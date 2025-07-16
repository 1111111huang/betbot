from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

class ModelWrapper(ABC):
    """Abstract base class for model wrappers"""
    def __init__(self, **params):
        self.params = params
        self.range_min = None
        self.range_max = None
        self.class_labels = None

    def _convert_to_classes(self, y, range_min, range_max):
        """
        Convert continuous values to class labels based on range.
        
        Classes are:
        - Class 0: values <= range_min
        - Class 1: values > range_min and <= range_min + 1
        - Class 2: values > range_min + 1 and <= range_min + 2
        - ...
        - Class N: values > range_max
        
        Args:
            y: array-like of continuous values
            range_min: minimum value for range
            range_max: maximum value for range
            
        Returns:
            class_labels: array of class indices
        """
        y = np.asarray(y)
        class_labels = np.zeros_like(y, dtype=np.int64)
        
        # Create class boundaries
        boundaries = list(range(range_min, range_max + 2))  # +2 to include > max class
        
        for i, boundary in enumerate(boundaries[:-1]):
            if i == 0:
                # First class: <= min
                mask = y <= boundary
            else:
                # Other classes: > previous_boundary and <= current_boundary
                mask = (y > boundaries[i-1]) & (y <= boundary)
            class_labels[mask] = i
        
        # Last class: > max
        class_labels[y > range_max] = len(boundaries) - 1
        
        return class_labels

    def _get_class_mapping(self, range_min, range_max):
        """
        Get the mapping from class indices to value ranges.
        
        Returns:
            dict: mapping from class index to (min_val, max_val, description)
        """
        boundaries = list(range(range_min, range_max + 2))
        mapping = {}
        
        for i, boundary in enumerate(boundaries[:-1]):
            if i == 0:
                mapping[i] = (float('-inf'), boundary, f"≤ {boundary}")
            else:
                mapping[i] = (boundaries[i-1], boundary, f"{boundaries[i-1]+1}-{boundary}")
        
        # Last class: > max
        mapping[len(boundaries) - 1] = (range_max, float('inf'), f"> {range_max}")
        
        return mapping

    @abstractmethod
    def fit(self, X, y, target_range=None):
        """
        Fit the model.
        
        Args:
            X: input features
            y: target values (continuous)
            target_range: tuple of (min, max) defining the range for class conversion
        """
        pass

    @abstractmethod
    def predict_proba(self, X, target_range=None):
        """
        Predict class probabilities.
        
        Args:
            X: input features
            target_range: tuple of (min, max) defining the range for class conversion
            
        Returns:
            array of shape (n_samples, n_classes) with class probabilities
        """
        pass

    def save(self, path):
        """Optional: save model to path"""
        pass

    def load(self, path):
        """Optional: load model from path"""
        pass

class SklearnWrapper(ModelWrapper):
    def __init__(self, model_class, **params):
        super().__init__(**params)
        self.model = model_class(**params)

    def fit(self, X, y, target_range=None):
        if target_range is not None:
            range_min, range_max = target_range
            self.range_min = range_min
            self.range_max = range_max
            y_classes = self._convert_to_classes(y, range_min, range_max)
            self.class_labels = y_classes
            self.class_mapping = self._get_class_mapping(range_min, range_max)
        else:
            # Fallback to original behavior
            y_classes = y
            
        self.model.fit(X, y_classes)

    def predict_proba(self, X, target_range=None):
        if target_range is not None and self.range_min is None:
            # If range is provided but not set during fit, set it now
            range_min, range_max = target_range
            self.range_min = range_min
            self.range_max = range_max
            self.class_mapping = self._get_class_mapping(range_min, range_max)
            
        return self.model.predict_proba(X)

    def save(self, path):
        import joblib
        joblib.dump(self.model, path)

    def load(self, path):
        import joblib
        self.model = joblib.load(path)

class TorchWrapper(ModelWrapper):
    show_graph = True  # class property

    def __init__(
        self,
        model_class,
        model_params={},
        learning_rate=0.001,
        batch_size=64,
        epochs=20,
        weight_decay=1e-4,
        device='auto',  # <=== NEW: device param
        show_graph=None,  # allow override
        **kwargs
    ):
        super().__init__(
            model_class=model_class,
            model_params=model_params,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            weight_decay=weight_decay,
            device=device
        )

        self.model_class = model_class
        self.model_params = model_params
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.show_graph = self.__class__.show_graph if show_graph is None else show_graph

        # === device auto-selection ===
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        print(f"[TorchWrapper] Using device: {self.device}")

        self.model = None
        self.input_dim = None
        self.output_dim = None

    def fit(self, X, y, target_range=None):
        X = np.asarray(X).astype(np.float32)
        y = np.asarray(y).astype(np.int64)

        if target_range is not None:
            range_min, range_max = target_range
            self.range_min = range_min
            self.range_max = range_max
            y_classes = self._convert_to_classes(y, range_min, range_max)
            self.class_labels = y_classes
            self.class_mapping = self._get_class_mapping(range_min, range_max)
            self.output_dim = len(self.class_mapping)
        else:
            # Fallback to original behavior
            y_classes = y
            self.output_dim = len(np.unique(y_classes))

        self.input_dim = X.shape[1]

        # instantiate model
        self.model = self.model_class(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            **self.model_params
        ).to(self.device)

        dataset = TensorDataset(torch.tensor(X), torch.tensor(y_classes))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        losses = []
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            avg_loss = epoch_loss / len(dataset)
            losses.append(avg_loss)
        if self.show_graph:
            plt.figure()
            plt.plot(range(1, self.epochs + 1), losses, marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss over Epochs')
            plt.show()
            print(f"Final training loss: {losses[-1]}")

    def predict_proba(self, X, target_range=None):
        X = np.asarray(X).astype(np.float32)
        
        if target_range is not None and self.range_min is None:
            # If range is provided but not set during fit, set it now
            range_min, range_max = target_range
            self.range_min = range_min
            self.range_max = range_max
            self.class_mapping = self._get_class_mapping(range_min, range_max)
            self.output_dim = len(self.class_mapping)
            
            # Need to recreate model with correct output dimension
            if self.model is None:
                self.model = self.model_class(
                    input_dim=self.input_dim,
                    output_dim=self.output_dim,
                    **self.model_params
                ).to(self.device)
        
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        assert self.model is not None  # Type checker hint
        self.model.eval()

        with torch.no_grad():
            logits = self.model(torch.tensor(X).to(self.device))
            probs = nn.functional.softmax(logits, dim=1)

        return probs.cpu().numpy()

    def save(self, path):
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        if self.model is None:
            self.model = self.model_class(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                **self.model_params
            ).to(self.device)

        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

class TorchSequenceWrapper(ModelWrapper):
    def __init__(
        self,
        model_class,
        model_params={},
        learning_rate=0.001,
        batch_size=64,
        epochs=20,
        weight_decay=1e-4,
        device='auto',
        show_graph=None,
        **kwargs
    ):
        super().__init__(
            model_class=model_class,
            model_params=model_params,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            weight_decay=weight_decay,
            device=device
        )
        self.model_class = model_class
        self.model_params = model_params
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay

        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.model = None

    def fit(self, X, y, target_range=None):
        # X: [N, T, F], y: [N]
        X = np.asarray(X).astype(np.float32)
        y = np.asarray(y).astype(np.int64)
        
        if target_range is not None:
            range_min, range_max = target_range
            self.range_min = range_min
            self.range_max = range_max
            y_classes = self._convert_to_classes(y, range_min, range_max)
            self.class_labels = y_classes
            self.class_mapping = self._get_class_mapping(range_min, range_max)
            self.output_dim = len(self.class_mapping)
        else:
            # Fallback to original behavior
            y_classes = y
            self.output_dim = len(np.unique(y_classes))
            
        self.input_dim = X.shape[2]
        self.seq_len = X.shape[1]

        self.model = self.model_class(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            seq_len=self.seq_len,
            **self.model_params
        ).to(self.device)

        dataset = TensorDataset(torch.tensor(X), torch.tensor(y_classes))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        # --- Learning rate scheduler with warm-up ---
        warmup_epochs = max(1, self.epochs // 10)
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / warmup_epochs
            else:
                return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        self.model.train()
        epoch_losses = []
        lrs = []
        max_grad_norm = 1.0  # gradient clipping value
        for epoch in range(self.epochs):
            batch_losses = []
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)  # gradient clipping
                optimizer.step()
                batch_losses.append(loss.item())
            avg_loss = np.mean(batch_losses)
            epoch_losses.append(avg_loss)
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
            # Optionally print progress
            if (epoch + 1) % max(1, self.epochs // 10) == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, LR: {lrs[-1]:.6f}")

        # Plot loss and LR curve
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(range(1, self.epochs + 1), epoch_losses, marker='o', label='Training Loss', color='tab:blue')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True)
        ax2 = ax1.twinx()
        ax2.plot(range(1, self.epochs + 1), lrs, marker='x', label='Learning Rate', color='tab:orange')
        ax2.set_ylabel('Learning Rate', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        plt.title('Training Loss and Learning Rate Curve')
        fig.tight_layout()
        plt.show()

    def predict_proba(self, X, target_range=None):
        X = np.asarray(X).astype(np.float32)
        
        if target_range is not None and self.range_min is None:
            # If range is provided but not set during fit, set it now
            range_min, range_max = target_range
            self.range_min = range_min
            self.range_max = range_max
            self.class_mapping = self._get_class_mapping(range_min, range_max)
            self.output_dim = len(self.class_mapping)
            
            # Need to recreate model with correct output dimension
            if self.model is None:
                self.model = self.model_class(
                    input_dim=self.input_dim,
                    output_dim=self.output_dim,
                    seq_len=self.seq_len,
                    **self.model_params
                ).to(self.device)
        
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        assert self.model is not None  # Type checker hint
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X).to(self.device)
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()

