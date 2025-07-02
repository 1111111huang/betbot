from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
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

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict_proba(self, X):
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

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
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

    def fit(self, X, y):
        X = np.asarray(X).astype(np.float32)
        y = np.asarray(y).astype(np.int64)

        self.input_dim = X.shape[1]
        self.output_dim = len(np.unique(y))

        # instantiate model
        self.model = self.model_class(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            **self.model_params
        ).to(self.device)

        dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
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

    def predict_proba(self, X):
        X = np.asarray(X).astype(np.float32)
        self.model.eval()

        with torch.no_grad():
            logits = self.model(torch.tensor(X).to(self.device))
            probs = nn.functional.softmax(logits, dim=1)

        return probs.cpu().numpy()

    def save(self, path):
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
    show_graph = True  # class property

    def __init__(
        self,
        model_class,
        model_params={},
        learning_rate=0.001,
        batch_size=64,
        epochs=20,
        weight_decay=1e-4,
        device='auto',
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

        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.model_class = model_class
        self.model_params = model_params
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.show_graph = self.__class__.show_graph if show_graph is None else show_graph

        self.model = None

    def fit(self, X, y):
        X = np.asarray(X).astype(np.float32)  # [batch, seq, features]
        y = np.asarray(y).astype(np.int64)

        self.input_dim = X.shape[2]
        self.output_dim = len(np.unique(y))

        self.model = self.model_class(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            **self.model_params
        ).to(self.device)

        dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
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

    def predict_proba(self, X):
        X = np.asarray(X).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X).to(self.device))
            probs = nn.functional.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def save(self, path):
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
