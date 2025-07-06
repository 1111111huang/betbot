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

class TorchSequenceWrapper:
    def __init__(
        self,
        model_class,
        model_params,
        sequence_length,
        date_col='date',
        home_col='home',
        away_col='away',
        learning_rate=0.001,
        batch_size=32,
        epochs=10,
        device='auto',
        use_padding=False,
    ):
        self.model_class = model_class
        self.model_params = model_params  # model init params only
        self.sequence_length = sequence_length
        self.date_col = date_col
        self.home_col = home_col
        self.away_col = away_col
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_padding = use_padding

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

    def _melt_matches(self, df):
        rows = []
        for _, row in df.iterrows():
            for is_home in [True, False]:
                team = row[self.home_col] if is_home else row[self.away_col]
                opponent = row[self.away_col] if is_home else row[self.home_col]
                new_row = row.copy()
                new_row['team'] = team
                new_row['opponent'] = opponent
                new_row['is_home'] = is_home
                rows.append(new_row)
        return pd.DataFrame(rows)

    def _build_sequences(self, feature_df, y_series):
        df = feature_df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df['season'] = df[self.date_col].apply(lambda d: d.year if d.month >= 7 else d.year - 1)
        df = self._melt_matches(df)
        df = df.sort_values(['team', 'season', self.date_col])

        exclude_cols = [self.home_col, self.away_col, self.date_col, 'season', 'team', 'opponent', 'is_home']
        feature_cols = [c for c in df.columns if c not in exclude_cols and c in feature_df.columns]

        sequences = []
        targets = []

        # y_series is a pandas Series, index should match feature_df
        for _, group in df.groupby(['team', 'season']):
            group = group.sort_values(self.date_col)
            values = group[feature_cols].values
            for i in range(1, len(group)):
                start_idx = max(0, i - self.sequence_length)
                seq = values[start_idx:i]
                if len(seq) < self.sequence_length:
                    if self.use_padding:
                        pad_len = self.sequence_length - len(seq)
                        seq = np.vstack([np.zeros((pad_len, seq.shape[1])), seq])
                    else:
                        continue  # skip sequences shorter than sequence_length if no padding
                match_index = group.iloc[i].name
                if match_index in y_series.index:
                    sequences.append(seq)
                    targets.append(y_series.loc[match_index])

        X = np.array(sequences)
        y = np.array(targets)
        return X, y

    def fit(self, X_df, y_series):
        # y_series is a pandas Series
        X_seq, y_seq = self._build_sequences(X_df, y_series)

        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = self.model_class(input_dim=X_seq.shape[2], **self.model_params).to(self.device)

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        for _ in range(self.epochs):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device).unsqueeze(1)
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

    def predict_proba(self, X_df):
        # Create dummy y_df to pass to _build_sequences
        print(X_df.shape)
        dummy_y_df = pd.Series(index=X_df.index)
        dummy_y_df.iloc[:] = 0  # dummy values

        X_seq, _ = self._build_sequences(X_df, dummy_y_df)

        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)  # (batch, n_classes)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs  # (n_data_points, n_classes)