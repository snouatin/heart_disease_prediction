import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from sqlalchemy import create_engine
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# ----------------------------
# Configuration MLflow
# ----------------------------
# Par défaut, MLflow cherche un server local sur le port 5000
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("heart_disease_prediction")

# ----------------------------
# Configuration PostgreSQL
# ----------------------------
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ----------------------------
# PyTorch Neural Network Model
# ----------------------------
class HeartDiseaseNet(nn.Module):
    def __init__(self, input_dim):
        super(HeartDiseaseNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

# ----------------------------
# Fonction pour entraîner modèle PyTorch
# ----------------------------
def train_pytorch_model(X_train, y_train, X_test, y_test, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Convertir en tenseurs
    train_tensor = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32),
                                 torch.tensor(y_train.values, dtype=torch.float32))
    test_tensor = TensorDataset(torch.tensor(X_test.values, dtype=torch.float32),
                                torch.tensor(y_test.values, dtype=torch.float32))
    train_loader = DataLoader(train_tensor, batch_size=params.get("batch_size", 32), shuffle=True)
    test_loader = DataLoader(test_tensor, batch_size=params.get("batch_size", 32))

    input_dim = X_train.shape[1]
    model = HeartDiseaseNet(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=params.get("lr", 1e-3))
    epochs = params.get("epochs", 20)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)
        # Logging loss
        mlflow.log_metric("pt_train_loss", epoch_loss, step=epoch)

    # Évaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())
    # Calculer métriques
    # Convertir probabilités en classes
    class_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, class_preds)
    auc = roc_auc_score(all_labels, all_preds)
    mlflow.log_metric("pt_accuracy", acc)
    mlflow.log_metric("pt_roc_auc", auc)

    # Log du modèle PyTorch dans MLflow
    mlflow.pytorch.log_model(model, "pt_heart_disease_model")

    return acc, auc

# ----------------------------
# Fonction principale
# ----------------------------
def train_model():
    # 1. Charger les données depuis PostgreSQL
    engine = create_engine(DATABASE_URI)
    df = pd.read_sql_table("heart_data", engine)

    # Supprimer la colonne d'identifiant
    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    # 2. Préparation des features et labels
    # One-hot encoding des variables catégorielles
    cat_cols = ["chest_pain_type", "rest_ecg", "st_slope"]
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)
    # df_encoded.columns = df_encoded.columns.astype(str)
    df_encoded = df_encoded.rename(columns={c : str(c) for c in df_encoded.columns})

    X = df_encoded.drop("target", axis=1)
    y = df_encoded["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_rf = df.drop("target", axis=1)
    y_rf = df["target"]
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

    # 3. Enregistrement dans MLflow : RandomForest
    with mlflow.start_run(run_name="RandomForest"):
        
        # Hyperparamètres
        n_estimators = 100
        max_depth = 5
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # 4. Entraînement
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train_rf, y_train_rf)

        # 5. Évaluation
        y_pred_rf = model.predict(X_test_rf)
        acc = accuracy_score(y_test_rf, y_pred_rf)
        auc = roc_auc_score(y_test_rf, model.predict_proba(X_test_rf)[:, 1])
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)

        # 6. Sauvegarde du modèle dans MLflow
        mlflow.sklearn.log_model(model, "rf_heart_disease_model")

        print(f"random Forest: accuracy={acc:.4f}, ROC_AUC={auc:.4f}")

    # 4. Enregistrement dans MLflow : PyTorch
    with mlflow.start_run(run_name="PyTorchNN"):
        pt_params = {"lr": 1e-3, "epochs": 20, "batch_size": 128}
        for k, v in pt_params.items():
            mlflow.log_param(f"pt_{k}", v)

        pt_acc, pt_auc = train_pytorch_model(X_train, y_train, X_test, y_test, pt_params)
        print(f"PyTorchNN: accuracy={pt_acc:.4f}, ROC_AUC={pt_auc:.4f}")

if __name__ == "__main__":
    train_model()
