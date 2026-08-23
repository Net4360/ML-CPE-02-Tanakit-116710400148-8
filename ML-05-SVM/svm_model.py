from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_svm(X_train, y_train, kernel_type="rbf", pca_components=150):
    scaler = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(pca_components, *X_train.shape),
                    whiten=True, random_state=42)),
    ])

    X_train_scaled = scaler.fit_transform(X_train)

    # เปลี่ยนบรรทัดนี้ให้รับค่า kernel_type
    model = SVC(
        kernel=kernel_type, C=10, gamma="scale", cache_size=1000
    )

    model.fit(X_train_scaled, y_train)
    return model, scaler


def predict_svm(model, scaler, X_test):

    # Apply the same scaling used for training data
    X_test_scaled = scaler.transform(X_test)
    # Predict
    predictions = model.predict(X_test_scaled)

    return predictions
