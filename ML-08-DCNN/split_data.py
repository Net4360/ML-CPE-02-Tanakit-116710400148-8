from sklearn.model_selection import train_test_split

def split_dataset(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Split the dataset into training, validation, and testing sets[cite: 1]"""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=random_state, stratify=y_train_val
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)