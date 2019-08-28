import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def prepare_data(
        X,
        y,
        batch_size,
        test_size=.2,
        valid_size=.1,
        random_state=42
):
    """
    Creating the data loaders
    :type X: np.ndarray
    :type y: np.ndarray
    :type batch_size: int
    :type test_size: float
    :type valid_size: float
    :type random_state: int
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=valid_size,
        random_state=random_state
    )
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    valid_dataset = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return train_loader, test_loader, valid_loader
