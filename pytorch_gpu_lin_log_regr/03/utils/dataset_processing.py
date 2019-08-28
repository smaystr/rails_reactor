import torch


def add_ones_column(data: torch.Tensor, device: torch.device) -> torch.Tensor:
    num_examples = data.shape[0]
    ones_col = torch.ones([num_examples, 1], dtype=torch.float64, device=device)
    return torch.cat((ones_col, data), 1)


def normalize_data(data: torch.Tensor) -> torch.Tensor:
    """
    Normalize features.
    Normalizes input features X. Returns a normalized version of X where
    the mean value of each feature is 0 and deviation is close to 1.
    """
    data = data.clone().type(torch.float64)

    features_mean = torch.mean(data, 0)
    features_deviation = torch.std(data, 0)
    if data.shape[0] > 1:
        data -= features_mean
    # Normalize each feature values so that all features are close to [-1:1].
    # Also prevent division by zero error.
    features_deviation[features_deviation == 0] = 1
    data /= features_deviation

    return data
