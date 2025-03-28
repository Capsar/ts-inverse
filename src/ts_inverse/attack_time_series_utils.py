import torch


def resolution_warping_function(indices, total_length_tensor, resolution_tensor):
    # Assuming index, total_length, and resolution_factor are all tensors
    return 1 - ((indices / total_length_tensor) * (resolution_tensor - 1) / resolution_tensor)


def temporal_resolution_warping(data, resolution_factor, warping_function=resolution_warping_function):
    batch_size, sequence_length, n_features = data.shape
    output_length = sequence_length // resolution_factor

    # Generate all indices for weighting
    indices = torch.arange(0, sequence_length, device=data.device).float()
    total_length_tensor = torch.tensor(sequence_length, device=data.device, dtype=torch.float)
    resolution_tensor = torch.tensor(resolution_factor, device=data.device, dtype=torch.float)

    # Calculate weights for all indices
    weights = warping_function(indices, total_length_tensor, resolution_tensor)
    weights = weights.view(1, sequence_length, 1).expand(batch_size, sequence_length, n_features)

    # Apply weights and reshape
    weighted_data = data * weights

    # Efficiently sum up the segments
    segmented_sum = weighted_data.reshape(batch_size, output_length, resolution_factor, n_features).sum(dim=2)
    weight_sum = weights.reshape(batch_size, output_length, resolution_factor, n_features).sum(dim=2)

    warped_data = segmented_sum / weight_sum

    return warped_data


def interpolate(warped_data, target_length):
    return torch.nn.functional.interpolate(warped_data.transpose(1, 2), size=target_length, mode='linear').transpose(1, 2)


def divide_no_nan(a, b):
    """
    Source: https://github.com/autonlab/nbeats/blob/master/nbeats/contrib/utils/pytorch/losses.py
    Auxiliary funtion to handle divide by 0
    """

    div = a / b
    div[div != div] = 0.0
    div[div == float('inf')] = 0.0
    return div


def SMAPELoss(y, y_hat, mask=None):
    """SMAPE2 Loss

    Calculates Symmetric Mean Absolute Percentage Error.
    SMAPE measures the relative prediction accuracy of a
    forecasting method by calculating the relative deviation
    of the prediction and the true value scaled by the sum of the
    absolute values for the prediction and true value at a
    given time, then averages these devations over the length
    of the series. This allows the SMAPE to have bounds between
    0% and 200% which is desireble compared to normal MAPE that
    may be undetermined.

    Parameters
    ----------
    y: tensor (batch_size, output_size)
        actual values in torch tensor.
    y_hat: tensor (batch_size, output_size)
        predicted values in torch tensor.

    Returns
    -------
    smape:
        symmetric mean absolute percentage error

    References
    ----------
    [1] https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)
    """
    if mask is None:
        mask = torch.ones_like(y_hat)
    delta_y = 2*(y - y_hat).abs()
    scale = y.abs() + y_hat.abs()
    smape = (divide_no_nan(delta_y, scale) * mask).mean()
    return smape


def pinball_loss(predictions, targets, quantiles):
    quantiles = torch.tensor(quantiles, device=targets.device, dtype=torch.float32)
    targets = targets.unsqueeze(-1).expand_as(predictions)  # Adding and broadcasting quantile dimension to targets
    errors = targets - predictions
    loss = torch.max((quantiles - 1) * errors, quantiles * errors)
    return loss.mean()

def get_sequence_trend(sequence):
    """
    Computes the linear trend of a sequence using linear regression.
    Args:
        sequence (torch.Tensor): Tensor of shape (batch_size, sequence_length).
    Returns:
        trend (torch.Tensor): Tensor of the same shape as input, representing the linear trend.
    """
    with torch.no_grad():
        # Calculate time indices
        time = torch.arange(sequence.shape[1], dtype=sequence.dtype, device=sequence.device).view(1, -1)

        # Calculate the means of time and sequence
        time_mean = time.mean()
        sequence_mean = sequence.mean(dim=1, keepdim=True)

        # Compute the slope (beta) of the linear trend
        beta = ((time - time_mean) * (sequence - sequence_mean)).sum(dim=1, keepdim=True) / ((time - time_mean) ** 2).sum()

        # Compute the trend
        trend = beta * (time - time_mean) + sequence_mean

    return trend


def trend_consistency_regularization(sequence, loss='l1_mean'):
    """
    Applies a linear trend consistency regularizer to a sequence.
    Args:
        sequence (torch.Tensor): Tensor of shape (batch_size, sequence_length).
        loss (str): Type of loss to apply. Choose from 'l1_mean', 'l1_sum', 'l2_mean', 'l2_sum'.
    Returns:
        torch.Tensor: The computed regularization loss.
    """
    trend = get_sequence_trend(sequence)

    # Calculate the loss
    if loss == 'l1_mean':
        regularization_loss = torch.mean(torch.abs(sequence - trend))
    elif loss == 'l1_sum':
        regularization_loss = torch.sum(torch.abs(sequence - trend))
    elif loss == 'l2_mean':
        regularization_loss = torch.mean((sequence - trend) ** 2)
    elif loss == 'l2_sum':
        regularization_loss = torch.sum((sequence - trend) ** 2)
    else:
        raise ValueError('Invalid loss method. Choose either "l1_mean", "l1_sum", "l2_mean", or "l2_sum".')

    return regularization_loss


def periodicity_regularization(sequence, period, loss='l1_mean'):
    # Minimize the difference between each point and its corresponding point one period away
    if loss == 'l1_mean':
        return torch.mean((sequence[:, :-period] - sequence[:, period:]).abs())
    elif loss == 'l1_sum':
        return torch.sum((sequence[:, :-period] - sequence[:, period:]).abs())
    elif loss == 'l2_mean':
        return torch.mean((sequence[:, :-period] - sequence[:, period:]).pow(2))
    elif loss == 'l2_sum':
        return torch.sum((sequence[:, :-period] - sequence[:, period:]).pow(2))
    else:
        raise ValueError('Invalid loss method. Choose either "l1_mean", "l1_sum", "l2_mean", or "l2_sum".')
