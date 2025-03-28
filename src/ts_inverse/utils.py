from matplotlib.collections import LineCollection
import random
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import warnings

warnings.filterwarnings("ignore")


class TorchStandardScaler:
    def fit(self, x: torch.Tensor):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x: torch.Tensor):
        x -= self.mean
        x /= self.std + 1e-7
        return x

    def fit_transform(self, x: torch.Tensor):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x: torch.Tensor):
        x *= self.std + 1e-7
        x += self.mean
        return x


def plot_multicolor_line(ax, x, y, colors, linewidth=2, alpha=0.7):
    """
    Plots a continuous line with multiple colors on a given axis.

    This function creates a continuous line on the provided matplotlib axis, where each
    segment of the line can have a different color. This is useful for visualizing changes
    or transitions in data along the line, as different colors can represent different
    data conditions or states.

    Parameters:
    ax (matplotlib.axes.Axes): The matplotlib axis on which to plot the multicolor line.
    x (list or array-like): The x-coordinates of the points defining the line.
    y (list or array-like): The y-coordinates of the points defining the line.
    colors (list): A list of colors for each segment of the line. The number of colors
                   should match the number of segments created from the x and y coordinates.
    linewidth (float, optional): The width of the line. Default is 2.
    alpha (float, optional): The opacity of the line. Default is 0.7.

    Returns:
    None: The function adds the multicolor line to the specified axis but does not return any value.
    """
    segments = [((x1, y1), (x2, y2)) for x1, x2, y1, y2 in zip(x[:-1], x[1:], y[:-1], y[1:])]
    lc = LineCollection(segments, colors=colors, linewidth=linewidth, alpha=alpha)
    ax.add_collection(lc)


def evaluate_model(model, dataset, criterion=F.mse_loss, device="cpu"):
    model.eval()
    model.to(device)
    with torch.no_grad():
        inputs, targets = dataset[:]
        inputs, targets = inputs[:, :, model.features].to(device), targets[:, :, 0].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    model.to("cpu")
    return loss.detach().item()


def train_model(model, optimizer, tr_dataloader, criterion=F.mse_loss, num_epochs=40, device="cpu"):
    train_loss_history = []

    model.train()
    model.to(device)
    # Train the model
    for _ in range(num_epochs):
        epoch_loss = []
        for inputs, targets in tr_dataloader:
            inputs, targets = inputs[:, :, model.features].to(device), targets[:, :, 0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.detach().item())
        train_loss_history.append(sum(epoch_loss) / len(epoch_loss))
    model.to("cpu")

    return train_loss_history


def get_model_parameters(model):
    """Returns a model's parameters."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    g = torch.Generator()
    g.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # torch.use_deterministic_algorithms(True)
    return g


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def grid_search_params(search_settings):
    """
    Given a dictionary of hyperparameters, if a value is a list, loop over all values
    and create a grid search.
    """
    if isinstance(search_settings, dict):
        param_keys = search_settings.keys()
        param_values = search_settings.values()
        param_combinations = list(itertools.product(*[v if isinstance(v, list) else [v] for v in param_values]))
        for combination in param_combinations:
            yield dict(zip(param_keys, combination))

    # A list of search settings dictionaries
    elif isinstance(search_settings, list):
        for settings in search_settings:
            yield from grid_search_params(settings)
