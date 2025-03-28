import torch
import torch.nn as nn
import torch.nn.functional as F

from .jit_gru import JitGRUDecoder
from .gru import GRUDecoder


class GradToInputNN(nn.Module):
    name = "GradToInputNN"
    # https://github.com/wrh14/Learning_to_Invert/blob/main/main_learn_dlg_large_model.py#L471

    def __init__(self, hidden_size=3000, gradients_size=-1, attack_input_shape=(1, 24 * 4, 1), attack_target_shape=None):
        super(GradToInputNN, self).__init__()
        self.fc = nn.Sequential(nn.Linear(gradients_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.fc_input = nn.Linear(hidden_size, attack_input_shape.numel())
        self.fc_target = None
        if attack_target_shape is not None:
            self.fc_target = nn.Linear(hidden_size, attack_target_shape.numel())
        self.attack_input_shape = attack_input_shape
        self.attack_target_shape = attack_target_shape
        self.extra_info = {}

    def forward(self, x):
        x = self.fc(x)
        inputs = self.fc_input(x).view(x.shape[0], *self.attack_input_shape)
        if self.fc_target is None:
            return inputs, None

        targets = self.fc_target(x).view(x.shape[0], *self.attack_target_shape)
        return inputs, targets

    def inference(self, x):
        with torch.no_grad():
            inputs, targets = self(x)
            return inputs, targets


class GradToInputNN_Sigmoid(nn.Module):
    name = "GradToInputNN_Sigmoid"
    # https://github.com/wrh14/Learning_to_Invert/blob/main/main_learn_dlg.py#L251

    def __init__(
        self, hidden_size=3000, gradients_size=-1, attack_input_shape=(1, 24 * 4, 1), attack_target_shape=(1, 24 * 4, 1)
    ):
        super(GradToInputNN_Sigmoid, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(gradients_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.fc_input = nn.Linear(hidden_size, attack_input_shape.numel())
        self.fc_target = None
        if attack_target_shape is not None:
            self.fc_target = nn.Linear(hidden_size, attack_target_shape.numel())
        self.sigmoid = nn.Sigmoid()

        self.attack_input_shape = attack_input_shape
        self.attack_target_shape = attack_target_shape
        self.extra_info = {}

    def forward(self, x):
        x = self.fc(x)
        inputs = self.sigmoid(self.fc_input(x)).view(x.shape[0], *self.attack_input_shape)
        if self.fc_target is None:
            return inputs, None

        targets = self.sigmoid(self.fc_target(x)).view(x.shape[0], *self.attack_target_shape)
        return inputs, targets

    def inference(self, x):
        with torch.no_grad():
            inputs, targets = self(x)
            return inputs, targets


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        self.adapt = nn.Sequential()
        if in_features != out_features:
            self.adapt = nn.Sequential(nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features))

    def forward(self, x):
        identity = x
        out = self.dropout(self.relu(self.bn1(self.fc1(x))))
        out = self.bn2(self.fc2(out))
        out += self.adapt(identity)
        out = self.relu(out)  # Activation after adding the residual
        return out


class ImprovedGradToInputNN(nn.Module):
    def __init__(
        self,
        hidden_sizes=[2500, 500],
        gradients_size=-1,
        attack_input_shape=(1, 24 * 4, 1),
        attack_target_shape=(1, 24 * 4),
        dropout_rate=0.05,
    ):
        super(ImprovedGradToInputNN, self).__init__()
        # Define the network structure with residual blocks
        self.blocks = nn.Sequential(
            *[ResidualBlock(gradients_size, hidden_sizes[0], dropout_rate=dropout_rate)]
            + [
                ResidualBlock(hidden_sizes[i - 1], hidden_sizes[i], dropout_rate=dropout_rate)
                for i in range(1, len(hidden_sizes))
            ]
        )
        self.fc_input = nn.Linear(hidden_sizes[-1], attack_input_shape.numel())
        self.fc_target = None
        if attack_target_shape is not None:
            self.fc_target = nn.Linear(hidden_sizes[-1], attack_target_shape.numel())

        self.attack_input_shape = attack_input_shape
        self.attack_target_shape = attack_target_shape

    def forward(self, x):
        x = self.blocks(x)
        inputs = self.fc_input(x).view(x.shape[0], *self.attack_input_shape)
        if self.fc_target is None:
            return inputs, None

        targets = self.fc_target(x).view(x.shape[0], *self.attack_target_shape)
        return inputs, targets

    def inference(self, x):
        with torch.no_grad():
            inputs, targets = self(x)
            return inputs, targets


class ImprovedGradToInputNN_2(nn.Module):
    def __init__(
        self,
        hidden_sizes=[2500, 500],
        gradients_size=-1,
        attack_input_shape=(1, 24 * 4, 1),
        attack_target_shape=(1, 24 * 4),
        dropout_rate=0.05,
    ):
        super(ImprovedGradToInputNN_2, self).__init__()
        # Define the network structure with residual blocks
        self.blocks_input = nn.Sequential(
            *[ResidualBlock(gradients_size, hidden_sizes[0], dropout_rate=dropout_rate)]
            + [
                ResidualBlock(hidden_sizes[i - 1], hidden_sizes[i], dropout_rate=dropout_rate)
                for i in range(1, len(hidden_sizes))
            ]
        )
        self.fc_input = nn.Linear(hidden_sizes[-1], attack_input_shape.numel())

        self.blocks_targets, self.fc_target = None, None
        if attack_target_shape is not None:
            self.blocks_targets = nn.Sequential(
                *[ResidualBlock(gradients_size, hidden_sizes[0], dropout_rate=dropout_rate)]
                + [
                    ResidualBlock(hidden_sizes[i - 1], hidden_sizes[i], dropout_rate=dropout_rate)
                    for i in range(1, len(hidden_sizes))
                ]
            )
            self.fc_target = nn.Linear(hidden_sizes[-1], attack_target_shape.numel())

        self.attack_input_shape = attack_input_shape
        self.attack_target_shape = attack_target_shape

    def forward(self, x):
        inputs_space = self.blocks_input(x)
        inputs = self.fc_input(inputs_space).view(inputs_space.shape[0], *self.attack_input_shape)
        if self.attack_target_shape is None:
            return inputs, None

        targets_space = self.blocks_targets(x)
        targets = self.fc_target(targets_space).view(targets_space.shape[0], *self.attack_target_shape)
        return inputs, targets

    def inference(self, x):
        with torch.no_grad():
            inputs, targets = self(x)
            return inputs, targets


class ImprovedGradToInputNN_Quantile(nn.Module):
    def __init__(
        self,
        hidden_sizes=[2500, 500],
        gradients_size=-1,
        attack_input_shape=(1, 24 * 4, 1),
        attack_target_shape=(1, 24 * 4),
        dropout_rate=0.05,
        quantiles=[0.1, 0.5, 0.9],
    ):
        super(ImprovedGradToInputNN_Quantile, self).__init__()
        self.quantiles = quantiles
        num_quantiles = len(quantiles)

        # Define the network structure with residual blocks for input and target
        self.blocks_input = nn.Sequential(
            *[ResidualBlock(gradients_size, hidden_sizes[0], dropout_rate=dropout_rate)]
            + [
                ResidualBlock(hidden_sizes[i - 1], hidden_sizes[i], dropout_rate=dropout_rate)
                for i in range(1, len(hidden_sizes))
            ]
        )
        self.fc_input = nn.Linear(hidden_sizes[-1], attack_input_shape.numel() * num_quantiles)

        self.blocks_targets = None
        self.fc_target = None
        if attack_target_shape is not None:
            self.blocks_targets = nn.Sequential(
                *[ResidualBlock(gradients_size, hidden_sizes[0], dropout_rate=dropout_rate)]
                + [
                    ResidualBlock(hidden_sizes[i - 1], hidden_sizes[i], dropout_rate=dropout_rate)
                    for i in range(1, len(hidden_sizes))
                ]
            )
            self.fc_target = nn.Linear(hidden_sizes[-1], attack_target_shape.numel() * num_quantiles)

        self.attack_input_shape = attack_input_shape
        self.attack_target_shape = attack_target_shape

    def forward(self, x):
        inputs_space = self.blocks_input(x)
        inputs = self.fc_input(inputs_space)
        inputs = inputs.view(inputs_space.shape[0], *self.attack_input_shape, len(self.quantiles))

        if self.attack_target_shape is None:
            return inputs, None

        targets_space = self.blocks_targets(x)
        targets = self.fc_target(targets_space)
        targets = targets.view(targets_space.shape[0], *self.attack_target_shape, len(self.quantiles))
        return inputs, targets

    def inference(self, x):
        with torch.no_grad():
            inputs, targets = self(x)
            return inputs, targets


class ImprovedGradToInputNN_Probabilistic(nn.Module):
    def __init__(
        self,
        hidden_sizes=[2500, 500],
        gradients_size=-1,
        attack_input_shape=(1, 24 * 4, 1),
        attack_target_shape=(1, 24 * 4),
        dropout_rate=0.05,
        distribution="normal",
    ):
        super(ImprovedGradToInputNN_Probabilistic, self).__init__()
        # Define the network structure with residual blocks
        self.blocks_input = nn.Sequential(
            *[ResidualBlock(gradients_size, hidden_sizes[0], dropout_rate=dropout_rate)]
            + [
                ResidualBlock(hidden_sizes[i - 1], hidden_sizes[i], dropout_rate=dropout_rate)
                for i in range(1, len(hidden_sizes))
            ]
        )
        self.fc_input_param_1 = nn.Linear(hidden_sizes[-1], attack_input_shape.numel())
        self.fc_input_param_2 = nn.Linear(hidden_sizes[-1], attack_input_shape.numel())

        self.blocks_targets, self.fc_target_param_1, self.fc_target_param_2 = None, None, None
        if attack_target_shape is not None:
            self.blocks_targets = nn.Sequential(
                *[ResidualBlock(gradients_size, hidden_sizes[0], dropout_rate=dropout_rate)]
                + [
                    ResidualBlock(hidden_sizes[i - 1], hidden_sizes[i], dropout_rate=dropout_rate)
                    for i in range(1, len(hidden_sizes))
                ]
            )
            self.fc_target_param_1 = nn.Linear(hidden_sizes[-1], attack_target_shape.numel())
            self.fc_target_param_2 = nn.Linear(hidden_sizes[-1], attack_target_shape.numel())

        self.attack_input_shape = attack_input_shape
        self.attack_target_shape = attack_target_shape
        self.distribution = distribution

    def forward(self, x):
        inputs_space = self.blocks_input(x)
        param_1_inputs = self.fc_input_param_1(inputs_space).view(inputs_space.shape[0], *self.attack_input_shape)
        param_2_inputs = self.fc_input_param_2(inputs_space).view(inputs_space.shape[0], *self.attack_input_shape)

        if self.distribution == "beta":
            param_1_inputs = F.softplus(param_1_inputs) + 1e-6
            param_2_inputs = F.softplus(param_2_inputs) + 1e-6

        if self.attack_target_shape is None:
            return param_1_inputs, param_2_inputs, None, None

        targets_space = self.blocks_targets(x)
        param_1_targets = self.fc_target_param_1(targets_space).view(targets_space.shape[0], *self.attack_target_shape)
        param_2_targets = self.fc_target_param_2(targets_space).view(targets_space.shape[0], *self.attack_target_shape)

        if self.distribution == "beta":
            param_1_targets = F.softplus(param_1_targets) + 1e-6
            param_2_targets = F.softplus(param_2_targets) + 1e-6

        return (param_1_inputs, param_2_inputs), (param_1_targets, param_2_targets)

    # Inference function
    def inference(self, x):
        with torch.no_grad():
            (param_1_inputs, param_2_inputs), (param_1_targets, param_2_targets) = self(x)

            if self.distribution == "cauchy":
                scale_inputs = torch.exp(param_2_inputs)
                inputs_dist = torch.distributions.Cauchy(param_1_inputs, scale_inputs)
                sampled_inputs = inputs_dist.rsample()

                if self.attack_target_shape is None:
                    return sampled_inputs, None

                scale_targets = torch.exp(param_2_targets)
                targets_dist = torch.distributions.Cauchy(param_1_targets, scale_targets)
                sampled_targets = targets_dist.sample()
            elif self.distribution == "normal":
                std_inputs = torch.exp(param_2_inputs)
                inputs_dist = torch.distributions.Normal(param_1_inputs, std_inputs)
                sampled_inputs = inputs_dist.rsample()

                if self.attack_target_shape is None:
                    return sampled_inputs, None

                std_targets = torch.exp(param_2_targets)
                targets_dist = torch.distributions.Normal(param_1_targets, std_targets)
                sampled_targets = targets_dist.sample()
            elif self.distribution == "beta":
                alpha_inputs = torch.exp(param_1_inputs)
                beta_inputs = torch.exp(param_2_inputs)
                inputs_dist = torch.distributions.Beta(alpha_inputs, beta_inputs)
                sampled_inputs = inputs_dist.rsample()

                if self.attack_target_shape is None:
                    return sampled_inputs, None

                alpha_targets = torch.exp(param_1_targets)
                beta_targets = torch.exp(param_2_targets)
                targets_dist = torch.distributions.Beta(alpha_targets, beta_targets)
                sampled_targets = targets_dist.rsample()
            else:
                raise ValueError(f"Unknown distribution: {self.distribution}")

            return sampled_inputs, sampled_targets

    def calculate_batchwise_prob_loss(self, predicted_inputs, predicted_targets, aux_inputs, aux_targets, config):
        # Calculate the loss for the probabilistic gradient inversion model
        probability_loss_function = None  # lambda param_1, param_2, target: F.mse_loss(param_1, target) # should not be used
        if self.distribution == "cauchy":

            def cauchy_nll(mu, logscale, target):
                # scale = torch.exp(logscale)
                # return torch.log(scale * (1 + ((target - mu) / scale)**2)).mean()
                return -torch.distributions.cauchy.Cauchy(mu, torch.exp(logscale)).log_prob(target).mean()

            probability_loss_function = cauchy_nll
        elif self.distribution == "normal":

            def normal_nll(mu, logscale, target):
                # scale = torch.exp(logscale)
                # return torch.log(scale) + 0.5 * ((target - mu) / scale)**2
                return -torch.distributions.normal.Normal(mu, torch.exp(logscale)).log_prob(target).mean()

            probability_loss_function = normal_nll
        elif self.distribution == "beta":

            def beta_nll(alpha, beta, target):
                dist = torch.distributions.beta.Beta(alpha, beta)
                if target.min() < 0 or target.max() > 1:
                    print("Target is not in the range [0, 1] for beta distribution, while it should be minmaxed between 0 and 1")

                return -dist.log_prob(torch.clamp(target, min=1e-6, max=1 - (1e-6))).mean()

            probability_loss_function = beta_nll

        predicted_inputs_param_1, predicted_inputs_param_2 = predicted_inputs
        predicted_targets_param_1, predicted_targets_param_2 = predicted_targets

        attack_batch_size = aux_inputs.shape[0]
        batch_size = aux_inputs.shape[1]

        # View the predicted inputs and auxiliary inputs as flat vectors
        predicted_inputs_param_1 = predicted_inputs_param_1.view(attack_batch_size, batch_size, -1)
        predicted_inputs_param_2 = predicted_inputs_param_2.view(attack_batch_size, batch_size, -1)
        aux_inputs = aux_inputs.view(attack_batch_size, batch_size, -1)

        # Check if targets are provided and concatenate them with inputs if they are
        if predicted_targets is not None:
            predicted_targets_param_1 = predicted_targets_param_1.view(attack_batch_size, batch_size, -1)
            predicted_targets_param_2 = predicted_targets_param_2.view(attack_batch_size, batch_size, -1)
            aux_targets = aux_targets.view(attack_batch_size, batch_size, -1)

            # Concatenate inputs with targets along the last dimension
            predicted_mu_combined = torch.cat((predicted_inputs_param_1, predicted_targets_param_1), dim=-1)
            predicted_logscale_combined = torch.cat((predicted_inputs_param_2, predicted_targets_param_2), dim=-1)
            aux_combined = torch.cat((aux_inputs, aux_targets), dim=-1)
        else:
            # Use only inputs if no targets are provided
            predicted_mu_combined = predicted_inputs_param_1
            predicted_logscale_combined = predicted_inputs_param_2
            aux_combined = aux_inputs

        # Calculate pairwise cauchy_nll between combined vectors
        batch_wise_combined_loss = torch.empty((attack_batch_size, batch_size, batch_size), device=config["device"])
        for i in range(attack_batch_size):
            for j in range(batch_size):
                for k in range(batch_size):
                    loss_ij = probability_loss_function(
                        predicted_mu_combined[i, j], predicted_logscale_combined[i, j], aux_combined[i, k]
                    )
                    batch_wise_combined_loss[i, j, k] = loss_ij.mean()

        return batch_wise_combined_loss


class ImprovedGradToInputNN_3(nn.Module):
    def __init__(
        self,
        hidden_sizes=[2500, 500],
        gradients_size=-1,
        attack_input_shape=(1, 24 * 4, 1),
        attack_target_shape=(1, 24 * 4),
        dropout_rate=0.05,
    ):
        super(ImprovedGradToInputNN_3, self).__init__()
        # Define the network structure with residual blocks
        self.blocks_input = nn.Sequential(
            *[ResidualBlock(gradients_size, hidden_sizes[0], dropout_rate=dropout_rate)]
            + [
                ResidualBlock(hidden_sizes[i - 1], hidden_sizes[i], dropout_rate=dropout_rate)
                for i in range(1, len(hidden_sizes))
            ]
        )
        self.gru_input = JitGRUDecoder(
            n_features=attack_input_shape[2], hidden_size=hidden_sizes[-1], output_length=attack_input_shape[1], backwards=True
        )

        self.blocks_targets = nn.Sequential(
            *[ResidualBlock(gradients_size, hidden_sizes[0], dropout_rate=dropout_rate)]
            + [
                ResidualBlock(hidden_sizes[i - 1], hidden_sizes[i], dropout_rate=dropout_rate)
                for i in range(1, len(hidden_sizes))
            ]
        )
        self.fc_input = nn.Linear(hidden_sizes[-1], attack_input_shape.numel())
        self.fc_target = nn.Linear(hidden_sizes[-1], attack_target_shape.numel())

        self.attack_input_shape = attack_input_shape
        self.attack_target_shape = attack_target_shape

    def forward(self, x):
        inputs_space = self.blocks_input(x)
        inputs = self.fc_input(inputs_space).view(inputs_space.shape[0], *self.attack_input_shape)
        targets_space = self.blocks_targets(x)
        targets = self.fc_target(targets_space).view(targets_space.shape[0], *self.attack_target_shape)
        return inputs, targets


class GradToTemporalInputNN(nn.Module):
    def __init__(
        self,
        hidden_sizes=[2500, 500],
        gradients_size=-1,
        attack_input_shape=(1, 24 * 4, 1),
        attack_target_shape=(1, 24 * 4),
        dropout_rate=0.05,
    ):
        super(GradToTemporalInputNN, self).__init__()

        # Define the network structure with residual blocks
        self.blocks = nn.Sequential(
            *[ResidualBlock(gradients_size, hidden_sizes[0], dropout_rate=dropout_rate)]
            + [
                ResidualBlock(hidden_sizes[i - 1], hidden_sizes[i], dropout_rate=dropout_rate)
                for i in range(1, len(hidden_sizes))
            ]
        )
        self.fc_input = nn.Linear(hidden_sizes[-1], hidden_sizes[-1])
        self.fc_target = nn.Linear(hidden_sizes[-1], hidden_sizes[-1])

        self.attack_input_shape = attack_input_shape
        self.attack_target_shape = attack_target_shape

        print(
            attack_input_shape, attack_target_shape
        )  # Target shape is now batch_size x seq_len, as number of features is always 1
        self.gru_decoder_input = GRUDecoder(
            n_features=attack_input_shape[0] * attack_input_shape[2],
            hidden_size=hidden_sizes[-1],
            output_length=attack_input_shape[1],
            backwards=True,
        )
        self.gru_decoder_target = GRUDecoder(
            n_features=attack_input_shape[0], hidden_size=hidden_sizes[-1], output_length=attack_target_shape[1], backwards=False
        )

    def forward(self, x, input_ii=None, input_t=None, input_tfp=0.0, target_ii=None, target_t=None, target_tfp=0.0):
        x = self.blocks(x)
        input_x = self.fc_input(x)
        target_x = self.fc_target(x)
        gru_input = self.gru_decoder_input(input_x, input_ii, input_t, input_tfp).view(x.shape[0], *self.attack_input_shape)
        gru_target = self.gru_decoder_target(target_x, target_ii, target_t, target_tfp).view(
            x.shape[0], *self.attack_target_shape
        )
        return gru_input, gru_target
