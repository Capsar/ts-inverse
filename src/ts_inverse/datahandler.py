from distutils.util import strtobool
import datetime
from pandas import Series
import torch
from torch.utils.data import Dataset, ConcatDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


class TimeSeriesDataSet(Dataset):
    """
    TimeSeriesDataSet is a dataset that takes a pandas Series and converts it into a dataset that can be used for training a time series model.
    """

    def __init__(
        self,
        series: Series,
        offset=datetime.timedelta(days=1),
        observation_length=datetime.timedelta(days=13),
        target_length=datetime.timedelta(days=1),
        step_size=datetime.timedelta(minutes=15),
        verbose=False,
        normalize="no",
    ):
        self.fig_size = (15, 3.5)
        self.sample_frequency = series.index[1] - series.index[0]
        self.freq_in_day = int((60 * 60 * 24) / self.sample_frequency.total_seconds())
        self.series = adjust_for_daylight_saving(series, freq=self.sample_frequency)
        self.name = self.series.name
        self.step_size = step_size
        self.obs_length = observation_length
        self.tar_length = target_length

        # Remove the first day because we want to predict the next day at midnight
        next_day_date = (self.series.index[0] + offset).replace(hour=0, minute=0, second=0, microsecond=0)
        if verbose:
            print("First date:", self.series.index[0], "Next day date:", next_day_date)
        self.series = self.series[self.series.index >= next_day_date]

        ## Transform series ##
        if isinstance(normalize, str):
            if normalize == "minmax":
                # Min max the series to 0 and 1
                self.original_min, self.original_max = self.series.min(), self.series.max()
                self.normalize = ("minmax", self.original_min, self.original_max)
                self.series = (self.series - self.original_min) / (self.original_max - self.original_min)
            elif normalize == "standard":
                # Standardize the series
                self.original_mean, self.original_std = self.series.mean(), self.series.std()
                self.normalize = ("standard", self.original_mean, self.original_std)
                self.series = (self.series - self.original_mean) / self.original_std
            elif normalize == "no":
                self.normalize = ("no", 0, 1)
        elif isinstance(normalize, tuple):
            if normalize[0] == "minmax":
                self.normalize, self.original_min, self.original_max = normalize, normalize[1], normalize[2]
                self.series = (self.series - self.original_min) / (self.original_max - self.original_min)
            elif normalize[0] == "standard":
                self.normalize, self.original_mean, self.original_std = normalize, normalize[1], normalize[2]
                self.series = (self.series - self.original_mean) / self.original_std
            elif normalize[0] == "no":
                self.normalize = normalize

        # Determine the number of values inside the observation and target sequences
        if verbose:
            print("Sample frequency:", self.sample_frequency)
        self.n_obs = int(self.obs_length / self.sample_frequency)
        self.n_targets = int(self.tar_length / self.sample_frequency)
        if verbose:
            print("n_obs:", self.n_obs, "n_targets:", self.n_targets)

        # Determin the date range according to the step size, these become the indexes of the dataset
        self.start_date = self.series.index[0] + self.obs_length  # Start one week in
        self.end_date = self.series.index[-1] - self.tar_length  # End one day before the end
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq=self.step_size)
        if verbose:
            print(
                "Date range:",
                date_range[0],
                "to",
                date_range[-1],
                "with step size",
                self.step_size,
                "equal to",
                len(date_range),
                "samples",
            )

        # Create the X and Y tensors
        self.n_features = 6
        self.X = []
        self.Y = []
        if verbose:
            print("X:", self.X.shape, "Y:", self.Y.shape)

        # Fill the X and Y tensors with rolling window
        for i, date in enumerate(date_range):
            observation = self.series.loc[date - self.obs_length : date - self.sample_frequency]
            target = self.series.loc[date : date + self.tar_length - self.sample_frequency]

            if observation.sum() == 0 or target.sum() == 0:
                # print('Skipping', date, 'because of missing data!')
                continue

            if len(observation) != self.n_obs or len(target) != self.n_targets:
                if verbose:
                    print(
                        "Skipping",
                        date,
                        "because of wrong length: ",
                        len(observation),
                        len(target),
                        "instead of",
                        self.n_obs,
                        self.n_targets,
                        "!",
                    )
                continue

            def get_time_encodings(df_datetime_index):
                hour_of_day = df_datetime_index.hour.values / 24
                day_of_week = df_datetime_index.dayofweek.values / 7
                month_of_year = df_datetime_index.month.values / 12
                day_of_month = df_datetime_index.day.values / 31
                day_of_year = df_datetime_index.dayofyear.values / 365
                return hour_of_day, day_of_week, month_of_year, day_of_month, day_of_year

            y_values = np.stack((target.values, *get_time_encodings(target.index)), axis=1)
            self.Y.append(torch.tensor(y_values).float())

            x_values = np.stack((observation.values, *get_time_encodings(observation.index)), axis=1)
            self.X.append(torch.tensor(x_values).float())

        self.X = torch.stack(self.X)
        self.Y = torch.stack(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def it(self, tensor):
        if self.normalize[0] == "minmax":
            return tensor * (self.original_max - self.original_min) + self.original_min
        elif self.normalize[0] == "standard":
            return tensor * self.original_std + self.original_mean
        elif self.normalize[0] == "no":
            return tensor

    def plot_series(self, start_date=None, duration=None, input_feature="kW", save_fig=None):
        if start_date is None:
            start_date = self.start_date
        if duration is None:
            duration = self.end_date - self.start_date

        self.series.index = self.series.index.map(lambda x: pd.Timestamp(x).floor("min"))

        fig, ax = plt.subplots(figsize=self.fig_size)
        self.series[start_date : start_date + duration].plot(ax=ax, label=f"{input_feature}")
        ax.set_title(f'Column "{self.name}" from {start_date.strftime("%d-%m-%Y")} to {(start_date + duration).strftime("%d-%m-%Y")} ({duration.days} days)')
        ax.set_xlabel("Time")

        plt.legend()
        plt.tight_layout()
        if save_fig is not None:
            plt.savefig(save_fig, bbox_inches="tight")
        plt.show()

    def plot(self, idx, features=[0, 1, 2, 3, 4, 5], input_feature="Input", it=False, save_fig=None):
        """
        Plot the idx'th sample
        """
        x, y = self[idx]
        xaxis = pd.date_range(start=self.start_date + (idx * self.step_size), periods=x.shape[0] + y.shape[0], freq=self.sample_frequency)
        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.set_xlabel("Time")
        ax.set_title(f'Column "{self.name}" sample {idx}: Observation vs. Prediction Target')
        input_lines = []
        if it:
            x[:, 0] = self.it(x[:, 0])
            y[:, 0] = self.it(y[:, 0])

        if 0 in features:
            (line1_1,) = ax.plot(xaxis[: x.shape[0]], x[:, 0], label=f"Observation {input_feature}")
            input_lines.append(line1_1)
        if 1 in features:
            (line1_2,) = ax.plot(xaxis[: x.shape[0]], x[:, 1], label="Observation Hour")
            input_lines.append(line1_2)
        if 2 in features:
            (line1_3,) = ax.plot(xaxis[: x.shape[0]], x[:, 2], label="Observation Weekday")
            input_lines.append(line1_3)
        if 3 in features:
            (line1_4,) = ax.plot(xaxis[: x.shape[0]], x[:, 3], label="Observation Month")
            input_lines.append(line1_4)
        if 4 in features:
            (line1_5,) = ax.plot(xaxis[: x.shape[0]], x[:, 4], label="Observation Day")
            input_lines.append(line1_5)
        if 5 in features:
            (line1_6,) = ax.plot(xaxis[: x.shape[0]], x[:, 5], label="Observation Yearday")
            input_lines.append(line1_6)

        (line2,) = ax.plot(xaxis[x.shape[0] :], y[:, 0], label=f"Target {input_feature}")

        def format_tick(tick):
            date = datetime.datetime(1970, 1, 1) + datetime.timedelta(days=tick)
            day_name = date.strftime("%a")
            return date.strftime("%d-%m-%Y") + "(" + day_name + ")"

        ax.xaxis.set_major_formatter(FuncFormatter(lambda tick, _: format_tick(tick)))
        ax.legend()
        plt.tight_layout()
        if save_fig is not None:
            plt.savefig(save_fig, bbox_inches="tight")

        return xaxis, fig, ax, tuple(input_lines), line2


class ElectricityDataSet(TimeSeriesDataSet):
    """
    Electricity datasets are different because they only want to predict the coming 24 hours at midnight given the previous week.

    - Notes:
        Predictions are to be made at midnight for the coming day, so the dataset is offset by one day.
        The datasets have daylight savings so the dataset deals with this by interpolating the missing values and averaging the dupicated values.
        The plot function plots the idx'th sample of the dataset.
        The plot_weekly_average function plots the average of the dataset over a weeks time. important to set observation length and target length to a total of 7 days.
    """

    def __init__(
        self,
        series: Series,
        offset=datetime.timedelta(days=1),
        observation_length=datetime.timedelta(days=13),
        target_length=datetime.timedelta(days=1),
        step_size=datetime.timedelta(minutes=15),
        verbose=False,
        normalize="no",
    ):
        super().__init__(
            series,
            offset=offset,
            observation_length=observation_length,
            target_length=target_length,
            step_size=step_size,
            verbose=verbose,
            normalize=normalize,
        )

    def plot(self, idx, features=[0, 1, 2, 3, 4, 5], it=False, save_fig=None):
        return super().plot(idx, features, input_feature="kW", it=it, save_fig=save_fig)

    def plot_weekly_load_profile(
        self,
        plot_individual=True,
        median_instead_of_average=False,
        model=None,
        device="cpu",
        verbose=False,
        it=False,
        save_fig=None,
    ):
        """
        Plot the average of the dataset over a weeks time. This allows to see the weekly pattern of the dataset.
        In case of electricity, it the week day influences the electricity consumption.
        For example, on a Sunday, the electricity consumption is higher than on a Monday.
        Week days are: Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        freq_in_day = self.freq_in_day

        xaxis = np.linspace(0, 7, freq_in_day * 7)  # Week in 15 minute intervals
        week_colors = ["red", "orange", "black", "green", "blue", "purple", "turquoise"]
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for day in range(7):
            day_mask = (self.Y[:, :, day_of_week_index := 2] * 7 == day).all(dim=1)  # Day of the week is index 2
            Y_day = self.Y[day_mask][:, :, 0].T
            if model is not None:
                model.eval()
                X_day = self.X[day_mask]
                with torch.no_grad():
                    Y_hat = model(X_day[:, :, model.features].to(device)).detach().cpu()
                if it:
                    Y_hat = self.it(Y_hat)
                    Y_day = self.it(Y_day)

                ax.plot(
                    xaxis[freq_in_day * day : freq_in_day * (day + 1)],
                    Y_hat.numpy().T,
                    color=week_colors[day],
                    alpha=0.2,
                    linewidth=0.5,
                )
                ax.plot(
                    xaxis[freq_in_day * day : freq_in_day * (day + 1)],
                    Y_hat.numpy().T.mean(axis=1),
                    color=week_colors[day],
                    linestyle="--",
                )
                if median_instead_of_average:
                    ax.plot(
                        xaxis[freq_in_day * day : freq_in_day * (day + 1)],
                        Y_day.median(axis=1)[0],
                        color="brown",
                        linestyle="-.",
                        linewidth=0.8,
                    )
                else:
                    ax.plot(
                        xaxis[freq_in_day * day : freq_in_day * (day + 1)],
                        Y_day.mean(dim=1),
                        color="brown",
                        linestyle="-.",
                        linewidth=0.8,
                    )
                model.train()
                del X_day, Y_hat, Y_day
            else:
                if it:
                    Y_day = self.it(Y_day)

                if plot_individual:
                    ax.plot(xaxis[freq_in_day * day : freq_in_day * (day + 1)], Y_day.numpy(), color=week_colors[day], alpha=0.01)
                if median_instead_of_average:
                    ax.plot(xaxis[freq_in_day * day : freq_in_day * (day + 1)], Y_day.median(dim=1)[0], color=week_colors[day])
                else:
                    ax.plot(xaxis[freq_in_day * day : freq_in_day * (day + 1)], Y_day.mean(dim=1), color=week_colors[day])

        ax.xaxis.set_major_formatter(FuncFormatter(lambda tick, _: days[int(tick) % len(days)]))
        ax.set_xlabel("Day of the Week")
        ax.set_ylabel("Electricity kW")
        aggregation_type = "Median" if median_instead_of_average else "Average"
        legend_elements = [Line2D([0], [0], color=week_colors[day], lw=4, label=f"{days[day]}") for day in range(7)]
        if model is not None:
            legend_elements.append(Line2D([0], [0], color="brown", lw=2, linestyle="-.", label=f"Dataset {aggregation_type}"))

        ax.legend(handles=legend_elements)
        title = f'Consumer "{self.name}": {aggregation_type} Weekday Electricity Consumption Over {self.end_date - self.start_date} Days'
        if model is not None:
            title += f" with {model.name}"
        ax.set_title(title)
        plt.tight_layout()
        if save_fig is not None:
            plt.savefig(save_fig, bbox_inches="tight")

        if verbose:
            plt.show()
        else:
            plt.close(fig)
        return fig, ax


def adjust_for_daylight_saving(df_before, method="time", freq="15min"):
    """
    Adjusts a DataFrame for daylight saving time (DST) changes by interpolating missing values and averaging duplicate values.

    This function creates a continuous time series without gaps or overlaps caused by DST transitions. Missing values
    at the start of DST are interpolated using the specified method, while duplicated values at the end of DST are
    averaged.

    Parameters:
    df_before (pd.DataFrame): DataFrame with a datetime index that may contain gaps or duplicates due to DST.
    method (str, optional): Interpolation method for filling missing values. Default is 'time'.
    freq (str, optional): Frequency of the expected time series in the DataFrame. Default is '15min'.

    Returns:
    pd.DataFrame: Adjusted DataFrame with continuous and consistent time indexing across DST changes.
    """
    date_range = pd.date_range(start=df_before.index.min(), end=df_before.index.max(), freq=freq)
    daylight_saving_begin = date_range.difference(df_before.index)
    df_after = df_before.copy()
    for i in range(len(daylight_saving_begin)):
        df_after.loc[daylight_saving_begin[i]] = None
    df_after.interpolate(method=method, inplace=True)
    df_after = df_after.groupby(df_after.index).mean()
    return df_after


def time_series_train_test_split(series, test_size=0.2) -> tuple[Series, Series]:
    """
    Splits the time series data into train and test sets.

    Parameters:
    series (pandas.Series or pandas.DataFrame): The time series data.
    test_size (float): The proportion of the dataset to include in the test split (0 to 1).

    Returns:
    train, test (tuple): Tuple containing the training set and the test set.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    split_idx = int(len(series) * (1 - test_size))
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]

    return train, test


def get_datasets(
    dataset,
    normalize,
    columns,
    train_stride,
    observation_days,
    future_days,
    validation_stride=24,
    split_ratio=0.2,
    should_dropna=True,
    drop_leading_zeros=True,
    smaller=False,
) -> tuple[list[TimeSeriesDataSet], list[TimeSeriesDataSet], list[TimeSeriesDataSet]]:
    df, dataset_class = get_dataset_df(dataset)

    train_sets, val_sets, test_sets = [], [], []
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(columns, int):
        columns = [df.columns[columns]]
    if isinstance(columns, list) and isinstance(columns[0], int):
        columns = df.columns[columns]

    for column in columns:
        data_series = df.loc[:, column]

        if should_dropna:
            data_series = data_series.dropna()

        if drop_leading_zeros:
            first_non_zero_index = data_series.ne(0).idxmax()
            data_series = data_series.loc[first_non_zero_index:]

        if smaller:
            print("Smaller dataset (hardcoded in datahandler class for quick testing)")
            _, data_series = time_series_train_test_split(data_series, test_size=0.2)

        train_series, test_series = time_series_train_test_split(data_series, test_size=split_ratio)
        train_series, val_series = time_series_train_test_split(train_series, test_size=split_ratio)

        train_sets.append(
            train_set := dataset_class(
                train_series,
                step_size=datetime.timedelta(hours=train_stride),
                observation_length=datetime.timedelta(days=observation_days),
                target_length=datetime.timedelta(days=future_days),
                normalize=normalize,
            )
        )

        test_sets.append(
            dataset_class(
                test_series,
                step_size=datetime.timedelta(hours=validation_stride),
                observation_length=datetime.timedelta(days=observation_days),
                target_length=datetime.timedelta(days=future_days),
                normalize=normalize,
            )
        )

        val_sets.append(
            dataset_class(
                val_series,
                step_size=datetime.timedelta(hours=validation_stride),
                observation_length=datetime.timedelta(days=observation_days),
                target_length=datetime.timedelta(days=future_days),
                normalize=normalize,
            )
        )
    return train_sets, val_sets, test_sets


class ConcatSliceDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            min_idx = idx.start if idx.start is not None else 0
            max_idx = idx.stop if idx.stop is not None else len(self)
            step = idx.step if idx.step is not None else 1

            # print('min_idx:', min_idx, 'max_idx:', max_idx, 'step:', step)

            datasets_to_stack = []
            for cum, dataset in zip(self.cumulative_sizes, self.datasets):
                # print('Cum:', cum, 'Dataset length:', len(dataset))

                if min_idx < cum and max_idx <= cum:
                    # print('Slice is completely in dataset')
                    datasets_to_stack.append(dataset[min_idx:max_idx:step])
                elif min_idx < cum and max_idx > cum:
                    # print('Slice starts in this dataset but ends further on')
                    datasets_to_stack.append(dataset[min_idx::step])
                elif min_idx > cum:
                    # print('Slice starts further on')
                    min_idx = max(min_idx - cum, 0)
                    max_idx = max(max_idx - cum, 0)
                    continue
            # datasets_to_stack contains tuples of X and Y tensors these should be stacked such that 2 new tensors are created
            # one for X and one for Y
            # print('Datasets to stack:', len(datasets_to_stack))

            X, Y = zip(*datasets_to_stack)
            del datasets_to_stack
            return torch.cat(X), torch.cat(Y)

        return super().__getitem__(idx)


# tsf file loading data_loader.py


# Converts the contents in a .tsf file into a dataframe and returns it along with other meta-data of the dataset: frequency, horizon, whether the dataset contains missing values and whether the series have equal lengths
#
# Parameters
# full_file_path_and_name - complete .tsf file path
# replace_missing_vals_with - a term to indicate the missing values in series in the returning dataframe
# value_column_name - Any name that is preferred to have as the name of the column containing series values in the returning dataframe
def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if len(line_content) != 3:  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if len(line_content) != 2:  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(strtobool(line_content[1]))
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception("Missing attribute section. Attribute section must come before data.")

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception("Missing attribute section. Attribute section must come before data.")
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(numeric_series):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.datetime.strptime(full_info[i], "%Y-%m-%d %H-%M-%S")
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length)


# Used for Flower Framework Client factory
def get_dataset_df(dataset):
    def try_read_csv_in_paths(file_path, root_paths=["./", "./data", "../data", "../../data"]):
        for root_path in root_paths:
            try:
                return pd.read_csv(
                    root_path + file_path,
                    index_col="Time",
                    parse_dates=["Time"],
                )
            except FileNotFoundError:
                pass

    dataset_class = TimeSeriesDataSet
    if dataset == "london_smartmeter":
        dataset_class = ElectricityDataSet
        df = try_read_csv_in_paths("/LondonSmartMeter/london_smart_meters_dataset_without_missing_values_first_30_consumers.csv")

    elif dataset == "electricity_321":
        dataset_class = ElectricityDataSet
        df = try_read_csv_in_paths("/Electricity321Hourly/electricity_hourly_dataset.csv")
    elif dataset == "electricity_370":
        dataset_class = ElectricityDataSet
        df = try_read_csv_in_paths("/Electricity370/LD2011_2014_first_40_consumers.csv")
    elif dataset == "kddcup":
        df = try_read_csv_in_paths("/KDDCup_2018/kdd_cup_2018_dataset_without_missing_values.csv")
    return df, dataset_class


def get_datasets_from_df(
    df,
    dataset_class,
    columns,
    normalize,
    train_stride,
    observation_days,
    future_days,
    validation_stride=24,
    split_ratio=0.2,
    should_dropna=True,
) -> tuple[list[TimeSeriesDataSet], list[TimeSeriesDataSet], list[TimeSeriesDataSet]]:
    train_sets, val_sets, test_sets = [], [], []
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(columns, int):
        columns = [df.columns[columns]]
    if isinstance(columns, list) and isinstance(columns[0], int):
        columns = df.columns[columns]

    for consumer in columns:
        data_series = df.loc[:, consumer].dropna() if should_dropna else df.loc[:, consumer]

        train_series, test_series = time_series_train_test_split(data_series, test_size=split_ratio)
        train_series, val_series = time_series_train_test_split(train_series, test_size=split_ratio)

        train_sets.append(
            train_set := dataset_class(
                train_series,
                step_size=datetime.timedelta(hours=train_stride),
                observation_length=datetime.timedelta(days=observation_days),
                target_length=datetime.timedelta(days=future_days),
                normalize=normalize,
            )
        )

        test_sets.append(
            dataset_class(
                test_series,
                step_size=datetime.timedelta(hours=validation_stride),
                observation_length=datetime.timedelta(days=observation_days),
                target_length=datetime.timedelta(days=future_days),
                normalize=train_set.normalize,
            )
        )

        val_sets.append(
            dataset_class(
                val_series,
                step_size=datetime.timedelta(hours=validation_stride),
                observation_length=datetime.timedelta(days=observation_days),
                target_length=datetime.timedelta(days=future_days),
                normalize=train_set.normalize,
            )
        )
    return train_sets, val_sets, test_sets


def get_mean_std_dataloader(data_loader, device="cpu"):
    """
    Calculate the mean and standard deviation of the input and target tensors in a dataset.
    The input and target tensors are assumed to be of shape (batch_size, sequence_length, num_features).
    """

    inputs_sum_, inputs_sum_sq = 0.0, 0.0
    targets_sum_, targets_sum_sq = 0.0, 0.0
    total_samples = 0

    for inputs, targets in data_loader:
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1, targets.size(-1))
        inputs_sum_ += inputs.sum(dim=0)
        inputs_sum_sq += (inputs**2).sum(dim=0)

        targets_sum_ += targets.sum(dim=0)
        targets_sum_sq += (targets**2).sum(dim=0)
        total_samples += inputs.size(0)

    inputs_mean = inputs_sum_ / total_samples
    inputs_std = (inputs_sum_sq / total_samples - inputs_mean**2) ** 0.5  # Variance formula: E[X^2] - (E[X])^2

    targets_mean = targets_sum_ / total_samples
    targets_std = (targets_sum_sq / total_samples - targets_mean**2) ** 0.5  # Variance formula: E[X^2] - (E[X])^2
    return inputs_mean.to(device), inputs_std.to(device), targets_mean.to(device), targets_std.to(device)
