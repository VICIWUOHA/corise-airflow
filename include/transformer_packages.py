#!/usr/bin/env python3.
# This script contains helper scripts abstracted away from the DAGS directory.
import numpy as np
from random import randint
from typing import List, Tuple, Dict
from zipfile import ZipFile
import xgboost as xgb
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


DATASET_NORM_WRITE_BUCKET = "corise-airflow-victor"
TRAINING_DATA_PATH = "week-2/price_prediction_training_data.csv"
VAL_END_INDEX = 31056

DATA_TYPES = ["generation", "weather"]

normalized_columns = {
    "generation": {
        "time": "time",
        "columns": [
            "total_load_actual",
            "price_day_ahead",
            "price_actual",
            "generation_fossil_hard_coal",
            "generation_fossil_gas",
            "generation_fossil_brown_coal_lignite",
            "generation_fossil_oil",
            "generation_other_renewable",
            "generation_waste",
            "generation_biomass",
            "generation_other",
            "generation_solar",
            "generation_hydro_water_reservoir",
            "generation_nuclear",
            "generation_hydro_run_of_river_and_poundage",
            "generation_wind_onshore",
            "generation_hydro_pumped_storage_consumption",
        ],
    },
    "weather": {
        "time": "dt_iso",
        "columns": [
            "city_name",
            "temp",
            "pressure",
            "humidity",
            "wind_speed",
            "wind_deg",
            "rain_1h",
            "rain_3h",
            "snow_3h",
            "clouds_all",
        ],
    },
}


def data_unzipper(file_path: str) -> Dict[str, pd.DataFrame]:
    """
    A Simple function to unzip a zipfile containing csv's

    args
    ----
    `file_path`: A path containing a zipfile.

    Returns:
    -------

    A dictionary containing the file names and theird= corresponding data in a pandas dataframe

    eg;

    `{"file_name_1": file_name_1_df,\n
    "file_name_2": file_name_2_df\n}
    `
    """

    dfs = [
        pd.read_csv(ZipFile(file_path).open(file))
        for file in ZipFile(file_path).namelist()
    ]
    return {"df_energy": dfs[0], "df_weather": dfs[1]}


def df_convert_dtypes(df, convert_from, convert_to):
    cols = df.select_dtypes(include=[convert_from]).columns
    for col in cols:
        df[col] = df[col].values.astype(convert_to)
    return df


def feature_addition(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    A placeholder function to add relevant features to a dataset during feature engineering.
    """

    # Extract helpful temporal, geographic, and highly correlated energy features
    # Calculate the weight of every city
    total_pop = 6155116 + 5179243 + 1645342 + 1305342 + 987000
    weight_Madrid = 6155116 / total_pop
    weight_Barcelona = 5179243 / total_pop
    weight_Valencia = 1645342 / total_pop
    weight_Seville = 1305342 / total_pop
    weight_Bilbao = 987000 / total_pop
    cities_weights = {
        "Madrid": weight_Madrid,
        "Barcelona": weight_Barcelona,
        "Valencia": weight_Valencia,
        "Seville": weight_Seville,
        "Bilbao": weight_Bilbao,
    }

    for i in range(len(dataset)):
        # Generate 'hour', 'weekday' and 'month' features
        position = dataset.index[i]
        hour = position.hour
        weekday = position.weekday()
        month = position.month
        dataset.loc[position, "hour"] = hour
        dataset.loc[position, "weekday"] = weekday
        dataset.loc[position, "month"] = month

        # Generate 'business hour' feature
        if (hour > 8 and hour < 14) or (hour > 16 and hour < 21):
            dataset.loc[position, "business hour"] = 2
        elif hour >= 14 and hour <= 16:
            dataset.loc[position, "business hour"] = 1
        else:
            dataset.loc[position, "business hour"] = 0
        print("business hours generated")

        # Generate 'weekend' feature

        if weekday == 6:
            dataset.loc[position, "weekday"] = 2
        elif weekday == 5:
            dataset.loc[position, "weekday"] = 1
        else:
            dataset.loc[position, "weekday"] = 0
        print("weekdays generated")

        # Generate 'temp_range' for each city
        temp_weighted = 0
        for city in cities_weights.keys():
            temp_max = dataset.loc[position, "temp_max_{}".format(city)]
            temp_min = dataset.loc[position, "temp_min_{}".format(city)]
            dataset.loc[position, "temp_range_{}".format(city)] = abs(
                temp_max - temp_min
            )

            # Generated city-weighted temperature
            temp = dataset.loc[position, "temp_{}".format(city)]
            temp_weighted += temp * cities_weights.get("{}".format(city))
        dataset.loc[position, "temp_weighted"] = temp_weighted

        print("city temp features generated")

    dataset["generation coal all"] = (
        dataset["generation fossil hard coal"]
        + dataset["generation fossil brown coal/lignite"]
    )
    print("=>Done with Feature Generation")
    return dataset


def model_input_preparation(dataset: pd.DataFrame):
    """
    Takes in a dataset and transforms it by feature reduction.

    Transform each feature to fall within a range from 0 to 1, pull out the target price from the features,
    and use PCA to reduce the features to those with an explained variance >= 0.80. Concatenate the scaled and
    dimensionality-reduced feature matrix with the scaled target vector, and return this result.
    """

    X = dataset[dataset.columns.drop("price actual")].values
    y = dataset["price actual"].values
    y = y.reshape(-1, 1)
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaler_X.fit(X[:VAL_END_INDEX])
    scaler_y.fit(y[:VAL_END_INDEX])
    X_norm = scaler_X.transform(X)
    y_norm = scaler_y.transform(y)

    pca = PCA(n_components=0.80)
    pca.fit(X_norm[:VAL_END_INDEX])
    X_pca = pca.transform(X_norm)
    dataset_norm = np.concatenate((X_pca, y_norm), axis=1)
    df_norm = pd.DataFrame(dataset_norm)
    print("Model Inputs Ready for Upload.")

    return df_norm


def indices_producer(
    max_end_index: int, min_indices_output=10, min_training_percentage=0.8
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates training and validation indices based on defined input by user.

    args
    ----

    `max_end_index`: maximum index of dataset
    `min_indices_output`: minimum no. of indices pairs to emit (output would range between this number to min_indices_output+3)
    `min_training_percentage`: minimum threshold for training data.

    returns
    ------
    A list containing tuples of train and test pairs of indices in which Each pair of training and validation
    indices should not overlap and never exceed the max of VAL_END_INDEX

    eg; [((start_train_index_1,end_train_index_1),(start_val_index_1,end_val_index_1))]
            Produces zipped list of training and validation indices

    The number of indices produced determine the number of downstream tasks to be generated.

    """

    print(f"Producing Indices.............")
    x = 0
    indices = []
    # randomize the number of generated indices pairs taking into account specified minimum
    while x <= randint(min_indices_output, min_indices_output + 3):
        # randomize cutoff point for splitting training
        train_cutoff_point = np.random.randint(
            max_end_index * min_training_percentage, max_end_index + 1
        )
        training_set = [0, train_cutoff_point]
        # ensure validation start is one after the cutoff point(training_end_index)
        validation_set = [train_cutoff_point + 1, max_end_index]
        train_and_val_pair = np.array(training_set), np.array(validation_set)

        indices.append(train_and_val_pair)
        x += 1

    print(f"`{len(indices)}` Indices Pairs were generated.")
    return indices


def multivariate_data(
    dataset, data_indices, history_size, target_size, step, single_step=False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce subset of dataset indexed by data_indices, with a window size of history_size hours
    """

    target = dataset[:, -1]
    data = []
    labels = []
    for i in data_indices:
        indices = range(i, i + history_size, step)
        # If within the last 23 hours in the dataset, skip
        if i + history_size > len(dataset) - 1:
            continue
        data.append(dataset[indices])
        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i : i + target_size])
    return np.array(data), np.array(labels)


def train_xgboost(X_train, y_train, X_val, y_val) -> xgb.Booster:
    """
    Train xgboost model using training set and evaluated against evaluation set, using
        a set of model parameters
    """

    X_train_xgb = X_train.reshape(-1, X_train.shape[1] * X_train.shape[2])
    X_val_xgb = X_val.reshape(-1, X_val.shape[1] * X_val.shape[2])
    param = {
        "eta": 0.03,
        "max_depth": 180,
        "subsample": 1.0,
        "colsample_bytree": 0.95,
        "alpha": 0.1,
        "lambda": 0.15,
        "gamma": 0.1,
        "objective": "reg:linear",
        "eval_metric": "rmse",
        "silent": 1,
        "min_child_weight": 0.1,
        "n_jobs": -1,
    }
    dtrain = xgb.DMatrix(X_train_xgb, y_train)
    dval = xgb.DMatrix(X_val_xgb, y_val)
    eval_list = [(dtrain, "train"), (dval, "eval")]
    xgb_model = xgb.train(param, dtrain, 10, eval_list, early_stopping_rounds=3)
    return xgb_model


def get_train_and_validation_pairs(
    dataset_norm, indices
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes in dataset and indices training pairs and returns a Tuple of train and validation labels and targets.

    Return format is X_train, y_train, X_val, y_val

    """
    past_history = 24
    future_target = 0
    train_indices, val_indices = indices
    print(f"train_indices is {train_indices}, val_indices is {val_indices}")
    X_train, y_train = multivariate_data(
        dataset_norm,
        train_indices,
        past_history,
        future_target,
        step=1,
        single_step=True,
    )
    X_val, y_val = multivariate_data(
        dataset_norm, val_indices, past_history, future_target, step=1, single_step=True
    )
    return X_train, y_train, X_val, y_val


def model_selector(models: List[xgb.Booster]) -> xgb.Booster:
    """
    This function simply chooses the best model based on the highest best score seen in the input array of models.

    returns
    ------

    `xgb.Booster`

    """
    # max_model_score = max([model.best_score for model in models])
    # get first pair of model and model score after sorting in descending order of scores
    best_model_pair = sorted(
        [(model, model.best_score) for model in models],
        key=lambda pair: pair[1],
        reverse=True,
    )[0]
    # for model in models:
    #     if model.best_score == max_model_score:
    print(f"The Best model had a score of {best_model_pair[1]}")
    return best_model_pair[0]


# Major Transformation scripts for Datasets


def process_energy_dataset(energy_dataset: pd.DataFrame):
    """
    Function containing tranformation processes for `energy` dataset
    """
    # Drop columns that are all 0s\

    energy_dataset = energy_dataset.drop(
        [
            "generation fossil coal-derived gas",
            "generation fossil oil shale",
            "generation fossil peat",
            "generation geothermal",
            "generation hydro pumped storage aggregated",
            "generation marine",
            "generation wind offshore",
            "forecast wind offshore eday ahead",
            "total load forecast",
            "forecast solar day ahead",
            "forecast wind onshore day ahead",
        ],
        axis=1,
    )

    # Extract timestamp
    energy_dataset["time"] = pd.to_datetime(
        energy_dataset["time"], utc=True, infer_datetime_format=True
    )
    energy_dataset = energy_dataset.set_index("time")

    # Interpolate the null price values
    energy_dataset.interpolate(
        method="linear", limit_direction="forward", inplace=True, axis=0
    )
    return energy_dataset


def process_weather_dataset(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    function containing tranformation processes for `weather` dataset
    """

    # Convert all ints to floats
    weather_df = df_convert_dtypes(weather_df, np.int64, np.float64)

    # Extract timestamp
    weather_df["time"] = pd.to_datetime(
        weather_df["dt_iso"], utc=True, infer_datetime_format=True
    )

    # Drop original time column
    weather_df = weather_df.drop(["dt_iso"], axis=1)
    weather_df = weather_df.set_index("time")

    # Reset index and drop records for the same city and time
    weather_df = (
        weather_df.reset_index()
        .drop_duplicates(subset=["time", "city_name"], keep="first")
        .set_index("time")
    )

    # Remove unnecessary qualitiative columns
    weather_df = weather_df.drop(
        ["weather_main", "weather_id", "weather_description", "weather_icon"], axis=1
    )

    # Filter out pressure and wind speed outliers
    weather_df.loc[weather_df.pressure > 1051, "pressure"] = np.nan
    weather_df.loc[weather_df.pressure < 931, "pressure"] = np.nan
    weather_df.loc[weather_df.wind_speed > 50, "wind_speed"] = np.nan

    # Interpolate for filtered values
    weather_df.interpolate(
        method="linear", limit_direction="forward", inplace=True, axis=0
    )
    return weather_df


def join_and_transform_datasets(
    df_energy: pd.DataFrame, df_weather: pd.DataFrame
) -> pd.DataFrame:
    """
    This function joins and transforms the energy and weather datasets and also drops city-specific features.
    """

    df_final = df_energy
    df_1, df_2, df_3, df_4, df_5 = [x for _, x in df_weather.groupby("city_name")]
    dfs = [df_1, df_2, df_3, df_4, df_5]

    for df in dfs:
        city = df["city_name"].unique()
        city_str = (
            str(city)
            .replace("'", "")
            .replace("[", "")
            .replace("]", "")
            .replace(" ", "")
        )
        df = df.add_suffix("_{}".format(city_str))
        df_final = df_final.merge(df, on=["time"], how="outer")
        df_final = df_final.drop("city_name_{}".format(city_str), axis=1)

    cities = ["Barcelona", "Bilbao", "Madrid", "Seville", "Valencia"]
    for city in cities:
        df_final = df_final.drop(["rain_3h_{}".format(city)], axis=1)

    return df_final


# WEEK 3 ARTIFACTS

PROJECT_ID = "corise-airflow"
DESTINATION_BUCKET = "corise-airflow-victor"
# BQ_DATASET_NAME = "timeseries_energy"


def extract_data(file_path: str) -> List[pd.DataFrame]:
    """
    Basic Function to extract dataframes from zip file

    """
    datasets = [
        pd.read_csv(ZipFile(file_path).open(file))
        for file in ZipFile(file_path).namelist()
    ]
    return datasets


def produce_select_statement(dataset_name, table_name) -> str:
    """
    This Function generates a select statement for a specified dataset table
    time columns are casted as TIMESTAMPS here and all other columns in normalized_columns are selected

    Returns
    ------
    `str`: A sql query string
    """

    timestamp_col = normalized_columns[table_name].get("time")
    columns = normalized_columns[table_name].get("columns")

    query = f"""SELECT CAST({timestamp_col} AS TIMESTAMP) AS {table_name}_time,
                {",".join(col_name for col_name in columns)} FROM {dataset_name}.{table_name}
               """

    return query


# Week 4 Artifacts

BQ_DATASET_NAME = "timeseries_energy_victor"

time_columns = {"generation": "time", "weather": "dt_iso"}

filepaths = {
    "generation": f"gs://{DESTINATION_BUCKET}/week-3/generation.parquet",
    "weather": f"gs://{DESTINATION_BUCKET}/week-3/weather.parquet",
}

DEFAULT_ARGS = {
    "retries": 2,
    "retry_delay": timedelta(seconds=60),
    "owner": "Victor.I",
}
