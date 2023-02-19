from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union
from zipfile import ZipFile
from include.transformer_packages import (
    df_convert_dtypes,
    data_unzipper,
    feature_addition,
    indices_producer,
    model_input_preparation,
    process_energy_dataset,
    process_weather_dataset,
    join_and_transform_datasets,
    DATASET_NORM_WRITE_BUCKET,
    TRAINING_DATA_PATH,
)

import numpy as np
import pandas as pd
import xgboost as xgb
from airflow.operators.empty import EmptyOperator
from airflow.models.dag import DAG
from airflow.decorators import task, task_group
from airflow.providers.google.cloud.hooks.gcs import GCSHook

VAL_END_INDEX = 31056


@task()
def extract() -> Dict[str, pd.DataFrame]:
    """
    #### Extract task
    A simple task that loads each file in the zipped file into a dataframe,
    building a list of dataframes that is returned
    """

    filename = "/usr/local/airflow/dags/data/energy-consumption-generation-prices-and-weather.zip"
    unzipped_files_dict = data_unzipper(file_path=filename)

    return unzipped_files_dict


@task
def post_process_energy_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare energy dataframe for merge with weather data
    """

    transformed_energy_df = process_energy_dataset(df)
    return transformed_energy_df


@task
def post_process_weather_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare weather dataframe for merge with energy data
    """
    transformed_weather_df = process_weather_dataset(df)
    return transformed_weather_df


@task
def join_dataframes_and_post_process(
    df_energy: pd.DataFrame, df_weather: pd.DataFrame
) -> pd.DataFrame:
    """
    Join dataframes and drop city-specific features
    """

    joined_datasets = join_and_transform_datasets(df_energy, df_weather)
    return joined_datasets


@task
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract helpful temporal, geographic, and highly correlated energy features
    Call a feature_addition custom function here.
    """
    dataset_with_features = feature_addition(df)
    return dataset_with_features


@task
def prepare_model_inputs(df_final: pd.DataFrame):
    """
    Transform each feature to fall within a range from 0 to 1, pull out the target price from the features,
    and use PCA to reduce the features to those with an explained variance >= 0.80. Concatenate the scaled and
    dimensionality-reduced feature matrix with the scaled target vector, and return this result.
    matrix with the
    """

    normalized_dataset = model_input_preparation(dataset=df_final)
    # alt method
    client = GCSHook()
    client.upload(
        bucket_name=DATASET_NORM_WRITE_BUCKET,
        object_name=TRAINING_DATA_PATH,
        data=normalized_dataset.to_csv(),
        timeout=600,
    )


@task_group
def join_data_and_add_features():
    """
    Task group responsible for feature engineering, including:
      1. Extracting dataframes from local zipped file
      2. Processing energy and weather dataframes
      3. Joining dataframes
      4. Adding features to the joined dataframe
      5. Producing a dimension-reduced numpy array containing the most
         significant features, and save it to GCS
    """
    output = extract()
    df_energy, df_weather = output["df_energy"], output["df_weather"]
    df_energy = post_process_energy_df(df_energy)
    df_weather = post_process_weather_df(df_weather)
    df_final = join_dataframes_and_post_process(df_energy, df_weather)
    df_final = add_features(df_final)
    prepare_task = prepare_model_inputs(df_final)


# @task_group_2 has been moved to a seperate Dag. dags/week_2/price-prediction-model_training_pipeline.py

DEFAULT_ARGS = {
    "retries": 2,  # If a task fails, it will retry 2 times.
    "retry_delay": timedelta(seconds=60),
    "owner": "Victor.I",
}

with DAG(
    "energy_price_prediction",
    schedule_interval="@daily",
    start_date=datetime(2021, 1, 1),
    tags=["feature-engineering"],
    render_template_as_native_obj=True,
    catchup=False,
    concurrency=5,
    default_args=DEFAULT_ARGS,
) as dag:
    group_1 = join_data_and_add_features()
