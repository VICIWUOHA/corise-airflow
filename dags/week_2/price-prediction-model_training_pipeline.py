import io
import numpy as np
import pandas as pd
import xgboost as xgb
from io import BytesIO
from datetime import datetime, timedelta
from typing import List, Tuple

from airflow.decorators import task, task_group, dag
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.sensors.external_task import ExternalTaskSensor
from include.transformer_packages import (
    indices_producer,
    model_selector,
    train_xgboost,
    get_train_and_validation_pairs,
    DATASET_NORM_WRITE_BUCKET,
    TRAINING_DATA_PATH,
    VAL_END_INDEX,
)


# This DAG would Run once the feature engineering is complete and it's data lands in GCS
# This Dag takes group 2 from the previous Dag as a means of decoupling.
#        1. Reading the dataset norm from GCS
#        2. Producing a list of training and validation indices numpy array tuples,
#        3. Mapping each element of that list onto the indices argument of format_data_and_train_model
#        4. Calling select_best_model on the output of all of the mapped tasks to select the best model and
#           write it to GCS

DEFAULT_ARGS = {
    "retries": 2,  # If a task fails, it will retry 2 times.
    "retry_delay": timedelta(seconds=60),
    "owner": "Victor.I",
}


@dag(
    dag_id="energy-price-prediction-model-training",
    start_date=datetime(2021, 1, 1),
    schedule="@daily",
    tags=["model_training"],
    render_template_as_native_obj=True,
    concurrency=5,
    catchup=False,
    default_args=DEFAULT_ARGS,
)
def taskflow():
    sense_features_dataset = ExternalTaskSensor(
        task_id="check_if_feature_dataset_is_ready",
        external_dag_id="energy_price_prediction",
        external_task_id="join_data_and_add_features.prepare_model_inputs",
        # execution_delta = timedelta(minutes=3),
        timeout=120,
        allowed_states=["success"],
        failed_states=["failed", "skipped"],
        mode="reschedule",
    )

    @task
    def read_dataset_norm():
        """
        This task will get triggered when it is sensed that energy_price_prediction.join_data_and_add_features.prepare_model_inputs
        has been completed successfully.

        This task runs after the External Task Sensor operation.
        """
        print(
            "Sensed upstream task completion signalling that model Inputs have landed in GCS bucket"
        )

        client = GCSHook().get_conn()
        read_bucket = client.bucket(DATASET_NORM_WRITE_BUCKET)
        dataset_norm = pd.read_csv(
            io.BytesIO(read_bucket.blob(TRAINING_DATA_PATH).download_as_bytes())
        ).to_numpy()

        return dataset_norm

    @task
    def produce_indices(
        val_end_index=VAL_END_INDEX,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Produces zipped list of training and validation indices.
        """

        # Indices generation has been abstracted.
        indices_list = indices_producer(max_end_index=val_end_index)
        return indices_list

    @task
    def format_data_and_train_model(
        dataset_norm: np.ndarray, indices: Tuple[np.ndarray, np.ndarray]
    ) -> xgb.Booster:
        """
        Extract training and validation sets and labels, and train a model with a given
        set of training and validation indices
        """
        X_train, y_train, X_val, y_val = get_train_and_validation_pairs(
            dataset_norm, indices
        )
        model = train_xgboost(X_train, y_train, X_val, y_val)
        print(f"Model eval score is {model.best_score}")

        return model

    @task
    def select_best_model(models: List[xgb.Booster]):
        """
        Select model that generalizes the best against the validation set, and
        write this to GCS. The best_score is an attribute of the model, and corresponds to
        the highest eval score yielded during training.
        """

        MODEL_PATH = "week-2/model.bst"
        best_model = model_selector(models)

        client = GCSHook()
        client.upload(
            bucket_name=DATASET_NORM_WRITE_BUCKET,
            object_name=MODEL_PATH,
            data=BytesIO(best_model.save_raw(raw_format="ubj")).getvalue(),
            timeout=600,
        )

    # instantiate tasks and set dependencies
    dataset_norm = read_dataset_norm()
    trained_models = format_data_and_train_model.partial(
        dataset_norm=dataset_norm
    ).expand(indices=produce_indices())
    best_model_selection = select_best_model(trained_models)

    (
        sense_features_dataset
        >> dataset_norm
        >> trained_models
        >> best_model_selection
    )


dag_run = taskflow()
