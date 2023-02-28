from datetime import datetime
import pandas as pd
from typing import List
from airflow.operators.empty import EmptyOperator
from airflow.decorators import dag, task, task_group
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.providers.google.cloud.operators.bigquery import (
    BigQueryCreateEmptyDatasetOperator,
    BigQueryCreateExternalTableOperator,
    BigQueryCreateEmptyTableOperator,
)

from include.transformer_packages import (
    extract_data,
    produce_select_statement,
    PROJECT_ID,
    DESTINATION_BUCKET,
    BQ_DATASET_NAME,
)

DATA_TYPES = ["generation", "weather"]

doc_md = """
    ### Data Warehouse Transform DAG
    `Description:`  An _ELT_ DAG that unzips datasets , loads to **GCS** as Parquet Files and Transforms in **BiqQuery** using federated queries.
    
    This DAG performs four operations:
    
    - 1. Extracts zip file into two dataframes
    - 2. Loads these dataframes into parquet files on GCS, with valid column names
    - 3. Builds external tables on top of these parquet files
    - 4. Builds normalized views on top of the external tables
    - 5. Builds a joined view on top of the normalized views, joined on time
    """


@dag(
    schedule_interval=None,
    start_date=datetime(2021, 1, 1),
    catchup=False,
    doc_md=doc_md,
)
def data_warehouse_transform_dag():
    @task
    def extract() -> List[pd.DataFrame]:
        """
        #### Extract task
        A simple task that loads each file in the zipped file into a dataframe,
        building a list of dataframes that is returned
        """
        filename = "/usr/local/airflow/dags/data/energy-consumption-generation-prices-and-weather.zip"
        datasets = extract_data(file_path=filename)
        return datasets

    @task
    def load(unzip_result: List[pd.DataFrame]):
        """
        #### Load task
        A simple "load" task that takes in the result of the "extract" task, formats
        columns to be BigQuery-compliant, and writes data to GCS Bucket.
        """
        client = GCSHook().get_conn()
        bucket = client.get_bucket(DESTINATION_BUCKET)

        for index, df in enumerate(unzip_result):
            df.columns = df.columns.str.replace(" ", "_")
            df.columns = df.columns.str.replace("/", "_")
            df.columns = df.columns.str.replace("-", "_")
            bucket.blob(f"week-3/{DATA_TYPES[index]}.parquet").upload_from_string(
                df.to_parquet(), "text/parquet"
            )
            print(df.dtypes)

    @task_group
    def create_bigquery_dataset():
        """
        Creates a BigQuery Dataset if it does not already exist.
        """
        BigQueryCreateEmptyDatasetOperator(
            task_id="create_bigquery_dataset_if_not_exists", dataset_id=BQ_DATASET_NAME
        )
        print(f"=> Dataset `{BQ_DATASET_NAME}` created successfully")

    @task_group
    def create_external_tables():
        """
        Creates two external tables, one for each data type, referencing the data stored in GCS in PARQUET format.
        """

        for data_artifact in DATA_TYPES:
            try:
                BigQueryCreateExternalTableOperator(
                    task_id=f"create_{data_artifact}_table",
                    table_resource={
                        "type": "EXTERNAL",
                        "tableReference": {
                            "projectId": f"{PROJECT_ID}",
                            "datasetId": f"{BQ_DATASET_NAME}",
                            "tableId": data_artifact,
                        },
                        "description": f"Table containing {data_artifact} data",
                        "externalDataConfiguration": {
                            "sourceUris": [
                                f"gs://{DESTINATION_BUCKET}/week-3/{data_artifact}.parquet"
                            ],
                            "sourceFormat": "PARQUET",
                        },
                    },
                )

                print(
                    f"=> Table `{PROJECT_ID}.{BQ_DATASET_NAME}.{data_artifact}` created Successfully. "
                )
            except Exception as e:
                print(f"** Error while creating External Table {data_artifact}.")
                raise e

    @task_group
    def produce_normalized_views():
        """
        Produces normalized views for data sources in BIGQUERY Tables.
        Uses a helper function called produce_select_statement to create the query that would be materialized as a view on BIGQUERY.
        """
        for table_name in DATA_TYPES:
            select_query = produce_select_statement(
                dataset_name=BQ_DATASET_NAME, table_name=table_name
            )
            BigQueryCreateEmptyTableOperator(
                task_id=f"create_{table_name}_normalized_view",
                dataset_id=BQ_DATASET_NAME,
                table_id=f"{table_name}_view",
                table_resource={
                    "type": "VIEW",
                    "tableReference": {
                        "projectId": f"{PROJECT_ID}",
                        "datasetId": f"{BQ_DATASET_NAME}",
                        "tableId": f"{table_name}_view",
                    },
                    "description": f"View containing {table_name} data",
                    "view": {"query": select_query, "useLegacySql": False},
                },
            )
            print(
                f"=> Normalized View `{BQ_DATASET_NAME}.{table_name}_view` created successfully."
            )

    # view_creation = create_view.expand(table_name=DATA_TYPES)
    # This didn't work when i tried using a task decorator to map over the datatypes.

    @task_group
    def produce_joined_view():
        """
        Produces a view that joins the two normalized views on their respective time columns.
        """

        joined_query = f"""SELECT g.*, w.* FROM {BQ_DATASET_NAME}.{DATA_TYPES[0]}_view AS g 
                        JOIN {BQ_DATASET_NAME}.{DATA_TYPES[1]}_view AS w
                        ON g.generation_time = w.weather_time"""
        joined_table_name = "_".join(name for name in DATA_TYPES) + "_view"
        BigQueryCreateEmptyTableOperator(
            task_id="create_joined_view",
            dataset_id=BQ_DATASET_NAME,
            table_id=joined_table_name,
            view={"query": joined_query, "useLegacySql": False},
        )
        print(
            f"=> Joined View for `{DATA_TYPES[0]}` and `{DATA_TYPES[1]}` created successfully"
        )

    unzip_task = extract()
    load_task = load(unzip_task)
    create_bigquery_dataset_task = create_bigquery_dataset()
    load_task >> create_bigquery_dataset_task
    external_table_task = create_external_tables()
    create_bigquery_dataset_task >> external_table_task
    normal_view_task = produce_normalized_views()
    external_table_task >> normal_view_task
    joined_view_task = produce_joined_view()
    normal_view_task >> joined_view_task


data_warehouse_transform_dag = data_warehouse_transform_dag()
