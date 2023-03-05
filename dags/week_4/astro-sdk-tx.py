import pandas as pd
from airflow.models.dag import DAG
from airflow.utils import timezone

import astro.sql as aql
from astro.files import File
from astro.table import Metadata, Table

from include.transformer_packages import (
    DEFAULT_ARGS,
    BQ_DATASET_NAME,
    time_columns,
    filepaths,
)


@aql.dataframe
def extract_nonzero_columns(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out columns that have only 0 or null values by first
    calling fillna(0) and only selecting columns that have any non-zero elements
    """

    filled_df = input_df.fillna(0)
    print(f"=> Number of Columns in Input data : {len(filled_df.columns.to_list())}")
    dataset_with_any_zeros = filled_df[filled_df.columns[(filled_df == 0).any()]]
    # drop cols where zeros were spotted.
    cleaned_dataset = input_df.drop(dataset_with_any_zeros.columns, axis=1)
    print(
        f"=> Number of Columns Left after cleanup : {len(cleaned_dataset.columns.to_list())}"
    )
    return cleaned_dataset


@aql.transform
def convert_timestamp_columns(input_table: Table, data_type: str):
    """
    Returns a SQL statement that selects the input table elements,
    and casts the time column specified in 'time_columns' to TIMESTAMP
    """
    return f"SELECT CAST({data_type} as TIMESTAMP) as {input_table.name}_time, * FROM {{{{input_table}}}}"


@aql.transform
def join_tables(generation_table: Table, weather_table: Table):  # skipcq: PYL-W0613
    """
    Joins `generation_table` and `weather_table` tables on their transformed time cols to create an output table
    """
    return f"""SELECT g.*, w.* 
              FROM {{{{generation_table}}}} AS g 
              JOIN {{{{weather_table}}}} AS w 
              ON g.generation_transformed_time = w.weather_transformed_time 
            """
    # The last line in the Join is hardcoded since the cols were renamed after type changes.
    # This is to make sure the table can be queried without errors and both original cols remain after the inner join


# Use this as DAG documentation in UI.
dag_docs = """
    ### Astro SDK Transform DAG

    This DAG performs four operations using the Astro SDK abstractions:

    - 1. Loads parquet files from GCS into BigQuery, referenced by a Table object using `aql.load_file`
    - 2. Extracts nonzero columns from an input table/dataframe, using a custom Python function extending `aql.dataframe`
    - 3. Converts the timestamp column from that table, using a custom SQL statement extending `aql.transform`
    - 4. Joins the two tables produced at step 3 for each datatype on time
    """


with DAG(
    dag_id="astro_sdk_transform_dag",
    schedule_interval=None,
    start_date=timezone.datetime(2022, 1, 1),
    default_args=DEFAULT_ARGS,
    doc_md=dag_docs,
) as dag:
    output_tables = []
    for dataset, path in filepaths.items():
        load_task = aql.load_file(
            task_id=f"load_{dataset}_data",
            input_file=File(path, conn_id="google_cloud_default"),
            # output_table :This was omitted to allow the file to be loaded into a tmp table.
        )

        biqguery_table = Table(
            name=f"{dataset}_transformed",
            metadata=Metadata(schema=BQ_DATASET_NAME),
            conn_id="google_cloud_default",
        )

        transform_task = extract_nonzero_columns(
            input_df=load_task, output_table=biqguery_table
        )

        type_conversion_task = convert_timestamp_columns(
            input_table=transform_task, data_type=time_columns.get(dataset)
        )

        output_tables.append(type_conversion_task)

    merged_table = Table(
        name=f"gen_weather_merged",
        metadata=Metadata(schema=BQ_DATASET_NAME),
        conn_id="google_cloud_default",
    )

    join_tables(output_tables[0], output_tables[1], output_table=merged_table)

    # Cleans up all temporary tables produced by the SDK
    # This task waits until all other tasks are complete before cleaning up the tmp tables.
    aql.cleanup()
