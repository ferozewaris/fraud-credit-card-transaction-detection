import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark_test import assert_pyspark_df_equal
from fraud_transaction_detection.data_processing.transformers import \
    NullThresholdRemover


@pytest.fixture(scope='module')
def spark():
    return SparkSession.builder.appName('fraud-detection-testing').getOrCreate()


def test_null_threshold_remover(spark):

    mock_df_schema = StructType([StructField("transactionAmount", FloatType(), True),
                                 StructField("creditLimit", FloatType(),
                                             True),
                                 StructField('merchantName', StringType(),
                                             True),
                                 StructField('merchantCountryCode', StringType(), True),
                                 StructField('posConditionCode', IntegerType(), True)])
    mock_df_data = [[88.21, None, None, None, 6],
                    [88.21, None, None, 'US', 6],
                    [88.21, 452.3, None, 'US', None],
                    [88.21, 452.3, 'Uber', 'US', 6],
                    [88.21, 452.3, 'Uber', 'US', 6],
                    [88.21, 452.3, 'Uber', 'US', None],
                    [88.21, 452.3, 'Uber', 'US', 6],
                    [88.21, None, 'Uber', 'US', 6],
                    [88.21, 452.3, 'Uber', 'US', 6],
                    [88.21, 452.3, 'Uber', 'US', None]]
    mock_df = spark.createDataFrame(mock_df_data, schema=mock_df_schema)
    expected_df_schema = StructType([StructField("transactionAmount", FloatType(),
                                                 True),
                                 StructField('merchantCountryCode', StringType(), True)])
    expected_df_data = [[88.21, None],
                        [88.21, 'US'],
                        [88.21, 'US'],
                        [88.21, 'US'],
                        [88.21, 'US'],
                        [88.21, 'US'],
                        [88.21, 'US'],
                        [88.21, 'US'],
                        [88.21, 'US'],
                        [88.21, 'US']]
    expected_df = spark.createDataFrame(expected_df_data,
                                        schema=expected_df_schema)
    null_remove = NullThresholdRemover(inputCols=mock_df.columns,
                                       threshold=0.2)
    transformed_df = null_remove.transform(mock_df)
    assert_pyspark_df_equal(expected_df, transformed_df)

def test_no_input_cols(spark):
    mock_df_schema = StructType([StructField("transactionAmount", FloatType(), True),
                                 StructField("creditLimit", FloatType(),
                                             True),
                                 StructField('merchantName', StringType(),
                                             True),
                                 StructField('merchantCountryCode', StringType(), True),
                                 StructField('posConditionCode', IntegerType(), True)])
    mock_df_data = [[88.21, None, None, None, 6],
                    [88.21, None, None, 'US', 6],
                    [88.21, 452.3, None, 'US', None],
                    [88.21, 452.3, 'Uber', 'US', 6],
                    [88.21, 452.3, 'Uber', 'US', 6],
                    [88.21, 452.3, 'Uber', 'US', None],
                    [88.21, 452.3, 'Uber', 'US', 6],
                    [88.21, None, 'Uber', 'US', 6],
                    [88.21, 452.3, 'Uber', 'US', 6],
                    [88.21, 452.3, 'Uber', 'US', None]]
    mock_df = spark.createDataFrame(mock_df_data, schema=mock_df_schema)
    null_remove = NullThresholdRemover(inputCols=[],
                                       threshold=0.2)
    transformed_df = null_remove.transform(mock_df)
    assert_pyspark_df_equal(mock_df, transformed_df)

def test_invalid_threshold(spark):
    mock_df_schema = StructType([StructField("transactionAmount", FloatType(), True),
                                 StructField("creditLimit", FloatType(),
                                             True),
                                 StructField('merchantName', StringType(),
                                             True),
                                 StructField('merchantCountryCode', StringType(), True),
                                 StructField('posConditionCode', IntegerType(), True)])
    mock_df_data = [[88.21, None, None, None, 6],
                    [88.21, None, None, 'US', 6],
                    [88.21, 452.3, None, 'US', None],
                    [88.21, 452.3, 'Uber', 'US', 6],
                    [88.21, 452.3, 'Uber', 'US', 6],
                    [88.21, 452.3, 'Uber', 'US', None],
                    [88.21, 452.3, 'Uber', 'US', 6],
                    [88.21, None, 'Uber', 'US', 6],
                    [88.21, 452.3, 'Uber', 'US', 6],
                    [88.21, 452.3, 'Uber', 'US', None]]
    mock_df = spark.createDataFrame(mock_df_data, schema=mock_df_schema)

    with pytest.raises(ValueError):
        null_remove = NullThresholdRemover(inputCols=mock_df.columns,
                                           threshold=1.2)
        transformed_df = null_remove.transform(mock_df)