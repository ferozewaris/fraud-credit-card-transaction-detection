from fraud_transaction_detection.pipeline.base import PipelineDriver
from fraud_transaction_detection.data_processing.transformers import ColumnRemover, ColumnRenamer, \
    NullThresholdRemover, CategoricalColImputer, DuplicateRowRemover, TypeCaster

from fraud_transaction_detection.features.transformers import CustomOneHotEncoder, CustomVectorAssembler, OverSampler
from fraud_transaction_detection.utils.database import save_data, load_data

from pyspark.ml.feature import Imputer, SQLTransformer, StringIndexer
from pyspark.ml import Pipeline


class PreprocessingPipeline(PipelineDriver):

    def __init__(self, spark, config, run_id):
        self.spark = spark
        self.config = config
        self.run_id = run_id
        self.raw_data = load_data(spark, config['raw_data_path'], config['raw_data_file_type'], config['raw_data_delimiter'])

    def create_cleaning_pipeline(self):
        initial_column_remover = ColumnRemover(inputCols=self.config['preprocessing_cols_remove'])
        duplicate_remover = DuplicateRowRemover()
        target_column_renamer = ColumnRenamer(inputCols=[self.config["target_col_name"]], outputCols=['label'])
        null_value_remover = NullThresholdRemover(
            inputCols=self.config['numerical_variables'] +
                      self.config['categorical_columns'] +
                      self.config['boolean_columns'] +
                      self.config['date_time_columns']
        )
        numerical_col_imputer = Imputer(inputCols=self.config['numerical_variables'],
                                        outputCols=self.config['numerical_variables'],
                                        strategy=self.config["numerical_imputation_strategy"])
        categorical_col_imputer = CategoricalColImputer(inputCols=self.config['categorical_columns'],
                                                        strategy=self.config["categorical_imputation_strategy"])
        bool_col_imputer = CategoricalColImputer(inputCols=self.config['boolean_columns'],
                                                 strategy=self.config["boolean_imputation_strategy"])

        return [
            initial_column_remover, duplicate_remover, target_column_renamer, null_value_remover,
            numerical_col_imputer, categorical_col_imputer, bool_col_imputer
        ]

    def create_feature_engineering_pipeline(self):
        wrong_cvv_featurizer = SQLTransformer(
            statement=f"SELECT *, CASE WHEN {self.config['card_cvv_col_name']} != {self.config['entered_cvv_col_name']} "
                      f"THEN 1 ELSE 0 END as {self.config['is_entered_wrong_cvv_col_name']} FROM __THIS__"
        )
        cvv_columns_remover = ColumnRemover(
            inputCols=[self.config['card_cvv_col_name'], self.config['entered_cvv_col_name']])
        address_change_featurizer = SQLTransformer(
            statement=f"SELECT *, CASE WHEN {self.config['account_open_date_col_name']} != "
                      f"{self.config['date_last_address_change_col_name']} THEN 1 ELSE 0 END as "
                      f"{self.config['is_address_change_col_name']} FROM __THIS__")
        account_dates_columns_remover = ColumnRemover(
            inputCols=[self.config['account_open_date_col_name'], self.config['date_last_address_change_col_name']])
        card_expiry_transformer = SQLTransformer(
            statement="Select *, Substring({0}, 1,Charindex('/', {0})-1) as {1}, Substring({0}, "
                      "Charindex('/', {0})+1) as {2} from __THIS__".format(
                self.config["current_exp_date_col_name"], self.config["card_expiry_month_col_name"],
                self.config["card_expiry_year_col_name"]))
        exp_col_remover = ColumnRemover(inputCols=[self.config["current_exp_date_col_name"]])
        expiry_col_caster = TypeCaster(
            inputCols=[self.config["card_expiry_month_col_name"], self.config["card_expiry_year_col_name"]],
            castType="int")
        expiry_month_cyclical = SQLTransformer(
            statement="select *, COS({0} * 2 * 3.14) as {0}COS, SIN({0} * 2 * 3.14) as {0}SIN FROM __THIS__".format(
                self.config["card_expiry_month_col_name"]))
        expiry_year_cyclical = SQLTransformer(
            statement="select *, COS({0} *2*3.14) as {0}COS, SIN({0} *2*3.14) as {0}SIN FROM __THIS__".format(
                self.config["card_expiry_year_col_name"]))
        extra_colmns_remover = ColumnRemover(
            inputCols=[self.config["card_expiry_year_col_name"], self.config["card_expiry_month_col_name"]])
        ohe_encoder = CustomOneHotEncoder()
        bool_cols_caster = TypeCaster(inputCols=self.config['boolean_columns'] + ["label"], castType="string")
        index_bool_cols = [x + "Index" for x in self.config['boolean_columns']]
        index_bool_cols = index_bool_cols + ['labelIndex']
        bool_string_indexer = StringIndexer(inputCols=self.config['boolean_columns'] + ["label"],
                                            outputCols=index_bool_cols,
                                            handleInvalid="skip")
        bool_cols_remover = ColumnRemover(inputCols=self.config['boolean_columns'] + ["label"])
        label_col_renamer = ColumnRenamer(inputCols=["labelIndex"], outputCols=["label"])
        vector_assembler = CustomVectorAssembler()
        over_sampler = OverSampler(majorityClass=0.0, minorityClass=1.0)
        return [
            wrong_cvv_featurizer, cvv_columns_remover, address_change_featurizer, account_dates_columns_remover,
            card_expiry_transformer, exp_col_remover, expiry_col_caster, expiry_month_cyclical, expiry_year_cyclical,
            extra_colmns_remover, ohe_encoder, bool_cols_caster, bool_string_indexer, bool_cols_remover,
            label_col_renamer, vector_assembler, over_sampler
        ]

    def run(self):
        cleaning_pipeline_stages = self.create_cleaning_pipeline()
        feature_engineering_pipeline_stages = self.create_feature_engineering_pipeline()
        all_stages = cleaning_pipeline_stages + feature_engineering_pipeline_stages
        feature_pipeline = Pipeline(stages=all_stages)
        feature_pipeline_model = feature_pipeline.fit(self.raw_data)
        feature_data = feature_pipeline_model.transform(self.raw_data)
        feature_table_path = self.config["feature_table_path"] + self.run_id
        return save_data(feature_data, feature_table_path, self.config["feature_table_format"])
