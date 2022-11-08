from pyspark.ml.classification import GBTClassifier, LogisticRegression, LinearSVC
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from fraud_transaction_detection.pipeline.base import PipelineDriver
from fraud_transaction_detection.utils.database import load_data, save_data
from fraud_transaction_detection.result_engine.training import TrainingResultManager


class TrainingPipeline(PipelineDriver):

    def __init__(self, spark, config, run_id):
        self.spark = spark
        self.config = config
        self.run_id = run_id
        self.feature_data = load_data(spark, config['feature_table_path'] + run_id, config['feature_table_format'])
        self.tuned_model_dict = {}

    def hyper_param_tuning(self, train_data, test_data):
        lr = LogisticRegression(maxIter=10, predictionCol="lrPrediction",
                                rawPredictionCol="lrRawPredictionCol", probabilityCol="lrProbabilityCol")
        lr_param_grid = ParamGridBuilder() \
            .addGrid(lr.regParam, [0.1, 0.01]) \
            .build()
        lr_cross_val = CrossValidator(estimator=lr,
                                      estimatorParamMaps=lr_param_grid,
                                      evaluator=BinaryClassificationEvaluator(rawPredictionCol='lrRawPredictionCol'),
                                      numFolds=2)
        lr_cross_val_model = lr_cross_val.fit(train_data)
        test_data = lr_cross_val_model.bestModel.transform(test_data)
        self.tuned_model_dict.update({"logistic_regression": {"best_model": lr_cross_val_model.bestModel}})

        gbt = GBTClassifier(maxIter=10, predictionCol="gbtPrediction")
        gbt.setRawPredictionCol("gbtRawPredictionCol")
        gbt.setProbabilityCol("gbtProbabilityCol")
        gbt_param_grid = ParamGridBuilder() \
            .addGrid(gbt.maxDepth, [5, 10]) \
            .addGrid(gbt.maxBins, [32, 64]) \
            .build()
        gbt_cross_val = CrossValidator(estimator=gbt,
                                       estimatorParamMaps=gbt_param_grid,
                                       evaluator=BinaryClassificationEvaluator(rawPredictionCol='gbtRawPredictionCol'),
                                       numFolds=2)
        gbt_cross_val_model = gbt_cross_val.fit(train_data)
        test_data = gbt_cross_val_model.bestModel.transform(test_data)
        self.tuned_model_dict.update({"gradient_boosting_trees": {"best_model": gbt_cross_val_model.bestModel}})

        lsvc = LinearSVC(maxIter=10, predictionCol="lsvcPrediction")
        lsvc.setRawPredictionCol("lsvcRawPredictionCol")

        lsvc_param_grid = ParamGridBuilder() \
            .addGrid(lsvc.regParam, [0.1, 0.01]) \
            .build()
        lsvc_cross_val = CrossValidator(estimator=lsvc,
                                        estimatorParamMaps=lsvc_param_grid,
                                        evaluator=BinaryClassificationEvaluator(
                                            rawPredictionCol='lsvcRawPredictionCol'),
                                        numFolds=2)
        lsvc_cross_val_model = lsvc_cross_val.fit(train_data)
        test_data = lsvc_cross_val_model.bestModel.transform(test_data)
        self.tuned_model_dict.update(
            {"linear_support_vector_classifier": {"best_model": lsvc_cross_val_model.bestModel}})
        return test_data

    def model_selection(self, test_data):
        lr_evaluator = BinaryClassificationEvaluator(rawPredictionCol='lrRawPredictionCol')
        gbt_evaluator = BinaryClassificationEvaluator(rawPredictionCol='gbtRawPredictionCol')
        lsvc_evaluator = BinaryClassificationEvaluator(rawPredictionCol='lsvcRawPredictionCol')

        lr_score = lr_evaluator.evaluate(test_data)
        self.tuned_model_dict["logistic_regression"]["score"] = lr_score
        gbt_score = gbt_evaluator.evaluate(test_data)
        self.tuned_model_dict["gradient_boosting_trees"]["score"] = gbt_score
        lsvc_score = lsvc_evaluator.evaluate(test_data)
        self.tuned_model_dict["linear_support_vector_classifier"]["score"] = lsvc_score
        best_score = 0.0
        best_model_dict = {}
        for model_algo in self.tuned_model_dict.keys():
            if best_score <= self.tuned_model_dict[model_algo]["score"]:
                best_score = self.tuned_model_dict[model_algo]['score']
                best_model_dict['name'] = model_algo
                best_model_dict['model'] = self.tuned_model_dict[model_algo]['best_model']
                best_model_dict['score'] = self.tuned_model_dict[model_algo]['score']
        return best_model_dict

    def run(self):
        train_data, test_data = self.feature_data.randomSplit([self.config["train_split"], self.config["test_split"]])
        transformed_data = self.hyper_param_tuning(train_data, test_data)
        best_model_dict = self.model_selection(transformed_data)
        result_manager = TrainingResultManager()
        result_manager.process_result(transformed_data, best_model_dict)
        test_result_file_path = self.config["test_result_table_path"] + self.run_id
        save_data(transformed_data, test_result_file_path, self.config["test_result_table_format"])
        return best_model_dict
