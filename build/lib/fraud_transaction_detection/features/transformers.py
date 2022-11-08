import pyspark.sql.functions as F
from pyspark.ml import Transformer
from pyspark import keyword_only
from pyspark.ml.param.shared import HasInputCol, Params, Param, TypeConverters, HasLabelCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from fraud_transaction_detection.features.constants import TIME_FEATURE_STRATEGY_LIST

__all__ = [
    'TimeFeaturizer',
    'CustomOneHotEncoder',
    'CustomVectorAssembler',
    'OverSampler'
]


class TimeFeaturizer(Transformer, HasInputCol,
                     DefaultParamsReadable, DefaultParamsWritable):
    strategy = Param(Params._dummy(), "strategy", "strategy", typeConverter=TypeConverters.toString)
    outputColsPrefix = Param(Params._dummy(), "outputColsPrefix", "outputColsPrefix",
                             typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self, inputCol=None, outputColsPrefix=None, strategy="hour") -> None:
        if strategy not in TIME_FEATURE_STRATEGY_LIST:
            raise AttributeError(f"Strategy {strategy} not supported")
        super().__init__()
        self._setDefault(inputCol=inputCol, outputColsPrefix=outputColsPrefix, strategy=strategy)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputColsPrefix=None, strategy="hour"):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def setOutputColsPrefix(self, value):
        self.setParams(outputColsPrefix=value)

    def getOutputColsPrefix(self):
        return self.getOrDefault(self.outputColsPrefix)

    def setStrategy(self, value):
        self.setParams(strategy=value)

    def getStrategy(self):
        return self.getOrDefault(self.strategy)

    def _day_of_week(self, dataset, output_cols_prefix):
        dataset = dataset.withColumn(output_cols_prefix + "Cos", F.cos(2 * 3.141 *
                                                                       F.dayofweek("transactionDateTime") / 7))
        dataset = dataset.withColumn(output_cols_prefix + "Sin", F.sin(2 * 3.141 *
                                                                       F.dayofweek("transactionDateTime") / 7))
        return dataset

    def _hour(self, dataset, output_cols_prefix):
        dataset = dataset.withColumn(output_cols_prefix + "Cos", F.cos(2 * 3.141 * F.hour("transactionDateTime") / 24))
        dataset = dataset.withColumn(output_cols_prefix + "Sin", F.sin(2 * 3.141 * F.hour("transactionDateTime") / 24))
        return dataset

    def _transform(self, dataset):

        input_col = self.getInputCol()
        if input_col not in list(dataset.columns):
            raise ValueError(f"column {input_col} not present in dataset")
        output_cols_prefix = self.getOutputColsPrefix()
        strategy = self.getStrategy()
        try:
            return getattr(self, "_" + strategy)(dataset, output_cols_prefix)
        except AttributeError:
            raise AttributeError(f"strategy {strategy} not implemented")


class CustomOneHotEncoder(Transformer, DefaultParamsReadable, DefaultParamsWritable):

    @keyword_only
    def __init__(self) -> None:
        super().__init__()
        self._setDefault()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _transform(self, dataset):
        categorical_cols = [field for (field, dataType) in dataset.dtypes
                           if dataType in ["string"]]
        index_output_cols = [x + "Index" for x in categorical_cols]
        ohe_output_cols = [x + "OHE" for x in categorical_cols]
        string_indexer = StringIndexer(inputCols=categorical_cols,
                                      outputCols=index_output_cols,
                                      handleInvalid="skip")
        ohe_encoder = OneHotEncoder(inputCols=index_output_cols,
                                   outputCols=ohe_output_cols)
        pipeline = Pipeline(stages=[string_indexer, ohe_encoder])
        pipeline_model = pipeline.fit(dataset)
        encoded_dataset = pipeline_model.transform(dataset)
        encoded_dataset = encoded_dataset.drop(*index_output_cols)
        encoded_dataset = encoded_dataset.drop(*categorical_cols)
        return encoded_dataset


class CustomVectorAssembler(Transformer, DefaultParamsReadable, DefaultParamsWritable):

    @keyword_only
    def __init__(self) -> None:
        super().__init__()
        self._setDefault()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _transform(self, dataset):
        cols = dataset.columns
        cols.remove("label")
        assembler = VectorAssembler(inputCols=cols, outputCol="features")
        return assembler.transform(dataset)


class OverSampler(Transformer, HasLabelCol,
                  DefaultParamsReadable, DefaultParamsWritable):
    majorityClass = Param(Params._dummy(), "majorityClass", "majorityClass", typeConverter=TypeConverters.toFloat)
    minorityClass = Param(Params._dummy(), "minorityClass", "minorityClass", typeConverter=TypeConverters.toFloat)

    @keyword_only
    def __init__(self, labelCol='label', majorityClass=None, minorityClass=None) -> None:
        super().__init__()
        self._setDefault(labelCol=labelCol, majorityClass=majorityClass, minorityClass=minorityClass)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, labelCol='label', majorityClass=None, minorityClass=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def setMajorityClass(self, value):
        self.setParams(majorityClass=value)

    def getMajorityClass(self):
        return self.getOrDefault(self.majorityClass)

    def setMinorityClass(self, value):
        self.setParams(minorityClass=value)

    def getMinorityClass(self):
        return self.getOrDefault(self.minorityClass)

    def _transform(self, dataset):
        label = self.getLabelCol()
        majority_class = self.getMajorityClass()
        minority_class = self.getMinorityClass()
        df_majority = dataset.filter(dataset[label] == majority_class)
        df_minority = dataset.filter(dataset[label] == minority_class)

        a_count = df_majority.count()
        b_count = df_minority.count()
        ratio = a_count / b_count

        df_minority_oversampled = df_minority.sample(withReplacement=True, fraction=ratio, seed=1)
        return df_majority.unionAll(df_minority_oversampled)
