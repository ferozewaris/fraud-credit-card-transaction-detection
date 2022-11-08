import pyspark.sql.functions as F
from pyspark.ml import Transformer
from pyspark import keyword_only
from pyspark.ml.param.shared import HasInputCols, HasOutputCols, Params, Param, TypeConverters, HasThreshold
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import Window

__all__ = [
    'ColumnRemover',
    'ColumnRenamer',
    'NullThresholdRemover',
    'CategoricalColImputer',
    'DuplicateRowRemover',
    'TypeCaster'
]


class ColumnRemover(Transformer, HasInputCols, DefaultParamsReadable, DefaultParamsWritable):
    @keyword_only
    def __init__(self, inputCols=None) -> None:
        super().__init__()
        self._setDefault(inputCols=inputCols)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _transform(self, dataset):
        cols = dataset.columns
        input_cols = list(set(self.getInputCols()).intersection(cols))
        return dataset.drop(*input_cols)


class ColumnRenamer(Transformer, HasInputCols, HasOutputCols,
                           DefaultParamsReadable, DefaultParamsWritable):
    @keyword_only
    def __init__(self, inputCols=None, outputCols=None) -> None:
        if len(inputCols) != len(outputCols):
            raise ValueError("input column number is not equal to output column number")
        super().__init__()
        self._setDefault(inputCols=inputCols, outputCols=outputCols)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCols=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _transform(self, dataset):
        input_cols = self.getInputCols()
        if not all(col in list(dataset.columns) for col in input_cols):
            raise ValueError("One or more input columns are not present in dataset")
        output_cols = self.getOutputCols()

        for i in range(len(input_cols)):
            dataset = dataset.withColumnRenamed(input_cols[i], output_cols[i])
        return dataset


class NullThresholdRemover(Transformer, HasThreshold, HasInputCols,
                           DefaultParamsReadable, DefaultParamsWritable):

    @keyword_only
    def __init__(self, inputCols=None, threshold=0.3) -> None:
        super().__init__()
        self._setDefault(inputCols=inputCols, threshold=threshold)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, threshold=0.3):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _transform(self, dataset):
        threshold = self.getThreshold()
        if not threshold or threshold > 1.0 or threshold < 0.0:
            raise ValueError("Invalid threshold value")
        cols = dataset.columns
        dataset_row_count = dataset.count()
        input_cols = list(set(self.getInputCols()).intersection(cols))

        cols_null_count = dataset.select(
            [(F.count(F.when(F.col(c).isNull(), c)) / dataset_row_count).alias(c)
             for c in input_cols]).collect()

        cols_null_count = [row.asDict() for row in cols_null_count][0]
        cols_gt_th = list(
            {i for i in cols_null_count if cols_null_count[i] > threshold})
        return dataset.drop(*cols_gt_th)


class CategoricalColImputer(Transformer, HasInputCols, DefaultParamsReadable, DefaultParamsWritable):

    strategy = Param(Params._dummy(), "strategy", "strategy",  typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self, inputCols=None, strategy="mode") -> None:
        super().__init__()
        self._setDefault(inputCols=inputCols, strategy=strategy)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, strategy="mode"):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def setStrategy(self, value):
        self.setParams(strategy=value)

    def getStrategy(self):
        return self.getOrDefault(self.strategy)

    def _mode(self, dataset, cols):
        mode_dict = dataset.select(*[
            F.struct(
                F.count(c).over(Window.partitionBy(c)).alias("cnt"),
                F.col(c).alias("val")
            ).alias(c) for c in cols
        ]).agg(*[
            F.slice(
                F.expr(f"transform(sort_array(collect_set({c}), false), x -> x.val)"),
                1, 1
            )[0].alias(c) for c in cols
        ]).collect()[0].asDict()
        return dataset.na.fill(mode_dict)

    def _transform(self, dataset):

        strategy = self.getStrategy()
        cols = dataset.columns
        input_cols = list(set(self.getInputCols()).intersection(cols))
        try:
            return getattr(self, "_"+strategy)(dataset, input_cols)
        except AttributeError as e:
            raise e


class DuplicateRowRemover(Transformer, DefaultParamsReadable, DefaultParamsWritable):

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
        return dataset.dropDuplicates()


class TypeCaster(Transformer, HasInputCols, HasOutputCols,
                           DefaultParamsReadable, DefaultParamsWritable):

    inPlace = Param(Params._dummy(), "inPlace", "inPlace", typeConverter=TypeConverters.toBoolean)
    castType = Param(Params._dummy(), "castType", "castType", typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, inPlace=True, castType=None) -> None:
        if not type:
            raise ValueError("Type is not specified")
        super().__init__()
        self._setDefault(inputCols=inputCols, outputCols=outputCols, inPlace=inPlace, castType=castType)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCols=None, inPlace=True, castType=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def setInplace(self, value):
        self.setParams(inPlace=value)

    def getInplace(self):
        return self.getOrDefault(self.inPlace)

    def setCastType(self, value):
        self.setParams(castType=value)

    def getCastType(self):
        return self.getOrDefault(self.castType)

    def _transform(self, dataset):
        in_place = self.getInplace()
        input_cols = self.getInputCols()
        cast_type = self.getCastType()
        if not in_place:
            output_cols = self.getOutputCols()
            if len(output_cols) != len(input_cols):
                raise ValueError("number of output columns are not equal to input columns")
            for i in range(len(input_cols)):
                dataset = dataset.withColumnRenamed(output_cols[i], F.col(input_cols[i]).cast(cast_type))
        else:
            input_cols = list(set(input_cols).intersection(dataset.columns))
            for col in input_cols:
                dataset = dataset.withColumn(col, F.col(col).cast(cast_type))

        return dataset
