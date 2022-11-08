
def load_data(spark, path, file_type, delimiter=','):
    try:
        if file_type == "parquet":
            return spark.read.parquet(path)
        return spark.read.format(file_type) \
            .option("inferSchema", "true") \
            .option("header", "true") \
            .option("sep", delimiter) \
            .load(path)
    except Exception as e:
        raise FileNotReadException(f"There is an error while reading the file {e!r}")


def save_data(dataset, path, file_format):
    try:
        if file_format == "parquet":
            dataset.write.parquet(path)
        else:
            dataset\
                .write.format(file_format)\
                .save(path)
        return path
    except Exception as e:
        raise FileNotSaveException(f"There is an error while saving the file {e!r}")


class FileNotSaveException(Exception):
    pass


class FileNotReadException(Exception):
    pass
