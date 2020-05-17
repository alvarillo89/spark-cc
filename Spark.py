from pyspark import SparkContext, SparkConf, SQLContext


def spark_init():
    """ Inicializa el Spark Context """
    conf = SparkConf().setAppName("Practica 4 - Alvaro Fernandez Garcia")
    sc = SparkContext(conf=conf)
    return sc


def select_columns(sc):
    """ Carga el fichero de datos y selecciona las columnas que se
    deben utilizar para el modelo.

    Keywords arguments:
    sc -- Spark Context
    """
    # headers = headers.filter(lambda line: "@inputs" in line)
    # Get Headers name:
    headers = sc.textFile(
        "/user/datasets/ecbdl14/ECBDL14_IR2.header").collect()
    headers = list(filter(lambda x: "@inputs" in x, headers))[0]
    headers = headers.replace(",", "").strip().split()
    del headers[0]  # Remove "@input"
    headers.append("class")

    # Read data:
    sqlc = SQLContext(sc)
    df = sqlc.read.csv(
        '/user/datasets/ecbdl14/ECBDL14_IR2.data', header=False, inferSchema=True)

    # Set headers names:
    old_schema = df.schema
    for old, new in zip(old_schema.fields, headers):
        df.withColumnRenamed(old, new)

    df.printSchema()


if __name__ == "__main__":
    sc = spark_init()
    select_columns(sc)


# Notas
# model.summary() -> collect
# Comandos para copiar el fichero de datos con hdfs:
# hdfs dfs -cp /user/datasets/ecbdl14/ECBDL14_IR2.header .
# hdfs dfs -cp /user/datasets/ecbdl14/ECBDL14_IR2.data .
# /opt/spark-2.2.0/bin/spark-submit --master spark://hadoop-master:7077 --total-executor-cores 5 --executor-memory 1g wordcount.py
