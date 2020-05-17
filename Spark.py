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
    spark = SQLContext(sc)
    df = spark.read.csv(
        '/user/datasets/ecbdl14/ECBDL14_IR2.data', header=False, inferSchema=True)
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
