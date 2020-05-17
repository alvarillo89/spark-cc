from pyspark import SparkContext, SparkConf


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
    df = sc.read.csv('/user/datasets/ecbdl14/ECBDL14_IR2.data', header=True)
    df.show()


if __name__ == "__main__":
    sc = spark_init()
    select_columns(sc)


# Notas
# 5 o 2 cores -> 1gb
# lr -> collect sumary
# Comandos para copiar el fichero de datos con hdfs:
# hdfs dfs -cp /user/datasets/ecbdl14/ECBDL14_IR2.header .
# hdfs dfs -cp /user/datasets/ecbdl14/ECBDL14_IR2.data .
