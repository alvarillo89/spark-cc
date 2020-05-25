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
    for i, colname in enumerate(df.columns):
        df = df.withColumnRenamed(colname, headers[i])

    # Get the assigned columns:
    df = df.select(
        "PSSM_r1_1_K", "PSSM_r2_-1_R", "PSSM_central_2_D",
        "PSSM_central_0_A", "PSSM_r1_1_W", "PSSM_central_-1_V", "class")

    # Write new csv:
    df.write.csv('./filteredC.small.training', header=True)


def load_data(sc):
    """ Carga el csv previamente creado con la función select_columns() """
    sqlc = SQLContext(sc)
    df = sqlc.read.csv('./filteredC.small.training/',
                       header=True, inferSchema=True)
    return df


if __name__ == "__main__":
    sc = spark_init()
    select_columns(sc)


# Notas
# model.summary() -> collect
# /opt/spark-2.2.0/bin/spark-submit --master spark://hadoop-master:7077 --total-executor-cores 5 --executor-memory 1g spark-cc/Spark.py
# hdfs dfs -getmerge filteredC.small.training/ ./filteredC.small.training
