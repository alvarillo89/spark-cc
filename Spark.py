###################################################################################################
# -*- coding: utf-8 -*-
# Author: Álvaro Fernández García
# Machine Learning con Spark
###################################################################################################

from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import FloatType, IntegerType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LinearSVC

###################################################################################################


def spark_init():
    """ Inicializa el Spark Context """
    conf = SparkConf().setAppName("Practica 4 - Alvaro Fernandez Garcia")
    sc = SparkContext(conf=conf)
    sc.setLogLevel('WARN')
    return sc

###################################################################################################


def select_columns(sc):
    """ Carga el fichero de datos y selecciona las columnas que se
    deben utilizar para el modelo.

    Keywords arguments:
    sc -- Spark Context
    """

    # Obtener los nombres de la cabeceras:
    headers = sc.textFile(
        "/user/datasets/ecbdl14/ECBDL14_IR2.header").collect()
    headers = list(filter(lambda x: "@inputs" in x, headers))[0]
    headers = headers.replace(",", "").strip().split()
    del headers[0]  # Borrar "@input"
    headers.append("class")

    # Leer los datos:
    sqlc = SQLContext(sc)
    df = sqlc.read.csv(
        '/user/datasets/ecbdl14/ECBDL14_IR2.data', header=False, inferSchema=True)

    # Asignar la cabecera a cada columna:
    for i, colname in enumerate(df.columns):
        df = df.withColumnRenamed(colname, headers[i])

    # Obtener las columnas asignadas:
    df = df.select(
        "PSSM_r1_1_K", "PSSM_r2_-1_R", "PSSM_central_2_D",
        "PSSM_central_0_A", "PSSM_r1_1_W", "PSSM_central_-1_V", "class")

    # Escribir un nuevo csv:
    df.write.csv('./filteredC.small.training', header=True)

###################################################################################################


def load_data(sc):
    """ Carga el csv previamente creado con la funcion select_columns() """
    sqlc = SQLContext(sc)

    schema = StructType([
        StructField("PSSM_r1_1_K", FloatType()),
        StructField("PSSM_r2_-1_R", FloatType()),
        StructField("PSSM_central_2_D", FloatType()),
        StructField("PSSM_central_0_A", FloatType()),
        StructField("PSSM_r1_1_W", FloatType()),
        StructField("PSSM_central_-1_V", FloatType()),
        StructField("class", FloatType())
    ])

    df = sqlc.read.csv('./filteredC.small.training/',
                       header=True, schema=schema)
    return df

###################################################################################################


def preprocess(df, should_undersample, scaler=None):
    """ Escala los datos y balancea usando Random Undersample (RUS) """
    # Agrupar las caracteristicas para poder usarlas en la MLlib:
    assembler = VectorAssembler(inputCols=[
        "PSSM_r1_1_K", "PSSM_r2_-1_R", "PSSM_central_2_D",
        "PSSM_central_0_A", "PSSM_r1_1_W", "PSSM_central_-1_V"
    ], outputCol="features")

    out = assembler.transform(df).select("features", "class")

    # Random Undersample (RUS)
    # Antes: POS = 550.140, NEG = 1.100.591
    # Despues: POS = 550.140, NEG = 549.668
    if should_undersample:
        positive = out.filter(out["class"] == 1.0)
        negative = out.filter(out["class"] == 0.0)
        fraction = float(positive.count()) / float(negative.count())
        negative = negative.sample(
            withReplacement=False, fraction=fraction, seed=89)
        out = negative.union(positive)

    # Escalar:
    if scaler == None:
        scaler = StandardScaler(withMean=True, withStd=True,
                                inputCol="features", outputCol="scaled_features")
        scaler = scaler.fit(out)
        out = scaler.transform(out)
    else:
        out = scaler.transform(out)

    return out, scaler

###################################################################################################


def train_test_split(data, test_size):
    """ Divide (estratificando) los datos en el conjunto de train y test """
    train = 1. - test_size
    zeros = data.filter(data["class"] == 0.)
    ones = data.filter(data["class"] == 1.)
    train0, test0 = zeros.randomSplit([train, test_size], seed=89)
    train1, test1 = ones.randomSplit([train, test_size], seed=89)
    train = train0.union(train1)
    test = test0.union(test1)
    return train, test

###################################################################################################


def validate(estimator, train, grid):
    """ Elige los hiperparámetros de "estimator" a partir de "grid" y utilizando el 
    20% de los datos de "train" como partición de validación. Como métrica de 
    comparación, utiliza AUC.
    """
    tvs = TrainValidationSplit(estimator=estimator,
                               estimatorParamMaps=grid,
                               evaluator=BinaryClassificationEvaluator(
                                   labelCol="class"),
                               trainRatio=0.8,
                               seed=89)

    model = tvs.fit(train)
    for i, item in enumerate(model.getEstimatorParamMaps()):
        grid = ["%s: %s" % (p.name, str(v)) for p, v in item.items()]
        print(grid, model.getEvaluator().getMetricName(),
              model.validationMetrics[i])

###################################################################################################


def train_logistic_regresion(train):
    """ Entrena un modelo de regresión logística probando varios hiperparámetros en
    un partición de validación 
    """
    lr = LogisticRegression(featuresCol="scaled_features",
                            labelCol="class", standardization=False)

    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.1, 0.01, 0.001]) \
        .addGrid(lr.maxIter, [5, 10, 15]) \
        .build()

    validate(estimator=lr, train=train, grid=paramGrid)

###################################################################################################


def train_random_forest(train):
    """ Entrena un modelo de Random Forest probando varios hiperparámetros en
    un partición de validación 
    """
    rf = RandomForestClassifier(
        featuresCol="features", labelCol="class", seed=89)

    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [5, 10, 15, 20]) \
        .addGrid(rf.impurity, ['entropy', 'gini']) \
        .build()

    validate(estimator=rf, train=train, grid=paramGrid)

###################################################################################################


def train_SVM(train):
    """ Entrena un modelo de SVM Lineal probando varios hiperparámetros en
    un partición de validación 
    """
    svm = LinearSVC(featuresCol="scaled_features",
                    labelCol="class", standardization=False)

    paramGrid = ParamGridBuilder() \
        .addGrid(svm.regParam, [0.1, 0.01, 0.001]) \
        .addGrid(svm.maxIter, [5, 10, 15]) \
        .build()

    validate(estimator=svm, train=train, grid=paramGrid)

###################################################################################################


if __name__ == "__main__":
    sc = spark_init()
    # select_columns(sc)
    df = load_data(sc)

    # Train 1.650.731 Test = 412.456
    train, test = train_test_split(df, test_size=0.2)
    train, scaler = preprocess(train, should_undersample=True)
    test, _ = preprocess(test, should_undersample=False, scaler=scaler)

    # Descomentar para ver la elección de hiperparámetros:
    # train_logistic_regresion(train)
    # train_random_forest(train)
    # train_SVM(train)

    # Test:
    # Primero entrenamos el mejor modelo con todos los datos de
    # train (sin reservar nada para validar):
    rf = RandomForestClassifier(
        featuresCol="features", labelCol="class", numTrees=15, impurity='gini', seed=89)
    model = rf.fit(train)
    evaluator = BinaryClassificationEvaluator()
    test = test.withColumnRenamed('class', 'label')
    evaluation = evaluator.evaluate(model.transform(test))
    print("[TEST] Area bajo la curva ROC:", evaluation)
    # ('[TEST] Area bajo la curva ROC:', 0.5391668462877764)
