from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName("TitanicClassification").getOrCreate()

data = [
    ("male", 22, 1, 0, 7.25, 0),
    ("female", 38, 1, 0, 71.28, 1),
    ("female", 26, 0, 0, 7.92, 1),
    ("male", 35, 0, 0, 8.05, 0),
    ("male", 28, 0, 0, 8.46, 0),
    ("female", 58, 0, 0, 26.55, 1),
]
columns = ["Sex", "Age", "SibSp", "Parch", "Fare", "Survived"]
df = spark.createDataFrame(data, columns)

df = StringIndexer(inputCol="Sex", outputCol="SexIndexed").fit(df).transform(df)

features = VectorAssembler(
    inputCols=["SexIndexed", "Age", "SibSp", "Parch", "Fare"],
    outputCol="features"
).transform(df)

train, test = features.randomSplit([0.8, 0.2], seed=1)

lr = LogisticRegression(featuresCol="features", labelCol="Survived")
model = lr.fit(train)

predictions = model.transform(test)

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="Survived")
accuracy = evaluator.evaluate(predictions)
print(f"Logistic Regression AUC: {accuracy:.3f}")

spark.stop()
