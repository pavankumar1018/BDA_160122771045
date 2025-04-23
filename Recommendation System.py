from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("BookRecommendation").getOrCreate()

data = [
    (1, 201, 4),
    (1, 202, 5),
    (2, 201, 3),
    (2, 203, 4),
    (3, 202, 2),
    (3, 203, 5),
]
columns = ["userId", "bookId", "rating"]
ratings = spark.createDataFrame(data, columns)

train, test = ratings.randomSplit([0.75, 0.25], seed=42)

als = ALS(
    maxIter=10,
    regParam=0.05,
    userCol="userId",
    itemCol="bookId",
    ratingCol="rating",
    coldStartStrategy="drop"
)
model = als.fit(train)

predictions = model.transform(test)
predictions.show()

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"Book Recommendation RMSE = {rmse:.3f}")

recommendations = model.recommendForAllUsers(2)
recommendations.show(truncate=False)

spark.stop()
