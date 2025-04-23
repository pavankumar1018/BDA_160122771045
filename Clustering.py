from pyspark.sql import SparkSession
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder.appName("GaussianClustering").getOrCreate()

data = [(0.1, 0.2), (0.3, 0.4), (9.0, 9.1), (9.2, 9.3), (5.0, 5.0), (5.2, 5.1)]
df = spark.createDataFrame(data, ["x", "y"])

assembler = VectorAssembler(inputCols=["x", "y"], outputCol="features")
df = assembler.transform(df)

gmm = GaussianMixture().setK(3).setSeed(42)
model = gmm.fit(df)

predictions = model.transform(df)
predictions.show()

for i, summary in enumerate(model.gaussiansDF.collect()):
    print(f"Cluster {i} mean = {summary['mean']}, covariance =\n{summary['cov']}\n")

spark.stop()
