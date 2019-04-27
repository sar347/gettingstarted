from pyspark import SparkConf, SparkContext

sc = SparkContext(master="local", appName="Spark Demo")
print(sc.textFile("/Users/sumitrane163/server/spark-2.4.1-bin-hadoop2.7/README.md").first())

#commit test