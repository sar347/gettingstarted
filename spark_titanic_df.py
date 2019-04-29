# Import packages
import time
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, OneHotEncoder, VectorAssembler, IndexToString
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import *

# Creatingt Spark SQL environment
from pyspark.sql import SparkSession
from pyspark.sql import Row

spark = SparkSession.builder.appName('spark sql titanic').master("local[*]").getOrCreate()

# spark is an existing SparkSession
dataTitanic = spark.read.csv("train.csv", header=True)
# Displays the content of the DataFrame to stdout
dataTitanic.show(10)
dataTitanic.printSchema()
dataTitanic.describe().show()


# String to float on some columns of the dataset : creates a new dataset
dataTitanic = dataTitanic.select(col("Survived"), col("Sex"), col("Embarked"), col("Pclass").cast("float"),
                     col("Age").cast("float"), col("SibSp").cast("float"), col("Fare").cast("float"))
dataTitanic.printSchema()

# dropping null values
dataTitanic = dataTitanic.dropna()


# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
dataTitanic = StringIndexer(inputCol="Sex", outputCol="indexedSex").fit(dataTitanic).transform(dataTitanic)
dataTitanic = StringIndexer(inputCol="Embarked", outputCol="indexedEmbarked").fit(dataTitanic).transform(dataTitanic)

dataTitanic = StringIndexer(inputCol="Survived", outputCol="indexedSurvived").fit(dataTitanic).transform(dataTitanic)

# One Hot Encoder on indexed features
dataTitanic = OneHotEncoder(inputCol="indexedSex", outputCol="sexVec").transform(dataTitanic)
dataTitanic = OneHotEncoder(inputCol="indexedEmbarked", outputCol="embarkedVec").transform(dataTitanic)

# Feature assembler as a vector
dataTitanic = VectorAssembler(inputCols=["Pclass", "sexVec", "embarkedVec", "Age", "SibSp", "Fare"],
                        outputCol="features").transform(dataTitanic)


# Spliting in train and test set. Beware : It sorts the dataset
(trainDF, testDF) = dataTitanic.randomSplit([0.7, 0.3], seed = 42)


rf = RandomForestClassifier(labelCol="indexedSurvived", featuresCol="features")

time_start = time.time()
model_rf = rf.fit(trainDF)

time_end = time.time()
time_rf = (time_end - time_start)
print("RF takes %d s" % (time_rf))


predictions = model_rf.transform(testDF)

# Select example rows to display.
predictions.select(col("prediction"), col("probability"),).show(5)
