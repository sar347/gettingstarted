# Import packages
import time
import pyspark
import csv
import numpy
from pyspark.mllib.regression import LabeledPoint
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

sc = SparkContext(master="local", appName="sparktitanic")


# load data
dataTitanic = sc.textFile("train.csv")
dataHeader = dataTitanic.first()
dataTitanic = dataTitanic.filter(lambda line: line != dataHeader).mapPartitions(lambda x: csv.reader(x))
print(dataTitanic.first())


# Data preprocessing
def transformMapper(stringWord):

    if stringWord == 'male':
        return [0]
    else:
        return [1]

dataTitanic = dataTitanic.map(lambda line: line[1:3] + transformMapper(line[4]) + line[5:11])

# filter lines with empty strings
dataTitanic = dataTitanic.filter(lambda line: line[3] != '').filter(lambda line: line[4] != '')
print(dataTitanic.take(10))


dataTitanicLP = dataTitanic.map(lambda line: LabeledPoint(line[0], [line[1:5]]))
print(dataTitanicLP.first())

# splitting dataset into train and test set
(trainData, testData) = dataTitanicLP.randomSplit([0.7, 0.3])


from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionWithSGD
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.evaluation import BinaryClassificationMetrics

# Logistic Regression model
t0 = time.time()

model_logreg = LogisticRegressionWithLBFGS.train(trainData)

tt = time.time() - t0
print("Classifier trained in {} seconds".format(round(tt,3)))

labelsAndPreds = testData.map(lambda p: (p.label, model_logreg.predict(p.features)))
# labelsAndPreds = model_logreg.predict(testData.map(lambda x: x.features))
trainAcc = labelsAndPreds.filter(lambda lp: lp[0] == lp[1]).count() / float(dataTitanicLP.count())
print("Training Accuracy = " + str(trainAcc))


## Random Forest model

time_start = time.time()

model_rf = RandomForest.trainClassifier(trainData, numClasses=2,
                                        categoricalFeaturesInfo={}, numTrees=100,
                                        featureSubsetStrategy='auto', impurity='gini', maxDepth=12,
                                        maxBins=32, seed=42)

print(model_rf.numTrees())
print(model_rf.totalNumNodes())

time_end = time.time()
time_rf = (time_end - time_start)
print("RF takes %d s" % (time_rf))

# Predictions on test set
predictions = model_rf.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
metrics = BinaryClassificationMetrics(labelsAndPredictions)

# Area under precision-recall curve & ROC curve:
print("Area under PR = %s" % metrics.areaUnderPR)
print("Area under ROC = %s" % metrics.areaUnderROC)

