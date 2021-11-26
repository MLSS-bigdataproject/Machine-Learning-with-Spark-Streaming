#importing all directories needed
import os
import sys, pyspark, json
from pyspark import SparkContext
import numpy as np
from pyspark.ml import Pipeline
import pyspark.sql.types as tp
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator,MulticlassClassificationEvaluator,BinaryClassificationEvaluator
from pyspark.ml.feature import Tokenizer
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql import SparkSession,Row,Column
from pyspark.ml.feature import StringIndexer, VectorAssembler,OneHotEncoder,VectorSlicer,StopWordsRemover, Word2Vec, RegexTokenizer,StandardScaler,RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import lit
from sklearn.preprocessing import MinMaxScaler
from pyspark.ml.classification import LogisticRegression,NaiveBayes,RandomForestClassifier
from pyspark.sql.functions import col
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.sql.functions import *
from pyspark.ml.linalg import Vectors
from sklearn.feature_selection import SelectKBest,chi2
from pyspark.sql.types import FloatType
from pyspark.mllib.evaluation import MulticlassMetrics
import pyspark.sql.functions as F

sc = SparkContext("local[2]", "Crime")
ssc = StreamingContext(sc, 1)
sql_context=SQLContext(sc)

#reading the training and testing dataset
train_df = sql_context.read.option("header",True).csv("train.csv")
test_df = sql_context.read.option("header",True).csv("test.csv")

#displaying the dataset
train_df.show()
test_df.show()

train_df = train_df.withColumn('X', train_df['X'].cast(FloatType()))
train_df = train_df.withColumn('Y', train_df['Y'].cast(FloatType()))

#diplaying the schema of the training dataset
train_df.printSchema() 

'''
#done for float columns
features = ['X', 'Y']
x = train_df.select([column for column in train_df.columns if column in features]).collect()
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
y = train_df.select(["Category"]).collect()
label_stringIdx = StringIndexer(inputCol = "Category", outputCol = "label")
indexed = label_stringIdx.fit(train_df).transform(train_df)
test = SelectKBest(score_func=chi2, k=2)
fit = test.fit(x, y)
print(fit.scores_)
#x and y columns are somewhat usefull with x giving 45% usability and y giving less than 10%

features = ['Dates', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address','Category']
x = train_df.select([column for column in train_df.columns if column in features]).collect()
y = train_df.select(["Category"]).collect()
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(train_df) for column in features]
pipeline = Pipeline(stages=indexers)
df_r = pipeline.fit(train_df).transform(train_df)
d_list = ['Dates', 'Descript','Category','Category_index','DayOfWeek', 'PdDistrict', 'Resolution', 'Address','X','Y','Dates_index']
x = df_r.select([column for column in df_r.columns if column not in d_list]).collect()
y = df_r.select(["Category_index"]).collect()
print(x[:][1])
#print(y)
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(x, y)
print(fit.scores_)
#even though the values are high these columns will be dropped because during streaming and upon using these features,accuracy becomes bad
'''
drop_list = ['Dates', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'Descript' ,'X', 'Y']
train_df = train_df.select([column for column in train_df.columns if column not in drop_list])
train_df.show(5)

#diplaying the schema of the training dataset
train_df.printSchema() 

train_df = train_df.sample(0.001)

train_df.groupBy("Category").count().orderBy(col("count").desc()).show()
train_df.groupBy("Descript").count().orderBy(col("count").desc()).show()

# regular expression tokenizer
regexTokenizer = RegexTokenizer(inputCol="Descript", outputCol="words", pattern="\\W")
# stop words
add_stopwords = ["http","https","amp","rt","t","c","the"] 
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)
# bag of words count
countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)

label_stringIdx = StringIndexer(inputCol = "Category", outputCol = "label")
pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_stringIdx])
pipelineFit = pipeline.fit(train_df)
dataset = pipelineFit.transform(train_df)
dataset.show(5)

(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testData.count()))

#Logistic Regression
print("Logistic Regression:")
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(trainingData)
predictions = lrModel.transform(testData)
predictions.filter(predictions['prediction'] == 0) \
    .select("Descript","Category","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)
    
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
LogRegAcc = evaluator.evaluate(predictions)

truepos = predictions.select('prediction').where(predictions["label"]=='1' & predictions["prediction"]=='1').count().show()
#trueneg = []
#falsepos = []
#falseneg = []

'''
pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_stringIdx])
pipelineFit = pipeline.fit(train_df)
dataset = pipelineFit.transform(train_df)
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.1, 0.3, 0.5]) # regularization parameter
             .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.2]) # Elastic Net Parameter (Ridge = 0)
#            .addGrid(model.maxIter, [10, 20, 50]) #Number of iterations
#            .addGrid(idf.numFeatures, [10, 100, 1000]) # Number of features
             .build())
# Create 5-fold CrossValidator
cv = CrossValidator(estimator=lr, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=5)
cvModel = cv.fit(trainingData)

predictions = cvModel.transform(testData)
# Evaluate best model
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator.evaluate(predictions)
'''
#Naive Bayes
print("Naive Bayes:")
nb = NaiveBayes(smoothing=1)
model = nb.fit(trainingData)
predictions = model.transform(testData)
predictions.filter(predictions['prediction'] == 0) \
    .select("Descript","Category","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
nativeBayesAcc = evaluator.evaluate(predictions)

#Random Forest
print("Random Forest")
rf = RandomForestClassifier(labelCol="label", \
                            featuresCol="features", \
                            numTrees = 100, \
                            maxDepth = 4, \
                            maxBins = 32)
rfModel = rf.fit(trainingData)
predictions = rfModel.transform(testData)
predictions.filter(predictions['prediction'] == 0) \
    .select("Descript","Category","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)
    

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
RandomForestAcc = evaluator.evaluate(predictions)

print("Logistic Regression Accuracy: ",LogRegAcc)
print("Naive Bayes Accuracy: ", nativeBayesAcc)
print("Random Forest Accuracy: ",RandomForestAcc)

#cross-validation tests
