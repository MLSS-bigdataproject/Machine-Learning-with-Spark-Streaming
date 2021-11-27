#importing all directories needed
import os
import sys, pyspark, json
from pyspark import SparkContext
from pyspark.ml import Pipeline
import pyspark.sql.types as tp
from pyspark.ml.feature import Tokenizer
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql import SparkSession,Row,Column
from pyspark.ml.feature import StringIndexer, VectorAssembler,OneHotEncoder,VectorSlicer
from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import lit
from shutil import rmtree
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression,NaiveBayes,RandomForestClassifier
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.sql.functions import *
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors


sc = SparkContext("local[2]", "Crime")
ssc = StreamingContext(sc, 1)
sql_context=SQLContext(sc)

#reading the training and testing dataset
train_df = sql_context.read.option("header",True).csv("train.csv")
test_df = sql_context.read.option("header",True).csv("test.csv")

#displaying the dataset
train_df.show()
test_df.show()

#diplaying the schema of the training dataset
train_df.printSchema() 

# one hot encoding and assembling
encoding_var = [i[0] for i in train_df.dtypes if (i[1]=='string') & (i[0]!='y')]
num_var = [i[0] for i in train_df.dtypes if ((i[1]=='int') | (i[1]=='double')) & (i[0]!='y')]
string_indexes = [StringIndexer(inputCol = c, outputCol = 'IDX_' + c, handleInvalid = 'keep') for c in encoding_var]
onehot_indexes = [OneHotEncoder(inputCols = ['IDX_' + c], outputCols = ['OHE_' + c]) for c in encoding_var]
label_indexes = StringIndexer(inputCol = 'y', outputCol = 'label', handleInvalid = 'keep')
assembler = VectorAssembler(inputCols = num_var + ['OHE_' + c for c in encoding_var], outputCol = "features")
rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed = 8464,numTrees=10, cacheNodeIds = True, subsamplingRate = 0.7)
pipe = Pipeline(stages = string_indexes + onehot_indexes + [assembler, label_indexes, rf])
mod = pipe.fit(train_df)
df2 = mod.transform(train_df)
mod.stages[-1].featureImportances


drop_list = ['Dates', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y']
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


