# importing required libraries
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
#spark = SparkSession(sc)
ssc = StreamingContext(sc, 1)
sql_context=SQLContext(sc)

def to_df(data):
	if data.isEmpty():
		return
	ss = SparkSession(data.context)
	data = data.collect()[0]
	columns = [f"feature{i}" for i in range(len(data[0]))]
	df = ss.createDataFrame(data, columns)
	drop_list = ['feature0','feature3','feature4','feature5','feature6','feature7','feature8']
	train_df = df.select([column for column in df.columns if column not in drop_list])
	train_df.show(5)
	train_df.groupBy("feature1").count().orderBy(col("count").desc()).show()
	train_df.groupBy("feature2").count().orderBy(col("count").desc()).show()
	# regular expression tokenizer
	regexTokenizer = RegexTokenizer(inputCol="feature2", outputCol="words", pattern="\\W")
	#stop words
	add_stopwords = ["http","https","amp","rt","t","c","the"] 
	stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)
	# bag of words count
	countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)
	label_stringIdx = StringIndexer(inputCol = "feature1", outputCol = "label")
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
	predictions.filter(predictions['prediction'] == 0).select("feature2","feature1","probability","label","prediction").orderBy("probability", ascending=False).show(n = 10, truncate = 30)
	evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
	LogRegAcc = evaluator.evaluate(predictions)
	#Naive Bayes
	print("Naive Bayes:")
	nb = NaiveBayes(smoothing=1)
	model = nb.fit(trainingData)
	predictions = model.transform(testData)
	predictions.filter(predictions['prediction'] == 0).select("feature2","feature1","probability","label","prediction").orderBy("probability", ascending=False).show(n = 10, truncate = 30)
	evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
	nativeBayesAcc = evaluator.evaluate(predictions)

    	#Random Forest
	print("Random Forest")
	rf = RandomForestClassifier(labelCol="label",featuresCol="features",numTrees = 100,maxDepth = 4,maxBins = 32)
	rfModel = rf.fit(trainingData)
	predictions = rfModel.transform(testData)
	predictions.filter(predictions['prediction'] == 0).select("feature2","feature1","probability","label","prediction").orderBy("probability", ascending=False).show(n = 10, truncate = 30)
	evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
	RandomForestAcc = evaluator.evaluate(predictions)

	print("Accuracy of different models used:")
	print("Logistic Regression Accuracy: ",LogRegAcc)
	print("Naive Bayes Accuracy: ", nativeBayesAcc)
	print("Random Forest Accuracy: ",RandomForestAcc)

def map_data(data):
	json_data=json.loads(data)
	list_rec = list()
	for rec in json_data:
		to_tuple = tuple(json_data[rec].values())
		list_rec .append(to_tuple)
	return list_rec 	

lines = ssc.socketTextStream("localhost",6100).map(map_data).foreachRDD(to_df)

ssc.start() 
#ssc.awaitTermination(100)
#ssc.stop()
ssc.awaitTermination()
