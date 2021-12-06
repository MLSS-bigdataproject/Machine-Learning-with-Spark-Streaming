# importing required libraries
#importing all directories needed
import os
import sys, pyspark, json
from pyspark import SparkContext
import numpy as np
import csv
import matplotlib.pyplot as plt
from pyspark.ml import Pipeline
import pyspark.sql.types as tp
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator,MulticlassClassificationEvaluator,BinaryClassificationEvaluator,ClusteringEvaluator
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
from sklearn.cluster import MiniBatchKMeans
from pyspark.ml.linalg import Vectors
from sklearn.feature_selection import SelectKBest,chi2
from pyspark.sql.types import FloatType
from pyspark.mllib.evaluation import MulticlassMetrics
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import lit,monotonically_increasing_id, row_number
from pyspark.sql import Window


sc = SparkContext("local[2]", "Crime")
#spark = SparkSession(sc)
ssc = StreamingContext(sc, 1)
sql_context=SQLContext(sc)


def dfto(data):
	if data.isEmpty():
		return
	s = SparkSession(data.context)
	data = data.collect()[0]
	#converting data into dataframe for easy usage
	cols = [f"feature{j}" for j in range(len(data[0]))]
	colm = ['Dates','Category','Descript', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address','X', 'Y']
	df = s.createDataFrame(data, colm)
	
	
	#converting into float
	df = df.withColumn('X', df['X'].cast(FloatType()))
	df = df.withColumn('Y', df['Y'].cast(FloatType()))



	#done for float columns
	#scaling of values
	features = ['X', 'Y']
	x = df.select([column for column in df.columns if column in features]).collect()
	scaler = MinMaxScaler()
	scaler.fit(x)
	x = scaler.transform(x)
	y = df.select(["Category"]).collect()
	label_stringIdx = StringIndexer(inputCol = "Category", outputCol = "label")
	#feature selection
	indexed = label_stringIdx.fit(df).transform(df)
	test = SelectKBest(score_func=chi2, k=2)
	fit = test.fit(x, y)
	print(fit.scores_)
	#x and y columns are useful with x giving 45% usability and y giving less than 10%
	
	
	
	#doing for categorical data
	features = ['Dates', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address','Category']
	x = df.select([column for column in df.columns if column in features]).collect()
	y = df.select(["Category"]).collect()
	indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in features]
	pipeline = Pipeline(stages=indexers)
	df_r = pipeline.fit(df).transform(df)
	d_list = ['Dates', 'Descript','Category','Category_index','DayOfWeek', 'PdDistrict', 'Resolution', 'Address','X','Y','Dates_index']
	x = df_r.select([column for column in df_r.columns if column not in d_list]).collect()
	y = df_r.select(["Category_index"]).collect()
	print(x[:][1])
	#print(y)
	test = SelectKBest(score_func=chi2, k=4)
	fit = test.fit(x, y)
	print(fit.scores_)
	#even though the values are high these columns will be dropped because during streaming and upon using these features,accuracy becomes bad
	
	
	
	#removing all unnecessary columns
	drop_list = ['Dates','DayOfWeek','PdDistrict','Resolution','Address','X','Y']
	train_df = df.select([column for column in df.columns if column not in drop_list])
	train_df.show(5)
	train_df.groupBy("Category").count().orderBy(col("count").desc()).show()
	train_df.groupBy("Descript").count().orderBy(col("count").desc()).show()
	# regular expression tokenizer
	regexTokenizer = RegexTokenizer(inputCol="Descript", outputCol="words", pattern="\\W")
	#stop words
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
	
	#print("Training Data count: " + str(trainingData.count()))
	#print("Validation Dataset count: " + str(testData.count()))
	
	
	
	
	#Logistic Regression
	print("Logistic Regression:")
	lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
	lrModel = lr.fit(trainingData)
	predictions = lrModel.transform(testData)
	predictions.filter(predictions['prediction'] == 0).select("Descript","Category","probability","label","prediction").orderBy("probability", ascending=False).show(n = 10, truncate = 30)
	evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
	LogRegAcc = evaluator.evaluate(predictions)
	lrModel.write().overwrite().save("models/lr")
	
	
	#data visualization
	#check ideal iterations and ploting graph
	iteracc = []
	for i in range(0,20,5):
	    lr = LogisticRegression(maxIter=i, regParam=0.3, elasticNetParam=0)
	    lrModela = lr.fit(trainingData)
	    predictions = lrModela.transform(testData)
	    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
	    iteracc.append(evaluator.evaluate(predictions))
	
	fig = plt.figure(figsize = (10, 5))
	plt.plot(iteracc, color = 'r')
	plt.xlabel("no of iterations")
	plt.ylabel("accuracy")
	plt.title("iterations vs accuracy")
	plt.xticks(rotation=90)
	plt.savefig("plots/LogisticRegression/Itervsacc/iteracc1.jpg")
	
	

	#checking ideal reg parameters and ploting graph
	regparamacc = []
	arr = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
	for i in arr:
    	    lr = LogisticRegression(maxIter=14, regParam=i, elasticNetParam=0)
    	    lrModelo = lr.fit(trainingData)
    	    predictions = lrModelo.transform(testData)
    	    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    	    regparamacc.append(evaluator.evaluate(predictions))
    	
	fig = plt.figure(figsize = (10, 5))
	plt.plot(regparamacc, color = 'r')
	plt.xlabel("no of params")
	plt.ylabel("accuracy")
	plt.title("params vs accuracy")
	plt.xticks(rotation=90)
	plt.savefig("plots/LogisticRegression/paramvsacc/paramvsacc.jpg")
	
	#Implemented calculations for Precision,Recall and F1-Score for Logistic Regression
	tp_lr = predictions[(predictions.label == 1) & (predictions.prediction == 1)].count()
	tn_lr = predictions[(predictions.label == 0) & (predictions.prediction == 0)].count()
	fp_lr = predictions[(predictions.label == 0) & (predictions.prediction == 1)].count()
	fn_lr = predictions[(predictions.label == 1) & (predictions.prediction == 0)].count()
	r_lr = float(tp_lr)/(tp_lr + fn_lr)
	p_lr = float(tp_lr) / (tp_lr + fn_lr)
	fone_lr = float(tp_lr) / (float(tp_lr) + 0.5*(fp_lr+fn_lr))
	
	
	
	
	#Implemented lr with tf-idf and also applied hyperparameter tuning
	print("logistic regression with tf-idf:")
	hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
	idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5) 
	pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, hashingTF, idf, label_stringIdx])
	pipelineFit = pipeline.fit(train_df)
	dataset = pipelineFit.transform(train_df)
	(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)
	lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
	lrModel = lr.fit(trainingData)
	predictions = lrModel.transform(testData)
	predictions.filter(predictions['prediction'] == 0) \
    	.select("Descript","Category","probability","label","prediction") \
    	.orderBy("probability", ascending=False) \
    	.show(n = 10, truncate = 30)
	evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
	logregidfacc = evaluator.evaluate(predictions)
	
	
	
	
	#hyperparameter tuning	
	pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_stringIdx])
	pipelineFit = pipeline.fit(train_df)
	dataset = pipelineFit.transform(train_df)
	(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)
	lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)

	paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.1, 0.3, 0.5]) # regularization parameter
#          .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.2]) # Elastic Net Param###eter (Ridge = 0)
            .addGrid(lr.maxIter, [10, 20, 50]) #Number of iterations
#            .addGrid(idf.numFeatures, [10, 100, 1000]) # Number of features
             .build())
	cv = CrossValidator(estimator=lr, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=5)
	cvModel = cv.fit(trainingData)
	predictions = cvModel.transform(testData)
	evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
	hyperlogregidfacc = evaluator.evaluate(predictions)
	
	
	
	
	
	
	#Naive Bayes
	print("\n")
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
	model.write().overwrite().save("models/nb")
	
	
	#data visualization
	#check ideal smoothing coefficient and ploting graph
	naivebacc = []
	for i in range(5):
    	    nbl = NaiveBayes(smoothing=i)
    	    model = nb.fit(trainingData)
    	    predictions = model.transform(testData)  
    	    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    	    naivebacc.append(evaluator.evaluate(predictions))
	
	fig = plt.figure(figsize = (10, 5))
	plt.plot(naivebacc, color = 'r')
	plt.xlabel("smoothing")
	plt.ylabel("accuracy")
	plt.title("smoothing vs accuracy")
	plt.xticks(rotation=90)
	plt.savefig("plots/NaiveBayes/smoothingvsacc/smooth1.jpg")


	#plotting distribution of predictions
	count = predictions.select("prediction").groupBy("prediction").count().rdd.flatMap(lambda x: x).collect()
	count = [count[i] for i in range(len(count)) if i % 2 != 0]
	temp = predictions.select("prediction").distinct().rdd.flatMap(lambda x: x).collect()

	fig = plt.figure(figsize = (10, 5))
	plt.bar(temp, count,width = 0.4)
	plt.xlabel("predictions")
	plt.ylabel("count")
	plt.title("distribution of prediction labels")
	plt.savefig("plots/NaiveBayes/predictionvsactual/pred1.jpg")
	
	

	#Implemented calculations for Precision,Recall and F1-Score for Naive Bayes
	tp_nb = predictions[(predictions.label == 1) & (predictions.prediction == 1)].count()
	tn_nb = predictions[(predictions.label == 0) & (predictions.prediction == 0)].count()
	fp_nb = predictions[(predictions.label == 0) & (predictions.prediction == 1)].count()
	fn_nb = predictions[(predictions.label == 1) & (predictions.prediction == 0)].count()
	r_nb = float(tp_nb)/(tp_nb + fn_nb)
	p_nb = float(tp_nb) / (tp_nb + fp_nb)
	fone_nb = float(tp_nb) / (float(tp_nb) + 0.5*(fp_nb+fn_nb))
	
	
	
	

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
	#Accuracy
	RandomForestAcc = evaluator.evaluate(predictions)
	rfModel.write().overwrite().save("models/rbf")



	#data visualization
	#variation of accuracy with changing no of depth:
	forestacc = []
	for i in range(5):
    	    rf = RandomForestClassifier(labelCol="label",featuresCol="features",numTrees = 100,maxDepth = i,maxBins = 32)
    	    rfModela = rf.fit(trainingData)
    	    predictions = rfModela.transform(testData)  
    	    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    	    forestacc.append(evaluator.evaluate(predictions))
	
	fig = plt.figure(figsize = (10, 5))
	plt.plot(forestacc, color = 'r')
	plt.xlabel("depth")
	plt.ylabel("accuracy")
	plt.title("depth vs accuracy")
	plt.xticks(rotation=90)
	plt.savefig("plots/RandomForest/depthvsacc/depth1.jpg")
	
	
	#variation of accuracy with changing no of trees:
	forestacc = []
	for i in range(10,100,10):
    	    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees = i, maxDepth = 4, maxBins = 32)
    	    rfModela = rf.fit(trainingData)
    	    predictions = rfModela.transform(testData)  
    	    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    	    forestacc.append(evaluator.evaluate(predictions))
	
	fig = plt.figure(figsize = (10, 5))
	plt.plot(forestacc, color = 'r')
	plt.xlabel("tree")
	plt.ylabel("accuracy")
	plt.title("depth vs accuracy")
	plt.xticks(rotation=90)
	plt.savefig("plots/RandomForest/treesvsacc/tree1.jpg")

	#Implemented calculations for Precision,Recall and F1-Score for Random Forest
	tp_rf = predictions[(predictions.label == 1) & (predictions.prediction == 1)].count()
	tn_rf = predictions[(predictions.label == 0) & (predictions.prediction == 0)].count()
	fp_rf = predictions[(predictions.label == 0) & (predictions.prediction == 1)].count()
	fn_rf = predictions[(predictions.label == 1) & (predictions.prediction == 0)].count()
	r_rf = float(tp_rf)/(tp_rf + fn_rf)
	p_rf = float(tp_rf) / (tp_rf + fp_rf)
	fone_rf = float(tp_rf) / (float(tp_rf) + 0.5*(fp_rf+fn_rf))


    
	#applied hyperparameter tuning on model
	pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_stringIdx])
	pipelineFit = pipeline.fit(train_df)
	dataset = pipelineFit.transform(train_df)
	(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)
	rbf = RandomForestClassifier(labelCol="label", \
                            featuresCol="features", \
                            numTrees = 100, \
                            maxDepth = 4, \
                            maxBins = 32)

	paramGrid = (ParamGridBuilder()
             .addGrid(rbf.numTrees, [10,100,1000]) # trees
             .addGrid(rbf.maxDepth, [1 ,4, 10]) # depth 
#            .addGrid(idf.numFeatures, [10, 100, 1000]) # Number of features
             .build())
	cv = CrossValidator(estimator=rbf, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=5)
	cvModel = cv.fit(trainingData)
	predictions = cvModel.transform(testData)
	evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
	hyperlograndfor = evaluator.evaluate(predictions)
	




	#Metrics of all Machine Learning Models
	print("Logistic Regression Metrics:")
	print("Logistic Regression Accuracy: ",hyperlogregidfacc)
	print("recall = ", r_lr)
	print("precision = ", p_lr)
	print("F1-score = ",fone_lr)


	print("Naive Bayes Metrics:")
	print("Naive Bayes Accuracy: ", nativeBayesAcc)
	print("recall = ", r_nb)
	print("precision = ", p_nb)
	print("F1-score = ",fone_nb)


	print("Random Forest Metrics: ")
	print("Random Forest Accuracy: ",RandomForestAcc)
	print("recall = ", r_rf)
	print("precision = ", p_rf)
	print("F1-score = ",fone_rf)
	
	
	#adding the accuracy after each stream
	'''
	acc = [hyperlogregidfacc,nativeBayesAcc,RandomForestAcc]
	file = open('stream1.csv', 'a+', newline ='')
	with file:
	    write = csv.writer(file)
	    write.writerows(acc)
	file.close()
	'''
	
	
	#plotting over the three models
	fig = plt.figure(figsize = (10, 5))
	plt.plot(iteracc, color = 'r')
	plt.plot(naivebacc, color = 'b')
	plt.plot(forestacc, color = 'g')
	plt.xlabel("model")
	plt.ylabel("accuracy")
	plt.title("comparision of models")
	plt.xticks(rotation=90)
	plt.legend(["Logistic Regression","Naive Bayes" ,"Random Forest"], loc ="lower right")
	plt.savefig("plots/comparisionbetweenmodels/compare1.jpg")
	
	
	
	#kmeanclustering
	drop_list = ['Dates','DayOfWeek','PdDistrict','Resolution','Address',"Descript","Category"]
	train_df = df.select([column for column in df.columns if column not in drop_list])
	data = train_df.select(col("X"),col("Y")).collect()
	kmeans = MiniBatchKMeans(n_clusters=39, random_state=0, batch_size=6)
	predictions = kmeans.fit_predict(data).tolist()
	b = sql_context.createDataFrame([(l,) for l in predictions], ['Cluster'])
	train_df = train_df.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))
	b = b.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))
	final_df = train_df.join(b, train_df.row_idx == b.row_idx).drop("row_idx")
	df2 = final_df.withColumn('Cluster',final_df['Cluster'].cast(IntegerType()))
	df2.show()
	ctr = df2.select("Cluster").groupBy("Cluster").count().rdd.flatMap(lambda x: x).collect()
	ctr = [ctr[i] for i in range(len(ctr)) if i % 2 != 0]
	t = df2.select("Cluster").distinct().rdd.flatMap(lambda x: x).collect()
	
	
	
	clusarr = []
	ct = []
	
	

	#varying number of clusters
	for i in range(10,40,10):
    	    data = train_df.select(col("X"),col("Y")).collect()
    	    kmeans = MiniBatchKMeans(n_clusters=i, random_state=0, batch_size=6)
    	    predictions = kmeans.fit_predict(data).tolist()
    	    b = sql_context.createDataFrame([(l,) for l in predictions], ['Cluster'])
    	    train_df = train_df.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))
    	    b = b.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))
    	    final_df = train_df.join(b, train_df.row_idx == b.row_idx).drop("row_idx")
    	    df2 = final_df.withColumn('Cluster',final_df['Cluster'].cast(IntegerType()))
    	    ctr = df2.select("Cluster").groupBy("Cluster").count().rdd.flatMap(lambda x: x).collect()
    	    ctr = [ctr[i] for i in range(len(ctr)) if i % 2 != 0]
    	    ct.append(ctr)
    	    t = df2.select("Cluster").distinct().rdd.flatMap(lambda x: x).collect()
    	    clusarr.append(t)
  
  
	#data visualization  
	for i in range(len(clusarr)):
    	    fig = plt.figure(figsize = (10, 5))
    	    plt.bar(clusarr[i], ct[i],width = 0.4)
    	    plt.xlabel("clusters")
    	    plt.ylabel("count")
    	    plt.title("distribution of clusters")
    	    plt.savefig("plots/kmeans/variationofclusters/test"+ str(i)+ ".jpg")


	clusarr1 = []
	ct1 = []


	#varying number of batch
	for i in range(1,6):
	    data = train_df.select(col("X"),col("Y")).collect()
	    kmeans = MiniBatchKMeans(n_clusters=39, random_state=0, batch_size=i)
	    predictions = kmeans.fit_predict(data).tolist()
	    b = sql_context.createDataFrame([(l,) for l in predictions], ['Cluster'])
	    train_df = train_df.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))
	    b = b.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))
	    final_df = train_df.join(b, train_df.row_idx == b.row_idx).drop("row_idx")
	    df2 = final_df.withColumn('Cluster',final_df['Cluster'].cast(IntegerType()))
	    ctr = df2.select("Cluster").groupBy("Cluster").count().rdd.flatMap(lambda x: x).collect()
	    ctr = [ctr[i] for i in range(len(ctr)) if i % 2 != 0]
	    ct1.append(ctr)
	    t = df2.select("Cluster").distinct().rdd.flatMap(lambda x: x).collect()
	    clusarr1.append(t)
    	    
    	    
	#data visualization
	for i in range(len(clusarr1)):
    	    fig = plt.figure(figsize = (10, 5))
    	    plt.bar(clusarr1[i], ct1[i],width = 0.4)
    	    plt.xlabel("clusters")
    	    plt.ylabel("count")
    	    plt.title("variation of batch size")
    	    plt.savefig("plots/kmeans/variationofbatch/test"+ str(i)+ ".jpg")



	clusarr2 = []
	ct2= []
	
	
	#varying number of random states
	for i in range(5):
    	    data = train_df.select(col("X"),col("Y")).collect()
    	    kmeans = MiniBatchKMeans(n_clusters=39, random_state=i, batch_size=6)
    	    predictions = kmeans.fit_predict(data).tolist()
    	    b = sql_context.createDataFrame([(l,) for l in predictions], ['Cluster'])
    	    train_df = train_df.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))
    	    b = b.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))
    	    final_df = train_df.join(b, train_df.row_idx == b.row_idx).drop("row_idx")
    	    df2 = final_df.withColumn('Cluster',final_df['Cluster'].cast(IntegerType()))
    	    ctr = df2.select("Cluster").groupBy("Cluster").count().rdd.flatMap(lambda x: x).collect()
    	    ctr = [ctr[i] for i in range(len(ctr)) if i % 2 != 0]
    	    ct2.append(ctr)
    	    t = df2.select("Cluster").distinct().rdd.flatMap(lambda x: x).collect()
    	    clusarr2.append(t)
    	    
    	    
	#data visualization
	for i in range(len(clusarr2)):
	    fig = plt.figure(figsize = (10, 5))
	    plt.bar(clusarr2[i], ct2[i],width = 0.4)
	    plt.xlabel("clusters")
	    plt.ylabel("count")
	    plt.title("variation of random state")
	    plt.savefig("plots/kmeans/variationofrandomstate/test"+ str(i)+ ".jpg")


def mapdat(data):
    lstrecord = list()
    js_dat=json.loads(data)
    for record in js_dat:
        tup = tuple(js_dat[record].values())
        lstrecord.append(tup)
    return lstrecord 	

lines = ssc.socketTextStream("localhost",6100).map(mapdat).foreachRDD(dfto)

ssc.start() 
#ssc.awaitTermination(100)
#ssc.stop()
ssc.awaitTermination()

