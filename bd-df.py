# importing required libraries
import sys, pyspark, json
from pyspark import SparkContext
from pyspark.ml import Pipeline
import pyspark.sql.types as tp
from pyspark.ml.feature import Tokenizer
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession,Row,Column
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
from pyspark.streaming import StreamingContext


sc = SparkContext("local[2]", "Sentiment")
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
	df.show()

def map_data(data):
	json_data=json.loads(data)
	list_rec = list()
	for rec in json_data:
		to_tuple = tuple(json_data[rec].values())
		list_rec .append(to_tuple)
	return list_rec 	

lines = ssc.socketTextStream("localhost",6100).map(map_data).foreachRDD(to_df)



ssc.start() 
ssc.awaitTermination(100)
ssc.stop()
