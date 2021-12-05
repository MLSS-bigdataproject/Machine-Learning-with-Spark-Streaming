
from pyspark.sql import SparkSession,Row,Column
from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
from pyspark.streaming import StreamingContext
from pyspark import SparkContext
import pyspark.sql.types as tp
from pyspark.ml.feature import Tokenizer
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
import sys, pyspark, json


sc = SparkContext("local[2]", "Crime")

ssc = StreamingContext(sc, 1)
sql_context=SQLContext(sc)

def dfto(data):
	if data.isEmpty():
		return
	s = SparkSession(data.context)
	data = data.collect()[0]
	cols = [f"feature{j}" for j in range(len(data[0]))]
	df = s.createDataFrame(data, cols)
	df.show()

def mapdat(data):
    lstrecord = list()
    js_dat=json.loads(data)
    for record in js_dat:
        tup = tuple(js_dat[record].values())
        lstrecord.append(tup)
    return lstrecord 	

lines = ssc.socketTextStream("localhost",6100).map(mapdat).foreachRDD(dfto)



ssc.start() 
#ssc.awaitTermination(500)
#ssc.stop()
ssc.awaitTermination()