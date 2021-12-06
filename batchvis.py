import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("batch.csv")


fig = plt.figure(figsize = (10, 5))
plt.plot(df["LogRegAcc"], color = 'r')
plt.plot(df["RandomForestAcc"], color = 'b')
plt.plot(df["NaiveBayesAcc"], color = 'g')
plt.xlabel("batch size")
plt.ylabel("accuracy")
plt.title("comparision of accuracy with varying batch size")
plt.xticks(rotation=90)
plt.legend(["Logistic Regression","Random Forest","Naive Bayes"], loc ="lower right")
plt.savefig("accuracygraph")


fig = plt.figure(figsize = (10, 5))
plt.plot(df["LogRegPre"], color = 'r')
plt.plot(df["RandomForestPre"], color = 'b')
plt.plot(df["NaiveBayesPre"], color = 'g')
plt.xlabel("batch size")
plt.ylabel("Precision")
plt.title("comparision of Precision with varying batch size")
plt.xticks(rotation=90)
plt.legend(["Logistic Regression" ,"Random Forest","Naive Bayes"], loc ="lower right")
plt.savefig("precisiongraph")


fig = plt.figure(figsize = (10, 5))
plt.plot(df["LogRecRe"], color = 'r')
plt.plot(df["RandomForestRe"], color = 'b')
plt.plot(df["NaiveBayesRe"], color = 'g')
plt.xlabel("batch size")
plt.ylabel("Recall")
plt.title("comparision of Recall with varying batch size")
plt.xticks(rotation=90)
plt.legend(["Logistic Regression" ,"Random Forest","Naive Bayes"], loc ="lower right")
plt.savefig("recallgraph")
