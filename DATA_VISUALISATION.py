import numpy as np 
import pandas as pd 

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import squarify

# for providing path
import os

data = pd.read_csv('crime/train.csv')
print(data.shape)

print(data.head())

print(data.describe())

#print(data.isnull().sum())


# different categories of crime

fig = plt.figure(figsize = (10, 5))
plt.rcParams['figure.figsize'] = (30,150)
plt.style.use('dark_background')

sns.countplot(data['Category'], palette = 'gnuplot')

plt.title('Major crimes', fontweight = 30, fontsize = 20)
plt.xticks(rotation = 90)
plt.savefig("majorcrimes.jpg")

#plotting a tree map
fig = plt.figure(figsize = (10, 5))
y = data['Category'].value_counts().head(25)
    
plt.rcParams['figure.figsize'] = (15, 15)
plt.style.use('fivethirtyeight')

color = plt.cm.magma(np.linspace(0, 1, 15))
squarify.plot(sizes = y.values, label = y.index, alpha=.8, color = color)
plt.title('Tree Map for Top 25 Crimes', fontsize = 20)

plt.axis('off')
plt.savefig("treemap.jpg")


# Regions with count of crimes graph
fig = plt.figure(figsize = (10, 5))
plt.rcParams['figure.figsize'] = (20, 9)
plt.style.use('seaborn')

color = plt.cm.ocean(np.linspace(0, 1, 15))
data['Address'].value_counts().head(15).plot.bar(color = color, figsize = (10,20))

plt.title('Top 15 Regions in Crime',fontsize = 20)

plt.xticks(rotation = 90)
plt.savefig("countofcrimes.jpg")


# Regions with count of crimes piechart
fig = plt.figure(figsize = (10, 5))
plt.style.use('seaborn')

data['DayOfWeek'].value_counts().head(15).plot.pie(figsize = (15, 8), explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1))

plt.title('Crime count on each day',fontsize = 20)

plt.xticks(rotation = 90)
plt.savefig("piechart.jpg")



# Resolutions for crimes
fig = plt.figure(figsize = (10, 5))
plt.style.use('seaborn')

color = plt.cm.winter(np.linspace(0, 10, 20))
data['Resolution'].value_counts().plot.bar(color = color, figsize = (15, 8))

plt.title('Resolutions for Crime',fontsize = 20)
plt.xticks(rotation = 90)
plt.savefig("Resolutions.jpg")



#crimes in each months
fig = plt.figure(figsize = (10, 5))
data['Dates'] = pd.to_datetime(data['Dates'])

data['Month'] = data['Dates'].dt.month

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (15, 8)

sns.countplot(data['Month'], palette = 'autumn',)
plt.title('Crimes in each Months', fontsize = 20)
plt.savefig("crimesineachmonths.jpg")


# checking the time at which crime occurs mostly

#color = plt.cm.twilight(np.linspace(0, 5, 100))
#data['Time'].value_counts().head(20).plot.bar(color = color, figsize = (15, 9))

#plt.title('Distribution of crime over the day', fontsize = 20)
#plt.show()


#district vs category of crime
fig = plt.figure(figsize = (10, 5))
df = pd.crosstab(data['Category'], data['PdDistrict'])
color = plt.cm.Greys(np.linspace(0, 1, 10))

df.div(df.sum(1).astype(float), axis = 0).plot.bar(stacked = True, color = color, figsize = (18, 12))
plt.title('District vs Category of Crime', fontweight = 30, fontsize = 20)

plt.xticks(rotation = 90)
plt.savefig("districtvscategoryofcrime.jpg")











