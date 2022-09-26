# %%
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
# from pathlib import Path
import pandas as pd
import numpy as np
# import os
from sklearn.decomposition import PCA

# %%
production_mode = True
local_mode = False

# %%
# conf = SparkConf().setAppName("reduction") \
#     .set("spark.driver.memory", "30g") \
#     .set("spark.executor.memory", "30g")
# sc = SparkContext.getOrCreate(conf)

sc = SparkContext()
spark = SparkSession(sc)

origin_path = 'F:/' if local_mode else 's3a://dr.hadinono/OC/P8/'

csv_dir = origin_path+'fruits-360/CSV/'
csv_separate_dir = csv_dir+'Separate/*'

# Create the data folders to store the outputs
# for name in os.listdir(img_dir[:-2]):
#     Path(output_dir+name).mkdir(parents=True, exist_ok=True)
# Path(csv_separate_dir+name).mkdir(parents=True, exist_ok=True)

# %%
# Read the images-csv files
# df = spark.read.options(delimiter=",", header=True,
#                         maxCharsPerColumn=-1, maxColumns=100*100*3+1).csv(csv_separate_dir)
df = spark.read.options(delimiter=";", header=True,
                        maxCharsPerColumn=-1).csv(csv_separate_dir)

# %%
# Keep the labels
labels = df.select('label')
if not production_mode:
    labels = labels.limit(5)
labels = labels.rdd.map(lambda x: {'label': x.label}).collect()
labels = pd.DataFrame(labels)

# %%
# Convert data to matrix
# rows = df.drop('label')
rows = df.select('features')
# rows = rows.select('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10')
if not production_mode:
    rows = rows.limit(5)
# rows = rows.rdd.map(list)
n = -1
if not production_mode:
    n = 10
rows = rows.rdd.map(lambda row: [float(x)
                    for x in row.features.strip('][').split(',')[:n]])
# rows.persist()

# %%
X = np.asarray(rows.collect())
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X)

# %%
dfr = pd.concat([labels, pd.DataFrame(
    {'features': [Vectors.dense(X_pca[i]) for i in range(len(X_pca))]})], axis=1)
# if not production_mode:
#     display(dfr.head())
dfr.to_csv(csv_dir+'data-reduced.csv', index=False, sep=";", quoting=3)

# %%
# Close Spark
print('>>>>>>>> all done!')
sc.stop()
spark.stop()
