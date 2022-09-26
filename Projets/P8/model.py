# %%
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.linalg import DenseVector
from pyspark.ml import Pipeline
from pyspark.ml.feature import IndexToString, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
import pandas as pd

# %%
production_mode = True
local_mode = False

# %%
sc = SparkContext()
spark = SparkSession(sc)

origin_path = 'F:/' if local_mode else 's3a://dr.hadinono/OC/P8/'

csv_path = origin_path+'fruits-360/CSV/'
csv_dir = origin_path+'fruits-360/CSV/Separate/'
csv_separate_dir = csv_dir+'Separate/'

# %%
# Read the images-csv file
data = spark.read.options(delimiter=";", header=True,
                          maxCharsPerColumn=-1).csv(csv_path+'data-reduced.csv')

# Convert to vectors
data = data.rdd.map(lambda row: Row(label=row.label, features=DenseVector(
    [float(x) for x in row.features.strip('][').split(',')]))).toDF()
if not production_mode:
    data.show()

# %%
# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(
    inputCol="label", outputCol="indexedLabel").fit(data)

if not production_mode:
    data.show()

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.6, 0.4])
trainingData.persist()
testData.persist()

if not production_mode:
    trainingData.show()
    testData.show()

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel",
                            featuresCol="features", numTrees=10)

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, rf, labelConverter])

# %%
# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

if not production_mode:
    # Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show()

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test accuracy = %g" % (accuracy))

# %%
df = pd.DataFrame({'label': predictions.select("label").rdd.map(lambda x: x[0]).collect(), 'predictedLabel': predictions.select(
    "predictedLabel").rdd.map(lambda x: x[0]).collect(), 'features': predictions.select("features").rdd.map(lambda x: x[0]).collect()})
df.to_csv(csv_path+'Predictions-accuracy='+str(round(accuracy, 2)) +
          '.csv', index=False, sep=";", quoting=3)

# %%
# Close Spark
print('>>>>>>>> all done!')
sc.stop()
spark.stop()
