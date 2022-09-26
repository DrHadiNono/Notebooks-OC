# %%
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.linalg import DenseVector

from PIL import Image
import numpy as np
import pandas as pd

# from pathlib import Path
# import os

# %%
production_mode = True
local_mode = False
OUTPUT_IMG = False
NB_CHANNEL = 1  # 1:grayscale, 3:RGB
REDUCTION_FACTOR = 2

# %%
sc = SparkContext()
spark = SparkSession(sc)

origin_path = 'F:/' if local_mode else 's3a://dr.hadinono/OC/P8/'

img_dir = origin_path+'fruits-360/Training/*'
if not production_mode:
    img_dir = origin_path+'fruits-360/Training/Apple_Braeburn/'
output_dir = origin_path+'fruits-360/PreprocessedTraining/'
csv_dir = origin_path+'fruits-360/CSV/Separate/'

# if local_mode and production_mode:
#     # Create the data folders to store the outputs
#     for name in os.listdir(img_dir[:-2]):
#         if OUTPUT_IMG:
#             Path(output_dir+name).mkdir(parents=True, exist_ok=True)
#         Path(csv_dir+name).mkdir(parents=True, exist_ok=True)

# %%


def save(image):
    filename = image[0]
    filedir = filename.split('/')[0]

    # Save preprocessed image
    # to JPG
    img = image[1]
    if OUTPUT_IMG:
        new_img = Image.fromarray(img.astype(
            'uint8'), 'RGB' if NB_CHANNEL == 3 else 'L')
        new_img.save(output_dir+filename)
        del(new_img)

    # to CSV with label
    # df = pd.concat(
    #     [pd.DataFrame({'label': [filedir]}), pd.DataFrame(img.flatten()).T], axis=1)
    df = pd.DataFrame({'label': [filedir], 'features': [
                      DenseVector(img.flatten())]})
    df.to_csv(csv_dir+filename.split('.')
              [0]+'.csv', index=False, sep=";", quoting=3)

    del(img)
    del(df)


def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    minval = None
    maxval = None
    if NB_CHANNEL == 1:
        arr = arr.flatten()
        arr *= (255.0/(arr.max()-arr.min()))
        arr = np.resize(arr, (100//REDUCTION_FACTOR, 100//REDUCTION_FACTOR))
    else:
        for i in range(NB_CHANNEL):
            minval = arr[..., i].min()
            maxval = arr[..., i].max()
            if minval != maxval:
                arr[..., i] -= minval
                arr[..., i] *= (255.0/(maxval-minval))
    del(minval)
    del(maxval)
    return arr


def reduce_image(img):
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    if NB_CHANNEL == 1:
        img = img.convert('L')
    img = img.resize((100//REDUCTION_FACTOR, 100 //
                     REDUCTION_FACTOR), Image.ANTIALIAS)
    return np.array(img)


def image_to_array(img):
    filename = '/'.join(img.origin.split('/')[-2:])
    img = np.resize(np.asarray(list(img.data)), (100, 100, 3))
    img = reduce_image(img)
    img = normalize(img)
    save((filename, img))
    # There is nothing to return, because there is no collect. All the processing and the exports is done for each image separately by each executors


# %%
imgs = spark.read.format("image").load(
    img_dir).select("image.origin", "image.data")
if not production_mode:
    imgs = imgs.limit(5)
imgs = imgs.rdd.map(image_to_array).collect()

# %%
# Close Spark
print('>>>>>>>> all done!')
sc.stop()
spark.stop()
