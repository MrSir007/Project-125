import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml as fo
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as lr
from PIL import Image
import PIL.ImageOps

# To fetch the data from "OpenML" library
X = np.load("image.npz")["arr_0"]
y = pd.read_csv("labels.csv")["labels"]
classes = ["A","B","C","D","E","F","G","H","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
nClasses = len(classes)

xTrain, xTest, yTrain, yTest = tts(X, y, random_state=1, train_size=7500, test_size=2500)
# To scale the data to make data points equal
xTrainScale = xTrain / 255.0
xTestScale = xTest / 255.0
# To fit the data inside the model for maximum accuracy
classifier = lr(solver="saga", multi_class="multinomial").fit(xTrainScale, yTrain)

def getPrediction (image) :
  # To open the uploaded images
  getImage = Image.open(image)
  # To convert the images to black-and-white
  imageBW = getImage.convert("L")
  # To resize the images
  imageResized = imageBW.resize((28, 28), Image.ANTIALIAS)
  # To get the minimum pixel for the images
  pixelFilter = 20
  pixelMinimum = np.percentile(imageResized, pixelFilter)
  # To number the images
  imageNumbered = np.clip(imageResized-pixelMinimum, 0, 255)
  # To get the maximum pixel for the images
  pixelMaximum = np.max(imageResized)
  imageNumbered = np.asarray(imageNumbered) / pixelMaximum
  # To create a test sample for prediction
  testSample = np.array(imageNumbered).reshape(1, 784)
  testPrediction = classifier.predict(testSample)
  return testPrediction[0]