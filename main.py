"""
use SoftMax instead of sigmoid, then change loss to categorical_crossentropy and finally try sgd optimizer
"""

import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

# read data
traindata = pd.read_csv("./KDDTrain+.txt", header=None)
#testdata = pd.read_csv("./KDDTest+.txt", header=None)


# =====preprocessing=====
# drop duplicates
traindata.drop_duplicates(inplace=True)
#testdata.drop_duplicates(inplace=True)
# removing unnecessary features
traindata.drop([0, 1, 2, 3, 41], inplace=True)
#testdata.drop([0, 1, 2, 3, 41], inplace=True)
# converting categorical data
traindata = pd.get_dummies(traindata, columns=[1, 2, 3])
#testdata = pd.get_dummies(testdata, columns=[1, 2, 3])

"""
missing_cols = set(traindata.columns) - set(testdata.columns)
for c in missing_cols:
    testdata[c] = 0
# Reorder test data columns to match train data columns order
testdata = testdata[traindata.columns]
"""

# splitting dataset into input features and target labels

x = traindata.drop(41, axis=1).values
y = traindata[41].apply(lambda x: 0 if x == "normal" else 1).values
"""
test_x = testdata.drop(41, axis=1).values
test_y = testdata[41].apply(lambda x: 0 if x == "normal" else 1).values
"""

train_x, test_x = train_test_split(x, test_size=0.2, random_state=42)
train_y, test_y = train_test_split(y, test_size=0.2, random_state=42)

# =====designing model=====
model = Sequential()
model.add(Dense(64, activation="relu", input_dim=train_x.shape[1]))
model.add(Dropout(0.5)) # dropout layer - depends on noise in data
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics="accuracy")
model.fit(train_x, train_y, epochs=10, batch_size=128)

#=====testing model=====
loss, accuracy = model.evaluate(test_x, test_y)

print(f"Accuracy: {accuracy}, Loss: {loss}")