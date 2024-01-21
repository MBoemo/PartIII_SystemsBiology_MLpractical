# Part III Systems Biology Machine Learning Practical

As hardware and deep learning methods continue to improve, neural networks are now all around us. They impact our day-to-day lives in more ways than most people realise: As you read this sentence, there is a very good chance that a neural network is running on your wrist or in your pocket in order to make real-time predictions from data. In this practical, we will explore the the application of neural networks to wearable technology and healthcare by predicting abnormal heartbeats from electrocardogram (ECG) data. ECGs are available on devices such as the Apple Watch (https://support.apple.com/en-us/HT208955) and can be used to identify irregular heart rhythms (such as atrial fibrillation) which are a major risk factor for stroke. As we work through the practical, we will give special consideration to the engineering constraints we might have to contend with in order to detect an irregular heartbeat on a wearable device.

# Outline and Outcomes

This practical will consist of two parts:
- Part I: A closely guided tutorial of how to import data, reshape it for training, design and train our first neural network, and benchmark its performance. 
- Part II: A loosely guided exploration of different neural network architectures and design considerations. Here, you will apply what you've learned in Part I and come up with different neural network architectures.

By the end of this practical, you should feel confident in:

- importing Tensorflow libraries,
- formatting data for neural network training,
- creating different neural network architectures with the Tensorflow Model API,
- benchmarking neural network performance. 

# Software and Data Sets

### Software
The software for this practical will be:

- Python 3.8,
- Tensorflow 2.12.0,
- Pandas,
- scikit-learn,
- matplotlib.

Neural network training will be done in Tensorflow. Pandas, scikit-learn, and matplotlib will be used as accessories to format data and make plots.

### Data Set and Format
The data set is the PTB Diagnostic ECG Database. This consists of 14552 ECGs divided into two classes: normal and abnormal. The data is organised into two spreadsheets in comma-separated value format (ptbdb_normal.csv and pdbdb_abnormal.csv) such that each row is a different ECG signal. The signals are sampled at 125 Hz with samples along the columns. Each signal has a maximum of 187 samples (i.e., columns). If the signal was longer this, it was truncated; if the signal was shorter than this, it is padded with zeros. Each signal is given a binary label in the rightmost column (1 for abnormal, 0 for normal).

# Part I: Building and Training a First Neural Network

## Import Libraries, Functions, and Classes

We can do the following to import the libraries listed above. 
```python 
import pandas as pd
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Activation, BatchNormalization, TimeDistributed, Masking, LSTM, Bidirectional, Add, Dropout, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.metrics import Recall, Precision
```

## Importing and Formatting Data

Let's get a feel for what our data looks like and make sure it agrees with our expectation. We can import some abnormal ECG data with the following line of code.

```python
dat_abnormal = pd.read_csv("ptbdb_abnormal.csv", header=None)
```

Let's take a look at the first five ECG signals to convince ourselves that the format is the same as what's described above.

```python
print(dat_abnormal.head(5))
```
You should be able to see the zero-padded signals with a Boolean label in the rightmost column (1.0 for abnormal ECG signals). We can also see how many signals we have and confirm the number of samples for each signal by checking the shape attribute of the Pandas dataframe.
```python
print(dat_abnormal.shape)
```
Repeat the above for the normal ECG signals paying particular attention to how the label in the rightmost column changes.

After checking the number of signals for each class (normal and abnormal) using the dataframe's shape attribute, you should have observed that we have more abnormal signals than normal signals by a factor of about 2:1. We want the number of signals for each class to be approximately balanced and we have plenty of data, so we're going to toss out some of the abnormal signals. While we're at it, we already said that the rightmost column is is going to be the label of the signal, so let's give it a column name called "Label" so we can easily pull out that column if we want to.

```python
dat_abnormal = pd.read_csv("ptbdb_abnormal.csv", header=None).head(4000)
dat_abnormal = dat_abnormal.rename({dat_abnormal.shape[1]-1:'Label'}, axis=1)

dat_normal = pd.read_csv("ptbdb_normal.csv", header=None).head(4000)
dat_normal = dat_normal.rename({dat_normal.shape[1]-1:'Label'}, axis=1)
```
Print out the first five signals of `dat_abnormal` and `dat_normal` to verify that renaming the rightmost column was successful.

We have two classes, normal and abnormal, and we have each of them in a separate dataframe. We want to concatenate them so that we can pass just one dataframe to our neural network at training time. To do that, we're going to use the Pandas concatenate function.

```python
dat_merged = pd.concat([dat_abnormal, dat_normal], axis=0).reset_index(drop=True)
```

It was useful having the rightmost column as the label for each signal, as these labels have been correctly shuffled with their signals when we shuffled the rows of the dataframe. However, we definitely don't want to input the label at the end of each signal during training time as it would make the training trivial. We're going to copy this label column into its own dataframe and then delete it from our training dataframe.

```python
label_merged = dat_merged['Label'].copy()
dat_merged = dat_merged.drop('Label',axis=1).copy()
```
Print out the first few rows of these two dataframes to make sure they're what you expect.

Our data is almost ready for training. However, it's important that we hold some data out for testing. This test data won't be involved in the training process whatsoever, and we'll use it later to benchmark our neural network's performance. We're going to split the data into training and test sets using the `train_test_split` function from scikit-learn. This function will shuffle our training data and labels together, then split off a fraction as test data. We want the shuffling to be reproducible, so we'll give it a random seed of 42.

```python
dat_train, dat_test, lab_train, lab_test = train_test_split(dat_merged, label_merged, test_size=0.2, random_state=42)
```

## Designing a First Neural Network

Our data is ready to go, but we still need to make our neural network. To do this, we're going to use the Tensorflow Model API which allows us to create arbitrary model topologies. To start, we're going to create a multilayer perceptron (MLP) using fully connected (i.e., "dense") layers. This type of neural network architecture isn't well-suited at all to this application. However, we're going to use it anyway for two reasons: First, it's useful to think about why this is a poor choice insofar as how we choose model architectures to fit the problems that we're trying to solve. Second, it will allow us to gain some familiarity with the syntax of Tensorflow's Model API which we can apply to create more appropriate models later. 

Every neural network needs at least one input layer and these input layers need to know the shape of the input they'll be passed. Recall from above that each ECG signal has a length of 187 columns. For each of these columns, we only have one real number (the magnitude of the signal) and so our input shape is: 
```python
input_layer= Input(shape=(187))
```
For the sake of the argument, suppose that we actually had measurements of two different features at each time step. In that case, we would have:
```python
input_layer= Input(shape=(187,2))
```
Now we need to define the rest of the layers in our network. We're using the Tensorflow Model API which uses the following general syntax for encoding the connectivity between layers:

```python
Layer_output = Layer_Name(...parameters...)(Layer_input)
```
We're going to use two different layers in this neural network:

- Dense (https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)
- Dropout (https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout)

Dropout is a type of regularisation that randomly sets some fraction of its inputs equal to zero to help prevent overtraining. In this network, we're going to use a fraction of 0.5, so half of the Dropout layer's inputs will be randomly set to zero. We can code the model layers and the connectivity between them as follows:

```python
X_layer = Dense(100, activation='relu')(input_layer)
X_layer  = Dropout(0.5)(X_layer)
X_layer  = Dense(50, activation='relu')(X_layer)
X_layer  = Dropout(0.5)(X_layer)
X_layer  = Dense(1, activation='sigmoid')(X_layer)
```

Check the documentation of the Dense and Dropout layers linked above and make sure you understand what's being done in the code block above.


For our nonlinear activation functions, we used ReLU in the first two dense layers and a sigmoid function (https://en.wikipedia.org/wiki/Sigmoid_function) in the final dense layer. Note that we're working on a binary classification problem (the heartbeat is either normal or abnormal). The labels are such that abnormal heartbeats had a label of 1.0 and normal heartbeats had a label of 0.0, so our network is predicting the probability of an abnormal heartbeat. Hence we set only one node in the final dense layer and use the sigmoid function to compute the probability of an abnormal heartbeat and conclude that the probability of a normal heartbeat is 1 - P(abnormal heartbeat). Note that if we had more than two classes (i.e., multi-class classification) the number of nodes in our final dense layer would be equal to the number of classes and it would be appropriate to use a softmax activation function instead of a sigmoid.

Finally, we're going to wrap these layers into a single Model object and compile it. As we train, we're going to keep track of accuracy, precision, and recall. Our labels are Boolean (0 or 1) so our loss function is going to be binary cross-entropy. A binary cross-entropy loss expects a single output node with sigmoid activation, and we can see that this is consistent with what we have above. Note that if we were working on a mult-class classification problem with softmax activation in the final dense layer, we would use 'categorical_crossentropy' instead of 'binary_crossentropy'.
 
```python
model = Model(inputs=input_layer, outputs=X_layer, name='ECG')
model.compile(optimizer='adam', metrics=['accuracy',Precision(),Recall()], loss='binary_crossentropy')
```

Draw a rough sketch of the different layers in this model and calculate how many trainable parameters each layer has. Check that you are correct by doing:

```python
print(model.summary())
```

Putting it all together, our model looks as follows:

```python
input_layer= Input(shape=(187))
X_layer = Dense(100, activation='relu')(input_layer)
X_layer  = Dropout(0.5)(X_layer)
X_layer  = Dense(50, activation='relu')(X_layer)
X_layer  = Dropout(0.5)(X_layer)
X_layer  = Dense(1, activation='sigmoid')(X_layer)

model = Model(inputs=input_layer, outputs=X_layer, name='ECG')
model.compile(optimizer='adam', metrics=['accuracy',Precision(),Recall()], loss='binary_crossentropy')
print(model.summary())
```
Now we're ready to start training.

## Training the Model

When training a model, it's often useful to have at least one callback function which are functions that execute at a specific time during training. The callback functions we'll use today execute at the end of each epoch. They are:

- EarlyStopping: Calculates loss on the validation dataset at the end of each epoch. If a certain number of epochs go by with no improvement (specified by the `patience` parameter) halt the training to guard against overtraining. Optionally, once training is halted, the model weights can be restored to that of the epoch with the lowest validation loss. (https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)
- CSVLogger: Record the training and validation loss into a csv file for recordkeeping. (https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/CSVLogger)

We can define these two functions as follows:

```python
es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto', restore_best_weights=True)
csv = CSVLogger('log.csv', separator=',', append=False)
```
We can train the model by using `model.fit`. Here, we can specify the training data (`dat_train`), the labels (`lab_train`), the number of epochs (100), how much of the training data we want to set aside for validation (a fraction of 0.2 or, equivalently, 20%), as well as our callbacks. Note that 100 epochs is probably quite high, but don't forget that we have our early stopping callback to help protect against overtraining.

Finally, we'll save our trained model in a directory called `model_trained`. We can then load up that trained model any time we want, deploy it, or incorporate it into another piece of software.

```
history = model.fit(x=dat_train, y=lab_train, batch_size=32, epochs=100, validation_split=0.2, verbose=1, callbacks=[es,csv])
model.save('model_trained')
```

## Evaluating the Model

Once the model is trained, we need to see how good it is. Let's start by plotting how the training loss and validation loss changed over our training epochs. When we called `model.fit`, this returned a `History` object that recorded the training and validation loss for us. We can plot them as follows:

```python
#make a figure showing the training loss and validation loss
epochs = range(1,len(history.history['loss'])+1)
plt.figure()
plt.plot(epochs,history.history['loss'],label='Training Loss')
plt.plot(epochs,history.history['val_loss'],label='Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(framealpha=0.3)
plt.savefig('loss.pdf')
plt.close()
```

We want to have a final benchmark for the accuracy, precision, and recall of our model and we set aside some test data at the beginning for this purpose. We can evaluate the model on our test data as follows.

```
model.evaluate(dat_test,lab_test)
```

Lastly, we can probe a little deeper into how our model is performing. We'll use our test data to compute a receiver operating characteristic (ROC) curve. For every ECG signal, our model will output a probability (between 0 and 1) of an abnormal heartbeat. We can consider a range of probability thresholds above which we say that a ECG signal is classified as abnormal. If our threshold is very low (say, 0.1) then you might imagine that we would have high true positive rate but a high false positive rate. If our probability threshold is very high (say, 0.9) then you might imagine that we would have a low true positive rate and a low false positive rate. We can visualise the relationship between probability thresholds, true positive rate, and false positive rate with a ROC curve. The following block of code will plot the ROC curve and annotate different locations along the curve with the corresponding probability threshold above which a an ECG signal is considered abnormal.

```python
#compute a ROC curve on the test data
pred = model.predict(dat_test)
fig,ax = plt.subplots()
roc = roc_curve(lab_test.astype(int),pred)
for i in range(1,len(roc[2]),int(len(roc[2])/10)):
	ax.annotate("{:.2f}".format(roc[2][i]),(roc[0][i],roc[1][i]),fontsize=6)
auc = roc_auc_score(lab_test.astype(int),pred)
plt.plot(roc[0], roc[1], label = 'AUC: '+"{:.3f}".format(auc))
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(framealpha=0.3)
plt.savefig('ROC.pdf')
plt.close()
```

ROC curves allow us to observe a few different properties to make sure our model is performing well:

- An ideal classifier has quite a sharp elbow-shaped ROC curve that closely hugs the top left of the plot. Hence, we can benchmark our model's performance by measuing the area under the curve (AUC). An AUC close to 1 suggests good performance of the classifier.
- We can look at the thresholds along the curve to assess how our models is making predictions and how confident it is. Ideally, we want our model's predictions to be confident and correct. Therefore, it's a positive sign if most of our abnormal heartbeat ECGs (on the True Positive axis) are assigned a probability of about 0.75 or higher. Likewise, we would like most of our normal heartbeat ECGs (on the False Positive axis) to be assigned probabilities lower than about 0.25. If, on the other hand, most of our abnormal heartbeat ECGs were assigned probabilities only just higher than 0.5 and most of our normal heartbeat ECGs were assigned probabilities only just lower than 0.5 then, while we may have an impressively high accuracy, our model isn't making very confident decisions.


### Optional Exploration Before Part II

Before moving on to Part II, you may wish to explore how the amount of training data affects your model's performance. Previously, we used 4000 abnormal and 4000 normal ECG signals. Experiment with downsampling the training data to get a feel for how it affects the evaluation metrics we computed in this section. 


# Part II: Finding a Better Architecture

In Part I, we used fully connected layers to classify the ECG data. While this was valuable for learning the Tensorflow Model API, we said that this model architecture isn't ideal for this purpose. Spend some time reflecting on why fully connected layers aren't ideal for this application and come up with as many reasons as you can.

We're now going to progress on a loosely guided exploration to apply what we've learned to some new neural network architectures and see if we can do better.

## Recurrent Neural Networks

ECGs are time series, hence it's very tempting to try to use a recurrent neural network for this application. Read through the documentation to implement a neural network architecture with LSTM layers (https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM). Pay particular attention to the following options and make sure you understand what they do:

- units,
- activation,
- return_sequences.

You may find it useful to think about the following design considerations:

- Previously, we used `input_layer= Input(shape=(187))` to define the shape of the input. For an recurrent neural network, we must make it explicit that we only have one feature per timestep by doing `input_layer= Input(shape=(187,1))`.
- LSTMs permit the masking of zeros via a masking layer (https://www.tensorflow.org/api_docs/python/tf/keras/layers/Masking). You may wish to use this layer after your input layer to tell your LSTM layers to ignore zero-padded positions.
- If we have a layer would output a matrix of shape (M, N), then a Flatten layer (https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) will turn it into a one-dimensional vector of length M*N. Depending on the architecture you design, you may find it useful to use a Flatten layer before your final Dense layer.
- Think carefully about when and where you want `return_sequences = True` and `return_sequences=False` in your LSTM layers. 
- Hint: The final layer of your network, i.e., `Dense(1, activation='sigmoid')`, shouldn't change from what we did before.
 

## Convolutional Neural Networks

We often associate convolutional neural networks (CNNs) with computer vision, but they can also be used for the analysis of signals. We can come up with an argument of why CNNs might be useful for this ECG classification task: We are interested in  the spatial distribution and magnitude of irregular heartbeats, a bit like we might be interested in the spatial distribution of objects in an image. Read through the documentation on one-dimensional convolutional layers (https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D) and make sure you understand the following options:

- filters,
- kernel_size,
- strides,
- padding,
- activation.

Then implement and train a CNN for the ECG classification task. You may want to think about the following:

- Like in recurrent neural networks, we must specify `input_layer= Input(shape=(187,1))`.
- Sadly, CNNs do not permit masking the way LSTMs do. Think about why.
- Flatten layers can be used with CNNs, but you might end up with quite a long vector before your final dense layer. You may find max pooling (https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPooling1D) helps with this.

## Residual Convolutional Neural Networks

In the Tensorflow Model API, we can create arbitrary model topologies which is useful for creating the "skip" connections in residual convolutional neural networks (ResNets). Recall that to make a skip connection, we need to save one "clean" copy of the input, put the other copy through multiple convolutional layers, and then add the two together element-wise. Here's an example of how we might do that:

```python
X_SC = X_layer
X_layer = Conv1D(32, 4, activation='relu',padding='same')(X_layer)
X_layer = Conv1D(32, 4, activation=None,padding='same')(X_layer)
X_layer = Add()([X_SC,X_layer])
X_layer = Activation('relu')(X_layer)
```

Create a ResNet for the ECG classification task and think about the following design considerations:

- Padding of the convolutional layers is very important here. The output of the convolutions must be the same size as the input, otherwise we will have a dimension mismatch when we go to do the element-wise layer addition.
- Pay careful attention to the layer variable names. Simple bugs in variable naming can result in you getting a different model topology than what you expect.


## Reflections

We're trying to build an accurate classifier for ECG data that can work on a wearable device such as a smartwatch. That device will have limited hardware capabilities, both in terms of computational power and in performance of the sensor. The device may move on the user's wrist, get wet in the rain, or experience other environmental insults that can impact sensor performance. The wrists of different people will vary in size and shape. In addition, the outcome of the classification can have serious implications for the health of the device's user. With those engineering constraints in mind, reflect upon the following questions.

- Of the models you developed, which was fastest? Why?
- Which was easiest to train?
- Which model had the best performance metrics (and define what you mean by "best")?
- According to that metric (or those metrics), is the model good enough to give health information to users?
- What are the ethical challenges associated with using deep learning to detect events that could be critical to a user's health?

## Extra Challenge (Optional)

In the above practical, we considered the PTB Diagnostic ECG Database which classifies ECG signals into normal and abnormal (two classes). For an extra challenge, consider the MIT-BIH Arrhythmia Database. The data has the same format (signal sampled at 125 Hz such that each signal has 187 samples with the last column as a label) but there are now five classes given as integers from 0 to 4:

- 0: normal heartbeat
- 1: supra-ventricular premature
- 2: ventricular escape
- 3: fusion of ventricular and normal
- 4: unclassified

Choose the best-performing model architecture above and update it to perform a five-way classification instead of a two-way classification.  Note that the MIT-BIH Arrhythmia Database contains considerably more data than the PTB Diagnostic ECG Database. Experiment with the effect of training data amount by subsampling the MIT-BIH Arrhythmia Database to 4000 signals (the same as you used above) and then increasing the amount of training data to determine its effect on your model's performance. If you decided your previous model was not sufficiently good enough to give health information to users, is it good enough after being trained on this larger dataset? Why or why not?

## References

- PTB Diagnostic ECG Database: https://www.physionet.org/content/ptbdb/1.0.0/
- MIT-BIH Arrhythmia Database: https://www.physionet.org/content/mitdb/1.0.0/
