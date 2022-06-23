import datetime
import itertools
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pip

try:
    import tensorflow as tf
    import keras
    import sklearn
except ImportError:
    pip.main(['install', 'tensorflow'])
    pip.main(['install', 'keras'])
    pip.main(['install', 'sklearn'])
    import tensorflow as tf
    import keras
    import sklearn

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, \
    recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import BatchNormalization, Input, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dropout, Dense, Flatten
from keras import regularizers, metrics
from keras_preprocessing.image import ImageDataGenerator
from keras import backend as K
import keras.optimizers


# Creates a simple two-layer MLP with inputs of the given dimension
def create_mlp(dim, regularizer=None):
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu", kernel_regularizer=regularizer))
    model.add(Dense(4, activation="relu", kernel_regularizer=regularizer))
    return model


# Creates a CNN with the given input dimension and filter numbers.
def create_cnn(width, height, depth, filters=(16, 32, 64), regularizer=None):
    # Initialize the input shape and channel dimension, where the number of channels is the last dimension
    inputShape = (height, width, depth)
    chanDim = -1

    # Define the model input
    inputs = Input(shape=inputShape)

    # Loop over the number of filters
    for (i, f) in enumerate(filters):
        # If this is the first CONV layer then set the input appropriately
        if i == 0:
            x = inputs

        # Create loops of CONV => RELU => BN => POOL layers
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # Final layers - flatten the volume, then Fully-Connected => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16, kernel_regularizer=regularizer)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

    # Apply another fully-connected layer, this one to match the number of nodes coming out of the MLP
    x = Dense(4, kernel_regularizer=regularizer)(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)

    # Construct the CNN
    model = Model(inputs, x)

    # Return the CNN
    return model


# Plots a confusion matrix
def show_cf(y_true, y_pred, class_names=None, model_name=None):
    cf = confusion_matrix(y_true, y_pred)
    plt.imshow(cf, cmap=plt.cm.Blues)

    if model_name:
        plt.title("Confusion Matrix: {}".format(model_name))
    else:
        plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    class_names = set(y_true)
    tick_marks = np.arange(len(class_names))
    if class_names:
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)

    thresh = cf.max() / 2.

    for i, j in itertools.product(range(cf.shape[0]), range(cf.shape[1])):
        plt.text(j, i, cf[i, j], horizontalalignment='center', color='white' if cf[i, j] > thresh else 'black')

    plt.colorbar()
    plt.show()


# Evaluates the performance of a CNN
def cnn_evaluation(model, history, train_features, train_images, train_labels, test_features, test_images, test_labels,
                   class_names=None, model_name=None):
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epch = range(1, len(train_acc) + 1)
    plt.plot(epch, train_acc, 'g.', label='Training Accuracy')
    plt.plot(epch, val_acc, 'g', label='Validation acc')
    plt.title('Accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epch, train_loss, 'r.', label='Training loss')
    plt.plot(epch, val_loss, 'r', label='Validation loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

    results_test = model.evaluate([test_features, test_images], test_labels)
    print('Test Loss:', results_test[0])
    print('Test Accuracy:', results_test[1])

    y_train_pred = np.round(model.predict([train_features, train_images]))
    y_pred = np.round(model.predict([test_features, test_images]))

    show_cf(test_labels, y_pred, class_names=class_names, model_name=model_name)

    print(classification_report(train_labels, y_train_pred))
    print(classification_report(test_labels, y_pred))


# speeds up training by using jit
def boost_perfomance():
    config = tf.compat.v1.ConfigProto()
    jit_level = tf.compat.v1.OptimizerOptions.ON_1
    config.graph_options.optimizer_options.global_jit_level = jit_level
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)


def build_model():
    boost_perfomance()
    image_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory("images/train", shuffle=False,
                                                                               class_mode='binary',
                                                                               target_size=(256, 256), batch_size=20000)
    images, labels = next(image_generator)
    (trainAttrX, testAttrX, trainImagesX, testImagesX) = train_test_split(
        pd.read_csv("train_2020_after_images_sort_bad_good_merged.csv").drop(
            columns=["grid_square"], axis=1),
        images, test_size=0.25,
        random_state=42)
    trainY = trainAttrX["safe"]
    testY = testAttrX["safe"]
    trainAttrX.drop(columns=["safe"], axis=1, inplace=True)
    testAttrX.drop(columns=["safe"], axis=1, inplace=True)

    # Create the MLP and CNN models
    mlp1 = create_mlp(trainAttrX.shape[1], regularizer=regularizers.l1(0.005))
    cnn1 = create_cnn(256, 256, 3, regularizer=regularizers.l1(0.005))

    # Create the input to the final set of layers as the output of both the MLP and CNN
    combinedInput = concatenate([mlp1.output, cnn1.output])
    print(mlp1.summary())
    print(cnn1.summary())

    # The final FC layer head will have two dense layers
    x = Dense(4, activation="relu", kernel_regularizer=regularizers.l1(0.005))(combinedInput)
    x = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l1(0.005))(x)
    start = datetime.datetime.now()

    model1 = Model(inputs=[mlp1.input, cnn1.input], outputs=x)

    # compile the model
    opt = keras.optimizers.adam_v2.Adam(learning_rate=1e-4, decay=1e-4 / 200)
    model1.compile(loss="binary_crossentropy", metrics=['acc', metrics.AUC(name="auc_score")], optimizer=opt)

    # train the model, and validate with the test set
    model1_history = model1.fit([trainAttrX, trainImagesX], trainY,
                                validation_data=([testAttrX[:1000], testImagesX[:1000]], testY[:1000]), epochs=5,
                                batch_size=32)
    end = datetime.datetime.now()
    print("Time taken to run:", end - start)

    model1.save('models/mixed_model.h5')
    cnn_evaluation(model1, model1_history, trainAttrX, trainImagesX, trainY, testAttrX[1000:], testImagesX[1000:],
                   testY[1000:], class_names=['bad_areas', 'good_areas'])


def test_model(test_dataset):
    model1 = load_model('models/mixed_model.h5')
    dataset = pd.read_csv(test_dataset)
    source_path = "images/source/"
    test_path = "images/all_images/all/"
    try:
        if os.path.isdir(test_path):
            if not os.listdir(test_path):
                for filename in os.listdir(source_path):
                    if filename.endswith(".jpg"):
                        shutil.copy2(source_path + filename, test_path)
        image_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory("images/all_images/", shuffle=False,
                                                                                   class_mode='binary',
                                                                                   target_size=(256, 256), batch_size=20000)

        # Checking the labels
        image_generator.class_indices
        images, labels = next(image_generator)
        # Using model to predict for each of the locations (with images & structured data attributes) in the test set
        testAttrX = dataset.drop(columns=["grid_square", "safe"], axis=1, inplace=False)
        testImagesX = images
        testY = dataset["safe"]
        y_pred = np.round(model1.predict([testAttrX[1000:], testImagesX[1000:]]))
        grid_square = pd.DataFrame(dataset["grid_square"][1000:]).reset_index(drop=True)

        # Reshaping the predictions and turning them into a pandas series
        test_predictions = pd.Series(y_pred.reshape((y_pred.size,)))
        test_predictions = pd.Series(test_predictions)

        # Getting a dataframe of the test set locations and actual classes
        test_actuals = pd.DataFrame(testY[1000:]).reset_index(drop=True)

        # Making a dataframe of True and Predicted labels so can look up images by grid_square
        test_df = pd.concat([grid_square, test_actuals, test_predictions], axis=1)
        test_df.columns = ['grid_square', 'True', 'Predicted']

        # metrics
        print("accuracy:", accuracy_score(test_actuals, test_predictions))
        print("precision:", precision_score(test_actuals, test_predictions))
        print("recall:", recall_score(test_actuals, test_predictions))
        print("f1_score:", f1_score(test_actuals, test_predictions))
        print("roc_auc_score:", roc_auc_score(test_actuals, test_predictions))

        # Example predictions
        print(test_df.head(test_df.shape[0]))
        show_cf(test_actuals, test_predictions)
    finally:
        # cleaning up test_path
        print("cleaning up test_path (images)")
        for filename in os.listdir(test_path):
            if filename.endswith(".jpg"):  # all other images are "good" so copy them into "good" folder
                os.remove(test_path + filename)

