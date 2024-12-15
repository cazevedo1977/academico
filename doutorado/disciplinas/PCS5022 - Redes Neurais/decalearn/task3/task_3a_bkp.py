import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_tuner import HyperModel
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D #, MaxPooling2D, Flatten # type: ignore
from tensorflow.keras.losses import CategoricalCrossentropy # type: ignore
from tensorflow.keras.metrics import CategoricalAccuracy # type: ignore
from keras_tuner import HyperModel, BayesianOptimization


import multiprocessing

#from dotenv import load_dotenv, find_dotenv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from google.cloud import storage
import numpy as np
from google.cloud import storage
from google.oauth2.service_account import Credentials
import io

# Ensure TensorFlow uses GPU and configures memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

current_path = Path(__file__).parent
output_dir = os.path.abspath(current_path)
os.makedirs(output_dir, exist_ok=True)

TRAIN_MODEL = True
VALIDATE_MODEL = True
PREDICT_DATA = True


APP_PATH = os.getcwd()
print(APP_PATH)
current_path = os.path.join(APP_PATH, os.path.join("data", "dataset.npz"))
train_path = APP_PATH #os.path.join(APP_PATH, os.path.join("decalearn\\task3", "dataset.npz"))
print(current_path)
num_cpus = multiprocessing.cpu_count()
class_names = ['granite', 'basalt', 'limestone', 'sandstone', 'slate']

class SimpleHyperModel(HyperModel):

    def __init__(self, num_classes, input_shape):
        super().__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape

    def build(self, hp):
        initializer = tf.keras.initializers.GlorotNormal(seed=12227)

        inputs = Input(shape=self.input_shape)

        # First Conv Layer
        H = Conv2D(
            filters=hp.Int('filters_1', min_value=32, max_value=128, step=32),
            kernel_size=hp.Choice('kernel_size_1', values=[3, 5]),
            activation=hp.Choice('activation_1', values=['relu', 'swish']),
            kernel_initializer=initializer,
            kernel_regularizer=tf.keras.regularizers.l2(hp.Float('l2_1', min_value=1e-5, max_value=1e-3, sampling='LOG'))
        )(inputs)
        H = MaxPooling2D(pool_size=(2, 2))(H)
        H = BatchNormalization()(H)
        H = Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.4, step=0.1))(H)

        # Second Conv Layer
        H = Conv2D(
            filters=hp.Int('filters_2', min_value=32, max_value=128, step=32),
            kernel_size=hp.Choice('kernel_size_2', values=[3, 5]),
            activation=hp.Choice('activation_2', values=['relu', 'swish']),
            kernel_initializer=initializer,
            kernel_regularizer=tf.keras.regularizers.l2(hp.Float('l2_2', min_value=1e-5, max_value=1e-3, sampling='LOG'))
        )(H)
        H = MaxPooling2D(pool_size=(2, 2))(H)
        H = BatchNormalization()(H)
        H = Dropout(rate=hp.Float('dropout_2', min_value=0.1, max_value=0.4, step=0.1))(H)

        H = Flatten()(H)

        # Fully Connected Layer
        H = Dense(
            units=hp.Int('units_fc_1', min_value=64, max_value=512, step=64),
            activation=hp.Choice('activation_fc_1', values=['relu', 'swish']),
            kernel_initializer=initializer,
            kernel_regularizer=tf.keras.regularizers.l2(hp.Float('l2_fc_1', min_value=1e-5, max_value=1e-3, sampling='LOG'))
        )(H)
        H = BatchNormalization()(H)
        H = Dropout(rate=hp.Float('dropout_fc_1', min_value=0.1, max_value=0.4, step=0.1))(H)

        outputs = Dense(self.num_classes, activation='softmax', kernel_initializer=initializer)(H)
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=hp.Choice('optimizer', values=['adam', 'sgd']),
            loss=CategoricalCrossentropy(),
            metrics=[CategoricalAccuracy()]
        )

        return model

class HyperModel_task3(HyperModel):

    def __init__(self, num_classes, input_shape):
        super().__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape

    def build(self, hp):
        initializer = tf.keras.initializers.GlorotNormal(seed=12227)

        inputs = Input(shape=self.input_shape)

        # First Conv Layer
        H = Conv2D(
            filters=hp.Int('filters_1', min_value=32, max_value=256, step=32),
            kernel_size=hp.Choice('kernel_size_1', values=[3, 5]),
            activation=hp.Choice('activation_1', values=['relu', 'swish']),
            kernel_initializer=initializer,
            kernel_regularizer=tf.keras.regularizers.l2(hp.Float('l2_1', min_value=1e-5, max_value=1e-2, sampling='LOG'))
        )(inputs)
        H = MaxPooling2D(pool_size=(2, 2))(H)
        H = BatchNormalization()(H)
        H = Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1))(H)

        # Additional Conv Layers
        for i in range(hp.Int('num_conv_layers', 1, 5)):
            H = Conv2D(
                filters=hp.Int(f'filters_{i+2}', min_value=32, max_value=256, step=32),
                kernel_size=hp.Choice(f'kernel_size_{i+2}', values=[3, 5]),
                activation=hp.Choice(f'activation_{i+2}', values=['relu', 'swish']),
                kernel_initializer=initializer,
                kernel_regularizer=tf.keras.regularizers.l2(hp.Float(f'l2_{i+2}', min_value=1e-5, max_value=1e-2, sampling='LOG'))
            )(H)
            H = MaxPooling2D(pool_size=(2, 2))(H)
            H = BatchNormalization()(H)
            H = Dropout(rate=hp.Float(f'dropout_{i+2}', min_value=0.1, max_value=0.5, step=0.1))(H)

        H = Flatten()(H)

        # Fully Connected Layers
        for i in range(hp.Int('num_fc_layers', 1, 3)):
            H = Dense(
                units=hp.Int(f'units_fc_{i+1}', min_value=64, max_value=1024, step=64),
                activation=hp.Choice(f'activation_fc_{i+1}', values=['relu', 'swish']),
                kernel_initializer=initializer,
                kernel_regularizer=tf.keras.regularizers.l2(hp.Float(f'l2_fc_{i+1}', min_value=1e-5, max_value=1e-2, sampling='LOG'))
            )(H)
            H = BatchNormalization()(H)
            H = Dropout(rate=hp.Float(f'dropout_fc_{i+1}', min_value=0.1, max_value=0.5, step=0.1))(H)

        outputs = Dense(self.num_classes, activation='softmax', kernel_initializer=initializer)(H)
        model = Model(inputs=inputs, outputs=outputs)

        '''
        model.compile(
            optimizer=hp.Choice('optimizer', values=['adam']),
            loss=CategoricalCrossentropy(),
            metrics=[CategoricalAccuracy()]
        )
        '''
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss=CategoricalCrossentropy(),
            metrics=[CategoricalAccuracy()]
        )

        return model
    
num_classes=5
input_shape=(112, 112,3)
#input_shape=(x_test_shrunk.shape[1], x_test_shrunk.shape[2],x_test_shrunk.shape[3])


hypermodel = HyperModel_task3(num_classes=num_classes, input_shape=input_shape)
#hypermodel = SimpleHyperModel(num_classes=num_classes, input_shape=input_shape)
    
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

APP_PATH = os.getcwd()
train_path = APP_PATH #os.path.join(APP_PATH, os.path.join("decalearn\\task3", "dataset.npz"))

# Instantiate the tuner
tuner = BayesianOptimization(
    hypermodel,
    objective='val_categorical_accuracy',
    max_trials=30,
    seed=12227,
    executions_per_trial=3,
    directory=output_dir,
    project_name='hyperparam_tuning_image',
    #overwrite=True  # Overwrite the previous results if needed
)


#region DATA VISUALIZATION
def plot_training_images(x_train, y_train, class_names, num_images=25):
    plt.figure(figsize=(10, 10))
    indices = np.random.choice(range(x_train.shape[0]), num_images, replace=False)
    for i, idx in enumerate(indices):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[idx])
        plt.xlabel(class_names[np.argmax(y_train[idx])])
    plt.show()

def plot_training_images_2_predict(unseen_data, class_names, num_images=25):
    plt.figure(figsize=(10, 10))
    indices = np.random.choice(range(unseen_data.shape[0]), num_images, replace=False)
    for i, idx in enumerate(indices):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(unseen_data[idx])
    plt.show()
#endregion

#region TRAIN MODEL
def train_model(X_train, y_train, x_test, y_test):

    tuner.search_space_summary()
    batch_size = 32
    epochs = 30
    #x_train_shrunk
    #x_test_shrunk
    tuner.search(
        x=X_train,  # Replace with your training data
        y=y_train,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping, reduce_lr],
        batch_size=batch_size,
        #multi_worker_mirrored_strategy=True,
        #workers=num_cpus,
        verbose=2
    )
#endregion

#region MODEL VALIDATION
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, classification_report


def find_best_threshold_for_f1score(y_true, y_pred_probs):
    """
    Compute the best threshold for each class in multi-class classification to maximize F1-score.
    
    Args:
    y_true (numpy array): True labels (one-hot encoded).
    y_probs (numpy array): Predicted probabilities from the model.

    Returns:
    dict: A dictionary containing the best threshold for each class.
    """
    num_classes = y_true.shape[1]
    thresholds = {}

    for i in range(num_classes):
        precision, recall, threshold = precision_recall_curve(y_true[:, i], y_pred_probs[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_threshold_index = np.argmax(f1_scores)
        best_threshold = threshold[best_threshold_index]
        thresholds[i] = best_threshold

    return thresholds

def find_best_threshold_for_accuracy(y_true, y_pred_probs):
    """
    Compute the best threshold for each class in multi-class classification to maximize accuracy.
    
    Args:
    y_true (numpy array): True labels (one-hot encoded).
    y_probs (numpy array): Predicted probabilities from the model.

    Returns:
    dict: A dictionary containing the best threshold for each class.
    """
    num_classes = y_true.shape[1]
    thresholds = {}

    for i in range(num_classes):
        best_threshold = 0.0
        best_accuracy = 0.0

        # Check thresholds from 0 to 1 with a step of 0.01
        for threshold in np.arange(0.0, 1.01, 0.01):
            y_pred = (y_pred_probs[:, i] >= threshold).astype(int)
            accuracy = accuracy_score(y_true[:, i], y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        thresholds[i] = best_threshold

    return thresholds

def apply_thresholds(y_probs, thresholds):
    """
    Apply the computed thresholds to the predicted probabilities to get the final class predictions.
    
    Args:
    y_probs (numpy array): Predicted probabilities from the model.
    thresholds (dict): Dictionary containing the best threshold for each class.

    Returns:
    numpy array: Final class predictions.
    """
    num_samples = y_probs.shape[0]
    num_classes = y_probs.shape[1]
    
    # Initialize the prediction array
    y_pred = np.zeros((num_samples, num_classes))
    
    for i in range(num_classes):
        # Apply the threshold for each class
        y_pred[:, i] = y_probs[:, i] >= thresholds[i]
    
    # Convert the binary predictions to class labels
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    return y_pred_labels

def default_threshold_classification(y_probs):
    """
    Classify samples based on the highest predicted probability (default threshold).
    
    Args:
    y_probs (numpy array): Predicted probabilities from the model.

    Returns:
    numpy array: Final class predictions.
    """
    y_pred_labels = np.argmax(y_probs, axis=1)
    return y_pred_labels

 

def validate_model(X_val, y_val):

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.summary()
    # Evaluate the best model
    y_pred = best_model.predict(X_val)
    print(y_pred)
    #y_pred = y_pred.ravel()
    
    best_thresholds = find_best_threshold_for_accuracy(y_true=y_val, y_pred_probs=y_pred)
    #best_thresholds = find_best_threshold_for_f1score(y_true=y_val, y_pred_probs=y_pred)
    print(f"Best Threshold: {best_thresholds}")

    y_pred_labels = apply_thresholds(y_probs=y_pred, thresholds=best_thresholds)
    accuracy = accuracy_score(np.argmax(y_val, axis=1), y_pred_labels)
    print('accuracy for best threshold: {:.2f}%'.format(100*accuracy))
    print(classification_report(np.argmax(y_val, axis=1), y_pred_labels))

    #threshold_default = np.array([0.5,0.5,0.5,0.5,0.5])
    threshold_default = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}

    y_test_pred = apply_thresholds(y_probs=y_pred, thresholds=threshold_default)
    accuracy = accuracy_score(np.argmax(y_val, axis=1), y_test_pred)
    print('accuracy default threshold: {:.2f}%'.format(accuracy*100))
    print(classification_report(np.argmax(y_val, axis=1), y_test_pred))
    
    y_test_pred = default_threshold_classification(y_probs=y_pred)
    accuracy = accuracy_score(np.argmax(y_val, axis=1), y_test_pred)
    print('accuracy default threshold: {:.2f}%'.format(accuracy*100))
    print(classification_report(np.argmax(y_val, axis=1), y_test_pred))

    return best_thresholds

#endregion

#region PREDICTIONS

def predictions(unseen_data, best_thresholds):
    best_model = tuner.get_best_models(num_models=1)[0]
    
    # Get the best hyperparameters
    best_hp = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters

    # Print the best hyperparameters
    print("Best hyperparameters:")
    for key in best_hp.values.keys():
        print(f"{key}: {best_hp.get(key)}")

    best_model.summary()
    
    y_pred = best_model.predict(unseen_data)
    
    # Apply the best threshold to make multiclass predictions
    predicted_classes = apply_thresholds(y_probs=y_pred, thresholds=best_thresholds)
    #predicted_classes = default_threshold_classification(y_probs=y_pred)
    
    save_result(predictions=predicted_classes, submission_file='submission.csv')

def save_result(predictions, submission_file):
    
    unique_values, indices, counts = np.unique(predictions, return_index=True, return_counts=True)
    print("Unique values:", unique_values)
    print("Indices:", indices)
    print("Counts:", counts)
    print("Proportion:", counts[0]/counts[1])
    
    
    # Create a DataFrame with zipped data and column names
    #num_samples = unseen_data.shape[0]
    
    df = pd.DataFrame(list(zip(range(1, len(predictions) + 1), predictions)), columns=['ID', 'Prediction'])
    #df = pd.DataFrame({'ID': np.arange(1, num_samples + 1),'Prediction': predictions})
    df.to_csv(submission_file, index=False)
    # Print the predictions
    #print("Predicted probabilities:\n", y_test_probs)
    #print("Predicted classes:\n", predicted_classes)


#endregion

#region DATA PREP
# Function to shrink image by 50%
def shrink_image(image):
    from skimage.transform import resize
    return resize(image, (image.shape[0] // 2, image.shape[1] // 2), anti_aliasing=True)
    
def data_prep(X,y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Normalize the images to a pixel value range of 0 to 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert class vectors to binary class matrices (one-hot encoding)
    y_train = to_categorical(y_train, 5)
    y_test = to_categorical(y_test, 5)
    
    
    # Print dimensions of the first 3 images
    print("Original dimensions of the first 3 images:")
    for i in range(3):
        print(f"Image {i+1}: {x_train[i].shape}")

    # Apply the shrink function to the first 3 images for demonstration
    x_train_shrunk = np.array([shrink_image(img) for img in x_train])

    # Apply the shrink function to the first 3 images for demonstration
    x_test_shrunk = np.array([shrink_image(img) for img in x_test])

    print("Shrunk dimensions of 3 images:")
    for i in range(5):
        print(f"Image {i+1} dimensions: {x_train_shrunk[i].shape}")

    
    print("{} - {} ".format(x_train_shrunk.shape, y_train.shape))
    print("{} - {} ".format(x_test_shrunk.shape, y_test.shape))
    
    #plot_training_images(x_train_shrunk, y_train, class_names)
    
    return x_train_shrunk, x_test_shrunk, y_train, y_test


def data_prep_2_predict(unseen_data):
    # Normalize the images to a pixel value range of 0 to 1
    unseen_data = unseen_data.astype('float32') / 255.0
    
    # Print dimensions of the first 3 images
    print("Original dimensions of the first 3 images:")
    for i in range(3):
        print(f"Image {i+1}: {unseen_data[i].shape}")

    # Apply the shrink function to the first 3 images for demonstration
    unseen_data_shrunk = np.array([shrink_image(img) for img in unseen_data])

    print("Shrunk dimensions of 3 images:")
    for i in range(5):
        print(f"Image {i+1} dimensions: {unseen_data_shrunk[i].shape}")

    #plot_training_images_2_predict(unseen_data_shrunk,class_names)
    
    return unseen_data_shrunk
#endregion


def main():
    print('\n####################  PCS-5022 - Redes Neurais e Aprendizado Profundo - Decalearn Task 1 #######################\n')
    
    # Carrega a base de dados a partir de seu caminho
    data = np.load(current_path)

    '''
    load_dotenv(find_dotenv())

    bucket_name = "pcs5022_usp"
    file_name = "dataset.npz"
    file_path = f"gs://{bucket_name}/{file_name}"
    training_path = f"gs://{bucket_name}"

    # Create a storage client
    storage_client = storage.Client()
    # Get the bucket and blob
    bucket = storage_client.bucket(bucket_name)

    from gcsfs import GCSFileSystem

    fs = GCSFileSystem()
    with fs.open(file_path) as f:
        data_bytes = f.read()

    data_stream = io.BytesIO(data_bytes)
    data = np.load(data_stream)
    '''

    X = data['X_train']
    y = data['y_train']
    unseen_data = data['X_test']
    #print('features: ', unseen_data.shape)
    
    x_train, x_test, y_train, y_test = data_prep(X,y)
    
    print("{} - {} ".format(x_train.shape, y_train.shape))
    print("{} - {} ".format(x_test.shape, y_test.shape))

    # Plot some training images
    #plot_training_images(x_train, y_train, class_names)

    best_thresholds = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}
    if TRAIN_MODEL:
        tuner = train_model(X_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        
    if VALIDATE_MODEL:    
        best_threshold = validate_model(X_val=x_test,y_val=y_test)
    
    if PREDICT_DATA:
        unseen_data_2_predict = data_prep_2_predict(unseen_data=unseen_data)
        #plot_best_model_training(X_train, X_val, y_train, y_val)
        predictions(unseen_data=unseen_data_2_predict, best_thresholds=best_threshold)
    
    
if __name__ == "__main__":
    main()
    
