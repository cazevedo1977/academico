#region LIBs
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import tensorflow as tf


from keras.utils import to_categorical # type: ignore
from tensorflow.keras.utils import to_categorical

from keras_tuner import HyperModel, BayesianOptimization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy # type: ignore
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
#from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import multiprocessing









from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, classification_report


from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2
from keras.layers import Input
from tensorflow.keras.callbacks import Callback


from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.initializers import GlorotNormal # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore



from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from pathlib import Path

#endregion 
TRAIN_MODEL = True
VALIDATE_MODEL = False
PREDICT_DATA = False
num_cpus = multiprocessing.cpu_count()
current_path = Path(__file__).parent
output_dir = os.path.abspath(current_path)
os.makedirs(output_dir, exist_ok=True)


#region NN Class, methods, optimizers and inicialization
#output_dir = os.path.abspath('C:\\Caio\\decalearn\\task2')

'''
The HyperModel Class:

Highlights:
 - Despite dimensionality reduction, there are still hundreds of features in the dataset. Additionally, I noticed some degree of overfitting, so I decided to test regularization techniques. 

 Considering the information above:
 - The input shape is set to the dimensions of the images (e.g., (224, 224, 3)).
 - Convolutional layers are added to extract spatial features from the images.
 - MaxPooling2D layers are used to reduce the spatial dimensions of the feature maps.
 - BatchNormalization and Dropout layers are used to improve training stability and prevent overfitting.
 - The number of convolutional and fully connected layers, along with their hyperparameters, are determined by the hyperparameter tuner (hp).

 - Increased Units and Layers: I increased the maximum number of units and allowed up to 5 hidden layers. More layers with different configurations to capture more complex patterns.
 - Regularization: Added L2 regularization and dropout layers to each dense layer to reduce overfitting.
 - Activation Functions: Added 'tanh' as an additional activation function choice.
 - Advanced Optimizers: Included 'sgd' and 'rmsprop' optimizers as choices.
 - Tuning Range: Increased the range for some hyperparameters to explore more configurations.
 - Max Trials: Increased max_trials to 50 for a more exhaustive search of hyperparameters.
 - Learning Rate Schedulers: Implement learning rate schedules for better convergence.
 - Feature Engineering: Apply techniques like Principal Component Analysis (PCA) for dimensionality reduction if feasible.
 -


 
'''


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
            activation=hp.Choice('activation_1', values=['relu', 'swish', 'tanh']),
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
                activation=hp.Choice(f'activation_{i+2}', values=['relu', 'swish', 'tanh']),
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
                activation=hp.Choice(f'activation_fc_{i+1}', values=['relu', 'swish', 'tanh']),
                kernel_initializer=initializer,
                kernel_regularizer=tf.keras.regularizers.l2(hp.Float(f'l2_fc_{i+1}', min_value=1e-5, max_value=1e-2, sampling='LOG'))
            )(H)
            H = BatchNormalization()(H)
            H = Dropout(rate=hp.Float(f'dropout_fc_{i+1}', min_value=0.1, max_value=0.5, step=0.1))(H)

        outputs = Dense(self.num_classes, activation='softmax', kernel_initializer=initializer)(H)
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop']),
            loss=CategoricalCrossentropy(),
            metrics=[CategoricalAccuracy()]
        )
        
        return model




def init_class(num_classes, input_shape):
    """
    Initializes the hyperparameter tuning process for an image classification model.

    This function creates a BayesianOptimization tuner, defines callbacks for early stopping, learning rate reduction, and model checkpointing,
    and returns the tuner and callbacks for use in the model training process.

    Args:
        num_classes (int): The number of classes in the multi-class classification problem.
        input_shape (tuple): The shape of the input images (height, width, channels).
        output_dir (str): The directory to save the tuning results and best model.

    Returns:
        tuple: A tuple containing the tuner, early stopping callback, learning rate reduction callback, and model checkpoint callback.
            - tuner (kerastuner.tuners.BayesianOptimization): The BayesianOptimization tuner for hyperparameter search.
            - early_stopping (keras.callbacks.EarlyStopping): The early stopping callback to prevent overfitting.
            - reduce_lr (keras.callbacks.ReduceLROnPlateau): The learning rate reduction callback to adjust the learning rate during training.
            - model_checkpoint (keras.callbacks.ModelCheckpoint): The model checkpoint callback to save the best model during training.
    """

    hypermodel = HyperModel_task3(num_classes=num_classes, input_shape=input_shape)

    # Instantiate the tuner
    tuner = BayesianOptimization(
        hypermodel,
        objective='val_categorical_accuracy',
        max_trials=50,
        executions_per_trial=3,
        directory=output_dir,
        project_name='hyperparam_tuning_image',
        #overwrite=True  # Overwrite the previous results if needed
    )

    # Display the search space summary
    tuner.search_space_summary()

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(output_dir, 'best_model.keras'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False
    )

    return tuner, early_stopping, reduce_lr, model_checkpoint

#endregion

#region DATA PREPARATION FUNCTIONS

"""
1. Image Resizing
Ensure all images are of the same size since neural networks require fixed input dimensions. Resize images to a consistent size, such as (224, 224, 3) if using common architectures like VGG, ResNet, or MobileNet.
2. Data Normalization
Normalize pixel values to a range of [0, 1] by dividing by 255. This helps in faster convergence during training.
3. Data Augmentation
Apply data augmentation to artificially expand the dataset. This helps in improving the model's generalization by creating variations of the images.
4. One-Hot Encoding of Labels
Convert class labels to one-hot encoded vectors. This is essential for multi-class classification tasks.
5. Train-Validation Split
Split the data into training and validation sets to evaluate the model's performance on unseen data during training.
6. Handling Imbalanced Data
If the dataset is imbalanced, consider techniques such as class weighting, oversampling the minority class, or undersampling the majority class.
# Example of class weighting
class_weights = {0: 1., 1: 50., 2: 1., 3: 1., 4: 1.}
# During model training
model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), class_weight=class_weights, callbacks=[...])
7. Data Generator for Large Datasets
For large datasets, use data generators to load data in batches to avoid memory issues.

"""
def dataprep_old(X,y, unseendata):
    # Normalize and augment data
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    unique_values, indices, counts = np.unique(y, return_index=True, return_counts=True)

    print("target unique values:", unique_values)
    print("Counts:", counts)

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes=unique_values.size)
    y_val = to_categorical(y_val, num_classes=unique_values.size)

    # Train generator
    train_generator = datagen.flow(X_train, y_train, batch_size=32)

    # Validation generator (without augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = val_datagen.flow(X_val, y_val, batch_size=32)

    # Prediction generator (without augmentation)
    unseen_datagen = ImageDataGenerator(rescale=1./255)
    unseendata_generator = unseen_datagen.flow(unseendata, batch_size=1, shuffle=False)
    
    
    return train_generator, validation_generator, unseendata_generator

def dataprep(X, y, unseen_data):
    """
    Prepares the data for training and validation.

    This function normalizes the image data and splits it into training and validation sets.
    It also converts the labels to one-hot encoding.

    Args:
        X (numpy.ndarray): The image data.
        y (numpy.ndarray): The labels.
        unseen_data (numpy.ndarray): The unseen data for prediction.

    Returns:
        tuple: A tuple containing the training generator, validation generator, and unseen data generator.
            - train_generator (ImageDataGenerator): The training data generator.
            - validation_generator (ImageDataGenerator): The validation data generator.
            - unseen_data_generator (ImageDataGenerator): The unseen data generator.
    """

    # Normalize the data
    datagen = ImageDataGenerator(rescale=1./255)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    unique_values, indices, counts = np.unique(y, return_index=True, return_counts=True)

    print("target unique values:", unique_values)
    print("Counts:", counts)

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes=unique_values.size)
    y_val = to_categorical(y_val, num_classes=unique_values.size)

    # Train generator
    train_generator = datagen.flow(X_train, y_train, batch_size=16)

    # Validation generator (without augmentation)
    validation_generator = datagen.flow(X_val, y_val, batch_size=16)

    # Prediction generator (without augmentation)
    #unseen_data_generator = datagen.flow(unseen_data, batch_size=1, shuffle=False)
    unseen_data_generator = None

    return train_generator, validation_generator, unseen_data_generator

def preprocess_unseen_data(unseen_data, target_size=(224, 224)):
    """
    Preprocess unseen data for prediction.

    Args:
        unseen_data (numpy.ndarray): Array of unseen images.
        target_size (tuple): Desired image size.

    Returns:
        numpy.ndarray: Preprocessed image array.
    """
    # Normalize the unseen data
    unseen_datagen = ImageDataGenerator(rescale=1./255)
    unseen_data = unseen_datagen.flow(unseen_data, batch_size=1, shuffle=False)
    
    return unseen_data

#endregion

#region MODEL TRAINING
    
def train_model(train_generator, validation_generator, tuner, early_stopping, reduce_lr, model_checkpoint):
    # Perform hyperparameter tuning
    tuner.search(
        train_generator,
        epochs=30,
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        batch_size=16,
        #multi_worker_mirrored_strategy=True,
        #workers=num_cpus,
        verbose=2
)
#endregion

#region MODEL VALIDATION

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

def evaluate_best_model(validation_generator):
    """
    Loads the best model saved during hyperparameter tuning and evaluates it on the validation data.

    Args:
        output_dir (str): Directory where the best model is saved.
        validation_generator (ImageDataGenerator): Data generator for the validation data.

    Returns:
        float: The validation accuracy of the best model.
    """
    # Load the best model
    best_model_path = os.path.join(output_dir, 'best_model.keras')
    best_model = load_model(best_model_path)
    
    # Evaluate the model on validation data
    val_loss, val_accuracy = best_model.evaluate(validation_generator)
    
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    
    return val_accuracy

def validate_model(validation_generator, tuner):
    """
    Validates the best model found during hyperparameter tuning.

    This function retrieves the best model from the tuner, evaluates it on the validation data,
    and calculates various performance metrics, including accuracy, classification report, and best thresholds for each class.
    It also compares the performance using the best thresholds with the default threshold of 0.5 for each class.

    Args:
        X_val (numpy.ndarray): The validation data.
        y_val (numpy.ndarray): The validation labels (one-hot encoded).
        tuner (keras_tuner.tuners.BayesianOptimization): The BayesianOptimization tuner used for hyperparameter search.

    Returns:
        dict: A dictionary containing the best thresholds for each class.
    """
    
    best_model = tuner.get_best_models(num_models=1)[0]
    #best_model.summary()
    # Evaluate the best model
    y_pred = best_model.predict(validation_generator)
    
    
    """
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
    """


#endregion

#region PREDICTIONS

def predictions(unseen_data, best_thresholds, tuner):
    best_model = tuner.get_best_models(num_models=1)[0]

    # Get the best hyperparameters
    best_hp = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters

    # Print the best hyperparameters
    print("Best hyperparameters:")
    for key in best_hp.values.keys():
        print(f"{key}: {best_hp.get(key)}")

    #best_model.summary()
    
    y_pred = best_model.predict(unseen_data)
    
    # Apply the best threshold to make multiclass predictions
    predicted_classes = apply_thresholds(y_probs=y_pred, thresholds=best_thresholds)
    #predicted_classes = default_threshold_classification(y_probs=y_pred)
    
    save_result(predictions=predicted_classes, submission_file='submission.csv')

def save_result(predictions, submission_file):
    #y_test_probs = predictions.ravel()
    # Set the best threshold on validation data
    
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

"""
X_train: um vetor em que cada elemento é uma imagem. 
Cada imagem pode ser representada por uma matriz de formato (altura, largura, canais de cor). 
Por exemplo uma imagem do formato (224, 224, 3) seria uma matriz de 224x224 em que cada elemento é um vetor de 3 inteiros (x, y, z) de 0 a 255. 
Nesse problema, todas as imagens são (224, 224, 3).

y_train: um vetor em que o n-ésimo elemento é a classificação da n-ésima amostra de X_train. 
Todos os elementos desse vetor são do tipo int64, sendo a classificação da amostra: 0, 1, 2, 3 ou 4.
"""
def main():
    # Carrega a base de dados a partir de seu caminho
    data = np.load("{}/data/dataset.npz".format(current_path))

    X = data['X_train']
    #X = X.reshape(-1,224,224,3) #reorganiza o array em um array 1 x 224 x 224 x 3
    
    y = data['y_train']
    #y = y.reshape(-1,1) #reorganiza o array em um array 1 x 1
    
    unseen_data = data['X_test']
    #print('features: ', unseen_data.shape)
    
    unseendata, train_generator, validation_generator = dataprep(X,y, unseen_data)

    # Check dimensions of X and y from train_generator
    for batch_x, batch_y in train_generator:
        print("Batch X shape:", batch_x.shape)
        print("Batch Y shape:", batch_y.shape)
        break  # Stop after checking one batch

    print('train/test shape: ', X.shape)
    print('target trainning/testing dimensions: ', y.shape)

    unique_values, indices, counts = np.unique(y, return_index=True, return_counts=True)



    tuner, early_stopping, reduce_lr, model_checkpoint = init_class(num_classes=unique_values.size, input_shape=(X.shape[1], X.shape[2],X.shape[3]))


    if TRAIN_MODEL:
        train_model(train_generator= train_generator, validation_generator=validation_generator,tuner=tuner, early_stopping=early_stopping, reduce_lr=reduce_lr , model_checkpoint=model_checkpoint)

    best_thresholds = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}
    if VALIDATE_MODEL:    
        evaluate_best_model(validation_generator=validation_generator)
        best_thresholds = validate_model(validation_generator=validation_generator, tuner=tuner)
    
    
    if PREDICT_DATA:
        predictions(unseen_data=unseendata, best_thresholds=best_thresholds, tuner=tuner)
    
if __name__ == "__main__":
    main()