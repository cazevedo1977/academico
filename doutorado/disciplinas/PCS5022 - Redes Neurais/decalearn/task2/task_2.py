#region LIBs
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import tensorflow as tf


from keras.utils import to_categorical # type: ignore
from keras_tuner import HyperModel, BayesianOptimization




from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2
from keras.layers import Input
from tensorflow.keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy

from tensorflow.keras.models import Model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from pathlib import Path

#endregion 
TRAIN_MODEL = False
VALIDATE_MODEL = True
PREDICT_DATA = True
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
 - Increased Units and Layers: I increased the maximum number of units and allowed up to 5 hidden layers. More layers with different configurations to capture more complex patterns.
 - Regularization: Added L2 regularization and dropout layers to each dense layer to reduce overfitting.
 - Activation Functions: Added 'tanh' as an additional activation function choice.
 - Advanced Optimizers: Included 'sgd' and 'rmsprop' optimizers as choices.
 - Tuning Range: Increased the range for some hyperparameters to explore more configurations.
 - Max Trials: Increased max_trials to 50 for a more exhaustive search of hyperparameters.
 - Learning Rate Schedulers: Implement learning rate schedules for better convergence.
 - Feature Engineering: Apply techniques like Principal Component Analysis (PCA) for dimensionality reduction if feasible.

'''

class HyperModel_task2(HyperModel):
    
    def __init__(self, num_classes, input_dim):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        
    def build(self, hp):
        initializer = tf.keras.initializers.GlorotNormal(seed=12227)

        inputs = Input(shape=(self.input_dim,))
        H = Dense(
            units=hp.Int('units_1', min_value=64, max_value=1024, step=64),
            activation=hp.Choice('activation_1', values=['relu', 'swish', 'tanh']),
            kernel_initializer=initializer,
            kernel_regularizer=tf.keras.regularizers.l2(hp.Float('l2_1', min_value=1e-5, max_value=1e-2, sampling='LOG'))
        )(inputs)
        H = BatchNormalization()(H)
        H = Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1))(H)
        
        for i in range(hp.Int('num_layers', 1, 5)):
            H = Dense(
                units=hp.Int(f'units_{i+2}', min_value=64, max_value=1024, step=64),
                activation=hp.Choice(f'activation_{i+2}', values=['relu', 'swish', 'tanh']),
                kernel_initializer=initializer,
                kernel_regularizer=tf.keras.regularizers.l2(hp.Float(f'l2_{i+2}', min_value=1e-5, max_value=1e-2, sampling='LOG'))
            )(H)
            H = BatchNormalization()(H)
            H = Dropout(rate=hp.Float(f'dropout_{i+2}', min_value=0.1, max_value=0.5, step=0.1))(H)

        outputs = Dense(self.num_classes, activation='softmax', kernel_initializer=initializer)(H)
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop']),
            loss=CategoricalCrossentropy(),
            metrics=[CategoricalAccuracy()]
        )
        
        return model

def init_class(num_classes, input_dim):
    """
    Initializes the hyperparameter tuning process for a multi-class classification model.

    This function creates a BayesianOptimization tuner, defines callbacks for early stopping, learning rate reduction, and model checkpointing,
    and returns the tuner and callbacks for use in the model training process.

    Args:
        num_classes (int): The number of classes in the multi-class classification problem.
        input_dim (int): The dimensionality of the input features.

    Returns:
        tuple: A tuple containing the tuner, early stopping callback, learning rate reduction callback, and model checkpoint callback.
            - tuner (keras_tuner.tuners.BayesianOptimization): The BayesianOptimization tuner for hyperparameter search.
            - early_stopping (keras.callbacks.EarlyStopping): The early stopping callback to prevent overfitting.
            - reduce_lr (keras.callbacks.ReduceLROnPlateau): The learning rate reduction callback to adjust the learning rate during training.
            - model_checkpoint (keras.callbacks.ModelCheckpoint): The model checkpoint callback to save the best model during training.
    """

    hypermodel = HyperModel_task2(num_classes=num_classes, input_dim=input_dim)

    # Instantiate the tuner
    tuner = BayesianOptimization(
        hypermodel,
        objective='val_categorical_accuracy',
        max_trials=50,
        executions_per_trial=3,
        directory=output_dir,
        project_name='hyperparam_tuning_numeric',
        # overwrite=True  # Overwrite the previous results
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

def dataprep_full_pca(dataset_train_val, target, unseendata):
    """
    Prepares data for a ML model by applying standardization and PCA. 
    It splits the data into training and testing sets, fits the pipeline to the training data, and transforms all the data using the learned parameters. 
    This process helps to improve the performance of many machine learning algorithms by scaling the features and reducing the dimensionality of the data.

    Args:
        dataset_train_val (numpy.ndarray): The dataset containing both training and validation data.
        target (numpy.ndarray): The target labels for the training and validation data.
        unseendata (numpy.ndarray): The unseen data for which predictions will be made.

    Returns:
        tuple: A tuple containing the transformed unseen data, training data, testing data, training labels, and testing labels.
            - unseendata (numpy.ndarray): The transformed unseen data.
            - X_train (numpy.ndarray): The transformed training data.
            - X_test (numpy.ndarray): The transformed testing data.
            - y_train (numpy.ndarray): The training labels.
            - y_test (numpy.ndarray): The testing labels.
    """
    
    X_train, X_test, y_train, y_test = None, None, None, None
    # Create a pipeline for standardization and PCA
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95))
    ])
    
    seed = 7
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(dataset_train_val, target, test_size=test_size, random_state=seed)
    
    # Transform the features
    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)
    #print('train shape:{} test shape: {} '.format(X_train.shape, X_test.shape))
    
    unseendata  = pipeline.transform(unseendata)
    
    return unseendata, X_train, X_test, y_train, y_test

def dataprep_full_scalar(dataset_train_val, target, unseendata):
    """
    Prepares data for a ML model by applying standardization. 
    It splits the data into training and testing sets, fits the pipeline to the training data, and transforms all the data using the learned parameters. 
    This process helps to improve the performance of many machine learning algorithms by scaling the features of the data.

    Args:
        dataset_train_val (numpy.ndarray): The dataset containing both training and validation data.
        target (numpy.ndarray): The target labels for the training and validation data.
        unseendata (numpy.ndarray): The unseen data for which predictions will be made.

    Returns:
        tuple: A tuple containing the transformed unseen data, training data, testing data, training labels, and testing labels.
            - unseendata (numpy.ndarray): The transformed unseen data.
            - X_train (numpy.ndarray): The transformed training data.
            - X_test (numpy.ndarray): The transformed testing data.
            - y_train (numpy.ndarray): The training labels.
            - y_test (numpy.ndarray): The testing labels.
    """
    
    X_train, X_test, y_train, y_test = None, None, None, None
    # Create a pipeline for standardization
    pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    seed = 7
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(dataset_train_val, target, test_size=test_size, random_state=seed)
    
    # Transform the features
    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)
    print('train shape:{} test shape: {} '.format(X_train.shape, X_test.shape))
    
    unseendata  = pipeline.transform(unseendata)
    
    return unseendata, X_train, X_test, y_train, y_test

def dataprep_pearson(dataset, target, split=True):
    """
    Prepares data for a ML model by removing highly correlated features.
    It calculates the Pearson correlation matrix for the dataset and removes columns based on a specified threshold.
    Optionally splits the data into training and testing sets for model evaluation.

    Args:
        dataset (numpy.ndarray): The dataset containing the features.
        target (numpy.ndarray): The target labels for the dataset.
        split (bool, optional): Whether to split the data into training and testing sets. Defaults to True.

    Returns:
        tuple: A tuple containing the prepared data:
            - dataset (numpy.ndarray): The dataset with highly correlated features removed.
            - X_train (numpy.ndarray): The training data (if split is True).
            - X_test (numpy.ndarray): The testing data (if split is True).
            - y_train (numpy.ndarray): The training labels (if split is True).
            - y_test (numpy.ndarray): The testing labels (if split is True).
    """
    X_train, X_test, y_train, y_test = None, None, None, None
    
    pearson_threshold = 0.9
    dataset = remove_highly_correlated_columns(dataset, pearson_threshold)
    print('non-correlated features: ', dataset.shape)

    if split:
        seed = 7
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(dataset, target, test_size=test_size, random_state=seed)
    
    return dataset, X_train, X_test, y_train, y_test

def remove_highly_correlated_columns(data, threshold=0.9):
    """
    Remove columns from a NumPy array where the correlation coefficient
    is greater than a specified threshold.

    Parameters:
    - data: np.ndarray, the input array.
    - threshold: float, the correlation coefficient threshold.

    Returns:
    - np.ndarray, the array with highly correlated columns removed.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a NumPy array.")
    
    corr_matrix = np.corrcoef(data, rowvar=False)
    to_remove = set()
    
    for i in range(corr_matrix.shape[0]):
        for j in range(i+1, corr_matrix.shape[0]):
            if abs(corr_matrix[i, j]) > threshold:
                to_remove.add(j)

    columns_to_keep = [i for i in range(data.shape[1]) if i not in to_remove]
    filtered_data = data[:, columns_to_keep]

    return filtered_data

#endregion

#region MODEL TRAINING
    
def train_model(X_train, y_train, X_val, y_val, tuner, early_stopping, reduce_lr, model_checkpoint):
    # Start the hyperparameter search
    tuner.search(
        x=X_train,  # Replace with your training data
        y=y_train,  # Replace with your training labels
        epochs=20, #50 (melhor)
        validation_data=(X_val, y_val),  # Replace with your validation data
        batch_size=64,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
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

    
def validate_model(X_val, y_val, tuner):
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
    y_pred = best_model.predict(X_val)
    
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



def main():
    # Carrega a base de dados a partir de seu caminho
    data = np.load('data/dataset.npz')
    #print(data.files)

    X = data['X_train']
    y = data['y_train']
    y = y.reshape(-1,1) #reorganiza o array em um array 1 x 1
    
    unseen_data = data['X_test']
    #print('features: ', unseen_data.shape)
    
    #X, X_train, X_test, y_train, y_test = dataprep_pearson(X, y, split = True)
    #X, X_train, X_test, y_train, y_test = dataprep_standardScaler(X, y, split = True)
    unseendata, X_train, X_test, y_train, y_test = dataprep_full_pca(unseendata= unseen_data, dataset_train_val=X, target=y)
    
    #combined_array = np.concatenate((X_train, X_test), axis=0)
    #df = pd.DataFrame(combined_array.reshape(X_train.shape[0] + X_test.shape[0], -1))
    #combined_array_y = np.concatenate((y_train, y_test), axis=0)
    #df['target'] = combined_array_y
    #df.to_csv('dataset.csv')
    
    #df = pd.DataFrame(unseen_data)
    #df['target'] = ''
    #df.to_csv('unseen_data.csv')

    print('train shape: ', X_train.shape)
    print('test shape: ', X_test.shape)
    print('target trainning dimensions: ', y_train.shape)
    print('target testing dimensions: ', y_test.shape)

    unique_values, indices, counts = np.unique(y_train, return_index=True, return_counts=True)

    print("target unique values:", unique_values)
    print("Counts:", counts)

    # One-hot encode the labels
    y_train = to_categorical(y_train, num_classes=unique_values.size)
    y_test = to_categorical(y_test, num_classes=unique_values.size)
    
    tuner, early_stopping, reduce_lr, model_checkpoint = init_class(num_classes=unique_values.size,input_dim=X_train.shape[1])

    if TRAIN_MODEL:
        train_model(X_train=X_train, y_train=y_train, X_val= X_test, y_val= y_test, tuner=tuner, early_stopping=early_stopping, reduce_lr=reduce_lr , model_checkpoint=model_checkpoint)

    best_thresholds = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}
    if VALIDATE_MODEL:    
        best_thresholds = validate_model(X_val=X_test, y_val=y_test,tuner=tuner)
    
    
    if PREDICT_DATA:
        predictions(unseen_data=unseendata, best_thresholds=best_thresholds, tuner=tuner)
    
if __name__ == "__main__":
    main()