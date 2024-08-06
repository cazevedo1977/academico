import keras
import keras_tuner
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Concatenate, Embedding, Flatten
from keras.initializers import GlorotNormal
from keras.optimizers import SGD, Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_tuner import HyperModel, RandomSearch
from keras_tuner.tuners import BayesianOptimization
import multiprocessing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from pathlib import Path
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


current_path = Path(__file__).parent
TRAIN_MODEL = False
VALIDATE_MODEL = True #tb recupera o threshold
PREDICT_DATA = False


# Ensure the directory exists or create it
output_dir = os.path.abspath('C:\\Caio\\decalearn\\task1')
os.makedirs(output_dir, exist_ok=True)

# Determine the number of available CPUs
num_cpus = multiprocessing.cpu_count()

class MyHyperModel(HyperModel):
    
    def __init__(self):
        super().__init__()
    

    def build(self, hp):
        initializer = GlorotNormal(seed=12227)

        
        inputs = Input(shape=(16,)) #quando no_data_prep
        #inputs = Input(shape=(13,)) #quando data_prep_full
        #inputs = Input(shape=(15,)) #quando standard_scaler
        H = Dense(
            units=hp.Int('units_1', min_value=32, max_value=1024, step=32),
            activation=hp.Choice('activation_1', values=['relu', 'swish']),
            kernel_initializer=initializer
        )(inputs)
        H = BatchNormalization()(H)
        
        
        for i in range(hp.Int('num_layers', 1, 3)):
            H = Dense(
                units=hp.Int(f'units_{i+2}', min_value=32, max_value=1024, step=32),
                activation=hp.Choice(f'activation_{i+2}', values=['relu', 'swish']),
                kernel_initializer=initializer
            )(H)
            H = BatchNormalization()(H)

        
        outputs = Dense(1, activation='sigmoid', kernel_initializer=initializer)(H)
        model = Model(inputs=inputs, outputs=outputs) 

        model.compile(
            optimizer=hp.Choice('optimizer', values=['adam']),
            loss=BinaryCrossentropy(),
            metrics=[BinaryAccuracy()]
        )
        
        return model

hypermodel = MyHyperModel()

# Bayesian Optimization with parallel execution
tuner = BayesianOptimization(
    hypermodel,
    objective='val_binary_accuracy',
    max_trials=20,
    executions_per_trial=3,
    directory=output_dir,
    project_name='hyperparam_tuning_numeric_no_data_prep',
    #distribution_strategy='multi_worker_mirrored'  # Or another appropriate strategy
)


tuner.search_space_summary()
# Add early stopping callback
#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)


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


def dataprep_split(dataset, target, split = True):
    X_train, X_test, y_train, y_test = None, None, None, None

    if split:
        seed = 7
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(dataset, target, test_size=test_size, random_state=seed)
    
    return dataset, X_train, X_test, y_train, y_test

def dataprep_standardScaler(dataset, target, split = True):
    X_train, X_test, y_train, y_test = None, None, None, None
    
    pearson_threshold = 0.9
    # Create a pipeline for standardization and PCA
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        #('pca', PCA(n_components=0.95))
    ])

    dataset = remove_highly_correlated_columns(dataset, pearson_threshold)
    print('features n-correlacionadas: ', dataset.shape)

    if split:
        seed = 7
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(dataset, target, test_size=test_size, random_state=seed)
        
        # Transform the features
        X_train = pipeline.fit_transform(X_train)
        X_test = pipeline.transform(X_test)
        #print(X_train.min(axis=0), X_train.max(axis=0))
    else:
        dataset  = pipeline.fit_transform(dataset)

    return dataset, X_train, X_test, y_train, y_test

def dataprep_full(dataset, target, split = True):
    X_train, X_test, y_train, y_test = None, None, None, None
    
    pearson_threshold = 0.9
    # Create a pipeline for standardization and PCA
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95))
    ])

    dataset = remove_highly_correlated_columns(dataset, pearson_threshold)
    print('features n-correlacionadas: ', dataset.shape)

    if split:
        seed = 7
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(dataset, target, test_size=test_size, random_state=seed)
        
        # Transform the features
        X_train = pipeline.fit_transform(X_train)
        X_test = pipeline.transform(X_test)
        #print(X_train.min(axis=0), X_train.max(axis=0))
    else:
        dataset  = pipeline.fit_transform(dataset)

    return dataset, X_train, X_test, y_train, y_test


def data_load_and_prep():
    # Carrega a base de dados a partir de seu caminho
    DATA_FILE = "{}/data/dataset.npz".format(current_path)
    data = np.load(DATA_FILE)

    XX_train = data['X_train']
    yy_train = data['y_train']
    yy_train = yy_train.reshape(-1,1) #reorganiza o array em um array 1 x 1
        
    X = XX_train
    y = yy_train

    unseen_data = data['X_test']
    print('features: ', unseen_data.shape)


    X, X_train, X_val, y_train, y_val = dataprep_split(X, y, split = True)
    
    #X, X_train, X_val, y_train, y_val = dataprep_pearson(X, y, split = True)

    #X, X_train, X_val, y_train, y_val = dataprep_standardScaler(X, y, split = True)
    #unseen_data, _, _, _, _ = dataprep_standardScaler(unseen_data, y, split = False)

    #X, X_train, X_val, y_train, y_val = dataprep_full(X, y, split = True)
    #unseen_data, _, _, _, _ = dataprep_full(unseen_data, y, split = False)

    print('features train: ', X_train.shape)
    print('features test: ', X_val.shape)
    print('target trainning dimensions: ', y_train.shape)
    print('target testing dimensions: ', y_val.shape)

    return X_train, y_train, X_val, y_val, unseen_data

def train_model(X_train, y_train, X_val, y_val):
    #tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val), batch_size=64, workers=num_cpus)
    
    # Perform the search
    tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val), batch_size=64, workers=num_cpus, callbacks=[early_stopping, reduce_lr])

# Function to find the best threshold
def find_best_threshold(y_true, y_pred_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)  # Avoid division by zero
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    return best_threshold

def validate_model(X_val, y_val):
    # Once you find promising hyperparameters, you can refine the tuning on the full dataset
    
    #best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    #model = hypermodel.build(best_hps)
    #model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[early_stopping])
    #model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))
    
    #y_pred = model.predict(X_val)

    #OUUUU
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.summary()
    # Evaluate the best model
    y_pred = best_model.predict(X_val).ravel()
    
    best_threshold = find_best_threshold(y_true=y_val, y_pred_probs=y_pred)
    print(f"Best Threshold: {best_threshold}")

    # Apply the best threshold to make binary predictions
    y_test_pred = (y_pred > best_threshold).astype(int)
    print('Acurácia best threshold: {:.2f}%'.format(100*accuracy_score(y_val, y_test_pred)))
    
    threshold_default = 0.5
    y_test_pred = (y_pred > threshold_default).astype(int)
    print('Acurácia default threshold: {:.2f}%'.format(100*accuracy_score(y_val, y_test_pred)))
    
    
    #acc = accuracy_score(np.round(y_val), np.round(y_pred))
    #print('Validation accuracy: {:.2f}%'.format(acc * 100))
    return best_threshold

def predictions(unseen_data,best_threshold):
    best_model = tuner.get_best_models(num_models=1)[0]
    

    # Get the best hyperparameters
    best_hp = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters

    # Print the best hyperparameters
    print("Best hyperparameters:")
    for key in best_hp.values.keys():
        print(f"{key}: {best_hp.get(key)}")

    best_model.summary()

    
    y_pred = best_model.predict(unseen_data)
    save_result(predictions=y_pred, submission_file='{}/submission.csv'.format(current_path), best_threshold=best_threshold)

def save_result(predictions, submission_file, best_threshold=0.5):
    y_test_probs = predictions.ravel()

    # Apply the best threshold to make binary predictions
    predicted_classes = (y_test_probs > best_threshold).astype(int)

    unique_values, indices, counts = np.unique(predicted_classes, return_index=True, return_counts=True)
    print("Unique values:", unique_values)
    print("Indices:", indices)
    print("Counts:", counts)
    print("Proportion:", counts[0]/counts[1])
    
    
    # Create a DataFrame with zipped data and column names
    #num_samples = unseen_data.shape[0]
    
    df = pd.DataFrame(list(zip(range(1, len(predicted_classes) + 1), predicted_classes)), columns=['ID', 'Prediction'])
    #df = pd.DataFrame({'ID': np.arange(1, num_samples + 1),'Prediction': predicted_classes})
    df.to_csv(submission_file, index=False)
    # Print the predictions
    #print("Predicted probabilities:\n", y_test_probs)
    #print("Predicted classes:\n", predicted_classes)

def main():
    print('\n####################  PCS-5022 - Redes Neurais e Aprendizado Profundo - Decalearn Task 1 #######################\n')
    X_train, y_train, X_val, y_val, unseen_data =  data_load_and_prep()
    # Define a subset of your data for quick tuning
    x_train_small, y_train_small = X_train[:5000], y_train[:5000]
    x_val_small, y_val_small = X_val[:400], y_val[:400]
    
    best_threshold = 0.5
    if TRAIN_MODEL:
        train_model(X_train=X_train, y_train=y_train, X_val= X_val, y_val= y_val)
        #train_model(X_train=x_train_small, y_train=y_train_small, X_val= x_val_small, y_val= y_val_small)

    if VALIDATE_MODEL:    
        best_threshold =  validate_model(X_val, y_val)
    
    if PREDICT_DATA:
        predictions(unseen_data=unseen_data, best_threshold = best_threshold)

if __name__ == "__main__":
    main()