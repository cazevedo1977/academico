#region LIBs 
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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
#import tensorflow as tf
import multiprocessing

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#endregion


TRAIN_MODEL = False
VALIDATE_MODEL = True
PREDICT_DATA = True
current_path = Path(__file__).parent
output_dir = os.path.abspath(current_path)
# Ensure the directory exists or create it
os.makedirs(output_dir, exist_ok=True)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Determine the number of available CPUs
num_cpus = multiprocessing.cpu_count()



class HyperModel_task1(HyperModel):
    
    def __init__(self, num_categories, embedding_dims):
        super().__init__()
        self.num_categories = num_categories  # List with the number of unique categories for each categorical feature
        self.embedding_dims = embedding_dims  # List with the embedding dimensions for each categorical feature
    

    def build(self, hp):
        initializer = GlorotNormal(seed=12227)


        # Input for categorical features
        categorical_inputs = []
        embeddings = []
        for i, num_category in enumerate(self.num_categories):
            input_i = Input(shape=(1,), name=f'cat_input_{i}')
            embedding_i = Embedding(input_dim=num_category, output_dim=self.embedding_dims[i], input_length=1)(input_i)
            embedding_i = Flatten()(embedding_i)
            categorical_inputs.append(input_i)
            embeddings.append(embedding_i)

        # Input for numeric features - only 5 are numeric
        numeric_input = Input(shape=(5,), name='numeric_input')

        # Concatenate all features
        all_features = Concatenate()([numeric_input] + embeddings)

        # Dense layers
        H = Dense(
            units=hp.Int('units_1', min_value=32, max_value=1024, step=32),
            activation=hp.Choice('activation_1', values=['relu', 'swish']),
            kernel_initializer=initializer
        )(all_features)
        H = BatchNormalization()(H)



        
        for i in range(hp.Int('num_layers', 1, 3)):
            H = Dense(
                units=hp.Int(f'units_{i+2}', min_value=32, max_value=1024, step=32),
                activation=hp.Choice(f'activation_{i+2}', values=['relu', 'swish']),
                kernel_initializer=initializer
            )(H)
            H = BatchNormalization()(H)

        
        outputs = Dense(1, activation='sigmoid', kernel_initializer=initializer)(H)
        #model = Model(inputs=inputs, outputs=outputs) - qdo todas as features são numericas
        model = Model(inputs=[numeric_input] + categorical_inputs, outputs=outputs)

        model.compile(
            optimizer=hp.Choice('optimizer', values=['adam']),
            loss=BinaryCrossentropy(),
            metrics=[BinaryAccuracy()]
        )
        
        return model

DATA_FILE = "{}/data/dataset.npz".format(current_path)
data = np.load(DATA_FILE)

X = data['X_train']
y = data['y_train']
y = y.reshape(-1,1) #reorganiza o array em um array 1 x 1
unseen_data = data['X_test']
    
categorical_indices = [0,3,4,5,6,8,9,11,12,13,14]
# Preprocess data to determine the number of unique categories
num_categories = [len(np.unique(X[:, idx])) for idx in categorical_indices]
# Set embedding dimensions (a simple heuristic, e.g., embedding dimension = min(num_categories // 2, 50))
embedding_dims = [min(num_category // 2, 50) for num_category in num_categories]

hypermodel = HyperModel_task1(num_categories=num_categories, embedding_dims=embedding_dims)

# Bayesian Optimization with parallel execution
tuner = BayesianOptimization(
    hypermodel,
    objective='val_binary_accuracy',
    max_trials=20,
    executions_per_trial=3,
    directory=output_dir,
    project_name='hyperparam_tuning_categorical'
    #distribution_strategy='multi_worker_mirrored'  # Or another appropriate strategy
)

# Combine processed categorical and numeric features
def preprocess_input(X):
    
    numeric_indices = [i for i in range(X.shape[1]) if i not in categorical_indices]
    
    # Initialize label encoders for categorical features
    label_encoders = [LabelEncoder() for _ in categorical_indices]

    # Encode categorical features
    X_cat = np.column_stack([
        label_encoders[i].fit_transform(X[:, idx]) for i, idx in enumerate(categorical_indices)
    ])

    X_train_cat = np.column_stack([
        label_encoders[i].transform(X[:, idx]) for i, idx in enumerate(categorical_indices)
    ])
    X_num = X[:, numeric_indices]
    return [X_num] + [X_cat[:, i].reshape(-1, 1) for i in range(X_cat.shape[1])]

def data_prep_categorical(X):
    # Categorical feature indices
    #categorical_indices = [0,3,4,5,6,8,9,11,12,13,14]

    # Preprocess data to determine the number of unique categories
    #num_categories = [len(np.unique(X[:, idx])) for idx in categorical_indices]
    # Set embedding dimensions (a simple heuristic, e.g., embedding dimension = min(num_categories // 2, 50))
    #embedding_dims = [min(num_category // 2, 50) for num_category in num_categories]

    # Initialize label encoders for categorical features
    #label_encoders = [LabelEncoder() for _ in categorical_indices]

    # Encode categorical features
    #X_train_cat = np.column_stack([
    #    label_encoders[i].fit_transform(X[:, idx]) for i, idx in enumerate(categorical_indices)
    #])

    # Get numeric features
    #numeric_indices = [i for i in range(X.shape[1]) if i not in categorical_indices]
    #X_num = X[:, numeric_indices]

    X_processed = preprocess_input(X)

    return X_processed


# Add early stopping callback
#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)


def dataprep_split(dataset, target, split = True):
    X_train, X_test, y_train, y_test = None, None, None, None

    if split:
        seed = 7
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(dataset, target, test_size=test_size, random_state=seed)
    
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
    

    X  = data_prep_categorical(X)
    unseen_data = data_prep_categorical(unseen_data)

    #X, X_train, X_val, y_train, y_val = dataprep_split(X, y, split = True)
    #X, X_train, X_test, y_train, y_test = dataprep_pearson(X, y, split = True)
    #X, X_train, X_test, y_train, y_test = dataprep_standardScaler(X, y, split = True)
    #X, X_train, X_test, y_train, y_test = dataprep_full(X, y, split = True)

    #print('features train: ', X_train.shape)
    #print('features test: ', X_val.shape)
    #print('target trainning dimensions: ', y_train.shape)
    #print('target testing dimensions: ', y_val.shape)

    #return X_train, y_train, X_val, y_val, unseen_data
    return X, y, unseen_data

#def train_model(X_train, y_train, X_val, y_val):
def train_model(X_train, y_train):

    tuner.search_space_summary()

    #tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val), batch_size=64, workers=num_cpus)
    
    # Perform the search
    #tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val), batch_size=64, workers=num_cpus, callbacks=[early_stopping, reduce_lr])
    # Perform the hyperparameter search
    tuner.search(
        X_train,
        y_train,
        epochs=50,
        validation_split=0.2,
        batch_size=64, 
        workers=num_cpus,
        callbacks=[early_stopping, reduce_lr]
    )


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
    y_pred = best_model.predict(X_val)
    y_pred = y_pred.ravel()
    
    #best_threshold = find_best_threshold_for_f1score(y_true=y, y_pred_probs=y_pred)
    best_threshold = find_best_threshold_for_accuracy(y_true=y, y_pred_probs=y_pred)
    print(f"Best Threshold: {best_threshold}")

    # Apply the best threshold to make binary predictions
    y_test_pred = (y_pred > best_threshold).astype(int)
    print('Acurácia best threshold: {:.2f}%'.format(100*accuracy_score(y_val, y_test_pred)))
    
    threshold_default = 0.5
    y_test_pred = (y_pred > threshold_default).astype(int)
    print('Acurácia default threshold: {:.2f}%'.format(100*accuracy_score(y_val, y_test_pred)))
    
    return best_threshold

def predictions(unseen_data, best_threshold):
    best_model = tuner.get_best_models(num_models=1)[0]

    # Get the best hyperparameters
    best_hp = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters

    # Print the best hyperparameters
    print("Best hyperparameters:")
    for key in best_hp.values.keys():
        print(f"{key}: {best_hp.get(key)}")

    #best_model.summary()

    
    y_pred = best_model.predict(unseen_data)
    save_result(predictions=y_pred, submission_file='submission.csv',best_threshold=best_threshold)

# Function to find the best threshold
def find_best_threshold_for_f1score(y_true, y_pred_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)  # Avoid division by zero
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    return best_threshold

def find_best_threshold_for_accuracy(y_true, y_pred_probs):
    """
    Compute the best threshold to maximize accuracy in a binary classification situation.

    Args:
    y_true (numpy array): True labels (binary, 0 or 1).
    y_pred_probs (numpy array): Predicted probabilities from the model.

    Returns:
    float: The best threshold that maximizes accuracy.
    """
    best_threshold = 0.0
    best_accuracy = 0.0

    # Check thresholds from 0 to 1 with a step of 0.01
    for threshold in np.arange(0.0, 1.01, 0.01):
        y_pred = (y_pred_probs >= threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold



def save_result(predictions, submission_file,best_threshold = 0.5):
    y_test_probs = predictions.ravel()
    # Set the best threshold on validation data
    
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
    
    
    #X_train, y_train, X_val, y_val, unseen_data =  data_load_and_prep()
    X, y, unseen_data =  data_load_and_prep()
    # Define a subset of your data for quick tuning
    #x_small, y_small = X[:10000], y[:10000]
    #x_val_small, y_val_small = X_val[:400], y_val[:400]
    best_threshold = 0.5
    if TRAIN_MODEL:
        tuner = train_model(X_train=X, y_train=y)
        
    if VALIDATE_MODEL:    
        best_threshold = validate_model(X, y)
    
    if PREDICT_DATA:
        predictions(unseen_data=unseen_data, best_threshold=best_threshold)


if __name__ == "__main__":
    main()