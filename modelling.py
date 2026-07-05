import os
import joblib
import numpy as np
import pandas as pd
import random
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    MinMaxScaler
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

from pyswarms.single.global_best import GlobalBestPSO
from sklearn.utils import shuffle

# REPRODUCIBILITY
SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# CREATE FOLDER
os.makedirs("saved_model", exist_ok=True)

# LOAD DATASET
df = pd.read_csv("dataset/eeg_emotion.csv")

# PREPROCESSING

# FEATURE & LABEL
X = df.drop("label", axis=1)
y = df["label"]

# HANDLE MISSING VALUES
X = X.fillna(0)

# LABEL ENCODING
label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y.astype(str))

# ONE HOT ENCODING
ohe = OneHotEncoder(sparse_output=False)

y_onehot = ohe.fit_transform(
    y_encoded.reshape(-1, 1)
)

# SCALING
scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

# TRAIN TEST SPLIT
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled,
    y_onehot,
    test_size=0.3,
    stratify=y_encoded,
    random_state=SEED
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    stratify=np.argmax(y_temp, axis=1),
    random_state=SEED
)

# SAVE DATA SPLIT

joblib.dump(
    X_train,
    "saved_model/X_train.pkl"
)

joblib.dump(
    y_train,
    "saved_model/y_train.pkl"
)

joblib.dump(
    X_val,
    "saved_model/X_val.pkl"
)

joblib.dump(
    y_val,
    "saved_model/y_val.pkl"
)

joblib.dump(
    X_test,
    "saved_model/X_test.pkl"
)

joblib.dump(
    y_test,
    "saved_model/y_test.pkl"
)

n_features = X_train.shape[1]
n_classes = y_train.shape[1]

# MODEL FUNCTION
def create_model():

    model = Sequential([
        Dense(32, activation='relu', input_shape=(n_features,)),
        Dense(16, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer=SGD(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# WEIGHT UTILITIES
def get_shapes(model):

    return [w.shape for w in model.get_weights()]


def flatten_weights(model):

    flat = []

    for w in model.get_weights():
        flat.extend(w.flatten())

    return np.array(flat)


def set_shape(flat, shapes):

    weights = []

    idx = 0

    for s in shapes:

        size = np.prod(s)

        weights.append(
            flat[idx:idx + size].reshape(s)
        )

        idx += size

    return weights

# TRAIN BACKPROPAGATION MODEL
bp_model = create_model()

bp_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=800,
    batch_size=16,
    verbose=0
)

# TRAIN BP + PSO MODEL
bp_pso_model = create_model()

# START FROM BP WEIGHTS
bp_pso_model.set_weights(
    bp_model.get_weights()
)

shapes = get_shapes(bp_pso_model)

dim = sum(
    np.prod(s) for s in shapes
)

flat0 = flatten_weights(bp_pso_model)

lb = flat0 - 0.05
ub = flat0 + 0.05

# FITNESS FUNCTION
def fitness_function(position):

    losses = []

    Xs, Ys = shuffle(
        X_train,
        y_train,
        random_state=SEED
    )

    Xs = Xs[:500]
    Ys = Ys[:500]

    temp_model = create_model()

    for particle in position:

        temp_model.set_weights(
            set_shape(particle, shapes)
        )

        loss = temp_model.evaluate(
            Xs,
            Ys,
            verbose=0
        )[0]

        losses.append(loss)

    return np.array(losses)

# PSO OPTIMIZATION
optimizer = GlobalBestPSO(
    n_particles=40,
    dimensions=dim,
    bounds=(lb, ub),
    options={
        'c1': 1.5,
        'c2': 1.5,
        'w': 0.5
    }
)

best_cost, best_pos = optimizer.optimize(
    fitness_function,
    iters=50,
    verbose=False
)

best_weights = set_shape(
    best_pos,
    shapes
)

bp_pso_model.set_weights(
    best_weights
)

# FINE TUNING
bp_pso_model.compile(
    optimizer=SGD(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

bp_pso_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    verbose=0
)

# SAVE MODELS
bp_model.save(
    "saved_model/bp_model.keras"
)

bp_pso_model.save(
    "saved_model/bp_pso_model.keras"
)

joblib.dump(
    scaler,
    "saved_model/scaler.pkl"
)

joblib.dump(
    label_encoder,
    "saved_model/label_encoder.pkl"
)

joblib.dump(
    list(X.columns),
    "saved_model/feature_columns.pkl"
)

joblib.dump(
    X_test,
    "saved_model/X_test.pkl"
)

joblib.dump(
    y_test,
    "saved_model/y_test.pkl"
)

from sklearn.metrics import accuracy_score

pred = bp_pso_model.predict(
    X_test,
    verbose=0
)

pred_class = np.argmax(pred, axis=1)

true_class = np.argmax(y_test, axis=1)

acc = accuracy_score(
    true_class,
    pred_class
)

print("BP+PSO Test Accuracy =", acc)

