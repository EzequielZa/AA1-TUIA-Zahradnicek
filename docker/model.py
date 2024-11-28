from tensorflow.keras import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import numpy as np
from sklearn.metrics import f1_score

# Modelo de red neuronal
class NeuralNetworkTensorFlow:

    def __init__(self, learning_rate=0.00021665975986665252, epochs=350):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Dense(48, activation='tanh', input_shape=(25,)),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        print("n° de parámetros:", model.count_params())
        return model

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        history = self.model.fit(X, y, epochs=self.epochs, batch_size=32, verbose=0)
        return history.history

    def predict(self, X):
        X = np.array(X)
        predictions = self.model.predict(X)
        return (predictions > 0.5).astype(int)

    def score(self, X, y):
        y_pred = self.predict(X)
        return f1_score(y, y_pred)