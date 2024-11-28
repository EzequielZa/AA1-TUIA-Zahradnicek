import joblib
import pandas as pd
from transformers import PreprocessDataTransformer, KNNImputerTransformer, ScaleTransformer
from model import NeuralNetworkTensorFlow

pipeline = joblib.load('pipeline.pkl')
input = pd.read_csv('/files/input.csv')

output = pipeline.predict(input)
pd.DataFrame(output, columns=['RainTomorrow']).to_csv('/files/output.csv', index=False)