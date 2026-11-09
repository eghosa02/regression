from model import LinearRegressionModel
from metrics import R_squared_real
from dataset import load_dataset


dataset = load_dataset("dataset").reset_index()
X, Y = dataset[['age', 'BMI', 'sodium', 'smoker', 'gender']], dataset['arterial_pressure']
model = LinearRegressionModel(beta=None)
model.train(X, Y)
res = model.get_stats()['residuals']

print(R_squared_real(res, Y))