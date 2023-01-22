from src.datasets import Datasets
from src.models import SingleLayerNeuralModel, MultiLayerNeuralModel

# SINGLE NEURON MODEL ON SINGLE LAYER

X_train, y_train = Datasets.get_random_blobs_set(n_samples=1000)
X_test, y_test = Datasets.get_random_blobs_set(n_samples=200)

model = SingleLayerNeuralModel(X_train, y_train)
model.start_training(iteration=2000, learning_rate=0.05)
model.show_training_results()
preds = model.predict(X_test)


# MULTIPLE NEURONS ON MULTIPLE LAYERS

X_train, y_train = Datasets.get_random_circles_set(n_samples=500)
X_test, y_test = Datasets.get_random_circles_set(n_samples=200)

model = MultiLayerNeuralModel(X_train, y_train, hidden_layers=(16, 16, 16))
model.start_training(iteration=10000, learning_rate=0.05)
model.show_training_results()
preds = model.predict(X_test)
