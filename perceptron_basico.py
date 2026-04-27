import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Inicializar los pesos y el sesgo a 0
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Entrenamiento
        for epoch in range(self.epochs):
            for i in range(n_samples):
                # Calcular la salida (combinación lineal)
                linear_output = np.dot(X[i], self.weights) + self.bias
                # Aplicar la función de activación (Escalón)
                y_predicted = self._activation_function(linear_output)

                # Regla de actualización de pesos
                update = self.learning_rate * (y[i] - y_predicted)
                self.weights += update * X[i]
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self._activation_function(linear_output)
        return y_predicted

    def _activation_function(self, x):
        # Función de activación de salto escalón (Step function)
        return np.where(x >= 0, 1, 0)

if __name__ == "__main__":
    print("--- Entrenando un Perceptrón Básico para resolver una puerta lógica OR ---")
    
    # Datos de entrenamiento para una puerta OR
    # X son las entradas, y son las salidas esperadas
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 1, 1, 1])

    # Inicializar y entrenar el modelo
    perceptron = Perceptron(learning_rate=0.1, epochs=10)
    perceptron.fit(X, y)

    print("\nEntrenamiento finalizado. Pesos aprendidos:", perceptron.weights)
    print("Sesgo aprendido:", perceptron.bias)

    # Probando el modelo
    print("\n--- Predicciones ---")
    predictions = perceptron.predict(X)
    for i in range(len(X)):
        print(f"Entrada: {X[i]} -> Predicción: {predictions[i]} | Esperado: {y[i]}")
