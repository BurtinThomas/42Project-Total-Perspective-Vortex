import joblib
import numpy as np

def predict(subject, task, bigtest):
    pipeline = joblib.load(f'save/Task{task}_Subject{subject}/pipeline.pkl')
    X_test = joblib.load(f'save/Task{task}_Subject{subject}/X_test.pkl')
    y_test = joblib.load(f'save/Task{task}_Subject{subject}/y_test.pkl')
    results = []
    for n in range(X_test.shape[0]):
        prediction = pipeline.predict(X_test[n:n + 1, :, :])[0]
        truth = y_test[n:n + 1][0]
        result = prediction == truth
        if not bigtest:
            print(f"epoch: {n}, prediction: {prediction}, truth: {truth}, equal?: {result}")
        results.append(result)
    return np.mean(results)