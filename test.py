from train import train
from predict import predict
import json

def affichage(dictionnaire):
    for key, value in dictionnaire.items():
        for sub_key, sub_value in value.items():
            if sub_key != 'meanRun':
                print(f"experiment {key}: subject {sub_key}: accuracy = {sub_value}")
    print("Mean accuracy of the four different experiments for all 109 subjects:")
    TotalRun = 0
    for key, value in dictionnaire.items():
        for sub_key, sub_value in value.items():
            if sub_key == 'meanRun':
                TotalRun += sub_value
                print(f"experiment {key}: accuracy = {sub_value}")
    print(f"Mean accuracy of 4 experiments: {TotalRun / len(dictionnaire)}")

def bigTest(tasks_runs):
    testAllRun = {}
    for run in tasks_runs:
        resultRun = 0.0
        testAllRun[f'Run{run}'] = {}
        for i in range (1, 110):
            print(f'subjetc {i} run {run}')
            train([i], tasks_runs[run], True)
            result = predict(True)
            resultRun += result
            testAllRun[f'Run{run}'][f'Subject{i}'] = result
        testAllRun[f'Run{run}']['meanRun'] = resultRun / 109

    with open('testAllRun.json', 'w') as json_file:
        json.dump(testAllRun, json_file, indent=4)
    affichage(testAllRun)

