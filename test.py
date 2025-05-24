from train import train
from predict import predict
import json

def affichage(dictionnaire):
    for key, value in dictionnaire.items():
        for sub_key, sub_value in value.items():
            if sub_key != 'meanTask':
                print(f"experiment {key}: subject {sub_key}: accuracy = {sub_value}")
    print("Mean accuracy of the four different experiments for all 109 subjects:")
    TotalRun = 0
    for key, value in dictionnaire.items():
        for sub_key, sub_value in value.items():
            if sub_key == 'meanTask':
                TotalRun += sub_value
                print(f"experiment {key}: accuracy = {sub_value}")
    print(f"Mean accuracy of 4 experiments: {TotalRun / len(dictionnaire)}")

def bigTest(tasks_runs):
    testAllRun = {}
    for task in tasks_runs:
        resultRun = 0
        testAllRun[f'Task{task}'] = {}
        for i in range (1, 110):
            print(f'subjetc {i} task {task}')
            train(i, tasks_runs[task], task, True)
            result = predict(i, task, True)
            resultRun += result
            testAllRun[f'Task{task}'][f'Subject{i}'] = result
        testAllRun[f'Task{task}']['meanTask'] = resultRun / 109

    with open('testAllRun.json', 'w') as json_file:
        json.dump(testAllRun, json_file, indent=4)
    affichage(testAllRun)

