import sys
from train import train
from predict import predict
from test import bigTest

tasks_runs = {'1': [3, 7, 11], '2': [4, 8, 12], '3': [5, 9, 13], 4: [6, 10, 14]}

def main():
    try:
        if len(sys.argv) == 4:
            task = sys.argv[1]
            if task not in tasks_runs:
                raise ValueError("task must be a number between 1 and 4 (inclusive)")
            intArgSubject = int(sys.argv[1])
            if intArgSubject < 1 or intArgSubject > 109:
                raise ValueError("il y a seulement 109 sujet (vous devez mettre l'arg1 entre 1 et 109)")
            if sys.argv[3] != 'train' and sys.argv[3] != 'predict':
                raise ValueError("vous devez mettre 'predict' ou 'train' comme 3eme argument")
            if sys.argv[3] == 'train':
                train([intArgSubject], tasks_runs[task], False)
            if sys.argv[3] == 'predict':
                print(f"Accuracy: {predict(False)}")
        elif len(sys.argv) == 1:
            bigTest(tasks_runs)
        else:
            raise ValueError("vous devez mettre 0 ou 3 arguments")
    except Exception as e:
        print(f"error: {e}")
        return
    
if __name__ == "__main__":
    main()
