from mne.io import concatenate_raws
import matplotlib.pyplot as plt
from mne import events_from_annotations, Epochs
from mne.datasets import eegbci
import mne
from csp import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import joblib
import os

rename_dict = {'Fc5.': 'FC5', 'Fc3.': 'FC3', 'Fc1.': 'FC1', 'Fcz.': 'FCz', 'Fc2.': 'FC2', 'Fc4.': 'FC4', 'Fc6.': 'FC6', 'C5..': 'C5', 'C3..': 'C3', 'C1..': 'C1', 'Cz..': 'Cz', 'C2..': 'C2', 'C4..': 'C4', 'C6..': 'C6', 'Cp5.': 'CP5', 'Cp3.': 'CP3', 'Cp1.': 'CP1', 'Cpz.': 'CPz', 'Cp2.': 'CP2', 'Cp4.': 'CP4', 'Cp6.': 'CP6', 'Fp1.': 'Fp1', 'Fpz.': 'Fpz', 'Fp2.': 'Fp2', 'Af7.': 'AF7', 'Af3.': 'AF3', 'Afz.': 'AFz', 'Af4.': 'AF4', 'Af8.': 'AF8', 'F7..': 'F7', 'F5..': 'F5', 'F3..': 'F3', 'F1..': 'F1', 'Fz..': 'Fz', 'F2..': 'F2', 'F4..': 'F4', 'F6..': 'F6', 'F8..': 'F8', 'Ft7.': 'FT7', 'Ft8.': 'FT8', 'T7..': 'T7', 'T8..': 'T8', 'T9..': 'T9', 'T10.': 'T10', 'Tp7.': 'TP7', 'Tp8.': 'TP8', 'P7..': 'P7', 'P5..': 'P5', 'P3..': 'P3', 'P1..': 'P1', 'Pz..': 'Pz', 'P2..': 'P2', 'P4..': 'P4', 'P6..': 'P6', 'P8..': 'P8', 'Po7.': 'PO7', 'Po3.': 'PO3', 'Poz.': 'POz', 'Po4.': 'PO4', 'Po8.': 'PO8', 'O1..': 'O1', 'Oz..': 'Oz', 'O2..': 'O2', 'Iz..': 'Iz'}

def train(subject, runs, task, bigtest):
    rows_files = eegbci.load_data(subjects=[subject], runs=runs)
    raws = []
    for row in rows_files:
        raws.append(mne.io.read_raw_edf(row, preload=True))
    raw = concatenate_raws(raws, verbose=False)
    filtered_raw = raw.copy().filter(l_freq=8, h_freq=30, verbose=False)

    filtered_raw.rename_channels(rename_dict)
    filtered_raw.set_montage('standard_1020')
    # if not bigtest:
    #     filtered_raw.plot(block=True)
    #     filtered_raw.compute_psd().plot()

    event_ids = dict(T1=0, T2=1)
    events, _ = events_from_annotations(filtered_raw, event_id=event_ids, verbose=False)
    epochs = Epochs(filtered_raw, events, event_ids, tmin=0.5, tmax=4, baseline=None, verbose=False)
    X = epochs.get_data(verbose=False)
    y = epochs.events[:, -1]

    pipeline = Pipeline(
        [('CSP', CSP()),
        ('LDA', LDA())
    ])

    if not bigtest:
        cv_scores = cross_val_score(pipeline, X, y, cv=5)
        print(cv_scores)
        print(f"cross-validation score: {cv_scores.mean():.2f}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    pipeline.fit(X_train, y_train)

    os.makedirs(f"save/Task{task}_Subject{subject}", exist_ok=True)
    joblib.dump(pipeline, f'save/Task{task}_Subject{subject}/pipeline.pkl')
    joblib.dump(X_test, f'save/Task{task}_Subject{subject}/X_test.pkl')
    joblib.dump(y_test, f'save/Task{task}_Subject{subject}/y_test.pkl')