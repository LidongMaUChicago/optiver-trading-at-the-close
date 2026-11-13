"""Run INSIDE a Kaggle notebook after adding trusted code and trained artifact as inputs.

    from optiver_close.inference import run_kaggle
    import joblib, optiver2023
    artifact = joblib.load('/kaggle/input/YOUR-MODEL/model.joblib')
    run_kaggle(artifact, optiver2023.make_env())

Use the same dependency versions as training. Do not load untrusted joblib files.
This module intentionally does not auto-submit, train, or install anything.
"""

