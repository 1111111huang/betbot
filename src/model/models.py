from sklearn.ensemble import RandomForestClassifier

def get_rf_model(params):
    return RandomForestClassifier(**params)
