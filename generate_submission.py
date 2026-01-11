import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# Attempt to import XGBoost and LightGBM
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
    print("XGBoost not found, skipping...")

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None
    print("LightGBM not found, skipping...")

def main():
    print("Loading data...")
    # Load data
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    # Sample for performance (adjust as needed)
    SAMPLE_SIZE = 50000
    if len(train) > SAMPLE_SIZE:
        print(f"Sampling {len(train)} -> {SAMPLE_SIZE} rows...")
        train_sample = train.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    else:
        train_sample = train

    # Feature definition
    num_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    cat_features = ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']
    target = 'exam_score'

    X = train_sample.drop(columns=[target, 'id'], errors='ignore')
    y = train_sample[target]

    # Preprocessing
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
    ])

    # Models definition
    models = {
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    }
    
    if XGBRegressor:
        models['XGBoost'] = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    
    if LGBMRegressor:
        models['LightGBM'] = LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)

    print("Training models...")
    best_score = -float('inf')
    best_model = None
    best_name = ""

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    for name, model in models.items():
        print(f"Training {name}...")
        pipe = Pipeline([('prep', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        
        y_pred = pipe.predict(X_val)
        score = r2_score(y_val, y_pred)
        print(f"  {name} R2: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = pipe
            best_name = name

    print(f"Best model: {best_name} with R2: {best_score:.4f}")

    # Refit best model on full sample (optional, but here we use the trained one)
    # Ideally refit on X, y
    print("Refitting best model on full sample...")
    best_model.fit(X, y)

    print("Generating submission...")
    # Prepare test data
    # Ensure test has same columns (drop id if present in X definition, wait X definition above dropped id)
    X_test = test.drop(columns=['id'], errors='ignore')
    
    predictions = best_model.predict(X_test)
    
    # Create submission file
    submission = pd.DataFrame({
        'id': test['id'],
        'exam_score': predictions.clip(0, 100).round(2)
    })
    
    submission_path = "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(f"Submission shape: {submission.shape}")

    # Also save the model
    joblib.dump(best_model, "model.pkl")
    print("Model saved to model.pkl")

if __name__ == "__main__":
    main()
