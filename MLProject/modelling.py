import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import argparse
from mlflow.models import infer_signature

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tuning', action='store_true', default=False)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--n_iter', type=int, default=5)
    args = parser.parse_args()
    
    df = pd.read_csv('data_preprocessed/diamonds_clean.csv')
    
    X = df.drop(['price', 'price_log'], axis=1)
    y = df['price_log']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    
    if args.tuning:
        param_dist = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        model = RandomForestRegressor(random_state=args.random_state, n_jobs=-1)
        random_search = RandomizedSearchCV(
            model, param_dist, n_iter=args.n_iter, cv=5,
            scoring='r2', random_state=args.random_state, n_jobs=-1
        )
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        cv_r2 = random_search.best_score_
    else:
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        best_model = model
        best_params = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth
        }
        cv_r2 = None
    
    y_pred_test_log = best_model.predict(X_test)
    y_test_original = np.expm1(y_test)
    y_pred_test_original = np.expm1(y_pred_test_log)
    
    test_r2 = r2_score(y_test_original, y_pred_test_original)
    test_rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_test_original))
    test_mae = mean_absolute_error(y_test_original, y_pred_test_original)
    
    signature = infer_signature(X_train, best_model.predict(X_train))
    
    with mlflow.start_run(run_name="CI_Run"):
        mlflow.log_params(best_params)
        mlflow.log_params({
            'test_size': args.test_size,
            'random_state': args.random_state,
            'tuning': args.tuning
        })
        
        mlflow.log_metrics({
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae
        })
        
        if cv_r2 is not None:
            mlflow.log_metric('best_cv_r2', cv_r2)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(y_test_original, y_pred_test_original, alpha=0.3, s=5)
        ax.plot([y_test_original.min(), y_test_original.max()], 
                [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Price (USD)')
        ax.set_ylabel('Predicted Price (USD)')
        ax.set_title(f'Random Forest - Actual vs Predicted\nR2: {test_r2:.4f}, RMSE: ${test_rmse:,.0f}')
        plt.tight_layout()
        plt.savefig('actual_vs_predicted.png')
        mlflow.log_artifact('actual_vs_predicted.png')
        plt.close()
        os.remove('actual_vs_predicted.png')
        
        estimator_info = f"""
        <html>
        <body>
        <h2>Random Forest Regressor - Diamond Price Prediction</h2>
        <h3>Best Parameters:</h3>
        <ul>
        """
        for key, value in best_params.items():
            estimator_info += f"<li>{key}: {value}</li>"
        
        estimator_info += f"""
        </ul>
        <h3>Performance Metrics:</h3>
        <ul>
        <li>Test R2: {test_r2:.4f}</li>
        <li>Test RMSE: ${test_rmse:,.2f}</li>
        <li>Test MAE: ${test_mae:,.2f}</li>
        </ul>
        </body>
        </html>
        """
        
        with open('estimator.html', 'w') as f:
            f.write(estimator_info)
        mlflow.log_artifact('estimator.html')
        os.remove('estimator.html')
        
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.iloc[:5]
        )

        # Simpan run_id ke file
        run_id = mlflow.active_run().info.run_id
        with open('run_id.txt', 'w') as f:
            f.write(run_id)
        print(f"Run ID saved: {run_id}")
    
        print(f"Test R2: {test_r2:.4f}")
        print(f"Test RMSE: ${test_rmse:,.2f}")

if __name__ == "__main__":
    main()
