import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

def main():
    """
    Main function to run the complete cybersecurity analysis.
    """
    # --- Setup ---
    # Create output directory if it doesn't exist
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the dataset from the 'data' folder
    try:
        df = pd.read_csv('data/Cybersecurity Intrusion Detection.csv')
    except FileNotFoundError:
        print("Error: 'data/Cybersecurity Intrusion Detection.csv' not found.")
        print("Please ensure the CSV file is in the 'data' directory.")
        return

    print("Dataset loaded successfully. Starting analysis...")

    # --- Figure 1: Distribution of the Target Variable ---
    plt.figure(figsize=(8, 6))
    sns.countplot(x='attack_detected', data=df)
    plt.title('Figure 1: Distribution of the Target Variable (attack_detected)')
    plt.xlabel('Attack Detected (0: Normal, 1: Attack)')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Normal', 'Attack'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure1_attack_distribution.png'))
    plt.clf()
    print("Generated Figure 1: Attack Distribution")

    # --- Figure 2: Correlation Matrix of Numerical Features ---
    numerical_features_corr = df.select_dtypes(include=np.number).columns.tolist()
    numerical_features_corr.remove('attack_detected')
    df_numerical = df[numerical_features_corr]

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_numerical.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Figure 2: Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure2_correlation_matrix.png'))
    plt.clf()
    print("Generated Figure 2: Correlation Matrix")

    # --- Figure 3: Feature Distributions by Attack Class ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Figure 3: Feature Distributions by Attack Class', fontsize=16)
    # Box plot for failed_logins
    sns.boxplot(ax=axes[0], x='attack_detected', y='failed_logins', data=df)
    axes[0].set_title('Distribution of Failed Logins')
    axes[0].set_xticklabels(['Normal', 'Attack'])
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Number of Failed Logins')
    # Box plot for ip_reputation_score
    sns.boxplot(ax=axes[1], x='attack_detected', y='ip_reputation_score', data=df)
    axes[1].set_title('Distribution of IP Reputation Score')
    axes[1].set_xticklabels(['Normal', 'Attack'])
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('IP Reputation Score')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'figure3_feature_distributions.png'))
    plt.clf()
    print("Generated Figure 3: Feature Distributions")

    # --- Model Training and Evaluation ---
    X = df.drop('attack_detected', axis=1)
    y = df['attack_detected']
    
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)}
    
    predictions, probas, trained_models = {}, {}, {}
    print("\nTraining models...")
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        predictions[name] = pipeline.predict(X_test)
        probas[name] = pipeline.predict_proba(X_test)[:, 1]
        print(f"  - {name} trained.")

    # --- Figure 4: Confusion Matrices ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle('Figure 4: Confusion Matrices for All Classifiers', fontsize=16)
    for i, (name, y_pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(name)
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'figure4_confusion_matrices.png'))
    plt.clf()
    print("Generated Figure 4: Confusion Matrices")

    # --- Figure 5: Random Forest Feature Importance ---
    rf_pipeline = trained_models['Random Forest']
    ohe_feature_names = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = np.concatenate([numerical_features, ohe_feature_names])
    importances = rf_pipeline.named_steps['classifier'].feature_importances_
    
    feature_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(10)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title('Figure 5: Random Forest Model Feature Importance (Top 10)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure5_feature_importance.png'))
    plt.clf()
    print("Generated Figure 5: Feature Importance")

    # --- Figure 6: ROC Curves ---
    plt.figure(figsize=(10, 8))
    for name, y_prob in probas.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Figure 6: Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure6_roc_curves.png'))
    plt.clf()
    print("Generated Figure 6: ROC Curves")

    print("\nAnalysis complete. All figures saved in the 'output' directory.")
    plt.close('all') # Close all plot figures

if __name__ == '__main__':
    main()