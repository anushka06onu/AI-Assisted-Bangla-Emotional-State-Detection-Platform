import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Import the custom Bangla preprocessing function
from utils.preprocessing import preprocess_bangla_text

def run_training_pipeline():
    """
    Main execution function to train, evaluate, and save the Bangla Emotion Classification model.
    """
    print("="*60)
    print("Starting Bangla Emotion AI Training Pipeline")
    print("="*60)
    
    # Define file paths
    dataset_path = 'data/raw/emotion_data.csv'
    models_dir = 'models'
    
    # 1. Create directories if they do not exist
    os.makedirs(models_dir, exist_ok=True)
    print(f"Verified directory: '{models_dir}/' exists.")
    
    # 2. Load the dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please create it first!")
        
    print(f"Loading raw dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} rows. Emotion label distribution:")
    print(df['label'].value_counts())
    print("-" * 40)
    
    # 3. Clean the text using our Bangla-compatible preprocessor
    print("Preprocessing Bangla text (removing emojis, punctuations, and extra whitespace)...")
    df['clean_text'] = df['text'].apply(preprocess_bangla_text)
    
    # Inspect a few cleaned rows for validation
    print("Sample of preprocessed data:")
    for idx, row in df.head(3).iterrows():
        print(f"  Raw:   {repr(row['text'])}")
        print(f"  Clean: {repr(row['clean_text'])}")
    print("-" * 40)
    
    # 4. Split data into Train and Test sets
    X = df['clean_text']
    y = df['label']
    
    # Use standard 80/20 train-test split. Stratify=y ensures balanced class distributions in splits.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print("-" * 40)
    
    # 5. Extract features using TF-IDF Vectorizer
    # We use word n-grams (1, 2) to capture local contextual pairs of Bangla words.
    print("Fitting TF-IDF Vectorizer on training data...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 6. Initialize and Train the Calibrated LinearSVC Model
    # Since LinearSVC does not support probability predictions directly,
    # we wrap it with CalibratedClassifierCV. This allows us to display confidence scores
    # and construct probability visualisations in the Streamlit frontend.
    print("Training Support Vector Machine (Calibrated LinearSVC)...")
    base_svc = LinearSVC(class_weight='balanced', random_state=42, max_iter=2000)
    calibrated_svc = CalibratedClassifierCV(estimator=base_svc, cv=3)
    calibrated_svc.fit(X_train_tfidf, y_train)
    print("Model training complete.")
    print("-" * 40)
    
    # 7. Evaluate the Model
    print("Evaluating model performance on the test set...")
    y_pred = calibrated_svc.predict(X_test_tfidf)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report (Precision, Recall, F1-Score):")
    print(classification_report(y_test, y_pred))
    
    # 8. Generate and Plot the Confusion Matrix
    print("Generating confusion matrix plot...")
    labels_order = ['happy', 'sad', 'angry', 'fear', 'neutral']
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    
    # Create matplotlib plot
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_order)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Bangla Emotion Classifier - Confusion Matrix")
    
    # Save the plot
    plot_path = os.path.join(models_dir, 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved successfully to: '{plot_path}'")
    print("-" * 40)
    
    # 9. Save Trained Models using joblib
    vectorizer_save_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
    model_save_path = os.path.join(models_dir, 'emotion_model.joblib')
    
    print(f"Saving TF-IDF Vectorizer to: '{vectorizer_save_path}'")
    joblib.dump(vectorizer, vectorizer_save_path)
    
    print(f"Saving Calibrated LinearSVC Model to: '{model_save_path}'")
    joblib.dump(calibrated_svc, model_save_path)
    
    print("\n" + "="*60)
    print("Training Pipeline Execution Finished Successfully!")
    print("="*60)

if __name__ == "__main__":
    run_training_pipeline()
