import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# --- STAGE 1: LOAD THE DATA ---
try:
    # Points directly to the file in your Language_AI folder
    df = pd.read_csv("Language Detection.csv")
    print("✅ Success! Real dataset loaded.")
    print(f"Loaded {len(df)} rows of data.")
except FileNotFoundError:
    print("❌ Error: Still can't find 'Language Detection.csv' in this folder.")
    exit()

# --- STAGE 2: PREPARE DATA ---
# Using the correct capitalized column names from your file
X = df["Text"]
y = df["Language"] 

# Vectorization: Turning words into numbers
cv = CountVectorizer()
X_vectors = cv.fit_transform(X)

# Split data: 80% to train, 20% to test
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2)

# --- STAGE 3: TRAIN AI ---
model = MultinomialNB() # Optimized for text frequency
print("🧠 Training AI Brain...")
model.fit(X_train, y_train)

# --- STAGE 4: INTERACTIVE TESTING ---
# Calculate the final accuracy score
accuracy = model.score(X_test, y_test)
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")

print("\n--- TEST YOUR AI (type 'quit' to stop) ---")
while True:
    user_input = input("\nEnter a sentence: ")
    if user_input.lower() == 'quit':
        break
    
    # Transform user input
    user_data = cv.transform([user_input]).toarray()
    
    # Calculate probability for each language
    probs = model.predict_proba(user_data)
    max_prob = np.max(probs) # Get the highest confidence score
    
    if max_prob < 0.5: # 50% threshold
        print(f"🤔 Detected Language: Uncertain (Confidence: {max_prob*100:.1f}%)")
        print("Note: This language might not be in the training set.")
    else:
        prediction = model.predict(user_data)
        print(f"🌍 Detected Language: {prediction[0]} (Confidence: {max_prob*100:.1f}%)")