import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/Admission_Predict.csv")

# Features and target
X = df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']]
y = df['Chance of Admit ']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Score
print("Model R² score:", model.score(X_test, y_test))





# Predict for a sample student
sample_student = [[327, 105, 3, 4, 4, 8.0, 0]]
predicted_chance = model.predict(sample_student)[0]
print(f"Predicted Chance of Admit: {predicted_chance:.2f}")
