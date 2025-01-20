# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the cleaned dataset
data = pd.read_csv("Alphabet.csv")

# Step 2: Feature Engineering
data['MA5'] = data['Price'].rolling(window=5).mean()  # 5-day moving average
data['MA10'] = data['Price'].rolling(window=10).mean()  # 10-day moving average
data['Price_Lag1'] = data['Price'].shift(1)  # Lagged price
data['Volatility'] = data['High'] - data['Low']  # Daily volatility
data['daily_sentiment'] = data.groupby('Date')['title_sentiment'].transform('mean')  # Daily sentiment average

# Drop NaN rows caused by rolling/lags
data = data.dropna()

# Step 3: Feature Selection (Exclude 'Change %')
features = ['title_sentiment', 'description_sentiment', 'Vol.', 'MA5', 'MA10', 'Price_Lag1', 'Volatility', 'daily_sentiment']
X = data[features]
y = (data['Change %'] > 0).astype(int)  # Binary target: 1 = price up, 0 = price down

# Step 4: Balance Classes with SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Step 6: Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Train the Model (Random Forest)
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Step 8: Evaluate the Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 9: Perform Cross-Validation
cv_scores = cross_val_score(model, X_balanced, y_balanced, cv=5, scoring='accuracy')
print("\nCross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", cv_scores.mean())

# Step 10: Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Step 11: Plot Feature Importance
importances = model.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Step 12: Plot Actual vs Predicted Labels
plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:50], label='Actual', marker='o', linestyle='')  # First 50 samples
plt.plot(y_pred[:50], label='Predicted', marker='x', linestyle='-')
plt.title('Actual vs Predicted Labels')
plt.xlabel('Sample Index')
plt.ylabel('Label (0 = Down, 1 = Up)')
plt.legend()
plt.show()
