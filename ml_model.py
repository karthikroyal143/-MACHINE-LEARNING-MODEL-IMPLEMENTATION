# Step 1: Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 2: Load the Iris dataset
iris = load_iris()
X = iris.data        # Features (sepal and petal measurements)
y = iris.target      # Labels (species)

# Step 3: Split dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Create and train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 6: Evaluate the model using accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
