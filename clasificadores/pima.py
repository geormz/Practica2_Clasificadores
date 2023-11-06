import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Cargar los datos desde un archivo CSV (asegúrate de que el archivo esté en la misma carpeta o proporciona la ruta completa)
data = pd.read_csv('pima.csv')

# Dividir los datos en características (X) y etiquetas (y)
X = data.drop('Outcome', axis=1)  # Excluir la columna 'Outcome' de las características
y = data['Outcome']  # 'Outcome' es la columna que contiene las etiquetas

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo de clasificación con Random Forest
clf_rf = RandomForestClassifier(random_state=42)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Precisión del modelo Random Forest:", accuracy_rf)

# Entrenar un modelo de clasificación con K-Vecinos Cercanos (K-Nearest Neighbors)
clf_knn = KNeighborsClassifier()
clf_knn.fit(X_train, y_train)
y_pred_knn = clf_knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("Precisión del modelo K-Vecinos Cercanos:", accuracy_knn)

# Entrenar un modelo de clasificación con Máquinas de Soporte Vectorial (Support Vector Machines)
clf_svm = SVC(random_state=42)
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Precisión del modelo Máquinas de Soporte Vectorial:", accuracy_svm)

# Entrenar un modelo de clasificación con Naive Bayes
clf_nb = GaussianNB()
clf_nb.fit(X_train, y_train)
y_pred_nb = clf_nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Precisión del modelo Naive Bayes:", accuracy_nb)
