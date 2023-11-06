from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar el conjunto de datos de calidad de vino
wine_quality = fetch_openml(data_id=186, as_frame=True)
X = wine_quality.data
y = wine_quality.target

# Imputar valores faltantes con la media de cada característica
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características (es importante para modelos como Regresión Logística y SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Regresión Logística
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_logistic_pred = logistic_model.predict(X_test)
accuracy_logistic = accuracy_score(y_test, y_logistic_pred)
print("Precisión del modelo de Regresión Logística:", accuracy_logistic)
print(classification_report(y_test, y_logistic_pred))
print(confusion_matrix(y_test, y_logistic_pred))
