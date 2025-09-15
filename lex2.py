# =======================
# Importar librerías necesarias
    # =======================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# =======================
# Generar datos sintéticos más complejos
# =======================
np.random.seed(42)      # Fijar semilla para reproducibilidad
n_samples = 2000        # Número de registros a generar

# =======================
# Crear características simuladas
# =======================
age = np.random.randint(18, 70, n_samples)                         # Edad entre 18 y 70
income = np.random.normal(60, 20, n_samples).astype(int)           # Ingreso promedio con desviación
income = np.clip(income, 20, 150)                                  # Limitar ingreso entre 20k y 150k
credit_score = np.random.normal(650, 100, n_samples).astype(int)   # Puntaje crediticio promedio
credit_score = np.clip(credit_score, 300, 850)                     # Limitar entre 300 y 850
loan_amount = np.random.normal(50, 25, n_samples).astype(int)      # Monto del préstamo
loan_amount = np.clip(loan_amount, 5, 150)                         # Limitar entre 5k y 150k
employment_years = np.random.randint(0, 40, n_samples)             # Años de empleo
debt_to_income_ratio = np.random.normal(35, 15, n_samples)         # Relación deuda/ingreso
debt_to_income_ratio = np.clip(debt_to_income_ratio, 5, 60)        # Limitar entre 5 y 60
existing_loans = np.random.poisson(1.5, n_samples)                 # Número de préstamos existentes
existing_loans = np.clip(existing_loans, 0, 5)                     # Limitar entre 0 y 5
education_level = np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.5, 0.2])  # Nivel educativo
marital_status = np.random.choice([1, 2, 3], n_samples, p=[0.4, 0.4, 0.2])  # Estado civil
has_dependents = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])          # Dependientes

# =======================
# Crear variable objetivo con reglas complejas
# =======================
loan_approved = np.zeros(n_samples)    # Inicializar todas en 0 (denegado)

# Reglas de aprobación de préstamos
for i in range(n_samples):
    # Regla 1: Ingreso alto y ratio bajo
    if income[i] > 80 and debt_to_income_ratio[i] < 30:
        loan_approved[i] = 1
    # Regla 2: Buen crédito y empleo estable
    elif credit_score[i] > 700 and employment_years[i] > 5:
        loan_approved[i] = 1
    # Regla 3: Monto pequeño con pocas deudas
    elif loan_amount[i] < 30 and existing_loans[i] < 2:
        loan_approved[i] = 1
    # Regla 4: Alta educación y deuda controlada
    elif education_level[i] == 3 and debt_to_income_ratio[i] < 40:
        loan_approved[i] = 1
    # Regla 5: Ingreso medio + buen crédito + sin deudas
    elif income[i] > 50 and credit_score[i] > 650 and existing_loans[i] == 0:
        loan_approved[i] = 1

# Añadir algo de ruido (5% de los datos invertidos aleatoriamente)
noise = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
loan_approved = np.logical_xor(loan_approved, noise).astype(int)

# =======================
# Crear DataFrame y guardar dataset
# =======================
data = pd.DataFrame({
    'age': age,
    'income': income,
    'credit_score': credit_score,
    'loan_amount': loan_amount,
    'employment_years': employment_years,
    'debt_to_income_ratio': debt_to_income_ratio,
    'existing_loans': existing_loans,
    'education_level': education_level,
    'marital_status': marital_status,
    'has_dependents': has_dependents,
    'loan_approved': loan_approved
})

# Guardar como CSV
data.to_csv('loan_dataset.csv', index=False)
print("Dataset generado y guardado como 'loan_dataset.csv'")

# =======================
# Cargar y explorar datos
# =======================
df = pd.read_csv('loan_dataset.csv')

print("Primeras filas del dataset:")
print(df.head())                 # Primeras filas
print("\nInformación del dataset:")
print(df.info())                 # Info general
print("\nEstadísticas descriptivas:")
print(df.describe())             # Estadísticas
print("\nDistribución de la variable objetivo:")
print(df['loan_approved'].value_counts())  # Balance de clases

# =======================
# Visualizar correlaciones
# =======================
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Matriz de Correlación')
plt.tight_layout()
plt.show()

# =======================
# Preparar datos para el modelo
# =======================
X = df.drop('loan_approved', axis=1)   # Variables predictoras
y = df['loan_approved']                # Variable objetivo

# =======================
# Dividir en entrenamiento y prueba
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# =======================
# Crear y entrenar modelo de Árbol de Decisión
# =======================
model = DecisionTreeClassifier(
    max_depth=5,            # Profundidad máxima (controlar sobreajuste)
    min_samples_split=20,   # Mínimo de muestras para dividir un nodo
    min_samples_leaf=10,    # Mínimo de muestras en una hoja
    random_state=42
)
model.fit(X_train, y_train)

# =======================
# Evaluar modelo
# =======================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo: {accuracy:.4f}")   # Exactitud del modelo

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))       # Precision, recall, f1-score

# =======================
# Matriz de confusión
# =======================
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicciones')
plt.ylabel('Verdaderos')
plt.show()

# =======================
# Visualizar árbol de decisión
# =======================
plt.figure(figsize=(20, 12))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Denegado', 'Aprobado'],
    filled=True,
    rounded=True,
    proportion=True,
    max_depth=3    # Mostrar solo primeros 3 niveles para claridad
)
plt.title('Árbol de Decisión para Aprobación de Préstamos (Primeros 3 niveles)')
plt.show()

# =======================
# Importancia de características
# =======================
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nImportancia de las características:")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Importancia de las Características')
plt.tight_layout()
plt.show()

# =======================
# Ejemplo de predicción con un nuevo solicitante
# =======================
print("\nEjemplo de predicción con nuevos datos:")
new_loan_applicant = pd.DataFrame({
    'age': [35],
    'income': [75],
    'credit_score': [720],
    'loan_amount': [45],
    'employment_years': [8],
    'debt_to_income_ratio': [25],
    'existing_loans': [1],
    'education_level': [2],
    'marital_status': [2],
    'has_dependents': [1]
})

prediction = model.predict(new_loan_applicant)               # Predicción (0 = denegado, 1 = aprobado)
prediction_proba = model.predict_proba(new_loan_applicant)   # Probabilidades asociadas

print(f"Datos del solicitante: {new_loan_applicant.to_dict('records')[0]}")
print(f"Préstamo aprobado: {'Sí' if prediction[0] == 1 else 'No'}")
print(f"Probabilidades: Denegado: {prediction_proba[0][0]:.4f}, Aprobado: {prediction_proba[0][1]:.4f}")
