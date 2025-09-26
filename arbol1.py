import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder 
import seaborn as sns 

# ==========================
# Generar datos sintéticos más complejos
# ==========================
np.random.seed(42) 
n_samples = 2000

# Crear características 
age = np.random.randint(18, 70, n_samples) 
income = np.random.normal(60, 20, n_samples).astype(int) 
income = np.clip(income, 20, 150)  # Limitar entre 20k y 150k 
credit_score = np.random.normal(650, 100, n_samples).astype(int) 
credit_score = np.clip(credit_score, 300, 850) 
loan_amount = np.random.normal(50, 25, n_samples).astype(int) 
loan_amount = np.clip(loan_amount, 5, 150) 
employment_years = np.random.randint(0, 40, n_samples) 
debt_to_income_ratio = np.random.normal(35, 15, n_samples) 
debt_to_income_ratio = np.clip(debt_to_income_ratio, 5, 60) 
existing_loans = np.random.poisson(1.5, n_samples) 
existing_loans = np.clip(existing_loans, 0, 5) 
education_level = np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.5, 0.2]) 
marital_status = np.random.choice([1, 2, 3], n_samples, p=[0.4, 0.4, 0.2]) 
has_dependents = np.random.choice([0, 1], n_samples, p=[0.6, 0.4]) 

# NUEVO 1: Historial de pagos anteriores
late_payments = np.random.poisson(1.0, n_samples)
late_payments = np.clip(late_payments, 0, 10)

# NUEVO 2: Tipo de empleo / contrato laboral
# 1 = permanente, 2 = temporal, 3 = independiente, 4 = desempleado
employment_type = np.random.choice([1, 2, 3, 4], n_samples, p=[0.4, 0.3, 0.2, 0.1])

# NUEVO 3: Propiedades o activos
owns_assets = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])

# NUEVO 4: Monto total de deudas activas
total_debt = np.clip(np.random.normal(40, 20, n_samples).astype(int), 0, 200)

# ==========================
# Crear variable objetivo con reglas complejas
# ==========================
loan_approved = np.zeros(n_samples) 

# Reglas complejas para aprobación 
for i in range(n_samples): 
    # Regla 1: Ingreso alto y ratio bajo 
    if income[i] > 80 and debt_to_income_ratio[i] < 30: 
        loan_approved[i] = 1 
     
    # Regla 2: Buen crédito y empleo estable 
    elif credit_score[i] > 700 and employment_years[i] > 5: 
        loan_approved[i] = 1 
         
    # Regla 3: Monto pequeño con garantías 
    elif loan_amount[i] < 30 and existing_loans[i] < 2: 
        loan_approved[i] = 1 
         
    # Regla 4: Educación avanzada compensa otros factores 
    elif education_level[i] == 3 and debt_to_income_ratio[i] < 40: 
        loan_approved[i] = 1 
         
    # Regla 5: Ingreso medio con buen historial crediticio 
    elif income[i] > 50 and credit_score[i] > 650 and existing_loans[i] == 0: 
        loan_approved[i] = 1 

    # 🔹 NUEVAS reglas usando las variables añadidas
    elif late_payments[i] == 0 and credit_score[i] > 600:
        loan_approved[i] = 1
    elif employment_type[i] == 1 and total_debt[i] < 80:
        loan_approved[i] = 1
    elif owns_assets[i] == 1 and loan_amount[i] < 60:
        loan_approved[i] = 1

# Añadir algo de ruido 
noise = np.random.choice([0, 1], n_samples, p=[0.95, 0.05]) 
loan_approved = np.logical_xor(loan_approved, noise).astype(int) 

# ==========================
# Crear DataFrame
# ==========================
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
    # Nuevas columnas
    'late_payments': late_payments,
    'employment_type': employment_type,
    'owns_assets': owns_assets,
    'total_debt': total_debt,
    'loan_approved': loan_approved 
}) 

# Guardar como CSV 
data.to_csv('loan_dataset.csv', index=False) 
print("Dataset generado y guardado como 'loan_dataset.csv'") 

# ==========================
# Cargar datos
# ==========================
df = pd.read_csv('loan_dataset.csv') 

# Exploración inicial 
print("Primeras filas del dataset:") 
print(df.head()) 
print("\nInformación del dataset:") 
print(df.info()) 
print("\nEstadísticas descriptivas:") 
print(df.describe()) 
print("\nDistribución de la variable objetivo:") 
print(df['loan_approved'].value_counts()) 

# ==========================
# Visualizar correlaciones 
# ==========================
plt.figure(figsize=(12, 8)) 
correlation_matrix = df.corr() 
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0) 
plt.title('Matriz de Correlación') 
plt.tight_layout() 
plt.show() 

# ==========================
# Preparar datos para el modelo 
# ==========================
X = df.drop('loan_approved', axis=1) 
y = df['loan_approved']

# Dividir en conjuntos de entrenamiento y prueba 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) 

# Crear y entrenar el modelo de árbol de decisión 
model = DecisionTreeClassifier( 
    max_depth=6,           # ajustado un poco para más variables
    min_samples_split=20,  
    min_samples_leaf=10,   
    random_state=42 
) 
model.fit(X_train, y_train) 

# ==========================
# Evaluar el modelo 
# ==========================
y_pred = model.predict(X_test) 
accuracy = accuracy_score(y_test, y_pred) 
print(f"\nPrecisión del modelo: {accuracy:.4f}") 

print("\nReporte de clasificación:") 
print(classification_report(y_test, y_pred)) 

# Matriz de confusión 
plt.figure(figsize=(8, 6)) 
cm = confusion_matrix(y_test, y_pred) 
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues') 
plt.title('Matriz de Confusión') 
plt.ylabel('Verdaderos') 
plt.xlabel('Predicciones') 
plt.show() 

# ==========================
# Visualizar el árbol de decisión 
# ==========================
plt.figure(figsize=(20, 12)) 
plot_tree( 
    model,  
    feature_names=X.columns,  
    class_names=['Denegado', 'Aprobado'],  
    filled=True,  
    rounded=True, 
    proportion=True, 
    max_depth=3  
) 
plt.title('Árbol de Decisión para Aprobación de Préstamos (Primeros 3 niveles)') 
plt.show()

# ==========================
# Importancia de características 
# ==========================
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

# ==========================
# Ejemplo de predicción con nuevos datos 
# ==========================
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
    'has_dependents': [1], 
    # 🔹 Nuevos datos
    'late_payments': [0],
    'employment_type': [1],
    'owns_assets': [1],
    'total_debt': [60]
}) 

prediction = model.predict(new_loan_applicant) 
prediction_proba = model.predict_proba(new_loan_applicant) 

print(f"Datos del solicitante: {new_loan_applicant.to_dict('records')[0]}") 
print(f"Préstamo aprobado: {'Sí' if prediction[0] == 1 else 'No'}") 
print(f"Probabilidades: [Denegado: {prediction_proba[0][0]:.4f}, Aprobado: {prediction_proba[0][1]:.4f}]") 
