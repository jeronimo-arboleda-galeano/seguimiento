import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# Generar datos sintéticos para becas
# ==========================
np.random.seed(42)
n_samples = 2000

# Categoría 1: Académicos
gpa = np.round(np.random.normal(3.2, 0.5, n_samples), 2)          # Promedio (0-5)
credits_completed = np.random.randint(10, 140, n_samples)         # Créditos aprobados

# Categoría 2: Socioeconómicos
family_income = np.random.normal(2, 1, n_samples).astype(int)     # Ingresos en salarios mínimos
family_income = np.clip(family_income, 0, 10)
socioeconomic_strata = np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.25,0.3,0.2,0.15,0.07,0.03])

# Categoría 3: Personales y sociales
belongs_to_group = np.random.choice([0,1], n_samples, p=[0.7,0.3])   # 1 = grupo étnico/víctima/discapacidad
special_condition = np.random.choice([0,1], n_samples, p=[0.8,0.2])  # Otras condiciones especiales

# Categoría 4: Compromisos con la universidad
full_time = np.random.choice([0,1], n_samples, p=[0.2,0.8])          # Estudiante tiempo completo
community_service = np.random.choice([0,1], n_samples, p=[0.7,0.3])  # Se compromete con servicio social

# Categoría 5: Historial educativo
lasallista_prev = np.random.choice([0,1], n_samples, p=[0.7,0.3])    # 1 = sí, 0 = no

# ==========================
# Crear variable objetivo con reglas
# ==========================
scholarship_awarded = np.zeros(n_samples)

for i in range(n_samples):
    # Regla 1: Excelente rendimiento académico
    if gpa[i] >= 4.0 and credits_completed[i] > 60:
        scholarship_awarded[i] = 1
    # Regla 2: Bajos ingresos y estrato bajo
    elif family_income[i] <= 2 and socioeconomic_strata[i] <= 2:
        scholarship_awarded[i] = 1
    # Regla 3: Pertenencia a grupos vulnerables con gpa aceptable
    elif belongs_to_group[i] == 1 and gpa[i] >= 3.0:
        scholarship_awarded[i] = 1
    # Regla 4: Estudiante comprometido
    elif full_time[i] == 1 and community_service[i] == 1 and gpa[i] >= 3.2:
        scholarship_awarded[i] = 1
    # Regla 5: Condición especial con ingresos bajos
    elif special_condition[i] == 1 and family_income[i] <= 3:
        scholarship_awarded[i] = 1
    # Regla 6: Exalumno de institución lasallista con buen promedio
    elif lasallista_prev[i] == 1 and gpa[i] >= 3.0:
        scholarship_awarded[i] = 1

# Añadir ruido
noise = np.random.choice([0,1], n_samples, p=[0.95,0.05])
scholarship_awarded = np.logical_xor(scholarship_awarded, noise).astype(int)

# ==========================
# Crear DataFrame
# ==========================
data = pd.DataFrame({
    'gpa': gpa,
    'credits_completed': credits_completed,
    'family_income': family_income,
    'socioeconomic_strata': socioeconomic_strata,
    'belongs_to_group': belongs_to_group,
    'special_condition': special_condition,
    'full_time': full_time,
    'community_service': community_service,
    'lasallista_prev': lasallista_prev,
    'scholarship_awarded': scholarship_awarded
})

data.to_csv("scholarship_dataset.csv", index=False)
print("Dataset generado y guardado como 'scholarship_dataset.csv'")

# ==========================
# Cargar y explorar
# ==========================
df = pd.read_csv("scholarship_dataset.csv")
print(df.head())
print(df.info())
print(df['scholarship_awarded'].value_counts())

# ==========================
# Visualizar correlaciones
# ==========================
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Matriz de Correlación - Factores de Beca")
plt.show()

# ==========================
# Preparar datos para el modelo
# ==========================
X = df.drop('scholarship_awarded', axis=1)
y = df['scholarship_awarded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)
model.fit(X_train, y_train)

# ==========================
# Evaluar modelo
# ==========================
y_pred = model.predict(X_test)
print("Precisión:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Matriz de Confusión")
plt.show()

# ==========================
# Visualizar árbol
# ==========================
plt.figure(figsize=(18,10))
plot_tree(model, feature_names=X.columns, class_names=["No beca","Beca"], filled=True, max_depth=3)
plt.title("Árbol de Decisión para Becas (primeros 3 niveles)")
plt.show()

# ==========================
# Importancia de características
# ==========================
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values("importance", ascending=False)
print(feature_importance)

sns.barplot(x="importance", y="feature", data=feature_importance)
plt.title("Importancia de Factores en la Beca")
plt.show()

# ==========================
# Ejemplo de predicción
# ==========================
new_applicant = pd.DataFrame({
    'gpa': [3.5],
    'credits_completed': [70],
    'family_income': [3],
    'socioeconomic_strata': [2],
    'belongs_to_group': [0],
    'special_condition': [0],
    'full_time': [1],
    'community_service': [1],
    'lasallista_prev': [1]
})

pred = model.predict(new_applicant)
proba = model.predict_proba(new_applicant)

print("\nSolicitante:", new_applicant.to_dict("records")[0])
print(f"Beca otorgada: {'Sí' if pred[0]==1 else 'No'}")
print(f"Probabilidades: [No: {proba[0][0]:.4f}, Sí: {proba[0][1]:.4f}]")
