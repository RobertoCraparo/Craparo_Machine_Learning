import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Caricamento del dataset
# Assicurati di avere il file 'Iris.csv' nella stessa cartella dello script
df = pd.read_csv('Iris.csv')

# 2. Preparazione dei dati
# Selezioniamo le misurazioni di sepali e petali come feature (X)
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
# Selezioniamo il nome della specie come variabile da prevedere (y)
y = df['Species']

# 3. Suddivisione in Training e Test set
# Usiamo l'80% dei dati per addestrare il modello e il 20% per testarlo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Standardizzazione (Scaling) dei dati
# Il k-NN calcola le distanze, quindi è fondamentale che tutte le misurazioni abbiano la stessa scala
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Inizializzazione e Addestramento del modello k-NN
# Scegliamo k=3 (guarderà le 3 piante più simili per prendere una decisione)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# 6. Previsione e Valutazione
y_pred = knn.predict(X_test_scaled)

print(f"Accuratezza del modello: {accuracy_score(y_test, y_pred):.2f}\n")
print("Report di Classificazione:")
print(classification_report(y_test, y_pred))


# 7. GENERAZIONE E SALVATAGGIO DELLA MATRICE COLORATA


# Calcola la matrice numerica incrociando i valori reali e le previsioni
cm = confusion_matrix(y_test, y_pred)

# Crea l'oggetto grafico associando i numeri alle etichette delle specie
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)

# Disegna la matrice
disp.plot(cmap='Reds')

plt.title("Matrice di Confusione - k-NN")

plt.savefig('matrice_confusione.png', dpi=300, bbox_inches='tight')

plt.close()

print("\nMatrice di confusione generata con successo.")