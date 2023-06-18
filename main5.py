import tkinter as tk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pandastable import Table, TableModel
import joblib
from tkinter import filedialog

model = None
model_loaded = False

def train_model():
    try:
        data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
        data_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                        'slope', 'ca',
                        'thal', 'target']
        df = pd.read_csv(data_url, header=None, names=data_columns, na_values='?')

        df = df.dropna()

        X = df.drop('target', axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        global model
        model = LogisticRegression(solver='saga', max_iter=10000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_label.config(text=f"Dokładność modelu: {accuracy}")

    except ValueError:
        accuracy_label.config(text="")


def evaluate_model():
    try:
        global model
        if model is None:
            raise ValueError("Najpierw trenuj model")

        data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
        data_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                        'slope', 'ca',
                        'thal', 'target']
        df = pd.read_csv(data_url, header=None, names=data_columns, na_values='?')
        df = df.dropna()

        X = df.drop('target', axis=1)
        y = df['target']
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        accuracy_label.config(text=f"Dokładność modelu (na pełnym zbiorze danych): {accuracy}")


    except ValueError as e:
        accuracy_label.config(text=str(e))


def rebuild_model():
    global model, model_loaded
    model = None
    model_loaded = False
    train_model()


def save_model():
    global model, model_loaded
    if model is not None:
        filename = filedialog.asksaveasfilename(defaultextension=".joblib")
        joblib.dump(model, filename)
        model_loaded = True


def load_model():
    global model, model_loaded
    filename = filedialog.askopenfilename(filetypes=[("Joblib files", "*.joblib")])
    model = joblib.load(filename)
    model_loaded = True
    accuracy_label.config(text="Model wczytany i przeszkolony")
    display_input_fields()


def rebuild_model():
    global model
    model = None
    train_model()
    accuracy_label.config(text=f"Model zostal zaladowany ponownie")


import matplotlib.pyplot as plt


def visualize_data():
    try:

        data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
        data_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                        'slope', 'ca',
                        'thal', 'target']
        df = pd.read_csv(data_url, header=None, names=data_columns, na_values='?')

        df = df.dropna()

        feature_names = df.columns[:-1]

        for feature_name in feature_names:
            plt.figure()
            plt.scatter(df[feature_name], df['target'])
            plt.xlabel(feature_name)
            plt.ylabel('target')
            plt.title(f'{feature_name} vs target')

        plt.show()

    except ValueError:
        accuracy_label.config(text="")
    except FileNotFoundError:
        print("Błąd: Nie znaleziono pliku.")
    except pd.errors.EmptyDataError:
        print("Błąd: Plik jest pusty.")
    except pd.errors.ParserError:
        print("Błąd: Błąd podczas parsowania pliku.")


def get_diagnosis_description(prediction):
    descriptions = {
        0: "Brak obecności choroby serca. (Wartość 0)",
        1: "Obecność choroby serca. (Wartość 1)",
        2: "Prawdopodobieństwo obecności choroby serca. (Wartość 2)",
        3: "Wysokie prawdopodobieństwo obecności choroby serca. (Wartość 3)",
        4: "Silne podejrzenie obecności choroby serca. (Wartość 4)"
    }
    return descriptions.get(prediction, "Brak opisu dla tej predykcji")


def predict():
    try:
        global model
        if model is None:
            raise ValueError("Najpierw trenuj model")

        new_data = {'age': float(age_entry.get()), 'sex': float(sex_entry.get()), 'cp': float(cp_entry.get()),
                    'trestbps': float(trestbps_entry.get()), 'chol': float(chol_entry.get()),
                    'fbs': float(fbs_entry.get()),
                    'restecg': float(restecg_entry.get()), 'thalach': float(thalach_entry.get()),
                    'exang': float(exang_entry.get()), 'oldpeak': float(oldpeak_entry.get()),
                    'slope': float(slope_entry.get()), 'ca': float(ca_entry.get()), 'thal': float(thal_entry.get())}
        new_data_df = pd.DataFrame([new_data])
        prediction = model.predict(new_data_df)
        prediction_label.config(text="Wynik predykcji: " + get_diagnosis_description(prediction[0]))


    except ValueError as e:
        prediction_label.config(text=str(e))


def browse_data():
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    data_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                    'ca',
                    'thal', 'target']
    df = pd.read_csv(data_url, header=None, names=data_columns, na_values='?')
    df = df.dropna()

    browse_window = tk.Toplevel(window)
    browse_window.title("Przeglądanie danych")

    frame = tk.Frame(browse_window)
    frame.pack(fill="both", expand=True)

    table = Table(frame, dataframe=df, showtoolbar=True, showstatusbar=True)
    table.show()


def load_data_from_csv():
    csv_path = tk.filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if csv_path:
        try:
            data_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                            'slope', 'ca', 'thal', 'target']
            df = pd.read_csv(csv_path, header=None, names=data_columns, na_values='?')
            df = df.dropna()

            display_input_fields()

            global model
            model = None

        except pd.errors.EmptyDataError:
            pass


def load_data_from_link():
    link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    if link:
        try:
            data_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                            'slope', 'ca', 'thal', 'target']
            df = pd.read_csv(link, header=None, names=data_columns, na_values='?')
            df = df.dropna()

            # Display the input fields for new data
            display_input_fields()

            # Update the global data variable
            global model
            model = None

        except pd.errors.EmptyDataError:
            pass


def display_input_fields():
    age_label.grid(row=5, column=0)
    age_entry.grid(row=5, column=1)
    sex_label.grid(row=6, column=0)
    sex_entry.grid(row=6, column=1)
    cp_label.grid(row=7, column=0)
    cp_entry.grid(row=7, column=1)
    trestbps_label.grid(row=8, column=0)
    trestbps_entry.grid(row=8, column=1)
    chol_label.grid(row=9, column=0)
    chol_entry.grid(row=9, column=1)
    fbs_label.grid(row=10, column=0)
    fbs_entry.grid(row=10, column=1)
    restecg_label.grid(row=11, column=0)
    restecg_entry.grid(row=11, column=1)
    thalach_label.grid(row=12, column=0)
    thalach_entry.grid(row=12, column=1)
    exang_label.grid(row=13, column=0)
    exang_entry.grid(row=13, column=1)
    oldpeak_label.grid(row=14, column=0)
    oldpeak_entry.grid(row=14, column=1)
    slope_label.grid(row=15, column=0)
    slope_entry.grid(row=15, column=1)
    ca_label.grid(row=16, column=0)
    ca_entry.grid(row=16, column=1)
    thal_label.grid(row=17, column=0)
    thal_entry.grid(row=17, column=1)
    predict_button.grid(row=18, column=0, columnspan=2)


def hide_input_fields():
    age_label.grid_remove()
    age_entry.grid_remove()
    sex_label.grid_remove()
    sex_entry.grid_remove()
    cp_label.grid_remove()
    cp_entry.grid_remove()
    trestbps_label.grid_remove()
    trestbps_entry.grid_remove()
    chol_label.grid_remove()
    chol_entry.grid_remove()
    fbs_label.grid_remove()
    fbs_entry.grid_remove()
    restecg_label.grid_remove()
    restecg_entry.grid_remove()
    thalach_label.grid_remove()
    thalach_entry.grid_remove()
    exang_label.grid_remove()
    exang_entry.grid_remove()
    oldpeak_label.grid_remove()
    oldpeak_entry.grid_remove()
    slope_label.grid_remove()
    slope_entry.grid_remove()
    ca_label.grid_remove()
    ca_entry.grid_remove()
    thal_label.grid_remove()
    thal_entry.grid_remove()
    predict_button.grid_remove()


window = tk.Tk()
window.title("Diagnostyka choroby serca")

load_link_button = tk.Button(window, text="Wczytaj dane z linku", command=load_data_from_link)
load_link_button.grid(row=0, column=3, padx=10, pady=5)

load_csv_button = tk.Button(window, text="Wczytaj dane z pliku CSV", command=load_data_from_csv)
load_csv_button.grid(row=0, column=4, padx=10, pady=5)

train_button = tk.Button(window, text="Trenuj model", command=train_model)
train_button.grid(row=0, column=5, padx=10, pady=5)

rebuild_button = tk.Button(window, text="Przebuduj model", command=rebuild_model)
rebuild_button.grid(row=0, column=6, padx=10, pady=5)

save_button = tk.Button(window, text="Zapisz model", command=save_model)
save_button.grid(row=0, column=7, padx=10, pady=5)

load_button = tk.Button(window, text="Wczytaj model", command=load_model)
load_button.grid(row=0, column=8, padx=10, pady=5)

evaluate_button = tk.Button(window, text="Oceń model", command=evaluate_model)
evaluate_button.grid(row=0, column=9, padx=10, pady=5)

visualize_button = tk.Button(window, text="Wizualizuj dane", command=visualize_data)
visualize_button.grid(row=0, column=10, padx=10, pady=5)

age_label = tk.Label(window, text="Wiek:")
age_entry = tk.Entry(window)

sex_label = tk.Label(window, text="Płeć:")
sex_entry = tk.Entry(window)

cp_label = tk.Label(window, text="Ból klatki piersiowej:")
cp_entry = tk.Entry(window)

trestbps_label = tk.Label(window, text="Ciśnienie krwi:")
trestbps_entry = tk.Entry(window)

chol_label = tk.Label(window, text="Cholesterol:")
chol_entry = tk.Entry(window)

fbs_label = tk.Label(window, text="Poziom cukru we krwi:")
fbs_entry = tk.Entry(window)

restecg_label = tk.Label(window, text="Wynik elektrokardiografii w spoczynku:")
restecg_entry = tk.Entry(window)

thalach_label = tk.Label(window, text="Maksymalne osiągnięte tętno:")
thalach_entry = tk.Entry(window)

exang_label = tk.Label(window, text="Dławica wysiłkowa:")
exang_entry = tk.Entry(window)

oldpeak_label = tk.Label(window, text="Obniżenie odcinka ST po wysiłku:")
oldpeak_entry = tk.Entry(window)

slope_label = tk.Label(window, text="Nachylenie odcinka ST:")
slope_entry = tk.Entry(window)

ca_label = tk.Label(window, text="Naczyń barwnikowanych fluoroskopowo:")
ca_entry = tk.Entry(window)

thal_label = tk.Label(window, text="Talasemia:")
thal_entry = tk.Entry(window)

predict_button = tk.Button(window, text="Predykcja", command=predict)

accuracy_label = tk.Label(window, text="")
accuracy_label.grid(row=19, column=0, columnspan=2)

prediction_label = tk.Label(window, text="")
prediction_label.grid(row=20, column=0, columnspan=2)

accuracy_label = tk.Label(window, text="", font=('Arial', 12))
accuracy_label.grid(row=1, column=0, columnspan=9)

hide_input_fields()

window.mainloop()
