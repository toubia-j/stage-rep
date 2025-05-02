import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold ,cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import time
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Dropout,LSTM
from tensorflow.keras.layers import Conv1D, MaxPooling1D  ,Conv2D, MaxPooling2D ,Flatten ,Reshape
import tensorflow as tf



def extract_columns(filepath, column_index=4):
    """
    Extrait une colonne spécifique d'un fichier CSV et la transforme en un DataFrame de 24 heures.
    """
    df = pd.read_csv(filepath, delimiter="\t")
    values = df.iloc[:, column_index].values
    return pd.DataFrame(values.reshape(-1, 24))

def extract_and_concat_consommation(city, column_index, prefix):
    """
    Extrait des colonnes spécifiques de plusieurs fichiers de données liés à différentes villes,
    crée un DataFrame global pour chaque ville avec le nom {prefix}{ville}, puis concatène tous les résultats
    dans un seul DataFrame global nommé df_combined_{ville}. 
    """
    extracted_data = []
    for city, path in city.items():
        data = extract_columns(path, column_index)
        globals()[f"{prefix}{city}"] = data
        extracted_data.append(data)
    combined_df = pd.concat(extracted_data, axis=0).reset_index(drop=True)
    globals()[f"df_combined_{city}"] = combined_df
    return combined_df



def add_binary_column(df, column_name="heat_on"):
    """
    Ajout d'une colonne binaire pour identifier les jours de consommation :
    - '1' indique un jour "ON" (consommation > 0)
    - '0' indique un jour "OFF" (consommation = 0)
    """
    df[column_name] = (df.drop(columns=[column_name], errors='ignore').sum(axis=1) > 0).astype(int)
    return df



def apply_kmeans(n_clusters,data):
    kmeans=KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    return kmeans 


    
def clustering(df, n_parts=1, status_column="heat_on", n_clusters_list=None):
    """
    Applique le clustering K-means automatiquement en divisant les colonnes horaires en n_parts égales.
    
    - df : DataFrame avec 24 colonnes horaires (0 à 23)
    - n_parts : nombre de parties à créer (1 = pas de partition)
    - status_column : colonne  indiquant si le chauffage est activé
    - n_clusters_list : liste du nombre de clusters à appliquer par partie (taille doit être = n_parts)
    
    Retour : df avec colonnes "clusters_1", "clusters_2", ..., une pour chaque partie ou  "clusters_1" si pas de partition
    """
    if n_clusters_list is None or len(n_clusters_list) != n_parts:
        raise ValueError("Tu dois fournir une liste n_clusters_list de même longueur que n_parts.")

    df_final = add_binary_column(df.copy(), column_name=status_column)
    df_final.columns = df_final.columns.astype(str)

    hour_columns = list(map(str, range(24)))
    step = 24 // n_parts
    parts_cols = [hour_columns[i * step: (i + 1) * step] for i in range(n_parts)]

    for i, (n_clusters, cols) in enumerate(zip(n_clusters_list, parts_cols), start=1):
        df_part = df_final[cols + [status_column]].copy()
        df_heat = df_part[df_part[status_column] == 1].drop(columns=[status_column])
        
        model = apply_kmeans(n_clusters=n_clusters, data=df_heat)

        cluster_col = f"clusters_{i}"
        df_final.loc[df_part[status_column] == 1, cluster_col] = model.labels_
        df_final.loc[df_part[status_column] == 0, cluster_col] = n_clusters 

    return df_final



def extract_and_store_data(files, prefix, column_index):
    """
    Pour chaque fichier dans `files`, extrait toutes les colonnes
    et les stocke dans des variables globales nommées comme <NomColonne>_<ville>
    Exemple : Text_toulouse, Hum_agen, wind_zurich_kloten
    """
    for city, path in files.items():
        data = extract_columns(path, column_index)
        globals()[f"{prefix}{city}"] = data


def extract_and_combine_all(city_groups, prefix_column_map):
    """
     Combine toutes les colonnes extraites (ayant le même préfixe) pour le groupe de villes actuel
     Par exemple, combine Text_agen, Text_albi, etc., en un seul DataFrame : Text_combined_toulouse

    """
    for group_name, files in city_groups.items():
        for prefix, col_index in prefix_column_map.items():
            extract_and_store_data(files, prefix, col_index)

            dfs = []
            for city in files.keys():
                var_name = f"{prefix}{city}"
                if var_name in globals():
                    dfs.append(globals()[var_name])
            if dfs:
                combined_name = f"{prefix}combined_{group_name}"
                globals()[combined_name] = pd.concat(dfs, axis=0).reset_index(drop=True)



def add_profil_and_status(input_df, conso_df, status_col="heat_on", profil_cols=None):
    """
    Ajoute la colonne 'status_col' et une ou plusieurs colonnes 'profil_cols' de 'conso_df' à 'input_df'.
    
    - input_df : DataFrame de base
    - conso_df : DataFrame contenant les colonnes à ajouter
    - status_col : colonne du statut (par défaut 'heat_on')
    - profil_cols : une chaîne (ex: "clusters_1") ou une liste (ex: ["clusters_1", "clusters_2", ...])
    """
    df = input_df.copy()
    df[status_col] = conso_df[status_col]

    if isinstance(profil_cols, str):
        profil_cols = [profil_cols]  

    if profil_cols:
        for col in profil_cols:
            df[col] = conso_df[col]

    df.columns = df.columns.astype(str)
    return df




import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, hamming_loss, zero_one_loss
from sklearn.preprocessing import MultiLabelBinarizer

def evaluate_models_split(df, target_cols, models, split_ratio=8):
    """
    Évalue plusieurs modèles (mono-label ou multi-label) avec séparation manuelle (80% par défaut).
    Si `target_cols` contient plusieurs colonnes => multi-label.
    Retourne :
      - un dictionnaire avec les métriques,
      - un DataFrame avec les vraies valeurs et prédictions.
    """
    multi_label = isinstance(target_cols, list) and len(target_cols) > 1
    y = df[target_cols] if multi_label else df[[target_cols]]
    X = df.drop(columns=target_cols if multi_label else [target_cols])

    split_index = int((X.shape[0] * split_ratio) / 10)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    results = {}
    df_test_results = X_test.copy()
    for col in (target_cols if multi_label else [target_cols]):
        df_test_results[f'y_true_{col}'] = y_test[col].values

    if multi_label:
        mlb = MultiLabelBinarizer()
        y_train_bin = mlb.fit_transform(y_train.values.tolist())
        y_test_bin = mlb.transform(y_test.values.tolist())

    for name, model in models.items():
        print(f"\nÉvaluation de {name}...")
        start_time = time.time()
        if multi_label:
            model.fit(X_train, y_train_bin)
            y_pred_bin = model.predict(X_test)
        else:
            model.fit(X_train, y_train.values.ravel())
            y_pred = model.predict(X_test)
        exec_time = time.time() - start_time

        if multi_label:
            # Scores multi-label
            f1 = f1_score(y_test_bin, y_pred_bin, average='weighted')
            acc = accuracy_score(y_test_bin, y_pred_bin)
            zero_one = zero_one_loss(y_test_bin, y_pred_bin)
            hamming = hamming_loss(y_test_bin, y_pred_bin)

            results[name] = {
                "f1_score": f1,
                "accuracy": acc,
                "zero_one_loss": zero_one,
                "hamming_loss": hamming,
                "execution_time (s)": exec_time
            }

            # Stocker les colonnes binaires en colonnes séparées
            for i, col in enumerate(target_cols):
                df_test_results[f'y_pred_{name}_{col}'] = y_pred_bin[:, i]

            print(f"{name} - F1: {f1:.4f} - Accuracy: {acc:.4f} - 0/1 Loss: {zero_one:.4f} - Hamming Loss: {hamming:.4f} - Temps: {exec_time:.2f}s")
        else:
            # Scores classification simple
            f1 = f1_score(y_test, y_pred, average='weighted')
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            results[name] = {
                "f1_score": f1,
                "accuracy": acc,
                "execution_time (s)": exec_time
            }

            df_test_results[f'y_pred_{name}_clusters_1'] = y_pred

            # Affichage matrice de confusion
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=model.classes_, yticklabels=model.classes_)
            plt.xlabel('Prédictions')
            plt.ylabel('Vraies classes')
            plt.title(f'Matrice de confusion - {name}')
            plt.show()

            print(f"{name} - F1: {f1:.4f} - Accuracy: {acc:.4f} - Temps: {exec_time:.2f}s")
        print("###################################################################")

    return results, df_test_results



def concat_and_create_final_df(city, prefixes):
    """
    Concatène les DataFrames spécifiés par des préfixes pour une ville donnée afin de former 
    une entrée multivariée  pour un modèle de prédiction.
    Le nom du DataFrame final est généré  selon la structure :
    "{prefix1}_{prefix2}_..._combined_{city}"
    Par exemple : Text_Solar_Ground_combined_toulouse
    """
    dfs = []
    for prefix in prefixes:
        var_name = f"{prefix}_combined_{city}"  
        if var_name in globals():
            dfs.append(globals()[var_name]) 

    final_df_name = f"{'_'.join(prefixes)}_combined_{city}" 
    final_df = pd.concat(dfs, axis=1).reset_index(drop=True)  
    globals()[final_df_name] = final_df  
    return final_df


def plot_f1_accuracy_results(results_dict, df_name):
    """
    Affiche un histogramme comparant le F1-score et l'Accuracy pour chaque modèle à partir des résultats,
    avec les valeurs affichées au-dessus des barres.
    """
    if "_conso_heat_" in df_name:
        parts = df_name.split("_conso_heat_")
        prefixes = parts[0].replace("_", ", ")
        city = parts[1].capitalize()
        title = f"Input: {prefixes} à {city}"
    else:
        title = "Comparaison des modèles"

    df = pd.DataFrame(results_dict).T[['f1_score', 'accuracy']]
    x = np.arange(len(df))  
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_f1 = ax.bar(x - width/2, df['f1_score'], width, label='F1 Score', color='skyblue')
    bars_acc = ax.bar(x + width/2, df['accuracy'], width, label='Accuracy', color='lightgreen')

    # Ajouter les valeurs en haut des barres
    for bar in bars_f1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}", ha='center', va='bottom', fontsize=8)

    for bar in bars_acc:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}", ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.show()




def train_and_evaluate(model, X_train, y_train, X_test, y_test, scaler_temp, scaler_cons):
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
    
    # Courbe de loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss during training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
   
    # Évaluation sur validation
    loss, mae, mse = model.evaluate(X_val, y_val)
    rmse = np.sqrt(mse) 
    print(f"Validation Loss : {loss}")
    print(f"Validation MAE: {mae}")
    print(f"Validation MSE: {mse}")
    print(f"Validation RMSE: {rmse}")
    
    # Prédiction et évaluation sur test
    predictions = model.predict(X_test) 
    y_test_reshape = y_test.reshape(-1, 24) 
    predictions_norm = scaler_cons.inverse_transform(predictions)
    y_test_reshape_norm = scaler_cons.inverse_transform(y_test_reshape)

    mae_test = mean_absolute_error(y_test_reshape_norm, predictions_norm)
    mse_test = mean_squared_error(y_test_reshape_norm, predictions_norm)
    rmse_test = np.sqrt(mse_test)
    r2 = r2_score(y_test_reshape_norm, predictions_norm)
    cvrmse = rmse_test / np.mean(y_test_reshape_norm)

    print(f"Test MAE: {mae_test}")
    print(f"Test MSE: {mse_test}")
    print(f"Test RMSE: {rmse_test}")
    print(f"Test R²: {r2}")
    print(f"Test CVRMSE: {cvrmse}")

    return history, loss, mae, mse, rmse, mae_test, mse_test, rmse_test, r2, cvrmse, predictions





from sklearn.utils import resample
def downsample_majority_class(df, target_column):
    """
    Réduire la classe majoritaire pour qu'elle soit égale au nombre de la classe maximale des autres classes.
    """
    counts = df[target_column].value_counts()
    majority_value = counts.idxmax()
    max_other = counts.drop(index=majority_value).max()
    df_majority = df[df[target_column] == majority_value]
    df_majority_downsampled = resample(df_majority, replace=False, n_samples=max_other, random_state=42)
    df_others = df[df[target_column] != majority_value]
    balanced_df = pd.concat([df_majority_downsampled, df_others]).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df



def model_lstm(X_train2, y_train2, X_test2, y_test2, scaler_temp, scaler_cons):
    model2 = Sequential()
    model2.add(LSTM(68, activation='tanh', input_shape=(X_train2.shape[1], X_train2.shape[2])))  
    model2.add(Dropout(0.2))
    model2.add(Dense(24, activation='linear'))   
    
    history2, loss2, mae2, mse2, rmse2, mae_test2, mse_test2, rmse_test2, r2, cvrmse, predictions2 = train_and_evaluate(
        model2, X_train2, y_train2, X_test2, y_test2, scaler_temp, scaler_cons
    )
    
    return model2, history2, loss2, mae2, mse2, rmse2, mae_test2, mse_test2, rmse_test2, r2, cvrmse, predictions2




import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def make_column_names_unique(columns):
    """
    Cette fonction rend les noms des colonnes uniques en ajoutant un suffixe aux doublons.
    """
    seen = {}
    result = []
    for col in columns:
        if col not in seen:
            seen[col] = 1
            result.append(col)
        else:
            seen[col] += 1
            result.append(f"{col}_{seen[col]}")
    return result

def preprocess_data(Text_combined, clustering_heat, Test_Text_heat, name_combined):
    """
    -Cette fonction prépare les données pour un modèle LSTM.
    -L'équilibrage de la classe majoritaire est effectué uniquement sur les jours prédits, 
    et n'est pas effectué sur les jours passés utilisés comme entrées (t-1).
    -La prédiction est faite en fonction des différentes données d'entrée et de consommation,
    ainsi que du profil réel à t-1 et des différentes données d'entrée et des profils prédits à t.
    """
    split_index = int(0.8 * len(clustering_heat))
    df = Text_combined.copy()
    df['heat_on'] = clustering_heat['heat_on']

    # Vérification de l'index après la copie de df

    # Vérification des colonnes dupliquées dans Text_combined et clustering_heat
    duplicates_df = df.columns[df.columns.duplicated()]
    duplicates_clustering_heat = clustering_heat.columns[clustering_heat.columns.duplicated()]
    if len(duplicates_df) > 0:
        df.columns = make_column_names_unique(df.columns)
    if len(duplicates_clustering_heat) > 0:
        clustering_heat.columns = make_column_names_unique(clustering_heat.columns)


    # Colonnes de clusters
    cluster_cols = clustering_heat.filter(like='cluster').columns
    df.loc[:split_index - 1, cluster_cols] = clustering_heat.loc[:split_index - 1, cluster_cols]

    # Affichage de l'index après l'ajout des clusters


    # Ajout des prédictions de clusters
    cluster_cols2 = Test_Text_heat.filter(like='y_pred_Gradient').columns
    for cluster_idx in range(1, len(cluster_cols2) + 1):
        cluster_col_name = f'y_pred_Gradient Boosting_clusters_{cluster_idx}'
        df.loc[split_index:, f'clusters_{cluster_idx}'] = Test_Text_heat.loc[:, cluster_col_name].values



    # Rendre les noms de colonnes uniques
    df.columns = make_column_names_unique(df.columns)

    # Ajout de colonnes pour l'équilibrage
    df = pd.concat([pd.Series(range(len(clustering_heat))), df, clustering_heat.iloc[:, :-(len(cluster_cols) + 1)]], axis=1).reset_index(drop=True)

    # Vérification de l'index après concaténation

    # Gestion des colonnes dupliquées
    duplicates = df.columns[df.columns.duplicated()]
    df.columns = make_column_names_unique(df.columns)
    df.columns = df.columns.astype(str)

    # Appliquer downsampling sur la classe majoritaire
    df2 = downsample_majority_class(df, 'heat_on')
    df2.columns = make_column_names_unique(df2.columns)

    # Vérification de l'index après downsampling
    print("Index après downsampling : ", df2.index)

    n_blocks = len(name_combined.split('_combined')[0].split('_'))
    parts = name_combined.split('_combined')[0].split('_')
    formatted = ' and '.join(parts)
    print(f"Prediction based on : {formatted}")

    n_temp_cols = 24 * n_blocks

    scaler_temp = StandardScaler()
    scaler_cons = StandardScaler()

    cluster_cols = df.columns[df.columns.str.contains('clusters_')]
    cluster_cols2 = df2.columns[df2.columns.str.contains('clusters_')]

    df_scaled = np.hstack([
        df.iloc[:, 0:1].values,  # ID du jour
        scaler_temp.fit_transform(df.iloc[:, 1:1 + n_temp_cols]),  # Température
        df.iloc[:, 1 + n_temp_cols:1 + n_temp_cols + 1].values,  # 'heat_on'
        df[cluster_cols].values,  # Clusters
        scaler_cons.fit_transform(df.iloc[:, -24:])  # Consommation
    ])

    df_scaled2 = np.hstack([
        df2.iloc[:, 0:1].values,  # ID du jour
        scaler_temp.fit_transform(df2.iloc[:, 1:1 + n_temp_cols]),  # Température
        df2.iloc[:, 1 + n_temp_cols:1 + n_temp_cols + 1].values,  # 'heat_on'
        df2[cluster_cols2].values,  # Clusters
        scaler_cons.fit_transform(df2.iloc[:, -24:])  # Consommation
    ])


    # Conversion en DataFrame
    df_final = pd.DataFrame(df_scaled, columns=df.columns)
    df_final2 = pd.DataFrame(df_scaled2, columns=df2.columns)

    data = df_final.values
    data2 = df_final2.values
    data2 = data2[data2[:, 0] != 0]  # Filtrer les jours où l'ID est 0

    # Création de X2 et y2
    X2, y2 = [], []
    for i in data2[:, 0]:
        prev_data = data[data[:, 0] == i - 1, 1:]
        current_data2 = data2[data2[:, 0] == i, 1:1 + n_temp_cols + len(cluster_cols) + 1]
        X2.append(np.hstack([prev_data, current_data2]))  # Données d'entrée
        y2.append(data2[data2[:, 0] == i, 1 + n_temp_cols + 1 + len(cluster_cols):])  # Cibles

    X2, y2 = np.array(X2), np.array(y2)
    X2 = X2.reshape(X2.shape[0], X2.shape[2])
    y2 = y2.reshape(y2.shape[0], y2.shape[2])
    X2 = X2.reshape(X2.shape[0], 1, X2.shape[1])

    # Séparation en train/test
    idx_split = int((X2.shape[0] * 8) / 10)
    X_train2 = X2[:idx_split, :].astype(float)
    X_test2 = X2[idx_split:, :].astype(float)
    y_train2 = y2[:idx_split, :].astype(float)
    y_test2 = y2[idx_split:, :].astype(float)

   

    return X_train2, X_test2, y_train2, y_test2
