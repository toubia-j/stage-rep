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
import sys
import importlib   
import utils
importlib.reload(utils)
from utils import *



conso_tou = pd.read_csv("../data/simulation_data/conso_kmeans_data/clusters_heat_perif_toulouse_kmeans")
conso_zur = pd.read_csv("../data/simulation_data/conso_kmeans_data/clusters_heat_perif_zurich_kmeans")
conso_sev = pd.read_csv("../data/simulation_data/conso_kmeans_data/clusters_cool_perif_seville_kmeans")
occupation=pd.read_csv("../data/simulation_data/Occupancy_per_hour",delimiter="\t")

files = {
    "agen": "../data/simulation_data/Meteo_Perif_Toulouse_Contemporain/Agen/Simulation_Outputs",
    "albi": "../data/simulation_data/Meteo_Perif_Toulouse_Contemporain/Albi/Simulation_Outputs",
    "auch": "../data/simulation_data/Meteo_Perif_Toulouse_Contemporain/Auch/Simulation_Outputs",
    "toulouse": "../data/simulation_data/Meteo_Perif_Toulouse_Contemporain/Toulouse/Simulation_Outputs",
    "Birmensdorf":"../data/simulation_data/Meteo_Perif_Zurich_Contemporain/Birmensdorf/Simulation_Outputs",
    "Taenikon":"../data/simulation_data/Meteo_Perif_Zurich_Contemporain/Taenikon/Simulation_Outputs",
    "Zurich_fluntern":"../data/simulation_data/Meteo_Perif_Zurich_Contemporain/Zuerich_Fluntern/Simulation_Outputs",
    "Zurich_kloten":"../data/simulation_data/Meteo_Perif_Zurich_Contemporain/Zuerich_kloten/Simulation_Outputs",
    "Cordoba": "../data/simulation_data/Meteo_Perif_Seville_Contemporain/Cordoba/Simulation_Outputs",
    "Granada": "../data/simulation_data/Meteo_Perif_Seville_Contemporain/Granada/Simulation_Outputs",
    "Malaga": "../data/simulation_data/Meteo_Perif_Seville_Contemporain/Malaga/Simulation_Outputs",
    "Sevilla": "../data/simulation_data/Meteo_Perif_Seville_Contemporain/Sevilla/Simulation_Outputs"   
}

files3 = {
    "agen": "../data/simulation_data/Meteo_Perif_Toulouse_Contemporain/Agen/Meteo_input",
    "albi": "../data/simulation_data/Meteo_Perif_Toulouse_Contemporain/Albi/Meteo_input",
    "auch": "../data/simulation_data/Meteo_Perif_Toulouse_Contemporain/Auch/Meteo_input",
    "toulouse": "../data/simulation_data/Meteo_Perif_Toulouse_Contemporain/Toulouse/Meteo_input",
    "Birmensdorf":"../data/simulation_data/Meteo_Perif_Zurich_Contemporain/Birmensdorf/Meteo_input",
    "Taenikon":"../data/simulation_data/Meteo_Perif_Zurich_Contemporain/Taenikon/Meteo_input",
    "Zurich_fluntern":"../data/simulation_data/Meteo_Perif_Zurich_Contemporain/Zuerich_Fluntern/Meteo_input",
    "Zurich_kloten":"../data/simulation_data/Meteo_Perif_Zurich_Contemporain/Zuerich_kloten/Meteo_input",
    "Cordoba": "../data/simulation_data/Meteo_Perif_Seville_Contemporain/Cordoba/Meteo_input",
    "Granada": "../data/simulation_data/Meteo_Perif_Seville_Contemporain/Granada/Meteo_input",
    "Malaga": "../data/simulation_data/Meteo_Perif_Seville_Contemporain/Malaga/Meteo_input",
    "Sevilla": "../data/simulation_data/Meteo_Perif_Seville_Contemporain/Sevilla/Meteo_input"  
}




"""
Décomposition des données : Température extérieure, intérieure, humidité, vitesse du vent, angle solaire, réflectivité du sol,
pour chaque ville sur une période de 24h. Chaque variable est stockée dans un DataFrame distinct par ville.
"""
for city, path in files.items():
      globals()[f"Text_{city}"] = extract_columns(files[city],1)

for city3, path3 in files3.items():
      globals()[f"hum_{city3}"] = extract_columns(files3[city3],3)  

for city4, path4 in files3.items():
      globals()[f"wind_{city4}"] = extract_columns(files3[city4],4)  

for city5, path5 in files3.items():
      globals()[f"solar_{city5}"] = extract_columns(files3[city5],5)  

for city6, path6 in files3.items():
      globals()[f"ground_{city6}"] = extract_columns(files3[city6],10)  

for city, path in files.items():
      globals()[f"Tint_{city}"] = extract_columns(files[city],2)
    

for city, path in files.items():
      globals()[f"consommation_heat_{city}"] = extract_columns(files[city])

for city2, path2 in files.items():
      globals()[f"consommation_cool_{city2}"] = extract_columns(files[city2],5)  



models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "SVC": SVC(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

models2 = {
    "Random Forest": MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)),
    "Logistic Regression": MultiOutputClassifier(LogisticRegression(random_state=42, max_iter=1000)),
    "SVC": MultiOutputClassifier(SVC(random_state=42)),
    "KNN": MultiOutputClassifier(KNeighborsClassifier()),
    "Gradient Boosting": MultiOutputClassifier(GradientBoostingClassifier(random_state=42))
}