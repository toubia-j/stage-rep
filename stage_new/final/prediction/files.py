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

files1 = {
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

# Données incluant l'occupation 
files4 = {
    "agen": "../data/simulation_data/Toulouse_et_perfi_Occupation_0a20/Agen/Simulation_Outputs",
    "albi": "../data/simulation_data/Toulouse_et_perfi_Occupation_0a20/Albi/Simulation_Outputs",
    "auch": "../data/simulation_data/Toulouse_et_perfi_Occupation_0a20/Auch/Simulation_Outputs",
    "toulouse": "../data/simulation_data/Toulouse_et_perfi_Occupation_0a20/Toulouse/Simulation_Outputs"  
}
files5 = {
    "agen": "../data/simulation_data/Toulouse_et_perfi_Occupation_0a20/Agen/Meteo_input",
    "albi": "../data/simulation_data/Toulouse_et_perfi_Occupation_0a20/Albi/Meteo_input",
    "auch": "../data/simulation_data/Toulouse_et_perfi_Occupation_0a20/Auch/Meteo_input",
    "toulouse": "../data/simulation_data/Toulouse_et_perfi_Occupation_0a20/Toulouse/Meteo_input"
}
occupation_new=pd.read_csv("../data/simulation_data/Occupancy_per_hour_Youssef.txt",delimiter="\t")


files6 = {
    "2018": "../data/simulated_data_7years/Mal_isole/2018/Simulation_Outputs",
    "2019": "../data/simulated_data_7years/Mal_isole/2019/Simulation_Outputs",
    "2020": "../data/simulated_data_7years/Mal_isole/2020/Simulation_Outputs",
    "2021": "../data/simulated_data_7years/Mal_isole/2021/Simulation_Outputs",
    "2022": "../data/simulated_data_7years/Mal_isole/2022/Simulation_Outputs",
    "2023": "../data/simulated_data_7years/Mal_isole/2023/Simulation_Outputs",
    "2024": "../data/simulated_data_7years/Mal_isole/2024/Simulation_Outputs"   
}
files7 = {
    "2018": "../data/simulated_data_7years/Mal_isole/2018/Weather_Dataset.txt",
    "2019": "../data/simulated_data_7years/Mal_isole/2019/Weather_Dataset.txt",
    "2020": "../data/simulated_data_7years/Mal_isole/2020/Weather_Dataset.txt",
    "2021": "../data/simulated_data_7years/Mal_isole/2021/Weather_Dataset.txt",
    "2022": "../data/simulated_data_7years/Mal_isole/2022/Weather_Dataset.txt",
    "2023": "../data/simulated_data_7years/Mal_isole/2023/Weather_Dataset.txt",
    "2024": "../data/simulated_data_7years/Mal_isole/2024/Weather_Dataset.txt"   
}

occupation_7years=pd.read_csv("../data/simulated_data_7years/Mal_isole/Occupancy_per_hour",delimiter="\t")




"""
Décomposition des données : Température extérieure, intérieure, humidité, vitesse du vent, angle solaire, réflectivité du sol,
pour chaque ville sur une période de 24h. Chaque variable est stockée dans un DataFrame distinct par ville.
"""
for city, path in files4.items():
      globals()[f"Text_2_{city}"] = extract_columns(files4[city],1)

for city3, path3 in files5.items():
      globals()[f"hum_2_{city3}"] = extract_columns(files5[city3],3)  

for city4, path4 in files5.items():
      globals()[f"wind_2_{city4}"] = extract_columns(files5[city4],4)  

for city5, path5 in files5.items():
      globals()[f"solar_2_{city5}"] = extract_columns(files5[city5],5)  

for city6, path6 in files5.items():
      globals()[f"ground_2_{city6}"] = extract_columns(files5[city6],10)  

for city, path in files4.items():
      globals()[f"Tint_2_{city}"] = extract_columns(files4[city],2)
    

for city, path in files1.items():
      globals()[f"Text_{city}"] = extract_columns(files1[city],1)

for city3, path3 in files3.items():
      globals()[f"hum_{city3}"] = extract_columns(files3[city3],3)  

for city4, path4 in files3.items():
      globals()[f"wind_{city4}"] = extract_columns(files3[city4],4)  

for city5, path5 in files3.items():
      globals()[f"solar_{city5}"] = extract_columns(files3[city5],5)  

for city6, path6 in files3.items():
      globals()[f"ground_{city6}"] = extract_columns(files3[city6],10)  

for city, path in files1.items():
      globals()[f"Tint_{city}"] = extract_columns(files1[city],2)
    


for years, path in files6.items():
      globals()[f"Text_{years}"] = extract_columns(files6[years],1)

for years3, path3 in files7.items():
      globals()[f"hum_{years3}"] = extract_columns(files7[years3],3)  

for years4, path4 in files7.items():
      globals()[f"wind_{years4}"] = extract_columns(files7[years4],4)  

for years5, path5 in files7.items():
      globals()[f"solar_{years5}"] = extract_columns(files7[years5],5)  

for years6, path6 in files7.items():
      globals()[f"ground_{years6}"] = extract_columns(files7[years6],10)  

for years, path in files6.items():
      globals()[f"Tint_{years}"] = extract_columns(files6[years],2)





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