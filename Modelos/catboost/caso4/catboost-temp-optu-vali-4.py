import pandas as pd
import numpy as np
import pickle
from optuna import create_study, Trial
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from darts.models import CatBoostModel
from darts import TimeSeries

#Constantes
TRIALS = 400
STUDY_NAME = "catboost-temp-4"
NUM_FOLDS = 10
SEMENTE_ALEATORIA = 352


# Ler dados
df = pd.read_csv("../../../dados/Dataset-REAL-OFICIAL-Cumulativo.csv", sep=',')

#Separar dados de treino e teste

df = df.drop(columns = ['Unnamed: 0'])
#df = df.drop(columns = ['Unnamed: 0.1'])

target = 'Casos Diários Cumulativos'
features = list(df.columns)
features.remove(target)
features.remove("Data") 

# Separando target e covariaveis, treino e teste
tamanho_treino = round(len(df) * 0.9)
# treino = df.loc[0:tamanho_treino]
# teste = df.loc[tamanho_treino + 1: len(df)]

series_target = TimeSeries.from_dataframe(df, time_col='Data', value_cols=[target], freq = 'D')
series_covariates = TimeSeries.from_dataframe(df, time_col='Data', value_cols=features, freq = 'D')

target_treino = series_target[0:tamanho_treino]
target_teste = series_target[tamanho_treino + 1:len(series_target)]

covariates_treino = series_covariates[0:tamanho_treino]
covariates_teste = series_covariates[tamanho_treino +1 : len(series_covariates)]

# Função p/ validação cruzada
def val_cruzada(modelo,folds,target_treino, covariates):
    n_segments = folds * 2
    segment_len = round(len(target_treino) // n_segments)

    if segment_len < 1:
        print(f"Error: Dataset length ({len(target_treino)}) is too short for {folds} non-overlapping windows.")
        print("Please reduce n_windows or use a larger dataset.")
        return

    train_len = segment_len
    test_len = segment_len
    rmses = []

    for i in range(folds):
        start_idx = i * (train_len + test_len)
        train_end_idx = start_idx + train_len
        test_end_idx = train_end_idx + test_len

        # Ensure the last window goes to the end of the data if there's a remainder
        if i == folds - 1:
            test_end_idx = len(target_treino)

        train = target_treino[start_idx:train_end_idx]
        test = target_treino[train_end_idx:test_end_idx]
        covariates_treino = covariates[start_idx:train_end_idx]

        modelo.fit(train, past_covariates = covariates_treino)

        pred = modelo.predict(len(test), past_covariates = covariates)
        rmse = mean_absolute_error(pred.values(), test.values())
        rmses.append(rmse)

    return sum(rmses)/len(rmses), rmses
 

#Otimização de hiperparâmetros
def cria_instancia_modelo(trial):

    parametros = {
        "lags": trial.suggest_int("lags", 1, 30),
        "lags_past_covariates": trial.suggest_int("lags_past_covariates", 1, 30),
        # hiperparâmetros do catboost
        "max_depth": trial.suggest_int("max_depth", 0, 12),
        "n_estimators": trial.suggest_int("n_estimators", 5, 1000, log=True),
        "subsample": trial.suggest_float("subsample", 0.1, 1),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1, log=True),
#        "grow_policy": trial.suggest_categorical("grow_policy", ["Depthwise", "Lossguide"]),
        "random_state": SEMENTE_ALEATORIA,
    }

    model = CatBoostModel(**parametros)

    return model

def funcao_objetivo(
    trial,
    X,
    y
):

    modelo = cria_instancia_modelo(trial)
    valor, folds = val_cruzada(modelo, NUM_FOLDS, X, y)

    return valor

study = create_study(
    study_name=STUDY_NAME,
    storage=f"sqlite:///hptuning/{STUDY_NAME}.db",
    direction="minimize",
    load_if_exists=True,
)

df_op = study.trials_dataframe()
if len(df_op) == 0:
    num_trials = TRIALS
else:
    num_trials = TRIALS - len(df_op.loc[(df_op["state"] == "COMPLETE")])



def funcao_objetivo_parcial(trial):
    return funcao_objetivo(trial, target_treino, covariates_treino)


study.optimize(funcao_objetivo_parcial, n_trials=num_trials)

trialdf = study.trials_dataframe()
trialdf.to_csv(f"hptuning/{STUDY_NAME}.csv", index=False)

melhor_trial = study.best_trial

print(f"Número do melhor trial: {melhor_trial.number}")
print(f"Parâmetros do melhor trial: {melhor_trial.params}")

modelo_final = cria_instancia_modelo(melhor_trial)

# Fzendo a validação cruzada de novo só p/ pegar os 10 kfolds
melhor_valor, melhor_folds = val_cruzada(modelo_final, NUM_FOLDS, target_treino, covariates_treino)

modelo_final = cria_instancia_modelo(melhor_trial)
modelo_final.fit(series = target_treino, past_covariates = covariates_treino)

y_verdadeiro = target_teste.values()
y_previsao = modelo_final.predict(n = len(target_teste), past_covariates = series_covariates)

MAE_teste = mean_absolute_error(y_verdadeiro, y_previsao.values())

print(f"Mae no teste:{MAE_teste}.")


# salvando modelo
with open(f"{STUDY_NAME}.pkl", "wb") as arquivo:
    pickle.dump(modelo_final, arquivo)
    
# salvando kfolds  
with open(f"{STUDY_NAME}_FOLDS.pkl", "wb") as arquivo:
    pickle.dump(melhor_folds, arquivo)