import pandas as pd 
import matplotlib.pyplot as plt 
import joblib
from sklearn.model_selection import train_test_split 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBRegressor 
from sklearn.metrics import(mean_absolute_error,mean_squared_error,r2_score)

df=pd.read_csv(
    "data/train_FD001.txt",
    sep=r"\s+",
    header=None
)
print("Data shape:")
print(df.shape)
print("\nFirst 5 rows:")
print(df.head())

#creating coulmns names

columns=['engine_id','cycle']
columns +=[f'operating_settings_{i}' for i in range(1,4)]
columns +=[f'sensor_{i}' for i in range(1,22)]
df.columns=columns
print(df.head())
print("\nShape:",df.shape)

print("\nNumber of engines:")
print(df["engine_id"].nunique())

#max cycles reached by each engine

max_cycles=df.groupby("engine_id")["cycle"].max()
print("\nMaximum cycle for first 5 engines:")
print(max_cycles.head())

# max cycle for each engine
df=df.merge(
    max_cycles,
    on="engine_id",
    suffixes=("","_max")
)

#calculating remaining useful life
#RUL=remaining useful life
df["RUL"]=df["cycle_max"]-df["cycle"]
print(df[["engine_id","cycle","cycle_max","RUL"]].head(10))

#EDA = Exploratory Data Analysis

print("\nRUL statistics:")
print(df["RUL"].describe())

print(df.isnull().sum())

print("\nRUL Range:")
print("Minimum RUL:", df["RUL"].min())
print("Maximum RUL:", df["RUL"].max())

plt.figure(figsize=(8,5))
plt.hist(df["RUL"],bins=30,color='green')
plt.title("Distribution of remaining useful life (RUL)")
plt.xlabel("RUL")
plt.ylabel("Frequency")
plt.show()

#sensor degradation analysis

engine1=df[df["engine_id"]==1]
plt.figure(figsize=(8,5))
plt.plot(
    engine1["cycle"],
    engine1["sensor_11"]
)
plt.title("Sensor 11 vs cycle (engine 1)")
plt.xlabel("Cycle")
plt.ylabel("sensor 11")
plt.show()

print("\nSensor Varience:")
sensor_cols=[f"sensor_{i}" for i in range(1,22)]
print(
    df[sensor_cols]
    .std()
    .sort_values()
    )

print("\nCorrelation with RUL:")
correlation=(
    df[sensor_cols+["RUL"]]
    .corr()["RUL"]
    .sort_values(ascending=False)
)
print(correlation)

# Feature selection

#features
X=df[sensor_cols]
#target
y=df["RUL"]
selector=SelectKBest(
    score_func=mutual_info_regression,
    k=10
)
X_selected=selector.fit_transform(X,y)
selected_feature=X.columns[
    selector.get_support()
]
print("\nTop 10 selected sensors:")
print(selected_feature)

X=df[selected_feature]
#train test split
X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
    )

print("\nTraining shape:")
print(X_train.shape)

print("\nTesting shape:")
print(X_test.shape)

#XGBoost model
model=XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

#model evaluation

mae=mean_absolute_error(y_test,y_pred)
rmse=mean_squared_error(
    y_test,
    y_pred
)** 0.5

r2=r2_score(y_test,y_pred)
print("\nModel Performance")
print("MAE:",mae)
print("RMSE:",rmse)
print("R2 :",r2)

results=pd.DataFrame({
    "Actual_RUL":y_test,
    "Predicted_RUL":y_pred
})
print("\nSample Prediction:")
print(results.head(20))

importance=pd.DataFrame({
    "Feature":selected_feature,
    "Importance":model.feature_importances_
})

importance=importance.sort_values(
    by="Importance",
    ascending=False
)
print("\nFeature Importance:")
print(importance)

def classify_risk(rul):

    if rul > 50:
        return "SAFE"

    elif rul > 20:
        return "WARNING"

    else:
        return "CRITICAL"
    
results["Status"] = results["Predicted_RUL"].apply(classify_risk)

print("\nMaintenance Decisions:")
print(results.head(20))

plt.figure(figsize=(8,5))
plt.bar(
    importance["Feature"],
    importance["Importance"]
)
plt.title("Feature Importance")
plt.xlabel("Sensors")
plt.ylabel("Importance")
plt.show()

# Save trained model

joblib.dump(
    model,
    "models/xgb_rul_model.pkl"
)

print("\nModel saved successfully!")