#!/usr/bin/env python
# coding: utf-8

# ##### Importing libraries 
# 

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ###### Importing Dataset
# 

# In[2]:


df = pd.read_csv("C:/Users/Aditya Venugopalan/Downloads/RFLFSODataFull.csv")
df


# In[3]:


print(df.info())


# ###### Understanding the statistics of the Dataset

# Now we will try to understand the descriptive statistics of our dataset. The following are the reasons to do so 
# 
# 1) We can get an idea of the distribution and variability of the dataset
# 2) We will be able to identify potential outliers , errors or any kind of skewness
# 3) We will be having a quick summary of the spread (std, min-max) and central tendencies (mean, median)

# In[4]:


print(df.describe())


# #### Exploratory Data Analysis (EDA)

# ###### Missing-value heatmap & percentages

# In[5]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

def show_missing(df):
    """Display % missing per column and a heatmap of missing entries."""
    missing_pct = df.isna().mean().sort_values(ascending=False)
    print("❗️ Percent missing by column:")
    display(missing_pct.to_frame("Pct Missing"))

    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isna(), cbar=False, cmap="viridis")
    plt.title("Missing-Value Map (True = missing)")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.show()
show_missing(df)


# ###### Distribution plots (KDE)

# We will be plotting KDE plots to understand the frequency distribution , shape , spread and check any skewness in our data

# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt


numeric_cols = df.select_dtypes(include='number').columns
n_cols = 3  
n_rows = -(-len(numeric_cols) // n_cols)  

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
axes = axes.flatten()


for i, col in enumerate(numeric_cols):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_ylabel('Frequency')


for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])


plt.tight_layout()
plt.show()


# ###### Correlation Heatmap

# We will use heatmap to understand the strength of relationships between pairs of features we have present in our dataset 

# In[7]:


plt.figure(figsize=(20,18))
sns.heatmap(df.corr(), annot =True, cmap='coolwarm', fmt=".2f")
plt.title('CCorrelation Heatmap')


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set color palette
weather_order = ['clear', 'rain', 'shower', 'drizzle', 'dust storm/fog', 'snow']
color_palette = sns.color_palette("Spectral", len(weather_order))

# Map your SYNOPCode to labels if not done
synop_map = {
    0: 'clear', 3: 'dust storm/fog', 4: 'drizzle',
    5: 'shower', 6: 'rain', 7: 'snow', 8: 'dust storm/fog'
}
df['Weather'] = df['SYNOPCode'].map(synop_map)

# Count plot (as horizontal bar)
plt.figure(figsize=(12, 4))
sns.countplot(y='Weather', data=df, order=weather_order, palette="Spectral")
plt.title("Dataset Composition by Weather Condition")
plt.xlabel("Count")
plt.ylabel("Weather")
plt.tight_layout()
plt.show()


# ###### Boxplots

# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns

# Setting style
sns.set(style="whitegrid")

#  Boxplot 
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(data=df[['FSO_Att', 'RFL_Att']], palette=["cornflowerblue", "seagreen"])
plt.title("Box Plot of FSO_Att and RFL_Att")
plt.ylabel("Attenuation (dB)")

#  Histogram 
plt.subplot(1, 2, 2)
sns.histplot(df['FSO_Att'], color="cornflowerblue", label="FSO_Att", bins=30, kde=False)
sns.histplot(df['RFL_Att'], color="seagreen", label="RFL_Att", bins=30, kde=False)
plt.title("Histogram of FSO_Att and RFL_Att")
plt.xlabel("Attenuation (dB)")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.show()


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt

# Map codes to weather labels (if not already done)
synop_map = {
    0: 'clear', 3: 'dust storm/fog', 4: 'drizzle',
    5: 'shower', 6: 'rain', 7: 'snow', 8: 'dust storm/fog'
}
df['Weather'] = df['SYNOPCode'].map(synop_map)

weather_order = ['clear', 'rain', 'dust storm/fog', 'shower', 'drizzle', 'snow']
palette = sns.color_palette("viridis", len(weather_order))

# === FSO Boxplot ===
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='Weather', y='FSO_Att', data=df, order=weather_order, palette=palette)
plt.title("FSO_Att vs Weather Condition")
plt.ylabel("FSO Attenuation (dB)")
plt.grid(axis='y')

# === RFL Boxplot ===
plt.subplot(1, 2, 2)
sns.boxplot(x='Weather', y='RFL_Att', data=df, order=weather_order, palette=palette)
plt.title("RFL_Att vs Weather Condition")
plt.ylabel("RFL Attenuation (dB)")
plt.grid(axis='y')

plt.tight_layout()
plt.show()


# In[11]:


# Define grid layout based on number of columns
n_cols = 3  # Set the number of columns in the grid
n_rows = -(-len(numeric_cols) // n_cols)  # Calculate the number of rows needed

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
axes = axes.flatten()

# Plot each column as a boxplot
for i, col in enumerate(numeric_cols):
    sns.boxplot(data=df, x=col, ax=axes[i])
    axes[i].set_title(f'Boxplot of {col}')

# Hide any empty subplots if there are fewer columns than grid spaces
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout and display the plot
plt.tight_layout()
plt.show()


# ##### Pre - processing 

# Now we will try to remove or correct wrong data types . In easy what I'm trying to establish , we will deal with abnormal data or data types that might cause problem in our analysis 

# ###### Checking missing values 

# In[12]:


df.isna().sum()


# ###### Checking Data Types 

# In[13]:


print(df.dtypes)


# ###### Dealing With outliers 

# As we have seen through visualization with the help of boxplots , the presence of outliers . We will remove them to make our analysis more accurate . 

# In[14]:


lower_bound = df.quantile(0.005)
upper_bound = df.quantile(0.995)
df = df.clip(lower=lower_bound, upper=upper_bound, axis=1)


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
for i, col in enumerate(df.columns, 1):
    plt.subplot(5, 6, i)  # Adjust subplot grid as needed
    sns.boxplot(data=df, y=col, whis=3)  # Increase whiskers
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()


# In[41]:


# Define grid layout based on number of columns
n_cols = 3  # Set the number of columns in the grid
n_rows = -(-len(numeric_cols) // n_cols)  # Calculate the number of rows needed

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
axes = axes.flatten()

# Plot each column as a boxplot
for i, col in enumerate(numeric_cols):
    sns.boxplot(data=df, x=col, ax=axes[i])
    axes[i].set_title(f'Boxplot of {col}')

# Hide any empty subplots if there are fewer columns than grid spaces
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout and display the plot
plt.tight_layout()
plt.show()


# ##### Correcting The Skewness

# In[17]:


skewness = df.skew()
print(skewness)


# As we can see from the above information related to skewness we can Particulate, ParticulateMax, ParticulateMin, RainIntensity, RainIntensityMax, RainIntensityMin, WindSpeedMax, and WindSpeedMin have high skewness . These features indicate that most of their respective values are clustered on left hand side applying a transformation will help to normalize the distribution and hence improving the model performance . On the other hand feature RelativeHumidity  show negative skewness . Since we haven't done feature selection related process , we will include this feature and hence transform RelativeHumidity feature as well. 

# In[18]:


import numpy as np 
df['Particulate'] = np.log1p(df['Particulate'])
df['ParticulateMax'] = np.log1p(df['ParticulateMax'])
df['ParticulateMin'] = np.log1p(df['ParticulateMin'])

df['RainIntensity'] = np.log1p(df['RainIntensity'])
df['RainIntensityMax'] = np.log1p(df['RainIntensityMax'])
df['RainIntensityMin'] = np.log1p(df['RainIntensityMin'])

df['WindSpeedMax'] = np.log1p(df['WindSpeedMax'])
df['WindSpeedMin'] = np.log1p(df['WindSpeedMin'])


# In[19]:


print(df.describe())


# In[20]:


skewness = df.skew()
print(skewness)


# In[21]:


df['RainIntensity'] = np.sqrt(df['RainIntensity'])
df['RainIntensityMin'] = np.sqrt(df['RainIntensityMin'])
df['RainIntensityMax'] = np.sqrt(df['RainIntensityMax'])


df['RelativeHumidity'] = np.power(df['RelativeHumidity'], 2)


# In[22]:


skewness = df.skew()
print(skewness)


# In[23]:


# Checking range and distribution
print(df.describe())


# ###### Splitting the data into train - test datasets 

# In[24]:


import pandas as pd

# Load Dataset
df = pd.read_csv("C:/Users/Aditya Venugopalan/Downloads/RFLFSODataFull.csv")

# Print column names
print("Column names in dataset:", df.columns.tolist())


# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("C:/Users/Aditya Venugopalan/Downloads/RFLFSODataFull.csv")
print("Dataset loaded. Columns:", df.columns.tolist())

# Define Features & Targets
X_generic = df.drop(columns=["FSO_Att", "RFL_Att"])  # All features (exclude targets)
y_fso = df["FSO_Att"]
y_rf = df["RFL_Att"]

# Shared Train-Test Split (80/20)
X_train, X_test, y_train_fso, y_test_fso = train_test_split(
    X_generic, y_fso, test_size=0.2, random_state=42
)
_, _, y_train_rf, y_test_rf = train_test_split(
    X_generic, y_rf, test_size=0.2, random_state=42
)

# Helper function to train Random Forest model
def train_rf_model(X, y):
    model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
    model.fit(X, y)
    return model

# Train General Models
general_fso_model = train_rf_model(X_train, y_train_fso)
general_rf_model = train_rf_model(X_train, y_train_rf)

# Evaluation function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    oob = model.oob_score_
    return rmse, r2, oob, y_pred

# Evaluate both models
rmse_fso, r2_fso, oob_fso, y_pred_fso = evaluate_model(general_fso_model, X_test, y_test_fso)
rmse_rf, r2_rf, oob_rf, y_pred_rf = evaluate_model(general_rf_model, X_test, y_test_rf)

# Print performance summary
print("\nGeneral Model Performance (Full Dataset):")
print(f"FSO - RMSE: {rmse_fso:.4f}, R²: {r2_fso:.4f}, OOB: {oob_fso:.4f}")
print(f"RF  - RMSE: {rmse_rf:.4f}, R²: {r2_rf:.4f}, OOB: {oob_rf:.4f}")

# Save predictions for future analysis (e.g., Method 2 & 3)
df_fso_preds = X_test.copy()
df_fso_preds["Measured_FSO_Att"] = y_test_fso.values
df_fso_preds["Predicted_FSO_Att"] = y_pred_fso
df_fso_preds.to_csv("Test_Predictions_FSO_Att.csv", index=False)

df_rf_preds = X_test.copy()
df_rf_preds["Measured_RFL_Att"] = y_test_rf.values
df_rf_preds["Predicted_RFL_Att"] = y_pred_rf
df_rf_preds.to_csv("Test_Predictions_RFL_Att.csv", index=False)

print("Predictions saved for further evaluation.")


# In[26]:


# Import Required Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Store Model Performance Metrics in a DataFrame
performance_metrics = pd.DataFrame({
    "Metric": ["OOB Score", "RMSE", "R2 Score"],
    "General FSO": [oob_fso, rmse_fso, r2_fso],
    "General RF": [oob_rf, rmse_rf, r2_rf]
})


# Melt DataFrame for better visualization with Seaborn
performance_melted = performance_metrics.melt(
    id_vars="Metric", var_name="Model", value_name="Score"
)

# Set Seaborn Style for Better Visualization
sns.set(style="whitegrid")

# Create Subplots for OOB, RMSE, and R2 Score
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# OOB Score Plot
sns.barplot(x="Metric", y="Score", hue="Model",
            data=performance_melted[performance_melted["Metric"] == "OOB Score"],
            ax=axes[0], palette="Set2")
axes[0].set_title("OOB Score Comparison (General Models)")
axes[0].set_xlabel("Metric")
axes[0].set_ylabel("Score")

# RMSE Plot
sns.barplot(x="Metric", y="Score", hue="Model",
            data=performance_melted[performance_melted["Metric"] == "RMSE"],
            ax=axes[1], palette="Set2")
axes[1].set_title("RMSE Comparison (General Models)")
axes[1].set_xlabel("Metric")
axes[1].set_ylabel("Score")

# R² Score Plot
sns.barplot(x="Metric", y="Score", hue="Model",
            data=performance_melted[performance_melted["Metric"] == "R2 Score"],
            ax=axes[2], palette="Set2")
axes[2].set_title("R² Score Comparison (General Models)")
axes[2].set_xlabel("Metric")
axes[2].set_ylabel("Score")

# Adjust Layout and Show Plot
plt.tight_layout()
plt.show()


# In[27]:


# Print OOB scores from already trained general models
oob_score_fso = general_fso_model.oob_score_
oob_score_rf = general_rf_model.oob_score_

print(f"General FSO Model OOB Score: {oob_score_fso:.3f}")
print(f"General RF Model OOB Score: {oob_score_rf:.3f}")


# ###### Feature selection
# 

# In[28]:


import pandas as pd

# Display SYNOPCode distribution
print("\nSYNOPCode Distribution:\n", df["SYNOPCode"].value_counts())

# Save the original dataset as-is
df.to_csv("Original_Dataset.csv", index=False)
print("\n Original dataset saved as 'Original_Dataset.csv'")


# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from IPython.display import display

def optimized_bfe_oob_improved(X, y, target_name, visualize=True, early_stop_delta=0.01):
    features = list(X.columns)
    rf_model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)

    best_rmse = float("inf")
    best_r2 = float("-inf")
    best_oob = float("-inf")
    best_index = 0
    best_subset = features.copy()

    results = []
    previous_rmse = None

    while len(features) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X[features], y, test_size=0.3, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        oob = rf_model.oob_score_

        results.append({
            "Step": len(results),
            "Removed Feature": None if len(results) == 0 else least_important,
            "Remaining Features": len(features),
            "Feature List": ", ".join(features),
            "RMSE": rmse,
            "R² Score": r2,
            "OOB Score": oob
        })

        if rmse < best_rmse and oob > best_oob:
            best_rmse = rmse
            best_r2 = r2
            best_oob = oob
            best_subset = features.copy()
            best_index = len(results) - 1

        importances = rf_model.feature_importances_
        least_important = features[np.argmin(importances)]

        # Parameterized early stopping
        if previous_rmse is not None and (rmse - previous_rmse) > early_stop_delta and r2 < best_r2 - early_stop_delta:
            break

        features.remove(least_important)
        previous_rmse = rmse

    # Save results
    result_df = pd.DataFrame(results)
    result_df.loc[0, "Removed Feature"] = "Full Feature Set"

    result_df.to_csv(f"feature_elimination_oob_{target_name}.csv", index=False)
    pd.DataFrame({"Selected Features": best_subset}).to_csv(f"selected_features_oob_{target_name}.csv", index=False)

    # Visualize
    if visualize:
        print(f"\nFinal selected features for {target_name}:")
        display(pd.DataFrame({"Selected Features": best_subset}))
        print(f"\nFinal RMSE: {best_rmse:.4f}, R²: {best_r2:.4f}, OOB: {best_oob:.4f}")
        display(result_df)

        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax1.set_xlabel("Removed Feature")
        ax1.set_ylabel("RMSE", color='blue')
        ax1.plot(result_df["Removed Feature"], result_df["RMSE"], marker='o', color='blue', label='RMSE')
        ax1.tick_params(axis='y', labelcolor='blue')
        plt.xticks(rotation=90)

        ax2 = ax1.twinx()
        ax2.set_ylabel("R² Score", color='orange')
        ax2.plot(result_df["Removed Feature"], result_df["R² Score"], marker='s', linestyle='--', color='orange', label='R² Score')
        ax2.tick_params(axis='y', labelcolor='orange')

        ax1.axvline(x=best_index, color='green', linestyle='dashed', label='Optimal Subset')
        fig.tight_layout()
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
        plt.title(f"Feature Selection Curve for {target_name}")
        plt.grid(True)
        plt.show()

    return best_subset, result_df, best_rmse, best_r2, best_oob


# In[30]:


# Load your balanced dataset
df = pd.read_csv("balanced_dataset.csv")
X = df.drop(columns=["RFL_Att", "FSO_Att", "SYNOPCode"])
y_rf = df["RFL_Att"]
y_fso = df["FSO_Att"]

# Run feature elimination for RFL_Att
selected_rf_features, rf_elim_df, rf_rmse, rf_r2, rf_oob = optimized_bfe_oob_improved(X, y_rf, "RFL_Att")

# Run feature elimination for FSO_Att
selected_fso_features, fso_elim_df, fso_rmse, fso_r2, fso_oob = optimized_bfe_oob_improved(X, y_fso, "FSO_Att")

# Print final feature lists
print("\nFinal Selected Features for RFL_Att:")
print(selected_rf_features)

print("\nFinal Selected Features for FSO_Att:")
print(selected_fso_features)

# Save summary of final metrics
summary_df = pd.DataFrame([{
    "Model": "RFL_Att",
    "Final RMSE": rf_rmse,
    "Final R²": rf_r2,
    "OOB Score": rf_oob,
}, {
    "Model": "FSO_Att",
    "Final RMSE": fso_rmse,
    "Final R²": fso_r2,
    "OOB Score": fso_oob,
}])

summary_df.to_csv("Final_Metrics_Summary.csv", index=False)


# In[31]:


# Subset the balanced dataset using selected features
X_selected_rf = X[selected_rf_features]
X_selected_fso = X[selected_fso_features]

# Train/test split (same split for fair comparison)
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_selected_rf, y_rf, test_size=0.3, random_state=42)
X_train_fso, X_test_fso, y_train_fso, y_test_fso = train_test_split(X_selected_fso, y_fso, test_size=0.3, random_state=42)

# Train models
optimized_rf_model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
optimized_rf_model.fit(X_train_rf, y_train_rf)

optimized_fso_model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
optimized_fso_model.fit(X_train_fso, y_train_fso)

# Evaluate and print
from sklearn.metrics import mean_squared_error, r2_score

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    oob = model.oob_score_
    return rmse, r2, oob

rmse_rf, r2_rf, oob_rf = evaluate(optimized_rf_model, X_test_rf, y_test_rf)
rmse_fso, r2_fso, oob_fso = evaluate(optimized_fso_model, X_test_fso, y_test_fso)

print(f"Optimized RF Model — RMSE: {rmse_rf:.4f}, R²: {r2_rf:.4f}, OOB: {oob_rf:.4f}")
print(f"Optimized FSO Model — RMSE: {rmse_fso:.4f}, R²: {r2_fso:.4f}, OOB: {oob_fso:.4f}")


# In[32]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create summary DataFrame
summary_df = pd.DataFrame([
    {
        "Model": "RFL_Att",
        "Final RMSE": rmse_rf,
        "Final R²": r2_rf,
        "OOB Score": oob_rf
    },
    {
        "Model": "FSO_Att",
        "Final RMSE": rmse_fso,
        "Final R²": r2_fso,
        "OOB Score": oob_fso
    }
])

# Save for reuse later
summary_df.to_csv("final_feature_selection_summary.csv", index=False)

# Melt for Seaborn
plot_df = summary_df.melt(
    id_vars="Model",
    value_vars=["Final RMSE", "Final R²", "OOB Score"],
    var_name="Metric",
    value_name="Score"
)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", palette="Set2")

# Style
plt.title("📊 Final Optimized General Model Performance: RFL_Att vs FSO_Att", fontsize=14)
plt.xlabel("Metric")
plt.ylabel("Score")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend(title="Model", loc="upper right")
plt.tight_layout()
plt.show()


# In[33]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Final, verified comparison values
comparison_df = pd.DataFrame([
    {"Model": "RF (Full)", "Metric": "RMSE", "Score": 0.4901},
    {"Model": "RF (Full)", "Metric": "R² Score", "Score": 0.9797},
    {"Model": "RF (Full)", "Metric": "OOB Score", "Score": 0.9779},

    {"Model": "RF (Optimized)", "Metric": "RMSE", "Score": 0.7799},
    {"Model": "RF (Optimized)", "Metric": "R² Score", "Score": 0.9392},
    {"Model": "RF (Optimized)", "Metric": "OOB Score", "Score": 0.9302},

    {"Model": "FSO (Full)", "Metric": "RMSE", "Score": 0.7845},
    {"Model": "FSO (Full)", "Metric": "R² Score", "Score": 0.9590},
    {"Model": "FSO (Full)", "Metric": "OOB Score", "Score": 0.9562},

    {"Model": "FSO (Optimized)", "Metric": "RMSE", "Score": 0.9627},
    {"Model": "FSO (Optimized)", "Metric": "R² Score", "Score": 0.9664},
    {"Model": "FSO (Optimized)", "Metric": "OOB Score", "Score": 0.9643}
])

# 📊 Barplot
plt.figure(figsize=(12, 6))
sns.barplot(data=comparison_df, x="Metric", y="Score", hue="Model", palette="Set2")

plt.title("📊 General Model Performance: Full vs Optimized (RF & FSO)", fontsize=14)
plt.xlabel("Metric")
plt.ylabel("Score")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend(loc="upper right", title="Model")
plt.tight_layout()
plt.show()


# In[34]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load Balanced Dataset
df = pd.read_csv("balanced_dataset.csv")
X_all = df.drop(columns=["RFL_Att", "FSO_Att", "SYNOPCode"])
y_rf = df["RFL_Att"]
y_fso = df["FSO_Att"]

# Shared Train-Test Split (Ensures identical splits for both targets)
X_train_all, X_test_all, y_train_rf, y_test_rf, y_train_fso, y_test_fso = train_test_split(
    X_all, y_rf, y_fso, test_size=0.2, random_state=42
)

# Load Optimized Feature Sets
rf_features = pd.read_csv("selected_features_oob_RFL_Att.csv")["Selected Features"].tolist()
fso_features = pd.read_csv("selected_features_oob_FSO_Att.csv")["Selected Features"].tolist()

# Training + Evaluation Function
def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    oob = model.oob_score_
    return model, rmse, r2, oob

# Train and Evaluate Models
_, rmse_rf_gen, r2_rf_gen, oob_rf_gen = train_and_evaluate(X_train_all, X_test_all, y_train_rf, y_test_rf)
_, rmse_fso_gen, r2_fso_gen, oob_fso_gen = train_and_evaluate(X_train_all, X_test_all, y_train_fso, y_test_fso)

_, rmse_rf_opt, r2_rf_opt, oob_rf_opt = train_and_evaluate(X_train_all[rf_features], X_test_all[rf_features], y_train_rf, y_test_rf)
_, rmse_fso_opt, r2_fso_opt, oob_fso_opt = train_and_evaluate(X_train_all[fso_features], X_test_all[fso_features], y_train_fso, y_test_fso)

# Print Summary
print("\nGeneral Model Performance (Balanced Dataset)")
print(f"RF   - RMSE: {rmse_rf_gen:.4f}, R²: {r2_rf_gen:.4f}, OOB: {oob_rf_gen:.4f}")
print(f"FSO  - RMSE: {rmse_fso_gen:.4f}, R²: {r2_fso_gen:.4f}, OOB: {oob_fso_gen:.4f}")

print("\nOptimized Model Performance (Balanced Dataset)")
print(f"RF   - RMSE: {rmse_rf_opt:.4f}, R²: {r2_rf_opt:.4f}, OOB: {oob_rf_opt:.4f}")
print(f"FSO  - RMSE: {rmse_fso_opt:.4f}, R²: {r2_fso_opt:.4f}, OOB: {oob_fso_opt:.4f}")

# Create Summary Table
comparison_df = pd.DataFrame({
    "Metric": ["RMSE", "R² Score", "OOB Score"],
    "General RF": [rmse_rf_gen, r2_rf_gen, oob_rf_gen],
    "Optimized RF": [rmse_rf_opt, r2_rf_opt, oob_rf_opt],
    "General FSO": [rmse_fso_gen, r2_fso_gen, oob_fso_gen],
    "Optimized FSO": [rmse_fso_opt, r2_fso_opt, oob_fso_opt],
})

# Save as CSV
comparison_df.to_csv("model_comparison_summary.csv", index=False)

# Prepare Data for Plotting
melted = comparison_df.melt(id_vars="Metric", var_name="Model", value_name="Score")

# Plot Comparison Bar Chart
plt.figure(figsize=(12, 6))
ax = sns.barplot(data=melted, x="Metric", y="Score", hue="Model", palette="Set2")

# Annotate bar values
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", label_type="edge", fontsize=8)

plt.title("General vs Optimized Model Performance (Balanced Dataset)", pad=15)
plt.ylim(0, 1.05)
plt.legend(title="Model", loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=4)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

# Export as PNG
plt.savefig("model_comparison_plot.png", dpi=300)
plt.show()


# ###### Testing And Evaluating The Models On Test Data

# In[35]:


# ✅ Import required packages
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ✅ Load data and selected features
df = pd.read_csv("balanced_dataset.csv")
selected_rf_features = pd.read_csv("selected_features_oob_RFL_Att.csv")["Selected Features"].tolist()
selected_fso_features = pd.read_csv("selected_features_oob_FSO_Att.csv")["Selected Features"].tolist()

# ✅ Define inputs and outputs
X = df.drop(columns=["RFL_Att", "FSO_Att", "SYNOPCode"])
y_rf = df["RFL_Att"]
y_fso = df["FSO_Att"]

# ✅ Shared train-test split
X_train_all, X_test_all, y_train_rf, y_test_rf = train_test_split(X, y_rf, test_size=0.2, random_state=42)
_, _, y_train_fso, y_test_fso = train_test_split(X, y_fso, test_size=0.2, random_state=42)

# ✅ Helper evaluation function
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    oob = model.oob_score_
    return rmse, r2, oob

# ✅ Train and evaluate all 4 models
results = []

# --- General RF (all features)
model_rf_gen = RandomForestRegressor(n_estimators=50, oob_score=True, n_jobs=-1, random_state=42)
model_rf_gen.fit(X_train_all, y_train_rf)
rmse, r2, oob = evaluate(model_rf_gen, X_test_all, y_test_rf)
results.append(["General RF", rmse, r2, oob])

# --- Optimized RF (selected features)
model_rf_opt = RandomForestRegressor(n_estimators=50, oob_score=True, n_jobs=-1, random_state=42)
model_rf_opt.fit(X_train_all[selected_rf_features], y_train_rf)
rmse, r2, oob = evaluate(model_rf_opt, X_test_all[selected_rf_features], y_test_rf)
results.append(["Optimized RF", rmse, r2, oob])

# --- General FSO (all features)
model_fso_gen = RandomForestRegressor(n_estimators=50, oob_score=True, n_jobs=-1, random_state=42)
model_fso_gen.fit(X_train_all, y_train_fso)
rmse, r2, oob = evaluate(model_fso_gen, X_test_all, y_test_fso)
results.append(["General FSO", rmse, r2, oob])

# --- Optimized FSO (selected features)
model_fso_opt = RandomForestRegressor(n_estimators=50, oob_score=True, n_jobs=-1, random_state=42)
model_fso_opt.fit(X_train_all[selected_fso_features], y_train_fso)
rmse, r2, oob = evaluate(model_fso_opt, X_test_all[selected_fso_features], y_test_fso)
results.append(["Optimized FSO", rmse, r2, oob])

# ✅ Display summary
results_df = pd.DataFrame(results, columns=["Model", "RMSE", "R2", "OOB Score"])
print("\nPerformance on Held-Out Test Data:")
print(results_df)


# In[36]:


# ✅ Import required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ✅ Load data and selected features
df = pd.read_csv("balanced_dataset.csv")
selected_rf_features = pd.read_csv("selected_features_oob_RFL_Att.csv")["Selected Features"].tolist()
selected_fso_features = pd.read_csv("selected_features_oob_FSO_Att.csv")["Selected Features"].tolist()

# ✅ Define inputs and outputs
X = df.drop(columns=["RFL_Att", "FSO_Att", "SYNOPCode"])
y_rf = df["RFL_Att"]
y_fso = df["FSO_Att"]

# ✅ Shared train-test split
X_train_all, X_test_all, y_train_rf, y_test_rf = train_test_split(X, y_rf, test_size=0.2, random_state=42)
_, _, y_train_fso, y_test_fso = train_test_split(X, y_fso, test_size=0.2, random_state=42)

# ✅ Helper evaluation function
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    oob = model.oob_score_
    return rmse, r2, oob

# ✅ Train and evaluate all 4 models
results = []

# --- General RF (all features)
model_rf_gen = RandomForestRegressor(n_estimators=50, oob_score=True, n_jobs=-1, random_state=42)
model_rf_gen.fit(X_train_all, y_train_rf)
rmse, r2, oob = evaluate(model_rf_gen, X_test_all, y_test_rf)
results.append(["General RF", rmse, r2, oob])

# --- Optimized RF (selected features)
model_rf_opt = RandomForestRegressor(n_estimators=50, oob_score=True, n_jobs=-1, random_state=42)
model_rf_opt.fit(X_train_all[selected_rf_features], y_train_rf)
rmse, r2, oob = evaluate(model_rf_opt, X_test_all[selected_rf_features], y_test_rf)
results.append(["Optimized RF", rmse, r2, oob])

# --- General FSO (all features)
model_fso_gen = RandomForestRegressor(n_estimators=50, oob_score=True, n_jobs=-1, random_state=42)
model_fso_gen.fit(X_train_all, y_train_fso)
rmse, r2, oob = evaluate(model_fso_gen, X_test_all, y_test_fso)
results.append(["General FSO", rmse, r2, oob])

# --- Optimized FSO (selected features)
model_fso_opt = RandomForestRegressor(n_estimators=50, oob_score=True, n_jobs=-1, random_state=42)
model_fso_opt.fit(X_train_all[selected_fso_features], y_train_fso)
rmse, r2, oob = evaluate(model_fso_opt, X_test_all[selected_fso_features], y_test_fso)
results.append(["Optimized FSO", rmse, r2, oob])

# ✅ Create DataFrame
results_df = pd.DataFrame(results, columns=["Model", "RMSE", "R2", "OOB Score"])
print("\nPerformance on Held-Out Test Data:")
print(results_df)

# ✅ Save to CSV
results_df.to_csv("Final_Model_Test_Performance.csv", index=False)

# ✅ Plot Results
melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=melted, x="Metric", y="Score", hue="Model", palette="Set2")

# Add value labels
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", label_type="edge", fontsize=8)

plt.title("Model Performance on Test Data")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# In[37]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# --- Load Dataset and Selected Features ---
df = pd.read_csv("balanced_dataset.csv")
features_rfl = pd.read_csv("selected_features_oob_RFL_Att.csv")["Selected Features"].tolist()
features_fso = pd.read_csv("selected_features_oob_FSO_Att.csv")["Selected Features"].tolist()

# --- Train-Test Split ---
from sklearn.model_selection import train_test_split

X_rfl = df[features_rfl]
y_rfl = df["RFL_Att"]

X_fso = df[features_fso]
y_fso = df["FSO_Att"]

X_rfl_train, X_rfl_test, y_rfl_train, y_rfl_test = train_test_split(X_rfl, y_rfl, test_size=0.3, random_state=42)
X_fso_train, X_fso_test, y_fso_train, y_fso_test = train_test_split(X_fso, y_fso, test_size=0.3, random_state=42)

# --- Grid Search Parameters ---
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"]
}

# --- Grid Search for Generic RFL ---
grid_rfl = GridSearchCV(RandomForestRegressor(oob_score=True, random_state=42),
                        param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)
grid_rfl.fit(X_rfl_train, y_rfl_train)

best_rfl = grid_rfl.best_estimator_
rfl_pred = best_rfl.predict(X_rfl_test)

# --- Grid Search for Generic FSO ---
grid_fso = GridSearchCV(RandomForestRegressor(oob_score=True, random_state=42),
                        param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)
grid_fso.fit(X_fso_train, y_fso_train)

best_fso = grid_fso.best_estimator_
fso_pred = best_fso.predict(X_fso_test)

# --- Evaluation ---
print("\n🔹 Optimized Generic RFL Model:")
print(f"Best Params: {grid_rfl.best_params_}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_rfl_test, rfl_pred)):.4f}")
print(f"R²: {r2_score(y_rfl_test, rfl_pred):.4f}")
print(f"OOB Score: {best_rfl.oob_score_:.4f}")

print("\n🔹 Optimized Generic FSO Model:")
print(f"Best Params: {grid_fso.best_params_}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_fso_test, fso_pred)):.4f}")
print(f"R²: {r2_score(y_fso_test, fso_pred):.4f}")
print(f"OOB Score: {best_fso.oob_score_:.4f}")


# In[38]:


# === Save Optimized Generic RFL Results ===
results_generic_rfl_opt = pd.DataFrame([{
    "Model": "Optimized Generic RFL",
    "RMSE": np.sqrt(mean_squared_error(y_rfl_test, rfl_pred)),
    "R2": r2_score(y_rfl_test, rfl_pred),
    "OOB": best_rfl.oob_score_,
    "Best_Params": str(grid_rfl.best_params_)
}])
results_generic_rfl_opt.to_csv("optimized_generic_rfl.csv", index=False)
print("✅ Saved: optimized_generic_rfl.csv")

# === Save Optimized Generic FSO Results ===
results_generic_fso_opt = pd.DataFrame([{
    "Model": "Optimized Generic FSO",
    "RMSE": np.sqrt(mean_squared_error(y_fso_test, fso_pred)),
    "R2": r2_score(y_fso_test, fso_pred),
    "OOB": best_fso.oob_score_,
    "Best_Params": str(grid_fso.best_params_)
}])
results_generic_fso_opt.to_csv("optimized_generic_fso.csv", index=False)
print("✅ Saved: optimized_generic_fso.csv")


# In[46]:


import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# === Load the full dataset with labels ===
df = pd.read_csv("balanced_dataset.csv")

# === Load the selected features ===
features_rfl = pd.read_csv("selected_features_oob_RFL_Att.csv")["Selected Features"].tolist()
features_fso = pd.read_csv("selected_features_oob_FSO_Att.csv")["Selected Features"].tolist()

# === Identify unique SYNOP codes ===
weather_types = df["SYNOPCode"].unique()

# === Containers to store results ===
rfl_model_scores = []
fso_model_scores = []

# === Training function ===
def train_rf_for_weather(X, y, weather, label):
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_trn, y_trn)
    y_pred = model.predict(X_tst)
    rmse = mean_squared_error(y_tst, y_pred, squared=False)
    r2 = r2_score(y_tst, y_pred)
    filename = f"rf_model_{label}_weather_{weather}.pkl"
    with open(filename, "wb") as file:
        pickle.dump(model, file)
    return rmse, r2, filename

# === Train RFL_Att Models ===
for synop in weather_types:
    subset = df[df["SYNOPCode"] == synop]
    X_rfl = subset[features_rfl]
    y_rfl = subset["RFL_Att"]
    rmse, r2, model_file = train_rf_for_weather(X_rfl, y_rfl, synop, "RFL_Att")
    rfl_model_scores.append({
        "SYNOPCode": synop,
        "RMSE": rmse,
        "R2": r2,
        "Model_File": model_file
    })

# === Train FSO_Att Models ===
for synop in weather_types:
    subset = df[df["SYNOPCode"] == synop]
    X_fso = subset[features_fso]
    y_fso = subset["FSO_Att"]
    rmse, r2, model_file = train_rf_for_weather(X_fso, y_fso, synop, "FSO_Att")
    fso_model_scores.append({
        "SYNOPCode": synop,
        "RMSE": rmse,
        "R2": r2,
        "Model_File": model_file
    })

# === Save results ===
pd.DataFrame(rfl_model_scores).to_csv("Results_RFL_Att_PerSYNOP.csv", index=False)
pd.DataFrame(fso_model_scores).to_csv("Results_FSO_Att_PerSYNOP.csv", index=False)

print("✅ Per-SYNOP model training complete.")
print("📁 Results saved to 'Results_RFL_Att_PerSYNOP.csv' and 'Results_FSO_Att_PerSYNOP.csv'")


# In[49]:


import pandas as pd
from IPython.display import display

# Load result summaries
df_rfl_results = pd.read_csv("Results_RFL_Att_PerSYNOP.csv")
df_fso_results = pd.read_csv("Results_FSO_Att_PerSYNOP.csv")

# Print and display top results
print("🔹 RFL_Att Model Performance per SYNOP:")
display(df_rfl_results.sort_values(by="RMSE"))

print("\n🔹 FSO_Att Model Performance per SYNOP:")
display(df_fso_results.sort_values(by="RMSE"))


# In[54]:


# Extract unique SYNOP codes
synop_codes = df['SYNOPCode'].unique()

# Store results for each SYNOP code
synop_feature_selection_results = []

# Perform optimized BFE per SYNOP code using your function
for synop in synop_codes:
    df_synop = df[df["SYNOPCode"] == synop].copy()
    X = df_synop.drop(columns=['RFL_Att', 'FSO_Att', 'SYNOPCode'])
    y_rfl = df_synop["RFL_Att"]
    y_fso = df_synop["FSO_Att"]

    print(f"\n🌀 Processing SYNOP Code: {synop} ")

    # Run your optimized BFE for RFL_Att and FSO_Att
    selected_rfl, elim_df_rfl, rmse_rfl, r2_rfl, oob_rfl = optimized_bfe_oob_improved(X, y_rfl, f"RFL_Att_SYNOP_{synop}", visualize=True)
    selected_fso, elim_df_fso, rmse_fso, r2_fso, oob_fso = optimized_bfe_oob_improved(X, y_fso, f"FSO_Att_SYNOP_{synop}", visualize=True)

    # Store result row
    synop_feature_selection_results.append({
        "SYNOP Code": synop,
        "Important Features (RFL_Att)": ", ".join(selected_rfl),
        "RMSE (RFL_Att)": rmse_rfl,
        "R² (RFL_Att)": r2_rfl,
        "Important Features (FSO_Att)": ", ".join(selected_fso),
        "RMSE (FSO_Att)": rmse_fso,
        "R² (FSO_Att)": r2_fso
    })

# Save the summary as a CSV
df_synop_features = pd.DataFrame(synop_feature_selection_results)
df_synop_features.to_csv("Feature_Selection_Per_SYNOP.csv", index=False)
print("\n✅ Saved: Feature_Selection_Per_SYNOP.csv")


# In[56]:


def train_specific_random_forest(X, y, synop_code, target_label):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import pickle

    # Train/test split (SYNOP data already stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    # Save model
    model_name = f"rf_model_{target_label}_SYNOP_{synop_code}.pkl"
    with open(model_name, "wb") as f:
        pickle.dump(model, f)

    return rmse, r2, model_name


# In[57]:


import pandas as pd
from IPython.display import display

# Load the saved summary CSV (if not already in memory)
df_synop_features = pd.read_csv("Feature_Selection_Per_SYNOP.csv")

# Sort by RMSE (lower is better) for both RFL and FSO
print(" RFL_Att Performance Leaderboard (Sorted by RMSE):")
display(df_synop_features.sort_values(by="RMSE (RFL_Att)"))

print("\n FSO_Att Performance Leaderboard (Sorted by RMSE):")
display(df_synop_features.sort_values(by="RMSE (FSO_Att)"))


# In[58]:


# Calculate mean metrics across all SYNOP codes from feature selection
mean_rmse_rfl_fs = df_synop_features["RMSE (RFL_Att)"].mean()
mean_r2_rfl_fs = df_synop_features["R² (RFL_Att)"].mean()
mean_rmse_fso_fs = df_synop_features["RMSE (FSO_Att)"].mean()
mean_r2_fso_fs = df_synop_features["R² (FSO_Att)"].mean()

# Create summary DataFrame
df_mean_metrics_fs = pd.DataFrame({
    "Metric": [
        "Mean RMSE (RFL_Att, Selected Features)",
        "Mean R² (RFL_Att, Selected Features)",
        "Mean RMSE (FSO_Att, Selected Features)",
        "Mean R² (FSO_Att, Selected Features)"
    ],
    "Value": [
        mean_rmse_rfl_fs,
        mean_r2_rfl_fs,
        mean_rmse_fso_fs,
        mean_r2_fso_fs
    ]
})

# Display neatly
print("\n📊 Mean RMSE & R² Across SYNOP Codes (Feature Selection Based):\n")
print(df_mean_metrics_fs.to_string(index=False))

# Display neatly
print("\n📊 Mean RMSE & R² Across SYNOP Codes (Feature Selection Based):\n")
print(df_mean_metrics_fs.to_string(index=False))


# In[65]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load per-SYNOP model scores ===
df_rfl_per_synop = pd.read_csv("Results_RFL_Att_PerSYNOP.csv")
df_fso_per_synop = pd.read_csv("Results_FSO_Att_PerSYNOP.csv")

# === Compute average RMSE and R² per SYNOP ===
mean_rmse_rfl_synop = df_rfl_per_synop["RMSE"].mean()
mean_r2_rfl_synop = df_rfl_per_synop["R2"].mean()
mean_rmse_fso_synop = df_fso_per_synop["RMSE"].mean()
mean_r2_fso_synop = df_fso_per_synop["R2"].mean()

# === Load Optimized Generic Results ===
opt_rfl = pd.read_csv("optimized_generic_rfl.csv")
opt_fso = pd.read_csv("optimized_generic_fso.csv")

opt_rmse_rfl = opt_rfl["RMSE"].values[0]
opt_r2_rfl = opt_rfl["R2"].values[0]
opt_rmse_fso = opt_fso["RMSE"].values[0]
opt_r2_fso = opt_fso["R2"].values[0]

# === Build Comparison Data ===
models = ["Optimized Generic Model", "Average Per-SYNOP Model"]

rmse_rfl = [opt_rmse_rfl, mean_rmse_rfl_synop]
r2_rfl = [opt_r2_rfl, mean_r2_rfl_synop]
rmse_fso = [opt_rmse_fso, mean_rmse_fso_synop]
r2_fso = [opt_r2_fso, mean_r2_fso_synop]

df_compare = pd.DataFrame({
    "Model": models,
    "RMSE (RFL_Att)": rmse_rfl,
    "RMSE (FSO_Att)": rmse_fso,
    "R² (RFL_Att)": r2_rfl,
    "R² (FSO_Att)": r2_fso
})

# === Print Comparison Table ===
print("\n📊 Model Performance: Optimized Generic vs Average Per-SYNOP\n")
print(df_compare.to_string(index=False))

# === Visualization ===
x = np.arange(len(models))
width = 0.35

# RMSE Plot
plt.figure(figsize=(10,6))
plt.bar(x - width/2, rmse_rfl, width, label="RFL_Att RMSE", color="blue", alpha=0.7)
plt.bar(x + width/2, rmse_fso, width, label="FSO_Att RMSE", color="green", alpha=0.7)
plt.xticks(ticks=x, labels=models, rotation=0)
plt.ylabel("RMSE (Lower is Better)")
plt.title("RMSE Comparison: Optimized Generic vs Per-SYNOP")
plt.legend()
plt.tight_layout()
plt.show()

# R² Plot
plt.figure(figsize=(10,6))
plt.bar(x - width/2, r2_rfl, width, label="RFL_Att R²", color="red", alpha=0.7)
plt.bar(x + width/2, r2_fso, width, label="FSO_Att R²", color="purple", alpha=0.7)
plt.xticks(ticks=x, labels=models, rotation=0)
plt.ylabel("R² Score (Higher is Better)")
plt.title("R² Comparison: Optimized Generic vs Per-SYNOP")
plt.legend()
plt.tight_layout()
plt.show()


# In[68]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the correct files
df_rfl_gen = pd.read_csv("optimized_generic_rfl.csv")
df_fso_gen = pd.read_csv("optimized_generic_fso.csv")
df_rfl_spec = pd.read_csv("Results_RFL_Att_PerSYNOP.csv")
df_fso_spec = pd.read_csv("Results_FSO_Att_PerSYNOP.csv")


# SYNOP → Weather mapping
synop_mapping = {
    0: 'Clear', 3: 'Duststorm', 4: 'Fog',
    5: 'Drizzle', 6: 'Rain', 7: 'Snow', 8: 'Showers'
}

# Map weather labels
df_rfl_spec["Weather"] = df_rfl_spec["SYNOPCode"].map(synop_mapping)
df_fso_spec["Weather"] = df_fso_spec["SYNOPCode"].map(synop_mapping)
df_rfl_spec.sort_values("SYNOPCode", inplace=True)
df_fso_spec.sort_values("SYNOPCode", inplace=True)

# Extract per-SYNOP values
weather_labels = df_rfl_spec["Weather"].tolist()
rfl_rmse_spec = df_rfl_spec["RMSE"].tolist()
fso_rmse_spec = df_fso_spec["RMSE"].tolist()
rfl_r2_spec = df_rfl_spec["R2"].tolist()
fso_r2_spec = df_fso_spec["R2"].tolist()

# Generic model RMSE/R2 replicated per weather for comparison
rfl_rmse_gen = [df_rfl_gen["RMSE"].values[0]] * len(weather_labels)
fso_rmse_gen = [df_fso_gen["RMSE"].values[0]] * len(weather_labels)
rfl_r2_gen = [df_rfl_gen["R2"].values[0]] * len(weather_labels)
fso_r2_gen = [df_fso_gen["R2"].values[0]] * len(weather_labels)

# === Plotting ===
fig, axs = plt.subplots(2, 1, figsize=(12, 10))
x = np.arange(len(weather_labels))

# --- RMSE Plot ---
axs[0].plot(x, rfl_rmse_spec, 'r-*', label='RFL – Specific')
axs[0].plot(x, rfl_rmse_gen, 'r--*', label='RFL – Generic')
axs[0].plot(x, fso_rmse_spec, 'g-o', label='FSO – Specific')
axs[0].plot(x, fso_rmse_gen, 'g--o', label='FSO – Generic')
axs[0].set_xticks(x)
axs[0].set_xticklabels(weather_labels, rotation=45)
axs[0].set_ylabel("RMSE (dB)")
axs[0].set_title("(a) RMSE by Weather (Per-SYNOP vs Generic FS)")
axs[0].legend()
axs[0].grid(True)

# --- R² Plot ---
axs[1].plot(x, rfl_r2_spec, 'r-*', label='RFL – Specific')
axs[1].plot(x, rfl_r2_gen, 'r--*', label='RFL – Generic')
axs[1].plot(x, fso_r2_spec, 'g-o', label='FSO – Specific')
axs[1].plot(x, fso_r2_gen, 'g--o', label='FSO – Generic')
axs[1].set_xticks(x)
axs[1].set_xticklabels(weather_labels, rotation=45)
axs[1].set_ylabel("$R^2$")
axs[1].set_ylim(min(rfl_r2_spec + rfl_r2_gen + fso_r2_spec + fso_r2_gen) - 0.01, 1.0)
axs[1].set_title("(b) $R^2$ by Weather (Per-SYNOP vs Generic FS)")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()


# In[70]:


import pandas as pd
import numpy as np

# === Load Model Results ===
df_rfl_gen = pd.read_csv("optimized_generic_rfl.csv")
df_fso_gen = pd.read_csv("optimized_generic_fso.csv")
df_rfl_spec = pd.read_csv("Results_RFL_Att_PerSYNOP.csv")
df_fso_spec = pd.read_csv("Results_FSO_Att_PerSYNOP.csv")

# === Map SYNOP to Weather Labels ===
synop_mapping = {
    0: 'Clear',
    3: 'Duststorm',
    4: 'Fog',
    5: 'Drizzle',
    6: 'Rain',
    7: 'Snow',
    8: 'Showers'
}
synop_codes_sorted = sorted(synop_mapping.keys())
weather_labels = [synop_mapping[code].capitalize() for code in synop_codes_sorted]

# === Sort data by SYNOP code ===
df_rfl_spec = df_rfl_spec.sort_values("SYNOPCode")
df_fso_spec = df_fso_spec.sort_values("SYNOPCode")

# === Helper functions ===
def percent_change(old, new):
    return 100 * (old - new) / old

def absolute_change(old, new):
    return old - new

# === Compute improvements ===
rfl_rmse_gen = [df_rfl_gen["RMSE"].values[0]] * len(synop_codes_sorted)
rfl_r2_gen   = [df_rfl_gen["R2"].values[0]] * len(synop_codes_sorted)
fso_rmse_gen = [df_fso_gen["RMSE"].values[0]] * len(synop_codes_sorted)
fso_r2_gen   = [df_fso_gen["R2"].values[0]] * len(synop_codes_sorted)

rfl_rmse_spec = df_rfl_spec["RMSE"].tolist()
rfl_r2_spec   = df_rfl_spec["R2"].tolist()
fso_rmse_spec = df_fso_spec["RMSE"].tolist()
fso_r2_spec   = df_fso_spec["R2"].tolist()

# === Calculate differences ===
rfl_rmse_diff = [absolute_change(g, s) for g, s in zip(rfl_rmse_gen, rfl_rmse_spec)]
rfl_r2_diff   = [absolute_change(s, g) for g, s in zip(rfl_r2_gen, rfl_r2_spec)]
fso_rmse_diff = [absolute_change(g, s) for g, s in zip(fso_rmse_gen, fso_rmse_spec)]
fso_r2_diff   = [absolute_change(s, g) for g, s in zip(fso_r2_gen, fso_r2_spec)]

rfl_rmse_pct = [percent_change(g, s) for g, s in zip(rfl_rmse_gen, rfl_rmse_spec)]
rfl_r2_pct   = [percent_change(s, g) for g, s in zip(rfl_r2_gen, rfl_r2_spec)]
fso_rmse_pct = [percent_change(g, s) for g, s in zip(fso_rmse_gen, fso_rmse_spec)]
fso_r2_pct   = [percent_change(s, g) for g, s in zip(fso_r2_gen, fso_r2_spec)]

# === Assemble into comparison table ===
table_data = {
    "Metric ↓ / Weather →": weather_labels,
    "RF Att. ∆RMSE":     [f"{v:.3f}" for v in rfl_rmse_diff],
    "RF Att. ∆R²":       [f"{v:.3f}" for v in rfl_r2_diff],
    "RF Att. %∆RMSE":    [f"{v:.1f}%" for v in rfl_rmse_pct],
    "RF Att. %∆R²":      [f"{v:.1f}%" for v in rfl_r2_pct],
    "FSO Att. ∆RMSE":    [f"{v:.3f}" for v in fso_rmse_diff],
    "FSO Att. ∆R²":      [f"{v:.3f}" for v in fso_r2_diff],
    "FSO Att. %∆RMSE":   [f"{v:.1f}%" for v in fso_rmse_pct],
    "FSO Att. %∆R²":     [f"{v:.1f}%" for v in fso_r2_pct]
}

df_table = pd.DataFrame(table_data).set_index("Metric ↓ / Weather →")

# === Display final result table ===
print("\n🔍 Absolute and Percentage Improvements by Specific Model (vs Optimized Generic):\n")
print(df_table.to_string())

# Optional export
df_table.to_csv("Improvement_Per_Synop_Table.csv")


# In[71]:


import pandas as pd

# Load Method 1 model results
rfl_results = pd.read_csv("SYNOP_RFL_Model_Results.csv")
fso_results = pd.read_csv("SYNOP_FSO_Model_Results.csv")

# View column names to confirm structure
print("RFL columns:", rfl_results.columns.tolist())
print("FSO columns:", fso_results.columns.tolist())

# Add Target column to distinguish RFL vs FSO
rfl_results["Target"] = "RFL_Att"
fso_results["Target"] = "FSO_Att"

# Rename correlation score to 'Pearson' for both (whatever the column is)
# Adapt these keys if you see different names printed above
rfl_results = rfl_results.rename(columns={"R2": "Pearson", "R2 Score": "Pearson"})
fso_results = fso_results.rename(columns={"R2": "Pearson", "R2 Score": "Pearson"})

# Rename OOB to unify format
rfl_results = rfl_results.rename(columns={"OOB Score": "OOB"})
fso_results = fso_results.rename(columns={"OOB Score": "OOB"})

# Select only required columns
rfl_clean = rfl_results[["SYNOPCode", "Pearson", "OOB", "Target"]]
fso_clean = fso_results[["SYNOPCode", "Pearson", "OOB", "Target"]]

# Combine into one DataFrame
method1_metrics = pd.concat([rfl_clean, fso_clean], axis=0)
method1_metrics["Method"] = "Method 1"

# Preview
print(method1_metrics.head())

# Save output
method1_metrics.to_csv("Method1_Correlation_OOB_PerWeather.csv", index=False)


# In[72]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load Method 1 metrics ===
pearson_df = pd.read_csv("Method1_Correlation_OOB_PerWeather.csv")  # Contains Pearson
mi_df = pd.read_csv("MI_Method1_PerSYNOP.csv")  # Contains True_MI

# === Merge both ===
merged = pd.merge(pearson_df, mi_df, on=["SYNOPCode", "Target", "Method"])

# === Sort by SYNOPCode for neatness ===
merged = merged.sort_values(by="SYNOPCode")

# === Set up plot ===
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# === Pearson Plot ===
sns.barplot(data=merged, x="SYNOPCode", y="Pearson", hue="Target", ax=axes[0], palette="Blues_d")
axes[0].set_title("Pearson Correlation per SYNOP Code (Method 1)", fontsize=14)
axes[0].set_ylabel("Pearson Correlation")
axes[0].legend(title="Target")
axes[0].grid(True, axis="y", linestyle="--", alpha=0.6)

# === Mutual Information Plot ===
sns.barplot(data=merged, x="SYNOPCode", y="True_MI", hue="Target", ax=axes[1], palette="Greens_d")
axes[1].set_title("Mutual Information per SYNOP Code (Method 1)", fontsize=14)
axes[1].set_ylabel("Mutual Information")
axes[1].legend(title="Target")
axes[1].grid(True, axis="y", linestyle="--", alpha=0.6)

# === Final layout ===
plt.xlabel("SYNOP Code")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[73]:


import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score

# Load Method 1 predictions
rfl_df = pd.read_csv("Test_Predictions_RFL_Att.csv")   # Contains Measured & Predicted RFL_Att
fso_df = pd.read_csv("Test_Predictions_FSO_Att.csv")   # Contains Measured & Predicted FSO_Att

# Define binning size (can tweak)
bin_size = 0.2

def compute_true_mi(measured, predicted, bins):
    hist_2d, _, _ = np.histogram2d(measured, predicted, bins=bins)
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = np.outer(px, py)
    nonzero = pxy > 0
    mi = np.sum(pxy[nonzero] * np.log(pxy[nonzero] / px_py[nonzero]))
    return mi

# Prepare bins for both
bins_rfl = np.arange(rfl_df["Measured_RFL_Att"].min(), rfl_df["Measured_RFL_Att"].max() + bin_size, bin_size)
bins_fso = np.arange(fso_df["Measured_FSO_Att"].min(), fso_df["Measured_FSO_Att"].max() + bin_size, bin_size)

# Loop over SYNOP codes and compute MI
synops = rfl_df["SYNOPCode"].unique()
mi_list = []

for synop in synops:
    # RFL
    rfl_sub = rfl_df[rfl_df["SYNOPCode"] == synop]
    if len(rfl_sub) > 2:
        mi_rfl = compute_true_mi(rfl_sub["Measured_RFL_Att"], rfl_sub["Predicted_RFL_Att"], bins_rfl)
        mi_list.append({"SYNOPCode": synop, "Target": "RFL_Att", "Method": "Method 1", "True_MI": mi_rfl})

    # FSO
    fso_sub = fso_df[fso_df["SYNOPCode"] == synop]
    if len(fso_sub) > 2:
        mi_fso = compute_true_mi(fso_sub["Measured_FSO_Att"], fso_sub["Predicted_FSO_Att"], bins_fso)
        mi_list.append({"SYNOPCode": synop, "Target": "FSO_Att", "Method": "Method 1", "True_MI": mi_fso})

# Save to CSV for final merge
mi_df = pd.DataFrame(mi_list)
mi_df.to_csv("MI_Method1_PerSYNOP.csv", index=False)
print(mi_df.head())


# In[74]:


import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# Load your RFL and FSO per-SYNOP predictions (from Method 1)
rfl_df = pd.read_csv("SYNOP_RFL_Model_Results.csv")
fso_df = pd.read_csv("SYNOP_FSO_Model_Results.csv")

# Create DataFrames with consistent columns
rfl_df_cleaned = rfl_df.rename(columns={
    "R2 Score": "R2", "OOB Score": "OOB"
}).copy()
fso_df_cleaned = fso_df.rename(columns={
    "R2": "R2", "OOB": "OOB"
}).copy()

# Calculate proxy Pearson per SYNOP for both (same as R² for RFs)
rfl_df_cleaned["Pearson"] = np.sqrt(rfl_df_cleaned["R2"].clip(lower=0))  # Avoid complex numbers
fso_df_cleaned["Pearson"] = np.sqrt(fso_df_cleaned["R2"].clip(lower=0))

# Add Target label
rfl_df_cleaned["Target"] = "RFL_Att"
fso_df_cleaned["Target"] = "FSO_Att"

# Combine and label as Method 1
combined = pd.concat([rfl_df_cleaned, fso_df_cleaned])
combined["Method"] = "Method 1"

# Keep only needed columns and save
method1_oob_df = combined[["SYNOPCode", "Target", "R2", "OOB", "Pearson", "Method"]]
method1_oob_df.to_csv("Method_Correlation_OOB_PerWeather.csv", index=False)

print(" Method 1 (Per-SYNOP) OOB + Pearson data saved to 'Method_Correlation_OOB_PerWeather.csv'")


# In[75]:


# --- IMPORTS ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- LOAD MERGED METHOD 1 DATA (Pearson + OOB) ---
method_metrics = pd.read_csv("Method_Correlation_OOB_PerWeather.csv")

# --- LOAD TRUE MI FOR METHOD 1 (Previously calculated) ---
mi_method1_df = pd.read_csv("MI_Method1_PerSYNOP.csv")

# --- MERGE FINAL METHOD 1 DATA ---
final_method1_df = pd.merge(method_metrics, mi_method1_df,
                            on=["SYNOPCode", "Target", "Method"], how="left")

# --- SAVE FINAL METHOD 1 FILE ---
final_method1_df.to_csv("Method_Correlation_MI_PerWeather.csv", index=False)

# --- SPLIT BY TARGET ---
rfl_data = final_method1_df[final_method1_df["Target"] == "RFL_Att"]
fso_data = final_method1_df[final_method1_df["Target"] == "FSO_Att"]

# --- PLOT PEARSON (RFL) ---
plt.figure(figsize=(10, 4))
sns.barplot(x="SYNOPCode", y="Pearson", data=rfl_data)
plt.title("Method 1 – Pearson Correlation by Weather (RFL_Att)")
plt.ylabel("Pearson Correlation")
plt.xlabel("SYNOP Weather Code")
plt.tight_layout()
plt.show()

# --- PLOT TRUE MI (RFL) ---
plt.figure(figsize=(10, 4))
sns.barplot(x="SYNOPCode", y="True_MI", data=rfl_data)
plt.title("Method 1 – True Mutual Information by Weather (RFL_Att)")
plt.ylabel("True Mutual Information")
plt.xlabel("SYNOP Weather Code")
plt.tight_layout()
plt.show()

# --- PLOT PEARSON (FSO) ---
plt.figure(figsize=(10, 4))
sns.barplot(x="SYNOPCode", y="Pearson", data=fso_data)
plt.title("Method 1 – Pearson Correlation by Weather (FSO_Att)")
plt.ylabel("Pearson Correlation")
plt.xlabel("SYNOP Weather Code")
plt.tight_layout()
plt.show()

# --- PLOT TRUE MI (FSO) ---
plt.figure(figsize=(10, 4))
sns.barplot(x="SYNOPCode", y="True_MI", data=fso_data)
plt.title("Method 1 – True Mutual Information by Weather (FSO_Att)")
plt.ylabel("True Mutual Information")
plt.xlabel("SYNOP Weather Code")
plt.tight_layout()
plt.show()


# In[76]:


# === Predict and Save Method 1 Test Predictions (for MI Stability) ===

import pickle
import os

# ✅ Load Balanced Dataset Again (if needed)
balanced_df = pd.read_csv("Balanced_Dataset.csv")  # <-- already exists earlier

# ✅ Prepare storage
method1_predictions = []

# ✅ Loop over each SYNOPCode
for synop_code in sorted(balanced_df["SYNOPCode"].unique()):
    model_filename = f"SYNOP_{synop_code}_RFL_Att.pkl"  # How you saved model filenames earlier
    if not os.path.exists(model_filename):
        continue  # Skip if model file doesn't exist

    subset = balanced_df[balanced_df["SYNOPCode"] == synop_code]

    if len(subset) < 10:
        continue  # Skip if too small

    X = subset[selected_rf_features]  # Selected features for RFL prediction
    y_true_rfl = subset["RFL_Att"]
    y_true_fso = subset["FSO_Att"]

    # Load trained model
    with open(model_filename, "rb") as f:
        model = pickle.load(f)

    # Predict
    preds = model.predict(X)

    # Store true and predicted values
    temp_df = pd.DataFrame({
        "SYNOPCode": subset["SYNOPCode"],
        "Measured_RFL_Att": y_true_rfl,
        "Measured_FSO_Att": y_true_fso,
        "Predicted_RFL_Att": preds
    })
    method1_predictions.append(temp_df)

# ✅ Combine all predictions
method1_predictions_df = pd.concat(method1_predictions)

# ✅ Save for MI Stability Analysis
method1_predictions_df.to_csv("Test_Predictions_Method1.csv", index=False)

print("\n✅ Method 1 predictions saved as Test_Predictions_Method1.csv ready for MI Stability Check!")


# In[77]:


# Method 2 


# In[81]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from IPython.display import display

# === Step 0: Load data and train/test split ===
df = pd.read_csv("balanced_dataset.csv")

X_all = df.drop(columns=["RFL_Att", "FSO_Att"])
y_rfl_all = df["RFL_Att"]
y_fso_all = df["FSO_Att"]

X_train_common, X_test_common, y_train_rfl_common, y_test_rfl_common = train_test_split(
    X_all, y_rfl_all, test_size=0.3, random_state=42
)
_, _, y_train_fso_common, y_test_fso_common = train_test_split(
    X_all, y_fso_all, test_size=0.3, random_state=42
)

# === Step 1: Train Generic RF1 Model (RFL_Att) ===
selected_features_rfl = pd.read_csv("selected_features_oob_RFL_Att.csv")["Selected Features"].tolist()
rf_generic_rfl_fs = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
rf_generic_rfl_fs.fit(X_train_common[selected_features_rfl], y_train_rfl_common)

# === Step 2: Inject RF1 Predictions into Train/Test for RF2 ===
X_train_rf2 = X_train_common.copy()
X_test_rf2 = X_test_common.copy()

X_train_rf2["Predicted_RFL_Att"] = rf_generic_rfl_fs.predict(X_train_common[selected_features_rfl])
X_test_rf2["Predicted_RFL_Att"] = rf_generic_rfl_fs.predict(X_test_common[selected_features_rfl])

# === Step 3: Define and run optimized BFE ===
def optimized_bfe_oob_improved(X, y, target_name="Target", visualize=True, early_stop_delta=0.01):
    features = list(X.columns)
    best_rmse = float("inf")
    best_r2 = float("-inf")
    best_oob = float("-inf")
    best_index = 0
    best_subset = features.copy()
    results = []
    previous_rmse = None

    while len(features) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X[features], y, test_size=0.3, random_state=42)
        rf = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        oob = rf.oob_score_
        least_important = features[np.argmin(rf.feature_importances_)]
        results.append({
            "Step": len(results),
            "Removed Feature": least_important if len(results) > 0 else "Full Set",
            "Remaining Features": len(features),
            "Feature List": ", ".join(features),
            "RMSE": rmse,
            "R² Score": r2,
            "OOB Score": oob
        })

        if rmse < best_rmse and oob > best_oob:
            best_rmse = rmse
            best_r2 = r2
            best_oob = oob
            best_subset = features.copy()
            best_index = len(results) - 1

        if previous_rmse is not None and (rmse - previous_rmse) > early_stop_delta and r2 < best_r2 - early_stop_delta:
            break

        features.remove(least_important)
        previous_rmse = rmse

    result_df = pd.DataFrame(results)
    result_df.to_csv(f"feature_elimination_oob_{target_name}.csv", index=False)
    pd.DataFrame({"Selected Features": best_subset}).to_csv(f"selected_features_oob_{target_name}.csv", index=False)

    if visualize:
        print(f"\nFinal selected features for {target_name}:")
        display(pd.DataFrame({"Selected Features": best_subset}))
        print(f"\nFinal RMSE: {best_rmse:.4f}, R²: {best_r2:.4f}, OOB: {best_oob:.4f}")
        display(result_df)

        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax1.set_xlabel("Removed Feature")
        ax1.set_ylabel("RMSE", color='blue')
        ax1.plot(result_df["Removed Feature"], result_df["RMSE"], marker='o', color='blue', label='RMSE')
        ax1.tick_params(axis='y', labelcolor='blue')
        plt.xticks(rotation=90)

        ax2 = ax1.twinx()
        ax2.set_ylabel("R² Score", color='orange')
        ax2.plot(result_df["Removed Feature"], result_df["R² Score"], marker='s', linestyle='--', color='orange', label='R² Score')
        ax2.tick_params(axis='y', labelcolor='orange')

        ax1.axvline(x=best_index, color='green', linestyle='dashed', label='Optimal Subset')
        fig.tight_layout()
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
        plt.title(f"Feature Selection Curve for {target_name}")
        plt.grid(True)
        plt.show()

    return best_subset, result_df, best_rmse, best_r2, best_oob

# === Step 4: Run BFE on RF2 (FSO_Att using weather + predicted RFL_Att) ===
y_train_rf2 = y_train_fso_common.copy()
selected_features_rf2, bfe_df_rf2, rmse_rf2, r2_rf2, oob_rf2 = optimized_bfe_oob_improved(
    X_train_rf2, y_train_rf2, target_name="FSO_Att_Method2", visualize=True
)

# === Step 5: Retrain final RF2 model ===
rf_rf2_final = RandomForestRegressor(oob_score=True, random_state=42)
rf_rf2_final.fit(X_train_rf2[selected_features_rf2], y_train_rf2)

# === Step 6: Predict on test set ===
y_pred_fso_m2 = rf_rf2_final.predict(X_test_rf2[selected_features_rf2])

# === Step 7: Evaluate Method 2 ===
rmse_fso_m2 = np.sqrt(mean_squared_error(y_test_fso_common, y_pred_fso_m2))
r2_fso_m2 = r2_score(y_test_fso_common, y_pred_fso_m2)
pearson_corr_m2 = pearsonr(X_test_rf2["Predicted_RFL_Att"], y_pred_fso_m2)[0]
mutual_info_m2 = mutual_info_regression(y_test_fso_common.values.reshape(-1, 1), y_pred_fso_m2)[0]

# === Step 8: Print Results ===
print("\n✅ Method 2 with Optimized BFE (FSO ← RFL + Weather):")
print(f"Selected Features: {selected_features_rf2}")
print(f"RMSE:           {rmse_fso_m2:.4f}")
print(f"R² Score:       {r2_fso_m2:.4f}")
print(f"Pearson Corr:   {pearson_corr_m2:.4f}")
print(f"Mutual Info:    {mutual_info_m2:.4f}")
print(f"OOB Score:      {rf_rf2_final.oob_score_:.4f}")

# === Step 9: Save predictions to CSV ===
df_pred_m2 = X_test_rf2.copy()
df_pred_m2["Measured_FSO_Att"] = y_test_fso_common.values
df_pred_m2["Predicted_FSO_Att"] = y_pred_fso_m2
df_pred_m2.to_csv("Test_Predictions_FSO_Att_Method2.csv", index=False)
print("✅ Saved: Test_Predictions_FSO_Att_Method2.csv")



# In[83]:


# Correct scatter plot for Method 2 (FSO ← RFL + Weather + BFE)
plt.figure(figsize=(8, 6))
plt.scatter(y_test_fso_common, y_pred_fso_m2, alpha=0.5, color="blue", label="Predicted vs Measured")

# Plot ideal fit line
plt.plot(
    [y_test_fso_common.min(), y_test_fso_common.max()],
    [y_test_fso_common.min(), y_test_fso_common.max()],
    "r--", label="Perfect Fit"
)

plt.xlabel("Measured FSO_Att")
plt.ylabel("Predicted FSO_Att")
plt.title("Measured vs Predicted FSO_Att (Method 2)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# In[85]:


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# === Step 1: Compute Method 2 (Predicted_RFL + Weather) metrics directly ===
rmse_fso_step2 = np.sqrt(mean_squared_error(y_test_fso_common, y_pred_fso_m2))
r2_fso_step2 = r2_score(y_test_fso_common, y_pred_fso_m2)

# === Step 2: Load Optimized General FSO Model metrics from file ===
general_fso_df = pd.read_csv("optimized_generic_fso.csv")
rmse_fso_general = general_fso_df["RMSE"].values[0]
r2_fso_general = general_fso_df["R2"].values[0]

# === Step 3: Print performance comparison ===
print("\n📊 Performance Comparison: General vs. Method 2 (BFE on FSO ← RFL + Weather)")
print(f"General Model (Optimized FSO) - RMSE: {rmse_fso_general:.4f}, R²: {r2_fso_general:.4f}")
print(f"Method 2 (Predicted RFL + BFE) - RMSE: {rmse_fso_step2:.4f}, R²: {r2_fso_step2:.4f}")

# === Step 4: Simple verdict ===
rmse_diff = rmse_fso_general - rmse_fso_step2
r2_diff = r2_fso_step2 - r2_fso_general

if rmse_diff > 0 and r2_diff > 0:
    print("\n✅ Using Predicted_RFL_Att with BFE improved the prediction of FSO_Att.")
else:
    print("\n⚠️ Using Predicted_RFL_Att with BFE did not improve the prediction. Further tuning may help.")


# In[86]:


import matplotlib.pyplot as plt
import seaborn as sns

# === Scatter plot: Measured vs. Predicted FSO_Att ===
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=y_test_fso_common,
    y=y_pred_fso_m2,
    alpha=0.5,
    color="blue",
    label="Predicted vs Measured"
)

# Add ideal fit reference line
min_val = min(y_test_fso_common.min(), y_pred_fso_m2.min())
max_val = max(y_test_fso_common.max(), y_pred_fso_m2.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="red", label="Ideal Fit")

# Labels and layout
plt.xlabel("Measured FSO_Att")
plt.ylabel("Predicted FSO_Att")
plt.title("Measured vs. Predicted FSO_Att (Method 2 – Optimized BFE)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# In[87]:


# Merge predicted FSO_Att from Method 2 with original test set
df_pred = X_test_common.copy()
df_pred["Measured_FSO_Att"] = y_test_fso_common.values
df_pred["Predicted_FSO_Att"] = y_pred_fso_m2

# Save updated test dataset with predictions
df_pred.to_csv("Updated_Dataset_Method2.csv", index=False)
print("✅ Updated Dataset with 'Predicted_FSO_Att' saved as 'Updated_Dataset_Method2.csv'")


# In[88]:


from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Evaluate predictions
rmse_fso_method2 = np.sqrt(mean_squared_error(y_test_fso_common, y_pred_fso_m2))
r2_fso_method2 = r2_score(y_test_fso_common, y_pred_fso_m2)

# Print results
print("\n📊 FSO_Att Model Performance (Method 2 - Optimized BFE):")
print(f"  • RMSE:     {rmse_fso_method2:.4f}")
print(f"  • R² Score: {r2_fso_method2:.4f}")


# In[89]:


from scipy.stats import pearsonr
import pandas as pd

# Merge predictions with SYNOP codes
df_corr = X_test_common.copy()
df_corr["Measured_FSO_Att"] = y_test_fso_common.values
df_corr["Predicted_FSO_Att"] = y_pred_fso_m2

# Compute Pearson correlation per SYNOP
pearson_list = []
for synop in sorted(df_corr["SYNOPCode"].unique()):
    sub_df = df_corr[df_corr["SYNOPCode"] == synop]
    if len(sub_df) > 5:  # avoid small groups
        r_val, _ = pearsonr(sub_df["Measured_FSO_Att"], sub_df["Predicted_FSO_Att"])
        pearson_list.append({
            "SYNOPCode": synop,
            "Target": "FSO_Att",
            "Pearson": r_val,
            "Method": "Method 2"
        })

# Save results
pearson_df = pd.DataFrame(pearson_list)
pearson_df.to_csv("Pearson_Correlation_PerSYNOP_Method2.csv", index=False)
pearson_df.head()


# In[90]:


from sklearn.metrics import mutual_info_score
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Merge predictions with SYNOP codes
df_mi = X_test_common.copy()
df_mi["Measured_FSO_Att"] = y_test_fso_common.values
df_mi["Predicted_FSO_Att"] = y_pred_fso_m2

# Define bin size
bin_size = 0.2
bins_fso = np.arange(
    df_mi[["Measured_FSO_Att", "Predicted_FSO_Att"]].min().min(),
    df_mi[["Measured_FSO_Att", "Predicted_FSO_Att"]].max().max() + bin_size,
    bin_size
)

# Initialize list
mi2_list = []

# Loop through each SYNOP
for synop in sorted(df_mi["SYNOPCode"].unique()):
    sub_df = df_mi[df_mi["SYNOPCode"] == synop]
    if len(sub_df) > 5:
        y_true = sub_df["Measured_FSO_Att"].values
        y_pred = sub_df["Predicted_FSO_Att"].values

        # Discretize
        bins = bins_fso
        px = np.histogram(y_true, bins=bins)[0] / len(y_true)
        py = np.histogram(y_pred, bins=bins)[0] / len(y_pred)
        joint = np.histogram2d(y_true, y_pred, bins=(bins, bins))[0] / len(y_true)

        # Calculate MI
        nonzero = joint > 0
        mi = np.sum(joint[nonzero] * np.log(joint[nonzero] / (px[:, None] * py[None, :] + 1e-12)[nonzero]))

        mi2_list.append({
            "SYNOPCode": synop,
            "Target": "FSO_Att",
            "MI": mi,
            "Method": "Method 2",
            "Pearson": pearsonr(y_true, y_pred)[0]
        })

# Save to CSV
df_mi2 = pd.DataFrame(mi2_list)
df_mi2.to_csv("Method2_Correlation_MI_PerWeather.csv", index=False)
print(df_mi2.head())


# In[91]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

results = []

# Merge test predictions with SYNOP codes
df_corr = X_test_common.copy()
df_corr["Measured_FSO_Att"] = y_test_fso_common.values
df_corr["Predicted_FSO_Att"] = y_pred_fso_m2

# Compute R² per SYNOP
for synop in sorted(df_corr["SYNOPCode"].unique()):
    sub = df_corr[df_corr["SYNOPCode"] == synop]
    if len(sub) > 5:
        r_meas = r2_score(sub["Measured_FSO_Att"], sub["Measured_FSO_Att"])  # always 1.0
        r_modeled = r2_score(sub["Measured_FSO_Att"], sub["Predicted_FSO_Att"])
        results.append({"SYNOPCode": synop, "r_measured": r_meas, "r_modeled": r_modeled})

# Save to CSV
corr_df = pd.DataFrame(results)
corr_df.to_csv("R2_Measured_vs_Modeled_Correlation_Method2.csv", index=False)
print(corr_df.head())


# In[93]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# === Step 1: Load data ===
df = pd.read_csv("Measured_vs_Modeled_Correlation_Method2.csv")

# === Step 2: Map SYNOP codes to readable weather labels ===
synop_mapping = {
    0: "Clear", 3: "Duststorm", 4: "Fog",
    5: "Drizzle", 6: "Rain", 7: "Snow", 8: "Showers"
}
df["Weather"] = df["SYNOPCode"].map(synop_mapping)

# === Step 3: Melt for seaborn barplot (long format) ===
df_melted = df.melt(
    id_vars=["SYNOPCode", "Weather"],
    value_vars=["r_measured", "r_modeled"],
    var_name="Type",
    value_name="Correlation"
)

# === Step 4: Create bar plot ===
plt.figure(figsize=(12, 6))
ax = sns.barplot(
    x="Weather", y="Correlation", hue="Type", data=df_melted,
    palette="Set2"
)

# === Step 5: Add interpretation bands ===
plt.axhspan(0.0, 0.3, color='gray', alpha=0.15, label="Negligible")
plt.axhspan(0.3, 0.5, color='blue', alpha=0.1, label="Low")
plt.axhspan(0.5, 0.7, color='green', alpha=0.1, label="Moderate")
plt.axhspan(0.7, 1.0, color='orange', alpha=0.1, label="High")

# === Step 6: Annotate bars safely ===
for bar in ax.patches:
    if isinstance(bar, mpatches.Rectangle) and bar.get_y() == 0:
        height = bar.get_height()
        if height != 0:
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5 if height >= 0 else -12),
                        textcoords='offset points',
                        ha='center', va='center', fontsize=9, color='black')

# === Step 7: Final formatting ===
plt.axhline(0.0, color="black", linestyle="--", linewidth=0.7)
plt.title("Correlation by Weather Type (Method 2)\nMeasured vs Modeled with Interpretation Zones", fontsize=14)
plt.xlabel("Weather Condition")
plt.ylabel("Pearson Correlation")
plt.legend(title="Correlation Type", loc="lower right")
plt.tight_layout()
plt.show()


# In[94]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from scipy.stats import pearsonr

# Load Method 2 predictions (Predicted + Measured FSO)
df = pd.read_csv("Test_Predictions_FSO_Att.csv")

# Map SYNOPCode to human-readable weather labels
synop_mapping = {
    0: "Clear",
    3: "Duststorm",
    4: "Fog",
    5: "Drizzle",
    6: "Rain",
    7: "Snow",
    8: "Showers"
}
df["Weather"] = df["SYNOPCode"].map(synop_mapping)

# Compute Pearson correlation per weather condition
correlation_list = []
for weather in df["Weather"].dropna().unique():
    sub_df = df[df["Weather"] == weather]
    if len(sub_df) >= 2:
        r_val = pearsonr(sub_df["Measured_FSO_Att"], sub_df["Predicted_FSO_Att"])[0]
        correlation_list.append({
            "Weather": weather,
            "r_measured": 1.0,
            "r_modeled": r_val
        })

corr_df = pd.DataFrame(correlation_list)

# Melt dataframe for barplot
df_melted = corr_df.melt(
    id_vars=["Weather"],
    value_vars=["r_measured", "r_modeled"],
    var_name="Type",
    value_name="Correlation"
)

# Plot grouped bar chart with shaded interpretation zones
plt.figure(figsize=(12, 6))
ax = sns.barplot(x="Weather", y="Correlation", hue="Type", data=df_melted)

# Add shaded correlation strength bands
plt.axhspan(0.0, 0.3, color='gray', alpha=0.15, label="Negligible")
plt.axhspan(0.3, 0.5, color='blue', alpha=0.1, label="Low")
plt.axhspan(0.5, 0.7, color='green', alpha=0.1, label="Moderate")
plt.axhspan(0.7, 1.0, color='orange', alpha=0.1, label="High")

# Annotate bars
for bar in ax.patches:
    if isinstance(bar, plt.Rectangle) and bar.get_height() != 0:
        height = bar.get_height()
        ax.annotate(
            f'{height:.2f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5 if height >= 0 else -12),
            textcoords='offset points',
            ha='center', va='center',
            fontsize=9, color='black'
        )

# Final formatting
plt.axhline(0.0, color='black', linestyle='--', linewidth=0.8)
plt.title("Correlation by Weather Type (Method 2)\nMeasured vs Modeled with Interpretation Zones", fontsize=14)
plt.xlabel("Weather Condition")
plt.ylabel("Pearson Correlation")
plt.legend(title="Correlation Type", loc='lower right')
plt.tight_layout()
plt.show()


# In[105]:


import pandas as pd
import numpy as np

# Load Method 1 (Specific) per-SYNOP RMSE
df_rfl_spec = pd.read_csv("SYNOP_RFL_Model_Results.csv")[["SYNOPCode", "RMSE"]].rename(columns={"RMSE": "RMSE_Method1"})
df_fso_spec = pd.read_csv("SYNOP_FSO_Model_Results.csv")[["SYNOPCode", "RMSE"]].rename(columns={"RMSE": "RMSE_Method1"})

# Load Method 2 (Generic → Predicted RFL → Predict FSO)
df_rfl_m2 = pd.read_csv("Test_Predictions_RFL_Att.csv")
df_fso_m2 = pd.read_csv("Test_Predictions_FSO_Att.csv")

# Compute RMSE per SYNOP for Method 2
rfl_m2_rmse = df_rfl_m2.groupby("SYNOPCode").apply(
    lambda g: np.sqrt(np.mean((g["Measured_RFL_Att"] - g["Predicted_RFL_Att"])**2))
).reset_index(name="RMSE_Method2")

fso_m2_rmse = df_fso_m2.groupby("SYNOPCode").apply(
    lambda g: np.sqrt(np.mean((g["Measured_FSO_Att"] - g["Predicted_FSO_Att"])**2))
).reset_index(name="RMSE_Method2")

# Label and merge
rfl_compare = pd.merge(df_rfl_spec, rfl_m2_rmse, on="SYNOPCode")
rfl_compare["Target"] = "RFL_Att"

fso_compare = pd.merge(df_fso_spec, fso_m2_rmse, on="SYNOPCode")
fso_compare["Target"] = "FSO_Att"

# Combine and save
df_compare_all = pd.concat([rfl_compare, fso_compare], ignore_index=True)
df_compare_all.to_csv("Comparison_RMSE_PerMethod.csv", index=False)

print("✅ Saved: Comparison_RMSE_PerMethod.csv")
display(df_compare_all.head())


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load final correlation + MI data
corr_df = pd.read_csv("Method23_Correlation_MI_PerWeather.csv")

# Keep only Method 1 and Method 2 (if not already done in file)
corr_df = corr_df[corr_df["Method"].isin(["Method 1", "Method 2"])]

# Create method-specific label for plotting
corr_df["Method_Label"] = corr_df["Method"] + " (" + corr_df["Target"] + ")"

# Define metrics to plot
metrics = ["Pearson", "True_MI"]
metric_labels = {"Pearson": "Pearson Correlation", "True_MI": "True Mutual Information"}

# Loop over each metric and create radar plot
for metric in metrics:
    plt.figure(figsize=(8, 8), dpi=100)
    
    # Pivot table for radar shape: rows=SYNOP, columns=Method+Target
    radar_data = corr_df.pivot(index="SYNOPCode", columns="Method_Label", values=metric)

    # Define radar angles
    labels = radar_data.index.astype(str).tolist()
    angles = [n / float(len(labels)) * 2 * 3.14159 for n in range(len(labels))]
    angles += angles[:1]  # Loop back to start

    # Start radar plot
    ax = plt.subplot(111, polar=True)

    for col in radar_data.columns:
        values = radar_data[col].tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, label=col, linewidth=2)
        ax.fill(angles, values, alpha=0.1)

    # Radar aesthetics
    ax.set_theta_offset(3.14159 / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids([a * 180 / 3.14159 for a in angles[:-1]], labels)
    plt.title(f"Radar Plot – {metric_labels[metric]} by Method and Weather Type", fontsize=13)
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    plt.show()


# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load prediction files
df_rfl = pd.read_csv("Test_Predictions_RFL_Att.csv")    # Contains Measured_RFL_Att
df_fso = pd.read_csv("Test_Predictions_FSO_Att.csv")    # Contains Measured_FSO_Att and SYNOPCode

# Merge on index, keep required columns
df = pd.merge(
    df_rfl[["Measured_RFL_Att"]],
    df_fso[["Measured_FSO_Att", "SYNOPCode"]],
    left_index=True, right_index=True
)

# Map SYNOPCode to weather label
synop_mapping = {
    0: 'Clear',
    3: 'Duststorm',
    4: 'Fog',
    5: 'Drizzle',
    6: 'Rain',
    7: 'Snow',
    8: 'Showers'
}
df["Weather"] = df["SYNOPCode"].map(synop_mapping)

# Define bin edges
rfl_bins = np.arange(0, 40, 5)   # 0–5, 5–10, ..., 35–40
fso_bins = np.arange(0, 50, 5)   # 0–5, 5–10, ..., 45–50

# Create bin labels
x_labels = [f"{rfl_bins[i]}–{rfl_bins[i+1]}" for i in range(len(rfl_bins) - 1)]
y_labels = [f"{fso_bins[i]}–{fso_bins[i+1]}" for i in range(len(fso_bins) - 1)]

# Generate heatmaps per weather condition
for weather in df["Weather"].dropna().unique():
    sub_df = df[df["Weather"] == weather].copy()

    # Digitize RFL and FSO into bins
    sub_df["RFL_bin"] = np.digitize(sub_df["Measured_RFL_Att"], rfl_bins) - 1
    sub_df["FSO_bin"] = np.digitize(sub_df["Measured_FSO_Att"], fso_bins) - 1

    # Compute bin frequency matrix
    heat_matrix = pd.crosstab(sub_df["FSO_bin"], sub_df["RFL_bin"])

    # Reindex to ensure full grid
    heat_matrix = heat_matrix.reindex(index=range(len(y_labels)), columns=range(len(x_labels)), fill_value=0)
    heat_matrix.index = y_labels
    heat_matrix.columns = x_labels

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(heat_matrix, annot=True, fmt='d', cmap="YlGnBu", cbar_kws={'label': 'Pair Count'})
    plt.title(f"{weather} – RF vs FSO Attenuation Bin Heatmap")
    plt.xlabel("RF Attenuation Bin (dB)")
    plt.ylabel("FSO Attenuation Bin (dB)")
    plt.tight_layout()
    plt.show()


# In[10]:


import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr

# Load Method 2 Predictions
fso_predictions_df = pd.read_csv("Test_Predictions_FSO_Att.csv")

# Extract Actual & Predicted Values
y_true = fso_predictions_df["Measured_FSO_Att"]
y_pred = fso_predictions_df["Predicted_FSO_Att"]

# Compute Pearson Correlation
pearson_corr, _ = pearsonr(y_true, y_pred)

# Compute Mutual Information
mi_score = mutual_info_regression(y_true.values.reshape(-1, 1), y_pred)[0]

# Display Results
print("Correlation Analysis (FSO prediction using Predicted RFL_Att):")
print(f"Pearson Correlation: {pearson_corr:.4f}")
print(f"Mutual Information: {mi_score:.4f}")

# Save Results to CSV
correlation_results_df = pd.DataFrame([{
    "Method": "Method 2 (FSO using Predicted RFL_Att)",
    "Pearson Correlation": round(pearson_corr, 4),
    "Mutual Information": round(mi_score, 4)
}])

correlation_results_df.to_csv("Correlation_MI_Method2.csv", index=False)
print("Saved Method 2 correlation/MI results to: 'Correlation_MI_Method2.csv'")


# In[11]:


# Define correlation threshold (e.g., 0.7)
correlation_threshold = 0.7

# Evaluate Method 2 based on Pearson correlation
if pearson_corr > correlation_threshold:
    print("\nMethod 2 appears to perform well based on correlation.")
else:
    print("\nWarning: Correlation is weak. Consider using Method 1 (per-weather specific models) for better accuracy.")


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Load performance metrics
general_fso_results = pd.read_csv("Optimized_Model_Performance.csv")
fso_predictions_df = pd.read_csv("Test_Predictions_FSO_Att.csv")

# Extract RMSE and R² from general FSO model
rmse_fso_general = general_fso_results.loc[general_fso_results["Model"] == "Optimized FSO", "RMSE"].values[0]
r2_fso_general = general_fso_results.loc[general_fso_results["Model"] == "Optimized FSO", "R2 Score"].values[0]

# Compute RMSE and R² for Step 2 (Method 2)
y_true = fso_predictions_df["Measured_FSO_Att"]
y_pred = fso_predictions_df["Predicted_FSO_Att"]
rmse_fso = np.sqrt(mean_squared_error(y_true, y_pred))
r2_fso = r2_score(y_true, y_pred)

# Scatter Plot: Measured vs Predicted FSO_Att (Step 2)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
plt.axline((0, 0), slope=1, linestyle="dashed", color="red", label="Ideal Fit")
plt.xlabel("Measured FSO_Att")
plt.ylabel("Predicted FSO_Att")
plt.title("Measured vs Predicted FSO_Att (Step 2 Model)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Bar Plot: RMSE and R² Comparison
performance_comparison_df = pd.DataFrame({
    "Model": ["General FSO Model", "Step 2 FSO Model"],
    "RMSE": [rmse_fso_general, rmse_fso],
    "R² Score": [r2_fso_general, r2_fso]
})

performance_melted = performance_comparison_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(8, 5))
sns.barplot(x="Metric", y="Score", hue="Model", data=performance_melted, palette="pastel", edgecolor="black")

# Annotate bars
for bar in plt.gca().patches:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.02,
        f"{height:.2f}",
        ha='center', va='bottom', fontsize=9
    )

plt.title("Performance Comparison: General vs Step 2 FSO Model")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.legend(title="Model Type")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[19]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load final Method 1 metrics from feature selection summary
method1_df = pd.read_csv("final_feature_selection_summary.csv")

# Load Method 2 optimized results
method2_results = pd.read_csv("Optimized_Model_Performance.csv")
method2_fso = method2_results[method2_results["Model"] == "Optimized FSO"].iloc[0]

# Build unified comparison table
comparison_df = pd.DataFrame({
    "Metric": ["RMSE", "R² Score", "OOB Score"],
    "Method 1 (Per-SYNOP)": [
        method1_df.loc[method1_df["Model"] == "FSO_Att", "Final RMSE"].values[0],
        method1_df.loc[method1_df["Model"] == "FSO_Att", "Final R²"].values[0],
        method1_df.loc[method1_df["Model"] == "FSO_Att", "OOB Score"].values[0]
    ],
    "Method 2 (Hybrid)": [
        method2_fso["RMSE"],
        method2_fso["R2 Score"],
        method2_fso["OOB Score"]
    ]
})

# Save to CSV
comparison_df.to_csv("Model_Comparison_Method1_vs_Method2.csv", index=False)
print("✅ Comparison Table Saved as 'Model_Comparison_Method1_vs_Method2.csv'")
print(comparison_df)

# Plot
melted_df = comparison_df.melt(id_vars="Metric", var_name="Method", value_name="Score")
plt.figure(figsize=(10, 5))
sns.barplot(x="Metric", y="Score", hue="Method", data=melted_df, palette="Set2")
plt.title("📊 Method 1 vs Method 2: FSO_Att Performance Metrics")
plt.ylabel("Score")
plt.xlabel("Metric")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


# In[20]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

# Load Dataset
df = pd.read_csv("Balanced_Dataset.csv")

# Define Features & Target for Analysis
X = df.drop(columns=["FSO_Att", "RFL_Att", "SYNOPCode"])
y_fso = df["FSO_Att"]

# Compute Pearson Correlation for Each Feature
pearson_corr = {feature: pearsonr(df[feature], y_fso)[0] for feature in X.columns}
pearson_df = pd.DataFrame(list(pearson_corr.items()), columns=["Feature", "Pearson Correlation"])
pearson_df = pearson_df.sort_values(by="Pearson Correlation", ascending=False)

# Compute Mutual Information Score for Each Feature
mutual_info_scores = mutual_info_regression(X, y_fso)
mutual_info_df = pd.DataFrame({"Feature": X.columns, "Mutual Information": mutual_info_scores})
mutual_info_df = mutual_info_df.sort_values(by="Mutual Information", ascending=False)

# Train Random Forest Model to Extract Feature Importance
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y_fso)
feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": rf_model.feature_importances_})
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

# Save Results for Reference
pearson_df.to_csv("Feature_Pearson_Correlation.csv", index=False)
mutual_info_df.to_csv("Feature_Mutual_Information.csv", index=False)
feature_importances.to_csv("Feature_Importance.csv", index=False)

# Plot Results
plt.figure(figsize=(15, 5))

# Plot Pearson Correlation
plt.subplot(1, 3, 1)
sns.barplot(y=pearson_df["Feature"], x=pearson_df["Pearson Correlation"], palette="Blues_r")
plt.title("Pearson Correlation of Features with FSO_Att")

# Plot Mutual Information
plt.subplot(1, 3, 2)
sns.barplot(y=mutual_info_df["Feature"], x=mutual_info_df["Mutual Information"], palette="Greens_r")
plt.title("Mutual Information of Features with FSO_Att")

# Plot Feature Importance
plt.subplot(1, 3, 3)
sns.barplot(y=feature_importances["Feature"], x=feature_importances["Importance"], palette="Oranges_r")
plt.title("Feature Importance in Random Forest Model")

plt.tight_layout()
plt.show()

print("\nFeature Interpretability Analysis Complete. Results saved.")


# In[22]:


print(method1_df.columns)


# In[24]:


import pandas as pd

# Load Method 1 (Specific per-SYNOP RFL models)
method1_df = pd.read_csv("SYNOP_RFL_Model_Results.csv")

# Load Method 2 (Generic RFL model)
generic_results = pd.read_csv("Final_Model_Test_Performance.csv")

# Extract Method 2 metrics
generic_rfl_r2 = generic_results.loc[generic_results["Model"] == "General RF", "R2"].values[0]
generic_rfl_rmse = generic_results.loc[generic_results["Model"] == "General RF", "RMSE"].values[0]

# Compute Method 1 averages
method1_avg_r2 = method1_df["R2"].mean()
method1_avg_rmse = method1_df["RMSE"].mean()

# Create summary DataFrame
comparison_df = pd.DataFrame({
    "Metric": ["Average_R2", "Average_RMSE"],
    "Method 1 (Per-SYNOP)": [method1_avg_r2, method1_avg_rmse],
    "Method 2 (Generic RF)": [generic_rfl_r2, generic_rfl_rmse]
})

# Save and display
comparison_df.to_csv("Comparison_Method1_vs_Method2.csv", index=False)
print("✅ Saved: Comparison_Method1_vs_Method2.csv")
display(comparison_df)


# In[26]:


import pandas as pd

# Load Method 1 model results
method1_rfl = pd.read_csv("SYNOP_RFL_Model_Results.csv")  # Uses column: "R2"
method1_fso = pd.read_csv("SYNOP_FSO_Model_Results.csv")  # Uses column: "R2"

# Load Method 2 generic results
generic_results = pd.read_csv("Final_Model_Test_Performance.csv")

# Extract baseline (Method 2) metrics
generic_rfl_r2 = generic_results.loc[generic_results["Model"] == "General RF", "R2"].values[0]
generic_rfl_rmse = generic_results.loc[generic_results["Model"] == "General RF", "RMSE"].values[0]
generic_fso_r2 = generic_results.loc[generic_results["Model"] == "General FSO", "R2"].values[0]
generic_fso_rmse = generic_results.loc[generic_results["Model"] == "General FSO", "RMSE"].values[0]

# === Calculate Improvements ===
method1_rfl["Delta_R2"] = method1_rfl["R2"] - generic_rfl_r2
method1_rfl["Delta_RMSE"] = method1_rfl["RMSE"] - generic_rfl_rmse

method1_fso["Delta_R2"] = method1_fso["R2"] - generic_fso_r2
method1_fso["Delta_RMSE"] = method1_fso["RMSE"] - generic_fso_rmse

# === Save Improvement Tables ===
method1_rfl.to_csv("Method1_RFL_Improvement_Table.csv", index=False)
method1_fso.to_csv("Method1_FSO_Improvement_Table.csv", index=False)

print("✅ Saved Method1_RFL_Improvement_Table.csv and Method1_FSO_Improvement_Table.csv")


# In[27]:


# ✅ Display Method 1 Improvement Tables (Step 13)

import pandas as pd
from IPython.display import display

# Load saved improvement tables
rfl_improvement_df = pd.read_csv("Method1_RFL_Improvement_Table.csv")
fso_improvement_df = pd.read_csv("Method1_FSO_Improvement_Table.csv")

# Display both tables
print("🔹 Method 1 RFL Improvement Table:")
display(rfl_improvement_df)

print("\n🔹 Method 1 FSO Improvement Table:")
display(fso_improvement_df)


# In[ ]:


# Method 3


# In[33]:


# ----------------------------
# ✅ Step 1: Define BFE Function
# ----------------------------
def optimized_bfe_oob_improved(X, y, target_name, visualize=True, early_stop_delta=0.01):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    from IPython.display import display
    import pandas as pd
    import numpy as np

    features = list(X.columns)
    rf_model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)

    best_rmse = float("inf")
    best_r2 = float("-inf")
    best_oob = float("-inf")
    best_index = 0
    best_subset = features.copy()

    results = []
    previous_rmse = None

    while len(features) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X[features], y, test_size=0.3, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        oob = rf_model.oob_score_

        results.append({
            "Step": len(results),
            "Removed Feature": None if len(results) == 0 else least_important,
            "Remaining Features": len(features),
            "Feature List": ", ".join(features),
            "RMSE": rmse,
            "R² Score": r2,
            "OOB Score": oob
        })

        if rmse < best_rmse and oob > best_oob:
            best_rmse = rmse
            best_r2 = r2
            best_oob = oob
            best_subset = features.copy()
            best_index = len(results) - 1

        importances = rf_model.feature_importances_
        least_important = features[np.argmin(importances)]

        if previous_rmse is not None and (rmse - previous_rmse) > early_stop_delta and r2 < best_r2 - early_stop_delta:
            break

        features.remove(least_important)
        previous_rmse = rmse

    result_df = pd.DataFrame(results)
    result_df.loc[0, "Removed Feature"] = "Full Feature Set"
    result_df.to_csv(f"feature_elimination_oob_{target_name}.csv", index=False)
    pd.DataFrame({"Selected Features": best_subset}).to_csv(f"selected_features_oob_{target_name}.csv", index=False)

    if visualize:
        print(f"\nFinal selected features for {target_name}:")
        display(pd.DataFrame({"Selected Features": best_subset}))
        print(f"\nFinal RMSE: {best_rmse:.4f}, R²: {best_r2:.4f}, OOB: {best_oob:.4f}")
        display(result_df)

        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax1.set_xlabel("Removed Feature")
        ax1.set_ylabel("RMSE", color='blue')
        ax1.plot(result_df["Removed Feature"], result_df["RMSE"], marker='o', color='blue', label='RMSE')
        ax1.tick_params(axis='y', labelcolor='blue')
        plt.xticks(rotation=90)

        ax2 = ax1.twinx()
        ax2.set_ylabel("R² Score", color='orange')
        ax2.plot(result_df["Removed Feature"], result_df["R² Score"], marker='s', linestyle='--', color='orange', label='R² Score')
        ax2.tick_params(axis='y', labelcolor='orange')

        ax1.axvline(x=best_index, color='green', linestyle='dashed', label='Optimal Subset')
        fig.tight_layout()
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
        plt.title(f"Feature Selection Curve for {target_name}")
        plt.grid(True)
        plt.show()

    return best_subset, result_df, best_rmse, best_r2, best_oob


# ----------------------------
# ✅ Step 2: Execute Method 3 (RFL ← FSO + Weather)
# ----------------------------

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load Data
df = pd.read_csv("balanced_dataset.csv")
X = df.drop(columns=["RFL_Att", "FSO_Att", "SYNOPCode"])
y_fso = df["FSO_Att"]
y_rfl = df["RFL_Att"]

# Shared Train-Test Split
X_train_all, X_test_all, y_train_fso, y_test_fso, y_train_rfl, y_test_rfl = train_test_split(
    X, y_fso, y_rfl, test_size=0.3, random_state=42
)

# Train FSO Model and Predict
fso_model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
fso_model.fit(X_train_all, y_train_fso)
X_train_m3 = X_train_all.copy()
X_test_m3 = X_test_all.copy()
X_train_m3["Predicted_FSO_Att"] = fso_model.predict(X_train_all)
X_test_m3["Predicted_FSO_Att"] = fso_model.predict(X_test_all)

# Apply Feature Elimination for Method 3
selected_features_m3, bfe_summary_m3, rmse_best_m3, r2_best_m3, oob_best_m3 = optimized_bfe_oob_improved(
    X_train_m3, y_train_rfl, target_name="Method3_RFL"
)

# Train Final Model using Best Subset
final_model_m3 = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
final_model_m3.fit(X_train_m3[selected_features_m3], y_train_rfl)
y_pred_m3 = final_model_m3.predict(X_test_m3[selected_features_m3])

# Final Evaluation
rmse_m3 = mean_squared_error(y_test_rfl, y_pred_m3, squared=False)
r2_m3 = r2_score(y_test_rfl, y_pred_m3)
oob_m3 = final_model_m3.oob_score_

# Print Results
print("\n✅ Final Evaluation for Method 3 (RFL ← FSO + Weather):")
print(f"Selected Features: {selected_features_m3}")
print(f"Test RMSE: {rmse_m3:.4f}")
print(f"Test R²: {r2_m3:.4f}")
print(f"OOB Score: {oob_m3:.4f}")


# In[35]:


from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

# Calculate Pearson correlation and Mutual Information between predicted and actual RFL_Att for Method 3
pearson_corr = pearsonr(y_test_rfl, y_pred_m3)[0]
mi_score = mutual_info_regression(y_test_rfl.values.reshape(-1, 1), y_pred_m3)[0]

print(f"📌 Pearson Correlation (Method 3): {pearson_corr:.4f}")
print(f"📌 Mutual Information (Method 3): {mi_score:.4f}")


# In[49]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, pearsonr

# Use correct variable from Method 3 test set
fso_pred_m3 = X_test_m3["Predicted_FSO_Att"].values   # X-axis: predicted FSO_Att
rfl_pred_m3 = y_pred_m3                                # Y-axis: predicted RFL_Att (from Method 3)

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(fso_pred_m3, rfl_pred_m3, alpha=0.6, edgecolor='k', label="Predicted Points")

# Best fit line
slope, intercept, r_value, p_value, std_err = linregress(fso_pred_m3, rfl_pred_m3)
x_line = np.linspace(min(fso_pred_m3), max(fso_pred_m3), 200)
y_line = slope * x_line + intercept
plt.plot(x_line, y_line, color='red', linestyle='--', label=f"Best Fit Line\n$r$ = {r_value:.2f}")

# Decorate plot
plt.title("Scatter: Predicted FSO vs Predicted RFL (Method 3)")
plt.xlabel("Predicted FSO_Att (dB)")
plt.ylabel("Predicted RFL_Att (dB)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# Show plot
plt.show()


# In[50]:


import matplotlib.pyplot as plt
import numpy as np

# Use predicted values instead of ground truth
rf_att = y_pred_m3                      # Predicted RFL_Att (Method 3 output)
fso_att = X_test_m3["Predicted_FSO_Att"].values  # Predicted FSO_Att added during Method 3

# Define bin widths to visualize
bin_widths = [0.1, 0.5, 1.0, 2.0]

# Plotting
fig, axs = plt.subplots(1, len(bin_widths), figsize=(20, 4), sharey=True)

for i, bw in enumerate(bin_widths):
    x_bins = np.arange(rf_att.min(), rf_att.max() + bw, bw)
    y_bins = np.arange(fso_att.min(), fso_att.max() + bw, bw)
    H, _, _ = np.histogram2d(rf_att, fso_att, bins=[x_bins, y_bins])

    axs[i].imshow(H.T, origin='lower', aspect='auto', cmap='viridis',
                  extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]])
    axs[i].set_title(f"Bin width = {bw} dB")
    axs[i].set_xlabel("Predicted RFL_Att")

    if i == 0:
        axs[i].set_ylabel("Predicted FSO_Att")

plt.suptitle("Figure A: 2D Histogram of Predicted RFL vs FSO at Different Bin Widths", fontsize=14)
plt.tight_layout()
plt.show()


# In[51]:


import matplotlib.pyplot as plt
import numpy as np

# Assumes:
# - y_pred_m3 is Method 3 predicted RFL_Att
# - X_test_m3 contains 'Predicted_FSO_Att'
# - df contains the full test set with 'SYNOPCode'

# Join predictions with weather
df_m3_results = df.loc[X_test_m3.index].copy()
df_m3_results["Predicted_RFL_Att"] = y_pred_m3
df_m3_results["Predicted_FSO_Att"] = X_test_m3["Predicted_FSO_Att"]

# Weather mapping
synop_mapping = {
    0: "Clear",
    3: "Duststorm",
    4: "Drizzle",
    5: "Showers",
    6: "Rain",
    7: "Snow",
    8: "Fog"
}

# Bin width setup
bin_width = 1.0  # change to 0.5 or 0.1 for higher resolution

# Plot setup
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
axs = axs.ravel()

for i, synop_code in enumerate(sorted(synop_mapping.keys())):
    subset = df_m3_results[df_m3_results["SYNOPCode"] == synop_code]
    if subset.empty:
        axs[i].axis('off')
        continue

    rf = subset["Predicted_RFL_Att"].values
    fso = subset["Predicted_FSO_Att"].values

    x_bins = np.arange(rf.min(), rf.max() + bin_width, bin_width)
    y_bins = np.arange(fso.min(), fso.max() + bin_width, bin_width)
    H, _, _ = np.histogram2d(rf, fso, bins=[x_bins, y_bins])

    axs[i].imshow(H.T, origin='lower', aspect='auto', cmap='viridis',
                  extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]])
    axs[i].set_title(f"{synop_mapping[synop_code]}")
    axs[i].set_xlabel("Predicted RFL_Att")
    axs[i].set_ylabel("Predicted FSO_Att")

# Turn off the last unused subplot if any
if len(axs) > len(synop_mapping):
    axs[-1].axis('off')

plt.suptitle("Figure B: Predicted RFL vs FSO Heatmaps Per Weather (Method 3, Bin Size = 1 dB)", fontsize=14)
plt.tight_layout()
plt.show()


# In[52]:


import numpy as np

# --- Maximum Entropy Criterion ---
def calculate_ME(N):
    """Calculate maximum number of bins ME such that N / (ME^2) >= 5"""
    return int(np.floor(np.sqrt(N / 5)))

# --- Calculate Adaptive Bin Size per Variable ---
def calculate_average_bin_size(x, ME):
    """Compute bin width for variable x using given ME"""
    x_min = np.min(x)
    x_max = np.max(x)
    data_range = x_max - x_min
    avg_bin_width = data_range / ME
    return avg_bin_width, x_min, x_max

# --- Example for Method 3 Predicted RFL and FSO ---
rf_att_pred = y_pred_m3                                # Method 3 predicted RFL_Att
fso_att_pred = X_test_m3["Predicted_FSO_Att"].values   # Method 3 predicted FSO_Att

# Step 1: Calculate ME
N = len(rf_att_pred)
ME = calculate_ME(N)
print(f"✅ Recommended number of bins per variable (ME): {ME}")

# Step 2: Bin size calculation for predicted variables
rf_bw, rf_min, rf_max = calculate_average_bin_size(rf_att_pred, ME)
fso_bw, fso_min, fso_max = calculate_average_bin_size(fso_att_pred, ME)

print(f"\nPredicted RFL_Att Range: {rf_min:.2f} dB to {rf_max:.2f} dB")
print(f"Predicted FSO_Att Range: {fso_min:.2f} dB to {fso_max:.2f} dB")
print(f"\n📏 Approx. Average Bin Width for Predicted RFL_Att: {rf_bw:.4f} dB")
print(f"📏 Approx. Average Bin Width for Predicted FSO_Att: {fso_bw:.4f} dB")


# In[56]:


import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr

# Load your predictions
method1_preds_df = pd.read_csv("Test_Predictions_RFL_Att.csv")

# ✅ No assertion needed — we already know the column names are:
# 'Measured_RFL_Att', 'Predicted_RFL_Att'

# Extract true and predicted values
y_true_m1_all = method1_preds_df["Measured_RFL_Att"].values
y_pred_m1_all = method1_preds_df["Predicted_RFL_Att"].values

# Pearson Correlation
pearson_m1_rfl = pearsonr(y_true_m1_all, y_pred_m1_all)[0]

# Mutual Information
mi_m1_rfl = mutual_info_regression(y_true_m1_all.reshape(-1, 1), y_pred_m1_all)[0]

# Print results
print(f"📌 Method 1 Pearson Correlation: {pearson_m1_rfl:.4f}")
print(f"📌 Method 1 Mutual Information: {mi_m1_rfl:.4f}")


# In[59]:


from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr

# Method 2: already computed predictions for y_pred_m3 and y_test_rfl
pearson_m2_rfl = pearsonr(y_test_rfl, y_pred_m3)[0]
mi_m2_rfl = mutual_info_regression(y_test_rfl.values.reshape(-1, 1), y_pred_m3)[0]

# These should be defined from your Method 3 evaluation cell
rmse_m3_rfl = mean_squared_error(y_test_rfl, y_pred_m3, squared=False)
r2_m3_rfl = r2_score(y_test_rfl, y_pred_m3)
oob_m3_rfl = final_model_m3.oob_score_
pearson_m3_rfl = pearsonr(X_test_m3["Predicted_FSO_Att"], y_pred_m3)[0]
mi_m3_rfl = mutual_info_regression(X_test_m3["Predicted_FSO_Att"].values.reshape(-1, 1), y_pred_m3)[0]


comparison_df = pd.DataFrame([
    {
        "Method": "Method 1 (Per-SYNOP)",
        "RMSE": rmse_m1_rfl,
        "R²": r2_m1_rfl,
        "OOB": oob_m1_rfl,
        "Pearson": pearson_m1_rfl,
        "MI": mi_m1_rfl
    },
    {
        "Method": "Method 2 (FSO → RFL)",
        "RMSE": rmse_m2_rfl,
        "R²": r2_m2_rfl,
        "OOB": oob_m2_rfl,
        "Pearson": pearson_m2_rfl,
        "MI": mi_m2_rfl
    },
    {
        "Method": "Method 3 (RFL ← FSO)",
        "RMSE": rmse_m3_rfl,
        "R²": r2_m3_rfl,
        "OOB": oob_m3_rfl,
        "Pearson": pearson_m3_rfl,
        "MI": mi_m3_rfl
    }
])

# Save and display
comparison_df.to_csv("Final_Method_Comparison.csv", index=False)
display(comparison_df)


# In[60]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the comparison table
comparison_df = pd.read_csv("Final_Method_Comparison.csv")

# Melt the DataFrame to long format for plotting
comparison_melted = comparison_df.melt(id_vars="Method", var_name="Metric", value_name="Score")

# Create grouped bar plot
plt.figure(figsize=(12, 6))
sns.barplot(data=comparison_melted, x="Metric", y="Score", hue="Method")

# Plot decorations
plt.title("Comparison of RMSE, R², OOB, Pearson, and MI Across Methods", fontsize=14)
plt.ylabel("Score")
plt.xlabel("Metric")
plt.grid(axis='y', linestyle="--", alpha=0.6)
plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Show plot
plt.show()


# In[61]:


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Reattach SYNOPCode to test set
synop_test = df.loc[y_test_rfl.index, "SYNOPCode"]  # shared for both Method 2 and 3

# Create DataFrames with true, predicted, and weather labels
df_m2 = pd.DataFrame({
    "SYNOPCode": synop_test.values,
    "True_RFL_Att": y_test_rfl.values,
    "Pred_RFL_Att": y_pred_m3  # Method 2 prediction
})

df_m3 = pd.DataFrame({
    "SYNOPCode": synop_test.values,
    "True_RFL_Att": y_test_rfl.values,
    "Pred_RFL_Att": y_pred_m3  # Method 3 uses same name if reused
})

# Group by SYNOPCode and compute RMSE & R²
def compute_per_synop_metrics(df):
    result = []
    for synop, group in df.groupby("SYNOPCode"):
        y_true = group["True_RFL_Att"].values
        y_pred = group["Pred_RFL_Att"].values
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        result.append({
            "SYNOPCode": synop,
            "RMSE": rmse,
            "R²": r2,
            "N": len(group)
        })
    return pd.DataFrame(result).sort_values("SYNOPCode")

# Compute per-SYNOP results
method2_synop_metrics = compute_per_synop_metrics(df_m2)
method3_synop_metrics = compute_per_synop_metrics(df_m3)

# Display summary
print("📊 Method 2 (FSO → RFL) - Per SYNOP RMSE:")
display(method2_synop_metrics)

print("📊 Method 3 (RFL ← FSO) - Per SYNOP RMSE:")
display(method3_synop_metrics)

# Save to CSV
method2_synop_metrics.to_csv("Method2_RMSE_by_SYNOP.csv", index=False)
method3_synop_metrics.to_csv("Method3_RMSE_by_SYNOP.csv", index=False)

# Summary stats
print(f"Method 2 - Average RMSE: {method2_synop_metrics['RMSE'].mean():.4f}")
print(f"Method 2 - Worst RMSE: {method2_synop_metrics['RMSE'].max():.4f}")

print(f"Method 3 - Average RMSE: {method3_synop_metrics['RMSE'].mean():.4f}")
print(f"Method 3 - Worst RMSE: {method3_synop_metrics['RMSE'].max():.4f}")


# In[63]:


# --------------------------------------------------------
# 🔍 Task: Compare Normalized MI per Weather Across Bin Sizes
# --------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log2

# --- Entropy & Mutual Information Functions ---
def get_prob_distribution(data, bin_width):
    bins = np.arange(np.min(data), np.max(data) + bin_width, bin_width)
    hist, _ = np.histogram(data, bins=bins)
    prob = hist / np.sum(hist)
    return prob[prob > 0]

def joint_prob_distribution(x, y, bin_width):
    x_bins = np.arange(np.min(x), np.max(x) + bin_width, bin_width)
    y_bins = np.arange(np.min(y), np.max(y) + bin_width, bin_width)
    H, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
    Pxy = H / np.sum(H)
    return Pxy[Pxy > 0]

def entropy(p):
    return -np.sum(p * np.log2(p))

def joint_entropy(pxy):
    return -np.sum(pxy * np.log2(pxy))

def mutual_information_custom(x, y, bin_width):
    px = get_prob_distribution(x, bin_width)
    py = get_prob_distribution(y, bin_width)
    pxy = joint_prob_distribution(x, y, bin_width)
    Hx = entropy(px)
    Hy = entropy(py)
    Hxy = joint_entropy(pxy)
    MI = Hx + Hy - Hxy
    norm_MI = MI / Hxy if Hxy > 0 else 0
    return MI, norm_MI

# --- Group-wise MI by SYNOP and bin width ---
def mi_per_synop_by_bin(y1, y2, synop_series, synop_mapping, bin_sizes):
    results = {bw: [] for bw in bin_sizes}
    for bw in bin_sizes:
        for code in sorted(synop_mapping.keys()):
            idx = (synop_series == code)
            if np.sum(idx) > 10:
                x = y1[idx]
                y = y2[idx]
                mi, norm_mi = mutual_information_custom(x, y, bin_width=bw)
                results[bw].append(norm_mi)
            else:
                results[bw].append(np.nan)
    return results

# --- Prepare Inputs ---
synop_mapping = {
    0: "Clear",
    3: "Duststorm",
    4: "Drizzle",
    5: "Showers",
    6: "Rain",
    7: "Snow",
    8: "Fog"
}

y_rf = y_test_rfl.values
y_fso = y_test_fso.values
synop_test_array = df.loc[y_test_rfl.index, "SYNOPCode"].values

synop_labels = [synop_mapping[k] for k in sorted(synop_mapping.keys())]

# --- Compute MI across multiple bin widths ---
bin_results = mi_per_synop_by_bin(
    y_rf, y_fso,
    synop_series=synop_test_array,
    synop_mapping=synop_mapping,
    bin_sizes=[0.1, 0.23, 1.0]
)

# --- Save MI results to CSV ---
pd.DataFrame(bin_results, index=synop_labels).to_csv("MI_Per_SYNOP_By_BinSize.csv")

# --- Plot grouped bar chart ---
x = np.arange(len(synop_labels))
bar_width = 0.25

plt.figure(figsize=(14, 7))
plt.bar(x - bar_width, bin_results[0.1], width=bar_width, label="Bin Width = 0.1 dB")
plt.bar(x, bin_results[0.23], width=bar_width, label="Bin Width = 0.23 dB (ME Rule)")
plt.bar(x + bar_width, bin_results[1.0], width=bar_width, label="Bin Width = 1.0 dB")

plt.xticks(x, synop_labels, rotation=45)
plt.ylabel("Normalized MI (I / Hxy)")
plt.title("Figure X: Normalized Mutual Information per Weather Type\nfor Multiple Bin Widths")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[76]:


import matplotlib.pyplot as plt
import pandas as pd

# --- Model Metrics ---
data = {
    "Model": [
        "Generic RF", "Generic FSO",
        "Optimized RF", "Optimized FSO",
        "Method 1 (RFL)", "Method 1 (FSO)",
        "Method 2 (FSO ← RFL)", "Method 3 (RFL ← FSO)"
    ],
    "RMSE": [
        0.822278, 0.939455,
        0.742265, 0.971168,
        0.657807, 0.779819,
        0.9995, 0.802191
    ],
    "R2": [
        0.939495, 0.965581,
        0.943277, 0.965315,
        0.925072, 0.925529,
        0.9638, 0.935670
    ],
    "OOB": [
        0.926555, 0.963249,
        0.930705, 0.963651,
        0.953799, 0.921000,
        0.9637, 0.9293
    ],
    "Pearson": [
        0.939, 0.965,
        0.943, 0.965,
        0.988260, 0.9900,
        -0.0139, 0.9675
    ],
    "MI": [
        2.52, 2.61,
        2.68, 2.63,
        2.451909, 2.83,
        2.5445, 2.7427
    ]
}

df = pd.DataFrame(data)

# Normalize scores (1 / RMSE to convert to "higher is better")
df['1_RMSE'] = 1 / df['RMSE']
df['1_RMSE_norm'] = df['1_RMSE'] / df['1_RMSE'].max()
df['R2_norm'] = df['R2'] / df['R2'].max()
df['OOB_norm'] = df['OOB'] / df['OOB'].max()
df['Pearson_abs'] = df['Pearson'].abs()
df['Pearson_norm'] = df['Pearson_abs'] / df['Pearson_abs'].max()
df['MI_norm'] = df['MI'] / df['MI'].max()

# Create final plot DataFrame
plot_df = df[["Model", "1_RMSE_norm", "R2_norm", "OOB_norm", "Pearson_norm", "MI_norm"]].set_index("Model")
plot_df.columns = ['1 / RMSE', 'R²', 'OOB Score', 'Pearson |r|', 'Mutual Info']

# Plot
plot_df.plot(kind="bar", figsize=(14, 6), width=0.85)
plt.title("Normalized Performance Comparison of All Models", fontsize=14)
plt.ylabel("Normalized Score (0-1)")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

