# Hybrid-RF-FSO-Signal-Attenuation-Modeling-Machine-Learning-for-Weather-Aware-Communication-Links

# 🛰️ Hybrid RF–FSO Signal Attenuation Modeling  
**Machine Learning for Weather-Aware Communication Links**

This project explores machine learning-based strategies to predict signal attenuation in **hybrid Free Space Optical (FSO)** and **Radio Frequency (RF)** communication systems. It was developed as part of the Master of Data Science capstone project at the University of Adelaide, focusing on high-performance prediction under varying atmospheric conditions using Random Forest regressors.

---

## 📄 Project Files

- 📑 [Final Report (PDF)](Final_project_data_science_a1899824%20(2)%20(1).pdf)  
- 🧠 [Main Code (Python)](Final_report_ds_a1899824.py)

---

## 📦 Dataset

- **91,379 samples**, each with 27 weather and system-related features
- Target variables:
  - `FSO_Att`: Free Space Optical signal attenuation
  - `RFL_Att`: Radio Frequency signal attenuation
- Weather conditions labeled using SYNOP codes (fog, rain, snow, etc.)

---

## 🎯 Objectives

- Build predictive models for RF and FSO attenuation under dynamic weather
- Compare:
  - Generic vs. Weather-Specific (SYNOP-based) models
  - Cross-modal models (FSO ← RF and RF ← FSO)
- Evaluate models using both accuracy and structural preservation:
  - RMSE, R², OOB Score
  - Pearson Correlation, Entropy-based Mutual Information

---

## 🧠 Modeling Approach

- **Algorithm**: Random Forest Regressors (`scikit-learn`)
- **Feature Selection**: Backward Feature Elimination using OOB Score
- **Model Variants**:
  - **General Models** (trained on all data)
  - **Method 1**: Separate model per weather type (SYNOPCode)
  - **Method 2**: FSO prediction using RF + weather features
  - **Method 3**: RF prediction using FSO + weather features
- **Evaluation Metrics**:  
  - RMSE, R², OOB Score  
  - Pearson Correlation  
  - Mutual Information (Entropy-Based)

---

## 📊 Results Summary

| Model                        | RMSE   | R² Score | OOB Score | Pearson r | MI   |
|-----------------------------|--------|----------|-----------|-----------|------|
| General RF (full)           | 0.8223 | 0.9325   | 0.9266    | 0.9087    | 2.34 |
| General FSO (full)          | 1.0709 | 0.8822   | 0.8579    | 0.9121    | 2.51 |
| **Method 1** (per-SYNOP)    | 0.6653 | 0.9232   | 0.9522    | **0.9883**| 2.83 |
| Method 2 (FSO ← RF)         | 0.8021 | 0.8681   | 0.9001    | -0.0139   | 2.54 |
| **Method 3** (RF ← FSO)     | **0.8022** | **0.9357** | **0.9413** | **0.9675** | **2.74** |

---

## 🛠 Technologies Used

- Python 3.11  
- pandas, NumPy  
- scikit-learn  
- seaborn, matplotlib  
- SciPy  
- Jupyter Notebook

---

## 🚀 Future Work

- Extend to **LSTM/CNN-based models** for sequence-aware prediction  
- Incorporate **live weather sensor integration** for adaptive link management  
- Use real-time **satellite FSO feeds** to simulate predictive switching in hybrid networks

---

## 📌 Author

**Aditya Venugopalan Nediyirippil**  
University of Adelaide – Master of Data Science  
Project Supervisor: Dr. Siu Wai Ho  
