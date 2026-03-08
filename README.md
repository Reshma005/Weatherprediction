#  Weather Temperature Prediction using Machine Learning

##  Project Overview

This project predicts **weather temperature** using machine learning techniques.
The model is trained on historical weather data and deployed through a **Streamlit web application** where users can enter weather conditions and get predicted temperature.

The project includes **data preprocessing, exploratory data analysis (EDA), model training, evaluation, and a web interface**.

---

##  Dataset

The dataset used in this project contains historical weather observations such as:

* Temperature (C)
* Humidity
* Wind Speed (km/h)
* Pressure (millibars)
* Visibility (km)
* Weather summary
* Date and time

Dataset file: `weatherHistory.csv`

---

##  Exploratory Data Analysis (EDA)

EDA was performed to understand patterns and relationships in the dataset.

Key visualizations include:

* Temperature distribution
* Humidity vs Temperature scatter plot
* Correlation heatmap
* Monthly temperature trends
* Weather condition frequency

These visualizations help identify important features affecting temperature prediction.

---

##  Data Preprocessing

The following preprocessing steps were applied:

1. Handling missing values
2. Converting date column to datetime format
3. Extracting features (year, month, day, hour)
4. Encoding categorical variables using one-hot encoding
5. Splitting dataset into training and testing sets

---

##  Machine Learning Models

The following models were tested:

* Decision Tree Regressor
* Random Forest Regressor

The best performing model was selected based on evaluation metrics.

---

## 📊Model Evaluation

Model performance was evaluated using:

* **R² Score**
* **RMSE (Root Mean Squared Error)**

Example results:

* Training Score: **0.99**
* Testing Score: **0.95**
* RMSE: **~2°C**

This indicates the model can predict temperature with good accuracy.

---

##  Web Application

A **Streamlit web application** was created to interact with the trained model.

Users can input:

* Humidity
* Wind Speed
* Pressure
* Visibility
* Month
* Day
* Hour

The model then predicts the **temperature in Celsius**.

---

##  How to Run the Project

### 1️ Install Required Libraries

```bash
pip install pandas numpy scikit-learn streamlit matplotlib seaborn
```

### 2️ Train the Model

```bash
python train_model.py
```

This generates:

* `weather_model.pkl`
* `features.pkl`

### 3️ Run the Web Application

```bash
streamlit run streamlit_app.py
```

Open in browser:

```
http://localhost:8501
```

---


## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit

---

##  Future Improvements

* Add more weather features for improved accuracy
* Deploy the application online
* Integrate real-time weather APIs
* Improve UI with advanced dashboards

---

## Author

**Reshma Jadhav**

Machine Learning & Data Analytics Enthusiast
