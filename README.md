<br><u>Project Title</b><br></u>
Medical Cost Personal Dataset (Linear Regression)

---

<b> Team Member Details<br></b>
Member 1 - Muskan: 2503031240067<br>
Member 2 - Meet: 2503031240063  <br>
Member 3 - Samir: 2503031240066<br>
Member 4 - Neelam: 2503031240070<br>
Member 5 - Nitin: 2503031240071<br>
Member 6 - Arnav: 2503031240072<br>
Member 7 - Sohel: 2503031240075<br>

---

<b> Problem Statement<br></b>
The objective of this project is to predict medical insurance charges based on individual attributes such as age, gender, BMI, number of children, smoking habits, and region.  <br>
The goal is to identify key factors affecting insurance costs and build a machine learning model to estimate charges accurately.<br>

---

<b> Dataset Description<br></b>
The dataset used in this project is taken from Kaggle Insurance Datset.<br>
<table border="1">
    <tr>
        <th>Feature</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>age</td>
        <td>Age of the individual</td>
    </tr>
    <tr>
        <td>sex</td>
        <td>Gender (male/female)</td>
    </tr>
    <tr>
        <td>bmi</td>
        <td>Body Mass Index</td>
    </tr>
    <tr>
        <td>children</td>
        <td>Number of dependents</td>
    </tr>
    <tr>
        <td>smoker</td>
        <td>Smoking status (yes/no)</td>
    </tr>
    <tr>
        <td>region</td>
        <td>Residential region</td>
    </tr>
    <tr>
        <td>charges</td>
        <td>Medical insurance charges (target variable)</td>
    </tr>
</table>
The dataset is structured and contains no missing values.

---

 <b>Data Preprocessing Steps<br></b>
- Checked dataset for null/missing values
- Converted categorical variables into numerical format using encoding:
  - sex → 0/1
  - smoker → 0/1
  - region → label encoding
- Split dataset into training and testing sets
- Feature scaling applied (if required)

---

<b> Model Used and Training Details<br></b>
- Model Used: Linear Regression  
- Library: Scikit-learn  
- The dataset was split into:
  - Training set (80%)
  - Testing set (20%)
- Model was trained using training data and predictions were made on test data

---

<b> Model Evaluation Results<br></b>
- The model performance was evaluated using:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score

---

<b> Key Observations:<br></b>
  - Smoking has a significant impact on insurance charges
  - Higher BMI leads to higher charges
  - Age also contributes to increased costs

---

<b> GitHub Collaboration Summary<br></b>
- Repository created and managed on GitHub
- Code and dataset uploaded to the repository
- Version control maintained using commits
- README file added for documentation
- Project structured for easy understanding and execution

---

 <b>Conclusion<br></b>
This project successfully demonstrates how machine learning can be used to predict insurance charges.  <br>
Linear Regression provides a good baseline model for this dataset.  <br>
Further improvements can be made using advanced algorithms and hyperparameter tuning.<br>
