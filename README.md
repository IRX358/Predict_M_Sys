# 🔮 Prediction Management System

[![Python](https://img.shields.io/badge/python-3.10-blue?logo=python)](https://www.python.org)
[![Made with Flask](https://img.shields.io/badge/Made%20with-Flask-blue.svg?logo=flask)](https://flask.palletsprojects.com/)

---


> A robust *Prediction Management System* that
Uses deep learning
to predict equipment failures from vibration data. It converts signals into spectrograms, processes them with a CNN model ,and delivers accurate predictions. With a Flask-powered web interface, it helps industries reduce downtime , optimize maintenance ,and improve efficiency through early failure detection. 🛠
<br>
---

<h2>Instructions to use</h2>
<div id="instructions-content">
        <ul>
            <li>📂 Upload a PNG or JPG file</li>
            <li>⏳ Wait for the prediction result</li>
            <li>✅ Check result or use Try Again if needed</li>
            <li>📊 Visit <a href="/metrics">Metrics</a> for model performance</li>
        </ul>
    </div>
   <br>
---
  
   <h2>
   Model Metrics : 
   </h2>
        <table>
            <thead>
                <tr class="table-info" >
                    <th class="table-info"  scope="col"># </th>
                    <th class="table-info"  scope="col">Metrics </th>
                    <th class="table-info"  scope="col">Values </th>
                </tr>
            </thead>
            <tbody>
                <tr class="table-info" >
                    <th class="table-info"  scope="row">1</th>
                    <td>Accuracy[Overall correct predictions] : </td>
                    <td>0.9867</td>
                </tr>
                <tr class="table-info" >
                    <th class="table-info"  scope="row">2</th>
      <td>Precision[True Positives / (TP + FP)]: </td>
      <td>0.9867</td>
    </tr>
    <tr class="table-info" >
        <th scope="row">3</th>
        <td>Recall[True Positives / (TP + FN)]: </td>
        <td>0.9867</td>
    </tr>
    <tr class="table-info" >
      <th class="table-info"  scope="row">4</th>
      <td>F1_Score[Harmonic mean of Precision & Recal]: </td>
      <td>0.9867</td>
    </tr>
  </tbody>
</table>  
---

## ✨ Features

✅ Interactive web interface  
✅ Upload and process data files  
✅ Serve machine learning predictions  
✅ Store and track prediction history  
✅ Modular and extensible project architecture  
✅ Dockerized for easy deployment

---

## 🛠 Tech Stack

- 🐍 *Python 3.10+*  
- ⚙ *Flask*    
- ⛩️ *Jinja*
- 🌐 *HTML - CSS - JS*


---

## 🏗 Setup

Clone the repository:

```bash
git clone https://github.com/IRX358/Prediction-Management-System.git
```
Create a virtual environment:
```
python -m venv venv
```
Activate the environment:

Windows:
```
venv\Scripts\activate
```
macOS/Linux:
```
source venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

---

🚀 Running the App

Start the Flask app:
```
python app.py
```
The system will be live on:
```
http://127.0.0.1:5000
```

---

📂 Project Structure
```
.
├── app.py
├── templates/
├── static/
├── models/
├── data/
├── requirements.txt
└── README.md
```

---


🤝 Connect

🐙 GitHub: <a href="https://github.com/IRX358">IRX358</a>

💼 LinkedIn: <a href="https://www.linkedin.com/irfan358"> Irfan Basha </a>

---

>  © 2025 Irfan IR || 
            Built with great MOOD😎 , EXCITEMENT🤩 and CURIOSITY🤔
