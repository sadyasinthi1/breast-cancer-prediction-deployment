![image](https://github.com/user-attachments/assets/c3d54747-b46a-414c-a2e8-b59e73ec4e4a)


Breast Cancer Prediction System
Developed a neural network using PyTorch and deployed it via Flask on Render. Preprocessed real-world data from the UCI Breast Cancer Dataset using scikit-learn and NumPy. Integrated with Power BI to visualize model performance and feature trends.
Tools: PyTorch, Flask, scikit-learn, NumPy, Power BI, Render, HTML/CSS

Full Tech Stack Used
🧠 Machine Learning & Data Processing
PyTorch – For building and training the neural network model.

Scikit-learn (sklearn) – For:

Loading the UCI Breast Cancer dataset

Preprocessing using StandardScaler

Splitting data using train_test_split

NumPy – For numerical operations and array handling.

Pandas (optional, if you use it in Power BI or for CSV export) – Useful for creating/exporting datasets.

🌐 Web Framework & Backend
Flask – Lightweight Python web framework used to:

Create the web server

Handle form input and predictions

Connect frontend to backend

🖥️ Frontend & UI
HTML/CSS – For creating the user interface (form for inputting 30 features).

Jinja2 – Flask’s templating engine used to render HTML and pass the prediction result.

🚀 Deployment
Render – For deploying the Flask app to the web.

OS (Python's os module) – To handle environment variables (PORT) for deployment.

📊 Visualization (Planned)
Power BI (planned) – To visualize:
