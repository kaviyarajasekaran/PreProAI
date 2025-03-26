from flask import Flask, request, jsonify
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

app = Flask(__name__)

# 📂 Folder to store uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 📂 Folder to store visualization images
VISUALIZATION_FOLDER = 'visualizations'
os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)
app.config['VISUALIZATION_FOLDER'] = VISUALIZATION_FOLDER

# 🏠 Home Route
@app.route('/')
def home():
    return "Flask Backend is Running!"

# 📂 Upload File Route
@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files')  # Get multiple files
    saved_files = []

    for file in files:
        if file.filename == '':
            continue

        if not file.filename.endswith('.csv'):
            return jsonify({'error': f'Invalid file format: {file.filename}'}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        saved_files.append(file.filename)

    return jsonify({'message': 'Files uploaded successfully', 'files': saved_files}), 200

# 🧹 Data Cleaning Route
@app.route('/clean', methods=['POST'])
def clean_data():
    data = request.get_json()  
    filenames = data.get('filenames')  # Expecting a list of filenames

    if not filenames or not isinstance(filenames, list):
        return jsonify({'error': 'List of filenames required'}), 400

    cleaned_files = []

    for filename in filenames:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if not os.path.exists(filepath):
            return jsonify({'error': f'File not found: {filename}'}), 404

        df = pd.read_csv(filepath)

        df.drop_duplicates(inplace=True)  
        df.fillna(df.mean(), inplace=True)

        cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'cleaned_' + filename)
        df.to_csv(cleaned_filepath, index=False)
        cleaned_files.append('cleaned_' + filename)

    return jsonify({'message': 'Data cleaned successfully', 'cleaned_files': cleaned_files}), 200

# 📊 Data Summary & Insights Route
@app.route('/summary', methods=['POST'])
def data_summary():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({'error': 'Filename required'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    df = pd.read_csv(filepath)

    # Get Column Names
    columns = df.columns.tolist()

    # Get Data Types
    data_types = df.dtypes.apply(lambda x: str()).to_dict()

    # Count Missing Values
    missing_values = df.isnull().sum().to_dict()

    # Basic Statistics
    stats = df.describe().to_dict()

    return jsonify({
        'columns': columns,
        'data_types': data_types,
        'missing_values': missing_values,
        'statistics': stats
    }), 200

# 📊 Model Evaluation Route
@app.route('/evaluate', methods=['POST'])
def evaluate_model():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({'error': 'Filename required'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    df = pd.read_csv(filepath)

    if df.shape[1] < 2:
        return jsonify({'error': 'Dataset must have at least one feature and one target column'}), 400

    X = df.iloc[:, :-1]  
    y = df.iloc[:, -1]  

    if not pd.api.types.is_numeric_dtype(y):
        return jsonify({'error': 'Target column must be numeric'}), 400

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    return jsonify({'mse': mse}), 200

# 📊 Data Visualization Route
@app.route('/visualize', methods=['POST'])
def visualize_data():
    data = request.get_json()
    filename = data.get('filename')
    chart_type = data.get('chart_type', 'histogram')

    if not filename:
        return jsonify({'error': 'Filename required'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    df = pd.read_csv(filepath)

    plt.figure(figsize=(10, 6))
    if chart_type == 'bar':
        df.iloc[:, 0].value_counts().plot(kind='bar')
    elif chart_type == 'line':
        df.plot(kind='line')
    elif chart_type == 'histogram':
        df.hist()
    elif chart_type == 'scatter' and df.shape[1] > 1:
        sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1])
    elif chart_type == 'pie':
        df.iloc[:, 0].value_counts().plot(kind='pie', autopct='%1.1f%%')
    else:
        return jsonify({'error': 'Invalid chart type or insufficient data'}), 400

    image_path = os.path.join(app.config['VISUALIZATION_FOLDER'], f'{chart_type}.png')
    plt.savefig(image_path)
    plt.close()

    return jsonify({'message': 'Visualization created', 'image_path': image_path}), 200

if __name__ == '__main__':
    app.run(debug=True)
