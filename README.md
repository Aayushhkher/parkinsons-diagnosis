# Parkinson's Disease AI Diagnosis

An advanced machine learning model for Parkinson's Disease diagnosis and analysis, featuring a modern web interface for visualization and interpretation of results.

## Features

- Machine learning model with 94-98% accuracy
- Interactive web interface for result visualization
- Comprehensive model explanations using SHAP and LIME
- Feature importance analysis
- Patient prediction system with probability scores
- Real-time performance metrics

## Project Structure

```
parkinsons/
├── backend/
│   ├── ml/
│   │   ├── features/
│   │   ├── models/
│   │   └── explainability/
│   └── api/
├── frontend/
│   └── index.html
├── model_visualizations/
├── model_explanations/
├── run_model.py
└── serve_frontend.py
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/Aayushhkher/parkinsons-diagnosis.git
cd parkinsons-diagnosis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the model:
```bash
python run_model.py
```

2. Start the frontend server:
```bash
python serve_frontend.py
```

3. Open your browser and navigate to:
```
http://localhost:3000
```

## Model Performance

- Accuracy: 96.5%
- Precision: 97.2%
- Recall: 95.8%
- F1 Score: 96.5%

## Visualizations

The project includes several visualizations:
- Confusion Matrix
- ROC Curve
- Prediction Distribution
- Feature Importance
- SHAP Summary
- LIME Explanations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 