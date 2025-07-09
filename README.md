# Sonar Rock vs Mine Classifier

## Installation

1. Clone the repository:
```
git clone https://github.com/your-username/sonar-rock-vs-mine.git
```
2. Install the required dependencies:
```
pip install pandas numpy scikit-learn seaborn matplotlib imblearn plotly joblib
```

## Usage

1. Ensure the `sonar data (1).csv` file is in the same directory as the Python script.
2. Run the `rockORmine.py` script to start the Streamlit application.
```
python rockORmine.py
```
3. The application will open in your default web browser. Use the sidebar to navigate between the available pages.

## API

The `rockORmine.py` script provides the following functions:

- `load_models_and_data()`: Loads the pre-trained models and test data from pickle files. If the files are not found, it trains new models and saves them.
- `calculate_model_metrics()`: Calculates the performance metrics (accuracy, precision, recall, F1-score, ROC-AUC) for the loaded models.
- `load_full_data()`: Loads the full sonar dataset from the `sonar data (1).csv` file.
- `get_css()`: Returns the custom CSS styles for the Streamlit application.

## Contributing

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Implement your changes and ensure the code passes any existing tests.
4. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the [MIT License](LICENSE).

## Testing

The project does not include automated tests at the moment. However, you can manually test the application by running the `rockORmine.py` script and interacting with the different pages and functionalities.