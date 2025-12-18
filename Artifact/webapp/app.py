"""
Property Price Predictor - Flask Web Application
Uses trained ML models to predict UK property prices based on EPC and property features
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys

app = Flask(__name__)

# Add parent directory to path to access models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load models and encoders
print("Loading models and encoders...")
MODEL_PATH = '../models/model_metadata.pkl'
LABEL_ENCODERS_PATH = '../models/label_encoders.pkl'
TARGET_ENCODINGS_PATH = '../models/target_encodings.pkl'

# Load model metadata
metadata = joblib.load(MODEL_PATH)
# Extract just the filename from the path
best_model_filename = os.path.basename(metadata['best_model_path'])
best_model_path = f'../models/{best_model_filename}'
model = joblib.load(best_model_path)
feature_names = metadata['feature_names']

# Load encoders
label_encoders = joblib.load(LABEL_ENCODERS_PATH)
target_encodings = joblib.load(TARGET_ENCODINGS_PATH)

print(f"Loaded model: {metadata['best_model_name']}")
print(f"Model R²: {metadata['test_r2']:.4f}")
print(f"Model MAE: £{metadata['test_mae']:,.0f}")

# Get dropdown options from encoders
PROPERTY_TYPES = sorted(label_encoders['PROPERTY_TYPE'].classes_.tolist())
PROPERTY_SUBTYPES = sorted(label_encoders['property_type'].classes_.tolist())
CONSTRUCTION_AGES = sorted(label_encoders['CONSTRUCTION_AGE_BAND'].classes_.tolist())
DISTRICTS = sorted(label_encoders['district'].classes_.tolist())
COUNTIES = sorted(label_encoders['COUNTY'].classes_.tolist())
ENERGY_RATINGS = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# Property type mapping for display
PROPERTY_SUBTYPE_MAP = {
    'D': 'Detached',
    'S': 'Semi-Detached',
    'T': 'Terraced',
    'F': 'Flat/Maisonette'
}

@app.route('/')
def index():
    """Home page with prediction form"""
    current_year = datetime.now().year
    current_month = datetime.now().month

    return render_template('index.html',
                         property_types=PROPERTY_TYPES,
                         property_subtypes=PROPERTY_SUBTYPES,
                         property_subtype_map=PROPERTY_SUBTYPE_MAP,
                         construction_ages=CONSTRUCTION_AGES,
                         districts=DISTRICTS,
                         counties=COUNTIES,
                         energy_ratings=ENERGY_RATINGS,
                         current_year=current_year,
                         current_month=current_month,
                         model_name=metadata['best_model_name'],
                         model_r2=f"{metadata['test_r2']*100:.1f}",
                         model_mae=f"{metadata['test_mae']:,.0f}")

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get form data
        floor_area = float(request.form['floor_area'])
        num_rooms = int(request.form['num_rooms'])
        property_type = request.form['property_type']
        property_subtype = request.form['property_subtype']
        construction_age = request.form['construction_age']
        district = request.form['district']
        county = request.form['county']
        is_freehold = 1 if request.form['is_freehold'] == 'yes' else 0
        energy_rating = request.form['energy_rating']
        sale_year = int(request.form['sale_year'])
        sale_month = int(request.form['sale_month'])

        # Validate inputs
        if floor_area < 20 or floor_area > 500:
            return render_template('result.html',
                                 error="Floor area must be between 20 and 500 sqm")

        if num_rooms < 1 or num_rooms > 15:
            return render_template('result.html',
                                 error="Number of rooms must be between 1 and 15")

        # Create feature dictionary
        features = {}

        # Numeric features
        features['TOTAL_FLOOR_AREA'] = floor_area
        features['NUMBER_HABITABLE_ROOMS'] = num_rooms

        # Encoded categorical features
        features['PROPERTY_TYPE_encoded'] = label_encoders['PROPERTY_TYPE'].transform([property_type])[0]
        features['property_type_encoded'] = label_encoders['property_type'].transform([property_subtype])[0]
        features['CONSTRUCTION_AGE_BAND_encoded'] = label_encoders['CONSTRUCTION_AGE_BAND'].transform([construction_age])[0]
        features['district_encoded'] = label_encoders['district'].transform([district])[0]
        features['COUNTY_encoded'] = label_encoders['COUNTY'].transform([county])[0]

        # Target encoded location features
        features['district_avg_price'] = target_encodings['district_avg_price'].get(district,
                                          np.mean(list(target_encodings['district_avg_price'].values())))
        features['county_avg_price'] = target_encodings['county_avg_price'].get(county,
                                        np.mean(list(target_encodings['county_avg_price'].values())))

        # Binary features
        features['is_freehold'] = is_freehold

        # Energy rating
        rating_map = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
        features['energy_rating_numeric'] = rating_map[energy_rating]

        # Interaction features
        features['floor_area_x_rooms'] = floor_area * num_rooms

        # Temporal features
        features['sale_year'] = sale_year
        features['sale_month'] = sale_month

        # Create DataFrame with correct feature order
        feature_df = pd.DataFrame([features])[feature_names]

        # Make prediction
        predicted_price = model.predict(feature_df)[0]

        # Calculate confidence interval (using model MAE)
        mae = metadata['test_mae']
        lower_bound = max(0, predicted_price - mae)
        upper_bound = predicted_price + mae

        # Prepare results
        result = {
            'predicted_price': f"£{predicted_price:,.0f}",
            'lower_bound': f"£{lower_bound:,.0f}",
            'upper_bound': f"£{upper_bound:,.0f}",
            'confidence': f"±£{mae:,.0f}",
            'model_accuracy': f"{metadata['test_r2']*100:.1f}%",
            'inputs': {
                'Floor Area': f"{floor_area:.0f} sqm",
                'Bedrooms': num_rooms,
                'Property Type': f"{property_type} ({PROPERTY_SUBTYPE_MAP.get(property_subtype, property_subtype)})",
                'Construction Period': construction_age,
                'Location': f"{district}, {county}",
                'Tenure': 'Freehold' if is_freehold else 'Leasehold',
                'Energy Rating': energy_rating,
                'Expected Sale': f"{sale_month}/{sale_year}"
            }
        }

        return render_template('result.html', result=result)

    except Exception as e:
        return render_template('result.html',
                             error=f"Error making prediction: {str(e)}")

@app.route('/about')
def about():
    """About page with model information"""
    return render_template('about.html',
                         model_name=metadata['best_model_name'],
                         model_r2=f"{metadata['test_r2']*100:.1f}",
                         model_mae=f"{metadata['test_mae']:,.0f}",
                         model_mape=f"{metadata['test_mape']:.2f}",
                         model_rmse=f"{metadata['test_rmse']:,.0f}",
                         n_features=metadata['n_features'],
                         training_date=metadata['training_date'])

@app.route('/data-sample')
def data_sample():
    """Display a sample of the training data"""
    try:
        # Load a sample from the final dataset
        data_path = '../data/final/modeling_data.csv'

        # Read just a sample for display (100 rows)
        df_sample = pd.read_csv(data_path, nrows=100)

        # Get full dataset info
        df_info = pd.read_csv(data_path, nrows=1000)  # Read more for stats

        # Prepare data for display
        columns = df_sample.columns.tolist()
        data = df_sample.head(20).values.tolist()  # Show 20 rows

        # Format values for display
        formatted_data = []
        for row in data:
            formatted_row = []
            for val in row:
                if isinstance(val, float):
                    if val > 1000:
                        formatted_row.append(f"{val:,.0f}")
                    else:
                        formatted_row.append(f"{val:.2f}")
                else:
                    formatted_row.append(str(val))
            formatted_data.append(formatted_row)

        # Calculate statistics
        total_records = "5.25M"  # From metadata
        num_features = len(columns)
        date_range = "2018-2024"

        if 'price' in df_info.columns:
            price_min = df_info['price'].min()
            price_max = df_info['price'].max()
            price_range = f"£{price_min:,.0f} - £{price_max:,.0f}"
        else:
            price_range = "N/A"

        return render_template('data_sample.html',
                             columns=columns,
                             data=formatted_data,
                             sample_size=20,
                             total_records=total_records,
                             num_features=num_features,
                             date_range=date_range,
                             price_range=price_range)

    except Exception as e:
        return render_template('result.html',
                             error=f"Could not load data sample: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
