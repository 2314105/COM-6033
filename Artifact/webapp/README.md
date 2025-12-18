
## Quick Start

### 1. Install Dependencies

```bash
# From project root directory
pip install -r requirements.txt

```

### 2. Ensure Models are Trained

Make sure you've run the Jupyter notebooks to generate:
- `../models/best_model_*.pkl` - Trained model
- `../models/label_encoders.pkl` - Categorical encoders
- `../models/target_encodings.pkl` - Location encoders
- `../models/model_metadata.pkl` - Model information

### 3. Run the Application

```bash
python app.py
```

The app will start at: http://localhost:5000

## Usage

1. **Open your browser** to http://localhost:5000
2. **Fill in property details**:
   - Size & layout (floor area, bedrooms)
   - Property type (house, flat, etc.)
   - Location (district, county)
   - Tenure and energy rating
   - Expected sale date
3. **Click "Get Price Prediction"**
4. **View results** with predicted price and confidence interval

