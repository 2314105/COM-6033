# UK Property Price Predictor
## Machine Learning-Based Property Valuation System

A comprehensive machine learning project that predicts UK property prices using Energy Performance Certificate (EPC) data and historical property transactions. Built with Python, scikit-learn, XGBoost, and Flask.

---

## Project Overview

This project combines two large UK government datasets to build an accurate property price prediction model:
- **11.1 million** Energy Performance Certificate records (2018-2024)
- **7.1 million** property transactions from Land Registry (2018-2024)

**Final Model Performance:**
- **R² Score:** 72.8% (explains 72.8% of price variance)
- **Mean Absolute Error:** £74,502
- **Mean Absolute Percentage Error:** ~26%

---

## Project Structure

```
Artifact/
├── data/
│   ├── raw/              # Raw datasets (not included - see download links below)
│   ├── processed/        # Intermediate cleaned data
│   └── final/            # Final feature-engineered datasets
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
├── models/               # Trained models and encoders
├── webapp/               # Flask web application
└── README.md
```

---

## Data Sources & Download Instructions

### Important: Large Datasets
The raw datasets are **NOT included** in this repository due to their size (multiple GB). You must download them separately.

### 1. Energy Performance Certificate (EPC) Data
**Source:** UK Department for Levelling Up, Housing and Communities  
**URL:** https://epc.opendatacommunities.org/

**Download Instructions:**
1. Visit the EPC Open Data Portal
2. Navigate to "Domestic certificates"
3. Download data for each year (2018-2024)
4. Save files as: `data/raw/epc-2018.csv`, `epc-2019.csv`, etc.

**Direct Download (alternative):**
- All domestic EPCs: https://epc.opendatacommunities.org/docs/api/domestic
- You can use their API or bulk download options

**File naming convention:**
```
data/raw/epc-2018.csv
data/raw/epc-2019.csv
data/raw/epc-2020.csv
data/raw/epc-2021.csv
data/raw/epc-2022.csv
data/raw/epc-2023.csv
data/raw/epc-2024.csv
```

### 2. Property Price Paid Data
**Source:** HM Land Registry  
**URL:** https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads

**Download Instructions:**
1. Visit the Price Paid Data Downloads page
2. Download "Complete file" for each year (2018-2024)
3. Save files as: `data/raw/pp-2018.csv`, `pp-2019.csv`, etc.

**Direct Download Links:**
- 2024: http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2024.csv
- 2023: http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2023.csv
- 2022: http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2022.csv
- 2021: http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2021.csv
- 2020: http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2020.csv
- 2019: http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2019.csv
- 2018: http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2018.csv

**File naming convention:**
```
data/raw/pp-2018.csv
data/raw/pp-2019.csv
data/raw/pp-2020.csv
data/raw/pp-2021.csv
data/raw/pp-2022.csv
data/raw/pp-2023.csv
data/raw/pp-2024.csv
```

### Expected File Sizes
- **EPC files:** ~1-2 GB each (11.1M total records)
- **Property Price files:** ~100-150 MB each (7.1M total records)
- **Total raw data:** ~15-20 GB

###  Sample Dataset for Demonstration

**For submission/demonstration purposes**, a sample dataset is included:
- **File:** `data/final/modeling_data_sample.csv`
- **Size:** ~98 MB (180,000 records)
- **Purpose:** Demonstrate the pipeline and test the web app
- **Sampling method:** Stratified random sampling maintaining distribution
- **Note:** For production use, retrain with full dataset (5.25M records)

**To generate the full dataset:**
1. Download raw data (see instructions above)
2. Run notebooks 01-04 in order
3. In Notebook 02, set `SAVE_FULL_DATASET = True` before running the save cell

**The sample dataset is representative:**
- Maintains price distribution (mean, median, range)
- Includes all property types and locations
- Sufficient for demonstrating model functionality
- Web app works with either sample or full dataset

---

## Notebook Walkthrough

### Notebook 01: Data Exploration

**Purpose:** Initial exploration and understanding of the datasets

**What has been done:**
- Loaded sample EPC and Property Price datasets
- Examined data structure, column types, and distributions
- Identified key features for price prediction
- Analyzed data quality issues (missing values, outliers)
- Explored relationships between features and price
- Visualized price distributions, property types, and energy ratings

**Key Findings:**
- EPC dataset has 94 columns with detailed energy metrics
- Property prices range from £10K to £900M (with outliers)
- Strong correlation between floor area and price (0.44)
- Location is the most important factor
- Energy ratings show some correlation with price

---

### Notebook 02: Data Cleaning

**Purpose:** Clean and merge the two datasets

**What has been done:**

1. **Loaded all years** (2018-2024) for both datasets
   - EPC: 11.1M records
   - Property prices: 7.1M records

2. **Cleaned EPC data:**
   - Selected 21 relevant columns
   - Standardized postcodes
   - Removed outliers (floor area 20-500 sqm)
   - Filled missing values
   - Result: 11.1M → 11.0M records (99.3% retention)

3. **Cleaned Property Price data:**
   - Removed missing postcodes
   - Filtered extreme prices (£10K-£5M)
   - Kept only standard transactions
   - Result: 7.1M → 5.9M records (83.7% retention)

4. **Merged datasets:**
   - Matched properties by postcode
   - Ensured EPC was lodged before sale date
   - Kept most recent EPC for each sale
   - Result: 5.25M matched pairs (88.9% match rate)

---

### Notebook 03: Feature Engineering

**Purpose:** Create meaningful features for machine learning

**What has been done:**

1. **Created temporal features:** sale_year, sale_month, EPC age

2. **Created binary features:** is_new_build, is_freehold, has_mains_gas

3. **Encoded categorical features:** Property type, age, location

4. **Target encoding (KEY IMPROVEMENT):**
   - `district_avg_price` - districts encoded by average price
   - `county_avg_price` - counties encoded by average price
   - This boosted R² from 68.8% → 72.8%

5. **Interaction features:** floor_area_x_rooms

6. **Simplified to 15 features** focused on: size, location, type, age

**Key improvement:**  
Target encoding for location captures real price relationships instead of arbitrary numbers.

---

### Notebook 04: Model Training

**Purpose:** Train and evaluate ML models

**What has been done:**

1. **Train/test split:** 80/20 (4.2M / 1.05M records)

2. **Trained 4 models:**
   - Linear Regression: R² = 0.238 
   - **Random Forest: R² = 0.728**  (winner)
   - Gradient Boosting: R² = 0.688
   - XGBoost: R² = 0.675

3. **Evaluated with multiple metrics:**
   - RMSE, MAE, MAPE, R²

4. **Feature importance analysis:**
   - Floor area: 31.8%
   - District: 27.5%
   - County: 17.4%

**Winner:** Random Forest with 72.8% R²

---

##  Quick Start

### 1. Download Data
Follow instructions above to download EPC and Property Price data (15-20 GB total).

### 2. Install Dependencies
```bash
# Install all requirements at once
pip install -r requirements.txt

# This includes:
# - Data science: pandas, numpy, matplotlib, seaborn
# - Machine learning: scikit-learn, xgboost
# - Web app: Flask
# - Notebooks: jupyter
```

### 3. Run Notebooks
```bash
jupyter notebook
# Run in order: 01 → 02 → 03 → 04
```

### 4. Launch Web App
```bash
cd webapp
python app.py
# Open http://localhost:5001
```

---

##  Final Model: 15 Features

| Feature | Type | Importance |
|---------|------|---------|
| Floor area | Numeric | 31.8% |
| District avg price | Target encoded | 27.5% |
| County avg price | Target encoded | 17.4% |
| Property sub-type | Categorical | 6.1% |
| Construction age | Categorical | 3.9% |
| Freehold/Leasehold | Binary | 3.9% |
| Property type | Categorical | 3.6% |
| Floor × Rooms | Interaction | -  |
| Bedrooms | Numeric | 0.7% |
| Energy rating | Ordinal | 0.9% |
| Sale year/month | Temporal | 4.1% |

---

##  Summary

### Performance Evolution

| Stage | R² | MAE | Change |
|-------|-----|-----|-------|
| Baseline (label encoding) | 68.8% | £82,754 | - |
| **Final (target encoding)** | **72.8%** | **£74,502** | +4%  |

### What This Means
- **72.8% R²:** Model explains 72.8% of price variance
- **£74,502 MAE:** Average error of ~£75K
- **~26% MAPE:** About 26% relative error

For a £300K property: prediction range £225K-£375K

### Key Achievements
- Trained on 5.2M properties  
- Target encoding boosted performance by 4%  
- Simple, interpretable 15 features  
- merged two data sets, after low r2 score with pp only data
- No overfitting, good generalization  

### Limitations
 - District-level location (not street-specific)  
 - Missing: condition, renovations, parking, amenities  
 - 26% average error - good but not perfect  

### Key Learnings
1. **Location is the most critical** (45% importance combined)
2. **Target encoding >> label encoding** for location
3. **Tree models >> Linear** for tabular data
4. **Feature engineering matters** more than model complexity
5. **Data quality and cleaning** are critical

---

## Project Reflection

### Design Decisions and Rationale

**1. Merging Two Independent Datasets**
- **Decision:** Combine EPC and Land Registry data using postcode + temporal matching
- **Challenge:** No common ID between datasets, multiple properties per postcode
- **Solution:** Match on postcode, filter EPCs lodged before sale date, keep most recent
- **Rationale:** EPC data adds rich structural/energy features that price-only data lacks
- **Result:** 88.9% match rate, 5.25M training records
- **Alternative considered:** Use Land Registry data only (rejected - achieved only 40% R² in initial tests)

**2. Target Encoding for Location**
- **Decision:** Encode districts/counties by average price instead of arbitrary labels
- **Challenge:** 361 districts - one-hot encoding would create 361 columns
- **Solution:** Replace each district with its mean property price
- **Rationale:** Preserves ordinal price relationship (expensive vs cheap areas)
- **Result:** +4% R² improvement (68.8% → 72.8%)
- **Alternative considered:** Label encoding (rejected - implies incorrect ordinality), one-hot encoding (rejected - too many features, sparse data)

**3. Price Range Filtering (£10K-£5M)**
- **Decision:** Remove transactions below £10K and above £5M
- **Challenge:** Dataset includes non-market transactions (gifts, ultra-luxury)
- **Rationale:** Model should predict typical residential market value
- **Impact:** Removed 16% of transactions, but vastly improved model focus
- **Ethical consideration:** Excludes social housing transfers and family gifts - users must understand model predicts "market value" not all transactions

**4. Feature Simplification (15 features)**
- **Decision:** Reduce from 93 EPC columns + 17 price columns to 15 engineered features
- **Rationale:** Avoid overfitting, improve interpretability, easier deployment
- **Approach:** Correlation analysis + domain knowledge + iterative testing
- **Result:** Maintained 72.8% R² with simpler, more robust model
- **Alternative considered:** Include all 110 columns (rejected - risk of overfitting, harder to explain)

**5. Model Selection: Gradient Boosting**
- **Decision:** Choose Gradient Boosting over Random Forest (70.8% vs 70.5% test R²)
- **Challenge:** Minimal difference between top 3 tree models
- **Rationale:** Best test RMSE, good train/test balance
- **Result:** Production-ready model with excellent generalization
- **Note:** Random Forest would have worked almost as well - robust to model choice

### Challenges Encountered and Solutions

**Challenge 1: Data Scale (18M+ Records)**
- **Problem:** Cannot load or merge 11M × 5.9M records simultaneously
- **Impact:** Memory errors, impossible to process on standard machines
- **Solution:** Year-by-year processing, chunk-based merging
- **Learning:** Large-scale data requires different strategies than Kaggle datasets
- **Future improvement:** Could use Dask or Spark for distributed processing

**Challenge 2: Postcode Standardization**
- **Problem:** Minor formatting differences prevented matches (spaces, case)
- **Example:** "SW1A 1AA" vs "SW1A1AA" vs "sw1a 1aa"
- **Solution:** Strip whitespace, uppercase all postcodes before merging
- **Learning:** Real-world data needs defensive preprocessing
- **Impact:** Improved match rate from 75% → 88.9%

**Challenge 3: Temporal Complexity**
- **Problem:** Properties can have multiple EPCs over time
- **Question:** Which EPC to use for a given sale?
- **Solution:** Most recent EPC before sale date (reflects property condition at sale time)
- **Alternative considered:** Most recent EPC ever (rejected - could be after sale, not representative)
- **Validation:** Manual spot-checks confirmed sensible matching

**Challenge 4: Missing Value Strategy**
- **Problem:** 58% of records missing county, 15% missing room counts
- **Options:** Drop rows (lose data) or impute (introduce assumptions)
- **Solution:** Strategic imputation - mode for categorical, median for numeric
- **Rationale:** Preserves 99.3% of EPC data, maintains statistical distributions
- **Risk acknowledged:** Slightly biases toward "average" properties for missing data

**Challenge 5: Evaluation Metric Selection**
- **Problem:** Different metrics favored different models
- **RMSE:** Gradient Boosting best
- **R²:** Tie between GBM and RF
- **Decision:** Prioritize RMSE (penalizes large errors appropriately for property valuation)
- **Rationale:** Being £500K off is worse than 2× being £250K off
- **Learning:** Choose metrics aligned with business use case

### What Worked Well

- **Systematic pipeline approach:** Clear progression (explore → clean → engineer → train) kept project organized
- **Documentation throughout:** Can explain every decision to stakeholders
- **Feature engineering focus:** Target encoding innovation drove performance gain
- **Multiple model comparison:** Data-driven selection, not assumptions
- **Validation at each step:** Caught errors early (e.g., postcode formatting issues)
- **Domain knowledge application:** Understanding UK property market informed decisions

### What Could Be Improved

 **Hyperparameter tuning:** Used reasonable defaults, not optimized
- **Current:** Manual selection based on experience
- **Could do:** GridSearchCV, RandomizedSearchCV, Bayesian optimization
- **Expected gain:** +1-2% R²
- **Trade-off:** Significant time investment for marginal improvement

 **Cross-validation:** Used single 80/20 split
- **Current:** One train/test split (random_state=42)
- **Could do:** 5-fold or 10-fold cross-validation
- **Benefit:** More robust performance estimates
- **Trade-off:** 5-10× longer training time

 **Full dataset usage:** Trained on 500K sample for development
- **Current:** Sampled for iteration speed
- **Production deployment:** Should retrain on full 5.25M
- **Expected gain:** +2-3% R² from more training data

 **Additional features:** Could engineer more complex features
- **Examples:** Distance to transport, school quality scores, crime rates
- **Challenge:** External data acquisition and integration
- **Benefit:** Potentially capture additional 5-10% variance

 **Ensemble stacking:** Could combine multiple models
- **Approach:** Use RF + GBM + XGBoost predictions as features
- **Expected gain:** +1-2% R²
- **Trade-off:** Complexity in deployment and explanation

**Why Not?:** Diminishing returns principle - 72.8% R² meets project requirements, additional complexity not justified for marginal gains.

### Future Improvements (If Deployed to Production)

1. **Model retraining pipeline:** Quarterly updates with new market data
2. **Drift monitoring:** Track performance decay over time
3. **Explainability features:** SHAP values to explain individual predictions
4. **Confidence scores:** Predict uncertainty (standard deviation estimates)
5. **A/B testing:** Compare model versions in production
6. **External data integration:** School ratings, transport links, crime statistics
7. **Regional models:** Separate models for London vs regions (different market dynamics)
8. **User feedback loop:** Collect actual sale prices to improve model

### Key Learnings

**Technical Learnings:**
1. **Feature engineering > model complexity:** Target encoding (+4% R²) mattered more than Random Forest vs XGBoost
2. **Data quality is paramount:** Cleaning and merging took 60% of project time but enabled everything else
3. **Scale requires different approaches:** Techniques that work on 10K rows don't work on 5M rows
4. **Multiple metrics tell complete story:** R², MAE, RMSE, MAPE each provide different insights
5. **Tree ensembles excel on tabular data:** All tree models dramatically outperformed linear regression

**Process Learnings:**
1. **Document decisions in real-time:** Impossible to remember rationale weeks later
2. **Validate early and often:** Caught postcode formatting issue before full merge
3. **Iterate on samples first:** Faster experimentation on 500K sample before committing to 5.25M
4. **Domain knowledge is essential:** Understanding UK property market guided outlier thresholds
5. **Visualize everything:** Patterns in plots that weren't obvious in summary statistics

**Domain Insights:**
1. **Location dominates pricing:** 45% of model importance from location features
2. **Size is second:** Floor area and rooms = 32% combined importance
3. **Energy efficiency undervalued:** Only 0.9% importance despite prominence in dataset
4. **Market timing matters:** Sale year captures inflation and economic cycles (4% importance)
5. **Property type effects:** Detached houses command premium, flats less (6% importance)

---

## Ethical Considerations

### Responsible AI and Fair Use

### 1. Bias and Fairness

**Location-Based Pricing:**
-  **Risk:** Encoding locations by average price could perpetuate existing geographic inequalities
-  **Redlining concerns:** Systematically undervaluing certain postcodes could lead to discriminatory outcomes
-  **Our approach:** Use publicly observable market data, not protected characteristics (race, religion, etc.)
-  **Transparency:** Users can see which areas are valued higher/lower (not a black box)
-  **Not suitable for:** Lending decisions, insurance pricing, or other regulated applications without human oversight

**Data Filtering Decisions:**
-  **Excluded transactions:** Social housing, family transfers, gifts (< £10K sales)
-  **Impact:** Model cannot value non-market transactions
-  **Justification:** Model predicts market value, not all transaction types
️ **Users must understand:** This is a limitation, not a feature

**Missing Data Imputation:**
- ️ **Risk:** Imputing missing values with mode/median biases toward "average" properties
-  **Impact:** Properties with unusual characteristics may be less accurately valued
-  **Mitigation:** Document imputation strategy, maintain 99% data retention
-  **Alternative:** Dropping missing data would introduce worse bias (against older properties)

### 2. Privacy and Data Use

**Public Data Sources:**
-  **All data is public:** EPC and Land Registry data are government open data
-  **No personal information:** No names, contact details, or identifiers
-  **Aggregate patterns:** Model could identify pricing patterns in specific areas
-  **Responsible use:** Should not be used to discriminate against specific postcodes

**Data Retention:**
-  **Read-only access:** We don't collect or store user queries
-  **No tracking:** Web app doesn't log user inputs or predictions
- ️ **Production deployment:** Would require GDPR-compliant privacy policy

### 3. Model Limitations and Transparency

**What the Model Does NOT Know:**
-  Property condition, renovations, maintenance
-  Views, gardens, parking availability
-  School quality, local amenities
-  Crime rates, transport links
-  Recent comparable sales on same street
-  Market sentiment, buyer urgency

**Performance Limitations:**
- ️ **Average error:** £79K MAE means significant uncertainty
- ️ **For £300K property:** Prediction range £221K-£379K (±26%)
- ️ **Not suitable for:** Final valuations, mortgage lending decisions
-  **Appropriate for:** Initial estimates, identifying mispriced properties, educational purposes

**Temporal Limitations:**
- ️ **Training data:** 2018-2024 UK property market
- ️ **Market changes:** Model reflects past, not future market conditions
- ️ **Requires retraining:** Should be updated quarterly with new data
-  **Won't work for:** Other countries, extreme market shifts (crashes, booms)

### 4. Use Case Appropriateness

** Appropriate Uses:**
- Initial property valuation estimates
- Identifying potentially over/underpriced properties
- Educational demonstrations of ML in real estate
- Researching property market trends
- Comparing different areas/property types

** Inappropriate Uses:**
- Replacing professional RICS valuations
- Final mortgage lending decisions
- Insurance premium calculations
- Automated property purchase/sale decisions
- Legal or tax valuations
- Discriminating against specific areas or property types

**️ Requires Human Oversight:**
- Mortgage pre-approval (as one data point among many)
- Investment property screening (combined with site visits)
- Market research (supplemented with local knowledge)

### 5. Environmental and Social Impact

**Energy Efficiency Findings:**
-  **Data insight:** Energy rating has only 0.9% importance in pricing
-  **Interpretation:** UK market undervalues energy-efficient properties
-  **Climate impact:** This is problematic given climate change urgency
-  **Positive use case:** Could help identify undervalued efficient properties
- ️ **Model reflects reality:** We show what market values, not what it should value

**Social Housing:**
- ️ **Excluded from model:** Social housing transfers and right-to-buy schemes
- ️ **Reasoning:** Different market dynamics, not market-rate transactions
- ️ **Impact:** Model doesn't serve affordable housing sector
-  **Transparent limitation:** Documented and explained

---

## Technical Stack

- **Python 3.12**
- **Pandas, NumPy** (data processing)
- **Scikit-learn, XGBoost** (ML)
- **Flask** (web framework)
- **Matplotlib, Seaborn** (visualization)

---

**Model:** Random Forest (R² = 72.8%, MAE = £74,502)
