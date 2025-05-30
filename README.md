# SportsScripts-Final: MLB Analytics Platform

This platform provides comprehensive analytics and prediction tools for various MLB betting markets, organized into specialized applications.

## Contents
- [Platform Overview](#platform-overview)
- [Apps](#apps)
  - [NRFI (No Runs First Inning)](#nrfi-no-runs-first-inning)
  - [Future Apps](#future-apps)
- [Getting Started](#getting-started)
- [Contact & Contributions](#contact--contributions)

---

## Platform Overview

This analytics platform features:

- Modular design with separate specialized applications
- Shared utilities and data pipelines across apps
- Consistent interface and visualization standards
- Extensible architecture for adding new apps

---

## Apps

### NRFI (No Runs First Inning)

The NRFI app predicts games where no runs will be scored in the first inning.

#### Features
- Advanced prediction model integrating lineup, pitcher, and ballpark data
- Time-based backtesting and confidence calibration
- Comprehensive visualizations and analytics
- Daily prediction outputs with confidence ratings

#### Key Files
- **Core Processing:** `Pre-Process.py`, `prepare_enhanced_data.py`, `NRFI.py`
- **Data Integration:** `nrfi_lineup_integration.py`, `enhance_nrfi_model.py`
- **Data Collection:** `Step1.py`, `Step2.py`, `Step3.py`, `Step4.py`
- **Daily Processing:** `daily_lineup_processor.py`
- **Analytics:** `graphs.py`

#### Technical Details

##### NRFI Calculation
```python
df['NRFI'] = ((df['home_inning_1_runs'] == 0) & (df['away_inning_1_runs'] == 0)).astype(int)
```

##### Feature Engineering Examples
- **Batter Hit Rate:** `hit_rate = hits / abs if abs > 0 else 0`
- **Pitcher Hits per PA:** `hits_per_pa = hits / pa if pa > 0 else 0.230`
- **Lineup Strength:** `lineup_strength = mean([batter_scoring_threat for top 4 batters])`
- **Scoring Threat:** `batter_scoring_threat = 0.3 * wOBA + 0.2 * SLG + 0.2 * OBP + 0.2 * Barrel Rate + 0.1 * (1 - K%)`

##### Model Details
- **Algorithm:** Random Forest Classifier with StandardScaler pipeline
- **Prediction:** `y_pred = (nrfi_probabilities >= 0.5).astype(int)`
- **Standard Confidence:** `standard_confidence = abs(nrfi_probabilities - 0.5) * 2`
- **Calibrated Confidence:** Combines historical accuracy, context similarity, and prediction consistency

##### Function Dependencies

| Function/Module | Dependent Functions/Modules |
|----------------|---------------------------|
| `prepare_features` (NRFI.py) | `backtest_model`, `main` (NRFI.py) |
| `calculate_calibrated_confidence` | `backtest_model`, `main` (NRFI.py) |
| `process_batter_stats`/`process_pitcher_stats` | `NRFILineupIntegration`, `prepare_features` |
| `NRFILineupIntegration.generate_features` | `enhance_nrfi_model.py`, `prepare_features` |
| `calculate_lineup_strength` | `prepare_features`, `NRFILineupIntegration` |
| `create_team_nrfi_trend` (graphs.py) | `main` (graphs.py) |
| `create_matchup_comparison` (graphs.py) | `analyze_upcoming_matchups` (graphs.py) |
| `load_batters/pitchers_first_inning_data` | `prepare_features`, `NRFILineupIntegration` |

##### Usage
```cmd
python NRFI/Scripts/NRFI.py
python NRFI/Scripts/graphs.py
```

#### Strengths
- **Advanced Feature Engineering:** Incorporates lineup strength, matchups, weather, and venue
- **Robust Data Pipeline:** Modular scripts for scraping, cleaning, and processing MLB data
- **Backtesting & Calibration:** Time-based backtesting and confidence calibration
- **Visualization Suite:** Trend charts, heatmaps, radar plots, and dashboards
- **Extensible Design:** Easy addition of new features, data sources, or models

#### Limitations
- **Lineup Prediction:** Falls back to historical data when actual lineups are missing
- **Data Quality Gaps:** Some stats are defaulted when missing
- **Model Explainability:** Limited to feature importances without SHAP/LIME integration
- **Data Source Dependency:** Vulnerable to MLB data format changes
- **Manual Data Steps:** Some data preparation requires human intervention

#### Future Improvements
- **Real-time Lineup Integration:** Improve accuracy with confirmed lineups
- **Advanced Imputation:** Better handling of missing data
- **Model Calibration:** Implement Platt scaling or isotonic regression
- **Explainability Tools:** Add SHAP/LIME for better interpretability
- **Environment Factors:** Enhanced weather and ballpark effects
- **Automation:** Build scheduler for daily updates and predictions
- **Testing:** Expand automated tests for data integrity

### Future Apps

The following apps are planned for future development:

#### Full Game Predictions
- Complete game outcome predictions
- Run totals and line analysis
- Specific inning scoring probabilities

#### Player Props
- Batter performance predictions (hits, runs, RBIs)
- Pitcher performance (strikeouts, innings pitched)
- Custom prop bet evaluations

#### Live Betting Advisor
- In-game situation analysis
- Win probability adjustments
- Value betting opportunities

---

## Getting Started

1. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```

2. Select the app you want to use:
   - For NRFI predictions, see [NRFI Usage](#usage)
   - For other apps, follow their specific instructions when available

3. Configure data sources in `config.py` for your selected app

4. Run the desired app's main script

---

## Contact & Contributions

For questions, suggestions, or contributions, please open an issue or submit a pull request.

---

**Last updated:** May 3, 2025
1. Pre-Process.py
- Primary data cleaning and initial processing
- Handles missing values, data validation, and standardization
- Creates the base processed dataset used by other scripts

2. prepare_enhanced_data.py  
- Processes specialized first inning stats for batters and pitchers
- Creates enhanced features for the model
- Runs after Pre-Process.py to add additional metrics

3. NRFI.py
- Main model training and prediction script
- Implements ensemble model with RandomForest, GradientBoosting, ExtraTrees
- Handles backtesting and model evaluation
- Contains core feature engineering logic

4. nrfi_lineup_integration.py
- Handles lineup data integration and processing
- Calculates lineup strength metrics
- Manages batter-pitcher matchup statistics

5. enhance_nrfi_model.py
- Enhances base model with lineup features
- Provides additional model calibration
- Generates enhanced predictions
1. Step1.py
- Scrapes first inning pitcher and batter historical stats

2. Step2.py
- Collects historical game schedules and first inning scores

3. Step3.py 
- Gathers first inning scores and results

4. Step4.py
- Collects historical lineup data

1. daily_lineup_processor.py
- Processes daily lineup data for predictions
- Integrates with live data sources

2. graphs.py
- Generates visualizations and analytics
- Creates team-specific and league-wide trend analysis

3. data_quality.py
- Monitors data quality metrics
- Validates data consistency

4. default_tracker.py
- Tracks default values and data imputation
- Helps monitor data quality

1. config.py
- Central configuration file
- Defines paths, constants, and settings
1. Dashboard.py (mentioned in README but not found in codebase)
- Could be removed from README as it's not implemented yet

2. weather.py
- Could be integrated into Pre-Process.py as weather handling is fairly simple

3. tests/ directory
- While testing is important, these appear to be incomplete/placeholder files
- Should be properly implemented or removed until ready

##### Execution Order
1. Data Collection
   ```cmd
   python NRFI/Scripts/Step1.py
   python NRFI/Scripts/Step2.py
   python NRFI/Scripts/Step3.py
   python NRFI/Scripts/Step4.py
   ```
2. Data Processing
   ```cmd
   python NRFI/Scripts/Pre-Process.py
   python NRFI/Scripts/prepare_enhanced_data.py
   ```
python NRFI/Scripts/prepare_enhanced_data.py
python NRFI/Scripts/NRFI.py
python NRFI/Scripts/daily_lineup_processor.py
python NRFI/Scripts/enhance_nrfi_model.py
python NRFI/Scripts/graphs.py