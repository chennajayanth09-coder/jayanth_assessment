# Dice Game – Data Processing & Analysis

This project processes raw CSV data from the Dice Game system into a clean data warehouse format, performs data quality checks, and generates basic insights.

## How to Run

1. Install dependencies:
   pip install pandas numpy pytest

2. Run the ETL process:
   python dice_game_etl.py

## Output

After running, you’ll find:

- Dimension & fact tables in `warehouse/csv/`:
  - `dim_*.csv` – dimension tables
  - `fact_*.csv` – fact tables
- Data quality results: `dq_results.csv` (PASS/FAIL checks)
- Insights:
  - `insight_sessions_by_channel.csv`
  - `insight_plan_adoption.csv`
- Console output showing estimated 2024 revenue.

## Data Model

Based on the provided “Dice Game Data Model - Published.pdf”:

- fact_play_session – session details with user, channel, status, score, and duration  
- fact_user_plan – user plan subscriptions and periods  
- fact_payment_detail – payment method details  
- dim_user – user profile information  
- dim_user_registration – registration records  
- dim_plan – plan pricing and payment frequency  
- dim_payment_frequency – descriptions for payment frequencies  
- dim_channel – channel descriptions  
- dim_status – status code descriptions  

## Running Tests

Once the ETL has generated the CSV outputs, you can run:

pytest -q
