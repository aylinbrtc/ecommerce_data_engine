# E-Commerce Data Warehouse Engine (1.2M Rows)

A high-performance Python engine developed for the Spring 2026 Data Mining course. This project generates a simulated e-commerce environment following a Star Schema, specifically designed to test Preprocessing, Association Rule Mining, and Classification algorithms.

## Technical Architecture
The engine is built on a **vectorized execution model**. By utilizing `numpy` and `pandas` array operations, the system bypasses the Python interpreter's Global Interpreter Lock (GIL) limitations during data generation, allowing for million-row throughput in seconds.

### Data Model
- **Fact Table (`fact_transactions`)**: 1M rows. Tracks ID, Customer_ID, Timestamp, Category, and Financials.
- **Dimension Table (`dim_customers`)**: 200k rows. Tracks demographic data and behavioral flags.
- **Star Schema**: Normalized on `customer_id` for efficient JOIN operations and OLAP "Drill-Down" analysis.

## Key Data Mining Features
- **Data Cleaning**: Automated injection of 0.5% null values and extreme age outliers (Age=150) to facilitate preprocessing pipelines.
- **Association Rules**: Probabilistic hidden patterns injected into subcategories (Gaming PC ➔ Mechanical Keyboard) to provide ground-truth for Apriori/FP-Growth.
- **Classification**: Programmatically derived `is_churn` labels based on a combination of Poisson-distributed support tickets and lifetime value (LTV).

## Usage
Ensure you are using Python 3.9+ and install requirements:
```bash
pip install -r requirements.txt
python main.py
