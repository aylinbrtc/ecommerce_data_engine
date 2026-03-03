import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DataEngine:
    """
    High-performance synthetic data generator for e-commerce analytics.
    Utilizes vectorized operations to simulate star-schema relational data.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(self.seed)
        self.regions = ['NA', 'EU', 'APAC', 'LATAM']
        self.tiers = ['Bronze', 'Silver', 'Gold', 'Platinum']

    def generate_customer_dim(self, n_rows: int = 200_000) -> pd.DataFrame:
        logger.info(f"Generating {n_rows} customer records...")
        
        df = pd.DataFrame({
            'customer_id': np.arange(1, n_rows + 1),
            'age': np.random.normal(34, 11, n_rows).astype(int),
            'gender': np.random.choice(['M', 'F', 'O'], n_rows, p=[0.49, 0.49, 0.02]),
            'region': np.random.choice(self.regions, n_rows),
            'membership': np.random.choice(self.tiers, n_rows, p=[0.4, 0.3, 0.2, 0.1]),
            'support_tickets': np.random.poisson(1.2, n_rows)
        })

        # Preprocessing exercise: Outlier injection
        df.loc[np.random.choice(df.index, 50), 'age'] = 150 
        df['age'] = df['age'].clip(lower=18)
        
        return df

    def generate_transaction_fact(self, cust_df: pd.DataFrame, n_rows: int = 1_000_000) -> pd.DataFrame:
        logger.info(f"Generating {n_rows} transaction records...")
        
        # Proper vectorized date generation
        start_date = pd.Timestamp('2025-01-01')
        random_seconds = np.random.randint(0, 31536000, n_rows)
        dates = start_date + pd.to_timedelta(random_seconds, unit='s')
        
        df = pd.DataFrame({
            'tx_id': np.arange(1, n_rows + 1),
            'customer_id': np.random.choice(cust_df['customer_id'], n_rows),
            'date': dates,
            'category': np.random.choice(['Electronics', 'Apparel', 'Home'], n_rows),
            'unit_price': np.random.uniform(15.0, 800.0, n_rows).round(2),
            'qty': np.random.randint(1, 6, n_rows),
            'discount_pct': np.random.choice([0, 0.05, 0.1, 0.2], n_rows, p=[0.6, 0.2, 0.1, 0.1])
        })

        # Default subcategory
        df['subcategory'] = 'General'

        #Gaming PC -> Mechanical Keyboard association
        pc_idx = df.sample(frac=0.1).index
        df.loc[pc_idx, 'subcategory'] = 'Gaming PC'
        
        kb_idx = np.random.choice(pc_idx, size=int(len(pc_idx)*0.8), replace=False)
        df.loc[kb_idx, 'subcategory'] = 'Mechanical Keyboard'

        df['total_amount'] = (df['unit_price'] * df['qty'] * (1 - df['discount_pct'])).round(2)

        # NaN and Negative Outliers
        df.loc[df.sample(5000).index, 'total_amount'] = np.nan
        df.loc[df.sample(100).index, 'total_amount'] = -99.99
        
        return df

    def run_analytics_engineering(self, cust_df: pd.DataFrame, tx_df: pd.DataFrame) -> pd.DataFrame:
        spend_map = tx_df.groupby('customer_id')['total_amount'].sum()
        cust_df = cust_df.set_index('customer_id')
        cust_df['total_spend'] = spend_map
        cust_df = cust_df.reset_index().fillna(0)

        #Classification Labeling
        cust_df['is_churn'] = ((cust_df['support_tickets'] > 3) & (cust_df['total_spend'] < 300)).astype(int)
        return cust_df