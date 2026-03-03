from src.generator import DataEngine
import logging
import os

def main():
    # Configuration
    CUST_COUNT = 200_000
    TX_COUNT = 1_000_000
    
    # Ensure data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
    
    engine = DataEngine(seed=42)
    
    # Execution
    customers = engine.generate_customer_dim(CUST_COUNT)
    transactions = engine.generate_transaction_fact(customers, TX_COUNT)
    customers = engine.run_analytics_engineering(customers, transactions)

    # Save to disk 
    print("Saving files to /data folder...")
    customers.to_csv('data/dim_customers.csv', index=False)
    transactions.to_csv('data/fact_transactions.csv', index=False)

    # Summary report
    print("\nGeneration & Export Complete")
    print(f"Transactions saved: data/fact_transactions.csv")
    print(f"Customers saved:    data/dim_customers.csv")
    print(f"Churn Rate:         {customers['is_churn'].mean():.2%}")

if __name__ == "__main__":
    main()