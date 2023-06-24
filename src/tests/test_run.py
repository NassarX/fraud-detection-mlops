"""
This module contains an example test.

Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py. They are simply functions
named ``test_*`` which test a unit of logic.

To run the tests, run ``kedro test`` from the project root directory.
"""
import numpy as np
import pandas as pd
import pytest
from kedro.config import ConfigLoader
from kedro.framework.hooks import _create_hook_manager
from pathlib import Path
from kedro.framework.context import KedroContext

from src.fraud_detection.pipelines.etl_app.nodes.data_generation import (
    generate_customer_profiles_data,
    generate_terminals_data,
    generate_dataset,
    generate_transactions_data,
    generate_fraud_Scenarios_data,
)

@pytest.fixture
def config_loader():
    return ConfigLoader(conf_source=str(Path.cwd()))

@pytest.fixture
def project_context(config_loader):
    return KedroContext(
        package_name="fraud_detection",
        project_path=Path.cwd(),
        config_loader=config_loader,
        hook_manager=_create_hook_manager(),
    )


class TestProjectContext:
    def test_project_path(self, project_context):
        assert project_context.project_path == Path.cwd()
        assert project_context._package_name == "fraud_detection"

    def test_generate_customer_profiles_data(self, project_context):
        # Define the expected column names in the generated dataframe
        expected_columns = [
            'CUSTOMER_ID',
            'x_customer_id',
            'y_customer_id',
            'mean_amount',
            'std_amount',
            'mean_nb_tx_per_day'
        ]

        # Define the number of customers to generate
        n_customers = 100

        # Call the function to generate customer profiles data
        customer_profiles_table = generate_customer_profiles_data(n_customers)

        # Assert that the generated dataframe has the expected column names
        assert list(customer_profiles_table.columns) == expected_columns

        # Assert that the number of rows in the generated dataframe matches the number of customers
        assert len(customer_profiles_table) == n_customers

        # Assert that the generated dataframe does not contain any NaN values
        assert not customer_profiles_table.isnull().values.any()

        # Assert that the generated dataframe has the correct data types for each column
        assert customer_profiles_table.dtypes['CUSTOMER_ID'] == np.int64
        assert customer_profiles_table.dtypes['x_customer_id'] == np.float64
        assert customer_profiles_table.dtypes['y_customer_id'] == np.float64
        assert customer_profiles_table.dtypes['mean_amount'] == np.float64
        assert customer_profiles_table.dtypes['std_amount'] == np.float64
        assert customer_profiles_table.dtypes['mean_nb_tx_per_day'] == np.float64

    

    def test_generate_terminals_data(self, project_context):
        # Define the expected column names in the generated dataframe
        expected_columns = [
            'TERMINAL_ID',
            'x_terminal_id',
            'y_terminal_id'
        ]

        # Define the number of terminals to generate
        n_terminals = 50

        # Call the function to generate terminals data
        terminal_data = generate_terminals_data(n_terminals)

        # Assert that the generated dataframe has the expected column names
        assert list(terminal_data.columns) == expected_columns

        # Assert that the number of rows in the generated dataframe matches the number of terminals
        assert len(terminal_data) == n_terminals

        # Assert that the generated dataframe does not contain any NaN values
        assert not terminal_data.isnull().values.any()

        # Assert that the generated dataframe has the correct data types for each column
        assert terminal_data.dtypes['TERMINAL_ID'] == np.int64
        assert terminal_data.dtypes['x_terminal_id'] == np.float64
        assert terminal_data.dtypes['y_terminal_id'] == np.float64

    
    def test_generate_dataset_invalid_date(self, project_context):
        n_customers = 100
        n_terminals = 50
        radius = 10.0
        start_date = '2023/01/01'
        nb_days = 7
        
        with pytest.raises(ValueError):
            generate_dataset(n_customers, n_terminals, radius, start_date, nb_days)

    def test_select_terminals_within_customer_radius(self, project_context):
        
        # load the customer profiles table
        customer_profiles_table = pd.read_csv('data/01_raw/03_customers_terminals_data.csv')
        
        # extract one row as example
        example = eval(customer_profiles_table['available_terminals'][0])
        assert len(example) > 0

        # Assert that all values in the example list are integers
        for value in example:
            assert isinstance(value, int)
            
        # Assert that all values in the example list are integers
        for value in customer_profiles_table['nb_terminals']:
            assert isinstance(value, int)
            
            
    def test_generate_transactions_data(self):

        # load the customer profiles table
        customer_profiles_table = pd.read_csv('data/01_raw/03_customers_terminals_data.csv')  
              
        # Define the expected column names in the generated dataframe
        expected_columns = [
        "TRANSACTION_ID",
        'TX_DATETIME',
        'CUSTOMER_ID',
        'TERMINAL_ID',
        'TX_AMOUNT',
        'TX_TIME_SECONDS',
        'TX_TIME_DAYS']
    
        start_date = "2020-01-01"
        nb_days = 30

        # Call the function to generate terminals data
        transactions_data = generate_transactions_data(customer_profiles_table, start_date, nb_days)

        # Assert that the generated dataframe has the expected column names
        assert list(transactions_data.columns) == expected_columns  

        # Assert that the generated dataframe does not contain any NaN values
        assert not transactions_data.isnull().values.any()
       
        # Assert that the generated dataframe has the correct data types for each column
        assert transactions_data.dtypes['TRANSACTION_ID'] == np.int64
        assert transactions_data.dtypes['CUSTOMER_ID'] == np.int64
        assert transactions_data.dtypes['TERMINAL_ID'] == np.int64
        assert transactions_data.dtypes['TX_AMOUNT'] == np.float64
        assert transactions_data.dtypes['TX_TIME_SECONDS'] == np.int64
        assert transactions_data.dtypes['TX_TIME_DAYS'] == np.int64        


    def test_generate_fraud_Scenarios_data(self):
        
        # load the customer profiles table
        fraud_scenarios_data = pd.read_csv('data/01_raw/06_fraud_transactions_data.csv')  

        # Find in TX_FRAUD column a value greater than 220
        # Assuming transactions_data is your DataFrame
        index_above_220 = fraud_scenarios_data.loc[fraud_scenarios_data['TX_AMOUNT'] > 220, 'TX_AMOUNT'].idxmax()
        index_below_220 = fraud_scenarios_data.loc[fraud_scenarios_data['TX_AMOUNT'] <= 220, 'TX_AMOUNT'].idxmax()
        
        # Access the value in the TX_FRAUD column at the found index
        fraud_above = fraud_scenarios_data.loc[index_above_220, 'TX_FRAUD']
        not_fraud = fraud_scenarios_data.loc[index_below_220, 'TX_FRAUD']
        
        # Assert that the value of fraud is equal to 1 and not_fraud is equal to 0
        assert fraud_above == 1
        assert not_fraud == 0
        
        # assert TX_FRAUD_SCENARIO
        index_fraud_1 = fraud_scenarios_data.loc[fraud_scenarios_data['TX_FRAUD_SCENARIO'] == 1, 'TX_FRAUD_SCENARIO'].idxmax()
        fraud_1 = fraud_scenarios_data.loc[index_fraud_1, 'TX_AMOUNT']  
        
        assert fraud_1 > 220
                                
        # Assert that in TX_FRAUD_SCENARIO exist only until 3 scenarios
        assert fraud_scenarios_data['TX_FRAUD_SCENARIO'].max() <= 3   
        
        assert fraud_scenarios_data.dtypes['TX_FRAUD_SCENARIO'] == np.int64
        assert fraud_scenarios_data.dtypes['TX_FRAUD'] == np.int64

        # Select only the rows where TX_FRAUD_SCENARIO different from 0
        fraud_scenarios_data_1 = fraud_scenarios_data[fraud_scenarios_data['TX_FRAUD_SCENARIO'] != 0]
        # Assert that the values of tx_fraud are 1
        assert fraud_scenarios_data_1['TX_FRAUD'].unique() == [1]

        # Select only the rows where TX_FRAUD_SCENARIO equals to 0
        fraud_scenarios_data_0 = fraud_scenarios_data[fraud_scenarios_data['TX_FRAUD_SCENARIO'] == 0]
        # Assert that the values of tx_fraud are 0
        assert fraud_scenarios_data_0['TX_FRAUD'].unique() == [0]
                                        
if __name__ == "__main__":
    pytest.main(["-v", __file__])