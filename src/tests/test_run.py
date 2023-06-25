"""
This module contains an example test.

Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py. They are simply functions
named ``test_*`` which test a unit of logic.

To run the tests, run ``kedro test`` from the project root directory.
"""
import numpy as np
import matplotlib.pyplot as plt
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
    select_terminals_within_customer_radius
    
)

from src.fraud_detection.pipelines.etl_app.nodes.data_exploration import (
    plot_transactions_daily_stats,
    plot_transactions_distribution
)

from src.fraud_detection.pipelines.etl_app.nodes.feature_transformation import (
    get_customer_spending_behaviour_features
    
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
    # Create a customer profile
    customer_profile = {
        'x_customer_id': 10.0,
        'y_customer_id': 10.0
    }

    # Create a DataFrame with terminal profiles
    terminals_data = pd.DataFrame({
        'x_terminal_id': [9.0, 10.0, 11.0, 12.0],
        'y_terminal_id': [10.0, 11.0, 12.0, 13.0]
    })

    # Define the radius
    radius = 5.0

    # Expected result: Indices of terminals within the radius
    expected_result = [0, 1]

    # Call the function to select terminals within the customer's radius
    result = select_terminals_within_customer_radius(customer_profile, terminals_data, radius)

    # Compare the result with the expected result
    assert result == expected_result

def sample_transactions_data():
    # Create a sample transactions dataframe
    transactions_data = pd.DataFrame({
        'CUSTOMER_ID': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'TRANSACTION_ID': [101, 102, 103, 201, 202, 203, 301, 302, 303],
        'TX_DATETIME': pd.to_datetime(['2023-06-01 09:00:00', '2023-06-05 14:00:00', '2023-06-10 10:00:00',
                                       '2023-06-02 08:00:00', '2023-06-04 11:00:00', '2023-06-07 16:00:00',
                                       '2023-06-03 13:00:00', '2023-06-06 15:00:00', '2023-06-10 18:00:00']),
        'TX_AMOUNT': [50.0, 30.0, 20.0, 100.0, 150.0, 80.0, 70.0, 90.0, 120.0]
    })
    return transactions_data

def test_get_customer_spending_behaviour_features(sample_transactions_data):
    # Call the function under test
    result = get_customer_spending_behaviour_features(sample_transactions_data, [1, 7, 30])

    # Verify the output dataframe structure and values
    assert result.columns.tolist() == ['CUSTOMER_ID', 'TRANSACTION_ID', 'TX_DATETIME', 'TX_AMOUNT',
                                       'CUSTOMER_ID_NB_TX_1DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW',
                                       'CUSTOMER_ID_NB_TX_7DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW',
                                       'CUSTOMER_ID_NB_TX_30DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW']
    assert result['CUSTOMER_ID_NB_TX_1DAY_WINDOW'].tolist() == [1, 1, 1, 1, 2, 3, 1, 2, 3]
    assert result['CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW'].tolist() == [50.0, 30.0, 20.0, 100.0, 125.0, 83.33333333333333, 70.0, 60.0, 70.0]
    assert result['CUSTOMER_ID_NB_TX_7DAY_WINDOW'].tolist() == [3, 3, 3, 3, 6, 6, 6, 6, 6]
    assert result['CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW'].tolist() == [33.333333333333336, 33.333333333333336, 33.333333333333336,
                                                                     33.333333333333336, 95.0, 81.66666666666667,
                                                                     80.0, 81.66666666666667, 93.33333333333333]
    assert result['CUSTOMER_ID_NB_TX_30DAY_WINDOW'].tolist() == [3, 3, 3, 3, 6, 6, 6, 6, 6]
    assert result['CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW']


def test_plot_transactions_distribution(sample_transactions_data):
    # Call the function with example transactions
    fig = plot_transactions_distribution(sample_transactions_data)

    # Check if the function returns a matplotlib figure object
    assert isinstance(fig, plt.Figure)

    # Check if the figure has two subplots
    assert len(fig.axes) == 2

    # Check the properties of the first subplot
    ax1 = fig.axes[0]
    assert ax1.get_title() == 'Distribution of transaction amounts'
    assert ax1.get_xlabel() == 'Amount'
    assert ax1.get_ylabel() == 'Number of transactions'

    # Check the properties of the second subplot
    ax2 = fig.axes[1]
    assert ax2.get_title() == 'Distribution of transaction times'
    assert ax2.get_xlabel() == 'Time (days)'
    assert ax2.get_ylabel() == 'Number of transactions'

    # Add additional assertions if needed

    # Clean up the plot after testing (optional)
    plt.close(fig)



def test_get_stats():
    df = pd.read_csv("data/01_raw/04_transactions_data.csv")
    assert df.dtypes['TRANSACTION_ID'] == np.int64
    assert df.dtypes['TX_DATETIME'] == np.float64
    assert df.dtypes['CUSTOMER_ID'] == np.float64
    assert df.dtypes['TERMINAL_ID'] == np.float64
    assert df.dtypes['TX_AMOUNT'] == np.float64
    assert df.dtypes['TX_TIME_SECONDS'] == np.float64
    assert df.dtypes['TX_TIME_DAYS'] == np.float64



def test_plot_transactions_daily_stats():
    # Create a sample transactions dataframe for testing
    transactions = pd.DataFrame({
        "TX_TIME_DAYS": [1, 2, 3, 4, 5],
        "value": [10, 20, 15, 30, 25]
    })

    # Call the function to plot the daily statistics
    fig = plot_transactions_daily_stats(transactions)

    # Assertions to check the expected behavior of the function
    assert isinstance(fig, plt.Figure)
    assert fig.get_size_inches() == (20, 8)

    # Check the plot elements
    axes = fig.get_axes()
    assert len(axes) == 1
    ax = axes[0]
    assert ax.get_title() == 'Total transactions, and number of fraudulent transactions \n and number of compromised cards per day'
    assert ax.get_xlabel() == 'Number of days since beginning of data generation'
    assert ax.get_ylabel() == 'Number'
    assert ax.get_ylim() == (0, 300)

    # Check the legend labels
    legend = ax.legend_
    assert legend.get_texts()[0].get_text() == '# transactions per day'
    assert legend.get_texts()[1].get_text() == '# fraudulent txs per day'
    assert legend.get_texts()[2].get_text() == '# fraudulent cards per day'