import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns

class Backtester:
    def __init__(self, ticker, lookback_years=5, strategy_type='mean_reversion'):
        """
        Initialize the backtester
        
        Parameters:
        - ticker: str, stock ticker symbol (e.g., 'AAPL', 'SPY')
        - lookback_years: int, number of years of historical data (default: 5)
        - strategy_type: str, 'mean_reversion' or 'momentum' (default: 'mean_reversion')
        """
        self.ticker = ticker
        self.lookback_years = lookback_years
        self.strategy_type = strategy_type
        self.data = None
        self.df = None
        
        # Validate strategy type
        if strategy_type not in ['mean_reversion', 'momentum']:
            raise ValueError("strategy_type must be 'mean_reversion' or 'momentum'")
        
    
    def get_lookback_date(self):
        """Calculate the start date based on lookback period"""
        today = dt.date.today()
        start = today - dt.timedelta(days=self.lookback_years * 365)
        return start
    
    def fetch_data(self, save_csv=False):
        """
        Download data from yfinance - the BBG alternative for personal computer
        This will be replaced by fetching from BBG and integrated into clean data function
        Parameters:
        - save_csv: bool, whether to save data to CSV (default: False)
        """
        start_date = self.get_lookback_date()
        self.data = yf.download(self.ticker, start=start_date, end=dt.date.today())
        
        if save_csv:
            filename = f"{self.ticker}_data.csv"
            self.data.to_csv(filename)
            print(f"Data saved to {filename}")
        
        return self.data
    
    def clean_data(self):
        """Clean the raw data and standardize column names"""
        if self.data is None:
            raise ValueError("No data to clean. Run fetch_data() first.")
        
        df = self.data.copy()
        
        # Handle multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        # Standardize column names
        df.columns = df.columns.str.lower()
        df.rename(columns={'close': 'price'}, inplace=True)
        
        # Drop volume column if present
        if 'volume' in df.columns:
            df = df.drop(columns='volume')
        
        return df
    
    def prepare_data(self, max_holding_period=22):
        """
        Calculate returns
        
        Parameters:
        - max_holding_period: int, maximum holding period to calculate forward returns for
        """
        self.fetch_data()
        df = self.clean_data()
    
        # Calculate daily returns
        df['return'] = df['price'].pct_change()
        #df['return'] = df['price'].diff() for daily returns in bps
    
        # Calculate forward returns for different holding periods
        for i in range(1, max_holding_period + 1):
            df[f'{i}d_fwd_returns'] = df['price'].pct_change(periods=i).shift(-i)
    
        df = df.dropna()
        # Store the prepared data in the instance variable
        self.df = df
    
        return self.df
    
    def plot_sensitivity_table(self, holding_period_range, trigger_values):
        """
        Plot heatmaps showing strategy performance across different parameters
        
        Parameters:
        - holding_period_range: list or range, holding periods to test (e.g., range(1, 23))
        - trigger_values: list or array, trigger values to test (e.g., np.arange(0.01, 0.05, 0.005))
        """
        if self.df is None:
            raise ValueError("No data prepared. Run prepare_data() first.")
        
        buy_strat = pd.DataFrame(index=list(holding_period_range), 
                                columns=[round(x, 4) for x in trigger_values], dtype=float)
        sell_strat = pd.DataFrame(index=list(holding_period_range), 
                                 columns=[round(x, 4) for x in trigger_values], dtype=float)

        for trigger_val in trigger_values:
            trigger_val = round(trigger_val, 4)
            
            data_analyze = self.df.copy()
            data_analyze['signal'] = data_analyze['return'].abs() >= trigger_val
            
            data_signal = data_analyze[data_analyze['signal'] == True].copy()
            
            # Determine trade direction based on strategy type
            if self.strategy_type == 'mean_reversion':
                # Mean reversion: buy after sell-offs, sell after rallies
                data_signal['trade_direction'] = np.where(data_signal['return'] < 0, 'buy', 'sell')
            elif self.strategy_type == 'momentum':
                # Momentum: buy after rallies, sell after sell-offs
                data_signal['trade_direction'] = np.where(data_signal['return'] > 0, 'buy', 'sell')

            pnl = np.where(data_signal['trade_direction'] == 'buy', 1, -1)

            for i in holding_period_range:
                if f'{i}d_fwd_returns' in data_signal.columns:
                    data_signal[f'{i}d_fwd_returns_adj'] = data_signal[f'{i}d_fwd_returns'] * pnl
                    
                    buy_pnl = data_signal.loc[data_signal['trade_direction'] == 'buy', 
                                             f'{i}d_fwd_returns_adj'].mean()
                    sell_pnl = data_signal.loc[data_signal['trade_direction'] == 'sell', 
                                               f'{i}d_fwd_returns_adj'].mean()
                    
                    buy_strat.loc[i, trigger_val] = buy_pnl
                    sell_strat.loc[i, trigger_val] = sell_pnl

        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
        cmap = sns.diverging_palette(0, 130, s=90, l=60, as_cmap=True)

        sns.heatmap(buy_strat.astype(float), ax=axes[0], cmap=cmap, annot=True, 
                   fmt=".4f", cbar=True, center=0)
        axes[0].set_title("Buy Strategy (Mean PnL)", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Trigger Value", fontsize=11)
        axes[0].set_ylabel("Holding Period (days)", fontsize=11)

        sns.heatmap(sell_strat.astype(float), ax=axes[1], cmap=cmap, annot=True, 
                   fmt=".4f", cbar=True, center=0)
        axes[1].set_title("Sell Strategy (Mean PnL)", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("Trigger Value", fontsize=11)
        axes[1].set_ylabel("Holding Period (days)", fontsize=11)

        strategy_label = 'Mean Reversion' if self.strategy_type == 'mean_reversion' else 'Momentum'
        plt.suptitle(f"{strategy_label} Strategy Sensitivity - {self.ticker}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return buy_strat, sell_strat
    
    def plot_pnl(self, buy_or_sell, holding_period, trigger_val):
        """
        Plot cumulative PnL for a specific strategy configuration
        
        Parameters:
        - buy_or_sell: str, 'buy', 'sell', or 'both'
        - holding_period: int, number of days to hold position
        - trigger_val: float, threshold for signal generation (e.g., 0.02 for 2%)
        
        Returns:
        - DataFrame containing signal data with PnL calculations
        """
        if self.df is None:
            raise ValueError("No data prepared. Run prepare_data() first.")
        
        data_analyze = self.df.copy()
        
        # Generate signals based on strategy type
        if self.strategy_type == 'mean_reversion':
            # Mean Reversion: buy after drops, sell after rallies
            if buy_or_sell == 'buy':
                # Buy after negative returns past trigger
                data_analyze['signal'] = data_analyze['return'] <= -trigger_val
                data_analyze['trade_direction'] = 'buy'
                
            elif buy_or_sell == 'sell':
                # Sell after positive returns past trigger
                data_analyze['signal'] = data_analyze['return'] >= trigger_val
                data_analyze['trade_direction'] = 'sell'
                
            elif buy_or_sell == 'both':
                #Trade both directions
                data_analyze['signal'] = data_analyze['return'].abs() >= trigger_val
                data_analyze['trade_direction'] = np.where(data_analyze['return'] < 0, 'buy', 'sell')
        
        elif self.strategy_type == 'momentum':
            # Momentum: buy after rallies, sell after drops
            if buy_or_sell == 'buy':
                data_analyze['signal'] = data_analyze['return'] >= trigger_val
                data_analyze['trade_direction'] = 'buy'
                
            elif buy_or_sell == 'sell':
                data_analyze['signal'] = data_analyze['return'] <= -trigger_val
                data_analyze['trade_direction'] = 'sell'
                
            elif buy_or_sell == 'both':
                data_analyze['signal'] = data_analyze['return'].abs() >= trigger_val
                data_analyze['trade_direction'] = np.where(data_analyze['return'] > 0, 'buy', 'sell')
        
        # Filter for signal days
        data_signal = data_analyze[data_analyze['signal'] == True].copy()
        
        if len(data_signal) == 0:
            print(f"No signals generated with trigger value {trigger_val}")
            return None
        
        # Calculate PnL for the holding period
        fwd_col = f'{holding_period}d_fwd_returns'
        if fwd_col not in data_signal.columns:
            print(f"Forward return column '{fwd_col}' not found in dataframe")
            return None
        
        # For mean reversion: buy expects positive forward returns, sell expects negative
        if buy_or_sell == 'buy':
            data_signal['pnl'] = data_signal[fwd_col]
        elif buy_or_sell == 'sell':
            data_signal['pnl'] = -data_signal[fwd_col]
        else:  # both
            data_signal['pnl'] = np.where(
                data_signal['trade_direction'] == 'buy',
                data_signal[fwd_col],
                -data_signal[fwd_col]
            )
        
        # Calculate cumulative PnL
        data_signal['cumulative_pnl'] = data_signal['pnl'].cumsum()
        
        # Create the plot
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Cumulative PnL
        axes[0].plot(data_signal.index, data_signal['cumulative_pnl'], linewidth=2, color='blue')
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        strategy_label = 'Mean Reversion' if self.strategy_type == 'mean_reversion' else 'Momentum'
        axes[0].set_title(f'Cumulative PnL - {strategy_label} {buy_or_sell.capitalize()} Strategy', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Cumulative Return')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Trade PnL distribution
        axes[1].bar(range(len(data_signal)), data_signal['pnl'], 
                    color=['green' if x > 0 else 'red' for x in data_signal['pnl']], alpha=0.6)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        axes[1].set_title('Individual Trade Returns', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Trade Number')
        axes[1].set_ylabel('Return')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Print statistics
        total_return = data_signal['cumulative_pnl'].iloc[-1]
        mean_return = data_signal['pnl'].mean()
        std_return = data_signal['pnl'].std()
        sharpe = mean_return / std_return * np.sqrt(252 / holding_period) if std_return > 0 else 0
        win_rate = (data_signal['pnl'] > 0).sum() / len(data_signal)
        
        print(f"Strategy Type: {self.strategy_type.upper().replace('_', ' ')}")
        print(f"Direction: {buy_or_sell.upper()}")
        print(f"Trigger Value: {trigger_val:.2%}")
        print(f"Holding Period: {holding_period} days")
        print(f"Total Trades: {len(data_signal)}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Mean Return per Trade: {mean_return:.4%}")
        print(f"Std Dev per Trade: {std_return:.4%}")
        print(f"Annualized Sharpe Ratio: {sharpe:.2f}")
        
        plt.show()
        return data_signal
    
    def summary(self):
        """Print a summary of the backtester configuration and data"""
        print(f"\n{'-'*60}")
        strategy_label = 'Mean Reversion' if self.strategy_type == 'mean_reversion' else 'Momentum'
        print(f"{strategy_label} Backtester Summary")
        print(f"Ticker: {self.ticker}")
        print(f"Strategy Type: {strategy_label}")
        print(f"Lookback Period: {self.lookback_years} years")
        print(f"Start Date: {self.get_lookback_date()}")
        
        if self.df is not None:
            print(f"Data Points: {len(self.df)}")
            print(f"Date Range: {self.df.index.min().date()} to {self.df.index.max().date()}")
            print(f"Average Daily Return: {self.df['return'].mean():.4%}")
            print(f"Daily Return Std Dev: {self.df['return'].std():.4%}")
            print(f"Max Holding Period Available: {max([int(col.split('d')[0]) for col in self.df.columns if 'd_fwd_returns' in col])} days")
        else:
            print("Data: Not yet loaded")

def run_backtest(ticker, 
                 lookback_years=5,
                 max_holding_period=22,
                 strategy_type='mean_reversion',
                 show_summary=True,
                 # Sensitivity table parameters (optional)
                 sensitivity_holding_periods=None,
                 sensitivity_trigger_values=None,
                 # Specific PnL plot parameters (optional)
                 pnl_strategy=None,
                 pnl_holding_period=None,
                 pnl_trigger_val=None):

    # Initialize and prepare data
    
    backtester = Backtester(ticker, lookback_years=lookback_years, 
                                        strategy_type=strategy_type)
    backtester.prepare_data(max_holding_period=max_holding_period)
    
    if show_summary:
        backtester.summary()
    
    # Run sensitivity analysis if parameters provided
    sensitivity_params_provided = all([
        sensitivity_holding_periods is not None,
        sensitivity_trigger_values is not None
    ])
    
    if sensitivity_params_provided:
        print("-"*60 + "\n")
        print(" Sensitivity Analysis Table")
        backtester.plot_sensitivity_table(sensitivity_holding_periods, 
                                         sensitivity_trigger_values)
    
    # Run specific PnL plot if parameters provided
    pnl_params_provided = all([
        pnl_strategy is not None,
        pnl_holding_period is not None,
        pnl_trigger_val is not None
    ])
    
    if pnl_params_provided:
        print("-"*60 + "\n")
        print(f"PnL PLOT - {pnl_strategy.upper()} STRATEGY")
        
        backtester.plot_pnl(pnl_strategy, pnl_holding_period, pnl_trigger_val)
    
    # Final message
    if not sensitivity_params_provided and not pnl_params_provided:
        print("TIP: Use backtester.plot_sensitivity_table() or backtester.plot_pnl()")
        print("     for further analysis")
    
    return backtester

st.set_page_config(page_title="Trading Strategy Backtester", layout="wide")

st.title(" Backtesting Dashboard")
st.markdown("---")

# Sidebar for inputs
st.sidebar.header("Backtest Configuration")

# Main inputs
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY", help="Enter ticker")
strategy_type = st.sidebar.selectbox(
    "Strategy Type",
    options=["mean_reversion", "momentum"],
    format_func=lambda x: x.replace("_", " ").title(),  # Display as "Mean Reversion" and "Momentum"
    help="Mean Reversion: Buy dips, sell rallies | Momentum: Buy rallies, sell dips"
)

lookback_years = st.sidebar.number_input(
    "Lookback Period (Years)",
    min_value=0.1,
    max_value=30.0,
    value=5.0,
    help="Number of years of historical data to use (can be decimal e.g. 5.3)"
)

st.sidebar.markdown("---")

# Sensitivity Table Section
st.sidebar.subheader("Sensitivity Analysis")
show_sensitivity = st.sidebar.radio("Show Sensitivity Table", options=["Yes", "No"], index=0)

sensitivity_params = {}
if show_sensitivity == "Yes":
    st.sidebar.markdown("**Holding Period Range**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        min_holding = st.number_input("Min Days", min_value=1, max_value=100, value=1)
    with col2:
        max_holding = st.number_input("Max Days", min_value=2, max_value=300, value=22)
    
    st.sidebar.markdown("**Trigger Value Range**")
    trigger_start = st.sidebar.number_input("Start Value", min_value=0.001, max_value=0.5, value=0.01, 
                                           format="%.3f", help="E.g., 0.01 = 1%")
    trigger_end = st.sidebar.number_input("End Value", min_value=0.001, max_value=0.5, value=0.05, 
                                         format="%.3f", help="E.g., 0.05 = 5%")
    trigger_interval = st.sidebar.number_input("Interval", min_value=0.001, max_value=0.1, value=0.005, 
                                              format="%.3f", help="E.g., 0.005 = 0.5%")
    
    sensitivity_params = {
        'holding_periods': range(min_holding, max_holding + 1),
        'trigger_values': np.arange(trigger_start, trigger_end + trigger_interval/2, trigger_interval)
    }

st.sidebar.markdown("---")

# PnL Plot Section
st.sidebar.subheader("PnL and Sharpe")
show_pnl = st.sidebar.radio("Show PnL and Sharpe", options=["Yes", "No"], index=0)

pnl_params = {}
if show_pnl == "Yes":
    pnl_direction = st.sidebar.selectbox(
        "Trading Direction",
        options=["buy", "sell", "both"],
        help="Buy only, Sell only, or Both directions"
    )
    
    pnl_holding = st.sidebar.number_input(
        "Holding Period (Days)", 
        min_value=1, 
        max_value=300, 
        value=5,
        help="Number of days to hold position"
    )
    
    pnl_trigger = st.sidebar.number_input(
        "Trigger Value", 
        min_value=0.001, 
        max_value=0.5, 
        value=0.02,
        format="%.3f",
        help="Signal threshold (e.g., 0.02 = 2%)"
    )
    
    pnl_params = {
        'direction': pnl_direction,
        'holding_period': pnl_holding,
        'trigger_val': pnl_trigger
    }

st.sidebar.markdown("---")
run_button = st.sidebar.button(" Run Backtest", type="primary", use_container_width=True)

# Main content area
if run_button:
    if not ticker:
        st.error("Please enter a ticker symbol")
    else:
        try:
            with st.spinner(f"Preparing data for {ticker}..."):
                # Initialize backtester
                backtester = Backtester(
                    ticker=ticker,
                    lookback_years=lookback_years,
                    strategy_type=strategy_type
                )
                
                # Prepare data 
                max_holding_period = 1  # Minimum possible
                
                if show_sensitivity == "Yes":
                    max_holding_period = max(max_holding_period, max(sensitivity_params['holding_periods']))
                    
                if show_pnl == "Yes" and 'holding_period' in pnl_params:
                    max_holding_period = max(max_holding_period, pnl_params['holding_period'])
                
                backtester.prepare_data(max_holding_period=max_holding_period)
                
            # Display summary
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Strategy", strategy_type.replace('_', ' ').title())
            with col2:
                st.metric("Mean Return", f"{backtester.df['return'].mean():.4%}")
            with col3:
                st.metric("Daily Vol (std)", f"{backtester.df['return'].std():.4%}")
            
            st.markdown("---")
            
            # Show sensitivity table if requested
            if show_sensitivity == "Yes":
                st.header("Sensitivity Analysis")
                holding_range = sensitivity_params['holding_periods']
                trigger_range = sensitivity_params['trigger_values']
                st.markdown(f"**Holding Periods:** {min(holding_range)} to {max(holding_range)} days")
                st.markdown(f"**Trigger Values:** {min(trigger_range):.1%} to {max(trigger_range):.1%}")
                
                with st.spinner("Displaying heatmap of sensitivity of returns to deviations in hyperparameters"):
                    fig, axes = plt.subplots(2, 1, figsize=(16, 14), sharex=True)
                    
                    # Run sensitivity analysis
                    buy_strat = pd.DataFrame(
                        index=list(sensitivity_params['holding_periods']), 
                        columns=[round(x, 4) for x in sensitivity_params['trigger_values']], 
                        dtype=float
                    )
                    sell_strat = pd.DataFrame(
                        index=list(sensitivity_params['holding_periods']), 
                        columns=[round(x, 4) for x in sensitivity_params['trigger_values']], 
                        dtype=float
                    )
                    
                    for trigger_val in sensitivity_params['trigger_values']:
                        trigger_val = round(trigger_val, 4)
                        
                        data_analyze = backtester.df.copy()
                        data_analyze['signal'] = data_analyze['return'].abs() >= trigger_val
                        data_signal = data_analyze[data_analyze['signal'] == True].copy()
                        
                        if backtester.strategy_type == 'mean_reversion':
                            data_signal['trade_direction'] = np.where(data_signal['return'] < 0, 'buy', 'sell')
                        elif backtester.strategy_type == 'momentum':
                            data_signal['trade_direction'] = np.where(data_signal['return'] > 0, 'buy', 'sell')
                        
                        pnl = np.where(data_signal['trade_direction'] == 'buy', 1, -1)
                        
                        for i in sensitivity_params['holding_periods']:
                            if f'{i}d_fwd_returns' in data_signal.columns:
                                data_signal[f'{i}d_fwd_returns_adj'] = data_signal[f'{i}d_fwd_returns'] * pnl
                                buy_pnl = data_signal.loc[data_signal['trade_direction'] == 'buy', 
                                                         f'{i}d_fwd_returns_adj'].mean()
                                sell_pnl = data_signal.loc[data_signal['trade_direction'] == 'sell', 
                                                           f'{i}d_fwd_returns_adj'].mean()
                                buy_strat.loc[i, trigger_val] = buy_pnl
                                sell_strat.loc[i, trigger_val] = sell_pnl
                    
                    # Plot heatmaps
                    cmap = sns.diverging_palette(0, 130, s=90, l=60, as_cmap=True)
                    
                    sns.heatmap(buy_strat.astype(float), ax=axes[0], cmap=cmap, annot=True, 
                               fmt=".4f", cbar=True, center=0)
                    axes[0].set_title("Buy Strategy (Mean PnL)", fontsize=12, fontweight='bold')
                    axes[0].set_xlabel("Trigger Value", fontsize=11)
                    axes[0].set_ylabel("Holding Period (days)", fontsize=11)
                    
                    sns.heatmap(sell_strat.astype(float), ax=axes[1], cmap=cmap, annot=True, 
                               fmt=".4f", cbar=True, center=0)
                    axes[1].set_title("Sell Strategy (Mean PnL)", fontsize=12, fontweight='bold')
                    axes[1].set_xlabel("Trigger Value", fontsize=11)
                    axes[1].set_ylabel("Holding Period (days)", fontsize=11)
                    
                    strategy_label = 'Mean Reversion' if strategy_type == 'mean_reversion' else 'Momentum'
                    plt.suptitle(f"{strategy_label} Strategy Sensitivity - {ticker}", 
                                fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    plt.close()
                
                st.markdown("---")
            
            # Show PnL plot if requested
            if show_pnl == "Yes":
                st.header("Strategy PnL and Sharpe Analysis")
                st.markdown(f"**Direction:** {pnl_params['direction'].upper()} | "
                          f"**Holding Period:** {pnl_params['holding_period']} days | "
                          f"**Trigger:** {pnl_params['trigger_val']:.2%}")
                
                with st.spinner("Generating PnL chart"):
                    result = backtester.plot_pnl(
                        buy_or_sell=pnl_params['direction'],
                        holding_period=pnl_params['holding_period'],
                        trigger_val=pnl_params['trigger_val']
                    )
                    
                    if result is not None:
                        # Display the matplotlib figure that was created
                        st.pyplot(plt.gcf())
                        plt.close()
                        
                        # Additional metrics in columns
                        st.subheader("Performance Metrics")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        total_return = result['cumulative_pnl'].iloc[-1]
                        mean_return = result['pnl'].mean()
                        std_return = result['pnl'].std()
                        sharpe = mean_return / std_return * np.sqrt(252 / pnl_params['holding_period']) if std_return > 0 else 0
                        win_rate = (result['pnl'] > 0).sum() / len(result)
                        
                        with col1:
                            st.metric("Total Return", f"{total_return:.2%}")
                        with col2:
                            st.metric("Win Rate", f"{win_rate:.2%}")
                        with col3:
                            st.metric("Total Trades", len(result))
                        with col4:
                            st.metric("Mean Return per Trade", f"{mean_return:.4%}")
                        with col5:
                            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    else:
                        st.warning("No signals generated with the selected parameters")
                
        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")
            st.exception(e)

else:
    # Welcome message
    st.info("Select backtest parameters then click 'Run Backtest'")
