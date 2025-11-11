import streamlit as st
import pandas as pd
import requests
from io import BytesIO, StringIO
import warnings
import numpy as np
import yfinance as yf
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
import plotly.express as px
from functools import lru_cache
import calendar

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Market Indices Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for black and gold theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Montserrat:wght@300;400;600&display=swap');
    
    /* Main background */
    .stApp {
        background-color: #0a0a0a;
        color: #d4af37;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        color: #d4af37;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Metrics and text */
    .stMetric {
        background-color: #1a1a1a;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #d4af37;
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: #1a1a1a !important;
        color: #d4af37 !important;
        border: 1px solid #d4af37 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #d4af37 0%, #f4d03f 100%);
        color: #0a0a0a;
        font-weight: 600;
        border: none;
        padding: 12px 30px;
        border-radius: 25px;
        font-size: 18px;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(212, 175, 55, 0.6);
    }
    
    /* Selectbox and multiselect */
    .stSelectbox, .stMultiSelect {
        background-color: #1a1a1a;
        color: #d4af37;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0a0a0a;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a1a;
        color: #d4af37;
        border: 1px solid #d4af37;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #d4af37 0%, #f4d03f 100%);
        color: #0a0a0a;
    }
    
    /* Landing page styles */
    .landing-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100vh;
        background-image: url('https://wallpapershome.com/images/pages/pic_h/26287.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    
    .landing-title {
        font-family: 'Playfair Display', serif;
        font-size: 80px;
        font-weight: 700;
        color: #d4af37;
        text-align: center;
        text-shadow: 4px 4px 8px rgba(0,0,0,0.8);
        margin-bottom: 50px;
        animation: fadeIn 2s;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Arrow styles */
    .arrow-up {
        color: #009900;
        font-size: 24px;
    }
    
    .arrow-down {
        color: #ad0000;
        font-size: 24px;
    }
    
    /* Table styling */
    table {
        background-color: #1a1a1a;
        color: #d4af37;
    }
    
    thead tr th {
        background-color: #0a0a0a !important;
        color: #d4af37 !important;
        border: 1px solid #d4af37 !important;
    }
    
    tbody tr td {
        border: 1px solid #333 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'started' not in st.session_state:
    st.session_state.started = False

# Cache data fetching functions
@st.cache_data(ttl=3600)
def baixar_indice(indice, name, source, start_date='2020-01-01'):
    """Download index data from various sources"""
    if source == 'anbima':
        url = f'https://s3-data-prd-use1-precos.s3.us-east-1.amazonaws.com/arquivos/indices-historico/{indice}-HISTORICO.xls'
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_excel(BytesIO(response.content))[['Data de Referência', 'Número Índice']]
        df.set_index('Data de Referência', inplace=True)
        df.rename(columns={'Número Índice': name}, inplace=True)
        return df.loc[df.index > start_date]
    
    elif source == 'yf':
        df = yf.download(indice, start=start_date, end=date.today(), interval='1d', progress=False)['Close']
        if isinstance(df, pd.Series):
            df = df.to_frame(name=name)
        else:
            df.columns = [name]
        return df
    
    elif source == 'cdi':
        serie_codigo = 12
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie_codigo}/dados"

        start_date = pd.to_datetime(start_date)
        start_date_str = start_date.strftime('%d/%m/%Y')
        end_date_str = datetime.today().strftime('%d/%m/%Y')
        
        params = {
            'formato': 'csv',
            'dataInicial': start_date_str,
            'dataFinal': end_date_str
        }
        headers = {'Cache-Control': 'no-cache', 'Pragma': 'no-cache'}
        response = requests.get(url, params=params, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = StringIO(response.text)
            cdi_df = pd.read_csv(data, sep=";", decimal=",", encoding="latin1")
            
            cdi_df['valor'] = pd.to_numeric(cdi_df['valor'].astype(str).str.replace(',', '.', regex=False), errors='coerce') / 100
            cdi_df['data'] = pd.to_datetime(cdi_df['data'], format='%d/%m/%Y')
            cdi_df.set_index('data', inplace=True)

            cdi_df['CDI'] = np.zeros(len(cdi_df))

            for i in range(0, len(cdi_df)):
                if i == 0:
                    cdi_df['CDI'].iloc[i] = 1 + cdi_df['valor'].iloc[i]
                else:
                    cdi_df['CDI'].iloc[i] = (1 + cdi_df['valor'].iloc[i]) * (cdi_df['CDI'].iloc[i-1])

            return pd.DataFrame(cdi_df['CDI'])
        else:
            st.error(f"Error accessing CDI API: {response.status_code}")
            return pd.DataFrame()
    else:
        st.error('Request a valid source')
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_all_indices():
    """Load all indices data"""
    with st.spinner('Loading market data...'):
        indices_data = {}
        
        # ANBIMA indices
        anbima_indices = [
            ('IMAB5', 'IMA-B5'),
            ('IMAB5MAIS', 'IMA-B5+'),
            ('IMAS', 'IMA-S'),
            ('IDADI', 'IDA-DI'),
            ('IDAIPCA', 'IDA-IPCA'),
            ('IRFM', 'IRF-M'),
            ('IHFA', 'IHFA')
        ]
        
        for code, name in anbima_indices:
            try:
                indices_data[name] = baixar_indice(code, name, 'anbima')
            except Exception as e:
                st.warning(f"Could not load {name}: {str(e)}")
        
        # Yahoo Finance indices
        yf_indices = [
            ('^BVSP', 'IBOVESPA'),
            ('^GSPC', 'S&P500'),
            ('XFIX11.SA', 'IFIX'),
            ('BRL=X', 'USD/BRL'),
            ('BTC-USD', 'BITCOIN'),
            ('GLD', 'OURO'),
            ('EUE.MI', 'STOXX50')
        ]
        
        for code, name in yf_indices:
            try:
                indices_data[name] = baixar_indice(code, name, 'yf')
            except Exception as e:
                st.warning(f"Could not load {name}: {str(e)}")
        
        # CDI
        try:
            indices_data['CDI'] = baixar_indice('CDI', 'CDI', 'cdi')
        except Exception as e:
            st.warning(f"Could not load CDI: {str(e)}")
        
        return indices_data

def calc_returns(df):
    """Compute MTD, YTD, 12M, 24M, and 36M returns from closing prices"""
    close = df.sort_index()
    col = close.columns[0]
    latest = close.index.max()
    end_price = close.loc[latest, col]

    def total_return(start, end):
        if pd.notna(start) and pd.notna(end) and start != 0:
            return (end / start) - 1
        else:
            return np.nan

    def get_start_price(start_date):
        """Get price from last available trading day before or on start_date."""
        if start_date <= close.index.min():
            return close.iloc[0, 0]
        idx = close.index.get_indexer([start_date], method='ffill')
        if idx[0] == -1:
            return np.nan
        return close.iloc[idx[0], 0]

    # Period starts
    start_mtd_date = latest.replace(day=1)
    start_ytd_date = pd.to_datetime(f"{latest.year}-01-01")
    start_12m_date = latest - pd.DateOffset(months=12)
    start_24m_date = latest - pd.DateOffset(months=24)
    start_36m_date = latest - pd.DateOffset(months=36)

    # Get start prices
    start_mtd = get_start_price(start_mtd_date - pd.Timedelta(days=1))
    start_ytd = get_start_price(start_ytd_date - pd.Timedelta(days=1))
    start_12m = get_start_price(start_12m_date)
    start_24m = get_start_price(start_24m_date)
    start_36m = get_start_price(start_36m_date)

    # Compute returns
    returns = {
        "MTD": total_return(start_mtd, end_price),
        "YTD": total_return(start_ytd, end_price),
        "12M": total_return(start_12m, end_price),
        "24M": total_return(start_24m, end_price),
        "36M": total_return(start_36m, end_price),
    }

    df_returns = pd.DataFrame(returns, index=[col]).T
    df_returns[col] = df_returns[col] * 100

    return df_returns

def get_daily_variation(df):
    """Calculate daily variation between last two trading days"""
    close = df.sort_index()
    col = close.columns[0]
    
    if len(close) < 2:
        return None, None, None, None
    
    last_date = close.index[-1]
    prev_date = close.index[-2]
    last_value = close.iloc[-1, 0]
    prev_value = close.iloc[-2, 0]
    
    variation = ((last_value / prev_value) - 1) * 100
    
    return last_date, last_value, prev_date, prev_value, variation

def calc_monthly_returns(indices_data, n_months=12):
    """Calculate monthly returns for all indices"""
    monthly_returns = {}
    
    for name, df in indices_data.items():
        if df is None or len(df) == 0:
            continue
        
        # Resample to monthly and calculate returns
        monthly = df.resample('ME').last()
        returns = monthly.pct_change() * 100
        
        # Get last n months
        returns = returns.tail(n_months)
        monthly_returns[name] = returns
    
    return monthly_returns

def create_monthly_ranking_matrix(monthly_returns):
    """Create ranking matrix for monthly returns - each month ranked independently"""
    if not monthly_returns:
        return None
    
    # Get all months from all indices
    all_months = set()
    for returns in monthly_returns.values():
        all_months.update(returns.index)
    
    all_months = sorted(list(all_months))
    
    # For each month, calculate returns and rank
    monthly_data = {}
    for month in all_months:
        month_returns = {}
        for name, returns in monthly_returns.items():
            if month in returns.index:
                ret_val = returns.loc[month]
                if isinstance(ret_val, pd.Series):
                    ret_val = ret_val.iloc[0]
                elif isinstance(ret_val, pd.DataFrame):
                    ret_val = ret_val.iloc[0, 0]
                
                if pd.notna(ret_val):
                    month_returns[name] = ret_val
        
        # Sort by return (descending) for this specific month
        sorted_month = sorted(month_returns.items(), key=lambda x: x[1], reverse=True)
        monthly_data[month] = sorted_month
    
    # Create ranking matrix
    max_indices = max(len(data) for data in monthly_data.values()) if monthly_data else 0
    ranking_matrix = pd.DataFrame(
        index=range(1, max_indices + 1),
        columns=[m.strftime('%m/%Y') for m in all_months]
    )
    
    # Fill the matrix
    for month, sorted_indices in monthly_data.items():
        month_str = month.strftime('%m/%Y')
        for rank, (idx_name, ret_val) in enumerate(sorted_indices, 1):
            ranking_matrix.loc[rank, month_str] = f"{idx_name}|{ret_val:.2f}"
    
    return ranking_matrix

def create_period_ranking_matrix(indices_data):
    """Create ranking matrix for different periods (36M, 24M, 12M, YTD, MTD) with returns"""
    periods = ['36M', '24M', '12M', 'YTD', 'MTD']
    
    # Calculate returns for all indices and periods
    all_returns = {}
    for name, df in indices_data.items():
        if df is None or len(df) == 0:
            continue
        returns = calc_returns(df)
        all_returns[name] = returns
    
    # Create ranking matrix
    max_indices = len(all_returns)
    ranking_matrix = pd.DataFrame(index=range(1, max_indices + 1), columns=periods)
    
    for period in periods:
        # Get returns for this period from all indices
        period_returns = {}
        for name, returns in all_returns.items():
            if period in returns.index:
                period_returns[name] = returns.loc[period].values[0]
        
        # Sort and rank
        sorted_returns = sorted(period_returns.items(), key=lambda x: x[1], reverse=True)
        for rank, (idx_name, ret_val) in enumerate(sorted_returns, 1):
            ranking_matrix.loc[rank, period] = f"{idx_name}|{ret_val:.2f}"
    
    return ranking_matrix

def resample_to_weekly(df):
    """Resample daily data to weekly for cumulative returns"""
    weekly = df.resample('W-FRI').last()
    return weekly

def calculate_cumulative_returns_weekly(indices_data, selected_indices, period):
    """Calculate cumulative returns on weekly basis"""
    # Determine start date based on period
    end_date = datetime.now()
    
    if period == 'MTD':
        start_date = end_date.replace(day=1)
    elif period == 'YTD':
        start_date = datetime(end_date.year, 1, 1)
    elif period == '12M':
        start_date = end_date - relativedelta(months=12)
    elif period == '24M':
        start_date = end_date - relativedelta(months=24)
    elif period == '36M':
        start_date = end_date - relativedelta(months=36)
    else:
        start_date = end_date - relativedelta(months=36)
    
    cumulative_returns = pd.DataFrame()
    
    for idx_name in selected_indices:
        if idx_name not in indices_data or indices_data[idx_name] is None:
            continue
        
        df = indices_data[idx_name].copy()
        
        # Filter by date
        df = df[df.index >= start_date]
        
        if len(df) == 0:
            continue
        
        # Resample to weekly
        weekly = resample_to_weekly(df)
        
        # Calculate cumulative returns
        col = weekly.columns[0]
        first_value = weekly.iloc[0, 0]
        weekly_cumret = ((weekly[col] / first_value) - 1) * 100
        
        cumulative_returns[idx_name] = weekly_cumret
    
    return cumulative_returns

# Landing Page
def show_landing_page():
    st.markdown("""
        <div class="landing-container">
            <h1 class="landing-title">MARKET INDICES DASHBOARD</h1>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("START", key="start_button", use_container_width=True):
            st.session_state.started = True
            st.rerun()

# Main Dashboard
def show_dashboard():
    st.title("📈 Market Indices Dashboard")
    
    # Load data
    indices_data = load_all_indices()
    
    if not indices_data:
        st.error("No data available. Please check your internet connection.")
        return
    
    # Sidebar (can be used for other settings later)
    st.sidebar.title("⚙️ Settings")
    st.sidebar.info("Use the controls in the main page to customize the chart.")
    
    available_indices = sorted(list(indices_data.keys()))
    
    # Set initial defaults only once
    default_indices = ['CDI', 'IBOVESPA', 'S&P500'] if all(x in available_indices for x in ['CDI', 'IBOVESPA', 'S&P500']) else available_indices[:3]
    
    # Main content
    st.header("📊 Cumulative Returns Chart")
    
    # Chart controls in main page
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_indices = st.multiselect(
            "Select Indices",
            options=available_indices,
            default=default_indices,
            help="Choose which indices to display on the chart"
        )
    
    with col2:
        period = st.selectbox(
            "Select Period",
            options=['36M', '24M', '12M', 'YTD', 'MTD'],
            index=0,
            help="Choose the time period for analysis"
        )
    
    st.markdown("---")
    st.subheader(f"Cumulative Returns ({period})")
    
    if selected_indices:
        cumulative_returns = calculate_cumulative_returns_weekly(indices_data, selected_indices, period)
        
        if not cumulative_returns.empty:
            # Create Plotly chart
            fig = go.Figure()
            
            # Generate colors
            colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
            
            for i, idx_name in enumerate(cumulative_returns.columns):
                fig.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns[idx_name],
                    mode='lines',
                    name=idx_name,
                    line=dict(width=2.5, color=colors[i % len(colors)]),
                    hovertemplate='%{y:.2f}%<extra></extra>'
                ))
            
            fig.update_layout(
                plot_bgcolor='#0a0a0a',
                paper_bgcolor='#1a1a1a',
                font=dict(color='#d4af37', family='Montserrat'),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='#333',
                    title='Date'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='#333',
                    title='Cumulative Return (%)'
                ),
                hovermode='x unified',
                legend=dict(
                    bgcolor='#1a1a1a',
                    bordercolor='#d4af37',
                    borderwidth=1
                ),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for selected indices and period.")
    else:
        st.info("Please select at least one index from the sidebar.")
    
    # Rankings side by side
    st.header("🏆 Performance Rankings")
    
    # Calculate all returns
    all_returns = {}
    for name, df in indices_data.items():
        if df is not None and len(df) > 0:
            returns = calc_returns(df)
            all_returns[name] = returns
    
    # Create returns DataFrame
    mtd_returns = pd.DataFrame({name: returns.loc['MTD'].values[0] for name, returns in all_returns.items()}, index=['Return']).T
    ytd_returns = pd.DataFrame({name: returns.loc['YTD'].values[0] for name, returns in all_returns.items()}, index=['Return']).T
    
    mtd_returns = mtd_returns.sort_values('Return', ascending=False)
    ytd_returns = ytd_returns.sort_values('Return', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        current_month = calendar.month_name[datetime.now().month]
        st.subheader(f"🥇 {current_month} Rankings (MTD)")
        
        # Format as table with rank
        mtd_display = mtd_returns.copy()
        mtd_display['Rank'] = range(1, len(mtd_display) + 1)
        mtd_display = mtd_display[['Rank', 'Return']]
        
        # Create HTML table with colored returns
        html_table = '<table style="width:100%; border-collapse: collapse;">'
        html_table += '<thead><tr style="background-color: #0a0a0a;"><th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Rank</th>'
        html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Index</th>'
        html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Return</th></tr></thead><tbody>'
        
        for idx_name, row in mtd_display.iterrows():
            ret_val = row['Return']
            color = '#009900' if ret_val >= 0 else '#ad0000'
            arrow = '▲' if ret_val >= 0 else '▼'
            html_table += '<tr>'
            html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: center; background-color: #1a1a1a; color: #d4af37;">{int(row["Rank"])}</td>'
            html_table += f'<td style="border: 1px solid #333; padding: 10px; background-color: #1a1a1a; color: #d4af37;">{idx_name}</td>'
            html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: right; background-color: #1a1a1a; color: {color}; font-weight: bold;">{arrow} {ret_val:.2f}%</td>'
            html_table += '</tr>'
        
        html_table += '</tbody></table>'
        st.markdown(html_table, unsafe_allow_html=True)
    
    with col2:
        current_year = datetime.now().year
        st.subheader(f"🥇 {current_year} Rankings (YTD)")
        
        # Format as table with rank
        ytd_display = ytd_returns.copy()
        ytd_display['Rank'] = range(1, len(ytd_display) + 1)
        ytd_display = ytd_display[['Rank', 'Return']]
        
        # Create HTML table with colored returns
        html_table = '<table style="width:100%; border-collapse: collapse;">'
        html_table += '<thead><tr style="background-color: #0a0a0a;"><th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Rank</th>'
        html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Index</th>'
        html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Return</th></tr></thead><tbody>'
        
        for idx_name, row in ytd_display.iterrows():
            ret_val = row['Return']
            color = '#009900' if ret_val >= 0 else '#ad0000'
            arrow = '▲' if ret_val >= 0 else '▼'
            html_table += '<tr>'
            html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: center; background-color: #1a1a1a; color: #d4af37;">{int(row["Rank"])}</td>'
            html_table += f'<td style="border: 1px solid #333; padding: 10px; background-color: #1a1a1a; color: #d4af37;">{idx_name}</td>'
            html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: right; background-color: #1a1a1a; color: {color}; font-weight: bold;">{arrow} {ret_val:.2f}%</td>'
            html_table += '</tr>'
        
        html_table += '</tbody></table>'
        st.markdown(html_table, unsafe_allow_html=True)
    
    # Variation Monitor
    st.header("📡 Daily Variation Monitor")
    
    variation_data = []
    for name, df in indices_data.items():
        if df is not None and len(df) >= 2:
            result = get_daily_variation(df)
            if result and len(result) == 5:
                last_date, last_value, prev_date, prev_value, variation = result
                variation_data.append({
                    'Index': name,
                    'Previous Date': prev_date.strftime('%Y-%m-%d'),
                    'Previous Value': prev_value,
                    'Last Date': last_date.strftime('%Y-%m-%d'),
                    'Last Value': last_value,
                    'Variation (%)': variation
                })
    
    if variation_data:
        var_df = pd.DataFrame(variation_data)
        
        # Sort by variation descending
        var_df = var_df.sort_values('Variation (%)', ascending=False)
        
        # Create HTML table with colored variations
        html_table = '<table style="width:100%; border-collapse: collapse;">'
        html_table += '<thead><tr style="background-color: #0a0a0a;">'
        html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Index</th>'
        html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Previous Date</th>'
        html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Previous Value</th>'
        html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Last Date</th>'
        html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Last Value</th>'
        html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Variation</th>'
        html_table += '</tr></thead><tbody>'
        
        for _, row in var_df.iterrows():
            variation = row['Variation (%)']
            color = '#009900' if variation >= 0 else '#ad0000'
            arrow = '▲' if variation >= 0 else '▼'
            
            html_table += '<tr>'
            html_table += f'<td style="border: 1px solid #333; padding: 10px; background-color: #1a1a1a; color: #d4af37; font-weight: bold;">{row["Index"]}</td>'
            html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: center; background-color: #1a1a1a; color: #d4af37;">{row["Previous Date"]}</td>'
            html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: right; background-color: #1a1a1a; color: #d4af37;">{row["Previous Value"]:.2f}</td>'
            html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: center; background-color: #1a1a1a; color: #d4af37;">{row["Last Date"]}</td>'
            html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: right; background-color: #1a1a1a; color: #d4af37;">{row["Last Value"]:.2f}</td>'
            html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: right; background-color: #1a1a1a; color: {color}; font-weight: bold;">{arrow} {abs(variation):.2f}%</td>'
            html_table += '</tr>'
        
        html_table += '</tbody></table>'
        st.markdown(html_table, unsafe_allow_html=True)
    else:
        st.warning("Insufficient data for variation monitor.")
    
    # Monthly Ranking Matrix
    st.header("📅 Monthly Performance Matrix (Last 12 Months)")
    
    monthly_returns = calc_monthly_returns(indices_data, n_months=12)
    monthly_ranking = create_monthly_ranking_matrix(monthly_returns)
    
    if monthly_ranking is not None:
        # Assign colors to indices
        unique_indices = list(indices_data.keys())
        color_palette = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Bold
        index_colors = {idx: color_palette[i % len(color_palette)] for i, idx in enumerate(unique_indices)}
        
        # Create colored HTML table
        html_table = '<table style="width:100%; border-collapse: collapse; font-size: 13px;">'
        html_table += '<thead><tr style="background-color: #0a0a0a;"><th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Rank</th>'
        
        for col in monthly_ranking.columns:
            html_table += f'<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">{col}</th>'
        html_table += '</tr></thead><tbody>'
        
        for idx in monthly_ranking.index:
            html_table += '<tr>'
            html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: center; background-color: #1a1a1a; color: #d4af37; font-weight: bold;">{idx}</td>'
            
            for col in monthly_ranking.columns:
                value = monthly_ranking.loc[idx, col]
                if pd.notna(value) and '|' in str(value):
                    idx_name, ret_str = value.split('|')
                    ret_val = float(ret_str)
                    bg_color = index_colors.get(idx_name, '#1a1a1a')
                    
                    # Color the return value
                    ret_color = '#009900' if ret_val >= 0 else '#ad0000'
                    arrow = '▲' if ret_val >= 0 else '▼'
                    
                    html_table += f'<td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: {bg_color};">'
                    html_table += f'<div style="color: #0a0a0a; font-weight: 600; font-size: 13px;">{idx_name}</div>'
                    html_table += f'<div style="color: {ret_color}; font-weight: bold; font-size: 12px; margin-top: 3px;">{arrow} {abs(ret_val):.2f}%</div>'
                    html_table += '</td>'
                else:
                    html_table += '<td style="border: 1px solid #333; padding: 10px; text-align: center; background-color: #1a1a1a;">-</td>'
            
            html_table += '</tr>'
        
        html_table += '</tbody></table>'
        
        st.markdown(html_table, unsafe_allow_html=True)
    else:
        st.warning("Unable to create monthly ranking matrix.")
    
    # Period Ranking Matrix
    st.header("🎯 Cumulative Performance Matrix (Multi-Period)")
    
    period_ranking = create_period_ranking_matrix(indices_data)
    
    if period_ranking is not None:
        # Create colored HTML table
        html_table = '<table style="width:100%; border-collapse: collapse; font-size: 13px; margin-top: 20px;">'
        html_table += '<thead><tr style="background-color: #0a0a0a;"><th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Rank</th>'
        
        for col in period_ranking.columns:
            html_table += f'<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">{col}</th>'
        html_table += '</tr></thead><tbody>'
        
        for idx in period_ranking.index:
            html_table += '<tr>'
            html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: center; background-color: #1a1a1a; color: #d4af37; font-weight: bold;">{idx}</td>'
            
            for col in period_ranking.columns:
                value = period_ranking.loc[idx, col]
                if pd.notna(value) and '|' in str(value):
                    idx_name, ret_str = value.split('|')
                    ret_val = float(ret_str)
                    bg_color = index_colors.get(idx_name, '#1a1a1a')
                    
                    # Color the return value
                    ret_color = "#009900" if ret_val >= 0 else '#ad0000'
                    arrow = '▲' if ret_val >= 0 else '▼'
                    
                    html_table += f'<td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: {bg_color};">'
                    html_table += f'<div style="color: #0a0a0a; font-weight: 600; font-size: 13px;">{idx_name}</div>'
                    html_table += f'<div style="color: {ret_color}; font-weight: bold; font-size: 12px; margin-top: 3px;">{arrow} {abs(ret_val):.2f}%</div>'
                    html_table += '</td>'
                else:
                    html_table += '<td style="border: 1px solid #333; padding: 10px; text-align: center; background-color: #1a1a1a;">-</td>'
            
            html_table += '</tr>'
        
        html_table += '</tbody></table>'
        
        st.markdown(html_table, unsafe_allow_html=True)
    else:
        st.warning("Unable to create period ranking matrix.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #d4af37; font-family: Montserrat;'>Data sources: ANBIMA, Yahoo Finance, Brazilian Central Bank</p>",
        unsafe_allow_html=True
    )

# Main app logic
if not st.session_state.started:
    show_landing_page()
else:
    show_dashboard()