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
        background-image: url('https://aquamarine-worthy-zebra-762.mypinata.cloud/ipfs/bafybeia6qj2jol4spdjraxdlohre7yg7wofe33awh2udn6harmg3an4mdq');
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
        color: #00d500;
        font-size: 24px;
    }
    
    .arrow-down {
        color: #c60000;
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
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Authorized users (in production, use encrypted database)
AUTHORIZED_USERS = {
    'admin': 'admin123',
    'vini': 'trader2024',
    'guest': 'guest123',
    'trader': 'trader2025'
}

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
    with st.spinner('Carregando dados do mercado...'):
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
                st.warning(f"Não foi possível carregar {name}: {str(e)}")
        
        # Yahoo Finance indices
        yf_indices = [
            ('^BVSP', 'IBOVESPA'),
            ('^GSPC', 'S&P500'),
            ('BRL=X', 'USD/BRL'),
            ('BTC-USD', 'BITCOIN'),
            ('GLD', 'OURO'),
            ('EUE.MI', 'STOXX50')
        ]
        
        for code, name in yf_indices:
            try:
                indices_data[name] = baixar_indice(code, name, 'yf')
            except Exception as e:
                st.warning(f"Não foi possível carregar {name}: {str(e)}")
        
        # CDI
        try:
            indices_data['CDI'] = baixar_indice('CDI', 'CDI', 'cdi')
        except Exception as e:
            st.warning(f"Não foi possível carregar CDI: {str(e)}")
        
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

def calc_monthly_returns(indices_data, n_months=12, method='isolated'):
    """Calculate monthly returns for all indices
    Uses last trading day of previous month as baseline for each month
    method: 'isolated' for month-only returns (each vs last day of previous month)
           'cumulative' for cumulative from last day before the period starts
    """
    monthly_returns = {}
    
    for name, df in indices_data.items():
        if df is None or len(df) == 0:
            continue
        
        # Get daily data
        daily_data = df.copy()
        
        # Get the last n_months + 1 month-end dates to have baseline
        monthly_ends = daily_data.resample('ME').last()
        monthly_period = monthly_ends.tail(n_months + 1)
        
        if len(monthly_period) < 2:
            continue
        
        # Create returns series
        returns_dict = {}
        
        for i in range(1, len(monthly_period)):
            month_end_date = monthly_period.index[i]
            prev_month_end_value = monthly_period.iloc[i-1, 0]
            
            if method == 'isolated':
                # Each month vs last day of previous month
                month_end_value = monthly_period.iloc[i, 0]
                ret = ((month_end_value / prev_month_end_value) - 1) * 100
            else:  # cumulative
                # Cumulative from the baseline (last day before period)
                baseline_value = monthly_period.iloc[0, 0]
                month_end_value = monthly_period.iloc[i, 0]
                ret = ((month_end_value / baseline_value) - 1) * 100
            
            returns_dict[month_end_date] = ret
        
        # Convert to Series
        returns_series = pd.Series(returns_dict)
        returns_series.index.name = monthly_period.index.name
        
        # Convert to DataFrame to match expected format
        returns_df = pd.DataFrame(returns_series, columns=[name])
        
        monthly_returns[name] = returns_df
    
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

def create_yearly_ranking_matrix(indices_data, method='isolated'):
    """Create ranking matrix for yearly periods (current year + last 4 years)
    Uses last trading day of previous year as baseline for each year
    method: 'isolated' for independent year returns (each vs last day of previous year)
           'cumulative' for compounded isolated returns year by year
    """
    current_year = datetime.now().year
    years = [current_year - i for i in range(5)]  # Current + 4 previous years
    years.reverse()  # Oldest to newest
    
    # First, calculate isolated returns for ALL years for ALL indices
    # Using last trading day of previous year as baseline
    all_isolated_returns = {}
    
    for name, df in indices_data.items():
        if df is None or len(df) == 0:
            continue
        
        isolated_returns = {}
        
        for year in years:
            year_data = df[df.index.year == year]
            
            if len(year_data) == 0:
                continue
            
            # Get last trading day of the year
            year_end_value = year_data.iloc[-1, 0]
            
            # Get last trading day of PREVIOUS year as baseline
            prev_year = year - 1
            prev_year_data = df[df.index.year == prev_year]
            
            if len(prev_year_data) > 0:
                prev_year_end_value = prev_year_data.iloc[-1, 0]
                year_return = ((year_end_value / prev_year_end_value) - 1) * 100
                isolated_returns[year] = year_return
        
        all_isolated_returns[name] = isolated_returns
    
    # Create ranking matrix
    max_indices = len(all_isolated_returns)
    ranking_matrix = pd.DataFrame(index=range(1, max_indices + 1), columns=[str(y) for y in years])
    
    # Now populate the matrix based on method
    for i, year in enumerate(years):
        year_returns = {}
        
        for name, isolated_rets in all_isolated_returns.items():
            if method == 'isolated':
                # Just use the isolated return for this year
                if year in isolated_rets:
                    ret_val = isolated_rets[year]
                else:
                    continue
            else:  # cumulative
                # Compound all returns from first year up to this year
                if i == 0:
                    # First year: same as isolated
                    if year in isolated_rets:
                        ret_val = isolated_rets[year]
                    else:
                        continue
                else:
                    # Compound returns: (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
                    cumulative_factor = 1.0
                    all_years_present = True
                    
                    for y in years[:i+1]:  # From first year to current year
                        if y in isolated_rets:
                            cumulative_factor *= (1 + isolated_rets[y] / 100)
                        else:
                            all_years_present = False
                            break
                    
                    if all_years_present:
                        ret_val = (cumulative_factor - 1) * 100
                    else:
                        continue
            
            if pd.notna(ret_val):
                year_returns[name] = ret_val
        
        # Sort and rank
        sorted_returns = sorted(year_returns.items(), key=lambda x: x[1], reverse=True)
        for rank, (idx_name, ret_val) in enumerate(sorted_returns, 1):
            ranking_matrix.loc[rank, str(year)] = f"{idx_name}|{ret_val:.2f}"
    
    return ranking_matrix

def calculate_cumulative_returns_daily(indices_data, selected_indices, period):
    """Calculate cumulative returns on daily basis with ffill for missing days"""
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
    elif period == 'Tudo':
        # Find the earliest date across all selected indices
        earliest_dates = []
        for idx_name in selected_indices:
            if idx_name in indices_data and indices_data[idx_name] is not None:
                earliest_dates.append(indices_data[idx_name].index.min())
        
        if earliest_dates:
            start_date = min(earliest_dates)
        else:
            start_date = end_date - relativedelta(months=36)
    else:
        start_date = end_date - relativedelta(months=36)
    
    # Collect all data for selected indices
    all_data = {}
    for idx_name in selected_indices:
        if idx_name not in indices_data or indices_data[idx_name] is None:
            continue
        
        df = indices_data[idx_name].copy()
        
        # Filter by date
        df = df[df.index >= start_date]
        
        if len(df) == 0:
            continue
        
        all_data[idx_name] = df
    
    if not all_data:
        return pd.DataFrame()
    
    # Get all unique dates from all indices
    all_dates = set()
    for df in all_data.values():
        all_dates.update(df.index)
    
    all_dates = sorted(list(all_dates))
    
    # Create a unified date range
    date_range = pd.DatetimeIndex(all_dates)
    
    # Calculate cumulative returns for each index
    cumulative_returns = pd.DataFrame(index=date_range)
    
    for idx_name, df in all_data.items():
        col = df.columns[0]
        
        # Reindex to all dates, forward fill prices
        prices = df[col].reindex(date_range, method='ffill')
        
        # Calculate daily returns
        daily_returns = prices.pct_change()
        
        # Fill NaN with 0 (0% return when no data and can't ffill)
        daily_returns = daily_returns.fillna(0)
        
        # Calculate cumulative returns from first value
        first_price = prices.iloc[0]
        cumulative_returns[idx_name] = ((prices / first_price) - 1) * 100
    
    return cumulative_returns

# Landing/Login Page
def show_landing_page():
    # Custom CSS for login page matching the uploaded app
    st.markdown("""
        <style>
        /* Remove default padding */
        .main .block-container {
            padding-top: 2rem;
            max-width: 100%;
        }
        
        .login-container {
            max-width: 200px;
            margin: 30px auto; /* reduced top margin */
            padding: 40px;
            background-image: url('https://aquamarine-worthy-zebra-762.mypinata.cloud/ipfs/bafybeigayrnnsuwglzkbhikm32ksvucxecuorcj4k36l4de7na6wcdpjsa');
            background-size: contain;
            background-position: center;
            background-repeat: no-repeat;
            background-color: black;
            border: 2px solid #D4AF37;
            border-radius: 10px;
            aspect-ratio: 16 / 16;
        

        .login-title {
            color: #D4AF37;
            text-align: center;
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 10px;
            letter-spacing: 2px;
            font-family: 'Montserrat', sans-serif;
        }
        
        .login-subtitle {
            color: #888888;
            text-align: center;
            margin-bottom: 30px;
            font-size: 14px;
        }
        
        /* Full page background */
        .stApp {
            background-image: url('https://aquamarine-worthy-zebra-762.mypinata.cloud/ipfs/bafybeia6qj2jol4spdjraxdlohre7yg7wofe33awh2udn6harmg3an4mdq');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<p class="login-title">ÍNDICES DE MERCADO</p>', unsafe_allow_html=True)
        
        # Use different keys for input widgets
        username_input = st.text_input("Usuário", key="login_username_input", placeholder="Digite seu usuário")
        password_input = st.text_input("Senha", type="password", key="login_password_input", placeholder="Digite sua senha")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("ENTRAR", key="login_button", use_container_width=True):
                if username_input in AUTHORIZED_USERS and AUTHORIZED_USERS[username_input] == password_input:
                    st.session_state.authenticated = True
                    st.session_state.started = True
                    st.session_state.user_logged_in = username_input  # Use different key
                    st.rerun()
                else:
                    st.error("❌ Usuário ou senha inválidos")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center; color: #666; font-size: 12px;'>Acesso autorizado apenas</p>",
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

# Main Dashboard
def show_dashboard():
    st.title("📈 Painel de Índices de Mercado")
    
    # Load data
    indices_data = load_all_indices()
    
    if not indices_data:
        st.error("Nenhum dado disponível. Por favor, verifique sua conexão com a internet.")
        return
    
    # Sidebar (can be used for other settings later)
    st.sidebar.title("⚙️ Configurações")
    st.sidebar.info("Use os controles na página principal para personalizar o gráfico.")
    
    available_indices = sorted(list(indices_data.keys()))
    
    # Set initial defaults only once
    default_indices = ['CDI', 'IBOVESPA', 'S&P500'] if all(x in available_indices for x in ['CDI', 'IBOVESPA', 'S&P500']) else available_indices[:3]
    
    # Main content
    st.header("📊 Gráfico de Retornos Acumulados")
    
    # Chart controls in main page
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_indices = st.multiselect(
            "Selecionar Índices",
            options=available_indices,
            default=default_indices,
            help="Escolha quais índices exibir no gráfico"
        )
    
    with col2:
        period = st.selectbox(
            "Selecionar Período",
            options=['Tudo', '36M', '24M', '12M', 'YTD', 'MTD'],
            index=1,
            help="Escolha o período de tempo para análise"
        )
    
    st.markdown("---")
    st.subheader(f"Retornos Acumulados ({period})")
    
    if selected_indices:
        cumulative_returns = calculate_cumulative_returns_daily(indices_data, selected_indices, period)
        
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
                    title='Data'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='#333',
                    title='Retorno Acumulado (%)'
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
            st.warning("Nenhum dado disponível para os índices e período selecionados.")
    else:
        st.info("Por favor, selecione pelo menos um índice.")
    
    # Rankings side by side
    st.header("🏆 Rankings de Performance")
    
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
        # Translate month names
        month_pt = {
            'January': 'Janeiro', 'February': 'Fevereiro', 'March': 'Março',
            'April': 'Abril', 'May': 'Maio', 'June': 'Junho',
            'July': 'Julho', 'August': 'Agosto', 'September': 'Setembro',
            'October': 'Outubro', 'November': 'Novembro', 'December': 'Dezembro'
        }
        current_month_pt = month_pt.get(current_month, current_month)
        st.subheader(f"🥇 Rankings de {current_month_pt} (MTD)")
        
        # Format as table with rank
        mtd_display = mtd_returns.copy()
        mtd_display['Rank'] = range(1, len(mtd_display) + 1)
        mtd_display = mtd_display[['Rank', 'Return']]
        
        # Create HTML table with colored returns
        html_table = '<table style="width:100%; border-collapse: collapse;">'
        html_table += '<thead><tr style="background-color: #0a0a0a;"><th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Rank</th>'
        html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Índice</th>'
        html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Retorno</th></tr></thead><tbody>'
        
        for idx_name, row in mtd_display.iterrows():
            ret_val = row['Return']
            color = '#00d500' if ret_val >= 0 else '#c60000'
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
        st.subheader(f"🥇 Rankings de {current_year} (YTD)")
        
        # Format as table with rank
        ytd_display = ytd_returns.copy()
        ytd_display['Rank'] = range(1, len(ytd_display) + 1)
        ytd_display = ytd_display[['Rank', 'Return']]
        
        # Create HTML table with colored returns
        html_table = '<table style="width:100%; border-collapse: collapse;">'
        html_table += '<thead><tr style="background-color: #0a0a0a;"><th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Rank</th>'
        html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Índice</th>'
        html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Retorno</th></tr></thead><tbody>'
        
        for idx_name, row in ytd_display.iterrows():
            ret_val = row['Return']
            color = '#00d500' if ret_val >= 0 else '#c60000'
            arrow = '▲' if ret_val >= 0 else '▼'
            html_table += '<tr>'
            html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: center; background-color: #1a1a1a; color: #d4af37;">{int(row["Rank"])}</td>'
            html_table += f'<td style="border: 1px solid #333; padding: 10px; background-color: #1a1a1a; color: #d4af37;">{idx_name}</td>'
            html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: right; background-color: #1a1a1a; color: {color}; font-weight: bold;">{arrow} {ret_val:.2f}%</td>'
            html_table += '</tr>'
        
        html_table += '</tbody></table>'
        st.markdown(html_table, unsafe_allow_html=True)
    
    # Variation Monitor
    st.header("📡 Monitor de Variação Diária")
    
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
        html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Índice</th>'
        html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Data Anterior</th>'
        html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Valor Anterior</th>'
        html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Última Data</th>'
        html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Último Valor</th>'
        html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Variação</th>'
        html_table += '</tr></thead><tbody>'
        
        for _, row in var_df.iterrows():
            variation = row['Variation (%)']
            color = '#00d500' if variation >= 0 else '#c60000'
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
        st.warning("Dados insuficientes para o monitor de variação.")
    
    # Monthly Ranking Matrix
    st.header("📅 Matriz de Performance Mensal (Últimos 12 Meses)")
    
    # Radio button for calculation method
    monthly_method = st.radio(
        "Selecione o método de cálculo:",
        options=['Retornos Mensais Isolados', 'Retornos Acumulados (até o fim do mês)'],
        index=0,
        horizontal=True,
        key='monthly_method'
    )
    
    method_key = 'isolated' if monthly_method == 'Retornos Mensais Isolados' else 'cumulative'
    
    monthly_returns = calc_monthly_returns(indices_data, n_months=12, method=method_key)
    monthly_ranking = create_monthly_ranking_matrix(monthly_returns)
    
    if monthly_ranking is not None:
        # Assign colors to indices
        unique_indices = list(indices_data.keys())
        color_palette = [
            "#1f77b4",  # strong blue
            "#9467bd",  # violet
            "#f3710f",  # cyan
            "#fff12f",  # olive yellow
            "#FDACF1",  # medium gray
            "#b06c5f",  # muted brown
            "#de5db7",  # pink (far from red)
            "#9f5495",  # purple
            "#9edae5",  # pale turquoise
            "#ffffff",  # lavender
            "#ffbb78",  # peach (orange but not red)
            "#dbdb8d",  # sand
            "#757575",  # light blue
            "#393b79"   # dark indigo
        ]


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
                    ret_color = "#00d500" if ret_val >= 0 else "#c60000"
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
        st.warning("Não foi possível criar a matriz de ranking mensal.")
    
    # Yearly Ranking Matrix
    st.header("🎯 Matriz de Performance Anual (Multi-Período)")
    
    # Radio button for calculation method
    yearly_method = st.radio(
        "Selecione o método de cálculo:",
        options=['Retornos Anuais Isolados', 'Retornos Acumulados (até o fim do ano)'],
        index=0,
        horizontal=True,
        key='yearly_method'
    )
    
    yearly_method_key = 'isolated' if yearly_method == 'Retornos Anuais Isolados' else 'cumulative'
    
    yearly_ranking = create_yearly_ranking_matrix(indices_data, method=yearly_method_key)
    
    if yearly_ranking is not None:
        # Create colored HTML table
        html_table = '<table style="width:100%; border-collapse: collapse; font-size: 13px; margin-top: 20px;">'
        html_table += '<thead><tr style="background-color: #0a0a0a;"><th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Rank</th>'
        
        for col in yearly_ranking.columns:
            html_table += f'<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">{col}</th>'
        html_table += '</tr></thead><tbody>'
        
        for idx in yearly_ranking.index:
            html_table += '<tr>'
            html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: center; background-color: #1a1a1a; color: #d4af37; font-weight: bold;">{idx}</td>'
            
            for col in yearly_ranking.columns:
                value = yearly_ranking.loc[idx, col]
                if pd.notna(value) and '|' in str(value):
                    idx_name, ret_str = value.split('|')
                    ret_val = float(ret_str)
                    bg_color = index_colors.get(idx_name, '#1a1a1a')
                    
                    # Color the return value
                    ret_color = '#00d500' if ret_val >= 0 else '#c60000'
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
        st.warning("Não foi possível criar a matriz de ranking anual.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #d4af37; font-family: Montserrat;'>Fontes de dados: ANBIMA, Yahoo Finance, Banco Central do Brasil</p>",
        unsafe_allow_html=True
    )

# Main app logic
if not st.session_state.started or not st.session_state.authenticated:
    show_landing_page()
else:
    show_dashboard()
