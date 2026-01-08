import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import requests
from io import BytesIO, StringIO
import warnings
import hashlib
import yfinance as yf
import calendar
import time
import os
from bs4 import BeautifulSoup
import trafilatura
from groq import Groq
import email
from email import policy
from email.parser import BytesParser
import json
import re

warnings.filterwarnings('ignore')


def generate_blue_gradient(n):
    """Generate n shades of blue from light (oldest) to dark (newest)"""
    import colorsys
    colors = []
    for i in range(n):
        # HSL: H=210 (blue), S=100%, L varies from 70% (light) to 20% (dark)
        lightness = 70 - (50 * i / (n - 1)) if n > 1 else 70
        r, g, b = colorsys.hls_to_rgb(210/360, lightness/100, 1.0)
        hex_color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
        colors.append(hex_color)
    return colors

# ═══════════════════════════════════════════════════════════════════════════════
# GROQ API KEY CONFIGURATION - ADD YOUR KEY HERE
# ═══════════════════════════════════════════════════════════════════════════════
# Get your FREE API key from: https://console.groq.com
# Load Groq API key securely
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except FileNotFoundError:
    # Local development fallback (you will set this in .streamlit/secrets.toml)
    st.error("⚠️ Groq API key not found. Add it to .streamlit/secrets.toml for local testing.")
    GROQ_API_KEY = None
except KeyError:
    st.error("⚠️ 'GROQ_API_KEY' not found in secrets. Configure it in Streamlit Cloud settings.")
    GROQ_API_KEY = None
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# SUPABASE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# Load Supabase credentials from secrets
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except:
    SUPABASE_URL = None
    SUPABASE_KEY = None
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Painel de Índices de Mercado",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM STYLING - BLACK & GOLD THEME
# ═══════════════════════════════════════════════════════════════════════════════

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

# Authorized users
AUTHORIZED_USERS = {
    'admin': 'admin123',
    'vini': 'trader2024',
    'guest': 'guest123',
    'trader': 'trader2025'
}

# ═══════════════════════════════════════════════════════════════════════════════
# TESOURO DIRETO FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_tesouro_direto_data():
    """Load Tesouro Direto data from government API"""
    try:
        url_td = "https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/precotaxatesourodireto.csv"
        td_df = pd.read_csv(url_td, encoding='utf-8', sep=';')
        td_df['Data Base'] = pd.to_datetime(td_df['Data Base'], dayfirst=True).dt.date
        td_df['Data Vencimento'] = pd.to_datetime(td_df['Data Vencimento'], dayfirst=True).dt.date
        return td_df
    except Exception as e:
        st.error(f"Erro ao carregar dados do Tesouro Direto: {str(e)}")
        return None

def products_td(td_df, bond):
    """Extract products data for a specific bond type"""
    df = td_df[td_df['Tipo Titulo'] == bond].copy()
    latest_dates = np.sort(pd.unique(df['Data Base']))[-10:]
    df = df[df['Data Base'] >= latest_dates[0]]
    df.set_index('Data Base', inplace=True)

    index_td = np.sort(pd.unique(df.index))
    maturity_td = np.sort(pd.unique(df['Data Vencimento']))
    products_df = pd.DataFrame(index=index_td, columns=maturity_td)

    for v in range(0, products_df.shape[1]):
        temp = df[df['Data Vencimento'] == maturity_td[v]].sort_index(ascending=True)
        products_df.iloc[:, v] = pd.to_numeric(temp['Taxa Compra Manha'].astype(str).str.replace(',', '.', regex=False), errors='coerce')

    return products_df

def create_td_chart(products_df, selected_dates, bond_name):
    """Create yield curve chart for Tesouro Direto"""
    fig = go.Figure()
    
    for date in selected_dates:
        if date in products_df.index:
            rates = products_df.loc[date].dropna()
            if len(rates) > 0:
                # Extract years from maturity dates
                maturity_years = [pd.Timestamp(mat).year for mat in rates.index]
                
                fig.add_trace(go.Scatter(
                    x=maturity_years,
                    y=rates.values,
                    mode='lines+markers',
                    name=str(date),
                    line=dict(width=2),
                    marker=dict(size=8)
                ))
    
    fig.update_layout(
        xaxis_title='Ano de Vencimento',
        yaxis_title='Taxa (%)',
        plot_bgcolor='#0a0a0a',
        paper_bgcolor='#1a1a1a',
        font=dict(color='#d4af37', family='Montserrat'),
        xaxis=dict(showgrid=True, gridcolor='#333'),
        yaxis=dict(showgrid=True, gridcolor='#333'),
        legend=dict(bgcolor='#1a1a1a', bordercolor='#d4af37', borderwidth=1),
        height=500
    )
    
    return fig

def calculate_td_rate_variations(products_df):
    """Calculate rate variations for different periods"""
    if len(products_df) < 2:
        return {}
    
    # Get latest rates (newest day)
    latest_data = products_df.iloc[-1].dropna()
    
    variations = {}
    days_back_list = [1, 2, 3, 5, 9]
    
    for days_back in days_back_list:
        if days_back >= len(products_df):
            continue
            
        past_data = products_df.iloc[-(days_back + 1)].dropna()
        
        # Calculate variation for each maturity
        var_dict = {}
        for maturity in latest_data.index:
            if maturity in past_data.index:
                variation = latest_data[maturity] - past_data[maturity]
                var_dict[maturity] = variation
        
        # Get top 10 by absolute value
        variations[days_back] = var_dict.items()
    
    return variations

def create_td_table(products_df):
    """Create table with all rates"""
    if products_df is None or len(products_df) == 0:
        return None
    
    # Reverse order: oldest dates first (ascending index), newest maturities first (descending columns)
    table_df = products_df.copy()
    table_df = table_df.sort_index(ascending=True)  # Oldest dates first
    table_df = table_df[sorted(table_df.columns, reverse=False)]  # Newest maturities first
    
    # Format dates for display
    table_df.index = [str(date) for date in table_df.index]
    table_df.columns = [str(date) for date in table_df.columns]
    
    return table_df

def get_maturity_time_series(td_df, bond, maturity_date):
    """Get complete time series for a specific maturity date"""
    df = td_df[td_df['Tipo Titulo'] == bond].copy()
    df = df[df['Data Vencimento'] == maturity_date]
    df = df.sort_values('Data Base')
    df.set_index('Data Base', inplace=True)
    
    # Extract rates
    rates = pd.to_numeric(df['Taxa Compra Manha'].astype(str).str.replace(',', '.', regex=False), errors='coerce')
    
    return rates

def get_all_maturities_time_series(td_df, bond):
    """Get time series for all current maturities of a bond"""
    df = td_df[td_df['Tipo Titulo'] == bond].copy()
    
    # Get current maturities (those available in recent data)
    recent_dates = np.sort(pd.unique(df['Data Base']))[-10:]
    recent_df = df[df['Data Base'].isin(recent_dates)]
    # Sort in descending order (newest maturity first)
    current_maturities = sorted(pd.unique(recent_df['Data Vencimento']), reverse=False)
    
    # Get time series for each maturity
    time_series_dict = {}
    for maturity in current_maturities:
        series = get_maturity_time_series(td_df, bond, maturity)
        if len(series) > 0:
            time_series_dict[maturity] = series
    
    return time_series_dict

def create_maturity_time_series_chart(time_series, maturity_date, bond_name):
    """Create time series chart for a specific maturity with percentile lines"""
    fig = go.Figure()
    
    if len(time_series) > 0:
        # Convert dates to datetime for plotting
        dates = [pd.Timestamp(d) for d in time_series.index]
        
        # Calculate percentiles
        p25 = time_series.quantile(0.25)
        p50 = time_series.quantile(0.50)
        p75 = time_series.quantile(0.75)
        
        # Add main time series line
        fig.add_trace(go.Scatter(
            x=dates,
            y=time_series.values,
            mode='lines',
            name=f'Vencimento {maturity_date.year}',
            line=dict(width=2, color='#d4af37'),
            marker=dict(size=6, color='#d4af37')
        ))
        
        # Add 25th percentile line
        fig.add_trace(go.Scatter(
            x=[dates[0], dates[-1]],
            y=[p25, p25],
            mode='lines',
            name=f'P25: {p25:.2f}%',
            line=dict(width=2, color='#4169E1', dash='dash'),
            showlegend=True
        ))
        
        # Add 50th percentile (median) line
        fig.add_trace(go.Scatter(
            x=[dates[0], dates[-1]],
            y=[p50, p50],
            mode='lines',
            name=f'P50: {p50:.2f}%',
            line=dict(width=2, color='#32CD32', dash='dash'),
            showlegend=True
        ))
        
        # Add 75th percentile line
        fig.add_trace(go.Scatter(
            x=[dates[0], dates[-1]],
            y=[p75, p75],
            mode='lines',
            name=f'P75: {p75:.2f}%',
            line=dict(width=2, color='#FF6347', dash='dash'),
            showlegend=True
        ))
        
        # Add annotations for percentiles on the right side
        fig.add_annotation(
            x=dates[-1],
            y=p25,
            text=f"P25: {p25:.2f}%",
            showarrow=False,
            xanchor='left',
            xshift=10,
            font=dict(size=10, color='#4169E1')
        )
        
        fig.add_annotation(
            x=dates[-1],
            y=p50,
            text=f"P50: {p50:.2f}%",
            showarrow=False,
            xanchor='left',
            xshift=10,
            font=dict(size=10, color='#32CD32')
        )
        
        fig.add_annotation(
            x=dates[-1],
            y=p75,
            text=f"P75: {p75:.2f}%",
            showarrow=False,
            xanchor='left',
            xshift=10,
            font=dict(size=10, color='#FF6347')
        )
    
    fig.update_layout(
        xaxis_title='Dia de Negociação',
        yaxis_title='Taxa (%)',
        plot_bgcolor='#0a0a0a',
        paper_bgcolor='#1a1a1a',
        font=dict(color='#d4af37', family='Montserrat'),
        xaxis=dict(showgrid=True, gridcolor='#333'),
        yaxis=dict(showgrid=True, gridcolor='#333'),
        legend=dict(bgcolor='#1a1a1a', bordercolor='#d4af37', borderwidth=1),
        height=500
    )
    
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# MARKET INDICES FUNCTIONS (from original code)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def baixar_indice(indice, name, source, start_date='2015-01-01'):
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
        df = yf.download(indice, start=start_date, end=datetime.today(), interval='1d', progress=False)['Close']
        if isinstance(df, pd.Series):
            df = df.to_frame(name=name)
        else:
            df.columns = [name]
        return df
    
    elif source == 'bcb':
        serie_codigo = 12
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie_codigo}/dados"

        def fetch_data(days_back):
            start_date = datetime.today() - relativedelta(days=days_back)
            params = {
                'formato': 'csv',
                'dataInicial': start_date.strftime('%d/%m/%Y'),
                'dataFinal': datetime.today().strftime('%d/%m/%Y')
            }
            headers = {'Cache-Control': 'no-cache', 'Pragma': 'no-cache'}
            response = requests.get(url, params=params, headers=headers, timeout=30)
            return response

        response = fetch_data(3653)

        if response.status_code != 200:
            response = fetch_data(3652)

        if response.status_code == 200:
            data = StringIO(response.text)
            cdi_df = pd.read_csv(data, sep=";", decimal=",", encoding="latin1")

            cdi_df['valor'] = pd.to_numeric(
                cdi_df['valor'].astype(str).str.replace(',', '.', regex=False),
                errors='coerce'
            ) / 100
            cdi_df['data'] = pd.to_datetime(cdi_df['data'], format='%d/%m/%Y')
            cdi_df.set_index('data', inplace=True)

            # Compute cumulative CDI index
            cdi_df['CDI'] = (1 + cdi_df['valor']).cumprod()

            return pd.DataFrame(cdi_df['CDI'])

        else:
            st.error(f"Erro ao acessar API CDI: {response.status_code}")
            return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_all_indices():
    """Load all indices data"""
    with st.spinner('Carregando dados do mercado...'):
        indices_data = {}
        
        # ANBIMA indices
        anbima_indices = [
            ('IMAB5', 'IMA-B5'), ('IMAB5MAIS', 'IMA-B5+'), ('IMAS', 'IMA-S'),
            ('IDADI', 'IDA-DI'), ('IDAIPCA', 'IDA-IPCA'), ('IRFM', 'IRF-M'), ('IHFA', 'IHFA')
        ]
        
        for code, name in anbima_indices:
            try:
                indices_data[name] = baixar_indice(code, name, 'anbima')
            except Exception as e:
                st.warning(f"Não foi possível carregar {name}: {str(e)}")
        
        # Yahoo Finance indices
        yf_indices = [
            ('^BVSP', 'IBOVESPA'), ('^GSPC', 'S&P 500 (USD)'), ('BRL=X', 'USD/BRL'), ('BTC-USD', 'BITCOIN'), ('GLD', 'OURO'), ('EUE.MI', 'STOXX50')
        ]
        
        for code, name in yf_indices:
            try:
                indices_data[name] = baixar_indice(code, name, 'yf')
            except Exception as e:
                st.warning(f"Não foi possível carregar {name}: {str(e)}")
        
        # CDI
        try:
            indices_data['CDI'] = baixar_indice('CDI', 'CDI', 'bcb')
        except Exception as e:
            st.warning(f"Não foi possível carregar CDI: {str(e)}")
        
        # Create S&P 500 (BRL) by multiplying S&P 500 (USD) by USD/BRL exchange rate
        if 'S&P 500 (USD)' in indices_data and 'USD/BRL' in indices_data:
            try:
                sp500_usd = indices_data['S&P 500 (USD)'].copy()
                usd_brl = indices_data['USD/BRL'].copy()
                
                # Get the column names
                sp500_col = sp500_usd.columns[0]
                usd_brl_col = usd_brl.columns[0]
                
                # Reindex USD/BRL to S&P 500 (USD) trading days and forward fill
                usd_brl_aligned = usd_brl.reindex(sp500_usd.index).ffill()
                
                # Multiply to get S&P 500 in BRL (element-wise multiplication)
                sp500_brl = pd.DataFrame(
                    sp500_usd[sp500_col].values * usd_brl_aligned[usd_brl_col].values,
                    index=sp500_usd.index,
                    columns=['S&P 500 (BRL)']
                )
                
                indices_data['S&P 500 (BRL)'] = sp500_brl
            except Exception as e:
                st.warning(f"Não foi possível criar S&P 500 (BRL): {str(e)}")
        
        return indices_data

def calc_returns(df):
    """Compute MTD, YTD, 12M, 24M, and 36M returns"""
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
        if start_date <= close.index.min():
            return close.iloc[0, 0]
        idx = close.index.get_indexer([start_date], method='ffill')
        if idx[0] == -1:
            return np.nan
        return close.iloc[idx[0], 0]

    start_mtd_date = latest.replace(day=1)
    start_ytd_date = pd.to_datetime(f"{latest.year}-01-01")
    start_12m_date = latest - pd.DateOffset(months=12)
    start_24m_date = latest - pd.DateOffset(months=24)
    start_36m_date = latest - pd.DateOffset(months=36)

    start_mtd = get_start_price(start_mtd_date - pd.Timedelta(days=1))
    start_ytd = get_start_price(start_ytd_date - pd.Timedelta(days=1))
    start_12m = get_start_price(start_12m_date)
    start_24m = get_start_price(start_24m_date)
    start_36m = get_start_price(start_36m_date)

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
    """Calculate monthly returns for all indices"""
    monthly_returns = {}
    
    for name, df in indices_data.items():
        if df is None or len(df) == 0:
            continue
        
        daily_data = df.copy()
        monthly_ends = daily_data.resample('ME').last()
        monthly_period = monthly_ends.tail(n_months + 1)
        
        if len(monthly_period) < 2:
            continue
        
        returns_dict = {}
        
        for i in range(1, len(monthly_period)):
            month_end_date = monthly_period.index[i]
            prev_month_end_value = monthly_period.iloc[i-1, 0]
            
            if method == 'isolated':
                # Isolated: compare current month end to previous month end
                month_end_value = monthly_period.iloc[i, 0]
                ret = ((month_end_value / prev_month_end_value) - 1) * 100
            else:
                # Cumulative: compare latest month end (last in series) to month before current column
                # For example: Nov cumulative = last day of Nov vs last day of Oct
                # Apr cumulative = last day of Nov (latest) vs last day of Mar (month before Apr column)
                latest_month_end_value = monthly_period.iloc[-1, 0]  # Last available month
                month_before_column = monthly_period.iloc[i-1, 0]  # Month before the column month
                ret = ((latest_month_end_value / month_before_column) - 1) * 100
            
            returns_dict[month_end_date] = ret
        
        returns_series = pd.Series(returns_dict)
        returns_series.index.name = monthly_period.index.name
        returns_df = pd.DataFrame(returns_series, columns=[name])
        
        monthly_returns[name] = returns_df
    
    return monthly_returns

def create_monthly_ranking_matrix(monthly_returns):
    """Create ranking matrix for monthly returns"""
    if not monthly_returns:
        return None
    
    all_months = set()
    for returns in monthly_returns.values():
        all_months.update(returns.index)
    
    all_months = sorted(list(all_months))
    
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
        
        sorted_month = sorted(month_returns.items(), key=lambda x: x[1], reverse=True)
        monthly_data[month] = sorted_month
    
    max_indices = max(len(data) for data in monthly_data.values()) if monthly_data else 0
    ranking_matrix = pd.DataFrame(
        index=range(1, max_indices + 1),
        columns=[m.strftime('%m/%Y') for m in all_months]
    )
    
    for month, sorted_indices in monthly_data.items():
        month_str = month.strftime('%m/%Y')
        for rank, (idx_name, ret_val) in enumerate(sorted_indices, 1):
            ranking_matrix.loc[rank, month_str] = f"{idx_name}|{ret_val:.2f}"
    
    return ranking_matrix

def create_yearly_ranking_matrix(indices_data, method='isolated'):
    """Create ranking matrix for yearly periods (10 years)"""
    current_year = datetime.now().year
    years = [current_year - i for i in range(10)]  # 10 years instead of 5
    years.reverse()
    
    all_isolated_returns = {}
    
    for name, df in indices_data.items():
        if df is None or len(df) == 0:
            continue
        
        isolated_returns = {}
        
        for year in years:
            year_data = df[df.index.year == year]
            
            if len(year_data) == 0:
                continue
            
            year_end_value = year_data.iloc[-1, 0]
            prev_year = year - 1
            prev_year_data = df[df.index.year == prev_year]
            
            if len(prev_year_data) > 0:
                prev_year_end_value = prev_year_data.iloc[-1, 0]
                year_return = ((year_end_value / prev_year_end_value) - 1) * 100
                isolated_returns[year] = year_return
        
        all_isolated_returns[name] = isolated_returns
    
    max_indices = len(all_isolated_returns)
    ranking_matrix = pd.DataFrame(index=range(1, max_indices + 1), columns=[str(y) for y in years])
    
    for i, year in enumerate(years):
        year_returns = {}
        
        for name, isolated_rets in all_isolated_returns.items():
            if method == 'isolated':
                if year in isolated_rets:
                    ret_val = isolated_rets[year]
                else:
                    continue
            else:
                # Cumulative: variation between latest return of current year and last trading day of year before the column year
                # For example, 2025 cumulative = variation between last day of 2025 and last day of 2024
                # For 2020 cumulative = variation between last day of 2025 and last day of 2019
                
                # Get the last available year with data
                available_years = sorted([y for y in isolated_rets.keys() if y >= year])
                if not available_years:
                    continue
                    
                latest_year = max(available_years)
                
                # Get the end value of the latest year
                year_data = indices_data[name][indices_data[name].index.year == latest_year]
                if len(year_data) == 0:
                    continue
                latest_year_end_value = year_data.iloc[-1, 0]
                
                # Get the end value of the year before the column year
                prev_year = year - 1
                prev_year_data = indices_data[name][indices_data[name].index.year == prev_year]
                
                if len(prev_year_data) > 0:
                    prev_year_end_value = prev_year_data.iloc[-1, 0]
                    ret_val = ((latest_year_end_value / prev_year_end_value) - 1) * 100
                else:
                    continue
            
            if pd.notna(ret_val):
                year_returns[name] = ret_val
        
        sorted_returns = sorted(year_returns.items(), key=lambda x: x[1], reverse=True)
        for rank, (idx_name, ret_val) in enumerate(sorted_returns, 1):
            ranking_matrix.loc[rank, str(year)] = f"{idx_name}|{ret_val:.2f}"
    
    return ranking_matrix

def calculate_cumulative_returns_daily(indices_data, selected_indices, period):
    """Calculate cumulative returns on daily basis"""
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
    elif period == '120M':
        start_date = end_date - relativedelta(months=120)
    elif period == 'Tudo':
        earliest_dates = []
        for idx_name in selected_indices:
            if idx_name in indices_data and indices_data[idx_name] is not None:
                df = indices_data[idx_name]
                if len(df) > 0 and hasattr(df.index, 'min'):
                    min_date = df.index.min()
                    # Only add if it's a valid Timestamp
                    if pd.notna(min_date) and isinstance(min_date, pd.Timestamp):
                        earliest_dates.append(min_date)
        
        if earliest_dates:
            start_date = min(earliest_dates)
        else:
            start_date = end_date - relativedelta(months=36)
    else:
        start_date = end_date - relativedelta(months=36)
    
    all_data = {}
    for idx_name in selected_indices:
        if idx_name not in indices_data or indices_data[idx_name] is None:
            continue
        
        df = indices_data[idx_name].copy()
        df = df[df.index >= start_date]
        
        if len(df) == 0:
            continue
        
        all_data[idx_name] = df
    
    if not all_data:
        return pd.DataFrame()
    
    all_dates = set()
    for df in all_data.values():
        all_dates.update(df.index)
    
    all_dates = sorted(list(all_dates))
    date_range = pd.DatetimeIndex(all_dates)
    
    cumulative_returns = pd.DataFrame(index=date_range)
    
    for idx_name, df in all_data.items():
        col = df.columns[0]
        prices = df[col].reindex(date_range, method='ffill')
        daily_returns = prices.pct_change()
        daily_returns = daily_returns.fillna(0)
        
        first_price = prices.iloc[0]
        cumulative_returns[idx_name] = ((prices / first_price) - 1) * 100
    
    return cumulative_returns

# ═══════════════════════════════════════════════════════════════════════════════
# LANDING/LOGIN PAGE
# ═══════════════════════════════════════════════════════════════════════════════

def show_landing_page():
    st.markdown("""
        <style>
        .main .block-container {
            padding-top: 2rem;
            max-width: 100%;
        }
        
        .login-container {
            max-width: 200px;
            margin: 50px auto;
            padding: 40px;
            background-image: url('https://aquamarine-worthy-zebra-762.mypinata.cloud/ipfs/bafybeigayrnnsuwglzkbhikm32ksvucxecuorcj4k36l4de7na6wcdpjsa');
            background-size: contain;
            background-position: center;
            background-repeat: no-repeat;
            background-color: black;
            border: 2px solid #D4AF37;
            border-radius: 10px;
            aspect-ratio: 1 / 1;
        }

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
        st.markdown('<p class="login-title">INDICES ANALYTICS</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #888888; text-align: center; margin-bottom: 20px;">Please sign in to continue</p>', unsafe_allow_html=True)
        
        username_input = st.text_input(key="login_username_input", placeholder="Digite seu usuário")
        password_input = st.text_input(type="password", key="login_password_input", placeholder="Digite sua senha")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("ENTRAR", key="login_button", use_container_width=True):
                if username_input in AUTHORIZED_USERS and AUTHORIZED_USERS[username_input] == password_input:
                    st.session_state.authenticated = True
                    st.session_state.started = True
                    st.session_state.user_logged_in = username_input
                    st.rerun()
                else:
                    st.error("❌ Usuário ou senha inválidos")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center; color: #666; font-size: 12px;'>Acesso autorizado apenas</p>",
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# NEWS AGGREGATOR FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# NEWSLETTER PROCESSING FUNCTIONS (from news_v5.py)
def init_groq_client():
    """Initialize Groq client"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        st.error("🔑 GROQ_API_KEY not found in environment variables")
        st.info("Please set your Groq API key in .streamlit/secrets.toml or environment variables")
        return None
    
    try:
        client = Groq(api_key=groq_api_key)
        return client
    except Exception as e:
        st.error(f"Error initializing Groq client: {e}")
        return None

# Try to import Supabase (optional)
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

def init_supabase_client():
    """Initialize Supabase client (optional)"""
    if not SUPABASE_AVAILABLE:
        return None
    
    # Use the variables loaded from st.secrets at the top of the file
    supabase_url = SUPABASE_URL
    supabase_key = SUPABASE_KEY
    
    if not supabase_url or not supabase_key:
        st.warning("⚠️ Supabase credentials not found. Database features disabled.")
        return None
    
    try:
        client = create_client(supabase_url, supabase_key)
        return client
    except Exception as e:
        st.warning(f"⚠️ Could not connect to Supabase: {e}")
        return None


def parse_eml_file(uploaded_file):
    """Parse .eml file and extract content"""
    try:
        # Read the uploaded file
        raw_email = uploaded_file.read()
        
        # Parse email
        msg = BytesParser(policy=policy.default).parsebytes(raw_email)
        
        # Extract metadata
        subject = msg.get('subject', 'No Subject')
        sender = msg.get('from', 'Unknown')
        recipient = msg.get('to', 'Unknown')
        date = msg.get('date', 'Unknown')
        
        # Extract HTML content
        html_content = None
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == 'text/html':
                    html_content = part.get_content()
                    break
        else:
            if msg.get_content_type() == 'text/html':
                html_content = msg.get_content()
        
        if not html_content:
            return None
        
        return {
            'subject': subject,
            'sender': sender,
            'recipient': recipient,
            'date': date,
            'html': html_content
        }
        
    except Exception as e:
        st.error(f"Error parsing email: {e}")
        return None


def clean_html_for_extraction(html_content):
    """Clean and reduce HTML size"""
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style tags
    for tag in soup(['script', 'style', 'meta', 'link', 'head']):
        tag.decompose()
    
    # Remove images (keep alt text if any)
    for img in soup.find_all('img'):
        alt_text = img.get('alt', '')
        if alt_text:
            img.replace_with(f"[IMAGE: {alt_text}]")
        else:
            img.decompose()
    
    # Get clean HTML
    clean_html = str(soup)
    
    return clean_html



def create_smart_summary(groq_client, text, max_chars=250, title=None):
    """Create AI-powered smart summary - NO ellipsis, NO redundancy with title"""
    if len(text) <= max_chars:
        return text
    
    try:
        # Prompt AI to create non-redundant summary
        if title:
            prompt = f"""Resuma este texto em EXATAMENTE {max_chars} caracteres ou menos.

TÍTULO: {title}

TEXTO: {text}

Instruções CRÍTICAS:
- Máximo de {max_chars} caracteres
- NÃO repita palavras ou conceitos do TÍTULO
- Foque em detalhes, contexto e informações COMPLEMENTARES ao título
- Se o título diz "Fed corta juros", o resumo deve focar em IMPACTOS, RAZÕES, DETALHES - nunca repita "Fed cortou juros"
- NÃO termine com "..."
- Frase completa e natural
- Em português

Resumo:"""
        else:
            prompt = f"""Resuma este texto em EXATAMENTE {max_chars} caracteres ou menos.

Texto: {text}

Instruções:
- Máximo de {max_chars} caracteres
- NÃO termine com "..."
- Frase completa e natural
- Em português

Resumo:"""
        
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        
        summary = response.choices[0].message.content.strip()
        
        # Clean up response
        summary = summary.replace('"', '').replace("'", '').replace('Resumo:', '').strip()
        
        # Remove trailing ellipsis if AI added it
        if summary.endswith('...'):
            summary = summary[:-3].strip()
        
        # If still too long, truncate at word boundary
        if len(summary) > max_chars:
            summary = summary[:max_chars]
            last_space = summary.rfind(' ')
            if last_space > max_chars * 0.8:
                summary = summary[:last_space]
        
        return summary.strip()
        
    except Exception as e:
        # Fallback: smart truncation
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.8:
            return truncated[:last_space].strip()
        return truncated.strip()




def extract_investnews(groq_client, html_content):
    """Extract InvestNews newsletter content"""
    
    # Clean HTML
    cleaned_content = clean_html_for_extraction(html_content)
    
    # Limit content size
    char_limit = 50000
    if len(cleaned_content) > char_limit:
        cleaned_content = cleaned_content[:char_limit]
    
    # Extraction prompt
    extraction_prompt = """Você é um especialista em análise de newsletters do InvestNews.

🎯 OBJETIVO: Extrair e resumir as 4 seções principais da newsletter de forma concisa.

📝 ESTRUTURA DA NEWSLETTER:
1. MATÉRIA PRINCIPAL - A primeira grande notícia do dia
2. DESTAQUES - 3 notícias curtas em bullets
3. SEGUNDA MATÉRIA - Segunda história importante, sempre aparece DEPOIS dos destaques ou "UMA IMAGEM"
4. VALE PARAR PARA LER - Recomendações de leitura (sempre 2 itens)

🔍 REGRAS DE RESUMO (IMPORTANTE):
- MATÉRIA PRINCIPAL: Extraia o texto completo OU resuma em até 500 caracteres (o que for mais conciso)
- Cada DESTAQUE: Extraia o texto completo OU resuma em até 250 caracteres (o que for mais conciso)
- SEGUNDA MATÉRIA: Extraia o texto completo OU resuma em até 500 caracteres (o que for mais conciso)
- Cada VALE PARAR: Extraia o texto completo OU resuma em até 250 caracteres (o que for mais conciso)

🔗 EXTRAÇÃO DE LINKS (MUITO IMPORTANTE):
- MATÉRIA PRINCIPAL: Procure por "Leia mais nesta reportagem" - capture o URL completo (https://...) que aparece próximo
- DESTAQUES: Cada destaque tem um link - procure por URLs (https://...) no texto ou próximo ao título
- SEGUNDA MATÉRIA: Procure por "Leia mais nesta reportagem do Wall Street Journal" - capture o URL
- VALE PARAR PARA LER: Cada item tem um link - capture URLs completos

📋 FORMATO JSON:
{
  "main_story": {"title": "...", "content": "≤500 chars", "url": "https://..."},
  "highlights": [
    {"title": "...", "content": "≤250 chars", "url": "https://..."},
    {"title": "...", "content": "≤250 chars", "url": "https://..."},
    {"title": "...", "content": "≤250 chars", "url": "https://..."}
  ],
  "segunda_materia": {"title": "...", "content": "≤500 chars", "url": "https://..."} ou null,
  "vale_parar_para_ler": [
    {"title": "...", "content": "≤250 chars", "url": "https://..."},
    {"title": "...", "content": "≤250 chars", "url": "https://..."}
  ]
}

⚠️ IMPORTANTE: Se não encontrar URL para algum item, use null para o campo "url".

Retorne APENAS JSON (sem ``` ou markdown):"""
    
    try:
        messages = [
            {
                "role": "user",
                "content": f"{extraction_prompt}\n\nCONTEÚDO:\n{cleaned_content}"
            }
        ]
        
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.05,
            max_tokens=8192,
            top_p=0.9
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Extract JSON
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        
        sections = json.loads(response_text)
        return sections
        
    except Exception as e:
        st.error(f"Error in Groq extraction: {e}")
        return None


def extract_exame(groq_client, html_content):
    """Extract Exame newsletter content"""
    
    cleaned_content = clean_html_for_extraction(html_content)
    
    char_limit = 40000
    if len(cleaned_content) > char_limit:
        cleaned_content = cleaned_content[:char_limit]
    
    extraction_prompt = """Você é um especialista em análise de newsletters da Exame.

🎯 OBJETIVO: Extrair TODAS as manchetes (headlines) da newsletter com seus conteúdos e links.

📝 ESTRUTURA DA NEWSLETTER EXAME:
- Começa com "VOCÊ VÊ NA DESPERTA:" (preview das notícias)
- Depois vem as manchetes elaboradas (tipicamente 5-7 manchetes)
- PRIMEIRA manchete: font-size 26px (maior)
- DEMAIS manchetes: font-size 20px
- Cada manchete tem conteúdo explicativo completo
- Cada manchete tem um link "SAIBA MAIS" ou link direto

🔍 CRITÉRIOS DE IDENTIFICAÇÃO:
**Como identificar uma manchete:**
1. Texto com fonte 26px OU 20px
2. Comprimento entre 20-200 caracteres
3. Parece ser um título de notícia
4. Seguido de conteúdo explicativo
5. Tem um link associado (geralmente click.comunicacao-exame.com)

**Exemplos de manchetes desta newsletter:**
- "O que muda no Imposto de Renda"
- "Moraes determina que Bolsonaro cumpra pena na Superintendência da PF"
- "Google volta ao topo: a gigante que hesitou na IA está rumo aos US$ 4 trilhões"
- "EUA despenca e Brasil sobe em ranking global de atração de talentos"
- "SoftBank perde R$ 549 bilhões após aposta pesada na OpenAI"
- "Restaurante de Belém deve faturar R$ 40 milhões — sem aumentar preços na COP30"
- "As apostas do YouTube para tomar a dianteira na corrida do streaming"

🚨 ATENÇÃO: Esta newsletter tem PELO MENOS 5-7 MANCHETES. Você DEVE encontrar TODAS elas!

🔍 ESTRATÉGIA DE EXTRAÇÃO:
1. Procure por TODOS os textos com font-size 26px ou 20px
2. Para cada manchete encontrada:
   - Título: texto da manchete
   - Conteúdo: parágrafo(s) explicativo(s) que vem logo após a manchete
   - URL: link "click.comunicacao-exame.com" mais próximo da manchete

3. NÃO pare após 3 ou 4 manchetes - CONTINUE até encontrar TODAS (5-7 típicas)

📋 FORMATO JSON:
{
  "headlines": [
    {
      "title": "título completo da manchete 1",
      "content": "texto resumido em 175 caracteres da notícia",
      "url": "URL completa do link https://click.comunicacao-exame.com/..."
    },
    {
      "title": "título completo da manchete 2",
      "content": "texto resumido em 175 caracteres da notícia",
      "url": "URL completa..."
    },
    ... (CONTINUE para TODAS as manchetes - mínimo 5, típico 5-7)
  ]
}

✅ CHECKLIST ANTES DE RETORNAR:
□ Encontrei a primeira manchete (26px)?
□ Encontrei TODAS as manchetes com 20px?
□ Tenho pelo menos 5 manchetes? (se não, PROCURE MAIS!)
□ Cada manchete tem título, conteúdo E URL?
□ Os URLs são do tipo "click.comunicacao-exame.com"?
□ O conteúdo está resumido em 175 caracteres?

⚠️ ERROS COMUNS A EVITAR:
❌ Parar após 3-4 manchetes (newsletter tem 5-7!)
❌ Pular manchetes no meio do documento
❌ Incluir apenas manchetes com URLs encontradas facilmente

✅ REGRAS:
- Extraia o conteúdo resumido em 175 caracteres de cada manchete
- Garanta que capturou a URL correta para cada manchete
- Se não encontrar URL, coloque null mas INCLUA a manchete
- Retorne TODAS as manchetes encontradas (mínimo 5, típico 5-7)
- A primeira manchete é maior (26px), trate todas igualmente

Retorne APENAS JSON (sem ``` ou markdown):"""
    
    try:
        messages = [
            {
                "role": "user",
                "content": f"{extraction_prompt}\n\nCONTEÚDO DA NEWSLETTER:\n{cleaned_content}"
            }
        ]
        
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.05,
            max_tokens=8192,
            top_p=0.9
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Extract JSON
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        
        sections = json.loads(response_text)
        
        # Validate we have enough headlines
        if sections.get('headlines') and len(sections['headlines']) < 5:
            st.warning(f"⚠️ Only {len(sections['headlines'])} headlines found. Expected 5-7. Try processing again.")
        
        return sections
        
    except Exception as e:
        st.error(f"Error in Groq extraction: {e}")
        return None



def extract_bloomberg(groq_client, html_content):
    """Extract Bloomberg newsletter with Portuguese translation and smart summaries"""
    
    cleaned_content = clean_html_for_extraction(html_content)
    
    char_limit = 50000
    if len(cleaned_content) > char_limit:
        cleaned_content = cleaned_content[:char_limit]
    
    extraction_prompt = """Você é um especialista em análise e tradução de newsletters da Bloomberg.

🎯 OBJETIVO: Extrair notícias e Deep Dive, traduzir para português, retornar conteúdo completo.

📝 ESTRUTURA:
1. "Good morning." (IGNORAR)
2. "Markets Snapshot" (IGNORAR)
3. **NOTÍCIAS EM PARÁGRAFOS** ← COMEÇAR
   - 8-12 notícias com headline em NEGRITO
4. **"Deep Dive:"** ← SEÇÃO 1
   - Conteúdo COMPLETO
5. **"The Big Take"** ← SEÇÃO 2 (CAPTURAR)
   - Conteúdo COMPLETO
6. **"Opinion"** ← SEÇÃO 3 (CAPTURAR)
   - Conteúdo COMPLETO
7. [Outras seções] (PARAR)

📋 FORMATO JSON:
{
  "news_paragraphs": [
    {
      "headline": "manchete concisa em português (5-10 palavras)",
      "content": "conteúdo em português (detalhes COMPLEMENTARES, não repita a manchete)"
    },
    ...
  ],
  "deep_dive": {
    "content": "conteúdo COMPLETO da seção Deep Dive em português"
  },
  "big_take": {
    "content": "conteúdo COMPLETO da seção The Big Take em português"
  },
  "opinion": {
    "content": "conteúdo COMPLETO da seção Opinion em português"
  }
}

⚠️ CRÍTICO - ESTRUTURA:
- news_paragraphs: headlines em NEGRITO no original
- Headline = manchete concisa (5-10 palavras)
- Content = detalhes COMPLEMENTARES (NÃO repita palavras da manchete!)
- Se headline é "Tesla anuncia lucros", content deve ser "A receita cresceu 40%...", NÃO "Tesla anunciou lucros..."
- deep_dive, big_take, opinion = conteúdo COMPLETO de cada seção
- NÃO misture seções

⚠️ IMPORTANTE:
- TRADUZA tudo para português brasileiro
- Maximize uso do limite com informação NOVA, não redundante
- Deep Dive, Big Take, Opinion = conteúdo COMPLETO de cada seção
- CAPTURE todas as 4 estruturas

Retorne APENAS JSON:"""
    
    try:
        messages = [{"role": "user", "content": f"{extraction_prompt}\n\nCONTEÚDO:\n{cleaned_content}"}]
        
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.05,
            max_tokens=8192,
            top_p=0.9
        )
        
        response_text = response.choices[0].message.content.strip()
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        import re
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        
        sections = json.loads(response_text)
        
        # News: 150 chars with non-redundant summaries
        if sections.get('news_paragraphs'):
            for item in sections['news_paragraphs']:
                headline = item.get('headline', '')
                full_content = item.get('content', '')
                if full_content:
                    # Pass headline to avoid redundancy in summary
                    item['summary'] = create_smart_summary(groq_client, full_content, max_chars=250, title=headline)
        
        # Deep Dive, Big Take, Opinion: Full content (no summarization needed)
        # They will be displayed in full
        
        news_count = len(sections.get('news_paragraphs', []))
        has_deep_dive = 'deep_dive' in sections and sections['deep_dive']
        has_big_take = 'big_take' in sections and sections['big_take']
        has_opinion = 'opinion' in sections and sections['opinion']
        
        if news_count < 5:
            st.warning(f"⚠️ Apenas {news_count} notícias encontradas. Esperado: 8-12.")
        
        sections_found = []
        if has_deep_dive:
            sections_found.append("Deep Dive")
        if has_big_take:
            sections_found.append("Big Take")
        if has_opinion:
            sections_found.append("Opinion")
        
        if not has_deep_dive:
            st.warning("⚠️ Seção Deep Dive não encontrada!")
        if not has_big_take:
            st.warning("⚠️ Seção The Big Take não encontrada!")
        if not has_opinion:
            st.warning("⚠️ Seção Opinion não encontrada!")
        
        return sections
        
    except Exception as e:
        st.error(f"Erro na extração: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


def display_investnews(sections):
    """Display InvestNews content with colored cards per section"""
    
    # Define colors for each section
    colors = {
        'main_story': '#2C4F7C',           # Dark blue
        'highlights': "#376299",           # Slightly lighter blue
        'segunda_materia': '#3A6B8F',      # Medium blue
        'vale_parar_para_ler': '#4A7FA3'   # Light blue
    }
    
    # Main Story
    if sections.get('main_story') and sections['main_story']:
        st.markdown("### 📰 MANCHETE PRINCIPAL")
        content = sections['main_story'].get('content', 'N/A')
        url = sections['main_story'].get('url')
        
        link_html = ""
        if url:
            link_html = f"<a href='{url}' target='_blank' style='color: #d4af37; text-decoration: none; font-weight: bold; font-size: 1.2rem;'>[+]</a>"
        
        st.markdown(f"""
        <div style='background-color: {colors['main_story']}; color: white; padding: 2rem; 
                    border-radius: 12px; margin-bottom: 1.5rem; border-left: 5px solid #d4af37;'>
            <p style='font-size: 1rem; line-height: 1.7; white-space: pre-wrap; margin: 0;'>{content} {link_html}</p>
        </div>
        """, unsafe_allow_html=True)

    # Highlights - formatted cards (3 columns)
    if sections.get('highlights') and len(sections['highlights']) > 0:
        st.markdown("### ⚡ DESTAQUES")
        cols = st.columns(3)

        for i, highlight in enumerate(sections['highlights'][:3]):
            with cols[i]:
                title = highlight.get('title', 'N/A')
                content = highlight.get('content', 'N/A')
                url = highlight.get('url')
                
                link_html = ""
                if url:
                    link_html = f"<a href='{url}' target='_blank' style='color: #d4af37; text-decoration: none; font-weight: bold; font-size: 1.2rem;'>[+]</a>"

                st.markdown(f"""
                <div style='background-color: {colors['highlights']}; color: white; padding: 1.5rem; 
                            border-radius: 12px; margin-bottom: 1.5rem; border-left: 5px solid #d4af37; height: 100%;'>
                    <strong style='font-size: 1.1rem; display: block; margin-bottom: 1rem;'>{title}</strong>
                    <p style='font-size: 0.95rem; line-height: 1.6; margin: 0;'>{content} {link_html}</p>
                </div>
                """, unsafe_allow_html=True)

    # Segunda Matéria
    if sections.get('segunda_materia') and sections['segunda_materia']:
        st.markdown("### 📋 SEGUNDA MATÉRIA")
        content = sections['segunda_materia'].get('content', 'N/A')
        url = sections['segunda_materia'].get('url')
        
        link_html = ""
        if url:
            link_html = f"<a href='{url}' target='_blank' style='color: #d4af37; text-decoration: none; font-weight: bold; font-size: 1.2rem;'>[+]</a>"

        st.markdown(f"""
        <div style='background-color: {colors['segunda_materia']}; color: white; padding: 2rem; 
                    border-radius: 12px; margin-bottom: 1.5rem; border-left: 5px solid #d4af37;'>
            <p style='font-size: 1rem; line-height: 1.7; white-space: pre-wrap; margin: 0;'>{content} {link_html}</p>
            
        </div>
        """, unsafe_allow_html=True)

    # Vale Parar Para Ler - 2 columns
    if sections.get('vale_parar_para_ler') and len(sections['vale_parar_para_ler']) > 0:
        st.markdown("### 📚 VALE PARAR PARA LER")
        cols = st.columns(2)
        
        for i, item in enumerate(sections['vale_parar_para_ler'][:2]):
            with cols[i]:
                title = item.get('title', 'N/A')
                content = item.get('content', 'N/A')
                url = item.get('url')
                
                link_html = ""
                if url:
                    link_html = f"<a href='{url}' target='_blank' style='color: #d4af37; text-decoration: none; font-weight: bold; font-size: 1.2rem;'>[+]</a>"

                st.markdown(f"""
                <div style='background-color: {colors['vale_parar_para_ler']}; color: white; padding: 1.5rem; 
                            border-radius: 12px; margin-bottom: 1rem; border-left: 5px solid #d4af37; height: 100%;'>
                    <strong style='font-size: 1.1rem; display: block; margin-bottom: 1rem;'>{title}</strong>
                    <p style='font-size: 0.95rem; line-height: 1.6; margin: 0;'>{content} {link_html}</p>
                    
                </div>
                """, unsafe_allow_html=True)



def display_exame(sections):
    """Display Exame content with colored cards - 2 blocks per row"""
    
    # Color for Exame headlines
    card_color = '#2D5A3B'  # Dark green
    
    if sections.get('headlines'):
        headlines = sections['headlines']
        
        # Display in rows of 2
        for i in range(0, len(headlines), 2):
            cols = st.columns(2)
            
            # First column
            with cols[0]:
                if i < len(headlines):
                    headline = headlines[i]
                    title = headline.get('title', 'N/A')
                    content_text = headline.get('content', 'N/A')
                    url = headline.get('url', '#')
                    
                    # Create link HTML
                    link_html = ""
                    if url and url != '#':
                        link_html = f"<a href='{url}' target='_blank' style='color: #d4af37; text-decoration: none; font-weight: bold; font-size: 1.2rem;'>[+]</a>"
                    
                    st.markdown(f"""
                    <div style='background-color: {card_color}; color: white; padding: 1.5rem; 
                                border-radius: 12px; margin-bottom: 1rem; border-left: 5px solid #d4af37; height: 100%; min-height: 200px;'>
                        <strong style='font-size: 1.1rem; color: #d4af37; display: block; margin-bottom: 1rem;'>{title}</strong>
                        <p style='font-size: 0.95rem; line-height: 1.6; margin: 0;'>{content_text} {link_html}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Second column
            with cols[1]:
                if i + 1 < len(headlines):
                    headline = headlines[i + 1]
                    title = headline.get('title', 'N/A')
                    content_text = headline.get('content', 'N/A')
                    url = headline.get('url', '#')
                    
                    # Create link HTML
                    link_html = ""
                    if url and url != '#':
                        link_html = f"<a href='{url}' target='_blank' style='color: #d4af37; text-decoration: none; font-weight: bold; font-size: 1.2rem;'>[+]</a>"
                    
                    st.markdown(f"""
                    <div style='background-color: {card_color}; color: white; padding: 1.5rem; 
                                border-radius: 12px; margin-bottom: 1rem; border-left: 5px solid #d4af37; height: 100%; min-height: 200px;'>
                        <strong style='font-size: 1.1rem; color: #d4af37; display: block; margin-bottom: 1rem;'>{title}</strong>
                        <p style='font-size: 0.95rem; line-height: 1.6; margin: 0;'>{content_text} {link_html}</p>
                    </div>
                    """, unsafe_allow_html=True)



def display_bloomberg(sections):
    """Display Bloomberg content with colored solid cards per section"""
    
    # Define solid colors for each Bloomberg section
    colors = {
        'deep_dive': '#4A5D8C',     # Purple-blue (solid)
        'big_take': '#8C4A7A',      # Pink-purple (solid)
        'opinion': '#4A8C8C'        # Cyan-teal (solid)
    }
    
    # Deep Dive section - Full content
    if sections.get('deep_dive') and sections['deep_dive']:
        st.markdown("### 🔍 DEEP DIVE")
        
        deep_dive = sections['deep_dive']
        content_text = deep_dive.get('content', 'N/A')
        
        st.markdown(f"""
        <div style='background-color: {colors['deep_dive']}; color: white; padding: 2rem; 
                    border-radius: 12px; margin-bottom: 1.5rem; border-left: 5px solid #d4af37;'>
            <p style='font-size: 0.95rem; line-height: 1.6; white-space: pre-wrap; margin: 0;'>{content_text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # The Big Take section - Full content
    if sections.get('big_take') and sections['big_take']:
        st.markdown("### 📊 THE BIG TAKE")
        
        big_take = sections['big_take']
        content_text = big_take.get('content', 'N/A')
        
        st.markdown(f"""
        <div style='background-color: {colors['big_take']}; color: white; padding: 2rem; 
                    border-radius: 12px; margin-bottom: 1.5rem; border-left: 5px solid #d4af37;'>
            <p style='font-size: 0.95rem; line-height: 1.6; white-space: pre-wrap; margin: 0;'>{content_text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Opinion section - Full content
    if sections.get('opinion') and sections['opinion']:
        st.markdown("### 💭 OPINION")
        
        opinion = sections['opinion']
        content_text = opinion.get('content', 'N/A')
        
        st.markdown(f"""
        <div style='background-color: {colors['opinion']}; color: white; padding: 2rem; 
                    border-radius: 12px; margin-bottom: 1.5rem; border-left: 5px solid #d4af37;'>
            <p style='font-size: 0.95rem; line-height: 1.6; white-space: pre-wrap; margin: 0;'>{content_text}</p>
        </div>
        """, unsafe_allow_html=True)





def extract_bloomberg_evening(groq_client, html_content):
    """
    Extract Bloomberg Evening Briefing content.
    Structure:
    1. Main news: Content after first image until "What You Need to Know Today"
    2. News items: Between <hr> tags, starting with bold text
       - Ignore items with "Read the Story" or "See More"
       - Stop at "What You'll Need to Know Tomorrow"
    All content translated to Portuguese.
    """
    from bs4 import BeautifulSoup
    import re
    import json
    
    try:
        # Limit content size
        if len(html_content) > 50000:
            html_content = html_content[:50000]
        
        # Clean HTML
        cleaned_content = html_content.replace('\r\n', '\n').replace('\t', ' ')
        
        prompt = f"""You are extracting content from a Bloomberg Evening Briefing newsletter.

CRITICAL INSTRUCTIONS:
1. Extract content in ENGLISH first
2. Translate EVERYTHING to Portuguese (Brasil)
3. Keep summaries CLOSE TO 400 characters (minimum 350, maximum 400)
4. Preserve key details, numbers, names, and context

TASK: Extract the following sections:

1. MAIN NEWS: Find the paragraphs AFTER the Bloomberg logo/image and BEFORE the section "What You Need to Know Today". This is typically 2-3 paragraphs about the main story. Extract the full content and translate to Portuguese, keeping it between 350-400 characters.

2. NEWS ITEMS: After "What You Need to Know Today", there are several news items separated by <hr> tags. Each item has:
   - A bold headline (in <strong> or <b> tags)
   - Content paragraphs following the headline
   
   For each item:
   - Translate headline to Portuguese
   - Translate content to Portuguese, keeping between 350-400 characters
   - SKIP items that contain "Read the Story" or "See More" buttons
   - STOP when you reach "What You'll Need to Know Tomorrow"

Return this EXACT JSON structure (no backticks, no markdown):
{{
  "main_news": "notícia principal traduzida para português (350-400 chars)",
  "news_items": [
    {{"headline": "manchete traduzida", "content": "conteúdo traduzido (350-400 chars)"}},
    {{"headline": "manchete traduzida", "content": "conteúdo traduzido (350-400 chars)"}}
  ]
}}

HTML:
{cleaned_content}
"""
        
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=8192
        )
        
        result = response.choices[0].message.content.strip()
        
        if not result:
            st.error("AI retornou resposta vazia")
            return {}
        
        # Clean response - remove markdown code blocks
        result = result.replace('```json', '').replace('```', '').strip()
        
        # Try to find JSON in response if it has extra text
        json_match = re.search(r'\{[\s\S]*\}', result)
        if json_match:
            result = json_match.group(0)
        
        # Parse JSON
        sections = json.loads(result)
        
        # Validate structure
        if not isinstance(sections, dict):
            st.error("Resposta não é um dicionário válido")
            return {}
        
        if not sections.get('main_news'):
            sections['main_news'] = ''
        
        if not sections.get('news_items'):
            sections['news_items'] = []
        
        # Ensure content is close to 400 chars (only trim if way over)
        for item in sections.get('news_items', []):
            if 'content' in item and len(item['content']) > 420:
                item['content'] = item['content'][:397] + '...'
        
        # Ensure main news is close to 400 chars (only trim if way over)
        if len(sections['main_news']) > 420:
            sections['main_news'] = sections['main_news'][:397] + '...'
        
        return sections
        
    except json.JSONDecodeError as e:
        st.error(f"Erro ao decodificar JSON: {str(e)}")
        st.error(f"Resposta da IA (primeiros 500 chars): {result[:500]}...")
        return {}
    except Exception as e:
        st.error(f"Erro na extração Bloomberg Evening: {str(e)}")
        return {}


def display_bloomberg_evening(sections):
    """Display Bloomberg Evening content with dark blue/evening theme"""
    
    if not sections:
        st.warning("⚠️ Nenhum conteúdo extraído")
        return
    
    # Main news
    main_news = sections.get('main_news', '')
    if main_news:
        st.markdown(f"""
        <div style="background-color: #1A2332; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <p style="color: white; font-size: 16px; line-height: 1.6; margin: 0;">
                {main_news}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # News items
    news_items = sections.get('news_items', [])
    if news_items:
        
        # Display in 2 columns
        for i in range(0, len(news_items), 2):
            col1, col2 = st.columns(2)
            
            # First item in row
            with col1:
                if i < len(news_items):
                    item = news_items[i]
                    headline = item.get('headline', '')
                    content = item.get('content', '')
                    
                    st.markdown(f"""
                    <div style="background-color: #2C3E5D; padding: 15px; border-radius: 8px; margin-bottom: 15px; min-height: 150px;">
                        <p style="color: white; font-weight: bold; margin-bottom: 10px; font-size: 15px;">
                            {headline}
                        </p>
                        <p style="color: #E0E0E0; font-size: 14px; line-height: 1.5; margin: 0;">
                            {content}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Second item in row
            with col2:
                if i + 1 < len(news_items):
                    item = news_items[i + 1]
                    headline = item.get('headline', '')
                    content = item.get('content', '')
                    
                    st.markdown(f"""
                    <div style="background-color: #2C3E5D; padding: 15px; border-radius: 8px; margin-bottom: 15px; min-height: 150px;">
                        <p style="color: white; font-weight: bold; margin-bottom: 10px; font-size: 15px;">
                            {headline}
                        </p>
                        <p style="color: #E0E0E0; font-size: 14px; line-height: 1.5; margin: 0;">
                            {content}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No news items found")

def save_to_database(supabase_client, newsletter_type, parsed_email, sections):
    """Save processed newsletter to Supabase"""
    try:
        # Create unique ID
        content_hash = hashlib.md5(
            f"{parsed_email['subject']}{parsed_email['date']}".encode()
        ).hexdigest()
        
        # Prepare data - use email_date instead of date
        data = {
            'id': content_hash,
            'newsletter_type': newsletter_type,
            'subject': parsed_email['subject'],
            'sender': parsed_email['sender'],
            'email_date': parsed_email['date'],  # Changed from 'date' to 'email_date'
            'sections': sections,
            'processed_at': datetime.now().isoformat()
        }
        
        # Insert or update
        result = supabase_client.table('newsletters').upsert(data).execute()
        
        return True
        
    except Exception as e:
        st.error(f"Error saving to database: {e}")
        return False




# ═══════════════════════════════════════════════════════════════════════════
# NOTICIÁRIO RENDA FIXA FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def parse_noticiario_renda_fixa(html_content):
    """
    Parse the Noticiário Renda Fixa HTML file and extract daily news sections.
    
    Returns:
        List of dictionaries, each containing:
        - date: The date string (e.g., "Quarta-feira, 07 de Janeiro")
        - sections: Dictionary with keys: 'leitura_da_curva', 'mercados_globais', 'mercado_domestico', 'noticiario_corporativo'
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        daily_reports = []
        
        # Find all h2 tags that contain the date pattern
        date_headers = soup.find_all('h2', class_='wp-block-heading')
        
        for date_header in date_headers:
            date_text = date_header.get_text(strip=True)
            
            # Check if this is a date header (contains day of week)
            days_of_week = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira']
            if not any(day in date_text for day in days_of_week):
                continue
            
            # Initialize the daily report structure
            daily_report = {
                'date': date_text,
                'leitura_da_curva_date': '',  # Will store the date from LEITURA DA CURVA
                'sections': {
                    'leitura_da_curva': {'title': 'LEITURA DA CURVA', 'content': ''},
                    'mercados_globais': {'title': 'MERCADOS GLOBAIS', 'items': []},
                    'mercado_domestico': {'title': 'MERCADO DOMÉSTICO', 'items': []},
                    'noticiario_corporativo': {'title': 'NOTICIÁRIO CORPORATIVO', 'subsections': {}}
                }
            }
            
            # Process elements after the date header
            current = date_header.find_next_sibling()
            current_section = None
            current_subsection = None
            
            while current:
                # Stop if we hit another date header or RELATÓRIOS DA SEMANA
                if current.name == 'h2':
                    h2_text = current.get_text(strip=True)
                    if any(day in h2_text for day in days_of_week):
                        break
                    if 'RELATÓRIOS DA SEMANA' in h2_text:
                        break
                
                if current.name == 'p':
                    text = current.get_text(strip=True)
                    
                    # Check for RELATÓRIOS DA SEMANA - end this day immediately
                    if 'RELATÓRIOS DA SEMANA' in text:
                        break
                    
                    # Check if this is LEITURA DA CURVA (has orange background)
                    if current.get('style') and '#fcb90078' in current.get('style'):
                        # Extract the date from LEITURA DA CURVA title
                        strong_tag = current.find('strong')
                        if strong_tag:
                            leitura_title = strong_tag.get_text(strip=True)
                            # Extract date part (after the dash)
                            if '–' in leitura_title:
                                daily_report['leitura_da_curva_date'] = leitura_title.split('–')[1].strip()
                            
                            # Remove the strong tag content from the text to get only the body
                            text_without_title = text.replace(leitura_title, '').strip()
                            daily_report['sections']['leitura_da_curva']['content'] = text_without_title
                        else:
                            daily_report['sections']['leitura_da_curva']['content'] = text
                        
                        current_section = 'leitura_da_curva'
                        current_subsection = None
                    
                    # Check if this paragraph contains section headers
                    elif current.find('strong'):
                        # Get all text from all strong tags combined
                        strong_tags = current.find_all('strong')
                        combined_strong_text = ' '.join([tag.get_text(strip=True) for tag in strong_tags])
                        
                        # Check for section headers
                        if 'MERCADOS GLOBAIS' in combined_strong_text:
                            current_section = 'mercados_globais'
                            current_subsection = None
                        elif 'MERCADO' in combined_strong_text and 'DOMÉSTICO' in combined_strong_text:
                            # Handle "MERCADO DOMÉSTICO" which might be split across multiple <strong> tags
                            current_section = 'mercado_domestico'
                            current_subsection = None
                        elif 'NOTICIÁRIO CORPORATIVO' in combined_strong_text:
                            current_section = 'noticiario_corporativo'
                            current_subsection = None
                        # Check for subsections in NOTICIÁRIO CORPORATIVO (e.g., "| Agronegócio")
                        elif current_section == 'noticiario_corporativo' and combined_strong_text.startswith('|'):
                            # Extract subsection name (remove the "|" and trim)
                            current_subsection = combined_strong_text[1:].strip()
                            if current_subsection not in daily_report['sections']['noticiario_corporativo']['subsections']:
                                daily_report['sections']['noticiario_corporativo']['subsections'][current_subsection] = []
                    
                    # Collect news items for the current section
                    else:
                        # Check if this paragraph has a link (news item)
                        link = current.find('a')
                        if link and current_section:
                            news_item = {
                                'title': link.get_text(strip=True),
                                'url': link.get('href', ''),
                                'source': '',
                                'description': ''
                            }
                            
                            # Get the full paragraph text
                            full_text = current.get_text()
                            
                            # Extract and remove the link text
                            link_text = link.get_text()
                            description = full_text.replace(link_text, '', 1).strip()
                            
                            # Extract source (usually in <em> tags or plain text in parentheses)
                            em_tag = current.find('em')
                            source_text = ''
                            
                            if em_tag:
                                # Source is in <em> tag
                                source_text = em_tag.get_text(strip=True).strip('().')
                                news_item['source'] = f"({source_text})"
                                # Remove the em tag text from description
                                em_text = em_tag.get_text()
                                description = description.replace(em_text, '', 1).strip()
                            else:
                                # Check if source is in plain text format like "(Source)." at the beginning
                                # Look for pattern: (Something). at the start
                                import re
                                source_pattern = r'^\(([^)]+)\)\.?\s*'
                                match = re.match(source_pattern, description)
                                if match:
                                    source_text = match.group(1).strip()
                                    # Only use if not empty
                                    if source_text:
                                        news_item['source'] = f"({source_text})"
                                    # Remove the source from description
                                    description = re.sub(source_pattern, '', description).strip()
                            
                            # Clean up leading unwanted characters and punctuation
                            description = description.lstrip('().,:; ')
                            
                            # Handle special case: "apurou o [Source]" at the end
                            # This pattern means the source name appears at the end
                            if 'apurou o ' in description.lower():
                                # Check if there's a source name after "apurou o"
                                parts = description.rsplit('apurou o ', 1)
                                if len(parts) == 2:
                                    # Extract what comes after "apurou o"
                                    after_apurou = parts[1].strip().rstrip('.')
                                    
                                    # If we don't have a source yet, use this as the source
                                    if not source_text and after_apurou:
                                        news_item['source'] = f"({after_apurou})"
                                    
                                    # Keep the full description including "apurou o [Source]"
                            
                            # Clean up extra spaces and normalize
                            description = ' '.join(description.split())
                            
                            # Remove any trailing periods before adding our own
                            description = description.rstrip('.')
                            
                            # Add ending period if there's content
                            if description:
                                description = description + '.'
                            
                            news_item['description'] = description
                            
                            # Add to appropriate section
                            if current_section == 'noticiario_corporativo' and current_subsection:
                                daily_report['sections']['noticiario_corporativo']['subsections'][current_subsection].append(news_item)
                            elif current_section in ['mercados_globais', 'mercado_domestico']:
                                daily_report['sections'][current_section]['items'].append(news_item)
                
                current = current.find_next_sibling()
            
            # Only add the report if it has content
            if daily_report['sections']['leitura_da_curva']['content'] or \
               daily_report['sections']['mercados_globais']['items'] or \
               daily_report['sections']['mercado_domestico']['items'] or \
               daily_report['sections']['noticiario_corporativo']['subsections']:
                daily_reports.append(daily_report)
        
        return daily_reports
        
    except Exception as e:
        st.error(f"Erro ao processar arquivo HTML: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return []


def display_noticiario_renda_fixa(daily_reports):
    """
    Display the parsed Noticiário Renda Fixa in a beautiful, collapsible format.
    
    Args:
        daily_reports: List of daily report dictionaries from parse_noticiario_renda_fixa
    """
    if not daily_reports:
        st.warning("Nenhum conteúdo encontrado no arquivo.")
        return
    
    # Custom CSS for the news display
    st.markdown("""
        <style>
        .news-date-header {
            font-family: 'Playfair Display', serif;
            font-size: 28px;
            font-weight: 700;
            color: #d4af37;
            margin-bottom: 20px;
            padding: 15px;
            background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
            border-left: 5px solid #d4af37;
            border-radius: 5px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        .section-box {
            background-color: #1a1a1a;
            border: 1px solid #d4af37;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .section-title {
            font-family: 'Montserrat', sans-serif;
            font-size: 20px;
            font-weight: 600;
            color: #f4d03f;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #d4af37;
        }
        
        .leitura-curva-box {
            background: linear-gradient(135deg, #2a1a0a 0%, #1a1a1a 100%);
            border: 2px solid #fcb900;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .leitura-curva-content {
            font-family: 'Montserrat', sans-serif;
            font-size: 15px;
            line-height: 1.8;
            color: #e0e0e0;
            text-align: justify;
        }
        
        .news-item {
            background-color: #0f0f0f;
            border-left: 3px solid #d4af37;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
            transition: all 0.3s ease;
        }
        
        .news-item:hover {
            background-color: #1a1a1a;
            border-left-color: #f4d03f;
            transform: translateX(5px);
        }
        
        .news-title {
            font-family: 'Montserrat', sans-serif;
            font-size: 16px;
            font-weight: 600;
            color: #d4af37;
            margin-bottom: 8px;
        }
        
        .news-title a {
            color: #d4af37;
            text-decoration: none;
        }
        
        .news-title a:hover {
            color: #f4d03f;
            text-decoration: underline;
        }
        
        .news-source {
            font-family: 'Montserrat', sans-serif;
            font-size: 13px;
            font-style: italic;
            color: #888;
            margin-bottom: 8px;
        }
        
        .news-description {
            font-family: 'Montserrat', sans-serif;
            font-size: 14px;
            line-height: 1.6;
            color: #b0b0b0;
            text-align: justify;
        }
        
        .subsection-header {
            font-family: 'Montserrat', sans-serif;
            font-size: 18px;
            font-weight: 700;
            color: #0a0a0a;
            background: linear-gradient(135deg, #d4af37 0%, #f4d03f 100%);
            margin-top: 25px;
            margin-bottom: 15px;
            padding: 12px 20px;
            border-radius: 8px;
            border-left: 5px solid #0a0a0a;
            box-shadow: 0 2px 8px rgba(212, 175, 55, 0.3);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Display each day in a collapsible expander
    for report in daily_reports:
        with st.expander(f"📅 {report['date']}", expanded=False):
            
            # LEITURA DA CURVA section
            if report['sections']['leitura_da_curva']['content']:
                # Build title with date if available
                leitura_title = report['sections']['leitura_da_curva']['title']
                if report.get('leitura_da_curva_date'):
                    leitura_title = f"{leitura_title} – {report['leitura_da_curva_date']}"
                
                st.markdown(f"""
                    <div class="leitura-curva-box">
                        <div class="section-title">📊 {leitura_title}</div>
                        <div class="leitura-curva-content">{report['sections']['leitura_da_curva']['content']}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # MERCADOS GLOBAIS section
            if report['sections']['mercados_globais']['items']:
                st.markdown(f"""
                    <div class="section-box">
                        <div class="section-title">🌍 {report['sections']['mercados_globais']['title']}</div>
                """, unsafe_allow_html=True)
                
                for item in report['sections']['mercados_globais']['items']:
                    st.markdown(f"""
                        <div class="news-item">
                            <div class="news-title"><a href="{item['url']}" target="_blank">{item['title']}</a></div>
                            <div class="news-source">{item['source']}</div>
                            <div class="news-description">{item['description']}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # MERCADO DOMÉSTICO section
            if report['sections']['mercado_domestico']['items']:
                st.markdown(f"""
                    <div class="section-box">
                        <div class="section-title">📍 {report['sections']['mercado_domestico']['title']}</div>
                """, unsafe_allow_html=True)
                
                for item in report['sections']['mercado_domestico']['items']:
                    st.markdown(f"""
                        <div class="news-item">
                            <div class="news-title"><a href="{item['url']}" target="_blank">{item['title']}</a></div>
                            <div class="news-source">{item['source']}</div>
                            <div class="news-description">{item['description']}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # NOTICIÁRIO CORPORATIVO section
            if report['sections']['noticiario_corporativo']['subsections']:
                st.markdown(f"""
                    <div class="section-box">
                        <div class="section-title">📰 {report['sections']['noticiario_corporativo']['title']}</div>
                """, unsafe_allow_html=True)
                
                # Display each subsection
                for subsection_name, items in report['sections']['noticiario_corporativo']['subsections'].items():
                    st.markdown(f"""
                        <div class="subsection-header">| {subsection_name}</div>
                    """, unsafe_allow_html=True)
                    
                    for item in items:
                        st.markdown(f"""
                            <div class="news-item">
                                <div class="news-title"><a href="{item['url']}" target="_blank">{item['title']}</a></div>
                                <div class="news-source">{item['source']}</div>
                                <div class="news-description">{item['description']}</div>
                            </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)


def save_noticiario_to_database(supabase_client, html_content, parsed_data, uploaded_by="admin"):
    """Save Noticiário Renda Fixa data to Supabase"""
    try:
        data = {
            'upload_date': datetime.now().isoformat(),
            'uploaded_by': uploaded_by,
            'html_content': html_content,
            'parsed_data': parsed_data
        }
        
        # Delete old records (keep only latest)
        supabase_client.table('noticiario_renda_fixa').delete().neq('id', 0).execute()
        
        # Insert new record
        result = supabase_client.table('noticiario_renda_fixa').insert(data).execute()
        
        return True
    except Exception as e:
        st.error(f"Erro ao salvar noticiário no banco: {e}")
        return False

def load_noticiario_from_database(supabase_client):
    """Load latest Noticiário Renda Fixa data from Supabase"""
    try:
        result = supabase_client.table('noticiario_renda_fixa').select('*').order('upload_date', desc=True).limit(1).execute()
        
        if result.data and len(result.data) > 0:
            return result.data[0]
        return None
    except Exception as e:
        st.error(f"Erro ao carregar noticiário do banco: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# CURVAS ANBIMA FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


# CURVAS ANBIMA FUNCTIONS (from news_v5.py)
def check_admin_access():
    """Check if user has admin access based on logged in user"""
    return st.session_state.get('user_logged_in') == 'admin' 

def save_curves_to_database(supabase_client, file_data, uploaded_by="admin"):
    """Save curves data to Supabase"""
    try:
        data = {
            'upload_date': datetime.now().isoformat(),
            'uploaded_by': uploaded_by,
            'file_data': file_data
        }
        
        # Delete old records (keep only latest)
        supabase_client.table('anbima_curves').delete().neq('id', 0).execute()
        
        # Insert new record
        result = supabase_client.table('anbima_curves').insert(data).execute()
        
        return True
    except Exception as e:
        st.error(f"Erro ao salvar no banco: {e}")
        return False

def load_curves_from_database(supabase_client):
    """Load latest curves data from Supabase"""
    try:
        result = supabase_client.table('anbima_curves').select('*').order('upload_date', desc=True).limit(1).execute()
        
        if result.data and len(result.data) > 0:
            return result.data[0]['file_data']
        return None
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

def process_excel_to_json(excel_file):
    """Convert Excel file to JSON format for storage"""
    import pandas as pd
    
    try:
        xl_file = pd.ExcelFile(excel_file)
        
        data = {}
        for sheet_name in xl_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            # Store column order explicitly to preserve it
            # This ensures Vértice (Anos) stays as first column
            data[sheet_name] = {
                'columns': df.columns.tolist(),  # Preserve column order
                'data': df.to_dict('list')
            }
        
        return data
    except Exception as e:
        st.error(f"Erro ao processar Excel: {e}")
        return None

def json_to_dataframes(json_data):
    """Convert JSON data back to DataFrames"""
    import pandas as pd
    
    try:
        dataframes = {}
        for sheet_name, sheet_data in json_data.items():
            # Handle both old format (direct dict) and new format (with columns key)
            if isinstance(sheet_data, dict) and 'columns' in sheet_data and 'data' in sheet_data:
                # New format: explicitly preserved column order
                df = pd.DataFrame(sheet_data['data'], columns=sheet_data['columns'])
            else:
                # Old format: backward compatibility
                df = pd.DataFrame(sheet_data)
            
            dataframes[sheet_name] = df
        return dataframes
    except Exception as e:
        st.error(f"Erro ao converter dados: {e}")
        return None

def create_curves_table(df, index_name):
    """Create table with rates for each vertex and date"""
    if df is None or df.empty:
        return None
    
    # Set the first column as index
    df_copy = df.copy()
    df_copy.set_index(df_copy.columns[0], inplace=True)
    
    return df_copy

def create_curves_chart(df, index_name):
    """Create chart for curves"""
    if df is None or df.empty:
        return None
    
    fig = go.Figure()
    
    # First column is vertices (anos), rest are dates
    vertices = df.iloc[:, 0].values
    date_columns = df.columns[1:]
    
    # Generate blue gradient (light to dark)
    colors = generate_blue_gradient(len(date_columns))
    
    for i, date_col in enumerate(date_columns):
        rates = df[date_col].values
        
        fig.add_trace(go.Scatter(
            x=vertices,
            y=rates,
            mode='lines+markers',
            line=dict(shape='spline', width=2.5, color=colors[i]),
            name=str(date_col)
        ))
    
    fig.update_layout(
        title=index_name,
        xaxis_title='Vértice (Anos)',
        yaxis_title='Taxa (%)',
        plot_bgcolor='#0a0a0a',
        paper_bgcolor='#1a1a1a',
        font=dict(color='#d4af37', family='Montserrat'),
        xaxis=dict(showgrid=True, gridcolor='#333'),
        yaxis=dict(showgrid=True, gridcolor='#333'),
        legend=dict(bgcolor='#1a1a1a', bordercolor='#d4af37', borderwidth=1),
        height=500
    )
    
    return fig

def calculate_curves_variations(df):
    """Calculate rate variations between dates"""
    if df is None or df.empty or len(df.columns) < 3:
        return {}
    
    vertices = df.iloc[:, 0].values
    date_columns = df.columns[1:]
    
    # Latest is last column, compare with previous dates
    latest_rates = df[date_columns[-1]].values
    
    variations = {}
    
    for days_back in range(1, min(6, len(date_columns))):
        past_col = date_columns[-(days_back + 1)]
        past_rates = df[past_col].values
        
        # Calculate variations
        var_list = []
        for i, vertex in enumerate(vertices):
            if pd.notna(latest_rates[i]) and pd.notna(past_rates[i]):
                variation = latest_rates[i] - past_rates[i]
                var_list.append((vertex, variation))
        
        # Sort by absolute value and take top 10
        var_list.sort(key=lambda x: abs(x[1]), reverse=True)
        variations[days_back] = var_list[:10]
    
    return variations
# ═══════════════════════════════════════════════════════════════════════════════
# CREDIT CURVES FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def save_credit_curves_to_database(supabase_client, file_data, uploaded_by="admin"):
    """Save credit curves to database"""
    if not supabase_client:
        return False
    
    try:
        # Delete old data
        supabase_client.table('credito_curves').delete().neq('id', 0).execute()
        
        # Insert new data
        data = {'file_data': file_data, 'uploaded_by': uploaded_by}
        result = supabase_client.table('credito_curves').insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Erro ao salvar: {e}")
        return False

def load_credit_curves_from_database(supabase_client):
    """Load latest credit curves from database"""
    if not supabase_client:
        return None
    
    try:
        result = supabase_client.table('credito_curves').select('*').order('upload_date', desc=True).limit(1).execute()
        
        if result.data and len(result.data) > 0:
            return result.data[0]['file_data']
        return None
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

def process_credit_excel_to_json(excel_file):
    """Convert credit Excel file to JSON format, removing rows with NaN"""
    import pandas as pd
    
    try:
        xl_file = pd.ExcelFile(excel_file)
        
        data = {}
        for sheet_name in xl_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            # Remove rows where ANY date column has NaN
            # Keep Vértice column, check only date columns (1 onwards)
            date_columns = df.columns[1:]
            df_clean = df.dropna(subset=date_columns, how='any')
            
            # Store with explicit column order
            data[sheet_name] = {
                'columns': df_clean.columns.tolist(),
                'data': df_clean.to_dict('list')
            }
        
        return data
    except Exception as e:
        st.error(f"Erro ao processar Excel: {e}")
        return None

def create_credit_comparison_chart(dataframes, selected_date):
    """Create comparison chart for all ratings on a selected date"""
    if not dataframes:
        return None
    
    fig = go.Figure()
    
    # Colors for ratings
    colors = {'AAA': '#00e100', 'AA': '#FFD700', 'A': '#f20000'}
    
    for rating, df in dataframes.items():
        if df is None or df.empty:
            continue
        
        vertices = df.iloc[:, 0].values
        
        # Find the selected date column
        if selected_date in df.columns:
            rates = df[selected_date].values
            
            fig.add_trace(go.Scatter(
                x=vertices,
                y=rates,
                mode='lines+markers',
                line=dict(shape='spline', width=3, color=colors.get(rating, '#d4af37')),
                marker=dict(size=6),
                name=rating
            ))
    
    fig.update_layout(
        title=f'Comparativo de Spreads - {selected_date}',
        xaxis_title='Vértice (Anos)',
        yaxis_title='Spread (%)',
        plot_bgcolor='#0a0a0a',
        paper_bgcolor='#1a1a1a',
        font=dict(color='#d4af37', family='Montserrat'),
        xaxis=dict(showgrid=True, gridcolor='#333'),
        yaxis=dict(showgrid=True, gridcolor='#333'),
        legend=dict(bgcolor='#1a1a1a', bordercolor='#d4af37', borderwidth=1),
        height=500
    )
    
    return fig

def calculate_spread_differences(dataframes, rating1, rating2, date):
    """Calculate average spread difference between two ratings for a specific date
    Returns: average of (lower_rated - higher_rated) across all vertices
    """
    df1 = dataframes.get(rating1)
    df2 = dataframes.get(rating2)
    
    if df1 is None or df2 is None or date not in df1.columns or date not in df2.columns:
        return None
    
    # Get common vertices
    vertices1 = df1.iloc[:, 0].values
    vertices2 = df2.iloc[:, 0].values
    
    # Use intersection of vertices
    common_vertices = sorted(set(vertices1) & set(vertices2))
    
    differences = []
    for vertex in common_vertices:
        # Find rates for this vertex
        rate1_idx = df1[df1.iloc[:, 0] == vertex].index
        rate2_idx = df2[df2.iloc[:, 0] == vertex].index
        
        if len(rate1_idx) > 0 and len(rate2_idx) > 0:
            rate1 = df1.loc[rate1_idx[0], date]
            rate2 = df2.loc[rate2_idx[0], date]
            
            if pd.notna(rate1) and pd.notna(rate2):
                # Lower rated (higher rate) - Higher rated (lower rate)
                diff = rate1 - rate2
                differences.append(diff)
    
    # Return average difference
    if differences:
        return sum(differences) / len(differences)
    return None

def create_credit_curves_chart(df, rating_name):
    """Create chart for credit curves (one rating)"""
    if df is None or df.empty:
        return None
    
    fig = go.Figure()
    
    vertices = df.iloc[:, 0].values
    date_columns = df.columns[1:]
    
    # Generate blue gradient (light to dark)
    colors = generate_blue_gradient(len(date_columns))
    
    for i, date_col in enumerate(date_columns):
        rates = df[date_col].values
        
        fig.add_trace(go.Scatter(
            x=vertices,
            y=rates,
            mode='lines+markers',
            line=dict(shape='spline', width=2.5, color=colors[i]),
            name=str(date_col)
        ))
    
    fig.update_layout(
        title=f'Spread {rating_name}',
        xaxis_title='Vértice (Anos)',
        yaxis_title='Spread (%)',
        plot_bgcolor='#0a0a0a',
        paper_bgcolor='#1a1a1a',
        font=dict(color='#d4af37', family='Montserrat'),
        xaxis=dict(showgrid=True, gridcolor='#333'),
        yaxis=dict(showgrid=True, gridcolor='#333'),
        legend=dict(bgcolor='#1a1a1a', bordercolor='#d4af37', borderwidth=1),
        height=500
    )
    
    return fig

def calculate_credit_variations(df):
    """Calculate rate variations for credit curves (all vertices in order)"""
    if df is None or df.empty or len(df.columns) < 3:
        return {}
    
    vertices = df.iloc[:, 0].values
    date_columns = df.columns[1:]
    
    # Latest is last column
    latest_rates = df[date_columns[-1]].values
    
    variations = {}
    
    for days_back in range(1, min(6, len(date_columns))):
        past_col = date_columns[-(days_back + 1)]
        past_rates = df[past_col].values
        
        var_list = []
        for i, vertex in enumerate(vertices):
            if pd.notna(latest_rates[i]) and pd.notna(past_rates[i]):
                variation = latest_rates[i] - past_rates[i]
                var_list.append((vertex, variation))
        
        # Keep original order (no sorting)
        variations[days_back] = var_list
    
    return variations

def create_credit_table(df):
    """Create table for credit curves"""
    if df is None or df.empty:
        return None
    
    df_copy = df.copy()
    df_copy.set_index(df_copy.columns[0], inplace=True)
    
    return df_copy







# ═══════════════════════════════════════════════════════════════════════════════
# SUPABASE HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class SupabaseNewsletter:
    """Helper class for Supabase newsletter operations"""
    
    def __init__(self):
        self.url = SUPABASE_URL
        self.key = SUPABASE_KEY
        self.headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json"
        }
        self.table_name = "bloomberg_linea_newsletters"
    
    def save_newsletter(self, newsletter_type, content, source_name):
        """Save or update newsletter in Supabase"""
        if not self.url or not self.key:
            return False
        
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            
            data = {
                "newsletter_type": newsletter_type,
                "content": content,
                "source_name": source_name,
                "upload_date": today,
                "created_at": datetime.now().isoformat()
            }
            
            # First, try to delete existing newsletter of same type for today
            delete_url = f"{self.url}/rest/v1/{self.table_name}?newsletter_type=eq.{newsletter_type}&upload_date=eq.{today}"
            requests.delete(delete_url, headers=self.headers)
            
            # Insert new newsletter
            insert_url = f"{self.url}/rest/v1/{self.table_name}"
            response = requests.post(insert_url, headers=self.headers, json=data)
            
            return response.status_code in [200, 201]
        except Exception as e:
            st.error(f"Erro ao salvar no Supabase: {str(e)}")
            return False
    
    def get_todays_newsletters(self):
        """Get all newsletters for today"""
        if not self.url or not self.key:
            return {}
        
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            url = f"{self.url}/rest/v1/{self.table_name}?upload_date=eq.{today}"
            
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                newsletters = response.json()
                result = {}
                for newsletter in newsletters:
                    result[newsletter['newsletter_type']] = {
                        'source': newsletter['source_name'],
                        'timestamp': newsletter['upload_date'],
                        'type': 'Newsletter Upload',
                        'content': newsletter['content'],
                        'status': 'success'
                    }
                return result
            else:
                return {}
        except Exception as e:
            return {}


# ═══════════════════════════════════════════════════════════════════════════════
# NEWSLETTER PARSERS - INVESTNEWS, EXAME, BLOOMBERG LÍNEA
# ═══════════════════════════════════════════════════════════════════════════════

class NewsletterParser:
    """Base parser for different newsletters"""
    
    @staticmethod
    def parse_eml_file(uploaded_file):
        """Parse .eml file and extract HTML content"""
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            msg = email.message_from_bytes(uploaded_file.read(), policy=policy.default)
            
            # Extract HTML content
            html_content = None
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == 'text/html':
                        html_content = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        break
            else:
                if msg.get_content_type() == 'text/html':
                    html_content = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            
            # Extract subject and date
            subject = msg.get('Subject', 'Unknown')
            date = msg.get('Date', 'Unknown')
            
            return {
                'subject': subject,
                'date': date,
                'html': html_content
            }
        except Exception as e:
            st.error(f"Erro ao analisar arquivo EML: {str(e)}")
            return None
    
    @staticmethod
    def extract_investnews(html_content):
        """Extract structured content from Investnews newsletter"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            sections = []
            
            # Section 1: First news from h1 until "Leia Mais"
            h1_tags = soup.find_all('h1')
            if h1_tags:
                first_section = {
                    'title': h1_tags[0].get_text(strip=True),
                    'content': []
                }
                
                current = h1_tags[0].find_next()
                while current:
                    text = current.get_text(strip=True) if current.name else ''
                    if 'Leia Mais' in text:
                        break
                    if current.name == 'p':
                        para_text = current.get_text(strip=True)
                        if para_text:
                            first_section['content'].append(para_text)
                    current = current.find_next()
                
                sections.append(first_section)
            
            # Section 2: HIGHLIGHTS (h4 + three highlights)
            highlights_h4 = soup.find('h4', string=lambda t: t and 'HIGHLIGHTS' in t.upper())
            if highlights_h4:
                highlights_section = {
                    'title': 'HIGHLIGHTS',
                    'content': []
                }
                
                current = highlights_h4.find_next()
                highlight_count = 0
                current_highlight = None
                
                while current and highlight_count < 3:
                    if current.name == 'h4':  # Stop if we hit another section
                        break
                    
                    if current.name == 'p':
                        text = current.get_text(strip=True)
                        if not text:
                            current = current.find_next()
                            continue
                        
                        # Check if this p contains a <b> tag (headline)
                        b_tag = current.find('b')
                        if b_tag:
                            # This is a new highlight headline
                            if current_highlight:
                                highlights_section['content'].append(current_highlight)
                                highlight_count += 1
                                if highlight_count >= 3:
                                    break
                            
                            current_highlight = {
                                'headline': b_tag.get_text(strip=True),
                                'text': text.replace(b_tag.get_text(strip=True), '').strip()
                            }
                        else:
                            # This is continuation of current highlight
                            if current_highlight:
                                current_highlight['text'] += ' ' + text
                    
                    current = current.find_next()
                
                # Add the last highlight
                if current_highlight:
                    highlights_section['content'].append(current_highlight)
                
                sections.append(highlights_section)
            
            # Section 4: Second news (next h1 until "Leia Mais")
            if len(h1_tags) > 1:
                second_section = {
                    'title': h1_tags[1].get_text(strip=True),
                    'content': []
                }
                
                current = h1_tags[1].find_next()
                while current:
                    text = current.get_text(strip=True) if current.name else ''
                    if 'Leia Mais' in text:
                        break
                    if current.name == 'p':
                        para_text = current.get_text(strip=True)
                        if para_text:
                            second_section['content'].append(para_text)
                    current = current.find_next()
                
                sections.append(second_section)
            
            # Section 8: VALE PARAR PARA LER (until "NOSSAS NEWSLETTERS")
            vale_h4 = soup.find('h4', string=lambda t: t and 'VALE PARAR PARA LER' in t.upper())
            if vale_h4:
                vale_section = {
                    'title': 'VALE PARAR PARA LER',
                    'content': []
                }
                
                current = vale_h4.find_next()
                current_item = None
                
                while current:
                    text = current.get_text(strip=True) if current.name else ''
                    if 'NOSSAS NEWSLETTERS' in text:
                        break
                    
                    if current.name == 'p':
                        para_text = current.get_text(strip=True)
                        if not para_text:
                            current = current.find_next()
                            continue
                        
                        # Check if this p contains a <b> tag (headline)
                        b_tag = current.find('b')
                        if b_tag:
                            # This is a new item headline
                            if current_item:
                                vale_section['content'].append(current_item)
                            
                            current_item = {
                                'headline': b_tag.get_text(strip=True),
                                'text': para_text.replace(b_tag.get_text(strip=True), '').strip()
                            }
                        else:
                            # This is continuation of current item
                            if current_item:
                                current_item['text'] += ' ' + para_text
                    
                    current = current.find_next()
                
                # Add the last item
                if current_item:
                    vale_section['content'].append(current_item)
                
                sections.append(vale_section)
            
            return sections
        
        except Exception as e:
            st.error(f"Erro ao extrair conteúdo da Investnews: {str(e)}")
            return None
    
    @staticmethod
    def extract_exame(html_content):
        """Extract structured content from Exame newsletter"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            sections = []
            
            # Section 1: First news from h2 until first "SAIBA MAIS"
            h2_tags = soup.find_all('h2')
            if h2_tags:
                first_section = {
                    'title': h2_tags[0].get_text(strip=True),
                    'content': []
                }
                
                current = h2_tags[0].find_next()
                while current:
                    if current.name == 'a' and 'SAIBA MAIS' in current.get_text().upper():
                        break
                    if current.name == 'span':
                        span_text = current.get_text(strip=True)
                        if span_text:
                            first_section['content'].append(span_text)
                    current = current.find_next()
                
                sections.append(first_section)
            
            # Section 2: "O que mais você precisa saber hoje" - multiple h2 headlines with spans
            marker_span = soup.find('span', string=lambda t: t and 'O que mais você precisa saber hoje' in t)
            if marker_span:
                # Find all h2 tags after this marker
                current = marker_span.find_next('h2')
                
                while current and current.name == 'h2':
                    # Check if we've hit "No radar" section
                    if 'No radar' in current.get_text():
                        break
                    
                    headline_section = {
                        'title': current.get_text(strip=True),
                        'content': []
                    }
                    
                    # Find the span with content after this h2
                    next_elem = current.find_next()
                    while next_elem:
                        if next_elem.name == 'h2':  # Next headline
                            break
                        if next_elem.name == 'a' and 'SAIBA MAIS' in next_elem.get_text().upper():
                            break
                        if next_elem.name == 'span':
                            span_text = next_elem.get_text(strip=True)
                            if span_text and 'O que mais você precisa saber hoje' not in span_text:
                                headline_section['content'].append(span_text)
                        next_elem = next_elem.find_next()
                    
                    if headline_section['content']:
                        sections.append(headline_section)
                    
                    # Move to next h2
                    current = next_elem if next_elem and next_elem.name == 'h2' else None
            
            return sections
        
        except Exception as e:
            st.error(f"Erro ao extrair conteúdo da Exame: {str(e)}")
            return None
    
    @staticmethod
    def extract_bloomberg_linea(html_content):
        """Extract structured content from Bloomberg Línea newsletter"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            sections = []
            
            # Section 1: Main headline (h1) until "Leia a matéria completa →"
            h1_tag = soup.find('h1')
            if h1_tag:
                main_section = {
                    'title': h1_tag.get_text(strip=True),
                    'content': []
                }
                
                current = h1_tag.find_next()
                while current:
                    if current.name == 'a' and 'Leia a matéria completa' in current.get_text():
                        break
                    if current.name == 'p' or current.name == 'span':
                        text = current.get_text(strip=True)
                        if text:
                            main_section['content'].append(text)
                    current = current.find_next()
                
                sections.append(main_section)
            
            # Section 2: "No Radar" - every <strong> is a new headline
            no_radar_span = soup.find('span', string=lambda t: t and 'No Radar' in t)
            if no_radar_span:
                radar_section = {
                    'title': 'No Radar',
                    'content': []
                }
                
                current = no_radar_span.find_next()
                current_item = None
                
                while current:
                    if current.name == 'a' and 'Leia sobre o que move os mercados hoje' in current.get_text():
                        break
                    
                    if current.name == 'strong':
                        # New headline
                        if current_item:
                            radar_section['content'].append(current_item)
                        current_item = {
                            'headline': current.get_text(strip=True),
                            'text': ''
                        }
                    elif current.name in ['p', 'span'] and current_item:
                        # Content for current headline
                        text = current.get_text(strip=True)
                        # Skip the headline itself if it appears again
                        if text != current_item['headline']:
                            current_item['text'] += ' ' + text
                    
                    current = current.find_next()
                
                # Add the last item
                if current_item:
                    radar_section['content'].append(current_item)
                
                sections.append(radar_section)
            
            # Section 3: "Para não ficar de fora" until "Saiba mais →"
            para_nao_span = soup.find('span', string=lambda t: t and 'Para não ficar de fora' in t)
            if para_nao_span:
                fora_section = {
                    'title': 'Para não ficar de fora',
                    'content': []
                }
                
                current = para_nao_span.find_next()
                while current:
                    if current.name == 'a' and 'Saiba mais' in current.get_text():
                        break
                    if current.name in ['p', 'span']:
                        text = current.get_text(strip=True)
                        if text and 'Para não ficar de fora' not in text:
                            fora_section['content'].append(text)
                    current = current.find_next()
                
                sections.append(fora_section)
            
            return sections
        
        except Exception as e:
            st.error(f"Erro ao extrair conteúdo da Bloomberg Línea: {str(e)}")
            return None


class NewsletterProcessor:
    """Process and summarize newsletter content with AI"""
    
    def __init__(self):
        self.client = None
        self.use_ai = False
        
        # Initialize Groq client
        if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
            try:
                self.client = Groq(api_key=GROQ_API_KEY)
                self.use_ai = True
            except:
                pass
    
    def sections_to_text(self, sections):
        """Convert parsed sections to plain text for AI processing"""
        if not sections:
            return ""
        
        text_parts = []
        for section in sections:
            # Add section title
            text_parts.append(f"\n## {section['title']}\n")
            
            # Add content based on structure
            if isinstance(section['content'], list):
                for item in section['content']:
                    if isinstance(item, dict):
                        # Has headline and text (like HIGHLIGHTS or VALE PARAR PARA LER)
                        if 'headline' in item and 'text' in item:
                            text_parts.append(f"**{item['headline']}**: {item['text']}")
                        else:
                            text_parts.append(str(item))
                    else:
                        text_parts.append(str(item))
            else:
                text_parts.append(str(section['content']))
        
        return "\n".join(text_parts)
    
    def summarize_investnews(self, sections):
        """Summarize Investnews with AI (max 1500 chars)"""
        if not sections:
            return None
        
        content = self.sections_to_text(sections)
        
        if not self.use_ai:
            return self.format_without_ai(content, 1500)
        
        specific_instructions = """
ESTRUTURA ESPECÍFICA PARA INVESTNEWS:
- Máximo 1500 caracteres no total
- Esta newsletter tem estrutura variável com seções obrigatórias e opcionais
- Seções obrigatórias: Primeira Notícia, HIGHLIGHTS (3 destaques), UMA IMAGEM, Segunda Notícia, VALE PARAR PARA LER
- Seções opcionais: UM NÚMERO, UM GRÁFICO, UMA FRASE (podem ou não estar presentes)

FORMATO DE SAÍDA OBRIGATÓRIO:
1. **Primeira Notícia**: Resuma a manchete principal em 2-3 frases concisas
2. **HIGHLIGHTS**: Liste os 3 destaques principais como tópicos numerados, cada um com 1 frase
3. **UMA IMAGEM**: Descreva brevemente o conteúdo visual em 1 frase
4. **Segunda Notícia**: Resuma a segunda manchete em 2-3 frases
5. Se houver **UM NÚMERO**: Mencione o dado numérico destacado em 1 frase
6. Se houver **UM GRÁFICO**: Descreva o gráfico destacado em 1 frase
7. Se houver **UMA FRASE**: Cite a frase destacada brevemente
8. **VALE PARAR PARA LER**: Liste 2-3 artigos recomendados como tópicos

REGRAS DE FORMATAÇÃO:
- Use **negrito** para títulos de seção
- Use tópicos (•) para listas
- Mantenha números e dados específicos exatos
- Linguagem direta e objetiva
"""
        
        return self.summarize_with_groq(content, "Investnews", specific_instructions, max_chars=1500)
    
    def summarize_exame(self, sections):
        """Summarize Exame with AI (max 1000 chars)"""
        if not sections:
            return None
        
        content = self.sections_to_text(sections)
        
        if not self.use_ai:
            return self.format_without_ai(content, 1000)
        
        specific_instructions = """
ESTRUTURA ESPECÍFICA PARA EXAME:
- Máximo 1000 caracteres no total
- Primeira notícia: Manchete principal do dia
- Seguida por múltiplos tópicos em "O que mais você precisa saber hoje"
- Cada tópico tem sua própria manchete e conteúdo

FORMATO DE SAÍDA OBRIGATÓRIO:
1. **Notícia Principal**: Resuma a manchete principal em 2-3 frases
2. **Tópicos do Dia**: Liste os principais tópicos como bullets numerados:
   • Tópico 1: [resumo em 1 frase]
   • Tópico 2: [resumo em 1 frase]
   • Tópico 3: [resumo em 1 frase]
   (continue para todos os tópicos, mas seja conciso)

REGRAS DE FORMATAÇÃO:
- Use **negrito** para "Notícia Principal" e "Tópicos do Dia"
- Use bullets (•) para cada tópico
- Máximo 1 frase por tópico
- Mantenha dados numéricos e nomes de empresas exatos
"""
        
        return self.summarize_with_groq(content, "Exame", specific_instructions, max_chars=1000)
    
    def summarize_bloomberg_linea(self, sections):
        """Summarize Bloomberg Línea with AI (max 1000 chars)"""
        if not sections:
            return None
        
        content = self.sections_to_text(sections)
        
        if not self.use_ai:
            return self.format_without_ai(content, 1000)
        
        specific_instructions = """
ESTRUTURA ESPECÍFICA PARA BLOOMBERG LÍNEA:
- Máximo 1000 caracteres no total
- Três seções principais: Notícia Principal, No Radar, Para não ficar de fora
- No Radar contém múltiplas notícias curtas com headlines

FORMATO DE SAÍDA OBRIGATÓRIO:
1. **Notícia Principal**: Resuma a manchete de abertura em 2-3 frases
2. **No Radar**: Liste as principais notícias como tópicos:
   • [Headline 1]: [resumo em 1 frase]
   • [Headline 2]: [resumo em 1 frase]
   • [Headline 3]: [resumo em 1 frase]
3. **Para não ficar de fora**: Resuma as notícias complementares em 1-2 frases

REGRAS DE FORMATAÇÃO:
- Use **negrito** para títulos de seção
- Use bullets (•) para itens do No Radar
- Mantenha headlines originais mas traduza o conteúdo
- Priorize informações sobre mercados brasileiros e latino-americanos
"""
        
        return self.summarize_with_groq(content, "Bloomberg Línea", specific_instructions, max_chars=1000)
    
    def summarize_with_groq(self, content, source_name, specific_instructions, max_chars=1000):
        """Use Groq AI to summarize and translate content"""
        if not self.use_ai or not content:
            return self.format_without_ai(content, max_chars)
        
        prompt = f"""Você é um analista financeiro especializado em resumir newsletters de mercado mantendo FIDELIDADE TOTAL à estrutura original e traduzindo para português brasileiro.

FONTE: {source_name}

CONTEÚDO ORIGINAL:
{content[:8000]}

{specific_instructions}

REGRAS CRÍTICAS DE TRADUÇÃO E FORMATAÇÃO:
1. Traduza TODO o conteúdo para PORTUGUÊS BRASILEIRO de forma natural e fluente
2. NUNCA deixe frases ou palavras em inglês no resultado final
3. Mantenha TODOS os números, percentuais e dados específicos EXATAMENTE como no original
4. Preserve nomes próprios de empresas, índices e pessoas sem tradução
5. Use terminologia financeira profissional em português (ex: "ações", "títulos", "rendimentos")
6. Seja CONCISO mas COMPLETO - capture todas as informações essenciais
7. RESPEITE RIGOROSAMENTE O LIMITE DE {max_chars} CARACTERES
8. Mantenha a estrutura de seções e formatação especificada acima
9. Use linguagem clara e profissional adequada ao mercado financeiro brasileiro

QUALIDADE DA TRADUÇÃO:
- Priorize naturalidade e fluidez em português brasileiro
- Evite traduções literais que soem artificiais
- Use construções de frase idiomáticas do português
- Contexto de mercado financeiro brasileiro

Responda APENAS com o resumo traduzido e formatado em HTML, sem introduções ou comentários adicionais. Use as tags HTML fornecidas abaixo para formatação:

HTML FORMATTING TAGS:
- Títulos de seção: <div style='font-size: 18px; font-weight: 700; color: #d4af37; margin: 15px 0 8px 0; font-family: Playfair Display;'>TÍTULO</div>
- Parágrafos: <p style='margin: 6px 0; line-height: 1.6;'>texto</p>
- Tópicos com headline: <div style='margin: 8px 0;'><strong style='color: #f4d03f;'>• Headline</strong> texto</div>
- Negrito: <strong style='color: #f4d03f;'>texto</strong>
"""
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Você é um tradutor financeiro especializado que cria resumos precisos, bem estruturados e em português brasileiro fluente, mantendo fidelidade total à estrutura original do conteúdo. Você sempre formata suas respostas em HTML seguindo as tags especificadas."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=1200,
                top_p=0.95,
                stream=False
            )
            
            result = chat_completion.choices[0].message.content.strip()
            
            # Ensure it's within character limit
            if len(result) > max_chars * 1.2:  # Allow 20% buffer for HTML tags
                # Truncate intelligently at last complete sentence
                result = result[:int(max_chars * 1.2)]
                last_period = result.rfind('</p>')
                if last_period > max_chars * 0.8:
                    result = result[:last_period + 4]
            
            return result
            
        except Exception as e:
            st.warning(f"Erro ao processar com IA: {str(e)}")
            return self.format_without_ai(content, max_chars)
    
    def format_without_ai(self, content, max_chars):
        """Fallback formatting without AI"""
        # Simple truncation with HTML formatting
        clean_content = content[:max_chars]
        return f"<p style='margin: 6px 0; line-height: 1.6;'>{clean_content}</p>"


def display_newsletter_card(newsletter_data):
    """Display a newsletter in a card format using Streamlit containers"""
    
    with st.container():
        # Status badge
        if newsletter_data['status'] == 'success':
            st.markdown('<div style="background-color: #1a4d2e; color: #4ade80; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 600; display: inline-block; margin-bottom: 10px;">✓ CARREGADO</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background-color: #4d1a1a; color: #f87171; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 600; display: inline-block; margin-bottom: 10px;">✗ ERRO</div>', unsafe_allow_html=True)
        
        # Source header
        st.markdown(f'<div style="color: #d4af37; font-size: 24px; font-weight: 700; margin-bottom: 8px; font-family: \'Helvetica Neue\', sans-serif; letter-spacing: 0.5px;">{newsletter_data["source"]}</div>', unsafe_allow_html=True)
        
        # Timestamp
        st.markdown(f'<div style="color: #888; font-size: 12px; margin-bottom: 15px; font-style: italic;">{newsletter_data["type"]} • {newsletter_data["timestamp"]}</div>', unsafe_allow_html=True)
        
        # Content
        if newsletter_data['status'] == 'success' and newsletter_data['content']:
            st.markdown(
                f"""
                <div style="
                    color: white;
                    font-size: 16px;
                    line-height: 1.7;
                    text-align: justify;
                    margin-top: 10px;
                ">
                {newsletter_data['content']}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="color: #ff6b6b; font-size: 16px; margin-top: 10px;">'
                f"⚠️ {newsletter_data.get('error', 'Conteúdo indisponível')}"
                '</div>',
                unsafe_allow_html=True
            )
        
        # Divider
        st.markdown("<div style='border-top: 1px solid #333; margin: 30px 0;'></div>", unsafe_allow_html=True)


# Initialize newsletter session state
if 'newsletters_data' not in st.session_state:
    st.session_state.newsletters_data = {}
if 'last_news_refresh' not in st.session_state:
    st.session_state.last_news_refresh = None


# ═══════════════════════════════════════════════════════════════════════════════
# SUPABASE HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class SupabaseNewsletter:
    """Helper class for Supabase newsletter operations"""
    
    def __init__(self):
        self.url = SUPABASE_URL
        self.key = SUPABASE_KEY
        self.headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json"
        }
        self.table_name = "newsletters"
    
    def save_newsletter(self, newsletter_type, content, source_name):
        """Save or update newsletter in Supabase"""
        if not self.url or not self.key:
            return False
        
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            
            data = {
                "newsletter_type": newsletter_type,
                "content": content,
                "source_name": source_name,
                "upload_date": today,
                "created_at": datetime.now().isoformat()
            }
            
            # First, try to delete existing newsletter of same type for today
            delete_url = f"{self.url}/rest/v1/{self.table_name}?newsletter_type=eq.{newsletter_type}&upload_date=eq.{today}"
            requests.delete(delete_url, headers=self.headers)
            
            # Insert new newsletter
            insert_url = f"{self.url}/rest/v1/{self.table_name}"
            response = requests.post(insert_url, headers=self.headers, json=data)
            
            return response.status_code in [200, 201]
        except Exception as e:
            st.error(f"Erro ao salvar no Supabase: {str(e)}")
            return False
    
    def get_todays_newsletters(self):
        """Get all newsletters for today"""
        if not self.url or not self.key:
            return {}
        
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            url = f"{self.url}/rest/v1/{self.table_name}?upload_date=eq.{today}"
            
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                newsletters = response.json()
                result = {}
                for newsletter in newsletters:
                    result[newsletter['newsletter_type']] = {
                        'source': newsletter['source_name'],
                        'timestamp': newsletter['upload_date'],
                        'type': 'Newsletter Upload',
                        'content': newsletter['content'],
                        'status': 'success'
                    }
                return result
            else:
                return {}
        except Exception as e:
            return {}


def show_dashboard():
    st.title("📈 Painel de Índices de Mercado")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Análise de Índices", "Curvas de Juros", "Curvas de Crédito", "Tesouro Direto", "Noticiário Renda Fixa"])
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1: ANÁLISE DE ÍNDICES
    # ═══════════════════════════════════════════════════════════════════════════
    with tab1:
        indices_data = load_all_indices()
        
        if not indices_data:
            st.error("Nenhum dado disponível. Por favor, verifique sua conexão com a internet.")
            return
        
        st.sidebar.title("⚙️ Configurações")
        st.sidebar.info("Use os controles na página principal para personalizar o gráfico.")
        
        available_indices = sorted(list(indices_data.keys()))
        default_indices = ['CDI', 'IBOVESPA', 'S&P 500 (USD)'] if all(x in available_indices for x in ['CDI', 'IBOVESPA', 'S&P 500 (USD)']) else available_indices[:3]
        
        st.header("📊 Gráfico de Retornos Acumulados")
        
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
                options=['Tudo', '120M', '36M', '24M', '12M', 'YTD', 'MTD'],
                index=1,
                help="Escolha o período de tempo para análise"
            )
        
        st.markdown("---")
        st.subheader(f"Retornos Acumulados ({period})")
        
        if selected_indices:
            cumulative_returns = calculate_cumulative_returns_daily(indices_data, selected_indices, period)
            
            if not cumulative_returns.empty:
                fig = go.Figure()
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
                    xaxis=dict(showgrid=True, gridcolor='#333', title='Data'),
                    yaxis=dict(showgrid=True, gridcolor='#333', title='Retorno Acumulado (%)'),
                    hovermode='x unified',
                    legend=dict(bgcolor='#1a1a1a', bordercolor='#d4af37', borderwidth=1),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Nenhum dado disponível para os índices e período selecionados.")
        else:
            st.info("Por favor, selecione pelo menos um índice.")
        
        # Rankings
        st.header("🏆 Rankings de Performance")
        
        all_returns = {}
        for name, df in indices_data.items():
            if df is not None and len(df) > 0:
                returns = calc_returns(df)
                all_returns[name] = returns
        
        mtd_returns = pd.DataFrame({name: returns.loc['MTD'].values[0] for name, returns in all_returns.items()}, index=['Return']).T
        ytd_returns = pd.DataFrame({name: returns.loc['YTD'].values[0] for name, returns in all_returns.items()}, index=['Return']).T
        
        mtd_returns = mtd_returns.sort_values('Return', ascending=False)
        ytd_returns = ytd_returns.sort_values('Return', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            current_month = calendar.month_name[datetime.now().month]
            month_pt = {
                'January': 'Janeiro', 'February': 'Fevereiro', 'March': 'Março',
                'April': 'Abril', 'May': 'Maio', 'June': 'Junho',
                'July': 'Julho', 'August': 'Agosto', 'September': 'Setembro',
                'October': 'Outubro', 'November': 'Novembro', 'December': 'Dezembro'
            }
            current_month_pt = month_pt.get(current_month, current_month)
            st.subheader(f"🥇 Rankings de {current_month_pt} (MTD)")
            
            mtd_display = mtd_returns.copy()
            mtd_display['Rank'] = range(1, len(mtd_display) + 1)
            mtd_display = mtd_display[['Rank', 'Return']]
            
            html_table = '<table style="width:100%; border-collapse: collapse;">'
            html_table += '<thead><tr style="background-color: #0a0a0a;"><th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Rank</th>'
            html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Índice</th>'
            html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Retorno</th></tr></thead><tbody>'
            
            for idx_name, row in mtd_display.iterrows():
                ret_val = row['Return']
                color = '#00e100' if ret_val >= 0 else '#f20000'
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
            
            ytd_display = ytd_returns.copy()
            ytd_display['Rank'] = range(1, len(ytd_display) + 1)
            ytd_display = ytd_display[['Rank', 'Return']]
            
            html_table = '<table style="width:100%; border-collapse: collapse;">'
            html_table += '<thead><tr style="background-color: #0a0a0a;"><th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Rank</th>'
            html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Índice</th>'
            html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Retorno</th></tr></thead><tbody>'
            
            for idx_name, row in ytd_display.iterrows():
                ret_val = row['Return']
                color = '#00e100' if ret_val >= 0 else '#f20000'
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
            var_df = var_df.sort_values('Variation (%)', ascending=False)
            
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
                color = '#00e100' if variation >= 0 else '#f20000'
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
        
        # Monthly Matrix
        st.header("📅 Matriz de Performance Mensal (Últimos 12 Meses)")
        
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
            unique_indices = list(indices_data.keys())
            color_palette = [
            "#020070",  
            "#383cff",  
            "#0078be",  
            "#22aba8",  
            "#FFFFFF",  
            "#a38e30",  
            "#fcff96",  
            "#c27e35",  
            "#ffb638",  
            "#fea269",  
            "#656565",  
            "#958F8F",  
            "#FC47FF",  
            "#AA6EF9",
            "#3EA47F",
            "#9028C8"
        ]
            index_colors = {idx: color_palette[i % len(color_palette)] for i, idx in enumerate(unique_indices)}
            
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
                        
                        ret_color = '#00e100' if ret_val >= 0 else "#f20000"
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
        
        # Yearly Matrix
        st.header("🎯 Matriz de Performance Anual (Multi-Período)")
        
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
                        
                        ret_color = '#00e100' if ret_val >= 0 else '#f20000'
                        arrow = '▲' if ret_val >= 0 else '▼'
                        
                        html_table += f'<td style="border: 1px solid #333; padding: 8px; text-align: center; background-color: {bg_color};">'
                        html_table += f'<div style="color: #0a0a0a; font-weight: 600; font-size: 13px;">{idx_name}</div>'
                        html_table += f'<div style="color: {ret_color}; font-weight: bold; font-size: 12px; margin-top: 3px;">{arrow} {abs(ret_val):.2f}%</div>'
                        html_table += '</td>'
                    else:
                        html_table += '<td style="border: 1px solid #333; padding: 10px; text-align: center; background-color: #1a1a1a;">-</td>'

    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2: TESOURO DIRETO
    # ═══════════════════════════════════════════════════════════════════════════
                        html_table += '<td style="border: 1px solid #333; padding: 10px; text-align: center; background-color: #1a1a1a;">-</td>'
                
                html_table += '</tr>'
            
            html_table += '</tbody></table>'
            st.markdown(html_table, unsafe_allow_html=True)
        else:
            st.warning("Não foi possível criar a matriz de ranking anual.")
        
        st.markdown("---")
        st.markdown(
            "<p style='text-align: center; color: #d4af37; font-family: Montserrat;'>Fontes de dados: ANBIMA, Yahoo Finance, Banco Central do Brasil</p>",
            unsafe_allow_html=True
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2: TESOURO DIRETO
    # ═══════════════════════════════════════════════════════════════════════════
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2: CURVAS ANBIMA (from news_v5.py)
    # ═══════════════════════════════════════════════════════════════════════
    
    with tab2:
        st.header("📈 Análise de Curvas ANBIMA")
        
        # Check if user is admin
        is_admin = st.session_state.get('user_logged_in') == 'admin'
        
        # Initialize Supabase client
        supabase_client = init_supabase_client()
        
        
        # Admin upload section
        if is_admin:
            st.markdown("### 🔐 Área do Administrador")
            
            uploaded_excel = st.file_uploader(
                "Upload arquivo Excel com curvas ANBIMA",
                type=['xlsx', 'xls'],
                key='curves_excel',
                help="Arquivo deve conter 3 sheets: 'ETTJ PRE', 'ETTJ IPCA', 'Inflação Implícita'"
            )
            
            if uploaded_excel:
                if st.button("💾 Salvar Curvas no Banco", type="primary"):
                    with st.spinner("Processando e salvando..."):
                        json_data = process_excel_to_json(uploaded_excel)
                        if json_data and supabase_client:
                            if save_curves_to_database(supabase_client, json_data):
                                st.success("✅ Curvas salvas com sucesso!")
                                # Clear cache so new data is loaded
                                st.session_state.curves_data_cache = None
                                st.rerun()
                            else:
                                st.error("❌ Erro ao salvar curvas")
                        elif not supabase_client:
                            st.error("❌ Cliente Supabase não inicializado")
            
            st.markdown("---")
            
            # Add refresh button for admin
            if st.button("🔄 Recarregar Dados do Banco", use_container_width=True):
                st.session_state.curves_data_cache = None
                st.success("✅ Cache limpo! Dados serão recarregados.")
                st.rerun()
        
        # Load curves data (use session state to avoid reloading on every button click)
        if 'curves_data_cache' not in st.session_state:
            st.session_state.curves_data_cache = None
        
        # Only load from database if not in cache
        if st.session_state.curves_data_cache is None:
            if supabase_client:
                try:
                    curves_data = load_curves_from_database(supabase_client)
                    if curves_data:
                        st.session_state.curves_data_cache = curves_data
                except Exception as e:
                    st.error(f"Erro ao carregar dados: {e}")
                    curves_data = None
            else:
                curves_data = None
        else:
            curves_data = st.session_state.curves_data_cache
        
        if not curves_data:
            st.warning("⚠️ Nenhuma curva disponível.")
            if not is_admin:
                st.info("💡 Aguardando upload do administrador.")
        else:
            # Convert JSON to DataFrames
            dataframes = json_to_dataframes(curves_data)
            
            if dataframes:
                # Buttons to select curve
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("ETTJ IPCA", use_container_width=True, key="btn_ipca"):
                        st.session_state.selected_curve = 'ETTJ IPCA'
                
                with col2:
                    if st.button("ETTJ PRE", use_container_width=True, key="btn_pre"):
                        st.session_state.selected_curve = 'ETTJ PRE'
                
                with col3:
                    if st.button("Inflação Implícita", use_container_width=True, key="btn_inflacao"):
                        st.session_state.selected_curve = 'Inflação Implícita'
                
                # Initialize selected curve
                if 'selected_curve' not in st.session_state:
                    st.session_state.selected_curve = 'ETTJ IPCA'
                
                selected_curve = st.session_state.selected_curve
                
                st.subheader(f"Curva Selecionada: {selected_curve}")
                
                # Get selected DataFrame
                df = dataframes.get(selected_curve)
                
                if df is not None:
                    # 1. Display curve chart
                    st.markdown("### 📊 Gráfico da Curva")
                    fig = create_curves_chart(df, selected_curve)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 2. Calculate and display rate variations
                    st.markdown("### 📉 Variações de Taxas (10 maiores em ordem)")
                    
                    variations = calculate_curves_variations(df)
                    
                    if variations:
                        cols = st.columns(min(5, len(variations)))
                        
                        for i, (days_back, var_list) in enumerate(sorted(variations.items())):
                            if i < len(cols):
                                with cols[i]:
                                    st.markdown(f"#### Variação de {days_back} Dia{'s' if days_back > 1 else ''}")
                                    
                                    html_table = '<table style="width:100%; border-collapse: collapse; font-size: 12px;">'
                                    html_table += '<thead><tr style="background-color: #0a0a0a;">'
                                    html_table += '<th style="border: 1px solid #d4af37; padding: 8px; color: #d4af37;">Vértice</th>'
                                    html_table += '<th style="border: 1px solid #d4af37; padding: 8px; color: #d4af37;">Variação (p.p.)</th>'
                                    html_table += '</tr></thead><tbody>'
                                    
                                    for vertex, variation in var_list:
                                        color = "#00e100" if variation >= 0 else "#f20000"
                                        arrow = '▲' if variation >= 0 else '▼'
                                        
                                        html_table += '<tr>'
                                        html_table += f'<td style="border: 1px solid #333; padding: 6px; text-align: center; background-color: #1a1a1a; color: #d4af37;">{vertex:.2f}</td>'
                                        html_table += f'<td style="border: 1px solid #333; padding: 6px; text-align: right; background-color: #1a1a1a; color: {color}; font-weight: bold;">{arrow} {abs(variation):.4f}%</td>'
                                        html_table += '</tr>'
                                    
                                    html_table += '</tbody></table>'
                                    st.markdown(html_table, unsafe_allow_html=True)
                    
                    # 3. Display rates table
                    st.markdown(f"### 📋 Tabela de Taxas - {selected_curve}")
                    
                    rates_table = create_curves_table(df, selected_curve)
                    if rates_table is not None:
                        st.dataframe(rates_table, use_container_width=True, height=400)
                    
                    # 4. Two-day comparison
                    st.markdown("### 🔄 Comparação Entre Dois Dias")
                    
                    date_columns = df.columns[1:].tolist()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        date1 = st.selectbox(
                            "Selecione o Dia Mais Antigo",
                            options=date_columns,
                            index=0,
                            key='comparison_date1'
                        )
                    
                    with col2:
                        date2 = st.selectbox(
                            "Selecione o Dia Mais Recente",
                            options=date_columns,
                            index=len(date_columns) - 1,
                            key='comparison_date2'
                        )
                    
                    if st.button("Gerar Comparação", use_container_width=False):
                        vertices = df.iloc[:, 0].values
                        rates1 = df[date1].values
                        rates2 = df[date2].values
                        
                        st.markdown(f"#### Comparação: {date1} vs {date2}")
                        
                        html_table = '<table style="width:100%; border-collapse: collapse; font-size: 13px;">'
                        html_table += '<thead><tr style="background-color: #0a0a0a;">'
                        html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Vértice (Anos)</th>'
                        html_table += f'<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">{date1}</th>'
                        html_table += f'<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">{date2}</th>'
                        html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Variação (p.p.)</th>'
                        html_table += '</tr></thead><tbody>'
                        
                        for i, vertex in enumerate(vertices):
                            if pd.notna(rates1[i]) and pd.notna(rates2[i]):
                                rate1 = rates1[i]
                                rate2 = rates2[i]
                                variation = rate2 - rate1
                                
                                color = '#00e100' if variation >= 0 else "#f20000"
                                arrow = '▲' if variation >= 0 else '▼'
                                
                                html_table += '<tr>'
                                html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: center; background-color: #1a1a1a; color: #d4af37; font-weight: bold;">{vertex:.2f}</td>'
                                html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: right; background-color: #1a1a1a; color: #d4af37;">{rate1:.4f}%</td>'
                                html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: right; background-color: #1a1a1a; color: #d4af37;">{rate2:.4f}%</td>'
                                html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: right; background-color: #1a1a1a; color: {color}; font-weight: bold;">{arrow} {abs(variation):.4f}%</td>'
                                html_table += '</tr>'
                        
                        html_table += '</tbody></table>'
                        st.markdown(html_table, unsafe_allow_html=True)
                        
                        # Summary statistics
                        st.markdown("#### Estatísticas da Comparação")
                        valid_variations = [rates2[i] - rates1[i] for i in range(len(vertices)) 
                                          if pd.notna(rates1[i]) and pd.notna(rates2[i])]
                        
                        if valid_variations:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Variação Média", f"{sum(valid_variations)/len(valid_variations):.4f}%")
                            with col2:
                                st.metric("Variação Máxima", f"{max(valid_variations):.4f}%")
                            with col3:
                                st.metric("Variação Mínima", f"{min(valid_variations):.4f}%")

    
    # ═══════════════════════════════════════════════════════════════════════════════
    # TAB 3: CURVAS DE CRÉDITO
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab3:
        st.header("📊 Análise de Curvas de Crédito")
        
        # Check if user is admin
        is_admin = st.session_state.get('user_logged_in') == 'admin'
        
        # Initialize Supabase client
        supabase_client = init_supabase_client()
        
        # Admin upload section
        if is_admin:
            st.markdown("### 🔐 Área do Administrador")
            
            uploaded_excel = st.file_uploader(
                "Upload arquivo Excel com curvas de crédito",
                type=['xlsx', 'xls'],
                key='credit_curves_excel',
                help="Arquivo deve conter 3 sheets: 'AAA', 'AA', 'A'"
            )
            
            if uploaded_excel:
                if st.button("💾 Salvar Curvas de Crédito no Banco", type="primary"):
                    with st.spinner("Processando e salvando..."):
                        json_data = process_credit_excel_to_json(uploaded_excel)
                        if json_data and supabase_client:
                            if save_credit_curves_to_database(supabase_client, json_data):
                                st.success("✅ Curvas de crédito salvas com sucesso!")
                                # Clear cache
                                st.session_state.credit_curves_data_cache = None
                                st.rerun()
                            else:
                                st.error("❌ Erro ao salvar curvas de crédito")
                        elif not supabase_client:
                            st.error("❌ Cliente Supabase não inicializado")
            
            st.markdown("---")
            
            # Add refresh button for admin
            if st.button("🔄 Recarregar Dados de Crédito do Banco", use_container_width=True):
                st.session_state.credit_curves_data_cache = None
                st.success("✅ Cache limpo! Dados serão recarregados.")
                st.rerun()
        
        # Load credit curves data (use session state to avoid reloading)
        if 'credit_curves_data_cache' not in st.session_state:
            st.session_state.credit_curves_data_cache = None
        
        # Only load from database if not in cache
        if st.session_state.credit_curves_data_cache is None:
            if supabase_client:
                try:
                    credit_curves_data = load_credit_curves_from_database(supabase_client)
                    if credit_curves_data:
                        st.session_state.credit_curves_data_cache = credit_curves_data
                except Exception as e:
                    st.error(f"Erro ao carregar dados: {e}")
                    credit_curves_data = None
            else:
                credit_curves_data = None
        else:
            credit_curves_data = st.session_state.credit_curves_data_cache
        
        if not credit_curves_data:
            st.warning("⚠️ Nenhuma curva de crédito disponível.")
            if not is_admin:
                st.info("💡 Aguardando upload do administrador.")
        else:
            # Convert JSON to DataFrames
            dataframes = json_to_dataframes(credit_curves_data)
            
            if dataframes:
                # Get available dates from any dataframe
                sample_df = list(dataframes.values())[0]
                available_dates = sample_df.columns[1:].tolist() if sample_df is not None else []
                
                # Create 4 main buttons
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button("Comparativo", use_container_width=True, key="btn_credit_comp"):
                        st.session_state.credit_view = 'comparativo'
                
                with col2:
                    if st.button("AAA", use_container_width=True, key="btn_credit_aaa"):
                        st.session_state.credit_view = 'AAA'
                
                with col3:
                    if st.button("AA", use_container_width=True, key="btn_credit_aa"):
                        st.session_state.credit_view = 'AA'
                
                with col4:
                    if st.button("A", use_container_width=True, key="btn_credit_a"):
                        st.session_state.credit_view = 'A'
                
                # Initialize view
                if 'credit_view' not in st.session_state:
                    st.session_state.credit_view = 'comparativo'
                
                st.markdown("---")
                
                # ═══════════════════════════════════════════════════════════════
                # SECTION 1: COMPARATIVO
                # ═══════════════════════════════════════════════════════════════
                if st.session_state.credit_view == 'comparativo':
                    st.subheader("Comparativo de Ratings")
                    
                    # Date selector
                    selected_date = st.selectbox(
                        "Selecione o Dia de Referência",
                        options=available_dates,
                        index=len(available_dates) - 1,  # Default to latest
                        key='credit_comp_date'
                    )
                    
                    # Chart
                    fig = create_credit_comparison_chart(dataframes, selected_date)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # ═══════════════════════════════════════════════════════════
                    # ═══════════════════════════════════════════════════════════
                    # ═══════════════════════════════════════════════════════════
                    # SPREAD DIFFERENCES SECTION
                    # ═══════════════════════════════════════════════════════════
                    st.markdown("### 📉 Diferença de Spreads entre Ratings")
                    st.markdown("*Média da diferença em p.p. (rating menor - rating maior)*")
                    
                    # Calculate all spread differences
                    rating_pairs = [('A', 'AAA'), ('A', 'AA'), ('AA', 'AAA')]
                    
                    # Create data for table
                    table_data = []
                    for rating_lower, rating_higher in rating_pairs:
                        row = {
                            'Comparação': f'{rating_lower} - {rating_higher}'
                        }
                        for date in available_dates[:6]:
                            avg_diff = calculate_spread_differences(
                                dataframes, rating_lower, rating_higher, date
                            )
                            if avg_diff is not None:
                                row[date] = f'{avg_diff:.4f}%'
                            else:
                                row[date] = '-'
                        table_data.append(row)
                    
                    # Create styled HTML table
                    html_table = '<table style="width:100%; border-collapse: collapse; font-size: 14px; margin: 20px 0;">'
                    
                    # Header
                    html_table += '<thead><tr style="background-color: #0a0a0a;">'
                    html_table += '<th style="border: 2px solid #d4af37; padding: 12px; color: #d4af37; text-align: center; font-weight: bold;">Comparação</th>'
                    for date in available_dates[:6]:
                        html_table += f'<th style="border: 2px solid #d4af37; padding: 12px; color: #d4af37; text-align: center; font-weight: bold;">{date}</th>'
                    html_table += '</tr></thead>'
                    
                    # Body
                    html_table += '<tbody>'
                    for row_data in table_data:
                        html_table += '<tr>'
                        html_table += f'<td style="border: 1px solid #d4af37; padding: 12px; text-align: center; background-color: #1a1a1a; color: #d4af37; font-weight: bold;">{row_data["Comparação"]}</td>'
                        for date in available_dates[:6]:
                            value = row_data.get(date, '-')
                            html_table += f'<td style="border: 1px solid #333; padding: 12px; text-align: center; background-color: #1a1a1a; color: #d4af37; font-size: 16px;">{value}</td>'
                        html_table += '</tr>'
                    html_table += '</tbody></table>'
                    
                    st.markdown(html_table, unsafe_allow_html=True)
                
                # SECTIONS 2-4: INDIVIDUAL RATINGS (AAA, AA, A)
                # ═══════════════════════════════════════════════════════════════
                elif st.session_state.credit_view in ['AAA', 'AA', 'A']:
                    rating = st.session_state.credit_view
                    df = dataframes.get(rating)
                    
                    if df is not None:
                        st.subheader(f"Rating {rating}")
                        
                        # 1. Display curve chart
                        st.markdown("### 📊 Gráfico de Spreads")
                        fig = create_credit_curves_chart(df, rating)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 2. Calculate and display rate variations
                        st.markdown("### 📉 Variações de Spreads")
                        
                        variations = calculate_credit_variations(df)
                        
                        if variations:
                            cols = st.columns(min(5, len(variations)))
                            
                            for i, (days_back, var_list) in enumerate(sorted(variations.items())):
                                if i < len(cols):
                                    with cols[i]:
                                        st.markdown(f"#### Variação de {days_back} Dia{'s' if days_back > 1 else ''}")
                                        
                                        html_table = '<table style="width:100%; border-collapse: collapse; font-size: 12px;">'
                                        html_table += '<thead><tr style="background-color: #0a0a0a;">'
                                        html_table += '<th style="border: 1px solid #d4af37; padding: 8px; color: #d4af37;">Vértice</th>'
                                        html_table += '<th style="border: 1px solid #d4af37; padding: 8px; color: #d4af37;">Variação (p.p.)</th>'
                                        html_table += '</tr></thead><tbody>'
                                        
                                        for vertex, variation in var_list:
                                            color = "#00e100" if variation >= 0 else "#f20000"
                                            arrow = '▲' if variation >= 0 else '▼'
                                            
                                            html_table += '<tr>'
                                            html_table += f'<td style="border: 1px solid #333; padding: 6px; text-align: center; background-color: #1a1a1a; color: #d4af37;">{vertex:.2f}</td>'
                                            html_table += f'<td style="border: 1px solid #333; padding: 6px; text-align: right; background-color: #1a1a1a; color: {color}; font-weight: bold;">{arrow} {abs(variation):.4f}%</td>'
                                            html_table += '</tr>'
                                        
                                        html_table += '</tbody></table>'
                                        st.markdown(html_table, unsafe_allow_html=True)
                        
                        # 3. Display rates table
                        st.markdown(f"### 📋 Tabela de Spreads - {rating}")
                        
                        rates_table = create_credit_table(df)
                        if rates_table is not None:
                            st.dataframe(rates_table, use_container_width=True, height=400)
                        
                        # 4. Two-day comparison
                        st.markdown("### 🔄 Comparação Entre Dois Dias")
                        
                        date_columns = df.columns[1:].tolist()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            date1 = st.selectbox(
                                "Selecione o Dia Mais Antigo",
                                options=date_columns,
                                index=0,
                                key=f'credit_comparison_date1_{rating}'
                            )
                        
                        with col2:
                            date2 = st.selectbox(
                                "Selecione o Dia Mais Recente",
                                options=date_columns,
                                index=len(date_columns) - 1,
                                key=f'credit_comparison_date2_{rating}'
                            )
                        
                        if st.button("Gerar Comparação", use_container_width=False, key=f'credit_gen_comp_{rating}'):
                            vertices = df.iloc[:, 0].values
                            rates1 = df[date1].values
                            rates2 = df[date2].values
                            
                            st.markdown(f"#### Comparação: {date1} vs {date2}")
                            
                            html_table = '<table style="width:100%; border-collapse: collapse; font-size: 13px;">'
                            html_table += '<thead><tr style="background-color: #0a0a0a;">'
                            html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Vértice (Anos)</th>'
                            html_table += f'<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">{date1}</th>'
                            html_table += f'<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">{date2}</th>'
                            html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Variação (p.p.)</th>'
                            html_table += '</tr></thead><tbody>'
                            
                            for i, vertex in enumerate(vertices):
                                if pd.notna(rates1[i]) and pd.notna(rates2[i]):
                                    rate1 = rates1[i]
                                    rate2 = rates2[i]
                                    variation = rate2 - rate1
                                    
                                    color = '#00e100' if variation >= 0 else "#f20000"
                                    arrow = '▲' if variation >= 0 else '▼'
                                    
                                    html_table += '<tr>'
                                    html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: center; background-color: #1a1a1a; color: #d4af37; font-weight: bold;">{vertex:.2f}</td>'
                                    html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: right; background-color: #1a1a1a; color: #d4af37;">{rate1:.4f}%</td>'
                                    html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: right; background-color: #1a1a1a; color: #d4af37;">{rate2:.4f}%</td>'
                                    html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: right; background-color: #1a1a1a; color: {color}; font-weight: bold;">{arrow} {abs(variation):.4f}%</td>'
                                    html_table += '</tr>'
                            
                            html_table += '</tbody></table>'
                            st.markdown(html_table, unsafe_allow_html=True)
                            
                            # Summary statistics
                            st.markdown("#### Estatísticas da Comparação")
                            valid_variations = [rates2[i] - rates1[i] for i in range(len(vertices)) 
                                              if pd.notna(rates1[i]) and pd.notna(rates2[i])]
                            
                            if valid_variations:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Variação Média", f"{sum(valid_variations)/len(valid_variations):.4f}%")
                                with col2:
                                    st.metric("Variação Máxima", f"{max(valid_variations):.4f}%")
                                with col3:
                                    st.metric("Variação Mínima", f"{min(valid_variations):.4f}%")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3: NOTÍCIAS DE MERCADO
    # ═══════════════════════════════════════════════════════════════════════════
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 4: NEWSLETTERS (from news_v5.py)
    # ═══════════════════════════════════════════════════════════════════════
    

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 4: TESOURO DIRETO
    # ═══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.header("💰 Análise de Taxas do Tesouro Direto")
        
        # Load Tesouro Direto data
        td_df = load_tesouro_direto_data()
        
        if td_df is not None:
            # Buttons to select bond type
            bond_types = [
                'Tesouro Selic',
                'Tesouro Prefixado',
                'Tesouro Prefixado com Juros Semestrais',
                'Tesouro IPCA+ com Juros Semestrais',
                'Tesouro IPCA+',
                'Tesouro Renda+ Aposentadoria Extra'
            ]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Tesouro Selic", use_container_width=True, key="btn_selic"):
                    st.session_state.selected_bond = 'Tesouro Selic'
                if st.button("Tesouro Renda+", use_container_width=True, key="btn_renda"):
                    st.session_state.selected_bond = 'Tesouro Renda+ Aposentadoria Extra'
            
            with col2:
                if st.button("Tesouro Prefixado", use_container_width=True, key="btn_pre_td"):
                    st.session_state.selected_bond = 'Tesouro Prefixado'
                if st.button("Tesouro Prefixado com Juros", use_container_width=True, key="btn_pre_juros"):
                    st.session_state.selected_bond = 'Tesouro Prefixado com Juros Semestrais'
            
            with col3:
                if st.button("Tesouro IPCA+", use_container_width=True, key="btn_ipca_td"):
                    st.session_state.selected_bond = 'Tesouro IPCA+'
                if st.button("Tesouro IPCA+ com Juros", use_container_width=True, key="btn_ipca_juros"):
                    st.session_state.selected_bond = 'Tesouro IPCA+ com Juros Semestrais'
            
            # Initialize selected bond if not set
            if 'selected_bond' not in st.session_state:
                st.session_state.selected_bond = 'Tesouro Selic'
            
            selected_bond = st.session_state.selected_bond
            
            st.subheader(f"Título Selecionado: {selected_bond}")
            
            # Get products data for selected bond
            products_df = products_td(td_df, selected_bond)
            
            if products_df is not None and len(products_df) > 0:
                # 1. Display curve chart
                st.markdown("### 📊 Gráfico da Curva de Taxas")
                
                # Get available dates (last 10 trading days)
                available_dates = list(products_df.index)
                
                # Date selection for chart
                default_dates = available_dates[-5:] if len(available_dates) >= 5 else available_dates
                
                selected_dates = st.multiselect(
                    "Selecione os dias para exibir no gráfico:",
                    options=available_dates,
                    default=default_dates,
                    format_func=lambda x: str(x),
                    key='td_chart_dates'
                )
                
                if selected_dates:
                    fig = create_td_chart(products_df, selected_dates, selected_bond)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Selecione pelo menos um dia para exibir o gráfico.")
                
                # 1.5. Display time series chart for each maturity
                st.markdown("### 📈 Série Histórica de Taxas por Vencimento")
                
                # Get all time series for current maturities
                all_time_series = get_all_maturities_time_series(td_df, selected_bond)
                
                if all_time_series:
                    
                    # Get maturities (already sorted in descending order from get_all_maturities_time_series)
                    # So first element is the newest maturity
                    sorted_maturities = list(all_time_series.keys())
                    
                    # Get the newest maturity (first in list since descending order)
                    newest_maturity = sorted_maturities[0]
                    
                    # Initialize or update selected maturity in session state
                    # Check if bond changed or if selected_maturity not set
                    if 'selected_maturity' not in st.session_state or \
                       'last_selected_bond' not in st.session_state or \
                       st.session_state.last_selected_bond != selected_bond:
                        st.session_state.selected_maturity = newest_maturity
                        st.session_state.last_selected_bond = selected_bond
                    
                    # If currently selected maturity is not in the available maturities, reset to newest
                    if st.session_state.selected_maturity not in sorted_maturities:
                        st.session_state.selected_maturity = newest_maturity
                    
                    # Create columns for buttons (max 6 per row)
                    n_cols = min(6, len(sorted_maturities))
                    cols = st.columns(n_cols)
                    
                    # Create buttons
                    for idx, maturity in enumerate(sorted_maturities):
                        col_idx = idx % n_cols
                        with cols[col_idx]:
                            if st.button(
                                str(maturity.year), 
                                use_container_width=True,
                                key=f"btn_maturity_{maturity}"
                            ):
                                st.session_state.selected_maturity = maturity
                    
                    # Display chart for selected maturity
                    selected_maturity = st.session_state.selected_maturity
                    
                    if selected_maturity in all_time_series:
                        time_series = all_time_series[selected_maturity]
                        fig_ts = create_maturity_time_series_chart(time_series, selected_maturity, selected_bond)
                        st.plotly_chart(fig_ts, use_container_width=True)
                        
                        # Display some statistics
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Taxa Atual", f"{time_series.iloc[-1]:.2f}%")
                        with col2:
                            st.metric("Taxa Média", f"{time_series.mean():.2f}%")
                        with col3:
                            st.metric("Taxa Máxima", f"{time_series.max():.2f}%")
                        with col4:
                            st.metric("Taxa Mínima", f"{time_series.min():.2f}%")
                        with col5:
                            st.metric("Desvio Padrão", f"{time_series.std():.2f}%")
                else:
                    st.warning("Não há dados de séries históricas disponíveis.")
                
                # 2. Calculate and display rate variations
                st.markdown("### 📉 Variações de Taxa")
                
                variations = calculate_td_rate_variations(products_df)
                
                if variations:
                    # Create columns for different periods
                    cols = st.columns(min(5, len(variations)))
                    
                    for i, (days_back, var_list) in enumerate(sorted(variations.items())):
                        if i < len(cols):
                            with cols[i]:
                                st.markdown(f"#### Variação de {days_back} Dia{'s' if days_back > 1 else ''}")
                                
                                # Create HTML table for this period
                                html_table = '<table style="width:100%; border-collapse: collapse; font-size: 12px;">'
                                html_table += '<thead><tr style="background-color: #0a0a0a;">'
                                html_table += '<th style="border: 1px solid #d4af37; padding: 8px; color: #d4af37;">Vencimento</th>'
                                html_table += '<th style="border: 1px solid #d4af37; padding: 8px; color: #d4af37;">Variação (p.p.)</th>'
                                html_table += '</tr></thead><tbody>'
                                
                                for maturity, variation in var_list:
                                    color = '#00ff00' if variation >= 0 else '#ff0000'
                                    arrow = '▲' if variation >= 0 else '▼'
                                    maturity_str = str(maturity)
                                    
                                    html_table += '<tr>'
                                    html_table += f'<td style="border: 1px solid #333; padding: 6px; text-align: center; background-color: #1a1a1a; color: #d4af37;">{maturity_str}</td>'
                                    html_table += f'<td style="border: 1px solid #333; padding: 6px; text-align: right; background-color: #1a1a1a; color: {color}; font-weight: bold;">{arrow} {abs(variation):.4f}%</td>'
                                    html_table += '</tr>'
                                
                                html_table += '</tbody></table>'
                                st.markdown(html_table, unsafe_allow_html=True)
                
                # 3. Display rates table
                st.markdown(f"### 📋 Tabela de Taxas - {selected_bond}")
                
                rates_table = create_td_table(products_df)
                if rates_table is not None:
                    st.dataframe(rates_table, use_container_width=True, height=400)
                else:
                    st.warning("Não foi possível criar a tabela de taxas.")
                
                # 4. Two-day comparison feature
                st.markdown("### 🔄 Comparação Entre Dois Dias")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    date1 = st.selectbox(
                        "Selecione o Dia Mais Antigo",
                        options=available_dates,
                        index=0,
                        format_func=lambda x: str(x),
                        key='td_comparison_date1'
                    )
                
                with col2:
                    date2 = st.selectbox(
                        "Selecione o Dia Mais Recente",
                        options=available_dates,
                        index=len(available_dates) - 1,
                        format_func=lambda x: str(x),
                        key='td_comparison_date2'
                    )
                
                if st.button("Gerar Comparação", use_container_width=False, key='td_comparison_btn'):
                    if date1 in products_df.index and date2 in products_df.index:
                        # Get data for selected dates
                        rates1 = products_df.loc[date1].dropna()
                        rates2 = products_df.loc[date2].dropna()
                        
                        # Find common maturities
                        common_maturities = sorted(set(rates1.index).intersection(set(rates2.index)))
                        
                        if common_maturities:
                            # Create comparison table
                            st.markdown(f"#### Comparação: {date1} vs {date2}")
                            
                            html_table = '<table style="width:100%; border-collapse: collapse; font-size: 13px;">'
                            html_table += '<thead><tr style="background-color: #0a0a0a;">'
                            html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Vencimento</th>'
                            html_table += f'<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">{date1}</th>'
                            html_table += f'<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">{date2}</th>'
                            html_table += '<th style="border: 1px solid #d4af37; padding: 10px; color: #d4af37;">Variação (p.p.)</th>'
                            html_table += '</tr></thead><tbody>'
                            
                            for maturity in common_maturities:
                                rate1 = rates1[maturity]
                                rate2 = rates2[maturity]
                                variation = rate2 - rate1
                                
                                maturity_str = str(maturity)
                                color = '#00ff00' if variation >= 0 else '#ff0000'
                                arrow = '▲' if variation >= 0 else '▼'
                                
                                html_table += '<tr>'
                                html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: center; background-color: #1a1a1a; color: #d4af37; font-weight: bold;">{maturity_str}</td>'
                                html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: right; background-color: #1a1a1a; color: #d4af37;">{rate1:.4f}%</td>'
                                html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: right; background-color: #1a1a1a; color: #d4af37;">{rate2:.4f}%</td>'
                                html_table += f'<td style="border: 1px solid #333; padding: 10px; text-align: right; background-color: #1a1a1a; color: {color}; font-weight: bold;">{arrow} {abs(variation):.4f}%</td>'
                                html_table += '</tr>'
                            
                            html_table += '</tbody></table>'
                            st.markdown(html_table, unsafe_allow_html=True)
                            
                            # Summary statistics
                            st.markdown("#### Estatísticas da Comparação")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            variations_list = [rates2[m] - rates1[m] for m in common_maturities]
                            avg_variation = np.mean(variations_list)
                            max_variation = max(variations_list)
                            min_variation = min(variations_list)
                            
                            with col1:
                                st.metric("Variação Média (p.p.)", f"{avg_variation:.4f}%")
                            with col2:
                                st.metric("Variação Máxima (p.p.)", f"{max_variation:.4f}%")
                            with col3:
                                st.metric("Variação Mínima (p.p.)", f"{min_variation:.4f}%")
                            with col4:
                                st.metric("Nº de Vencimentos", len(common_maturities))
                        else:
                            st.warning("Não há vencimentos comuns entre os dias selecionados.")
                    else:
                        st.error("Erro ao carregar dados dos dias selecionados.")
            else:
                st.warning(f"Não há dados disponíveis para {selected_bond}.")
        else:
            st.error("Não foi possível carregar os dados do Tesouro Direto. Verifique sua conexão.")

    with tab5:
        st.markdown("### 📰 Noticiário Renda Fixa")
        
        # Initialize Supabase client
        supabase_client = init_supabase_client()
        
        # Check if user is admin
        is_admin = st.session_state.get('user_logged_in') == 'admin'
        
        # Try to load saved data from database first
        saved_data = None
        if supabase_client:
            saved_data = load_noticiario_from_database(supabase_client)
        
        # Admin section - upload and save
        if is_admin:
            st.markdown("#### 🔐 Admin: Upload e Salvar Novo Noticiário")
            
            uploaded_file = st.file_uploader(
                "Escolha o arquivo HTML",
                type=['html', 'htm'],
                help="Arquivo HTML do noticiário semanal de renda fixa",
                key='admin_upload'
            )
            
            if uploaded_file is not None:
                try:
                    # Read HTML content
                    html_content = uploaded_file.read().decode('utf-8')
                    
                    # Parse the content
                    with st.spinner("📖 Processando o noticiário..."):
                        daily_reports = parse_noticiario_renda_fixa(html_content)
                    
                    if daily_reports:
                        st.success(f"✅ {len(daily_reports)} dia(s) de notícias encontrado(s)!")
                        
                        # Save to database button
                        if supabase_client:
                            if st.button("💾 Salvar no Banco de Dados (Substituir Anterior)", type="primary"):
                                with st.spinner("Salvando..."):
                                    if save_noticiario_to_database(supabase_client, html_content, daily_reports, "admin"):
                                        st.success("✅ Noticiário salvo com sucesso! Todos os usuários verão esta versão.")
                                        # Force reload
                                        st.rerun()
                                    else:
                                        st.error("❌ Erro ao salvar no banco de dados.")
                        
                        # Display the reports
                        st.markdown("---")
                        st.markdown("#### Preview:")
                        display_noticiario_renda_fixa(daily_reports)
                    else:
                        st.warning("⚠️ Nenhuma notícia foi encontrada no arquivo. Verifique o formato.")
                        
                except Exception as e:
                    st.error(f"❌ Erro ao processar o arquivo: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
            
            st.markdown("---")
        
        # Display section - show saved data or instructions
        if saved_data:
            # Show when it was uploaded
            upload_date = saved_data.get('upload_date', '')
            if upload_date:
                upload_datetime = datetime.fromisoformat(upload_date)
                formatted_date = upload_datetime.strftime("%d/%m/%Y às %H:%M")
            
            # Display the saved reports
            parsed_data = saved_data.get('parsed_data', [])
            if parsed_data:
                display_noticiario_renda_fixa(parsed_data)
            else:
                st.warning("⚠️ Dados salvos não puderam ser exibidos.")
        else:
            # No saved data - show instructions
            if not is_admin:
                st.info("""
                **Aguardando primeiro upload**
                
                O administrador ainda não fez upload do noticiário semanal.
                Assim que disponível, as notícias aparecerão aqui automaticamente.
                """)
            else:
                st.info("""
                **Como usar:**
                1. Faça upload do arquivo HTML do noticiário semanal
                2. O sistema irá extrair automaticamente as notícias organizadas por dia
                3. Clique em "Salvar no Banco de Dados" para disponibilizar para todos os usuários
                4. Cada dia terá as seguintes seções:
                   - 📊 **Leitura da Curva**: Análise dos juros futuros
                   - 🌍 **Mercados Globais**: Notícias internacionais
                   - 📍 **Mercado Doméstico**: Notícias do Brasil
                   - 📰 **Noticiário Corporativo**: Notícias por setor
                """)
    

# MAIN APP LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

if not st.session_state.started or not st.session_state.authenticated:
    show_landing_page()
else:
    show_dashboard()





