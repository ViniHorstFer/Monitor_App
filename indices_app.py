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
import yfinance as yf
import calendar
import time
import os

warnings.filterwarnings('ignore')

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
    
    elif source == 'b3':
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        # Folder to save CSV
        download_dir = os.path.join(os.getcwd(), "downloads")
        os.makedirs(download_dir, exist_ok=True)

        prefs = {
            "download.default_directory": download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        chrome_options.add_experimental_option("prefs", prefs)

        # ------------------ DRIVER ------------------
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        wait = WebDriverWait(driver, 20)

        try:
            # Step 1. Access IFIX Monthly Evolution page directly
            url = "https://sistemaswebb3-listados.b3.com.br/indexStatisticsPage/monthly-evolution/IFIX?language=pt-br"
            driver.get(url)

            # Step 2. Wait for and click "Download (período selecionado)"
            download_link = wait.until(EC.element_to_be_clickable(
                (By.XPATH, "//a[contains(@href, '/indexStatisticsPage/monthly-evolution/IFIX/monthly-evolution/IFIX')]")
            ))
            driver.execute_script("arguments[0].click();", download_link)
            time.sleep(10)  # wait for download to finish

            # Step 3. Find and load the CSV into pandas
            csv_files = [f for f in os.listdir(download_dir) if f.endswith(".csv")]
            if not csv_files:
                raise FileNotFoundError("No CSV file downloaded.")
            
            csv_path = os.path.join(download_dir, csv_files[0])
            ifix_df = pd.read_csv(csv_path, sep=';', encoding='latin1')
            ifix_df = ifix_df.iloc[1:, :]
            ifix_df.index = ifix_df.index.map(lambda x: f"{x[0]}/{x[1]}")
            ifix_df.index = pd.to_datetime(ifix_df.index)

            from pandas.tseries.offsets import BMonthEnd

            def get_last_trading_day(date):
                """Get last trading day for a given month date"""
                today = pd.Timestamp.today().normalize()
                
                # Get last business day of the month
                last_bday = date + BMonthEnd()
                
                # If it's the current month and last business day is in the future
                if last_bday.year == today.year and last_bday.month == today.month:
                    if last_bday >= today:
                        # Return last business day before today
                        return today - pd.offsets.BDay(1)
                
                return last_bday

            ifix_df.index = [get_last_trading_day(date) for date in ifix_df.index]

            ifix_df.columns = ['IFIX']

            ifix_df['IFIX'] = pd.to_numeric(
                                ifix_df['IFIX'].astype(str)
                                .str.replace('.', '', regex=False)
                                .str.replace(',', '.', regex=False),
                                errors='coerce'
                            )

            return ifix_df.loc[ifix_df.index > start_date]

        except Exception as e:
            st.error(f"Erro ao acessar IFIX.")
            return pd.DataFrame()
        
        finally:
            driver.quit()

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
        
        username_input = st.text_input("Usuário", key="login_username_input", placeholder="Digite seu usuário")
        password_input = st.text_input("Senha", type="password", key="login_password_input", placeholder="Digite sua senha")
        
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

def show_dashboard():
    st.title("📈 Painel de Índices de Mercado")
    
    # Create tabs
    tab1, tab3 = st.tabs(["Análise de Índices", "Tesouro Direto"])
    
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
    # TAB 3: TESOURO DIRETO
    # ═══════════════════════════════════════════════════════════════════════════
    
    with tab3:
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

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

if not st.session_state.started or not st.session_state.authenticated:
    show_landing_page()
else:
    show_dashboard()
