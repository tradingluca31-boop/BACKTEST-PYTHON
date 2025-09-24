"""
🎯 BACKTEST ANALYZER PRO - Professional Trading Analytics
=======================================================
Application Streamlit pour analyser les backtests de trading quantitatif
Générer des rapports HTML professionnels avec QuantStats + métriques custom

Version: Streamlit Cloud Optimized
Auteur: tradingluca31-boop
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import warnings
import io
import base64

# Patch complet pour IPython sur Streamlit Cloud
import sys
from unittest.mock import MagicMock

# Mock complet d'IPython pour éviter toutes les erreurs
class MockIPython:
    def __getattr__(self, name):
        return MagicMock()

# Mock tous les modules IPython
sys.modules['IPython'] = MockIPython()
sys.modules['IPython.core'] = MagicMock()
sys.modules['IPython.display'] = MagicMock()
sys.modules['IPython.core.display'] = MagicMock()

# Import QuantStats avec patch IPython complet
try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except ImportError as e:
    st.warning(f"QuantStats non disponible: {e}")
    QUANTSTATS_AVAILABLE = False

warnings.filterwarnings('ignore')

class BacktestAnalyzerPro:
    """
    Analyseur de backtest professionnel avec style institutionnel
    """

    def __init__(self):
        self.returns = None
        self.equity_curve = None
        self.trades_data = None
        self.benchmark = None
        self.custom_metrics = {}

    def load_data(self, data_source, data_type='returns', file_extension=None):
        """
        Charger les données de backtest

        Args:
            data_source: DataFrame, CSV path ou données
            data_type: 'returns', 'equity' ou 'trades'
            file_extension: Extension du fichier pour déterminer le format
        """
        try:
            if isinstance(data_source, str):
                # Fichier path
                if file_extension and file_extension.lower() in ['.xlsx', '.xls']:
                    df = pd.read_excel(data_source, index_col=0, parse_dates=True)
                elif file_extension and file_extension.lower() == '.html':
                    # Lire table HTML
                    tables = pd.read_html(data_source)
                    df = tables[0]  # Prendre la première table
                    df = df.set_index(df.columns[0])
                    df.index = pd.to_datetime(df.index)
                else:
                    df = pd.read_csv(data_source, index_col=0, parse_dates=True)
            elif hasattr(data_source, 'name'):
                # Uploaded file object
                file_name = data_source.name.lower()
                if file_name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(data_source, index_col=0, parse_dates=True)
                elif file_name.endswith('.html'):
                    # Lire table HTML depuis uploaded file
                    content = data_source.read().decode('utf-8')
                    tables = pd.read_html(content)
                    df = tables[0]  # Prendre la première table
                    df = df.set_index(df.columns[0])
                    df.index = pd.to_datetime(df.index)
                else:
                    df = pd.read_csv(data_source, index_col=0, parse_dates=True)
            elif isinstance(data_source, pd.DataFrame):
                df = data_source.copy()
            else:
                raise ValueError("Format de données non supporté")

            # Prendre la première colonne si DataFrame avec plusieurs colonnes
            if len(df.columns) > 1:
                data_series = df.iloc[:, 0]  # Première colonne
            else:
                data_series = df.squeeze()

            # S'assurer que c'est numérique
            data_series = pd.to_numeric(data_series, errors='coerce').dropna()

            if data_type == 'returns':
                self.returns = data_series
            elif data_type == 'equity':
                self.equity_curve = data_series
                # Calculer les returns depuis equity curve
                self.returns = self.equity_curve.pct_change().dropna()
            elif data_type == 'trades':
                self.trades_data = df
                # Si trades, créer des returns à partir des P&L
                # Simuler des returns basés sur les trades
                self.returns = data_series / 10000  # Normaliser (assume capital de 10k)

            return True

        except Exception as e:
            st.error(f"Erreur lors du chargement: {e}")
            return False

    def calculate_rr_ratio(self):
        """
        Calculer le R/R moyen par trade (métrique personnalisée)
        """
        if self.trades_data is None:
            # Estimation basée sur les returns si pas de trades détaillés
            positive_returns = self.returns[self.returns > 0]
            negative_returns = self.returns[self.returns < 0]

            if len(negative_returns) > 0 and len(positive_returns) > 0:
                avg_win = positive_returns.mean()
                avg_loss = abs(negative_returns.mean())
                rr_ratio = avg_win / avg_loss
            else:
                rr_ratio = 0
        else:
            # Calcul précis avec données de trades
            wins = self.trades_data[self.trades_data['PnL'] > 0]['PnL']
            losses = abs(self.trades_data[self.trades_data['PnL'] < 0]['PnL'])

            if len(losses) > 0 and len(wins) > 0:
                rr_ratio = wins.mean() / losses.mean()
            else:
                rr_ratio = 0

        self.custom_metrics['RR_Ratio'] = rr_ratio
        return rr_ratio

    def calculate_all_metrics(self):
        """
        Calculer toutes les métriques avec QuantStats (si disponible) ou implémentation custom
        """
        metrics = {}

        try:
            if QUANTSTATS_AVAILABLE:
                # Utiliser QuantStats si disponible
                metrics['CAGR'] = qs.stats.cagr(self.returns)
                metrics['Sharpe'] = qs.stats.sharpe(self.returns)
                metrics['Sortino'] = qs.stats.sortino(self.returns)
                metrics['Calmar'] = qs.stats.calmar(self.returns)
                metrics['Max_Drawdown'] = qs.stats.max_drawdown(self.returns)
                metrics['Volatility'] = qs.stats.volatility(self.returns)
                metrics['VaR'] = qs.stats.var(self.returns)
                metrics['CVaR'] = qs.stats.cvar(self.returns)
                metrics['Win_Rate'] = qs.stats.win_rate(self.returns)
                metrics['Profit_Factor'] = qs.stats.profit_factor(self.returns)
                metrics['Omega_Ratio'] = qs.stats.omega(self.returns)
                metrics['Recovery_Factor'] = qs.stats.recovery_factor(self.returns)
                metrics['Skewness'] = qs.stats.skew(self.returns)
                metrics['Kurtosis'] = qs.stats.kurtosis(self.returns)
            else:
                # Implémentation custom fallback
                returns = self.returns.dropna()

                if len(returns) == 0:
                    return {key: 0.0 for key in ['CAGR', 'Sharpe', 'Sortino', 'Max_Drawdown',
                                   'Win_Rate', 'Profit_Factor', 'RR_Ratio_Avg']}

                # CAGR (Compound Annual Growth Rate)
                total_return = (1 + returns).prod() - 1
                years = len(returns) / 252  # Assuming 252 trading days per year
                metrics['CAGR'] = (1 + total_return) ** (1/years) - 1 if years > 0 else 0

                # Volatilité annualisée
                metrics['Volatility'] = returns.std() * np.sqrt(252)

                # Sharpe Ratio
                excess_returns = returns.mean() * 252  # Annualized return
                metrics['Sharpe'] = excess_returns / metrics['Volatility'] if metrics['Volatility'] > 0 else 0

                # Sortino Ratio (downside deviation)
                negative_returns = returns[returns < 0]
                downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else metrics['Volatility']
                metrics['Sortino'] = excess_returns / downside_std if downside_std > 0 else 0

                # Max Drawdown
                cumulative_returns = (1 + returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                metrics['Max_Drawdown'] = abs(drawdown.min())

                # Calmar Ratio
                metrics['Calmar'] = metrics['CAGR'] / metrics['Max_Drawdown'] if metrics['Max_Drawdown'] > 0 else 0

                # Win Rate
                winning_trades = len(returns[returns > 0])
                total_trades = len(returns)
                metrics['Win_Rate'] = winning_trades / total_trades if total_trades > 0 else 0

                # Profit Factor
                gross_profits = returns[returns > 0].sum()
                gross_losses = abs(returns[returns < 0].sum())
                metrics['Profit_Factor'] = gross_profits / gross_losses if gross_losses > 0 else 0

                # VaR et autres métriques
                metrics['VaR'] = np.percentile(returns, 5)
                var_threshold = metrics['VaR']
                tail_losses = returns[returns <= var_threshold]
                metrics['CVaR'] = tail_losses.mean() if len(tail_losses) > 0 else metrics['VaR']

                try:
                    from scipy import stats as scipy_stats
                    metrics['Skewness'] = scipy_stats.skew(returns)
                    metrics['Kurtosis'] = scipy_stats.kurtosis(returns)
                except:
                    metrics['Skewness'] = 0
                    metrics['Kurtosis'] = 0

                metrics['Recovery_Factor'] = total_return / metrics['Max_Drawdown'] if metrics['Max_Drawdown'] > 0 else 0

                threshold = 0
                gains = returns[returns > threshold].sum()
                losses = abs(returns[returns <= threshold].sum())
                metrics['Omega_Ratio'] = gains / losses if losses > 0 else 0

            # Métrique personnalisée R/R (toujours calculée)
            metrics['RR_Ratio_Avg'] = self.calculate_rr_ratio()

        except Exception as e:
            st.warning(f"Erreur calcul métriques: {e}")
            # Métriques par défaut en cas d'erreur
            metrics = {key: 0.0 for key in ['CAGR', 'Sharpe', 'Sortino', 'Max_Drawdown',
                      'Win_Rate', 'Profit_Factor', 'RR_Ratio_Avg', 'Volatility', 'Calmar',
                      'VaR', 'CVaR', 'Skewness', 'Kurtosis', 'Recovery_Factor', 'Omega_Ratio']}

        return metrics

    def create_equity_curve_plot(self):
        """
        Graphique equity curve professionnel
        """
        if self.equity_curve is None:
            self.equity_curve = (1 + self.returns).cumprod()

        fig = go.Figure()

        # Equity curve principale
        fig.add_trace(go.Scatter(
            x=self.equity_curve.index,
            y=self.equity_curve.values,
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>'
        ))

        # Benchmark si disponible
        if self.benchmark is not None:
            fig.add_trace(go.Scatter(
                x=self.benchmark.index,
                y=self.benchmark.values,
                name='Benchmark',
                line=dict(color='#ff7f0e', width=1, dash='dash'),
                hovertemplate='<b>Date:</b> %{x}<br><b>Benchmark:</b> %{y:.2f}<extra></extra>'
            ))

        fig.update_layout(
            title={
                'text': 'Portfolio Equity Curve',
                'x': 0.5,
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            xaxis_title='Date',
            yaxis_title='Portfolio Value',
            template='plotly_white',
            hovermode='x unified',
            height=500
        )

        return fig

    def create_drawdown_plot(self):
        """
        Graphique des drawdowns (avec QuantStats si disponible)
        """
        try:
            if QUANTSTATS_AVAILABLE:
                # Utiliser QuantStats pour les drawdowns
                drawdown = qs.stats.to_drawdown_series(self.returns)
            else:
                # Calculer les drawdowns manuellement
                cumulative_returns = (1 + self.returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.3)',
                line=dict(color='red', width=1),
                name='Drawdown %',
                hovertemplate='<b>Date:</b> %{x}<br><b>Drawdown:</b> %{y:.2f}%<extra></extra>'
            ))

            fig.update_layout(
                title={
                    'text': 'Drawdown Periods',
                    'x': 0.5,
                    'font': {'size': 18, 'color': '#2c3e50'}
                },
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                template='plotly_white',
                height=400,
                yaxis=dict(ticksuffix='%')
            )

            return fig
        except Exception as e:
            st.warning(f"Erreur création graphique drawdown: {e}")
            return go.Figure()

    def create_monthly_heatmap(self):
        """
        Heatmap des rendements mensuels (sans QuantStats)
        """
        try:
            # Calculer les rendements mensuels manuellement
            monthly_returns = self.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

            # Créer une matrice année/mois
            monthly_df = monthly_returns.to_frame('returns')
            monthly_df['year'] = monthly_df.index.year
            monthly_df['month'] = monthly_df.index.month

            # Pivot pour créer la heatmap
            heatmap_data = monthly_df.pivot(index='year', columns='month', values='returns').fillna(0) * 100

            # Créer des labels pour les mois
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=[month_labels[i-1] if i-1 < len(month_labels) else f'{i:02d}' for i in heatmap_data.columns],
                y=heatmap_data.index,
                colorscale='RdYlGn',
                zmid=0,
                hovertemplate='<b>Year:</b> %{y}<br><b>Month:</b> %{x}<br><b>Return:</b> %{z:.2f}%<extra></extra>'
            ))

            fig.update_layout(
                title={
                    'text': 'Monthly Returns Heatmap (%)',
                    'x': 0.5,
                    'font': {'size': 18, 'color': '#2c3e50'}
                },
                xaxis_title='Month',
                yaxis_title='Year',
                template='plotly_white',
                height=400
            )

            return fig
        except Exception as e:
            st.warning(f"Erreur création heatmap: {e}")
            return go.Figure()

    def create_returns_distribution(self):
        """
        Distribution des rendements
        """
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=self.returns * 100,
            nbinsx=50,
            name='Returns Distribution',
            marker_color='skyblue',
            opacity=0.7
        ))

        fig.update_layout(
            title={
                'text': 'Returns Distribution',
                'x': 0.5,
                'font': {'size': 18, 'color': '#2c3e50'}
            },
            xaxis_title='Daily Returns (%)',
            yaxis_title='Frequency',
            template='plotly_white',
            height=400
        )

        return fig

    def generate_downloadable_report(self, metrics):
        """
        Générer un rapport HTML téléchargeable
        """
        try:
            # HTML simplifié pour téléchargement
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Backtest Report Professional</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        background-color: #f8f9fa;
                    }}
                    .header {{
                        text-align: center;
                        background: linear-gradient(135deg, #1e3c72, #2a5298);
                        color: white;
                        padding: 30px;
                        border-radius: 10px;
                        margin-bottom: 30px;
                    }}
                    .metrics-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 15px;
                        margin: 20px 0;
                    }}
                    .metric-card {{
                        background: white;
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        text-align: center;
                    }}
                    .metric-value {{
                        font-size: 24px;
                        font-weight: bold;
                        color: #2980b9;
                    }}
                    .metric-label {{
                        font-size: 14px;
                        color: #7f8c8d;
                        margin-top: 5px;
                    }}
                    .rr-highlight {{
                        background: linear-gradient(135deg, #f093fb, #f5576c);
                        color: white;
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🎯 BACKTEST REPORT PROFESSIONNEL</h1>
                    <h2>Trader Quantitatif Analysis</h2>
                    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>

                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{metrics.get('CAGR', 0):.2%}</div>
                        <div class="metric-label">CAGR</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.get('Sharpe', 0):.2f}</div>
                        <div class="metric-label">Sharpe Ratio</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.get('Max_Drawdown', 0):.2%}</div>
                        <div class="metric-label">Max Drawdown</div>
                    </div>
                    <div class="metric-card rr-highlight">
                        <div class="metric-value">{metrics.get('RR_Ratio_Avg', 0):.2f}</div>
                        <div class="metric-label">R/R Moyen par Trade</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.get('Win_Rate', 0):.2%}</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.get('Profit_Factor', 0):.2f}</div>
                        <div class="metric-label">Profit Factor</div>
                    </div>
                </div>

                <div style="margin-top: 30px; padding: 20px; background: white; border-radius: 10px;">
                    <h3>Toutes les Métriques</h3>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr style="background: #f8f9fa;">
                            <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Métrique</th>
                            <th style="padding: 10px; text-align: right; border: 1px solid #ddd;">Valeur</th>
                        </tr>
            """

            for key, value in metrics.items():
                if isinstance(value, float):
                    if 'Ratio' in key or key in ['CAGR', 'Max_Drawdown', 'Win_Rate', 'Volatility']:
                        formatted_value = f"{value:.2%}"
                    else:
                        formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)

                html_content += f"""
                        <tr>
                            <td style="padding: 10px; border: 1px solid #ddd;">{key.replace('_', ' ')}</td>
                            <td style="padding: 10px; text-align: right; border: 1px solid #ddd;">{formatted_value}</td>
                        </tr>
                """

            html_content += """
                    </table>
                </div>
            </body>
            </html>
            """

            return html_content
        except Exception as e:
            st.error(f"Erreur génération rapport: {e}")
            return None

def main():
    """
    Application Streamlit principale
    """
    st.set_page_config(
        page_title="Backtest Analyzer Pro",
        page_icon="🎯",
        layout="wide"
    )

    # CSS personnalisé
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            color: #1e3c72;
            text-align: center;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .subtitle {
            text-align: center;
            color: #2a5298;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            color: white;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .rr-metric {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🎯 BACKTEST ANALYZER PRO</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Professional Trading Analytics - Wall Street Quantitative Analysis - v2.1</p>', unsafe_allow_html=True)

    # Sidebar pour configuration
    with st.sidebar:
        st.header("📊 Configuration")

        # Upload de fichiers
        uploaded_file = st.file_uploader(
            "Upload fichier de backtest",
            type=['csv', 'xlsx', 'xls', 'html'],
            help="Formats supportés: CSV, Excel (xlsx/xls), HTML"
        )

        data_type = st.selectbox(
            "Type de données",
            ['returns', 'equity', 'trades'],
            help="returns: rendements quotidiens, equity: valeur portefeuille, trades: détail trades"
        )

        st.markdown("---")
        st.markdown("### Options d'affichage")
        show_charts = st.checkbox("Afficher tous les graphiques", value=True)
        show_advanced = st.checkbox("Métriques avancées", value=True)

    # Interface principale
    if uploaded_file is not None:
        try:
            # Initialiser l'analyseur
            analyzer = BacktestAnalyzerPro()

            # Charger les données selon le format
            file_name = uploaded_file.name.lower()
            try:
                if file_name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file, index_col=0)
                    # Essayer de parser les dates de l'index
                    try:
                        df.index = pd.to_datetime(df.index)
                    except:
                        pass
                elif file_name.endswith('.html'):
                    # Lire table HTML depuis uploaded file
                    content = uploaded_file.read().decode('utf-8')
                    tables = pd.read_html(content)
                    df = tables[0]  # Prendre la première table
                    df = df.set_index(df.columns[0])
                    try:
                        df.index = pd.to_datetime(df.index)
                    except:
                        pass
                    uploaded_file.seek(0)  # Reset file pointer
                else:
                    df = pd.read_csv(uploaded_file, index_col=0)
                    # Essayer de parser les dates de l'index
                    try:
                        df.index = pd.to_datetime(df.index)
                    except:
                        pass

            except Exception as e:
                st.error(f"Erreur lecture fichier: {e}")
                st.stop()

            if analyzer.load_data(df, data_type):
                st.success("✅ Données chargées avec succès!")

                # Afficher aperçu des données
                with st.expander("👀 Aperçu des données"):
                    st.dataframe(df.head(10))
                    st.write(f"**Nombre de lignes:** {len(df)}")
                    st.write(f"**Colonnes:** {list(df.columns)}")
                    try:
                        start_date = pd.to_datetime(df.index[0]).strftime('%Y-%m-%d')
                        end_date = pd.to_datetime(df.index[-1]).strftime('%Y-%m-%d')
                        st.write(f"**Période:** {start_date} à {end_date}")
                    except:
                        st.write(f"**Période:** {df.index[0]} à {df.index[-1]}")

                    # Debug des valeurs
                    if data_type == 'returns':
                        min_val = df.iloc[:,0].min()
                        max_val = df.iloc[:,0].max()
                        mean_val = df.iloc[:,0].mean()
                        st.write(f"**Returns stats:** Min={min_val:.6f}, Max={max_val:.6f}, Mean={mean_val:.6f}")
                    elif data_type == 'equity':
                        returns = df.iloc[:,0].pct_change().dropna()
                        min_val = df.iloc[:,0].min()
                        max_val = df.iloc[:,0].max()
                        ret_min = returns.min()
                        ret_max = returns.max()
                        ret_mean = returns.mean()
                        st.write(f"**Equity stats:** Min={min_val:.2f}, Max={max_val:.2f}")
                        st.write(f"**Returns from equity:** Min={ret_min:.6f}, Max={ret_max:.6f}, Mean={ret_mean:.6f}")

                    # Auto-détection du type de données
                    col_values = df.iloc[:,0]
                    if col_values.min() >= 0 and col_values.max() > 10:
                        st.info("💡 **Auto-détection:** Vos données ressemblent à une **equity curve** (valeurs > 10). Essayez le type 'equity'")
                    elif abs(col_values.min()) < 1 and abs(col_values.max()) < 1:
                        st.info("💡 **Auto-détection:** Vos données ressemblent à des **returns** (valeurs entre -1 et 1). Le type 'returns' est bon")
                    elif col_values.min() < 0 or col_values.max() > col_values.mean() * 2:
                        st.info("💡 **Auto-détection:** Vos données ressemblent à des **trades** (P&L). Essayez le type 'trades'")

                # Générer l'analyse
                if st.button("🚀 GÉNÉRER L'ANALYSE COMPLÈTE", type="primary"):
                    with st.spinner("Génération de l'analyse professionnelle..."):

                        # Calculer métriques
                        metrics = analyzer.calculate_all_metrics()

                        # Afficher métriques clés en cartes stylées
                        st.markdown("## 📈 Métriques Principales")

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>CAGR</h3>
                                <h2>{metrics.get('CAGR', 0):.2%}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>Sharpe Ratio</h3>
                                <h2>{metrics.get('Sharpe', 0):.2f}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>Max Drawdown</h3>
                                <h2>{metrics.get('Max_Drawdown', 0):.2%}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        with col4:
                            st.markdown(f"""
                            <div class="metric-card rr-metric">
                                <h3>🎯 R/R Moyen</h3>
                                <h2>{metrics.get('RR_Ratio_Avg', 0):.2f}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        # Métriques secondaires
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Win Rate", f"{metrics.get('Win_Rate', 0):.2%}")

                        with col2:
                            st.metric("Profit Factor", f"{metrics.get('Profit_Factor', 0):.2f}")

                        with col3:
                            st.metric("Sortino Ratio", f"{metrics.get('Sortino', 0):.2f}")

                        with col4:
                            st.metric("Volatilité", f"{metrics.get('Volatility', 0):.2%}")

                        if show_charts:
                            # Graphiques
                            st.markdown("## 📊 Visualisations")

                            st.subheader("📈 Equity Curve")
                            st.plotly_chart(analyzer.create_equity_curve_plot(), use_container_width=True)

                            col1, col2 = st.columns(2)

                            with col1:
                                st.subheader("📉 Drawdowns")
                                st.plotly_chart(analyzer.create_drawdown_plot(), use_container_width=True)

                            with col2:
                                st.subheader("📊 Distribution des Returns")
                                st.plotly_chart(analyzer.create_returns_distribution(), use_container_width=True)

                            st.subheader("🔥 Heatmap Rendements Mensuels")
                            st.plotly_chart(analyzer.create_monthly_heatmap(), use_container_width=True)

                        if show_advanced:
                            # Tableau complet des métriques
                            st.markdown("## 📋 Toutes les Métriques")

                            metrics_df = pd.DataFrame([
                                {'Métrique': k.replace('_', ' '), 'Valeur': f"{v:.4f}" if isinstance(v, float) else str(v)}
                                for k, v in metrics.items()
                            ])
                            st.dataframe(metrics_df, use_container_width=True)

                        # Générer et télécharger rapport
                        html_report = analyzer.generate_downloadable_report(metrics)

                        if html_report:
                            # Export CSV des métriques
                            csv_data = pd.DataFrame([metrics]).T.reset_index()
                            csv_data.columns = ['Métrique', 'Valeur']

                            # Créer les différents formats d'export
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

                            st.markdown("## Options de telechargement")

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.download_button(
                                    "CSV Metriques",
                                    data=csv_data.to_csv(index=False),
                                    file_name=f"metrics_{timestamp}.csv",
                                    mime="text/csv",
                                    type="primary"
                                )

                            with col2:
                                # Export Excel XML (MS Office Excel 2007) - Simplifié
                                excel_data = None
                                try:
                                    excel_buffer = io.BytesIO()
                                    csv_data.to_excel(excel_buffer, index=False, engine='openpyxl')
                                    excel_data = excel_buffer.getvalue()

                                    st.download_button(
                                        "Excel XML (MS Office)",
                                        data=excel_data,
                                        file_name=f"metrics_{timestamp}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        type="primary"
                                    )
                                except Exception as e:
                                    st.button("Excel Error", disabled=True)
                                    st.caption(f"Erreur: {str(e)[:50]}...")

                            with col3:
                                st.download_button(
                                    "HTML (Internet Explorer)",
                                    data=html_report,
                                    file_name=f"report_IE_{timestamp}.html",
                                    mime="text/html",
                                    type="primary"
                                )

        except Exception as e:
            st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
            st.exception(e)

    else:
        st.info("👆 Uploadez votre fichier CSV de backtest pour commencer l'analyse")

        # Instructions détaillées
        with st.expander("ℹ️ Instructions d'utilisation"):
            st.markdown("""
            ## 📋 Formats de fichiers supportés

            **Formats acceptés:**
            - **CSV** (.csv)
            - **Excel** (.xlsx, .xls) - MS Office Excel 2007+
            - **HTML** (.html) - Tables HTML

            **Structure du fichier:**
            - **Index:** Dates au format YYYY-MM-DD
            - **Colonnes:** Selon le type de données choisi

            ### Types de données supportés:

            **1. Returns (Rendements quotidiens)**
            ```
            Date,returns
            2023-01-01,0.01
            2023-01-02,-0.005
            2023-01-03,0.02
            ```

            **2. Equity (Valeur du portefeuille)**
            ```
            Date,equity
            2023-01-01,10000
            2023-01-02,10100
            2023-01-03,10050
            ```

            **3. Trades (Détail des trades)**
            ```
            Date,PnL
            2023-01-01,150
            2023-01-02,-75
            2023-01-03,200
            ```

            ### Notes pour formats spéciaux:
            - **Excel**: Première feuille utilisée, dates en colonne A
            - **HTML**: Première table trouvée dans le fichier

            ## 📊 Métriques calculées

            ### Standards QuantStats:
            - **CAGR** - Taux de croissance annuel composé
            - **Sharpe Ratio** - Ratio rendement/risque
            - **Sortino Ratio** - Sharpe ajusté downside
            - **Calmar Ratio** - CAGR/Max Drawdown
            - **Max Drawdown** - Perte maximale
            - **Win Rate** - Taux de trades gagnants
            - **Profit Factor** - Gains/Pertes bruts

            ### Métriques avancées:
            - **VaR/CVaR** - Value at Risk
            - **Omega Ratio** - Probabilité gains/pertes
            - **Recovery Factor** - Récupération après DD
            - **Skewness/Kurtosis** - Asymétrie/Aplatissement

            ### Métrique personnalisée:
            - **🎯 R/R Moyen** - Risk/Reward ratio par trade

            ## 🎯 Fonctionnalités
            - ✅ Analyse complète automatisée
            - ✅ Graphiques interactifs professionnels
            - ✅ Rapport HTML téléchargeable
            - ✅ Export CSV des métriques
            - ✅ Interface responsive et moderne
            """)

        # Exemple de données
        with st.expander("📝 Générer des données d'exemple"):
            st.markdown("**Créer un fichier CSV d'exemple pour tester l'application:**")

            if st.button("🎲 Générer données exemple"):
                # Générer des données de backtest simulées
                np.random.seed(42)
                dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
                returns = np.random.normal(0.0008, 0.02, len(dates))  # Returns quotidiens

                # Ajouter quelques tendances
                trend = np.linspace(0, 0.1, len(dates)) / 365
                returns += trend

                df_example = pd.DataFrame({
                    'returns': returns
                }, index=dates)

                st.download_button(
                    "📥 Télécharger exemple returns",
                    data=df_example.to_csv(),
                    file_name="example_backtest_returns.csv",
                    mime="text/csv"
                )

                # Générer equity curve
                equity = (1 + df_example['returns']).cumprod() * 10000
                df_equity = pd.DataFrame({
                    'equity': equity
                }, index=dates)

                st.download_button(
                    "📥 Télécharger exemple equity",
                    data=df_equity.to_csv(),
                    file_name="example_backtest_equity.csv",
                    mime="text/csv"
                )

                st.success("✅ Fichiers d'exemple générés! Téléchargez et uploadez pour tester.")

if __name__ == "__main__":
    main()