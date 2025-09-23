"""
🎯 BACKTEST ANALYZER CLAUDE V1 - Professional Trading Analytics
=============================================================
Trader quantitatif Wall Street - Script de backtesting institutionnel
Générer des rapports HTML professionnels avec QuantStats + métriques custom

Auteur: tradingluca31-boop
Version: 1.0
Date: 2025
"""

import pandas as pd
import numpy as np
import quantstats as qs
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import warnings
import io
import base64

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
        
    def load_data(self, data_source, data_type='returns'):
        """
        Charger les données de backtest
        
        Args:
            data_source: DataFrame, CSV path ou données
            data_type: 'returns', 'equity' ou 'trades'
        """
        try:
            if isinstance(data_source, str):
                df = pd.read_csv(data_source, index_col=0, parse_dates=True)
            elif isinstance(data_source, pd.DataFrame):
                df = data_source.copy()
            else:
                raise ValueError("Format de données non supporté")
                
            if data_type == 'returns':
                self.returns = df.squeeze()
            elif data_type == 'equity':
                self.equity_curve = df.squeeze()
                # Calculer les returns depuis equity curve
                self.returns = self.equity_curve.pct_change().dropna()
            elif data_type == 'trades':
                self.trades_data = df
                
            return True
            
        except Exception as e:
            st.error(f"Erreur lors du chargement: {e}")
            return False
    
    def calculate_rr_ratio(self):
        """
        Calculer le R/R moyen par trade (métrique personnalisée obligatoire)
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
        Calculer toutes les métriques via QuantStats + custom
        """
        metrics = {}
        
        # Métriques QuantStats standards
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
        
        # Métriques avancées
        metrics['Omega_Ratio'] = qs.stats.omega(self.returns)
        metrics['Recovery_Factor'] = qs.stats.recovery_factor(self.returns)
        metrics['Skewness'] = qs.stats.skew(self.returns)
        metrics['Kurtosis'] = qs.stats.kurtosis(self.returns)
        
        # Métrique personnalisée obligatoire
        metrics['RR_Ratio_Avg'] = self.calculate_rr_ratio()
        
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
        Graphique des drawdowns
        """
        drawdown = qs.stats.to_drawdown_series(self.returns)
        
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
    
    def create_monthly_heatmap(self):
        """
        Heatmap des rendements mensuels
        """
        monthly_rets = qs.utils.group_returns(self.returns, groupby='M') * 100
        
        # Restructurer pour heatmap
        heatmap_data = monthly_rets.unstack().fillna(0)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=[f'{month:02d}' for month in heatmap_data.columns],
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
    
    def create_metrics_table(self, metrics):
        """
        Tableau des métriques stylé
        """
        # Formater les métriques
        formatted_metrics = []
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'Ratio' in key or key in ['CAGR', 'Max_Drawdown', 'Volatility']:
                    formatted_value = f"{value:.2%}"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
                
            formatted_metrics.append({
                'Métrique': key.replace('_', ' '),
                'Valeur': formatted_value
            })
        
        df_metrics = pd.DataFrame(formatted_metrics)
        return df_metrics
    
    def generate_html_report(self, output_path='backtest_report.html'):
        """
        Générer le rapport HTML institutionnel complet
        """
        try:
            # Calculer métriques
            metrics = self.calculate_all_metrics()
            
            # Créer les graphiques
            equity_fig = self.create_equity_curve_plot()
            drawdown_fig = self.create_drawdown_plot()
            heatmap_fig = self.create_monthly_heatmap()
            dist_fig = self.create_returns_distribution()
            
            # Template HTML professionnel
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Professional Backtest Report - Claude V1</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{
                        font-family: 'Arial', sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: #f8f9fa;
                        color: #2c3e50;
                    }}
                    .header {{
                        text-align: center;
                        background: linear-gradient(135deg, #1e3c72, #2a5298);
                        color: white;
                        padding: 30px;
                        border-radius: 10px;
                        margin-bottom: 30px;
                    }}
                    .metrics-container {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 20px;
                        margin-bottom: 30px;
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
                    .chart-container {{
                        background: white;
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        margin-bottom: 30px;
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
                    <h2>Claude V1 - Trader Quantitatif Analysis</h2>
                    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="metrics-container">
                    <div class="metric-card">
                        <div class="metric-value">{metrics['CAGR']:.2%}</div>
                        <div class="metric-label">CAGR</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics['Sharpe']:.2f}</div>
                        <div class="metric-label">Sharpe Ratio</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics['Max_Drawdown']:.2%}</div>
                        <div class="metric-label">Max Drawdown</div>
                    </div>
                    <div class="metric-card rr-highlight">
                        <div class="metric-value">{metrics['RR_Ratio_Avg']:.2f}</div>
                        <div class="metric-label">R/R Moyen par Trade</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics['Win_Rate']:.2%}</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics['Profit_Factor']:.2f}</div>
                        <div class="metric-label">Profit Factor</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <div id="equity-chart"></div>
                </div>
                
                <div class="chart-container">
                    <div id="drawdown-chart"></div>
                </div>
                
                <div class="chart-container">
                    <div id="heatmap-chart"></div>
                </div>
                
                <div class="chart-container">
                    <div id="distribution-chart"></div>
                </div>
                
                <script>
                    Plotly.newPlot('equity-chart', {equity_fig.to_json()});
                    Plotly.newPlot('drawdown-chart', {drawdown_fig.to_json()});
                    Plotly.newPlot('heatmap-chart', {heatmap_fig.to_json()});
                    Plotly.newPlot('distribution-chart', {dist_fig.to_json()});
                </script>
            </body>
            </html>
            """
            
            # Sauvegarder le rapport
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_template)
                
            return output_path, metrics
            
        except Exception as e:
            st.error(f"Erreur génération rapport: {e}")
            return None, None

def main():
    """
    Application Streamlit principale
    """
    st.set_page_config(
        page_title="Backtest Analyzer Pro",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 BACKTEST ANALYZER PROFESSIONAL")
    st.subheader("Wall Street Quantitative Trading Analytics - Claude V1")
    
    # Sidebar pour configuration
    with st.sidebar:
        st.header("📊 Configuration")
        
        # Upload de fichiers
        uploaded_file = st.file_uploader(
            "Upload CSV de backtest",
            type=['csv'],
            help="Format: Date (index) + Returns/Equity column"
        )
        
        data_type = st.selectbox(
            "Type de données",
            ['returns', 'equity', 'trades']
        )
        
        benchmark_option = st.checkbox("Ajouter benchmark (S&P500)")
        
    # Interface principale
    if uploaded_file is not None:
        # Initialiser l'analyseur
        analyzer = BacktestAnalyzerPro()
        
        # Charger les données
        df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        
        if analyzer.load_data(df, data_type):
            st.success("✅ Données chargées avec succès!")
            
            # Afficher aperçu des données
            with st.expander("👀 Aperçu des données"):
                st.dataframe(df.head())
                
            # Générer l'analyse
            if st.button("🚀 GÉNÉRER LE RAPPORT COMPLET", type="primary"):
                with st.spinner("Génération du rapport institutionnel..."):
                    
                    # Calculer métriques
                    metrics = analyzer.calculate_all_metrics()
                    
                    # Afficher métriques clés
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("CAGR", f"{metrics['CAGR']:.2%}")
                        st.metric("Sharpe Ratio", f"{metrics['Sharpe']:.2f}")
                        
                    with col2:
                        st.metric("Max Drawdown", f"{metrics['Max_Drawdown']:.2%}")
                        st.metric("Sortino Ratio", f"{metrics['Sortino']:.2f}")
                        
                    with col3:
                        st.metric("Win Rate", f"{metrics['Win_Rate']:.2%}")
                        st.metric("Profit Factor", f"{metrics['Profit_Factor']:.2f}")
                        
                    with col4:
                        st.metric("🎯 R/R Moyen", f"{metrics['RR_Ratio_Avg']:.2f}", 
                                help="Métrique personnalisée Wall Street")
                        st.metric("Volatilité", f"{metrics['Volatility']:.2%}")
                    
                    # Graphiques
                    st.subheader("📈 Equity Curve")
                    st.plotly_chart(analyzer.create_equity_curve_plot(), use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📉 Drawdowns")
                        st.plotly_chart(analyzer.create_drawdown_plot(), use_container_width=True)
                        
                    with col2:
                        st.subheader("📊 Distribution")
                        st.plotly_chart(analyzer.create_returns_distribution(), use_container_width=True)
                    
                    st.subheader("🔥 Heatmap Mensuelle")
                    st.plotly_chart(analyzer.create_monthly_heatmap(), use_container_width=True)
                    
                    # Générer rapport HTML
                    report_path, _ = analyzer.generate_html_report("backtest_report_pro.html")
                    
                    if report_path:
                        st.success("🎉 Rapport HTML généré avec succès!")
                        
                        # Bouton de téléchargement
                        with open(report_path, 'rb') as f:
                            st.download_button(
                                "📥 TÉLÉCHARGER RAPPORT HTML",
                                data=f.read(),
                                file_name="backtest_report_professional.html",
                                mime="text/html"
                            )
    
    else:
        st.info("👆 Uploadez votre fichier CSV de backtest pour commencer l'analyse")
        
        # Instructions
        with st.expander("ℹ️ Instructions d'utilisation"):
            st.markdown("""
            **Format CSV requis:**
            - Index: Dates (format YYYY-MM-DD)
            - Colonnes: Returns (decimal) ou Equity values
            
            **Types de données supportés:**
            - `returns`: Rendements quotidiens (ex: 0.01 pour 1%)
            - `equity`: Valeur du portefeuille (ex: 1000, 1050, etc.)
            - `trades`: Détail des trades avec colonnes PnL
            
            **Métriques générées:**
            - Toutes les métriques QuantStats professionnelles
            - **R/R moyen par trade** (métrique personnalisée)
            - Rapport HTML institutionnel exportable
            """)

if __name__ == "__main__":
    main()