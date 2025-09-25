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
                # Pour MT5, utiliser les profits directement comme equity curve
                pnl_cumulative = data_series.cumsum()

                # Créer une equity curve réaliste
                initial_capital = 10000  # Capital initial par défaut
                self.equity_curve = initial_capital + pnl_cumulative

                # Calculer les returns depuis l'equity curve
                self.returns = self.equity_curve.pct_change().dropna()

                # Nettoyer les valeurs infinies/NaN
                import numpy as np
                self.returns = self.returns.replace([np.inf, -np.inf], np.nan).dropna()

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
            try:
                # Essayer de trouver la colonne de profits (PnL, profit, etc.)
                profit_col = None
                for col in self.trades_data.columns:
                    if col.lower() in ['pnl', 'profit', 'p&l', 'pl']:
                        profit_col = col
                        break

                if profit_col is not None:
                    wins = self.trades_data[self.trades_data[profit_col] > 0][profit_col]
                    losses = abs(self.trades_data[self.trades_data[profit_col] < 0][profit_col])

                    if len(losses) > 0 and len(wins) > 0:
                        rr_ratio = wins.mean() / losses.mean()
                    else:
                        rr_ratio = 0
                else:
                    # Fallback si pas de colonne trouvée
                    rr_ratio = 0
            except Exception:
                # En cas d'erreur, utiliser les returns
                positive_returns = self.returns[self.returns > 0]
                negative_returns = self.returns[self.returns < 0]

                if len(negative_returns) > 0 and len(positive_returns) > 0:
                    avg_win = positive_returns.mean()
                    avg_loss = abs(negative_returns.mean())
                    rr_ratio = avg_win / avg_loss
                else:
                    rr_ratio = 0

        self.custom_metrics['RR_Ratio'] = rr_ratio
        return rr_ratio

    def calculate_all_metrics(self, target_dd=None, target_profit=None, initial_capital=10000, target_profit_euro=None, target_profit_total_euro=None):
        """
        Calculer toutes les métriques avec QuantStats (si disponible) ou implémentation custom

        Args:
            target_dd: Drawdown target personnalisé (décimal, ex: 0.10 pour 10%)
            target_profit: Profit target annuel personnalisé (décimal, ex: 0.20 pour 20%)
            initial_capital: Capital initial en euros
            target_profit_euro: Profit target annuel en euros
            target_profit_total_euro: Profit target total en euros (sur toute la période)
        """
        metrics = {}

        try:
            # Vérifier que nous avons des returns valides
            if self.returns is None or len(self.returns) == 0:
                st.warning("⚠️ Aucun return calculé - vérifiez vos données")
                return {key: 0.0 for key in ['CAGR', 'Sharpe', 'Sortino', 'Max_Drawdown',
                          'Win_Rate', 'Profit_Factor', 'RR_Ratio_Avg', 'Volatility']}

            # Nettoyer les returns
            returns = self.returns.dropna()
            if len(returns) == 0:
                st.warning("⚠️ Tous les returns sont NaN - vérifiez vos données")
                return {key: 0.0 for key in ['CAGR', 'Sharpe', 'Sortino', 'Max_Drawdown',
                          'Win_Rate', 'Profit_Factor', 'RR_Ratio_Avg', 'Volatility']}

            # FORCER l'utilisation du fallback personnalisé pour les données de trading
            # QuantStats assume des données journalières ce qui donne des résultats faux
            if False: # Désactivé pour éviter les calculs incorrects
                try:
                    # Utiliser QuantStats si disponible (DÉSACTIVÉ)
                    metrics['CAGR'] = qs.stats.cagr(returns)
                    metrics['Sharpe'] = qs.stats.sharpe(returns)
                    metrics['Sortino'] = qs.stats.sortino(returns)
                    metrics['Calmar'] = qs.stats.calmar(returns)
                    metrics['Max_Drawdown'] = qs.stats.max_drawdown(returns)
                    metrics['Volatility'] = qs.stats.volatility(returns)
                    metrics['VaR'] = qs.stats.var(returns)
                    metrics['CVaR'] = qs.stats.cvar(returns)
                    metrics['Win_Rate'] = qs.stats.win_rate(returns)
                    metrics['Profit_Factor'] = qs.stats.profit_factor(returns)
                    metrics['Omega_Ratio'] = qs.stats.omega(returns)
                    metrics['Recovery_Factor'] = qs.stats.recovery_factor(returns)
                    metrics['Skewness'] = qs.stats.skew(returns)
                    metrics['Kurtosis'] = qs.stats.kurtosis(returns)
                except Exception as e:
                    st.warning(f"Erreur QuantStats: {e} - Utilisation fallback")
                    # Forcer l'utilisation du fallback
                    raise Exception("QuantStats failed")
            else:
                # Implémentation custom fallback
                returns = self.returns.dropna()

                if len(returns) == 0:
                    return {key: 0.0 for key in ['CAGR', 'Sharpe', 'Sortino', 'Max_Drawdown',
                                   'Win_Rate', 'Profit_Factor', 'RR_Ratio_Avg']}

                # CAGR (Compound Annual Growth Rate) - Corrigé pour données de trading
                try:
                    total_return = (1 + returns).prod() - 1
                    # Calculer la période réelle en années basée sur les dates de trade
                    time_period = (returns.index[-1] - returns.index[0]).days / 365.25
                    if time_period > 0 and total_return > -1:
                        metrics['CAGR'] = (1 + total_return) ** (1/time_period) - 1
                    else:
                        metrics['CAGR'] = total_return  # Si moins d'un an, return total
                except:
                    metrics['CAGR'] = 0

                # Calculs corrigés pour données de trading (pas journalières)
                # Calculer la fréquence de trading réelle
                time_period = (returns.index[-1] - returns.index[0]).days / 365.25
                trades_per_year = len(returns) / time_period if time_period > 0 else len(returns)

                # Volatilité (standard deviation des returns sans annualisation forcée)
                vol = returns.std()
                metrics['Volatility'] = vol

                # Return annualisé basé sur CAGR réel
                annual_return = metrics['CAGR']

                # Sharpe Ratio (excess return vs volatility) - simplifié
                metrics['Sharpe'] = annual_return / vol if vol > 0 else 0

                # Sortino Ratio (downside deviation)
                negative_returns = returns[returns < 0]
                downside_std = negative_returns.std() if len(negative_returns) > 0 else vol
                metrics['Sortino'] = annual_return / downside_std if downside_std > 0 else 0

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
                metrics['VaR'] = returns.quantile(0.05)
                var_threshold = metrics['VaR']
                tail_losses = returns[returns <= var_threshold]
                metrics['CVaR'] = tail_losses.mean() if len(tail_losses) > 0 else metrics['VaR']

                try:
                    from scipy import stats as scipy_stats
                    metrics['Skewness'] = scipy_stats.skew(returns)
                    metrics['Kurtosis'] = scipy_stats.kurtosis(returns)

                    # Monthly Returns Distribution
                    monthly_returns = self.returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
                    if len(monthly_returns) > 1:
                        metrics['Monthly_Volatility'] = monthly_returns.std()
                        metrics['Monthly_Skewness'] = scipy_stats.skew(monthly_returns.dropna())
                        metrics['Monthly_Kurtosis'] = scipy_stats.kurtosis(monthly_returns.dropna())
                    else:
                        metrics['Monthly_Volatility'] = 0
                        metrics['Monthly_Skewness'] = 0
                        metrics['Monthly_Kurtosis'] = 0

                except:
                    metrics['Skewness'] = 0
                    metrics['Kurtosis'] = 0
                    metrics['Monthly_Volatility'] = 0
                    metrics['Monthly_Skewness'] = 0
                    metrics['Monthly_Kurtosis'] = 0

                metrics['Recovery_Factor'] = total_return / metrics['Max_Drawdown'] if metrics['Max_Drawdown'] > 0 else 0

                threshold = 0
                gains = returns[returns > threshold].sum()
                losses = abs(returns[returns <= threshold].sum())
                metrics['Omega_Ratio'] = gains / losses if losses > 0 else 0

            # Métrique personnalisée R/R (toujours calculée)
            metrics['RR_Ratio_Avg'] = self.calculate_rr_ratio()

            # === NOUVELLES MÉTRIQUES POUR STRATEGY OVERVIEW ===

            # Log Return et Absolute Return
            if len(self.returns) > 0:
                total_return = (1 + self.returns).prod() - 1
                metrics['Log_Return'] = np.log(1 + total_return) if total_return > -1 else 0
                metrics['Absolute_Return'] = total_return
            else:
                metrics['Log_Return'] = 0
                metrics['Absolute_Return'] = 0

            # Alpha (excess return vs benchmark - ici on assume 0% benchmark)
            metrics['Alpha'] = metrics['CAGR']  # Alpha vs cash (0%)

            # Number of Trades
            metrics['Number_of_Trades'] = len(self.returns)

            # === RISK-ADJUSTED METRICS ===

            # Probabilistic Sharpe Ratio (estimation)
            if len(self.returns) > 1 and metrics['Volatility'] > 0:
                # Calcul approximatif du Probabilistic Sharpe Ratio
                n_observations = len(self.returns)
                sharpe = metrics['Sharpe']

                # Formule approximative pour PSR
                if sharpe > 0:
                    import math
                    # PSR basé sur distribution normale des returns
                    psr_stat = (sharpe * math.sqrt(n_observations - 1)) / math.sqrt(1 - sharpe**2/n_observations) if n_observations > 1 else 0
                    # Approximation: convertir en pourcentage de confiance
                    if sharpe >= 2:
                        psr = 0.95  # Très bon Sharpe
                    elif sharpe >= 1.5:
                        psr = 0.85  # Bon Sharpe
                    elif sharpe >= 1:
                        psr = 0.70  # Correct
                    else:
                        psr = max(0.50, 0.50 + 0.20 * sharpe)
                else:
                    psr = max(0.01, 0.50 + 0.15 * sharpe)  # Sharpe négatif

                metrics['Probabilistic_Sharpe_Ratio'] = psr
            else:
                metrics['Probabilistic_Sharpe_Ratio'] = 0.5

            # === DRAWDOWN METRICS ===

            if len(self.returns) > 0:
                # Calculer la courbe de cumul pour les drawdowns
                cumulative_returns = (1 + self.returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdowns = (cumulative_returns - running_max) / running_max

                # Max Drawdown (déjà calculé mais on s'assure)
                metrics['Max_Drawdown'] = abs(drawdowns.min())

                # Calculs de drawdown basés sur les JOURS CALENDAIRES
                # Identifier les périodes de drawdown (< 0)
                in_drawdown = drawdowns < 0
                if in_drawdown.any():
                    # Calculer les périodes de drawdown en JOURS entre les dates
                    drawdown_periods_days = []
                    current_start_date = None

                    for i, is_dd in enumerate(in_drawdown):
                        current_date = self.returns.index[i]

                        if is_dd and current_start_date is None:
                            # Début d'une période de drawdown
                            current_start_date = current_date
                        elif not is_dd and current_start_date is not None:
                            # Fin d'une période de drawdown
                            period_days = (current_date - current_start_date).days
                            drawdown_periods_days.append(period_days)
                            current_start_date = None

                    # Ajouter la dernière période si elle se termine par un drawdown
                    if current_start_date is not None:
                        period_days = (self.returns.index[-1] - current_start_date).days
                        drawdown_periods_days.append(period_days)

                    # Longest et Average Drawdown en jours
                    if drawdown_periods_days:
                        metrics['Longest_Drawdown'] = max(drawdown_periods_days)
                        metrics['Average_Drawdown_Days'] = int(sum(drawdown_periods_days) / len(drawdown_periods_days))
                    else:
                        metrics['Longest_Drawdown'] = 0
                        metrics['Average_Drawdown_Days'] = 0
                else:
                    metrics['Longest_Drawdown'] = 0
                    metrics['Average_Drawdown_Days'] = 0

                # Average Drawdown (moyenne des drawdowns négatifs en pourcentage)
                negative_drawdowns = drawdowns[drawdowns < 0]
                if len(negative_drawdowns) > 0:
                    metrics['Average_Drawdown_Pct'] = abs(negative_drawdowns.mean())
                else:
                    metrics['Average_Drawdown_Pct'] = 0
            else:
                metrics['Max_Drawdown'] = 0
                metrics['Longest_Drawdown'] = 0
                metrics['Average_Drawdown_Pct'] = 0
                metrics['Average_Drawdown_Days'] = 0

            # Métriques personnalisées selon les targets
            if target_dd is not None:
                actual_dd = metrics.get('Max_Drawdown', 0)
                metrics['DD_Target'] = target_dd
                metrics['DD_Respect'] = "✅ Respecté" if actual_dd <= target_dd else "❌ Dépassé"
                metrics['DD_Marge'] = (target_dd - actual_dd) / target_dd if target_dd > 0 else 0
                metrics['DD_Score'] = min(100, (target_dd - actual_dd) / target_dd * 100) if target_dd > 0 else 0

            if target_profit is not None and target_profit_euro is not None:
                actual_cagr = metrics.get('CAGR', 0)
                actual_profit_euro = actual_cagr * initial_capital

                metrics['Profit_Target'] = target_profit
                metrics['Profit_Target_Euro'] = target_profit_euro
                metrics['Profit_Actual_Euro'] = actual_profit_euro
                metrics['Profit_Atteint'] = "✅ Atteint" if actual_profit_euro >= target_profit_euro else "❌ Non atteint"
                metrics['Profit_Ratio'] = actual_profit_euro / target_profit_euro if target_profit_euro > 0 else 0
                metrics['Profit_Score'] = min(100, actual_profit_euro / target_profit_euro * 100) if target_profit_euro > 0 else 0

            # Métriques profit total
            if target_profit_total_euro is not None:
                # Calculer le profit total réalisé = (valeur finale - valeur initiale)
                if self.equity_curve is None:
                    self.equity_curve = (1 + self.returns).cumprod()

                total_return = (self.equity_curve.iloc[-1] - 1) if len(self.equity_curve) > 0 else 0
                actual_profit_total_euro = total_return * initial_capital

                metrics['Profit_Total_Target_Euro'] = target_profit_total_euro
                metrics['Profit_Total_Actual_Euro'] = actual_profit_total_euro
                metrics['Profit_Total_Atteint'] = "✅ Atteint" if actual_profit_total_euro >= target_profit_total_euro else "❌ Non atteint"
                metrics['Profit_Total_Ratio'] = actual_profit_total_euro / target_profit_total_euro if target_profit_total_euro > 0 else 0
                metrics['Profit_Total_Score'] = min(100, actual_profit_total_euro / target_profit_total_euro * 100) if target_profit_total_euro > 0 else 0

            # Métriques combinées si les deux targets sont définis
            if target_dd is not None and target_profit is not None and target_profit_euro is not None:
                dd_ok = metrics.get('Max_Drawdown', 0) <= target_dd
                profit_ok = metrics.get('Profit_Actual_Euro', 0) >= target_profit_euro

                if dd_ok and profit_ok:
                    metrics['Strategy_Status'] = "🎯 EXCELLENT"
                elif profit_ok:
                    metrics['Strategy_Status'] = "📈 PROFITABLE (DD élevé)"
                elif dd_ok:
                    metrics['Strategy_Status'] = "🛡️ CONSERVATEUR (Profit faible)"
                else:
                    metrics['Strategy_Status'] = "⚠️ À AMÉLIORER"

                # Score global
                dd_score = metrics.get('DD_Score', 0)
                profit_score = metrics.get('Profit_Score', 0)
                metrics['Global_Score'] = (dd_score + profit_score) / 2

        except Exception as e:
            st.warning(f"Erreur calcul métriques: {e}")
            # Métriques par défaut en cas d'erreur
            metrics = {key: 0.0 for key in ['CAGR', 'Sharpe', 'Sortino', 'Max_Drawdown',
                      'Win_Rate', 'Profit_Factor', 'RR_Ratio_Avg', 'Volatility', 'Calmar',
                      'VaR', 'CVaR', 'Skewness', 'Kurtosis', 'Recovery_Factor', 'Omega_Ratio']}

        return metrics

    def calculate_streaks(self):
        """
        Calculer les séries de gains et pertes maximales
        """
        if self.returns is None or len(self.returns) == 0:
            return {'max_winning_streak': 0, 'max_losing_streak': 0}

        returns = self.returns.dropna()
        if len(returns) == 0:
            return {'max_winning_streak': 0, 'max_losing_streak': 0}

        # Déterminer les gains et pertes
        winning_trades = returns > 0
        losing_trades = returns < 0

        # Calculer les séries
        max_winning_streak = 0
        max_losing_streak = 0
        current_winning_streak = 0
        current_losing_streak = 0

        for is_win, is_loss in zip(winning_trades, losing_trades):
            if is_win:
                current_winning_streak += 1
                current_losing_streak = 0
                max_winning_streak = max(max_winning_streak, current_winning_streak)
            elif is_loss:
                current_losing_streak += 1
                current_winning_streak = 0
                max_losing_streak = max(max_losing_streak, current_losing_streak)
            else:
                # Trade neutre (breakeven)
                current_winning_streak = 0
                current_losing_streak = 0

        return {
            'max_winning_streak': max_winning_streak,
            'max_losing_streak': max_losing_streak
        }

    def calculate_tail_and_outlier_ratios(self):
        """
        Calculer les ratios de queue et d'outliers
        """
        if self.returns is None or len(self.returns) == 0:
            return {
                'tail_ratio': 0.0,
                'outlier_win_ratio': 0.0,
                'outlier_loss_ratio': 0.0
            }

        returns = self.returns.dropna()
        if len(returns) == 0:
            return {
                'tail_ratio': 0.0,
                'outlier_win_ratio': 0.0,
                'outlier_loss_ratio': 0.0
            }

        # Calculer les statistiques de base
        mean_return = returns.mean()
        std_return = returns.std()

        # Définir les seuils d'outliers (par exemple, au-delà de 2 écarts-types)
        outlier_threshold = 2 * std_return

        # Séparer les gains et les pertes
        gains = returns[returns > 0]
        losses = returns[returns < 0]

        # 1. Tail Ratio : rapport entre les rendements extrêmes positifs et négatifs
        # Utiliser les percentiles 95% et 5%
        if len(returns) >= 20:  # Assez de données pour des percentiles fiables
            top_5_percent = np.percentile(returns, 95)
            bottom_5_percent = np.percentile(returns, 5)

            if bottom_5_percent != 0:
                tail_ratio = abs(top_5_percent / bottom_5_percent)
            else:
                tail_ratio = float('inf') if top_5_percent > 0 else 0.0
        else:
            # Pas assez de données, utiliser max/min
            max_return = returns.max()
            min_return = returns.min()
            if min_return != 0:
                tail_ratio = abs(max_return / min_return)
            else:
                tail_ratio = float('inf') if max_return > 0 else 0.0

        # 2. Outlier Win Ratio : rapport des gains extrêmes
        if len(gains) > 0:
            extreme_gains = gains[gains > (mean_return + outlier_threshold)]
            if len(gains) > 0:
                outlier_win_ratio = len(extreme_gains) / len(gains)
            else:
                outlier_win_ratio = 0.0
        else:
            outlier_win_ratio = 0.0

        # 3. Outlier Loss Ratio : rapport des pertes extrêmes
        if len(losses) > 0:
            extreme_losses = losses[losses < (mean_return - outlier_threshold)]
            if len(losses) > 0:
                outlier_loss_ratio = len(extreme_losses) / len(losses)
            else:
                outlier_loss_ratio = 0.0
        else:
            outlier_loss_ratio = 0.0

        # Limiter les valeurs extrêmes pour l'affichage
        if tail_ratio == float('inf'):
            tail_ratio = 999.0
        elif tail_ratio > 999:
            tail_ratio = 999.0

        return {
            'tail_ratio': tail_ratio,
            'outlier_win_ratio': outlier_win_ratio,
            'outlier_loss_ratio': outlier_loss_ratio
        }

    def calculate_average_wins_losses(self):
        """
        Calculer les moyennes des gains et pertes par période
        """
        if self.returns is None or len(self.returns) == 0:
            return {
                'avg_winning_month': 0.0,
                'avg_losing_month': 0.0,
                'avg_winning_trade': 0.0,
                'avg_losing_trade': 0.0
            }

        returns = self.returns.dropna()
        if len(returns) == 0:
            return {
                'avg_winning_month': 0.0,
                'avg_losing_month': 0.0,
                'avg_winning_trade': 0.0,
                'avg_losing_trade': 0.0
            }

        # Calculs mensuels
        try:
            monthly_returns = returns.resample('MS').apply(lambda x: (1 + x).prod() - 1)
            monthly_returns = monthly_returns.dropna()

            if len(monthly_returns) > 0:
                winning_months = monthly_returns[monthly_returns > 0]
                losing_months = monthly_returns[monthly_returns < 0]

                avg_winning_month = winning_months.mean() if len(winning_months) > 0 else 0.0
                avg_losing_month = losing_months.mean() if len(losing_months) > 0 else 0.0

                # Vérifier si les valeurs sont NaN et les remplacer par 0
                if pd.isna(avg_winning_month):
                    avg_winning_month = 0.0
                if pd.isna(avg_losing_month):
                    avg_losing_month = 0.0
            else:
                avg_winning_month = 0.0
                avg_losing_month = 0.0
        except Exception as e:
            # Si on ne peut pas calculer mensuellement, utiliser des moyennes simples
            winning_daily = returns[returns > 0]
            losing_daily = returns[returns < 0]

            avg_winning_month = winning_daily.mean() * 21 if len(winning_daily) > 0 else 0.0  # ~21 jours/mois
            avg_losing_month = losing_daily.mean() * 21 if len(losing_daily) > 0 else 0.0

        # Calculs par trade (utiliser les returns individuels comme proxy)
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]

        avg_winning_trade = winning_trades.mean() if len(winning_trades) > 0 else 0.0
        avg_losing_trade = losing_trades.mean() if len(losing_trades) > 0 else 0.0

        return {
            'avg_winning_month': avg_winning_month,
            'avg_losing_month': avg_losing_month,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade
        }

    def calculate_winning_rates(self):
        """
        Calculer les taux de réussite par période
        """
        if self.returns is None or len(self.returns) == 0:
            return {
                'winning_days': 0.0,
                'winning_months': 0.0,
                'winning_quarters': 0.0,
                'winning_years': 0.0,
                'win_rate': 0.0
            }

        returns = self.returns.dropna()
        if len(returns) == 0:
            return {
                'winning_days': 0.0,
                'winning_months': 0.0,
                'winning_quarters': 0.0,
                'winning_years': 0.0,
                'win_rate': 0.0
            }

        # Winning Days (trades individuels)
        winning_days_count = len(returns[returns > 0])
        total_days = len(returns[returns != 0])  # Exclure les jours neutres
        winning_days_rate = winning_days_count / total_days if total_days > 0 else 0.0

        # Winning Months
        try:
            monthly_returns = returns.resample('MS').apply(lambda x: (1 + x).prod() - 1)
            monthly_returns = monthly_returns.dropna()
            winning_months_count = len(monthly_returns[monthly_returns > 0])
            total_months = len(monthly_returns[monthly_returns != 0])
            winning_months_rate = winning_months_count / total_months if total_months > 0 else 0.0
        except:
            winning_months_rate = 0.0

        # Winning Quarters
        try:
            quarterly_returns = returns.resample('QS').apply(lambda x: (1 + x).prod() - 1)
            quarterly_returns = quarterly_returns.dropna()
            winning_quarters_count = len(quarterly_returns[quarterly_returns > 0])
            total_quarters = len(quarterly_returns[quarterly_returns != 0])
            winning_quarters_rate = winning_quarters_count / total_quarters if total_quarters > 0 else 0.0
        except:
            winning_quarters_rate = 0.0

        # Winning Years
        try:
            yearly_returns = returns.resample('YS').apply(lambda x: (1 + x).prod() - 1)
            yearly_returns = yearly_returns.dropna()
            winning_years_count = len(yearly_returns[yearly_returns > 0])
            total_years = len(yearly_returns[yearly_returns != 0])
            winning_years_rate = winning_years_count / total_years if total_years > 0 else 0.0
        except:
            winning_years_rate = 0.0

        # Win Rate global (identique aux winning days pour les trades)
        win_rate = winning_days_rate

        return {
            'winning_days': winning_days_rate,
            'winning_months': winning_months_rate,
            'winning_quarters': winning_quarters_rate,
            'winning_years': winning_years_rate,
            'win_rate': win_rate
        }

    def calculate_transaction_costs(self):
        """
        Calculer les coûts de transaction (estimation basée sur les données)
        """
        if self.returns is None or len(self.returns) == 0:
            return {
                'total_transaction_costs': 0.0,
                'commission_costs': 0.0,
                'swap_costs': 0.0
            }

        returns = self.returns.dropna()
        if len(returns) == 0:
            return {
                'total_transaction_costs': 0.0,
                'commission_costs': 0.0,
                'swap_costs': 0.0
            }

        # Compter le nombre de trades (positions non nulles)
        num_trades = len(returns[returns != 0])

        if num_trades == 0:
            return {
                'total_transaction_costs': 0.0,
                'commission_costs': 0.0,
                'swap_costs': 0.0
            }

        # Calculer les rendements bruts vs nets pour estimer les coûts
        gross_returns = returns.sum()  # Rendements bruts cumulés

        # Calculs plus réalistes basés sur les performances réelles
        total_return = gross_returns

        # Estimation plus conservative des coûts de trading
        # Forex/CFD typique : 0.05% à 0.2% par trade en commission/spread
        estimated_commission_per_trade = 0.0005  # 0.05% par trade (plus réaliste)
        total_commission_cost = num_trades * estimated_commission_per_trade

        # Swap plus réaliste : uniquement sur positions gardées overnight
        # Estimation basée sur les taux d'intérêt et la volatilité
        estimated_daily_swap = -0.00005  # -0.005% par jour (plus conservateur)
        avg_holding_days = 1.5  # Moyenne de jours par position
        overnight_ratio = 0.2  # 20% des positions sont overnight (plus conservateur)
        total_swap_cost = num_trades * overnight_ratio * estimated_daily_swap * avg_holding_days

        # Coûts totaux
        total_costs = total_commission_cost + abs(total_swap_cost)

        # Calculer en pourcentage du capital total échangé (plus réaliste)
        # Supposer un capital initial et calculer le turnover
        estimated_capital = 10000  # Capital de base estimé
        total_volume_traded = num_trades * estimated_capital * 0.1  # 10% du capital par trade moyen

        if total_volume_traded > 0:
            commission_percentage = (total_commission_cost * estimated_capital / total_volume_traded) * 100
            swap_percentage = (total_swap_cost * estimated_capital / total_volume_traded) * 100
            total_costs_percentage = (total_costs * estimated_capital / total_volume_traded) * 100
        else:
            # Fallback avec les vraies performances
            if abs(total_return) > 0.01:  # Si rendements significatifs
                commission_percentage = (total_commission_cost / abs(total_return)) * 100 * 0.1  # Réduire le facteur
                swap_percentage = (total_swap_cost / total_return) * 100 * 0.1
                total_costs_percentage = (total_costs / abs(total_return)) * 100 * 0.1
            else:
                commission_percentage = min(5.0, num_trades * 0.05)  # Max 5% ou 0.05% par trade
                swap_percentage = max(-2.0, num_trades * overnight_ratio * -0.01)  # Max -2%
                total_costs_percentage = commission_percentage + abs(swap_percentage)

        # Limiter à des valeurs réalistes pour le trading
        commission_percentage = max(0, min(10, commission_percentage))  # 0-10%
        swap_percentage = max(-5, min(2, swap_percentage))  # -5% à +2%
        total_costs_percentage = max(0, min(12, total_costs_percentage))  # 0-12%

        return {
            'total_transaction_costs': total_costs_percentage,
            'commission_costs': commission_percentage,
            'swap_costs': swap_percentage
        }

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
        Heatmap professionnelle des rendements mensuels avec calculs réels
        """
        try:
            if self.returns is None or len(self.returns) == 0:
                return go.Figure()

            # Calculer les rendements mensuels réels
            # Utiliser 'M' pour fin de mois et calculer correctement
            monthly_returns = self.returns.resample('M').apply(lambda x: (1 + x).prod() - 1 if len(x) > 0 else np.nan)

            # Créer DataFrame avec années et mois
            monthly_df = monthly_returns.to_frame('returns')
            monthly_df = monthly_df.dropna()  # Supprimer les NaN d'abord

            if len(monthly_df) == 0:
                st.warning("Pas assez de données pour créer la heatmap mensuelle")
                return go.Figure()

            monthly_df['year'] = monthly_df.index.year
            monthly_df['month'] = monthly_df.index.month

            # Créer la matrice pivot pour la heatmap
            heatmap_data = monthly_df.pivot(index='year', columns='month', values='returns')

            # Convertir en pourcentages
            heatmap_data = heatmap_data * 100

            # S'assurer que nous avons toutes les colonnes de 1 à 12
            for month in range(1, 13):
                if month not in heatmap_data.columns:
                    heatmap_data[month] = np.nan

            # Réorganiser les colonnes dans l'ordre chronologique
            heatmap_data = heatmap_data.reindex(columns=range(1, 13))

            # Labels des mois
            month_labels = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                           'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

            # Obtenir les années pour les labels y
            years = [str(year) for year in sorted(heatmap_data.index)]

            # Créer les annotations avec les valeurs
            annotations = []
            for i, year in enumerate(heatmap_data.index):
                for j, month in enumerate(range(1, 13)):
                    value = heatmap_data.loc[year, month]
                    if pd.notna(value):
                        # Couleur du texte basée sur la valeur
                        text_color = 'white' if abs(value) > 2 else 'black'
                        annotations.append(
                            dict(
                                x=j,
                                y=i,
                                text=f'{value:.2f}',
                                showarrow=False,
                                font=dict(color=text_color, size=10, family="Arial Black"),
                                xref="x",
                                yref="y"
                            )
                        )

            # Remplacer les NaN par une valeur spéciale pour les cases vides
            heatmap_data_display = heatmap_data.copy()

            # Créer une matrice pour les couleurs custom
            z_values = heatmap_data_display.values.copy()

            # Créer la heatmap
            fig = go.Figure(data=go.Heatmap(
                z=z_values,
                x=month_labels,
                y=years,  # Utiliser les années comme strings
                colorscale=[
                    [0.0, '#d73027'],    # Rouge foncé pour -8%
                    [0.2, '#fc8d59'],    # Rouge clair pour -4%
                    [0.35, '#fee08b'],   # Jaune pour -2%
                    [0.5, '#ffffbf'],    # Crème pour 0%
                    [0.65, '#d9ef8b'],   # Vert clair pour +2%
                    [0.8, '#91bfdb'],    # Bleu clair pour +4%
                    [1.0, '#4575b4']     # Bleu foncé pour +8%
                ],
                zmid=0,
                zmin=-8,
                zmax=8,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Return (%)", side="right"),
                    tickmode="linear",
                    tick0=-8,
                    dtick=2,
                    thickness=15,
                    len=0.7
                ),
                hovertemplate='<b>%{y}</b> - <b>%{x}</b><br><b>Return:</b> %{z:.2f}%<extra></extra>',
                showlegend=False,
                text=[[f'{val:.2f}' if pd.notna(val) else '' for val in row] for row in heatmap_data.values],
                texttemplate='%{text}',
                textfont={"size": 10, "color": "white", "family": "Arial Black"},
                connectgaps=False,  # Ne pas connecter les valeurs manquantes
                hoverongaps=False   # Pas de hover sur les cases vides
            ))

            # Ajouter des rectangles gris pour les cases vides
            for i, year in enumerate(heatmap_data.index):
                for j, month in enumerate(range(1, 13)):
                    value = heatmap_data.loc[year, month]
                    if pd.isna(value):
                        fig.add_shape(
                            type="rect",
                            x0=j-0.5, y0=i-0.5,
                            x1=j+0.5, y1=i+0.5,
                            fillcolor="#2d2d2d",
                            line=dict(color="#404040", width=1),
                            layer="below"
                        )

            # Style de la heatmap
            fig.update_layout(
                title={
                    'text': 'Metrics - Monthly Returns (%)',
                    'x': 0.5,
                    'font': {'size': 20, 'color': 'white', 'family': 'Arial Black'}
                },
                plot_bgcolor='#1e1e1e',
                paper_bgcolor='#1e1e1e',
                font=dict(color='white'),
                height=500,
                xaxis=dict(
                    title='',
                    tickfont=dict(size=12, color='white', family='Arial Black'),
                    side='bottom',
                    showgrid=False
                ),
                yaxis=dict(
                    title='',
                    tickfont=dict(size=12, color='white', family='Arial Black'),
                    showgrid=False,
                    autorange='reversed'  # Pour avoir les années dans l'ordre croissant de haut en bas
                ),
                margin=dict(l=80, r=100, t=80, b=50)
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

    def create_monthly_returns_distribution(self):
        """
        Distribution des rendements mensuels - calculs corrigés
        """
        try:
            # Vérifier que nous avons des données
            if self.returns is None or len(self.returns) == 0:
                return go.Figure()

            # Essayer différentes approches selon le type de données
            # D'abord essayer le regroupement mensuel standard
            monthly_returns = self.returns.resample('MS').apply(lambda x: (1 + x).prod() - 1)
            monthly_returns_clean = monthly_returns.dropna()

            # Si nous avons très peu de mois (< 12), essayer une approche différente
            if len(monthly_returns_clean) < 12:
                # Option 1: Essayer avec des fenêtres glissantes de 30 jours
                if len(self.returns) > 30:
                    # Créer des "rendements mensuels" avec des fenêtres glissantes
                    rolling_returns = []
                    for i in range(0, len(self.returns) - 30, 10):  # Tous les 10 jours
                        window_data = self.returns.iloc[i:i+30]
                        if len(window_data) == 30:
                            monthly_return = (1 + window_data).prod() - 1
                            rolling_returns.append(monthly_return)

                    if len(rolling_returns) > 0:
                        monthly_returns_clean = pd.Series(rolling_returns)

                # Option 2: Si toujours pas assez, utiliser des rendements hebdomadaires
                elif len(self.returns) > 7:
                    weekly_returns = self.returns.resample('W').apply(lambda x: (1 + x).prod() - 1)
                    weekly_returns_clean = weekly_returns.dropna()
                    if len(weekly_returns_clean) > 0:
                        # Simuler des rendements mensuels en multipliant par ~4.33
                        monthly_returns_clean = weekly_returns_clean * 4.33

            # Filtrer les valeurs extrêmes
            if len(monthly_returns_clean) > 0:
                monthly_returns_clean = monthly_returns_clean[
                    (monthly_returns_clean > -0.5) & (monthly_returns_clean < 2.0)
                ]

            if len(monthly_returns_clean) < 1:
                return go.Figure()

            monthly_returns_pct = monthly_returns_clean * 100  # Convertir en pourcentage

            # Statistiques réelles
            mean_return = monthly_returns_pct.mean()
            std_return = monthly_returns_pct.std()

            # Créer l'histogramme avec un nombre approprié de bins
            fig = go.Figure()

            # Calculer le nombre optimal de bins
            n_bins = min(30, max(10, int(len(monthly_returns_pct) / 2)))

            # Histogramme des rendements mensuels réels
            fig.add_trace(go.Histogram(
                x=monthly_returns_pct,
                nbinsx=n_bins,
                name='Monthly Returns',
                marker_color='#00d4aa',
                opacity=0.8,
                showlegend=False,
                autobinx=True
            ))

            # Ajouter la courbe normale lisse basée sur les statistiques réelles
            if len(monthly_returns_pct) > 3 and not np.isnan(std_return) and std_return > 0:
                try:
                    # Calculer la plage pour la courbe
                    data_min = monthly_returns_pct.min()
                    data_max = monthly_returns_pct.max()
                    data_range = data_max - data_min

                    # Étendre la plage de 20% de chaque côté
                    x_range = np.linspace(
                        data_min - data_range * 0.2,
                        data_max + data_range * 0.2,
                        300
                    )

                    # Fonction de densité de probabilité normale
                    normal_density = (1 / (std_return * np.sqrt(2 * np.pi))) * \
                                   np.exp(-0.5 * ((x_range - mean_return) / std_return) ** 2)

                    # Mettre à l'échelle pour correspondre à l'histogramme
                    hist_counts, _ = np.histogram(monthly_returns_pct, bins=n_bins)
                    max_hist_count = max(hist_counts) if len(hist_counts) > 0 else 1
                    max_density = max(normal_density) if len(normal_density) > 0 else 1

                    # Calculer le facteur d'échelle
                    scale_factor = max_hist_count / max_density if max_density > 0 else 1
                    scaled_density = normal_density * scale_factor

                    # Ajouter la courbe normale lisse
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=scaled_density,
                        mode='lines',
                        name='Normal Distribution',
                        line=dict(color='white', width=3),
                        showlegend=False,
                        hovertemplate='<extra></extra>'
                    ))
                except:
                    pass  # Si erreur dans le calcul de la courbe, continuer sans

            # Ajouter la ligne de moyenne
            if not np.isnan(mean_return) and not np.isinf(mean_return):
                fig.add_vline(
                    x=mean_return,
                    line=dict(color='red', dash='dash', width=2),
                    annotation_text=f"Mean: {mean_return:.1f}%"
                )

            # Calculer les périodes réelles
            if len(monthly_returns_clean) > 0:
                start_year = monthly_returns_clean.index[0].year
                end_year = monthly_returns_clean.index[-1].year
            else:
                start_year = 2018
                end_year = 2024

            # Titre avec info de debug pour comprendre les données
            title_text = f'Distribution of Monthly Returns<br><span style="font-size:14px">{start_year} - {end_year} | {len(monthly_returns_clean)} data points</span>'

            # Mise à jour du layout
            fig.update_layout(
                title={
                    'text': title_text,
                    'x': 0.5,
                    'font': {'size': 18, 'color': 'white'}
                },
                xaxis_title='',
                yaxis_title='Occurrences',
                template='plotly_dark',
                height=400,
                plot_bgcolor='#1a1a1a',
                paper_bgcolor='#1a1a1a',
                font=dict(color='white'),
                xaxis=dict(
                    gridcolor='#333333',
                    tickformat='.1f',
                    ticksuffix='%',
                    showgrid=True
                ),
                yaxis=dict(
                    gridcolor='#333333',
                    showgrid=True
                )
            )

            return fig

        except Exception as e:
            # En cas d'erreur, retourner un graphique vide plutôt que planter
            return go.Figure()

    def create_var_visualization(self, confidence_level=0.05):
        """
        Créer une visualisation VaR avec courbe normale et zone de risque
        """
        try:
            if self.returns is None or len(self.returns) == 0:
                return go.Figure()

            returns = self.returns.dropna()
            if len(returns) == 0:
                return go.Figure()

            # Calculer les statistiques du portefeuille
            mean_return = returns.mean()
            std_return = returns.std()

            # Calculer VaR au niveau de confiance spécifié (par défaut 5%)
            var_percentile = np.percentile(returns, confidence_level * 100)

            # VaR est exprimé comme une perte potentielle (valeur positive)
            var_loss = abs(var_percentile) if var_percentile < 0 else var_percentile

            # Créer la distribution normale pour visualisation
            x_min = min(returns.min(), mean_return - 4 * std_return)
            x_max = max(returns.max(), mean_return + 4 * std_return)
            x_range = np.linspace(x_min, x_max, 1000)

            # Distribution normale basée sur les statistiques réelles
            normal_pdf = (1 / (std_return * np.sqrt(2 * np.pi))) * \
                        np.exp(-0.5 * ((x_range - mean_return) / std_return) ** 2)

            # Créer le graphique
            fig = go.Figure()

            # Courbe de distribution normale complète
            fig.add_trace(go.Scatter(
                x=x_range * 100,  # Convertir en pourcentage
                y=normal_pdf,
                mode='lines',
                name='Distribution',
                line=dict(color='lightblue', width=2),
                fill='tozeroy',
                fillcolor='rgba(173, 216, 230, 0.3)',
                showlegend=False
            ))

            # Zone de risque VaR - seulement la queue gauche (5%)
            var_threshold_x = x_range[x_range <= var_percentile]
            var_threshold_y = (1 / (std_return * np.sqrt(2 * np.pi))) * \
                   np.exp(-0.5 * ((var_threshold_x - mean_return) / std_return) ** 2)

            fig.add_trace(go.Scatter(
                x=var_threshold_x * 100,
                y=var_threshold_y,
                mode='lines',
                name=f'{confidence_level*100}% VaR',
                line=dict(color='darkblue', width=2),
                fill='tozeroy',
                fillcolor='rgba(70, 130, 180, 0.7)',
                showlegend=False
            ))

            # Ligne verticale VaR
            fig.add_vline(
                x=var_percentile * 100,
                line=dict(color='red', dash='dash', width=3),
                annotation_text=f"VaR {confidence_level*100}%: {var_percentile*100:.1f}%"
            )

            # Ligne verticale moyenne
            fig.add_vline(
                x=mean_return * 100,
                line=dict(color='black', dash='dash', width=2),
                annotation_text=f"Mean: {mean_return*100:.1f}%"
            )

            # Calculer les valeurs monétaires (utiliser capital initial si disponible)
            # Estimer avec 10,000€ par défaut (plus réaliste pour trading individuel)
            portfolio_value = 10000
            var_loss_amount = portfolio_value * var_loss

            fig.update_layout(
                title={
                    'text': f'{confidence_level*100:.0f}% VaR<br><span style="font-size:14px">Portfolio Risk Assessment</span>',
                    'x': 0.5,
                    'font': {'size': 18, 'color': '#2c3e50'}
                },
                xaxis_title='Returns (%)',
                yaxis_title='Probability Density',
                template='plotly_white',
                height=400,
                showlegend=False,
                xaxis=dict(
                    tickformat='.1f',
                    ticksuffix='%'
                ),
                annotations=[
                    dict(
                        x=var_percentile * 100,
                        y=max(normal_pdf) * 0.4,
                        text=f"€{var_loss_amount:,.0f}<br>Max Loss<br>({var_loss*100:.1f}%)",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="red",
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="red",
                        font=dict(size=10)
                    ),
                    dict(
                        x=mean_return * 100 + std_return * 150,
                        y=max(normal_pdf) * 0.7,
                        text=f"€{portfolio_value:,.0f}<br>Portfolio Value",
                        showarrow=False,
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="gray",
                        font=dict(size=10)
                    ),
                    dict(
                        x=var_percentile * 100 - std_return * 50,
                        y=max(normal_pdf) * 0.15,
                        text=f"{confidence_level*100:.0f}% Probability<br>Tail Risk",
                        showarrow=False,
                        bgcolor="rgba(70, 130, 180, 0.8)",
                        font=dict(color="white", size=9)
                    )
                ]
            )

            return fig

        except Exception as e:
            return go.Figure()

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

        # Tutoriel interactif pour les types de données
        with st.expander("🎓 TUTORIEL - Comment choisir le type de données ?", expanded=False):
            st.markdown("### 🔍 Guide de sélection du type de données")

            # Tabs pour chaque type
            tab1, tab2, tab3 = st.tabs(["📈 Returns", "💼 Equity", "🎯 Trades"])

            with tab1:
                st.markdown("""
                #### 📈 **RETURNS (Rendements quotidiens)**

                **✅ Utilisez ce type si vos données contiennent :**
                - Rendements quotidiens exprimés en décimal (ex: 0.01 = 1%)
                - Valeurs généralement entre -0.20 et +0.20 (-20% à +20%)
                - Performance journalière de votre stratégie

                **💡 Exemples de valeurs :**
                ```
                Date        returns
                2024-01-01    0.0150   (gain de 1.5%)
                2024-01-02   -0.0075   (perte de 0.75%)
                2024-01-03    0.0220   (gain de 2.2%)
                ```

                **🎯 Parfait pour :**
                - Stratégies de trading algorithmique
                - Backtests MetaTrader, TradingView
                - Données de performance journalière
                """)

                if st.button("📥 Télécharger exemple Returns"):
                    import pandas as pd
                    import numpy as np
                    np.random.seed(42)
                    dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
                    returns = np.random.normal(0.001, 0.015, len(dates))
                    df_example = pd.DataFrame({'returns': returns}, index=dates)
                    st.download_button(
                        "💾 Fichier exemple Returns",
                        data=df_example.to_csv(),
                        file_name="exemple_returns.csv",
                        mime="text/csv"
                    )

            with tab2:
                st.markdown("""
                #### 💼 **EQUITY (Valeur du portefeuille)**

                **✅ Utilisez ce type si vos données contiennent :**
                - Valeur totale du portefeuille jour par jour
                - Montants en euros/dollars (ex: 10000, 10150, 9925...)
                - Évolution du capital au fil du temps

                **💡 Exemples de valeurs :**
                ```
                Date        equity
                2024-01-01  10000.00  (capital initial)
                2024-01-02  10150.75  (gain de 150.75€)
                2024-01-03  10075.25  (perte de 75.50€)
                ```

                **🎯 Parfait pour :**
                - Exports de courtiers (Interactive Brokers, etc.)
                - Suivi de compte de trading réel
                - Courbes d'équité MT4/MT5

                **⚡ L'app calculera automatiquement les returns !**
                """)

                if st.button("📥 Télécharger exemple Equity"):
                    np.random.seed(42)
                    dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
                    returns = np.random.normal(0.001, 0.015, len(dates))
                    equity = (1 + pd.Series(returns)).cumprod() * 10000
                    df_example = pd.DataFrame({'equity': equity}, index=dates)
                    st.download_button(
                        "💾 Fichier exemple Equity",
                        data=df_example.to_csv(),
                        file_name="exemple_equity.csv",
                        mime="text/csv"
                    )

            with tab3:
                st.markdown("""
                #### 🎯 **TRADES (Détail des trades)**

                **✅ Utilisez ce type si vos données contiennent :**
                - P&L de chaque trade individuel
                - Profits/pertes en euros/dollars
                - Historique trade par trade

                **💡 Exemples de valeurs :**
                ```
                Date        PnL
                2024-01-01  +125.50  (trade gagnant)
                2024-01-02   -85.25  (trade perdant)
                2024-01-03  +200.75  (trade gagnant)
                ```

                **🎯 Parfait pour :**
                - Exports détaillés de trades
                - Analysis trade par trade
                - Calcul précis du R/R ratio

                **⚡ L'app créera une equity curve à partir des trades !**
                """)

                if st.button("📥 Télécharger exemple Trades"):
                    np.random.seed(42)
                    dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')[:30]
                    trades_pnl = np.random.normal(15, 45, len(dates))
                    df_example = pd.DataFrame({'PnL': trades_pnl}, index=dates)
                    st.download_button(
                        "💾 Fichier exemple Trades",
                        data=df_example.to_csv(),
                        file_name="exemple_trades.csv",
                        mime="text/csv"
                    )

            # Guide de diagnostic
            st.markdown("---")
            st.markdown("### 🔬 **DIAGNOSTIC RAPIDE**")

            diagnostic_col1, diagnostic_col2 = st.columns(2)

            with diagnostic_col1:
                st.markdown("""
                **🟢 Vos valeurs sont entre -1 et +1 ?**
                → Utilisez **RETURNS**

                **🟢 Vos valeurs commencent autour de votre capital initial ?**
                → Utilisez **EQUITY**
                """)

            with diagnostic_col2:
                st.markdown("""
                **🟢 Vos valeurs sont des gains/pertes par trade ?**
                → Utilisez **TRADES**

                **❓ Pas sûr ?**
                → L'app fait de l'auto-détection en bas !
                """)

        st.markdown("---")

        st.markdown("---")
        st.markdown("### 🎯 Personnalisation Trading")

        # Section Drawdown personnalisé
        st.markdown("**Drawdown Target**")
        custom_dd_enabled = st.checkbox("Utiliser DD personnalisé", value=False)
        if custom_dd_enabled:
            target_dd = st.slider("Max Drawdown Target (%)", 1.0, 50.0, 10.0, 0.5)
            target_dd = target_dd / 100  # Convertir en décimal
        else:
            target_dd = None

        # Capital initial
        st.markdown("**Capital Initial**")
        initial_capital = st.number_input("Capital de départ (€)", min_value=100, max_value=10000000, value=10000, step=1000)

        # Section Profit personnalisé
        st.markdown("**Profit Targets**")
        custom_profit_enabled = st.checkbox("Utiliser Profit personnalisé", value=False)
        if custom_profit_enabled:
            # Profit annuel
            target_profit_euro = st.number_input("Profit Target Annuel (€)", min_value=100, max_value=1000000, value=2000, step=100)
            target_profit = target_profit_euro / initial_capital  # Convertir en ratio

            # Profit total
            target_profit_total_euro = st.number_input("Profit Target Total (€)", min_value=100, max_value=10000000, value=5000, step=500,
                                                      help="Profit total cible sur toute la période du backtest")
        else:
            target_profit = None
            target_profit_euro = None
            target_profit_total_euro = None

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
                import pandas as pd  # Import pandas au début

                if file_name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                elif file_name.endswith('.html'):
                    # Lire table HTML depuis uploaded file
                    content = uploaded_file.read().decode('utf-8')
                    tables = pd.read_html(content)
                    df = tables[0]  # Prendre la première table
                    uploaded_file.seek(0)  # Reset file pointer
                else:
                    df = pd.read_csv(uploaded_file)

                # Détecter le format MT5 (avec colonnes magic, symbol, type, etc.)
                if 'profit' in df.columns and 'time_close' in df.columns:
                    st.info("🎯 **Fichier MT5 détecté !** Conversion automatique en cours...")

                    # Convertir les timestamps MT5 en dates
                    df['time_close_dt'] = pd.to_datetime(df['time_close'], unit='s', errors='coerce')

                    # Créer DataFrame avec dates en index et profit en valeur
                    df_processed = df[['time_close_dt', 'profit']].copy()
                    df_processed = df_processed.dropna()
                    df_processed = df_processed.set_index('time_close_dt')
                    df_processed = df_processed.sort_index()
                    df = df_processed

                    st.success("✅ Conversion MT5 terminée ! Utilisez le type 'trades'")

                # Sinon, essayer le format standard avec dates en première colonne
                elif len(df.columns) > 1:
                    # Prendre la première colonne comme index
                    df = df.set_index(df.columns[0])
                    try:
                        df.index = pd.to_datetime(df.index)
                    except:
                        # Si ça ne marche pas, essayer avec les colonnes suivantes
                        pass

            except Exception as e:
                st.error(f"Erreur lecture fichier: {e}")
                st.exception(e)
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

                    # Auto-détection avancée du type de données
                    col_values = df.iloc[:,0]
                    min_val = col_values.min()
                    max_val = col_values.max()
                    mean_val = col_values.mean()
                    std_val = col_values.std()

                    st.markdown("### 🤖 Auto-détection Intelligence")

                    # Analyse statistique
                    detection_col1, detection_col2 = st.columns(2)

                    with detection_col1:
                        st.markdown("**📊 Statistiques de vos données:**")
                        st.write(f"• Min: {min_val:.6f}")
                        st.write(f"• Max: {max_val:.6f}")
                        st.write(f"• Moyenne: {mean_val:.6f}")
                        st.write(f"• Écart-type: {std_val:.6f}")

                    with detection_col2:
                        st.markdown("**🎯 Recommandation IA:**")

                        # Logique d'auto-détection améliorée
                        confidence = 0
                        recommendation = ""
                        reasons = []

                        # Test pour RETURNS
                        if abs(min_val) < 1 and abs(max_val) < 1 and abs(mean_val) < 0.1:
                            confidence += 80
                            recommendation = "RETURNS"
                            reasons = [
                                "✅ Valeurs entre -1 et +1",
                                "✅ Moyenne proche de 0",
                                "✅ Typique des rendements"
                            ]

                        # Test pour EQUITY
                        elif min_val >= 0 and max_val > 100 and mean_val > 1000:
                            confidence += 85
                            recommendation = "EQUITY"
                            reasons = [
                                "✅ Toutes valeurs positives",
                                "✅ Valeurs > 100 (capital)",
                                "✅ Croissance progressive typique"
                            ]

                        # Test pour TRADES
                        elif (min_val < 0 and max_val > abs(min_val) * 0.5) or (std_val > abs(mean_val) * 2):
                            confidence += 75
                            recommendation = "TRADES"
                            reasons = [
                                "✅ Mix gains/pertes",
                                "✅ Volatilité élevée",
                                "✅ Typique P&L trades"
                            ]

                        # Test alternatif pour EQUITY (valeurs moyennes)
                        elif min_val > 1000 and max_val > min_val * 1.1:
                            confidence += 70
                            recommendation = "EQUITY"
                            reasons = [
                                "✅ Valeurs > 1000€",
                                "✅ Progression positive",
                                "✅ Semble être un capital"
                            ]

                        # Affichage de la recommandation
                        if confidence >= 70:
                            if recommendation == "RETURNS":
                                st.success(f"🎯 **{recommendation}** ({confidence}% confiance)")
                            elif recommendation == "EQUITY":
                                st.success(f"💼 **{recommendation}** ({confidence}% confiance)")
                            elif recommendation == "TRADES":
                                st.success(f"🎯 **{recommendation}** ({confidence}% confiance)")

                            for reason in reasons:
                                st.write(reason)

                            if recommendation.lower() != data_type:
                                st.warning(f"⚠️ Vous avez sélectionné '{data_type}' mais l'IA recommande '{recommendation.lower()}'")
                        else:
                            st.info("🤔 **Détection incertaine** - Vérifiez le tutoriel ci-dessus")
                            st.write("• Données ambiguës")
                            st.write("• Consultez les exemples")

                    st.markdown("---")

                # Générer l'analyse
                if st.button("🚀 GÉNÉRER L'ANALYSE COMPLÈTE", type="primary"):
                    with st.spinner("Génération de l'analyse professionnelle..."):

                        # Calculer métriques
                        metrics = analyzer.calculate_all_metrics(target_dd, target_profit, initial_capital, target_profit_euro, target_profit_total_euro)

                        # Strategy Overview Section
                        st.markdown("## 🎯 Strategy Overview")

                        # Calculate strategy overview metrics
                        try:
                            # Debug: vérifier les données disponibles

                            # S'assurer que les returns existent et ne sont pas vides
                            if analyzer.returns is not None and len(analyzer.returns) > 0:
                                # Get date range
                                start_date = analyzer.returns.index[0]
                                end_date = analyzer.returns.index[-1]

                                # Calculate trading period in years
                                trading_period_years = (end_date - start_date).days / 365.25
                                start_date_str = start_date.strftime('%Y-%m-%d')
                                end_date_str = end_date.strftime('%Y-%m-%d')

                                # Calculate returns
                                total_return = (1 + analyzer.returns).prod() - 1
                                import math
                                log_return = math.log(1 + total_return) if total_return > -1 else 0

                                # Number of periods
                                num_periods = len(analyzer.returns)

                            elif analyzer.equity_curve is not None and len(analyzer.equity_curve) > 0:
                                # Fallback: utiliser equity_curve si returns n'est pas disponible
                                start_date = analyzer.equity_curve.index[0]
                                end_date = analyzer.equity_curve.index[-1]

                                trading_period_years = (end_date - start_date).days / 365.25
                                start_date_str = start_date.strftime('%Y-%m-%d')
                                end_date_str = end_date.strftime('%Y-%m-%d')

                                # Calculate returns from equity curve
                                equity_returns = analyzer.equity_curve.pct_change().dropna()
                                total_return = (analyzer.equity_curve.iloc[-1] / analyzer.equity_curve.iloc[0]) - 1
                                log_return = math.log(1 + total_return) if total_return > -1 else 0

                                num_periods = len(analyzer.equity_curve)

                            else:
                                # Aucune donnée disponible
                                trading_period_years = 0
                                start_date_str = "N/A"
                                end_date_str = "N/A"
                                total_return = 0
                                log_return = 0
                                num_periods = 0

                            # Number of trades
                            if analyzer.trades_data is not None:
                                num_trades = len(analyzer.trades_data)

                                # Average holding period (for trades data)
                                avg_holding_period = "1 day"  # Valeur par défaut
                                if 'time_open' in analyzer.trades_data.columns and 'time_close' in analyzer.trades_data.columns:
                                    try:
                                        open_times = pd.to_datetime(analyzer.trades_data['time_open'], unit='s')
                                        close_times = pd.to_datetime(analyzer.trades_data['time_close'], unit='s')
                                        holding_periods = close_times - open_times
                                        avg_holding = holding_periods.mean()
                                        if pd.notna(avg_holding):
                                            days = avg_holding.days
                                            seconds = avg_holding.seconds
                                            hours = seconds // 3600
                                            minutes = (seconds % 3600) // 60
                                            avg_holding_period = f"{days} days {hours:02d}:{minutes:02d}"
                                    except Exception as e:
                                        # Estimation basée sur le nombre de trades
                                        if num_trades > 100:
                                            avg_holding_period = "2-6 hours"
                                        elif num_trades > 50:
                                            avg_holding_period = "1 day"
                                        else:
                                            avg_holding_period = "1-3 days"
                                else:
                                    # Pas de timestamps, estimer selon le nombre de trades
                                    total_days = (end_date - start_date).days if trading_period_years > 0 else 365
                                    avg_trades_per_day = num_trades / total_days if total_days > 0 else 1

                                    if avg_trades_per_day > 10:
                                        avg_holding_period = "2-4 hours"
                                    elif avg_trades_per_day > 1:
                                        avg_holding_period = "4-12 hours"
                                    else:
                                        avg_holding_period = "1-3 days"
                            else:
                                num_trades = num_periods
                                # Estimation basée sur les returns
                                if analyzer.returns is not None and len(analyzer.returns) > 1000:
                                    avg_holding_period = "2-6 hours"  # Day trading
                                elif analyzer.returns is not None and len(analyzer.returns) > 252:
                                    avg_holding_period = "1 day"  # Daily trading
                                else:
                                    avg_holding_period = "1-3 days"  # Swing trading

                        except Exception as e:
                            st.error(f"Erreur calcul Strategy Overview: {e}")
                            trading_period_years = 0
                            start_date_str = "N/A"
                            end_date_str = "N/A"
                            total_return = 0
                            log_return = 0
                            num_trades = 0
                            avg_holding_period = "N/A"

                        # Display Strategy Overview in a styled box
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                    padding: 25px; border-radius: 15px; color: white; margin: 20px 0;">
                            <h3 style="text-align: center; margin: 0 0 20px 0;">📊 STRATEGY OVERVIEW</h3>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                                <div style="text-align: center;">
                                    <h4 style="margin: 5px 0; color: #e8f4f8;">Trading Period</h4>
                                    <h3 style="margin: 5px 0; color: white;">{trading_period_years:.1f} Years</h3>
                                </div>
                                <div style="text-align: center;">
                                    <h4 style="margin: 5px 0; color: #e8f4f8;">Start Period</h4>
                                    <h3 style="margin: 5px 0; color: white;">{start_date_str}</h3>
                                </div>
                                <div style="text-align: center;">
                                    <h4 style="margin: 5px 0; color: #e8f4f8;">End Period</h4>
                                    <h3 style="margin: 5px 0; color: white;">{end_date_str}</h3>
                                </div>
                                <div style="text-align: center;">
                                    <h4 style="margin: 5px 0; color: #e8f4f8;">Log Return</h4>
                                    <h3 style="margin: 5px 0; color: white;">{log_return:.2%}</h3>
                                </div>
                                <div style="text-align: center;">
                                    <h4 style="margin: 5px 0; color: #e8f4f8;">Absolute Return</h4>
                                    <h3 style="margin: 5px 0; color: white;">{total_return:.2%}</h3>
                                </div>
                                <div style="text-align: center;">
                                    <h4 style="margin: 5px 0; color: #e8f4f8;">Number of Trades</h4>
                                    <h3 style="margin: 5px 0; color: white;">{num_trades}</h3>
                                </div>
                            </div>
                            <div style="text-align: center; margin-top: 15px;">
                                <h4 style="margin: 5px 0; color: #e8f4f8;">Average Holding Period</h4>
                                <h3 style="margin: 5px 0; color: white;">{avg_holding_period}</h3>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Additional Strategy Metrics Section
                        try:

                            # Calculate additional metrics
                            if analyzer.returns is not None and len(analyzer.returns) > 0:
                                # Best and Worst periods
                                best_day = analyzer.returns.max()
                                worst_day = analyzer.returns.min()

                                # Best and Worst months
                                try:
                                    monthly_returns = analyzer.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                                    best_month = monthly_returns.max() if len(monthly_returns) > 0 else 0
                                    worst_month = monthly_returns.min() if len(monthly_returns) > 0 else 0
                                except Exception as e:
                                    best_month = worst_month = 0
                                    monthly_returns = pd.Series()

                                # Average periods
                                avg_return = analyzer.returns.mean()
                                avg_month = monthly_returns.mean() if len(monthly_returns) > 0 else 0

                                # Win/Loss streaks
                                wins = analyzer.returns > 0
                                losses = analyzer.returns < 0

                                # Calculate winning streak
                                win_streaks = []
                                current_streak = 0
                                for win in wins:
                                    if win:
                                        current_streak += 1
                                    else:
                                        if current_streak > 0:
                                            win_streaks.append(current_streak)
                                        current_streak = 0
                                if current_streak > 0:
                                    win_streaks.append(current_streak)

                                # Calculate losing streak
                                loss_streaks = []
                                current_streak = 0
                                for loss in losses:
                                    if loss:
                                        current_streak += 1
                                    else:
                                        if current_streak > 0:
                                            loss_streaks.append(current_streak)
                                        current_streak = 0
                                if current_streak > 0:
                                    loss_streaks.append(current_streak)

                                best_streak = max(win_streaks) if win_streaks else 0
                                worst_streak = max(loss_streaks) if loss_streaks else 0

                                # Positive/Negative periods
                                positive_periods = len([x for x in analyzer.returns if x > 0])
                                negative_periods = len([x for x in analyzer.returns if x < 0])
                                positive_pct = (positive_periods / len(analyzer.returns)) * 100 if len(analyzer.returns) > 0 else 0
                                negative_pct = (negative_periods / len(analyzer.returns)) * 100 if len(analyzer.returns) > 0 else 0

                                # Debug des valeurs calculées

                            elif analyzer.equity_curve is not None and len(analyzer.equity_curve) > 0:
                                # Fallback: utiliser equity_curve
                                equity_returns = analyzer.equity_curve.pct_change().dropna()
                                if len(equity_returns) > 0:
                                    # Best and Worst periods
                                    best_day = equity_returns.max()
                                    worst_day = equity_returns.min()

                                    # Best and Worst months
                                    monthly_returns = equity_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                                    best_month = monthly_returns.max() if len(monthly_returns) > 0 else 0
                                    worst_month = monthly_returns.min() if len(monthly_returns) > 0 else 0

                                    # Average periods
                                    avg_return = equity_returns.mean()
                                    avg_month = monthly_returns.mean() if len(monthly_returns) > 0 else 0

                                    # Win/Loss streaks
                                    wins = equity_returns > 0
                                    losses = equity_returns < 0

                                    # Calculate winning streak
                                    win_streaks = []
                                    current_streak = 0
                                    for win in wins:
                                        if win:
                                            current_streak += 1
                                        else:
                                            if current_streak > 0:
                                                win_streaks.append(current_streak)
                                            current_streak = 0
                                    if current_streak > 0:
                                        win_streaks.append(current_streak)

                                    # Calculate losing streak
                                    loss_streaks = []
                                    current_streak = 0
                                    for loss in losses:
                                        if loss:
                                            current_streak += 1
                                        else:
                                            if current_streak > 0:
                                                loss_streaks.append(current_streak)
                                            current_streak = 0
                                    if current_streak > 0:
                                        loss_streaks.append(current_streak)

                                    best_streak = max(win_streaks) if win_streaks else 0
                                    worst_streak = max(loss_streaks) if loss_streaks else 0

                                    # Positive/Negative periods
                                    positive_periods = len([x for x in equity_returns if x > 0])
                                    negative_periods = len([x for x in equity_returns if x < 0])
                                    positive_pct = (positive_periods / len(equity_returns)) * 100 if len(equity_returns) > 0 else 0
                                    negative_pct = (negative_periods / len(equity_returns)) * 100 if len(equity_returns) > 0 else 0
                                else:
                                    best_day = worst_day = 0
                                    best_month = worst_month = 0
                                    avg_return = avg_month = 0
                                    best_streak = worst_streak = 0
                                    positive_periods = negative_periods = 0
                                    positive_pct = negative_pct = 0
                            else:
                                best_day = worst_day = 0
                                best_month = worst_month = 0
                                avg_return = avg_month = 0
                                best_streak = worst_streak = 0
                                positive_periods = negative_periods = 0
                                positive_pct = negative_pct = 0

                        except Exception as e:
                            st.error(f"Erreur calcul métriques détaillées: {e}")
                            best_day = worst_day = 0
                            best_month = worst_month = 0
                            avg_return = avg_month = 0
                            best_streak = worst_streak = 0
                            positive_periods = negative_periods = 0
                            positive_pct = negative_pct = 0

                        # S'assurer que toutes les variables sont définies avant l'affichage
                        if 'best_day' not in locals():
                            best_day = worst_day = 0
                            best_month = worst_month = 0
                            avg_return = avg_month = 0
                            best_streak = worst_streak = 0
                            positive_periods = negative_periods = 0
                            positive_pct = negative_pct = 0

                        # Debug final des valeurs avant affichage

                        # === PERFORMANCES DÉTAILLÉES ===

                        # 1. PERFORMANCE MENSUELLE
                        st.markdown("### 📅 Performance Mensuelle")
                        col1, col2, col3 = st.columns(3)

                        # Calculs mensuels RÉELS avec vérifications robustes
                        try:
                            # Calculer les returns mensuels directement ici pour garantir qu'ils existent
                            if analyzer.returns is not None and len(analyzer.returns) > 0:
                                monthly_returns_calc = analyzer.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

                                if len(monthly_returns_calc) > 0:
                                    best_month_val = monthly_returns_calc.max()
                                    worst_month_val = monthly_returns_calc.min()
                                    avg_month_val = monthly_returns_calc.mean()
                                    positive_months = len([x for x in monthly_returns_calc if x > 0])
                                    negative_months = len([x for x in monthly_returns_calc if x < 0])
                                    total_months = len(monthly_returns_calc)
                                else:
                                    # Si pas assez de données pour les mois, utiliser les returns journaliers
                                    best_month_val = analyzer.returns.max()
                                    worst_month_val = analyzer.returns.min()
                                    avg_month_val = analyzer.returns.mean()
                                    positive_months = len([x for x in analyzer.returns if x > 0])
                                    negative_months = len([x for x in analyzer.returns if x < 0])
                                    total_months = len(analyzer.returns)
                            else:
                                # Aucune donnée disponible
                                best_month_val = worst_month_val = avg_month_val = 0
                                positive_months = negative_months = total_months = 0
                        except Exception as e:
                            # En cas d'erreur, essayer avec les variables existantes
                            try:
                                best_month_val = best_month if 'best_month' in locals() else 0
                                worst_month_val = worst_month if 'worst_month' in locals() else 0
                                avg_month_val = avg_month if 'avg_month' in locals() else 0
                                positive_months = len([x for x in monthly_returns if x > 0]) if 'monthly_returns' in locals() else 0
                                negative_months = len([x for x in monthly_returns if x < 0]) if 'monthly_returns' in locals() else 0
                                total_months = len(monthly_returns) if 'monthly_returns' in locals() else 0
                            except:
                                best_month_val = worst_month_val = avg_month_val = 0
                                positive_months = negative_months = total_months = 0

                        # Affichage avec métriques Streamlit natives (plus fiable)
                        with col1:
                            st.success("📈 **Meilleures Performances**")
                            st.metric("Meilleur Mois", f"{best_month_val:.2%}")
                            st.metric("Mois Positifs", f"{positive_months}")

                        with col2:
                            st.error("📉 **Pires Performances**")
                            st.metric("Pire Mois", f"{worst_month_val:.2%}")
                            st.metric("Mois Négatifs", f"{negative_months}")

                        with col3:
                            st.info("📊 **Moyennes**")
                            st.metric("Mois Moyen", f"{avg_month_val:.2%}")
                            st.metric("Total Mois", f"{total_months}")

                        st.markdown("---")

                        # 2. PERFORMANCE ANNUELLE
                        st.markdown("### 📆 Performance Annuelle")
                        col1, col2, col3 = st.columns(3)

                        # Calculs annuels RÉELS
                        try:
                            if analyzer.returns is not None and len(analyzer.returns) > 0:
                                yearly_returns_calc = analyzer.returns.resample('A').apply(lambda x: (1 + x).prod() - 1)

                                if len(yearly_returns_calc) > 0:
                                    best_year_val = yearly_returns_calc.max()
                                    worst_year_val = yearly_returns_calc.min()
                                    avg_year_val = yearly_returns_calc.mean()
                                    positive_years = len([x for x in yearly_returns_calc if x > 0])
                                    negative_years = len([x for x in yearly_returns_calc if x < 0])
                                    total_years = len(yearly_returns_calc)
                                else:
                                    # Si pas assez pour années, utiliser returns totaux comme année unique
                                    total_return = (1 + analyzer.returns).prod() - 1
                                    best_year_val = total_return
                                    worst_year_val = total_return
                                    avg_year_val = total_return
                                    positive_years = 1 if total_return > 0 else 0
                                    negative_years = 1 if total_return < 0 else 0
                                    total_years = 1
                            else:
                                best_year_val = worst_year_val = avg_year_val = 0
                                positive_years = negative_years = total_years = 0
                        except Exception as e:
                            best_year_val = worst_year_val = avg_year_val = 0
                            positive_years = negative_years = total_years = 0

                        # Affichage avec métriques Streamlit natives (plus fiable)
                        with col1:
                            st.success("📈 **Meilleures Performances**")
                            st.metric("Meilleure Année", f"{best_year_val:.2%}")
                            st.metric("Années Positives", f"{positive_years}")

                        with col2:
                            st.error("📉 **Pires Performances**")
                            st.metric("Pire Année", f"{worst_year_val:.2%}")
                            st.metric("Années Négatives", f"{negative_years}")

                        with col3:
                            st.info("📊 **Moyennes**")
                            st.metric("Année Moyenne", f"{avg_year_val:.2%}")
                            st.metric("Total Années", f"{total_years}")

                        st.markdown("---")

                        # 3. PERFORMANCE PAR TRADE
                        st.markdown("### 🎯 Performance par Trade")
                        col1, col2, col3 = st.columns(3)

                        # Calculs par trade RÉELS
                        try:
                            if analyzer.returns is not None and len(analyzer.returns) > 0:
                                best_trade_val = analyzer.returns.max()
                                worst_trade_val = analyzer.returns.min()
                                avg_trade_val = analyzer.returns.mean()
                                positive_trades = len([x for x in analyzer.returns if x > 0])
                                negative_trades = len([x for x in analyzer.returns if x < 0])
                                total_trades = len(analyzer.returns)
                            else:
                                best_trade_val = worst_trade_val = avg_trade_val = 0
                                positive_trades = negative_trades = total_trades = 0
                        except:
                            best_trade_val = worst_trade_val = avg_trade_val = 0
                            positive_trades = negative_trades = total_trades = 0

                        # Affichage avec métriques Streamlit natives
                        with col1:
                            st.success("📈 **Meilleures Performances**")
                            st.metric("Meilleur Trade", f"{best_trade_val:.2%}")
                            st.metric("Trades Gagnants", f"{positive_trades}")

                        with col2:
                            st.error("📉 **Pires Performances**")
                            st.metric("Pire Trade", f"{worst_trade_val:.2%}")
                            st.metric("Trades Perdants", f"{negative_trades}")

                        with col3:
                            st.info("📊 **Moyennes**")
                            st.metric("Trade Moyen", f"{avg_trade_val:.2%}")
                            st.metric("Total Trades", f"{total_trades}")

                        st.markdown("---")

                        # 4. PERFORMANCE TRIMESTRIELLE
                        st.markdown("### 🗓️ Performance Trimestrielle")
                        col1, col2, col3 = st.columns(3)

                        # Calculs trimestriels RÉELS
                        try:
                            if analyzer.returns is not None and len(analyzer.returns) > 0:
                                quarterly_returns_calc = analyzer.returns.resample('Q').apply(lambda x: (1 + x).prod() - 1)

                                if len(quarterly_returns_calc) > 0:
                                    best_quarter_val = quarterly_returns_calc.max()
                                    worst_quarter_val = quarterly_returns_calc.min()
                                    avg_quarter_val = quarterly_returns_calc.mean()
                                    positive_quarters = len([x for x in quarterly_returns_calc if x > 0])
                                    negative_quarters = len([x for x in quarterly_returns_calc if x < 0])
                                    total_quarters = len(quarterly_returns_calc)
                                else:
                                    # Si pas assez pour trimestres, utiliser returns totaux
                                    total_return = (1 + analyzer.returns).prod() - 1
                                    best_quarter_val = total_return
                                    worst_quarter_val = total_return
                                    avg_quarter_val = total_return
                                    positive_quarters = 1 if total_return > 0 else 0
                                    negative_quarters = 1 if total_return < 0 else 0
                                    total_quarters = 1
                            else:
                                best_quarter_val = worst_quarter_val = avg_quarter_val = 0
                                positive_quarters = negative_quarters = total_quarters = 0
                        except Exception as e:
                            best_quarter_val = worst_quarter_val = avg_quarter_val = 0
                            positive_quarters = negative_quarters = total_quarters = 0

                        # Affichage avec métriques Streamlit natives
                        with col1:
                            st.success("📈 **Meilleures Performances**")
                            st.metric("Meilleur Trimestre", f"{best_quarter_val:.2%}")
                            st.metric("Trimestres Positifs", f"{positive_quarters}")

                        with col2:
                            st.error("📉 **Pires Performances**")
                            st.metric("Pire Trimestre", f"{worst_quarter_val:.2%}")
                            st.metric("Trimestres Négatifs", f"{negative_quarters}")

                        with col3:
                            st.info("📊 **Moyennes**")
                            st.metric("Trimestre Moyen", f"{avg_quarter_val:.2%}")
                            st.metric("Total Trimestres", f"{total_quarters}")

                        # Expected Returns and VaR Section
                        st.markdown("### 🎯 Expected Returns and VaR")

                        try:

                            # S'assurer que les returns existent et ne sont pas vides
                            if analyzer.returns is not None and len(analyzer.returns) > 0:
                                # Expected Return par Trade (moyenne des returns)
                                expected_per_trade = analyzer.returns.mean()

                                # Pour calculer expected monthly/yearly, on a besoin de savoir la fréquence des trades
                                # Calculons la durée totale et le nombre de trades
                                total_days = (analyzer.returns.index[-1] - analyzer.returns.index[0]).days
                                num_trades = len(analyzer.returns)

                                if total_days > 0 and num_trades > 0:
                                    # Trades per day
                                    trades_per_day = num_trades / total_days

                                    # Expected daily return (en supposant que tous les trades ne se font pas chaque jour)
                                    expected_daily = expected_per_trade * trades_per_day

                                    # Expected monthly (21 jours de trading)
                                    expected_monthly = expected_daily * 21

                                    # Expected yearly (252 jours de trading)
                                    expected_yearly = expected_daily * 252
                                else:
                                    # Fallback: utiliser directement les moyennes sans compound
                                    expected_daily = expected_per_trade
                                    expected_monthly = expected_per_trade * 21
                                    expected_yearly = expected_per_trade * 252

                                # Risk of Ruin (calcul corrigé)
                                daily_vol = analyzer.returns.std()
                                if daily_vol > 0:
                                    # Calcul basé sur la probabilité de drawdown important
                                    negative_returns = analyzer.returns[analyzer.returns < 0]
                                    if len(negative_returns) > 0:
                                        # Probabilité d'avoir des trades perdants
                                        loss_probability = len(negative_returns) / len(analyzer.returns)

                                        # Risk of ruin basé sur le win rate et average win/loss
                                        winning_trades = analyzer.returns[analyzer.returns > 0]
                                        if len(winning_trades) > 0:
                                            avg_win = winning_trades.mean()
                                            avg_loss = abs(negative_returns.mean())

                                            # Formule Risk of Ruin classique adaptée
                                            if avg_win > 0:
                                                win_loss_ratio = avg_win / avg_loss
                                                win_rate = len(winning_trades) / len(analyzer.returns)

                                                # Risk of ruin simplifié: si win_rate < 50% et win/loss < 1
                                                if win_rate < 0.5 and win_loss_ratio < 1:
                                                    risk_of_ruin = min(0.8 * (1 - win_rate) * (1 - win_loss_ratio), 0.95)
                                                else:
                                                    # Stratégie profitable: risk of ruin faible
                                                    risk_of_ruin = max(0.05, 0.3 * (1 - win_rate))
                                            else:
                                                risk_of_ruin = 0.5
                                        else:
                                            # Que des trades perdants = 100% risk of ruin
                                            risk_of_ruin = 1.0
                                    else:
                                        # Aucun trade perdant = 0% risk of ruin
                                        risk_of_ruin = 0.0
                                else:
                                    risk_of_ruin = 0.0

                                # Daily VaR (5% VaR - perte maximale dans 95% des cas)
                                daily_var = analyzer.returns.quantile(0.05)


                            elif analyzer.equity_curve is not None and len(analyzer.equity_curve) > 0:
                                # Fallback: calculer à partir de equity_curve (ici c'est vraiment journalier)
                                equity_returns = analyzer.equity_curve.pct_change().dropna()
                                if len(equity_returns) > 0:
                                    expected_daily = equity_returns.mean()
                                    # Pour equity curve, on peut utiliser des moyennes simples car c'est journalier
                                    expected_monthly = expected_daily * 21
                                    expected_yearly = expected_daily * 252

                                    daily_vol = equity_returns.std()
                                    if daily_vol > 0:
                                        negative_returns = equity_returns[equity_returns < 0]
                                        if len(negative_returns) > 0:
                                            # Calculer Risk of Ruin basé sur equity returns
                                            winning_days = equity_returns[equity_returns > 0]
                                            if len(winning_days) > 0:
                                                avg_win = winning_days.mean()
                                                avg_loss = abs(negative_returns.mean())
                                                win_rate = len(winning_days) / len(equity_returns)

                                                if avg_win > 0:
                                                    win_loss_ratio = avg_win / avg_loss
                                                    if win_rate < 0.5 and win_loss_ratio < 1:
                                                        risk_of_ruin = min(0.6 * (1 - win_rate), 0.8)
                                                    else:
                                                        risk_of_ruin = max(0.05, 0.2 * (1 - win_rate))
                                                else:
                                                    risk_of_ruin = 0.5
                                            else:
                                                risk_of_ruin = 1.0
                                        else:
                                            risk_of_ruin = 0.0
                                    else:
                                        risk_of_ruin = 0.0

                                    daily_var = equity_returns.quantile(0.05)
                                else:
                                    expected_daily = expected_monthly = expected_yearly = 0
                                    risk_of_ruin = 0
                                    daily_var = 0
                            else:
                                expected_daily = expected_monthly = expected_yearly = 0
                                risk_of_ruin = 0
                                daily_var = 0

                        except Exception as e:
                            st.error(f"Erreur calcul VaR: {e}")
                            expected_daily = expected_monthly = expected_yearly = 0
                            risk_of_ruin = 0
                            daily_var = 0

                        # Display Expected Returns and VaR in a dark themed section
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                                    padding: 25px; border-radius: 15px; color: white; margin: 20px 0;">
                            <h3 style="text-align: center; margin: 0 0 20px 0; color: #ecf0f1;">🎯 Expected Returns and VaR</h3>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 20px;">
                                <div style="text-align: center; background: rgba(52, 152, 219, 0.2); padding: 15px; border-radius: 10px;">
                                    <h4 style="margin: 5px 0; color: #3498db;">Expected Daily %</h4>
                                    <h2 style="margin: 5px 0; color: #ecf0f1; font-size: 24px;">{expected_daily:.2%}</h2>
                                </div>
                                <div style="text-align: center; background: rgba(46, 204, 113, 0.2); padding: 15px; border-radius: 10px;">
                                    <h4 style="margin: 5px 0; color: #2ecc71;">Expected Monthly %</h4>
                                    <h2 style="margin: 5px 0; color: #ecf0f1; font-size: 24px;">{expected_monthly:.2%}</h2>
                                </div>
                                <div style="text-align: center; background: rgba(155, 89, 182, 0.2); padding: 15px; border-radius: 10px;">
                                    <h4 style="margin: 5px 0; color: #9b59b6;">Expected Yearly %</h4>
                                    <h2 style="margin: 5px 0; color: #ecf0f1; font-size: 24px;">{expected_yearly:.2%}</h2>
                                </div>
                                <div style="text-align: center; background: rgba(231, 76, 60, 0.2); padding: 15px; border-radius: 10px;">
                                    <h4 style="margin: 5px 0; color: #e74c3c;">Risk of Ruin</h4>
                                    <h2 style="margin: 5px 0; color: #ecf0f1; font-size: 24px;">{risk_of_ruin:.2%}</h2>
                                </div>
                                <div style="text-align: center; background: rgba(241, 196, 15, 0.2); padding: 15px; border-radius: 10px;">
                                    <h4 style="margin: 5px 0; color: #f1c40f;">Daily VaR</h4>
                                    <h2 style="margin: 5px 0; color: #ecf0f1; font-size: 24px;">{daily_var:.2%}</h2>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("---")

                        # === STRATEGY OVERVIEW SECTION ===
                        st.markdown("## 🎯 Strategy Overview")

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 20px; border-radius: 12px; text-align: center; color: white; margin: 5px;">
                                <h4 style="margin: 5px 0; font-size: 14px;">Log Return</h4>
                                <h2 style="margin: 10px 0; color: #4fc3f7;">{metrics.get('Log_Return', 0):.2%}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 20px; border-radius: 12px; text-align: center; color: white; margin: 5px;">
                                <h4 style="margin: 5px 0; font-size: 14px;">Absolute Return</h4>
                                <h2 style="margin: 10px 0; color: #4fc3f7;">{metrics.get('Absolute_Return', 0):.2%}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        with col3:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 20px; border-radius: 12px; text-align: center; color: white; margin: 5px;">
                                <h4 style="margin: 5px 0; font-size: 14px;">Alpha</h4>
                                <h2 style="margin: 10px 0; color: #4fc3f7;">{metrics.get('Alpha', 0):.2%}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        with col4:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 20px; border-radius: 12px; text-align: center; color: white; margin: 5px;">
                                <h4 style="margin: 5px 0; font-size: 14px;">Number of Trades</h4>
                                <h2 style="margin: 10px 0; color: #90a4ae;">{metrics.get('Number_of_Trades', 0)}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("---")

                        # === RISK-ADJUSTED METRICS SECTION ===
                        st.markdown("## ⚖️ Risk-Adjusted Metrics")

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%); padding: 20px; border-radius: 12px; text-align: center; color: white; margin: 5px;">
                                <h4 style="margin: 5px 0; font-size: 14px;">Sharpe Ratio</h4>
                                <h2 style="margin: 10px 0; color: #68d391;">{metrics.get('Sharpe', 0):.2f}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%); padding: 20px; border-radius: 12px; text-align: center; color: white; margin: 5px;">
                                <h4 style="margin: 5px 0; font-size: 14px;">Probabilistic Sharpe Ratio</h4>
                                <h2 style="margin: 10px 0; color: #4fc3f7;">{metrics.get('Probabilistic_Sharpe_Ratio', 0):.2%}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        with col3:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%); padding: 20px; border-radius: 12px; text-align: center; color: white; margin: 5px;">
                                <h4 style="margin: 5px 0; font-size: 14px;">Sortino Ratio</h4>
                                <h2 style="margin: 10px 0; color: #9f7aea;">{metrics.get('Sortino', 0):.2f}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        with col4:
                            calmar_color = "#f56565" if metrics.get('Calmar', 0) < 1 else "#68d391"
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%); padding: 20px; border-radius: 12px; text-align: center; color: white; margin: 5px;">
                                <h4 style="margin: 5px 0; font-size: 14px;">Calmar Ratio</h4>
                                <h2 style="margin: 10px 0; color: {calmar_color};">{metrics.get('Calmar', 0):.2f}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("---")

                        # === DRAWDOWNS SECTION ===
                        st.markdown("## 📉 Drawdowns")

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%); padding: 20px; border-radius: 12px; text-align: center; color: white; margin: 5px;">
                                <h4 style="margin: 5px 0; font-size: 14px;">Max Drawdown</h4>
                                <h2 style="margin: 10px 0; color: #4fc3f7;">{metrics.get('Max_Drawdown', 0):.2%}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%); padding: 20px; border-radius: 12px; text-align: center; color: white; margin: 5px;">
                                <h4 style="margin: 5px 0; font-size: 14px;">Longest Drawdown</h4>
                                <h2 style="margin: 10px 0; color: #f56565;">{metrics.get('Longest_Drawdown', 0)}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        with col3:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%); padding: 20px; border-radius: 12px; text-align: center; color: white; margin: 5px;">
                                <h4 style="margin: 5px 0; font-size: 14px;">Average Drawdown</h4>
                                <h2 style="margin: 10px 0; color: #4fc3f7;">{metrics.get('Average_Drawdown_Pct', 0):.2%}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        with col4:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%); padding: 20px; border-radius: 12px; text-align: center; color: white; margin: 5px;">
                                <h4 style="margin: 5px 0; font-size: 14px;">Average Drawdown Days</h4>
                                <h2 style="margin: 10px 0; color: #f56565;">{metrics.get('Average_Drawdown_Days', 0)}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("---")

                        # === RETURNS DISTRIBUTION SECTIONS ===
                        st.markdown("## 📊 Returns Distribution")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 12px; text-align: center; color: white; margin: 5px;">
                                <h4 style="margin: 5px 0; font-size: 14px;">Volatilité</h4>
                                <h2 style="margin: 10px 0; color: #4fc3f7;">{metrics.get('Volatility', 0):.2%}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 12px; text-align: center; color: white; margin: 5px;">
                                <h4 style="margin: 5px 0; font-size: 14px;">Skew</h4>
                                <h2 style="margin: 10px 0; color: #68d391;">{metrics.get('Skewness', 0):.3f}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        with col3:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 12px; text-align: center; color: white; margin: 5px;">
                                <h4 style="margin: 5px 0; font-size: 14px;">Kurtosis</h4>
                                <h2 style="margin: 10px 0; color: #fbb6ce;">{metrics.get('Kurtosis', 0):.3f}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("---")

                        st.markdown("## 📊 Monthly Returns Distribution")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #ff9a56 0%, #ff6b9d 100%); padding: 20px; border-radius: 12px; text-align: center; color: white; margin: 5px;">
                                <h4 style="margin: 5px 0; font-size: 14px;">Volatilité</h4>
                                <h2 style="margin: 10px 0; color: #4fc3f7;">{metrics.get('Monthly_Volatility', 0):.2%}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #ff9a56 0%, #ff6b9d 100%); padding: 20px; border-radius: 12px; text-align: center; color: white; margin: 5px;">
                                <h4 style="margin: 5px 0; font-size: 14px;">Skew</h4>
                                <h2 style="margin: 10px 0; color: #68d391;">{metrics.get('Monthly_Skewness', 0):.3f}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        with col3:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #ff9a56 0%, #ff6b9d 100%); padding: 20px; border-radius: 12px; text-align: center; color: white; margin: 5px;">
                                <h4 style="margin: 5px 0; font-size: 14px;">Kurtosis</h4>
                                <h2 style="margin: 10px 0; color: #fbb6ce;">{metrics.get('Monthly_Kurtosis', 0):.3f}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("---")

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
                                <h3>Win Rate</h3>
                                <h2>{metrics.get('Win_Rate', 0):.2%}</h2>
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

                        # Métriques secondaires (sans doublons)
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Profit Factor", f"{metrics.get('Profit_Factor', 0):.2f}")

                        with col2:
                            st.metric("Volatilité", f"{metrics.get('Volatility', 0):.2%}")

                        with col3:
                            st.metric("VaR (5%)", f"{metrics.get('VaR', 0):.2%}")

                        with col4:
                            st.metric("Recovery Factor", f"{metrics.get('Recovery_Factor', 0):.2f}")

                        # Section Streaks
                        st.markdown("---")
                        st.markdown("## 🔥 Streaks")

                        # Calculer les streaks
                        streaks = analyzer.calculate_streaks()

                        # Affichage des streaks en colonnes
                        streak_col1, streak_col2 = st.columns(2)

                        with streak_col1:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #28a745, #20c997);
                                        color: white; padding: 20px; border-radius: 10px; text-align: center;">
                                <h4 style="margin: 0; color: white;">Max Winning Streak</h4>
                                <h1 style="margin: 10px 0; color: white; font-size: 3em;">{}</h1>
                            </div>
                            """.format(streaks['max_winning_streak']), unsafe_allow_html=True)

                        with streak_col2:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #dc3545, #e83e8c);
                                        color: white; padding: 20px; border-radius: 10px; text-align: center;">
                                <h4 style="margin: 0; color: white;">Max Losing Streak</h4>
                                <h1 style="margin: 10px 0; color: white; font-size: 3em;">{}</h1>
                            </div>
                            """.format(streaks['max_losing_streak']), unsafe_allow_html=True)

                        st.markdown("---")

                        # Section Tail and Outlier Ratios
                        st.markdown("## 📊 Tail and Outlier Ratios")

                        # Calculer les ratios
                        tail_ratios = analyzer.calculate_tail_and_outlier_ratios()

                        # Affichage en trois colonnes
                        tail_col1, tail_col2, tail_col3 = st.columns(3)

                        with tail_col1:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #17a2b8, #138496);
                                        color: white; padding: 20px; border-radius: 10px; text-align: center;">
                                <h4 style="margin: 0; color: white;">Tail Ratio</h4>
                                <h1 style="margin: 10px 0; color: white; font-size: 2.5em;">{:.2f}</h1>
                            </div>
                            """.format(tail_ratios['tail_ratio']), unsafe_allow_html=True)

                        with tail_col2:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #28a745, #20c997);
                                        color: white; padding: 20px; border-radius: 10px; text-align: center;">
                                <h4 style="margin: 0; color: white;">Outlier Win Ratio</h4>
                                <h1 style="margin: 10px 0; color: white; font-size: 2.5em;">{:.2f}</h1>
                            </div>
                            """.format(tail_ratios['outlier_win_ratio']), unsafe_allow_html=True)

                        with tail_col3:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #dc3545, #c82333);
                                        color: white; padding: 20px; border-radius: 10px; text-align: center;">
                                <h4 style="margin: 0; color: white;">Outlier Loss Ratio</h4>
                                <h1 style="margin: 10px 0; color: white; font-size: 2.5em;">{:.2f}</h1>
                            </div>
                            """.format(tail_ratios['outlier_loss_ratio']), unsafe_allow_html=True)

                        # Explication des ratios
                        with st.expander("ℹ️ Explication des Tail and Outlier Ratios"):
                            st.markdown("""
                            **Tail Ratio**: Rapport entre les rendements extrêmes positifs (95e percentile) et négatifs (5e percentile).
                            Une valeur > 1 indique que les gains extrêmes sont plus importants que les pertes extrêmes.

                            **Outlier Win Ratio**: Proportion des gains qui sont considérés comme des outliers (> moyenne + 2σ).
                            Une valeur élevée indique des gains exceptionnels fréquents.

                            **Outlier Loss Ratio**: Proportion des pertes qui sont considérées comme des outliers (< moyenne - 2σ).
                            Une valeur élevée indique des pertes exceptionnelles fréquentes.
                            """)

                        st.markdown("---")

                        # Section Average Wins and Losses
                        st.markdown("## 💰 Average Wins and Losses")

                        # Calculer les moyennes
                        avg_stats = analyzer.calculate_average_wins_losses()

                        # Affichage en quatre colonnes
                        avg_col1, avg_col2, avg_col3, avg_col4 = st.columns(4)

                        with avg_col1:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #007bff, #0056b3);
                                        color: white; padding: 15px; border-radius: 8px; text-align: center;">
                                <h5 style="margin: 0; color: white; font-size: 0.9em;">Average Winning Month</h5>
                                <h1 style="margin: 5px 0; color: white; font-size: 1.8em;">{:.2%}</h1>
                            </div>
                            """.format(avg_stats['avg_winning_month']), unsafe_allow_html=True)

                        with avg_col2:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #dc3545, #c82333);
                                        color: white; padding: 15px; border-radius: 8px; text-align: center;">
                                <h5 style="margin: 0; color: white; font-size: 0.9em;">Average Losing Month</h5>
                                <h1 style="margin: 5px 0; color: white; font-size: 1.8em;">{:.2%}</h1>
                            </div>
                            """.format(avg_stats['avg_losing_month']), unsafe_allow_html=True)

                        with avg_col3:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #007bff, #0056b3);
                                        color: white; padding: 15px; border-radius: 8px; text-align: center;">
                                <h5 style="margin: 0; color: white; font-size: 0.9em;">Average Winning Trade</h5>
                                <h1 style="margin: 5px 0; color: white; font-size: 1.8em;">{:.2%}</h1>
                            </div>
                            """.format(avg_stats['avg_winning_trade']), unsafe_allow_html=True)

                        with avg_col4:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #dc3545, #c82333);
                                        color: white; padding: 15px; border-radius: 8px; text-align: center;">
                                <h5 style="margin: 0; color: white; font-size: 0.9em;">Average Losing Trade</h5>
                                <h1 style="margin: 5px 0; color: white; font-size: 1.8em;">{:.2%}</h1>
                            </div>
                            """.format(avg_stats['avg_losing_trade']), unsafe_allow_html=True)

                        st.markdown("---")

                        # Section Winning Rates
                        st.markdown("## 📈 Winning Rates")

                        # Calculer les taux de réussite
                        winning_stats = analyzer.calculate_winning_rates()

                        # Affichage en cinq colonnes
                        win_col1, win_col2, win_col3, win_col4, win_col5 = st.columns(5)

                        with win_col1:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #007bff, #0056b3);
                                        color: white; padding: 15px; border-radius: 8px; text-align: center;">
                                <h5 style="margin: 0; color: white; font-size: 0.9em;">Winning Days</h5>
                                <h1 style="margin: 5px 0; color: white; font-size: 1.6em;">{:.2%}</h1>
                            </div>
                            """.format(winning_stats['winning_days']), unsafe_allow_html=True)

                        with win_col2:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #007bff, #0056b3);
                                        color: white; padding: 15px; border-radius: 8px; text-align: center;">
                                <h5 style="margin: 0; color: white; font-size: 0.9em;">Winning Months</h5>
                                <h1 style="margin: 5px 0; color: white; font-size: 1.6em;">{:.2%}</h1>
                            </div>
                            """.format(winning_stats['winning_months']), unsafe_allow_html=True)

                        with win_col3:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #007bff, #0056b3);
                                        color: white; padding: 15px; border-radius: 8px; text-align: center;">
                                <h5 style="margin: 0; color: white; font-size: 0.9em;">Winning Quarters</h5>
                                <h1 style="margin: 5px 0; color: white; font-size: 1.6em;">{:.2%}</h1>
                            </div>
                            """.format(winning_stats['winning_quarters']), unsafe_allow_html=True)

                        with win_col4:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #007bff, #0056b3);
                                        color: white; padding: 15px; border-radius: 8px; text-align: center;">
                                <h5 style="margin: 0; color: white; font-size: 0.9em;">Winning Years</h5>
                                <h1 style="margin: 5px 0; color: white; font-size: 1.6em;">{:.2%}</h1>
                            </div>
                            """.format(winning_stats['winning_years']), unsafe_allow_html=True)

                        with win_col5:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #007bff, #0056b3);
                                        color: white; padding: 15px; border-radius: 8px; text-align: center;">
                                <h5 style="margin: 0; color: white; font-size: 0.9em;">Win Rate</h5>
                                <h1 style="margin: 5px 0; color: white; font-size: 1.6em;">{:.2%}</h1>
                            </div>
                            """.format(winning_stats['win_rate']), unsafe_allow_html=True)

                        st.markdown("---")

                        # Section Transaction Costs
                        st.markdown("## 💳 Transaction Costs")

                        # Calculer les coûts de transaction
                        transaction_costs = analyzer.calculate_transaction_costs()

                        # Affichage en trois colonnes
                        cost_col1, cost_col2, cost_col3 = st.columns(3)

                        with cost_col1:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #007bff, #0056b3);
                                        color: white; padding: 20px; border-radius: 10px; text-align: center;">
                                <h4 style="margin: 0; color: white;">Transaction Costs</h4>
                                <h1 style="margin: 10px 0; color: white; font-size: 2.5em;">{:.2f}%</h1>
                            </div>
                            """.format(transaction_costs['total_transaction_costs']), unsafe_allow_html=True)

                        with cost_col2:
                            commission_color = "#dc3545" if transaction_costs['commission_costs'] < 0 else "#28a745"
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, {commission_color}, #c82333);
                                        color: white; padding: 20px; border-radius: 10px; text-align: center;">
                                <h4 style="margin: 0; color: white;">Commission</h4>
                                <h1 style="margin: 10px 0; color: white; font-size: 2.5em;">{transaction_costs['commission_costs']:+.2f}%</h1>
                            </div>
                            """, unsafe_allow_html=True)

                        with cost_col3:
                            swap_color = "#dc3545" if transaction_costs['swap_costs'] < 0 else "#28a745"
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #007bff, #0056b3);
                                        color: white; padding: 20px; border-radius: 10px; text-align: center;">
                                <h4 style="margin: 0; color: white;">Swap</h4>
                                <h1 style="margin: 10px 0; color: white; font-size: 2.5em;">{transaction_costs['swap_costs']:+.2f}%</h1>
                            </div>
                            """, unsafe_allow_html=True)

                        # Explication des coûts
                        with st.expander("ℹ️ Explication des Transaction Costs"):
                            st.markdown(f"""
                            **Transaction Costs**: Estimation des coûts totaux de trading basés sur {len(analyzer.returns.dropna())} trades.

                            **Commission**: Coûts estimés des commissions de courtage (typiquement 0.1-0.5% par trade).

                            **Swap**: Coûts/gains estimés des positions overnight (frais de financement).

                            *Note: Ces valeurs sont des estimations basées sur des standards de marché.
                            Les coûts réels peuvent varier selon votre courtier.*
                            """)

                        st.markdown("---")

                        # Affichage des métriques personnalisées si définies
                        if target_dd is not None or (target_profit is not None and target_profit_euro is not None) or target_profit_total_euro is not None:
                            st.markdown("## 🎯 Analyse Personnalisée")

                            if target_dd is not None and target_profit is not None and target_profit_euro is not None:
                                # Affichage du statut global avec style
                                strategy_status = metrics.get('Strategy_Status', 'N/A')
                                global_score = metrics.get('Global_Score', 0)

                                if global_score >= 80:
                                    status_color = "success"
                                elif global_score >= 60:
                                    status_color = "warning"
                                else:
                                    status_color = "error"

                                st.markdown(f"""
                                <div style="text-align: center; padding: 20px; border-radius: 10px;
                                     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin: 20px 0;">
                                    <h2 style="margin: 0;">{strategy_status}</h2>
                                    <h3 style="margin: 10px 0;">Score Global: {global_score:.1f}/100</h3>
                                </div>
                                """, unsafe_allow_html=True)

                            # Métriques détaillées en colonnes
                            col1, col2 = st.columns(2)

                            with col1:
                                if target_dd is not None:
                                    st.markdown("### 🛡️ Analyse Drawdown")
                                    dd_respect = metrics.get('DD_Respect', 'N/A')
                                    dd_score = metrics.get('DD_Score', 0)
                                    dd_marge = metrics.get('DD_Marge', 0)

                                    st.metric(
                                        "Target DD",
                                        f"{target_dd:.1%}",
                                        help="Drawdown maximum acceptable défini"
                                    )
                                    st.metric(
                                        "DD Réalisé",
                                        f"{metrics.get('Max_Drawdown', 0):.2%}",
                                        delta=f"{dd_marge:.1%}" if dd_marge != 0 else None
                                    )
                                    st.metric("Statut DD", dd_respect)
                                    st.metric("Score DD", f"{dd_score:.1f}/100")

                            with col2:
                                if target_profit is not None and target_profit_euro is not None:
                                    st.markdown("### 💰 Analyse Profit (€)")
                                    profit_atteint = metrics.get('Profit_Atteint', 'N/A')
                                    profit_score = metrics.get('Profit_Score', 0)
                                    profit_ratio = metrics.get('Profit_Ratio', 0)
                                    actual_profit_euro = metrics.get('Profit_Actual_Euro', 0)

                                    st.metric(
                                        "Target Profit",
                                        f"{target_profit_euro:,.0f}€",
                                        help="Profit annuel cible en euros"
                                    )
                                    st.metric(
                                        "Profit Réalisé",
                                        f"{actual_profit_euro:,.0f}€",
                                        delta=f"{actual_profit_euro - target_profit_euro:+,.0f}€" if target_profit_euro != 0 else None
                                    )
                                    st.metric("Statut Profit", profit_atteint)
                                    st.metric("Score Profit", f"{profit_score:.1f}/100")

                                    # Affichage additionnel du CAGR pour référence
                                    st.caption(f"📊 CAGR équivalent: {metrics.get('CAGR', 0):.2%}")
                                    st.caption(f"💼 Capital initial: {initial_capital:,.0f}€")

                        # Affichage du profit total si défini
                        if target_profit_total_euro is not None:
                            st.markdown("### 🏆 Analyse Profit Total")

                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric(
                                    "Target Total",
                                    f"{target_profit_total_euro:,.0f}€",
                                    help="Profit total cible sur toute la période"
                                )

                            with col2:
                                actual_profit_total = metrics.get('Profit_Total_Actual_Euro', 0)
                                st.metric(
                                    "Profit Total Réalisé",
                                    f"{actual_profit_total:,.0f}€",
                                    delta=f"{actual_profit_total - target_profit_total_euro:+,.0f}€" if target_profit_total_euro != 0 else None
                                )

                            with col3:
                                total_profit_status = metrics.get('Profit_Total_Atteint', 'N/A')
                                st.metric("Statut Total", total_profit_status)

                            with col4:
                                total_profit_score = metrics.get('Profit_Total_Score', 0)
                                st.metric("Score Total", f"{total_profit_score:.1f}/100")

                            # Informations additionnelles
                            if len(analyzer.returns) > 0:
                                period_days = len(analyzer.returns)
                                period_years = period_days / 365.25
                                st.caption(f"📅 Période: {period_days} jours ({period_years:.1f} années)")

                                if actual_profit_total != 0 and period_years > 0:
                                    avg_profit_per_year = actual_profit_total / period_years
                                    st.caption(f"📈 Profit moyen par an: {avg_profit_per_year:,.0f}€")

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

                            st.subheader("📊 Distribution des Rendements Mensuels")
                            st.plotly_chart(analyzer.create_monthly_returns_distribution(), use_container_width=True)

                            st.subheader("📉 5% VaR Analysis")
                            st.plotly_chart(analyzer.create_var_visualization(), use_container_width=True)

                        if show_advanced:
                            # Tableau complet des métriques
                            st.markdown("## 📋 Toutes les Métriques")

                            metrics_df = pd.DataFrame([
                                {'Métrique': k.replace('_', ' '), 'Valeur': f"{v:.4f}" if isinstance(v, float) else str(v)}
                                for k, v in metrics.items()
                            ])
                            st.dataframe(metrics_df, use_container_width=True)

                            # Section détaillée des métriques
                            with st.expander("📚 Guide détaillé des métriques"):
                                st.markdown("""
                                ## 📊 **Guide Complet des Métriques Trading**

                                ### **🎯 Métriques de Performance**

                                **📈 CAGR (Compound Annual Growth Rate)**
                                - **Définition :** Taux de croissance annuel composé
                                - **Calcul :** (Valeur finale/Valeur initiale)^(1/années) - 1
                                - **Bon niveau :** > 10% excellent, > 20% exceptionnel
                                - **Usage :** Mesure la croissance annuelle moyenne

                                **⚡ Sharpe Ratio**
                                - **Définition :** Ratio rendement/risque ajusté
                                - **Calcul :** (Rendement - Taux sans risque) / Volatilité
                                - **Interprétation :** > 1 = bon, > 1.5 = excellent, > 2 = exceptionnel
                                - **Usage :** Compare l'efficacité risque/rendement

                                **🛡️ Sortino Ratio**
                                - **Définition :** Sharpe ajusté pour le downside uniquement
                                - **Calcul :** Rendement / Volatilité des pertes
                                - **Avantage :** Ne pénalise pas la volatilité haussière
                                - **Bon niveau :** > 1.5 = très bon

                                **🎪 Calmar Ratio**
                                - **Définition :** CAGR / Max Drawdown
                                - **Usage :** Mesure l'efficacité par rapport au pire scénario
                                - **Bon niveau :** > 1 = bon, > 3 = excellent
                                - **Avantage :** Focus sur le contrôle du risque

                                ### **📉 Métriques de Risque**

                                **💥 Max Drawdown**
                                - **Définition :** Perte maximale depuis un sommet
                                - **Calcul :** (Valeur max - Valeur min suivante) / Valeur max
                                - **Bon niveau :** < 10% = excellent, < 20% = acceptable
                                - **Critique :** Mesure le pire scénario vécu

                                **📊 Volatility**
                                - **Définition :** Écart-type annualisé des rendements
                                - **Calcul :** Écart-type × √252 jours
                                - **Interprétation :** Mesure l'amplitude des variations
                                - **Trading :** 15-40% = normal, > 50% = très risqué

                                **⚠️ VaR (Value at Risk)**
                                - **Définition :** Perte maximale probable (95% confiance)
                                - **Usage :** "5% de chance de perdre plus que X%"
                                - **Gestion risque :** Limite d'exposition quotidienne
                                - **Calcul :** 5ème percentile des rendements

                                **🔻 CVaR (Conditional VaR)**
                                - **Définition :** Perte moyenne au-delà du VaR
                                - **Usage :** "Quand les 5% pires jours arrivent, perte moyenne = X%"
                                - **Avantage :** Mesure le risque de queue (tail risk)
                                - **Plus conservateur :** Que le VaR simple

                                ### **🎲 Métriques de Distribution**

                                **📈 Skewness (Asymétrie)**
                                - **Définition :** Mesure l'asymétrie de la distribution
                                - **Positif :** Plus de gros gains que de grosses pertes ✅
                                - **Négatif :** Plus de grosses pertes que de gros gains ❌
                                - **Idéal :** Positif pour les stratégies

                                **🏔️ Kurtosis (Aplatissement)**
                                - **Définition :** Mesure la "queue" de la distribution
                                - **Positif :** Plus d'événements extrêmes que normal
                                - **Négatif :** Moins d'événements extrêmes ✅
                                - **Trading :** Négatif = moins de risques extrêmes

                                ### **💼 Métriques de Trading**

                                **🎯 Win Rate**
                                - **Définition :** Pourcentage de trades/périodes gagnants
                                - **Calcul :** Trades gagnants / Total trades
                                - **Paradoxe :** Peut être faible avec excellent R/R
                                - **Équilibre :** 40-60% = bon, mais R/R plus important

                                **💰 Profit Factor**
                                - **Définition :** Gains bruts / Pertes brutes
                                - **Calcul :** Somme(gains) / |Somme(pertes)|
                                - **Interprétation :** "Chaque € perdu génère X€ de gain"
                                - **Excellent :** > 2.0, > 3.0 = exceptionnel

                                **🔄 Recovery Factor**
                                - **Définition :** Rendement total / Max Drawdown
                                - **Usage :** Vitesse de récupération après pertes
                                - **Bon niveau :** > 5 = excellent récupération
                                - **Stratégie :** Plus c'est haut, mieux c'est

                                **⚖️ Omega Ratio**
                                - **Définition :** Probabilité de gains vs pertes (seuil = 0%)
                                - **Calcul :** Gains(>0%) / |Pertes(<0%)|
                                - **Usage :** Alternative au Profit Factor
                                - **Avantage :** Prend en compte toute la distribution

                                ### **🎯 Métriques Personnalisées**

                                **🏆 RR Ratio Avg (Risk/Reward)**
                                - **Définition :** Rapport gain moyen / perte moyenne
                                - **Calcul :** |Gain moyen par trade| / |Perte moyenne par trade|
                                - **Excellent :** > 2 = très bon, > 3 = exceptionnel
                                - **Stratégie :** Compense un Win Rate faible

                                ---

                                ## 📈 **Comment Interpréter Votre Performance**

                                ### **🟢 Stratégie Excellente :**
                                - Sharpe > 1.5 ✅
                                - CAGR > 15% ✅
                                - Max DD < 15% ✅
                                - Profit Factor > 2 ✅
                                - RR Ratio > 2 ✅

                                ### **🟡 Stratégie Correcte :**
                                - Sharpe 1-1.5
                                - CAGR 8-15%
                                - Max DD 15-25%
                                - Profit Factor 1.5-2
                                - RR Ratio 1-2

                                ### **🔴 À Améliorer :**
                                - Sharpe < 1
                                - CAGR < 8%
                                - Max DD > 25%
                                - Profit Factor < 1.5
                                - RR Ratio < 1

                                **💡 Astuce :** Une stratégie avec Win Rate faible (30-40%) peut être excellente si RR Ratio > 3 !
                                """)


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

        # Conseils rapides pour débuter
        st.markdown("## 🚀 Conseils pour débuter rapidement")

        quick_col1, quick_col2, quick_col3 = st.columns(3)

        with quick_col1:
            st.markdown("""
            ### 💡 **Nouveau ?**
            1. 📥 Téléchargez un exemple via le tutoriel
            2. 🔄 Uploadez le fichier
            3. ✅ Vérifiez l'auto-détection
            4. 🚀 Lancez l'analyse !
            """)

        with quick_col2:
            st.markdown("""
            ### 🎯 **Problème courant**
            - **Erreur de format ?** → Vérifiez le tutoriel
            - **Mauvais type ?** → Utilisez l'auto-détection
            - **Pas de données ?** → Index = dates obligatoire
            """)

        with quick_col3:
            st.markdown("""
            ### 🔧 **Sources compatibles**
            - MetaTrader 4/5
            - TradingView
            - Interactive Brokers
            - Fichiers Excel manuels
            """)

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