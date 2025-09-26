#!/usr/bin/env python3
"""
Script de debug pour analyser les drawdowns et les zones affichées
"""
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

def debug_drawdown_calculation(csv_path):
    print("=== DEBUG CALCUL DRAWDOWNS ===")

    # Lire le CSV et créer l'equity curve réelle
    df = pd.read_csv(csv_path)
    df['close_date'] = pd.to_datetime(df['time_close'], unit='s')
    df_sorted = df.sort_values('close_date')

    initial_capital = 10000
    df_sorted['equity'] = initial_capital + df_sorted['profit'].cumsum()

    equity_series = pd.Series(df_sorted['equity'].values, index=df_sorted['close_date'])

    # Calculer High Water Mark et drawdowns
    hwm = equity_series.expanding().max()
    drawdowns = (equity_series - hwm) / hwm * 100

    print(f"Equity range: {equity_series.min():.2f} → {equity_series.max():.2f}")
    print(f"Drawdown range: {drawdowns.max():.2f}% → {drawdowns.min():.2f}%")

    # Debug: détecter les périodes comme dans le code
    drawdown_periods = []
    in_drawdown = False
    start_idx = None
    start_date = None
    peak_equity = None

    print("\n=== DETECTION DES PERIODES ===")

    for i, (date, eq) in enumerate(equity_series.items()):
        current_dd = drawdowns.iloc[i]
        current_hwm = hwm.iloc[i]

        if current_dd < -0.1 and not in_drawdown:  # Début drawdown
            in_drawdown = True
            start_idx = i
            start_date = date
            peak_equity = current_hwm
            print(f"DEBUT DD #{len(drawdown_periods)+1}: {date.strftime('%Y-%m-%d')} | DD: {current_dd:.2f}% | Peak: {peak_equity:.2f}")

        elif current_dd >= 0 and in_drawdown:  # Fin drawdown
            in_drawdown = False
            end_date = date
            end_idx = i
            max_dd = drawdowns.iloc[start_idx:end_idx+1].min()

            drawdown_periods.append({
                'start': start_date,
                'end': end_date,
                'max_drawdown': max_dd,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'peak_equity': peak_equity
            })

            print(f"FIN DD #{len(drawdown_periods)}: {end_date.strftime('%Y-%m-%d')} | Max DD: {max_dd:.2f}%")

    # Si on finit en drawdown, l'ajouter
    if in_drawdown:
        end_date = equity_series.index[-1]
        end_idx = len(equity_series) - 1
        max_dd = drawdowns.iloc[start_idx:end_idx+1].min()
        drawdown_periods.append({
            'start': start_date,
            'end': end_date,
            'max_drawdown': max_dd,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'peak_equity': peak_equity
        })
        print(f"DD EN COURS: {end_date.strftime('%Y-%m-%d')} | Max DD: {max_dd:.2f}%")

    # Top 5
    worst_5 = sorted(drawdown_periods, key=lambda x: x['max_drawdown'])[:5]

    print(f"\n=== TOP 5 DRAWDOWNS ===")
    for i, dd in enumerate(worst_5):
        print(f"#{i+1}: {dd['start'].strftime('%Y-%m-%d')} → {dd['end'].strftime('%Y-%m-%d')}")
        print(f"     Peak: {dd['peak_equity']:.2f} | DD: {dd['max_drawdown']:.2f}%")

    return worst_5, equity_series, drawdowns

if __name__ == "__main__":
    csv_file = "/Users/luca/Downloads/XAUUSD_CLAUDE_FIXED_2025.09.19.csv"
    worst_5, equity_series, drawdowns = debug_drawdown_calculation(csv_file)