#!/usr/bin/env python3
"""
Script de diagnostic pour analyser le problème de janvier 2020
"""
import pandas as pd
import numpy as np
from datetime import datetime

def analyze_january_2020(csv_path):
    print("=== DIAGNOSTIC JANVIER 2020 ===")

    # Lire le fichier CSV
    df = pd.read_csv(csv_path)
    print(f"📊 Total trades dans le fichier: {len(df)}")

    # Convertir les timestamps
    df['time_close_dt'] = pd.to_datetime(df['time_close'], unit='s', errors='coerce')

    # Filtrer janvier 2020
    jan_2020 = df[(df['time_close_dt'] >= '2020-01-01') & (df['time_close_dt'] < '2020-02-01')]
    print(f"📊 Trades en janvier 2020: {len(jan_2020)}")

    if len(jan_2020) > 0:
        print("\n=== DÉTAILS TRADES JANVIER 2020 ===")
        for idx, row in jan_2020.iterrows():
            print(f"Date: {row['time_close_dt'].strftime('%Y-%m-%d %H:%M')} | Profit: {row['profit']:.2f}")

        total_profit_jan = jan_2020['profit'].sum()
        print(f"\n💰 Total profit janvier 2020: {total_profit_jan:.2f}")

        # Simuler le calcul comme dans l'application
        print("\n=== SIMULATION CALCUL APPLI ===")

        # Créer la série comme l'application
        df_processed = df[['time_close_dt', 'profit']].copy()
        df_processed = df_processed.dropna()
        df_processed = df_processed.set_index('time_close_dt')
        df_processed = df_processed.sort_index()

        print(f"📊 Données après traitement: {len(df_processed)} entrées")

        # Calculer equity curve
        initial_capital = 10000
        pnl_cumulative = df_processed['profit'].cumsum()
        equity_curve = initial_capital + pnl_cumulative

        print(f"📈 Equity début janvier 2020: {equity_curve.loc[equity_curve.index >= '2020-01-01'].iloc[0]:.2f}")
        print(f"📈 Equity fin janvier 2020: {equity_curve.loc[equity_curve.index < '2020-02-01'].iloc[-1]:.2f}")

        # Calculer returns
        returns = equity_curve.pct_change().dropna()

        # Rendements mensuels
        monthly_returns_sum = returns.resample('M').sum()
        monthly_returns_compound = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

        jan_2020_return_sum = monthly_returns_sum.loc['2020-01-31'] if '2020-01-31' in monthly_returns_sum.index else 0
        jan_2020_return_compound = monthly_returns_compound.loc['2020-01-31'] if '2020-01-31' in monthly_returns_compound.index else 0

        print(f"📊 Rendement janvier 2020 (somme): {jan_2020_return_sum*100:.4f}%")
        print(f"📊 Rendement janvier 2020 (composé): {jan_2020_return_compound*100:.4f}%")

        # Alternative: rendement direct
        equity_start = equity_curve.loc[equity_curve.index < '2020-01-01'].iloc[-1] if len(equity_curve.loc[equity_curve.index < '2020-01-01']) > 0 else initial_capital
        equity_end = equity_curve.loc[equity_curve.index < '2020-02-01'].iloc[-1]
        direct_return = (equity_end - equity_start) / equity_start

        print(f"📊 Rendement janvier 2020 (direct): {direct_return*100:.4f}%")

    else:
        print("❌ Aucun trade trouvé en janvier 2020")

    print("\n=== VÉRIFICATION AUTRES MOIS ===")
    for month in range(1, 13):
        month_data = df[(df['time_close_dt'].dt.year == 2020) & (df['time_close_dt'].dt.month == month)]
        if len(month_data) > 0:
            print(f"2020-{month:02d}: {len(month_data)} trades, profit total: {month_data['profit'].sum():.2f}")

if __name__ == "__main__":
    csv_file = "/Users/luca/Downloads/XAUUSD_CLAUDE_FIXED_2025.09.19.csv"
    analyze_january_2020(csv_file)