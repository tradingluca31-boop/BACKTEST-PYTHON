#!/usr/bin/env python3
"""
Validation script to verify all critical fixes were applied
"""

import re

def check_fixes(filename):
    """Check if all critical fixes are present"""

    with open(filename, 'r') as f:
        content = f.read()

    results = []

    # Check 1: Division by zero checks for avg_loss
    avg_loss_checks = len(re.findall(r'avg_loss\s*>\s*1e-10', content))
    results.append(("Division by zero checks (avg_loss > 1e-10)", avg_loss_checks >= 5, f"{avg_loss_checks} found"))

    # Check 2: Specific exceptions instead of bare except
    bare_excepts = len(re.findall(r'^\s*except:\s*$', content, re.MULTILINE))
    results.append(("Bare except: statements removed", bare_excepts <= 1, f"{bare_excepts} remaining"))

    # Check 3: Exception handlers with 'as e'
    specific_excepts = len(re.findall(r'except\s+\w+.*as\s+e:', content))
    results.append(("Specific exception handlers", specific_excepts >= 30, f"{specific_excepts} found"))

    # Check 4: NaN handling with .dropna() after resample
    dropna_after_resample = len(re.findall(r'resample.*\.dropna\(\)', content))
    results.append(("NaN handling (.dropna() after resample)", dropna_after_resample >= 4, f"{dropna_after_resample} found"))

    # Check 5: Initial capital parameter in __init__
    init_capital_param = 'def __init__(self, initial_capital=10000):' in content
    results.append(("Initial capital parameter", init_capital_param, "Found" if init_capital_param else "Missing"))

    # Check 6: Minimum data validation
    min_data_check = 'if len(returns) < 10:' in content
    results.append(("Minimum 10 data points validation", min_data_check, "Found" if min_data_check else "Missing"))

    # Check 7: Timestamp conversion error handling
    timestamp_handling = 'except (ValueError, TypeError) as e:' in content
    results.append(("Timestamp conversion error handling", timestamp_handling, "Found" if timestamp_handling else "Missing"))

    # Check 8: Empty DataFrame validation
    empty_df_check = 'if len(returns) < 2:' in content
    results.append(("Empty DataFrame validation", empty_df_check, "Found" if empty_df_check else "Missing"))

    # Check 9: Risk of Ruin division by zero fix
    risk_of_ruin_fix = 'if avg_win > 0 and avg_loss > 1e-10:' in content
    results.append(("Risk of Ruin division by zero fix", risk_of_ruin_fix, "Found" if risk_of_ruin_fix else "Missing"))

    # Check 10: RR Ratio infinity handling
    rr_inf_handling = "rr_ratio = float('inf') if avg_win > 0 else 0" in content
    results.append(("RR Ratio infinity handling", rr_inf_handling, "Found" if rr_inf_handling else "Missing"))

    # Print results
    print("=" * 80)
    print("VALIDATION RESULTS FOR: {}".format(filename))
    print("=" * 80)
    print()

    all_passed = True
    for check_name, passed, details in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} | {check_name:50s} | {details}")
        if not passed:
            all_passed = False

    print()
    print("=" * 80)
    if all_passed:
        print("✅ ALL CHECKS PASSED!")
    else:
        print("❌ SOME CHECKS FAILED - Review above")
    print("=" * 80)

    return all_passed

if __name__ == "__main__":
    import sys

    filename = "/tmp/BACKTEST-PYTHON/app_fixed.py"
    success = check_fixes(filename)
    sys.exit(0 if success else 1)
