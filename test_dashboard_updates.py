import urllib.request

def verify_dashboard():
    url = "http://127.0.0.1:8000/"
    req = urllib.request.urlopen(url)
    html = req.read().decode('utf-8')

    checks = {
        "Model Status Card Removed": "Model Status" not in html,
        "sys-model-name Element Removed": "sys-model-name" not in html,
        "sys-model-status Element Removed": "sys-model-status" not in html,
        "PRIME INVESTMENT ASSET Removed": "PRIME INVESTMENT ASSET" not in html,
        "Properties Analyzed Initialized to 0": 'id="kpi-props">0<' in html,
        "Metropolitan Coverage Initialized to 0": 'id="kpi-cities">0 Cities<' in html,
        "Avg Price Initialized to 0.00 L": 'id="kpi-avg-price">₹ 0.00 L<' in html,
        "Median Price Initialized to 0.00 L": 'id="kpi-med-price">₹ 0.00 L<' in html,
        "Model R2 Initialized to 0.0000": 'id="kpi-r2">0.0000<' in html,
        "Model MAE Initialized to 0.00 L": 'id="kpi-mae">₹ 0.00 L<' in html,
        "Recent Valuation Initialized to 0.00 Lakhs": 'id="overview-recent-val-num">₹ 0.00 Lakhs<' in html,
        "Score Initialized to 0/100": 'id="overview-recent-val-score">0/100<' in html,
        "Submit Button is PREDICT PROPERTY VALUE": '⚡ PREDICT PROPERTY VALUE' in html,
    }

    all_passed = True
    print("\n" + "=" * 60)
    print("DASHBOARD UI SPECIFICATION VERIFICATION")
    print("=" * 60)
    for check_name, passed in checks.items():
        status = "PASSED" if passed else "FAILED"
        print(f"[{status}] {check_name}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL DASHBOARD VERIFICATION TESTS PASSED SUCCESSFULLY!")
    else:
        print("SOME VERIFICATION CHECKS FAILED.")
    print("=" * 60 + "\n")
    return all_passed

if __name__ == '__main__':
    verify_dashboard()
