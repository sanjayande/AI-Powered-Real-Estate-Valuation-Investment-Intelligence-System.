import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8000"

def test_endpoint(name, url, method="GET", data=None):
    print(f"Testing {name} [{method} {url}]...")
    req = urllib.request.Request(url, method=method)
    if data:
        json_bytes = json.dumps(data).encode('utf-8')
        req.add_header('Content-Type', 'application/json')
        req.data = json_bytes
    
    try:
        with urllib.request.urlopen(req) as resp:
            status_code = resp.status
            body = json.loads(resp.read().decode('utf-8'))
            print(f"  [SUCCESS] Status: {status_code}")
            preview = json.dumps(body, indent=2)[:300]
            print(f"  Response Preview: {preview}...\n")
            return status_code, body
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        print(f"  [HTTP Error Response] Status: {e.code}")
        print(f"  Error Response: {error_body[:200]}...\n")
        return e.code, error_body
    except Exception as e:
        print(f"  [ERROR] Failed: {e}\n")
        return None, None

def run_all_tests():
    print("=" * 60)
    print("FASTAPI BACKEND & API VERIFICATION SUITE")
    print("=" * 60 + "\n")

    # 1. Health
    test_endpoint("1. GET /health", f"{BASE_URL}/health")

    # 2. Model Info
    test_endpoint("2. GET /model-info", f"{BASE_URL}/model-info")

    # 3. Features
    test_endpoint("3. GET /features", f"{BASE_URL}/features")

    # 4. Overview Metrics
    test_endpoint("4. GET /overview-metrics", f"{BASE_URL}/overview-metrics")

    # 5. Valid Predict
    valid_payload = {
        "City": "Delhi",
        "Locality_Type": "Premium",
        "Property_Type": "Apartment",
        "BHK": 3,
        "Bathrooms": 3,
        "Super_Area_SqFt": 1650.0,
        "Carpet_Area_SqFt": 1250.0,
        "Floor_Number": 8,
        "Total_Floors": 20,
        "Age_of_Property": 4,
        "Furnishing_Status": "Semi-Furnished",
        "Parking": 1,
        "Lift_Available": 1,
        "Gated_Community": 1,
        "Distance_to_Metro_km": 1.2,
        "Distance_to_City_Center_km": 6.5
    }
    test_endpoint("5. POST /predict (Valid Payload)", f"{BASE_URL}/predict", method="POST", data=valid_payload)

    # 6. Invalid Predict (Carpet Area > Super Area)
    invalid_payload = valid_payload.copy()
    invalid_payload["Carpet_Area_SqFt"] = 2500.0  # Invalid!
    test_endpoint("6. POST /predict (Invalid Carpet Area > Super Area)", f"{BASE_URL}/predict", method="POST", data=invalid_payload)

    # 7. Invalid Predict (Invalid City)
    invalid_city_payload = valid_payload.copy()
    invalid_city_payload["City"] = "Atlantis"  # Invalid!
    test_endpoint("7. POST /predict (Invalid City Name)", f"{BASE_URL}/predict", method="POST", data=invalid_city_payload)

    # 8. Explain (Global)
    test_endpoint("8. POST /explain (Global)", f"{BASE_URL}/explain", method="POST", data={})

    # 9. Explain (Local Contextual)
    test_endpoint("9. POST /explain (Local Contextual)", f"{BASE_URL}/explain", method="POST", data=valid_payload)

    # 10. Market Comparison
    test_endpoint("10. POST /market-comparison", f"{BASE_URL}/market-comparison", method="POST", data=valid_payload)

    # 11. What-If Analysis
    modified_payload = valid_payload.copy()
    modified_payload["Super_Area_SqFt"] = 2000.0
    modified_payload["Carpet_Area_SqFt"] = 1500.0
    what_if_req = {
        "base_property": valid_payload,
        "modified_property": modified_payload
    }
    test_endpoint("11. POST /what-if", f"{BASE_URL}/what-if", method="POST", data=what_if_req)

    # 12. Frontend Root HTML
    print("Testing Root Static Frontend Server [GET http://127.0.0.1:8000/]...")
    try:
        with urllib.request.urlopen(BASE_URL) as resp:
            html_content = resp.read().decode('utf-8')
            if "<title>REAL ESTATE AI" in html_content:
                print("  [SUCCESS] Frontend Index HTML successfully served at root / !\n")
            else:
                print("  [WARNING] HTML served but title did not match.\n")
    except Exception as e:
        print(f"  [ERROR] Frontend Root GET failed: {e}\n")

if __name__ == "__main__":
    run_all_tests()
