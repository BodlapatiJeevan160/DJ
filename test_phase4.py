from skills.web_browser import open_browser, open_url, search_web, get_current_page, read_page, click_element, type_into_element, go_back, close_browser
import time
import json

def test_phase4():
    print("--- Test 1: Open Browser ---")
    print(open_browser())
    time.sleep(2)
    
    print("\n--- Test 2: Open Wikipedia ---")
    print(open_url("wikipedia.org"))
    time.sleep(2)
    print("Current page:", get_current_page())
    
    print("\n--- Test 3 & 4: Search Web ---")
    print(search_web("Python programming"))
    time.sleep(2)
    
    print("\n--- Test 6: Read Page ---")
    res = read_page()
    # Truncate output for logging
    try:
        data = json.loads(res)
        text = data.get("data", {}).get("text", "")
        print("Read length:", len(text))
        print("Snippet:", text[:200].replace("\n", " "))
    except Exception:
        print(res)
    
    print("\n--- Test: Go Back ---")
    print(go_back())
    time.sleep(2)
    
    print("\n--- Test 10: Close Browser ---")
    print(close_browser())

if __name__ == "__main__":
    test_phase4()
