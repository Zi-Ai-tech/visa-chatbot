"""
Test script for multiple search methods
"""
import time
import requests
from bs4 import BeautifulSoup

def test_ddg_new():
    """Test using the new ddgs package."""
    print("\n" + "="*50)
    print("Testing DDGS (new package)...")
    print("="*50)
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text("Ireland visa requirements", max_results=3))
            if results:
                print(f"✅ Found {len(results)} results")
                for i, r in enumerate(results[:2], 1):
                    print(f"   {i}. {r.get('title', 'N/A')[:60]}...")
            else:
                print("❌ No results returned")
    except ImportError:
        print("❌ ddgs not installed. Run: pip install ddgs")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_googlesearch():
    """Test using googlesearch-python."""
    print("\n" + "="*50)
    print("Testing Google Search...")
    print("="*50)
    try:
        from googlesearch import search
        results = list(search("Ireland visa requirements", num_results=3, advanced=True))
        if results:
            print(f"✅ Found {len(results)} results")
            for i, r in enumerate(results[:2], 1):
                print(f"   {i}. {r.title[:60]}...")
        else:
            print("❌ No results returned")
    except ImportError:
        print("❌ googlesearch-python not installed. Run: pip install googlesearch-python")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_direct_request():
    """Test direct HTTP request to check network."""
    print("\n" + "="*50)
    print("Testing Network Connectivity...")
    print("="*50)
    
    urls = [
        ("https://www.google.com", "Google"),
        ("https://duckduckgo.com", "DuckDuckGo"),
        ("https://www.irishimmigration.ie", "Irish Immigration")
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    for url, name in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            print(f"✅ {name}: Status {response.status_code}")
        except requests.exceptions.Timeout:
            print(f"⚠️ {name}: Timeout")
        except requests.exceptions.ConnectionError:
            print(f"❌ {name}: Connection Error - Check network/firewall")
        except Exception as e:
            print(f"❌ {name}: {str(e)[:50]}")

def test_requests_search():
    """Simple search using requests to DuckDuckGo HTML."""
    print("\n" + "="*50)
    print("Testing DuckDuckGo HTML Search...")
    print("="*50)
    try:
        query = "Ireland visa requirements"
        url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            results = soup.find_all('a', class_='result__a')
            if results:
                print(f"✅ Found {len(results)} results")
                for i, r in enumerate(results[:2], 1):
                    print(f"   {i}. {r.get_text()[:60]}...")
            else:
                print("❌ No results found in HTML")
        else:
            print(f"❌ Status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("\n🔍 SEARCH DIAGNOSTIC TOOL")
    print("="*50)
    
    # Test network first
    test_direct_request()
    
    # Test DDG HTML (doesn't require API)
    test_requests_search()
    
    # Test new ddgs package
    test_ddg_new()
    
    # Test Google search
    test_googlesearch()
    
    print("\n" + "="*50)
    print("RECOMMENDATIONS:")
    print("="*50)
    print("If all methods fail: Check internet connection and firewall")
    print("If only DDG fails: IP may be rate-limited, try VPN or wait")
    print("If Google works: Use googlesearch-python as fallback")