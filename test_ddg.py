from duckduckgo_search import DDGS
import time

def test_ddg_search():
    print("Testing DuckDuckGo Search...")
    
    try:
        with DDGS() as ddgs:
            # Test 1: Simple search
            print("\n1. Testing simple search...")
            results = list(ddgs.text("Ireland visa requirements", max_results=2))
            print(f"   Found {len(results)} results")
            
            # Test 2: Search with delay
            print("\n2. Testing with delay...")
            time.sleep(2)
            results2 = list(ddgs.text("UK student visa IELTS", max_results=2))
            print(f"   Found {len(results2)} results")
            
            if len(results) == 0 and len(results2) == 0:
                print("\n❌ No results - Possible issues:")
                print("   - Rate limiting (IP temporarily blocked)")
                print("   - Network/firewall blocking DDG")
                print("   - DNS issues")
            else:
                print("\n✅ DDG search is working!")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPossible fixes:")
        print("1. Check internet connection")
        print("2. Try using a VPN")
        print("3. Wait 5-10 minutes and try again")
        print("4. Update duckduckgo-search: pip install --upgrade duckduckgo-search")

if __name__ == "__main__":
    test_ddg_search()