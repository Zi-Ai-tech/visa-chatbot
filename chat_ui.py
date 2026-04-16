import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from typing import List, Tuple, Optional, Dict
import time
import random
import os
import requests
from bs4 import BeautifulSoup
import urllib.parse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Global Visa Assistant", page_icon="🌍", layout="wide")
st.title("🌍 Global Visa Assistant")
st.caption("Real-time visa information from web search")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "search_cache" not in st.session_state:
    st.session_state.search_cache = {}
if "button_counter" not in st.session_state:
    st.session_state.button_counter = 0

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = None
    vector_store_path = "visa_vector_store"
    
    if os.path.exists(vector_store_path):
        faiss_file = os.path.join(vector_store_path, "index.faiss")
        pkl_file = os.path.join(vector_store_path, "index.pkl")
        if os.path.exists(faiss_file) and os.path.exists(pkl_file):
            try:
                vector_store = FAISS.load_local(
                    vector_store_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception:
                vector_store = None

    llm = OllamaLLM(model="llama3.2", temperature=0.1, streaming=True)

    return vector_store, llm

vector_store, llm = load_models()

# -----------------------------
# COUNTRY DETECTION - 70+ Countries
# -----------------------------
COUNTRIES = {
    "afghanistan": ["afghanistan", "afghan", "kabul"],
    "albania": ["albania", "albanian", "tirana"],
    "algeria": ["algeria", "algerian", "algiers"],
    "argentina": ["argentina", "argentinian", "buenos aires"],
    "australia": ["australia", "australian", "sydney", "melbourne", "brisbane", "perth"],
    "austria": ["austria", "austrian", "vienna"],
    "bahrain": ["bahrain", "bahraini", "manama"],
    "bangladesh": ["bangladesh", "bangladeshi", "dhaka"],
    "belgium": ["belgium", "belgian", "brussels"],
    "brazil": ["brazil", "brazilian", "rio", "sao paulo", "brasilia"],
    "canada": ["canada", "canadian", "toronto", "vancouver", "montreal", "ottawa"],
    "chile": ["chile", "chilean", "santiago"],
    "china": ["china", "chinese", "beijing", "shanghai", "guangzhou"],
    "colombia": ["colombia", "colombian", "bogota"],
    "czech republic": ["czech", "czech republic", "prague"],
    "denmark": ["denmark", "danish", "copenhagen"],
    "egypt": ["egypt", "egyptian", "cairo"],
    "finland": ["finland", "finnish", "helsinki"],
    "france": ["france", "french", "paris", "lyon", "marseille"],
    "germany": ["germany", "german", "berlin", "munich", "frankfurt"],
    "ghana": ["ghana", "ghanaian", "accra"],
    "greece": ["greece", "greek", "athens"],
    "hong kong": ["hong kong", "hongkong"],
    "hungary": ["hungary", "hungarian", "budapest"],
    "iceland": ["iceland", "icelandic", "reykjavik"],
    "india": ["india", "indian", "delhi", "mumbai", "bangalore", "kolkata"],
    "indonesia": ["indonesia", "indonesian", "jakarta", "bali"],
    "iran": ["iran", "iranian", "tehran"],
    "iraq": ["iraq", "iraqi", "baghdad"],
    "ireland": ["ireland", "irish", "dublin"],
    "israel": ["israel", "israeli", "tel aviv", "jerusalem"],
    "italy": ["italy", "italian", "rome", "milan", "venice", "florence"],
    "japan": ["japan", "japanese", "tokyo", "osaka", "kyoto"],
    "jordan": ["jordan", "jordanian", "amman"],
    "kenya": ["kenya", "kenyan", "nairobi"],
    "kuwait": ["kuwait", "kuwaiti"],
    "lebanon": ["lebanon", "lebanese", "beirut"],
    "malaysia": ["malaysia", "malaysian", "kuala lumpur"],
    "mexico": ["mexico", "mexican", "mexico city", "cancun"],
    "morocco": ["morocco", "moroccan", "casablanca", "rabat"],
    "nepal": ["nepal", "nepali", "kathmandu"],
    "netherlands": ["netherlands", "dutch", "holland", "amsterdam"],
    "new zealand": ["new zealand", "nz", "auckland", "wellington"],
    "nigeria": ["nigeria", "nigerian", "lagos", "abuja"],
    "norway": ["norway", "norwegian", "oslo"],
    "oman": ["oman", "omani", "muscat"],
    "pakistan": ["pakistan", "pakistani", "islamabad", "karachi", "lahore"],
    "peru": ["peru", "peruvian", "lima"],
    "philippines": ["philippines", "filipino", "philippine", "manila"],
    "poland": ["poland", "polish", "warsaw"],
    "portugal": ["portugal", "portuguese", "lisbon"],
    "qatar": ["qatar", "qatari", "doha"],
    "romania": ["romania", "romanian", "bucharest"],
    "russia": ["russia", "russian", "moscow"],
    "saudi arabia": ["saudi", "saudi arabia", "ksa", "riyadh", "jeddah"],
    "singapore": ["singapore", "singaporean"],
    "south africa": ["south africa", "south african", "johannesburg", "cape town"],
    "south korea": ["south korea", "korea", "korean", "seoul", "busan"],
    "spain": ["spain", "spanish", "madrid", "barcelona"],
    "sri lanka": ["sri lanka", "sri lankan", "colombo"],
    "sweden": ["sweden", "swedish", "stockholm"],
    "switzerland": ["switzerland", "swiss", "zurich", "geneva"],
    "taiwan": ["taiwan", "taiwanese", "taipei"],
    "thailand": ["thailand", "thai", "bangkok"],
    "turkey": ["turkey", "turkish", "türkiye", "istanbul", "ankara"],
    "uae": ["uae", "dubai", "abu dhabi", "emirates", "sharjah", "united arab emirates"],
    "uk": ["uk", "united kingdom", "britain", "british", "london", "england", "scotland", "wales"],
    "ukraine": ["ukraine", "ukrainian", "kyiv", "kiev"],
    "usa": ["usa", "united states", "america", "american", "new york", "california", "texas", "washington"],
    "vietnam": ["vietnam", "vietnamese", "hanoi", "ho chi minh", "saigon"],
}

# City to Country mapping
CITY_TO_COUNTRY = {
    "dubai": "uae", "abu dhabi": "uae", "sharjah": "uae", "ajman": "uae",
    "london": "uk", "manchester": "uk", "birmingham": "uk", "edinburgh": "uk",
    "paris": "france", "lyon": "france", "marseille": "france",
    "berlin": "germany", "munich": "germany", "frankfurt": "germany",
    "rome": "italy", "milan": "italy", "venice": "italy",
    "madrid": "spain", "barcelona": "spain",
    "amsterdam": "netherlands",
    "new york": "usa", "los angeles": "usa", "chicago": "usa", "miami": "usa",
    "toronto": "canada", "vancouver": "canada", "montreal": "canada",
    "sydney": "australia", "melbourne": "australia", "brisbane": "australia",
    "tokyo": "japan", "osaka": "japan", "kyoto": "japan",
    "singapore": "singapore",
    "doha": "qatar",
    "riyadh": "saudi arabia", "jeddah": "saudi arabia",
}

# -----------------------------
# OFFICIAL DOMAINS
# -----------------------------
OFFICIAL_DOMAINS = {
    "australia": ["homeaffairs.gov.au", "immi.gov.au", "border.gov.au"],
    "canada": ["canada.ca", "cic.gc.ca", "ircc.canada.ca"],
    "france": ["france-visas.gouv.fr", "diplomatie.gouv.fr", "service-public.fr"],
    "germany": ["auswaertiges-amt.de", "bamf.de", "bva.bund.de"],
    "india": ["indianvisaonline.gov.in", "mea.gov.in", "mha.gov.in"],
    "ireland": ["irishimmigration.ie", "justice.ie", "dfa.ie", "gov.ie"],
    "italy": ["esteri.it", "vistoperitalia.esteri.it"],
    "japan": ["mofa.go.jp", "moj.go.jp", "immigration.go.jp"],
    "netherlands": ["ind.nl", "government.nl", "netherlandsworldwide.nl"],
    "new zealand": ["immigration.govt.nz", "govt.nz"],
    "pakistan": ["dgip.gov.pk", "visa.nadra.gov.pk"],
    "saudi arabia": ["visa.mofa.gov.sa", "mofa.gov.sa"],
    "singapore": ["ica.gov.sg", "mom.gov.sg"],
    "south africa": ["dha.gov.za", "dirco.gov.za"],
    "south korea": ["immigration.go.kr", "mofa.go.kr", "visa.go.kr"],
    "spain": ["exteriores.gob.es", "maec.es"],
    "sweden": ["migrationsverket.se", "swedenabroad.se"],
    "switzerland": ["sem.admin.ch", "eda.admin.ch"],
    "uae": ["icp.gov.ae", "u.ae", "gdrfad.gov.ae", "mofa.gov.ae"],
    "uk": ["gov.uk", "homeoffice.gov.uk"],
    "usa": ["uscis.gov", "state.gov", "travel.state.gov", "cbp.gov"],
}

# -----------------------------
# SCHENGEN COUNTRIES
# -----------------------------
SCHENGEN_COUNTRIES = [
    "austria", "belgium", "czech republic", "denmark", "estonia", "finland", "france",
    "germany", "greece", "hungary", "iceland", "italy", "latvia", "liechtenstein", 
    "lithuania", "luxembourg", "malta", "netherlands", "norway", "poland", "portugal",
    "slovakia", "slovenia", "spain", "sweden", "switzerland"
]

# -----------------------------
# KNOWN REQUIREMENTS DATABASE
# -----------------------------
KNOWN_REQUIREMENTS = {
    "ireland": {
        "student": {
            "ielts_required": True,
            "min_score": "6.0 - 6.5",
            "min_bands": "5.5 - 6.0",
            "test_type": "IELTS Academic",
            "alternatives": "TOEFL iBT (80-90), Cambridge CAE/CPE, PTE Academic (59-63)",
            "note": "Required for non-native English speakers. Minimum IELTS 5.0 for language courses.",
            "financial": "€10,000 for courses >8 months; €833/month or €6,665 total for courses ≤8 months",
            "processing": "8 weeks",
            "fees": "€60 single entry, €100 multi-entry",
            "documents": "Valid passport (6+ months), acceptance letter from Irish institution, proof of funds (€10,000), medical insurance, IELTS/English proficiency certificate, passport photos, completed application form",
            "official_url": "https://www.irishimmigration.ie"
        },
        "tourist": {
            "ielts_required": False,
            "note": "No language test required for tourist visas.",
            "financial": "Proof of sufficient funds (typically €50-100 per day of stay)",
            "processing": "8 weeks for visa-required nationals",
            "fees": "€60 single entry, €100 multi-entry",
            "documents": "Valid passport (minimum 6 months validity beyond intended stay), completed AVATS online application form, recent passport-sized photographs, travel itinerary (flight reservations), proof of accommodation, proof of sufficient funds (bank statements), travel/medical insurance",
            "official_url": "https://www.irishimmigration.ie"
        },
        "work": {
            "ielts_required": False,
            "note": "No general English requirement, employer-specific.",
            "official_url": "https://www.irishimmigration.ie"
        }
    },
    "uk": {
        "student": {
            "ielts_required": True,
            "min_score": "5.5 - 7.0",
            "min_bands": "4.0 - 5.5",
            "test_type": "IELTS for UKVI Academic",
            "alternatives": "Trinity College London, Pearson PTE Academic UKVI, LanguageCert",
            "note": "B1 (IELTS 4.0) for below degree, B2 (IELTS 5.5) for degree level.",
            "financial": "£1,334/month in London; £1,023/month outside London (up to 9 months)",
            "processing": "3 weeks (priority service available)",
            "fees": "£490 for Student visa (standard)",
            "documents": "Valid passport, CAS letter from licensed sponsor, proof of financial maintenance, TB test certificate (if applicable), ATAS certificate (if applicable), parental consent (if under 18)",
            "official_url": "https://www.gov.uk/student-visa"
        },
        "tourist": {
            "ielts_required": False,
            "note": "No English test required for visitor visas.",
            "processing": "3 weeks",
            "fees": "£115 for Standard Visitor visa",
            "documents": "Valid passport, travel itinerary, proof of sufficient funds, accommodation details, proof of ties to home country",
            "official_url": "https://www.gov.uk/standard-visitor-visa"
        }
    },
    "canada": {
        "student": {
            "ielts_required": True,
            "min_score": "6.0 - 6.5",
            "min_bands": "6.0 (SDS)",
            "test_type": "IELTS Academic",
            "alternatives": "TOEFL iBT, CAEL, PTE Academic, Duolingo (some institutions)",
            "note": "Student Direct Stream (SDS) requires IELTS 6.0 in each band.",
            "financial": "$10,000 CAD + tuition fees for first year",
            "processing": "SDS: 20 calendar days; Regular: varies by country",
            "fees": "$150 CAD study permit fee",
            "documents": "Valid passport, Letter of Acceptance from DLI, proof of financial support, Provincial Attestation Letter (PAL), immigration medical exam (if required), statement of purpose",
            "official_url": "https://www.canada.ca/en/immigration-refugees-citizenship/services/study-canada.html"
        },
        "tourist": {
            "ielts_required": False,
            "note": "No language test for visitor visas.",
            "processing": "Varies by country (typically 2-8 weeks)",
            "fees": "$100 CAD visitor visa fee + $85 biometrics",
            "documents": "Valid passport, proof of funds, travel itinerary, ties to home country (employment, property, family), letter of invitation (if applicable)",
            "official_url": "https://www.canada.ca/en/immigration-refugees-citizenship/services/visit-canada.html"
        }
    },
    "australia": {
        "student": {
            "ielts_required": True,
            "min_score": "5.5 - 6.5",
            "min_bands": "5.0 - 6.0",
            "test_type": "IELTS Academic",
            "alternatives": "TOEFL iBT, PTE Academic, CAE, OET",
            "note": "Vocational courses: IELTS 5.5; University: IELTS 6.0-6.5.",
            "financial": "$21,041 AUD per year living costs + tuition fees",
            "processing": "1-4 months (varies by assessment level)",
            "fees": "$710 AUD for Student visa (subclass 500)",
            "documents": "Valid passport, Confirmation of Enrolment (CoE), Genuine Temporary Entrant statement, proof of funds, Overseas Student Health Cover (OSHC), English test results",
            "official_url": "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/student-500"
        },
        "tourist": {
            "ielts_required": False,
            "note": "No English test for visitor visas.",
            "processing": "Varies (typically 2-4 weeks)",
            "fees": "$195 AUD for Visitor visa (subclass 600)",
            "documents": "Valid passport, proof of funds, travel itinerary, evidence of intention to return home",
            "official_url": "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/visitor-600"
        }
    },
    "usa": {
        "student": {
            "ielts_required": True,
            "min_score": "6.5+ (varies by institution)",
            "min_bands": "Varies by institution",
            "test_type": "IELTS Academic or TOEFL iBT",
            "alternatives": "TOEFL iBT, PTE Academic, Duolingo English Test",
            "note": "Requirements set by individual institutions. Most require TOEFL 80+ or IELTS 6.5+.",
            "financial": "Proof of funds for first year (tuition + living expenses)",
            "processing": "Interview wait times vary by location",
            "fees": "$185 SEVIS fee + $160 visa application fee",
            "documents": "Valid passport (6+ months), Form I-20 from SEVP-approved school, DS-160 confirmation, SEVIS fee receipt, financial documents, academic transcripts",
            "official_url": "https://travel.state.gov/content/travel/en/us-visas/study/student-visa.html"
        },
        "tourist": {
            "ielts_required": False,
            "note": "No English test for B-2 tourist visas.",
            "processing": "Interview wait times vary by embassy",
            "fees": "$185 for B-2 visa",
            "documents": "Valid passport (6+ months beyond stay), DS-160 form confirmation, photo, interview required, proof of ties to home country",
            "official_url": "https://travel.state.gov/content/travel/en/us-visas/tourism-visit/visitor.html"
        }
    },
    "new zealand": {
        "student": {
            "ielts_required": True,
            "min_score": "5.5 - 6.5",
            "min_bands": "5.0 - 6.0",
            "test_type": "IELTS Academic",
            "alternatives": "TOEFL iBT, PTE Academic, Cambridge English",
            "note": "University: IELTS 6.0-6.5; Polytechnics: 5.5-6.0.",
            "financial": "$15,000 - $20,000 NZD per year living costs + tuition",
            "processing": "4-8 weeks",
            "fees": "$375 NZD for Student Visa",
            "documents": "Valid passport, Offer of Place from NZ institution, proof of funds, medical/travel insurance, medical certificates (if staying >6 months)",
            "official_url": "https://www.immigration.govt.nz/new-zealand-visas/visas/visa/student-visa"
        },
        "tourist": {
            "ielts_required": False,
            "note": "No English test for visitor visas.",
            "processing": "20-30 working days",
            "fees": "$211 NZD for Visitor Visa",
            "documents": "Valid passport (3 months beyond departure), proof of funds, onward travel ticket, evidence of ties to home country",
            "official_url": "https://www.immigration.govt.nz/new-zealand-visas/visas/visa/visitor-visa"
        }
    },
    "uae": {
        "student": {
            "ielts_required": True,
            "min_score": "5.0 - 6.5",
            "test_type": "IELTS Academic or TOEFL iBT",
            "alternatives": "PTE Academic, EmSAT",
            "note": "Foundation: IELTS 5.0; Bachelor's: 5.5-6.0; Master's: 6.0-6.5.",
            "processing": "2-4 weeks",
            "documents": "Valid passport, acceptance letter from UAE institution, proof of financial means, medical fitness certificate, Emirates ID application",
            "official_url": "https://icp.gov.ae/"
        },
        "tourist": {
            "ielts_required": False,
            "note": "No language test required for tourist visas.",
            "processing": "3-5 working days",
            "documents": "Valid passport (minimum 6 months validity), recent passport-sized photograph, completed application form, travel itinerary, hotel booking or sponsor letter",
            "official_url": "https://icp.gov.ae/"
        },
        "work": {
            "ielts_required": False,
            "note": "English proficiency required for professional positions but no formal test mandated.",
            "processing": "2-4 weeks",
            "documents": "Valid passport, job offer from UAE employer, educational certificates attested, medical fitness certificate",
            "official_url": "https://icp.gov.ae/"
        }
    }
}

# -----------------------------
# COPY FUNCTION
# -----------------------------
def copy_to_clipboard(text: str):
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception:
        return False

# -----------------------------
# SEARCH FUNCTIONS
# -----------------------------
def is_official_source(url: str, country: str) -> bool:
    url_lower = url.lower()
    if country in OFFICIAL_DOMAINS:
        for domain in OFFICIAL_DOMAINS[country]:
            if domain in url_lower:
                return True
    gov_indicators = [".gov", "gov.", "europa.eu", ".int", "immigration", "embassy", "consulate"]
    if any(indicator in url_lower for indicator in gov_indicators):
        return True
    return False

def fetch_page_content(url: str) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=8)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form']):
                element.decompose()
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.find('body')
            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
                text = re.sub(r'\s+', ' ', text)
                return text[:2000]
    except Exception:
        pass
    return ""

def search_with_fallback(query: str, max_results: int = 3) -> List[dict]:
    results = []
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                if r.get('href') and r.get('title'):
                    results.append({
                        "title": r.get('title', ''),
                        "url": r.get('href', ''),
                        "body": r.get('body', ''),
                        "source": "DDGS"
                    })
            if results:
                return results
    except Exception:
        pass
    
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for r in soup.find_all('a', class_='result__a')[:max_results]:
                href = r.get('href', '')
                if 'uddg=' in href:
                    parsed = urllib.parse.urlparse(href)
                    query_params = urllib.parse.parse_qs(parsed.query)
                    if 'uddg' in query_params:
                        href = urllib.parse.unquote(query_params['uddg'][0])
                results.append({
                    "title": r.get_text(strip=True),
                    "url": href,
                    "body": "",
                    "source": "DDG_HTML"
                })
            if results:
                return results
    except Exception:
        pass
    return results

def search_single_query(search_query: str, country: str) -> List[dict]:
    results = []
    try:
        web_results = search_with_fallback(search_query, max_results=2)
        for r in web_results:
            url = r.get('url', '').lower()
            title = r.get('title', '')
            body = r.get('body', '')
            
            skip_domains = ["facebook.com", "twitter.com", "youtube.com", "instagram.com", 
                          "linkedin.com", "pinterest.com", "reddit.com", "tiktok.com", "quora.com"]
            if any(skip in url for skip in skip_domains):
                continue
            
            if not body or len(body) < 300:
                full_content = fetch_page_content(r.get('url', ''))
                if full_content:
                    body = full_content
            
            results.append({
                "title": title,
                "url": r.get('url', ''),
                "body": body,
                "is_official": is_official_source(url, country)
            })
    except Exception:
        pass
    return results

def search_visa_information_parallel(query: str, country: str, nationality: Optional[str] = None, 
                                     visa_type: Optional[str] = None, is_language_question: bool = False,
                                     is_ban_question: bool = False) -> List[dict]:
    
    cache_key = f"{country}_{visa_type}_{is_language_question}_{is_ban_question}"
    if cache_key in st.session_state.search_cache:
        cached_data, cache_time = st.session_state.search_cache[cache_key]
        if time.time() - cache_time < 300:
            return cached_data
    
    search_queries = []
    
    if is_ban_question:
        if nationality:
            search_queries.append(f"{country} travel ban {nationality} citizens 2025 2026")
            search_queries.append(f"can {nationality} travel to {country} entry restrictions")
            search_queries.append(f"{country} visa ban for {nationality} nationals")
        search_queries.append(f"{country} travel ban entry restrictions 2025")
        search_queries.append(f"is {country} issuing visas to {nationality}" if nationality else f"{country} visa ban update")
    else:
        if country in OFFICIAL_DOMAINS:
            domain = OFFICIAL_DOMAINS[country][0]
            search_queries.append(f"site:{domain} {visa_type} visa requirements")
        
        if is_language_question:
            search_queries.append(f"{country} {visa_type} visa IELTS English requirements {nationality if nationality else ''}")
        else:
            if nationality:
                search_queries.append(f"{country} {visa_type} visa requirements for {nationality} citizens")
            search_queries.append(f"{country} {visa_type} visa requirements documents fees processing")
    
    search_queries = list(dict.fromkeys(search_queries))[:4]
    
    all_results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_query = {executor.submit(search_single_query, q, country): q for q in search_queries}
        for future in as_completed(future_to_query):
            try:
                results = future.result()
                all_results.extend(results)
            except Exception:
                pass
    
    all_results.sort(key=lambda x: (not x.get('is_official', False), x.get('title', '')))
    
    seen_urls = set()
    unique_results = []
    for r in all_results:
        url = r.get('url', '')
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)
    
    st.session_state.search_cache[cache_key] = (unique_results[:5], time.time())
    return unique_results[:5]

# -----------------------------
# DETECTION FUNCTIONS
# -----------------------------
def detect_country(query: str) -> Optional[str]:
    q = query.lower()
    
    for city, country in CITY_TO_COUNTRY.items():
        if city in q:
            return country
    
    destination_patterns = ["visa for", "visa to", "travel to", "visit", "going to", "go to", 
                           "requirements for", "ban on", "restrictions for", "allowed in", "visas for"]
    
    for pattern in destination_patterns:
        if pattern in q:
            for country, keywords in COUNTRIES.items():
                for kw in keywords:
                    if kw in q:
                        idx = q.find(kw)
                        pattern_idx = q.find(pattern)
                        if idx > pattern_idx:
                            return country
            break
    
    for country, keywords in COUNTRIES.items():
        if any(keyword in q for keyword in keywords):
            return country
    
    return None

def detect_nationality(query: str) -> Optional[str]:
    q = query.lower()
    
    for country in COUNTRIES.keys():
        patterns = [
            f"{country} citizen", f"{country} national", f"{country} passport",
            f"from {country}", f"i am {country}", f"i'm {country}",
            f"{country}s", f"for {country}"
        ]
        for pattern in patterns:
            if pattern in q:
                return country
    
    nationality_words = ["pakistani", "indian", "bangladeshi", "nigerian", "chinese", 
                         "filipino", "american", "british", "canadian", "australian"]
    for word in nationality_words:
        if word in q:
            mapping = {
                "pakistani": "pakistan", "indian": "india", "bangladeshi": "bangladesh",
                "nigerian": "nigeria", "chinese": "china", "filipino": "philippines",
                "american": "usa", "british": "uk", "canadian": "canada", "australian": "australia"
            }
            return mapping.get(word, word)
    
    return None

def detect_visa_type(query: str) -> str:
    q = query.lower()
    if any(word in q for word in ["student", "study", "university", "college", "course", "education"]):
        return "student"
    elif any(word in q for word in ["work", "job", "employment", "working", "career", "skilled"]):
        return "work"
    else:
        return "tourist"

def is_ban_question(query: str) -> bool:
    q = query.lower()
    ban_keywords = ["ban", "banned", "restrict", "restriction", "allowed", "allow", "permit", "permission", "issuing visas"]
    return any(word in q for word in ban_keywords)

def is_ambiguous_nationality(query: str) -> bool:
    ambiguous = ["asian", "african", "european", "middle eastern", "arab"]
    return any(term in query.lower() for term in ambiguous)

def get_requirement_fallback(country: str, visa_type: str) -> Optional[Dict]:
    if country in KNOWN_REQUIREMENTS:
        country_reqs = KNOWN_REQUIREMENTS[country]
        if visa_type in country_reqs:
            return country_reqs[visa_type]
    return None

def generate_ban_answer(query: str, country: str, nationality: Optional[str], web_context: str) -> str:
    prompt = f"""You are answering a question about travel bans or visa restrictions.

USER QUESTION: {query}
DESTINATION: {country.upper()}
NATIONALITY: {nationality.upper() if nationality else 'Not specified'}

WEB SEARCH RESULTS:
{web_context[:2000]}

CRITICAL INSTRUCTIONS:
1. Answer based ONLY on the web search results above
2. If the search results indicate a ban or restriction, clearly state it
3. If NO ban is mentioned in the results, state "Based on current search results, there is NO evidence of a ban. Visas appear to be available."
4. Provide specific details from the search results (dates, conditions, exceptions)
5. DO NOT invent information - only use what's in the context
6. If search results are insufficient, honestly state that

FORMAT:
## 🛂 Travel Status: {country.upper()} Visas for {nationality.upper() if nationality else 'Travelers'}

### 📋 Current Status
[Clear statement about ban/restriction based on search results]

### 📝 Details from Search
[Specific information found in web results]

### 🔗 Sources
[Sources used]

### ⚠️ Important
Always verify with official {country.upper()} government sources as policies change.
"""
    
    try:
        answer = llm.invoke(prompt)
        return answer
    except Exception:
        return f"""## 🛂 Travel Status: {country.upper()} Visas for {nationality.upper() if nationality else 'Travelers'}

### 📋 Current Status
Based on available information, I could not find specific ban/restriction details.

### 🔗 Recommendation
- Check the official {country.upper()} immigration website
- Contact the nearest {country.upper()} embassy or consulate
- Verify current policies before making travel plans

### ⚠️ Important
Travel policies can change. Always verify with official sources.
"""

def generate_complete_answer(query: str, country: str, nationality: Optional[str], 
                             visa_type: str, is_language_question: bool,
                             is_ambiguous: bool, web_context: str = "",
                             is_ban: bool = False) -> str:
    
    if is_ban and web_context:
        return generate_ban_answer(query, country, nationality, web_context)
    
    if is_ambiguous:
        return f"""## ⚠️ Please Specify Your Nationality

You mentioned an ambiguous nationality (e.g., "Asian", "African"). Visa requirements vary significantly by specific country.

**Please clarify which country's passport you hold** (e.g., "Pakistani", "Indian", "Chinese", "Nigerian").

---
### General {visa_type.title()} Visa Information for {country.title()}:

{web_context[:800] if web_context else 'Please check official sources for requirements.'}

---
*Rephrase your question with your specific nationality for accurate information.*
"""
    
    schengen_note = ""
    if country and country in SCHENGEN_COUNTRIES:
        schengen_note = f"\n**🌍 Schengen Note:** {country.title()} is part of the Schengen Area. A visa issued by {country.title()} allows travel to all 27 Schengen countries.\n"
    
    fallback_data = get_requirement_fallback(country, visa_type)
    
    if web_context and not fallback_data:
        prompt = f"""You are a visa expert. Answer based ONLY on the context provided.

QUESTION: {query}
COUNTRY: {country}
VISA TYPE: {visa_type}

WEB SEARCH CONTEXT:
{web_context[:1800]}

CRITICAL: Only state information EXPLICITLY present in the context.
If information is not in context, say "Not specified in search results - check official sources."

FORMAT:
## 📋 {visa_type.title()} Visa Requirements for {country.title()}

### ✅ Required Documents
[Only from context]

### 💰 Financial Requirements
[Only from context]

### ⏱️ Processing Time
[Only from context]

### 💵 Visa Fees
[Only from context]

### 🔗 Sources
Based on web search results.

### ⚠️ Official Verification
Always confirm with official {country.upper()} government sources.
"""
        try:
            answer = llm.invoke(prompt)
            return answer
        except Exception:
            pass
    
    if fallback_data:
        ielts_section = ""
        if is_language_question or visa_type == "student":
            ielts_req = "YES" if fallback_data.get('ielts_required') else "NO"
            ielts_section = f"""
### 🌐 Language Requirements (IELTS/English)
- **IELTS Required:** {ielts_req}
"""
            if fallback_data.get('ielts_required'):
                ielts_section += f"""- **Minimum Overall Score:** {fallback_data.get('min_score', 'N/A')}
- **Minimum Band Scores:** {fallback_data.get('min_bands', 'N/A')}
- **Test Type:** {fallback_data.get('test_type', 'N/A')}
- **Alternative Tests:** {fallback_data.get('alternatives', 'N/A')}
- **Note:** {fallback_data.get('note', 'N/A')}
"""
        
        nationality_note = ""
        if nationality:
            nationality_note = f"\n\n**📌 Note for {nationality.title()} Nationals:** Requirements apply as stated."
        
        answer = f"""## 📋 {visa_type.title()} Visa Requirements for {country.title()}
{schengen_note}
### ✅ Required Documents
{fallback_data.get('documents', 'Check official sources for complete document list.')}

### 💰 Financial Requirements
{fallback_data.get('financial', 'Check official sources for current financial requirements.')}

### ⏱️ Processing Time
{fallback_data.get('processing', 'Check official sources for current processing times.')}

### 💵 Visa Fees
{fallback_data.get('fees', 'Check official sources for current visa fees.')}
{ielts_section}
### 📌 Additional Information
{fallback_data.get('note', 'Please verify all requirements with official sources.')}{nationality_note}

### 🔗 Official Source
**{country.title()} Immigration Authority**  
[{fallback_data.get('official_url', '#')}]({fallback_data.get('official_url', '#')})

### ⚠️ Official Verification
Always confirm with the official immigration website before applying, as requirements may change.
"""
        return answer
    
    return f"""## 📋 {visa_type.title()} Visa Requirements for {country.title()}

### ✅ Required Documents
- Valid passport (minimum 6 months validity)
- Completed visa application form
- Recent passport-sized photographs
- Travel itinerary and proof of accommodation
- Proof of sufficient financial means
- Travel/medical insurance

### 💰 Financial Requirements
Please check the official {country.title()} immigration website for specific financial requirements.

### ⏱️ Processing Time
Processing times vary. Check official sources for current estimates.

### 🔗 Official Source
Please visit the official {country.title()} immigration website for complete and current requirements.

### ⚠️ Official Verification
Always confirm requirements with official government sources before applying.
"""

# -----------------------------
# CHAT INTERFACE
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask any visa question (e.g., 'Do Pakistani students require IELTS for Ireland?' or 'Is there any ban on Dubai visas for Pakistanis?')")

if prompt:
    st.session_state.button_counter += 1
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    country = detect_country(prompt)
    nationality = detect_nationality(prompt)
    visa_type = detect_visa_type(prompt)
    is_language_question = any(word in prompt.lower() for word in ["ielts", "english", "language", "toefl", "pte"])
    is_ambiguous = is_ambiguous_nationality(prompt)
    is_ban = is_ban_question(prompt)
    
    with st.chat_message("assistant"):
        answer_container = st.empty()
        status_container = st.empty()
        
        if not country:
            answer = "❓ Please specify a country (e.g., 'Ireland', 'UK', 'Canada', 'UAE', 'Japan', etc.)"
            answer_container.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.stop()
        
        status_container.info(f"🔍 Searching web for {country.upper()} visa information...")
        
        fallback_data = get_requirement_fallback(country, visa_type)
        
        web_context = ""
        if is_ban or not fallback_data or is_ambiguous:
            web_results = search_visa_information_parallel(prompt, country, nationality, visa_type, is_language_question, is_ban)
            if web_results:
                context_parts = []
                for r in web_results[:4]:
                    context_parts.append(f"SOURCE: {r.get('title', 'N/A')}\nURL: {r.get('url', 'N/A')}\nCONTENT: {r.get('body', '')[:800]}\n---")
                web_context = "\n".join(context_parts)
        
        answer = generate_complete_answer(prompt, country, nationality, visa_type, 
                                          is_language_question, is_ambiguous, web_context, is_ban)
        
        answer_container.markdown(answer)
        st.session_state.last_answer = answer
        status_container.empty()
        
        col1, col2, col3 = st.columns([0.7, 0.15, 0.15])
        with col2:
            if st.button("📋 Copy", key=f"copy_btn_{st.session_state.button_counter}", use_container_width=True):
                success = copy_to_clipboard(answer)
                if success:
                    st.toast("✅ Copied!", icon="📋")
        with col3:
            st.download_button(
                label="⬇️ Save",
                data=answer,
                file_name=f"visa_{country}_{visa_type}.txt",
                mime="text/plain",
                key=f"download_btn_{st.session_state.button_counter}",
                use_container_width=True
            )

        st.session_state.messages.append({"role": "assistant", "content": answer})

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown("## 🌍 Global Visa Assistant")
    st.markdown("### ✅ Works for ANY Country")
    
    st.markdown("**Ask about:**")
    st.markdown("- Visa requirements (all types)")
    st.markdown("- Travel bans & restrictions")
    st.markdown("- IELTS/Language requirements")
    st.markdown("- Processing times & fees")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.search_cache = {}
        st.session_state.button_counter = 0
        st.rerun()
    
    st.caption(f"📝 Messages: {len(st.session_state.messages)}")