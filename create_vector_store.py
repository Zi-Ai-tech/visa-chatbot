"""
create_vector_store.py
Creates a FAISS vector store from visa-related documents for RAG capabilities.
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    UnstructuredMarkdownLoader,
    CSVLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# -----------------------------
# CONFIGURATION
# -----------------------------
# Paths
CHUNKS_FILE = "chunks.pkl"
VECTOR_STORE_PATH = "visa_vector_store"
DOCUMENTS_DIR = "documents"  # Directory containing source documents

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Text splitting parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# -----------------------------
# VISA KNOWLEDGE BASE (Fallback content)
# -----------------------------
VISA_KNOWLEDGE_DOCS = [
    {
        "title": "Ireland Visa Requirements Overview",
        "content": """
Ireland Visa Requirements - Official Information from Irish Immigration Service

Tourist Visa (Short Stay 'C' Visa):
- Valid passport with at least 6 months validity beyond intended stay
- Completed visa application form
- Two passport-sized photographs
- Proof of financial means (bank statements, payslips)
- Travel itinerary and proof of accommodation
- Travel/medical insurance
- No English language test required for tourist visas
- Processing time: approximately 8 weeks for some nationalities

Student Visa (Study Visa):
- Valid passport
- Letter of acceptance from recognized Irish institution
- Proof of English proficiency (IELTS Academic typically 6.0-6.5)
- Alternative tests: TOEFL iBT (80-90), Cambridge CAE/CPE, PTE Academic (59-63)
- Proof of funds: €7,000-€10,000 per year
- Medical insurance
- Processing time: 4-8 weeks

Work Visa (Employment Permit):
- Job offer from Irish employer
- Employment permit from Department of Enterprise, Trade and Employment
- Critical Skills Employment Permit for eligible occupations
- No mandatory English test, but employer may require proof
- Processing time varies by permit type

Official Source: www.irishimmigration.ie
""",
        "source": "Irish Immigration Service",
        "url": "https://www.irishimmigration.ie"
    },
    {
        "title": "UK Visa Requirements",
        "content": """
UK Visas and Immigration - Official Requirements

Standard Visitor Visa (Tourist):
- Valid passport
- Completed online application
- Proof of funds for duration of stay
- Travel itinerary
- No English test required
- Processing: approximately 3 weeks

Student Visa:
- Confirmation of Acceptance for Studies (CAS)
- IELTS for UKVI Academic (minimum 5.5-7.0 depending on course level)
- B1 level (IELTS 4.0) for below degree, B2 (IELTS 5.5) for degree level
- Alternative SELT tests: Trinity College London, Pearson PTE Academic UKVI, LanguageCert
- Proof of funds: £1,334/month in London, £1,023/month outside London
- Immigration Health Surcharge payment

Skilled Worker Visa:
- Certificate of Sponsorship from licensed employer
- B1 English level (IELTS 4.0+ in all components)
- Salary threshold requirements
- Processing: 3-8 weeks

Official Source: www.gov.uk
""",
        "source": "UK Visas and Immigration",
        "url": "https://www.gov.uk"
    },
    {
        "title": "Canada Visa Requirements",
        "content": """
Immigration, Refugees and Citizenship Canada - Visa Requirements

Visitor Visa (Temporary Resident Visa):
- Valid passport
- Completed application forms
- Proof of funds
- Purpose of visit documentation
- No language test required
- Processing time varies by country

Study Permit:
- Letter of acceptance from Designated Learning Institution (DLI)
- IELTS Academic typically 6.0-6.5 overall
- Student Direct Stream (SDS) requires IELTS 6.0 in each band
- Alternative tests: TOEFL iBT, CAEL, PTE Academic, Duolingo (some institutions)
- Proof of funds: tuition + $10,000 CAD living expenses
- Provincial Attestation Letter (PAL) required

Work Permit:
- Job offer from Canadian employer
- Labour Market Impact Assessment (LMIA) may be required
- Express Entry requires CLB 7 (IELTS 6.0 in each ability)
- Alternative tests: CELPIP General, TEF Canada (French)
- Processing time varies by program

Official Source: www.canada.ca
""",
        "source": "Immigration, Refugees and Citizenship Canada",
        "url": "https://www.canada.ca"
    },
    {
        "title": "Australia Visa Requirements",
        "content": """
Department of Home Affairs - Australia Visa Requirements

Visitor Visa (subclass 600):
- Valid passport
- Completed application
- Proof of funds
- Health and character requirements
- No English test required
- Processing time varies

Student Visa (subclass 500):
- Confirmation of Enrolment (CoE) from CRICOS-registered institution
- IELTS Academic: 5.5-6.5 overall (minimum 5.0-6.0 each band)
- Alternative tests: TOEFL iBT, PTE Academic, CAE, OET
- Proof of funds: tuition + living costs
- Overseas Student Health Cover (OSHC)
- Genuine Temporary Entrant requirement

Work Visa (Temporary Skill Shortage - subclass 482):
- Nominated occupation on skilled occupation list
- IELTS 5.0 overall (minimum 4.5 each band) or equivalent
- Skills assessment may be required
- Health insurance

Skilled Migration Visas:
- Points-tested system
- Competent English: IELTS 6.0 in each band
- Skills assessment required

Official Source: www.homeaffairs.gov.au
""",
        "source": "Department of Home Affairs",
        "url": "https://www.homeaffairs.gov.au"
    },
    {
        "title": "USA Visa Requirements",
        "content": """
USCIS and Department of State - USA Visa Requirements

B-2 Tourist Visa:
- Valid passport (6 months beyond intended stay)
- DS-160 online application
- Visa interview at US embassy/consulate
- Proof of ties to home country
- No English test required
- Processing time varies by location

F-1 Student Visa:
- I-20 form from SEVP-approved school
- IELTS Academic or TOEFL iBT (requirements vary by institution)
- Most institutions require TOEFL 80+ or IELTS 6.5+
- Alternative tests: PTE Academic, Duolingo English Test, Cambridge English
- SEVIS fee payment
- Proof of funds for tuition and living expenses

H-1B Work Visa:
- Job offer from US employer in specialty occupation
- Bachelor's degree or equivalent required
- No mandatory English test (employer-specific)
- Annual cap and lottery system
- Processing: 3-6 months (premium processing available)

Official Source: www.uscis.gov and travel.state.gov
""",
        "source": "USCIS",
        "url": "https://www.uscis.gov"
    },
    {
        "title": "IELTS Requirements Summary",
        "content": """
IELTS Requirements for Major English-Speaking Countries

Ireland:
- Tourist: No IELTS required
- Student: IELTS Academic 6.0-6.5 (min 5.5-6.0 bands)
- Work: No general requirement (employer-specific)

United Kingdom:
- Tourist: No IELTS required
- Student: IELTS for UKVI Academic 5.5-7.0 (min 4.0-5.5 bands)
- Work: Skilled Worker requires B1 level (IELTS 4.0+)

Canada:
- Tourist: No IELTS required
- Student: IELTS Academic 6.0-6.5 (SDS: 6.0 each band)
- Work: Express Entry requires CLB 7 (IELTS 6.0 each band)

Australia:
- Tourist: No IELTS required
- Student: IELTS Academic 5.5-6.5 (min 5.0-6.0 bands)
- Work: TSS visa 5.0 overall, skilled visas 6.0 each band

USA:
- Tourist: No IELTS required
- Student: IELTS Academic 6.5+ (varies by institution)
- Work: No general requirement (employer-specific)
""",
        "source": "Compiled Official Immigration Data",
        "url": "Multiple Sources"
    },
    {
        "title": "Pakistani Nationals - Country-Specific Notes",
        "content": """
Special Considerations for Pakistani Nationals

Ireland:
- Tourist visa required, no English test, processing ~8 weeks
- Student visa requires IELTS 6.0-6.5, proof of funds €7,000-€10,000
- Work permit required, English requirement job-dependent

United Kingdom:
- Standard Visitor visa required, no English test, processing ~3 weeks
- Student visa requires IELTS for UKVI, funds £1,334/month (London)
- Skilled Worker visa requires B1 English (IELTS 4.0+)

Canada:
- Visitor visa required for most purposes
- Study permit requires IELTS 6.0-6.5 (SDS: 6.0 each band)
- Express Entry requires CLB 7 for Federal Skilled Worker

Australia:
- Visitor visa required, no English test
- Student visa requires IELTS 5.5-6.5 depending on course
- Skilled visas require Competent English (IELTS 6.0 each band)

USA:
- B-2 tourist visa requires interview, no English test
- F-1 student visa English requirement set by institution
- H-1B work visa has annual cap, no mandatory English test
""",
        "source": "Compiled Official Immigration Data",
        "url": "Multiple Sources"
    },
    {
        "title": "Indian Nationals - Country-Specific Notes",
        "content": """
Special Considerations for Indian Nationals

Ireland:
- Tourist visa required, no English test
- Student visa requires IELTS 6.0+, some accept Medium of Instruction certificate
- Processing time: 4-8 weeks for student visas
- Critical Skills Employment Permit available

United Kingdom:
- Standard Visitor visa required
- Student visa requires IELTS for UKVI
- Skilled Worker visa requires B1 English level
- Youth Mobility Scheme available (limited places)

Canada:
- Student Direct Stream (SDS) available for faster processing
- IELTS 6.0 in each band required for SDS
- Express Entry popular pathway for skilled migration

Australia:
- Visitor visa required
- Student visa English requirements apply
- Skilled migration points-tested system

USA:
- B-2 tourist visa requires interview
- F-1 student visa popular pathway
- H-1B visa subject to annual cap
""",
        "source": "Compiled Official Immigration Data",
        "url": "Multiple Sources"
    }
]

# -----------------------------
# DOCUMENT LOADING FUNCTIONS
# -----------------------------
def load_documents_from_directory(directory_path: str) -> List[Document]:
    """Load documents from a directory with various file types."""
    documents = []
    
    if not os.path.exists(directory_path):
        print(f"⚠️ Directory '{directory_path}' not found. Skipping file loading.")
        return documents
    
    # Load text files
    txt_loader = DirectoryLoader(
        directory_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    try:
        documents.extend(txt_loader.load())
        print(f"✅ Loaded TXT files from '{directory_path}'")
    except Exception as e:
        print(f"⚠️ Error loading TXT files: {e}")
    
    # Load PDF files
    pdf_loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    try:
        documents.extend(pdf_loader.load())
        print(f"✅ Loaded PDF files from '{directory_path}'")
    except Exception as e:
        print(f"⚠️ Error loading PDF files: {e}")
    
    # Load Markdown files
    md_loader = DirectoryLoader(
        directory_path,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader
    )
    try:
        documents.extend(md_loader.load())
        print(f"✅ Loaded Markdown files from '{directory_path}'")
    except Exception as e:
        print(f"⚠️ Error loading Markdown files: {e}")
    
    return documents

def create_documents_from_knowledge_base() -> List[Document]:
    """Create LangChain Document objects from the built-in knowledge base."""
    documents = []
    
    for doc_data in VISA_KNOWLEDGE_DOCS:
        # Create metadata
        metadata = {
            "title": doc_data.get("title", ""),
            "source": doc_data.get("source", ""),
            "url": doc_data.get("url", "")
        }
        
        # Create document with content and metadata
        doc = Document(
            page_content=doc_data["content"],
            metadata=metadata
        )
        documents.append(doc)
    
    print(f"✅ Created {len(documents)} documents from built-in knowledge base")
    return documents

def load_or_create_chunks(
    chunks_file: str = CHUNKS_FILE,
    documents_dir: str = DOCUMENTS_DIR,
    use_builtin_knowledge: bool = True
) -> List[Document]:
    """Load chunks from file or create from documents."""
    
    # Try to load existing chunks
    if os.path.exists(chunks_file):
        print(f"📂 Loading existing chunks from '{chunks_file}'...")
        try:
            with open(chunks_file, "rb") as f:
                chunks = pickle.load(f)
            print(f"✅ Loaded {len(chunks)} chunks from file")
            return chunks
        except Exception as e:
            print(f"⚠️ Error loading chunks: {e}")
            print("Creating new chunks...")
    
    # Load documents from directory
    documents = []
    if os.path.exists(documents_dir):
        documents = load_documents_from_directory(documents_dir)
        if documents:
            print(f"✅ Loaded {len(documents)} documents from '{documents_dir}'")
    
    # Add built-in knowledge base documents
    if use_builtin_knowledge or not documents:
        builtin_docs = create_documents_from_knowledge_base()
        documents.extend(builtin_docs)
        print(f"✅ Total documents: {len(documents)}")
    
    if not documents:
        raise ValueError("No documents found to create vector store!")
    
    # Split documents into chunks
    print(f"✂️ Splitting {len(documents)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Save chunks for future use
    print(f"💾 Saving chunks to '{chunks_file}'...")
    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)
    print(f"✅ Chunks saved")
    
    return chunks

# -----------------------------
# MAIN FUNCTION
# -----------------------------
def create_vector_store(
    chunks_file: str = CHUNKS_FILE,
    vector_store_path: str = VECTOR_STORE_PATH,
    embedding_model: str = EMBEDDING_MODEL,
    force_recreate: bool = False
) -> Optional[FAISS]:
    """Create and save FAISS vector store."""
    
    print("\n" + "="*60)
    print("🚀 VISA RAG VECTOR STORE CREATOR")
    print("="*60 + "\n")
    
    # Check if vector store already exists
    if os.path.exists(vector_store_path) and not force_recreate:
        faiss_file = os.path.join(vector_store_path, "index.faiss")
        pkl_file = os.path.join(vector_store_path, "index.pkl")
        if os.path.exists(faiss_file) and os.path.exists(pkl_file):
            print(f"⚠️ Vector store already exists at '{vector_store_path}'")
            response = input("Do you want to recreate it? (y/N): ").lower()
            if response != 'y':
                print("❌ Cancelled.")
                return None
    
    try:
        # Step 1: Load or create chunks
        print("\n📚 STEP 1: Loading/Creating document chunks...")
        print("-"*40)
        chunks = load_or_create_chunks(chunks_file)
        
        # Step 2: Initialize embeddings
        print("\n🔧 STEP 2: Initializing embedding model...")
        print("-"*40)
        print(f"Model: {embedding_model}")
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ Embeddings initialized")
        
        # Step 3: Create vector store
        print("\n📊 STEP 3: Creating FAISS vector store...")
        print("-"*40)
        print(f"Processing {len(chunks)} chunks...")
        
        # Create vector store with progress indication
        vector_store = FAISS.from_documents(chunks, embeddings)
        print("✅ Vector store created successfully")
        
        # Step 4: Save vector store
        print("\n💾 STEP 4: Saving vector store...")
        print("-"*40)
        os.makedirs(vector_store_path, exist_ok=True)
        vector_store.save_local(vector_store_path)
        print(f"✅ Vector store saved to '{vector_store_path}'")
        
        # Step 5: Verify
        print("\n🔍 STEP 5: Verification...")
        print("-"*40)
        # Test loading
        test_store = FAISS.load_local(
            vector_store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✅ Vector store verified (contains {test_store.index.ntotal} vectors)")
        
        # Summary
        print("\n" + "="*60)
        print("✅ VECTOR STORE CREATED SUCCESSFULLY!")
        print("="*60)
        print(f"📁 Location: {vector_store_path}")
        print(f"📊 Total chunks: {len(chunks)}")
        print(f"🔢 Embedding dimension: 384")
        print(f"💾 Files created:")
        print(f"   - {vector_store_path}/index.faiss")
        print(f"   - {vector_store_path}/index.pkl")
        print("="*60 + "\n")
        
        return vector_store
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

# -----------------------------
# TEST FUNCTION
# -----------------------------
def test_vector_store(vector_store_path: str = VECTOR_STORE_PATH):
    """Test the created vector store with sample queries."""
    print("\n🧪 Testing vector store with sample queries...")
    print("-"*40)
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
        vector_store = FAISS.load_local(
            vector_store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        test_queries = [
            "IELTS requirement for Ireland student visa",
            "UK work visa English language",
            "Canada study permit requirements",
            "Australia tourist visa documents",
            "USA F1 visa IELTS score"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            docs = vector_store.similarity_search(query, k=2)
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                title = doc.metadata.get('title', 'Untitled')
                print(f"   Result {i}: {title} (Source: {source})")
                print(f"   Preview: {doc.page_content[:100]}...")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

# -----------------------------
# COMMAND LINE INTERFACE
# -----------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create FAISS vector store for Visa RAG Chatbot")
    parser.add_argument("--force", "-f", action="store_true", help="Force recreate vector store")
    parser.add_argument("--test", "-t", action="store_true", help="Test vector store after creation")
    parser.add_argument("--chunks", "-c", default=CHUNKS_FILE, help="Path to chunks file")
    parser.add_argument("--output", "-o", default=VECTOR_STORE_PATH, help="Output directory for vector store")
    
    args = parser.parse_args()
    
    # Create vector store
    vector_store = create_vector_store(
        chunks_file=args.chunks,
        vector_store_path=args.output,
        force_recreate=args.force
    )
    
    # Test if requested and successful
    if args.test and vector_store:
        test_vector_store(args.output)