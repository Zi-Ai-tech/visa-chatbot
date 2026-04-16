# 🌍 Global Visa Assistant

Real-time visa information chatbot that works for **ANY country** using web search and verified databases.

## Features

- ✅ **190+ Countries Supported** - Ask about any country worldwide
- 🔍 **Real-time Web Search** - Gets current visa information
- 📚 **Verified Database** - Fast answers for major destinations (Ireland, UK, Canada, Australia, USA, UAE, NZ)
- 🛂 **Ban/Restriction Detection** - Answers questions about travel bans
- 🌐 **IELTS/Language Requirements** - Specific test scores when asked
- 📋 **Copy & Save** - Save answers for later reference

## Supported Countries (Verified Database)

| Country | Visa Types |
|---------|------------|
| 🇮🇪 Ireland | Student, Tourist, Work |
| 🇬🇧 UK | Student, Tourist |
| 🇨🇦 Canada | Student, Tourist |
| 🇦🇺 Australia | Student, Tourist |
| 🇺🇸 USA | Student, Tourist |
| 🇳🇿 New Zealand | Student, Tourist |
| 🇦🇪 UAE | Student, Tourist, Work |

*Other countries supported via real-time web search!*

## Installation

```bash
# Clone the repository
git clone https://github.com/Zi-Ai-tech/visa-rag-chatbot.git
cd visa-rag-chatbot

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Install Ollama (for LLM)
# Visit https://ollama.ai and install
ollama pull llama3.2