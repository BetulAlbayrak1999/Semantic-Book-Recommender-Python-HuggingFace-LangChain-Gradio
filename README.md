# 📚 Semantic Book Recommender System

A smart book recommendation system that uses **semantic search**, **emotion analysis**, and **vector embeddings** to find the perfect books based on user descriptions and emotional preferences.
<img width="1917" height="870" alt="image" src="https://github.com/user-attachments/assets/a2256968-6b6d-4263-a421-c3079860d0af" />


## ✨ Features

- **Semantic Search**: Uses HuggingFace embeddings to understand book descriptions contextually
- **Emotion-Based Filtering**: Analyze book descriptions and filter by emotional tone (Happy, Sad, Angry, Surprising, Suspenseful)
- **Category Filtering**: Filter recommendations by book categories
- **Vector Database**: Leverages Chroma for fast similarity search across thousands of books
- **Interactive Web UI**: Built with Gradio for an intuitive user experience
- **GPU Acceleration**: CUDA-enabled for fast inference with NVIDIA GPUs

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- NVIDIA GPU - Optional but recommended
- pip and virtual environment

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/semantic-book-recommender.git
cd semantic-book-recommender
```

2. **Create a virtual environment**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # macOS/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Add your API keys if needed
```

5. **Run the application**
```bash
python gradio-dashboard.py
```

The web interface will be available at `http://localhost:7860`

## 📋 Requirements
```
pandas==2.0.0+
numpy==1.24.0+
torch==2.0.0+
torchvision==0.15.0+
torchaudio==2.0.0+
langchain==0.1.0+
langchain-community==0.1.0+
langchain-huggingface==0.0.1+
langchain-chroma==0.1.0+
chromadb==1.3.0+
sentence-transformers==2.2.0+
gradio==4.0.0+
python-dotenv==1.0.0+
humanfriendly==10.0+
```

## 📁 Project Structure
```
semantic-book-recommender/
├── gradio-dashboard.py          # Main Gradio application
├── books_with_emotions.csv      # Book dataset with emotion scores
├── books_cleaned.csv            # Cleaned book data
├── tagged_description.txt       # Book descriptions for embeddings
├── chroma_db/                   # Vector database (persisted)
├── .env                         # Environment variables
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── .gitignore                   # Git ignore rules
```

## How It Works

### 1. **Data Processing**
- Books dataset is loaded with emotion scores (joy, sadness, anger, fear, surprise)
- Descriptions are processed and split into chunks
- Large thumbnail URLs are generated for display

### 2. **Semantic Embeddings**
- Uses `sentence-transformers/all-MiniLM-L6-v2` to convert text to embeddings
- Embeddings are stored in Chroma vector database
- Enables semantic similarity search

### 3. **Recommendation Pipeline**
```
User Query
    ↓
Semantic Search (Top 50 results)
    ↓
Category Filtering (if selected)
    ↓
Emotion Sorting (by tone preference)
    ↓
Return Top 16 Recommendations
```

### 4. **Emotion Filtering**
- **Happy** → Sorts by `joy` score
- **Sad** → Sorts by `sadness` score
- **Angry** → Sorts by `anger` score
- **Surprising** → Sorts by `surprise` score
- **Suspenseful** → Sorts by `fear` score

## 💻 Usage

### Web Interface

1. **Enter Book Description**: Describe what you're looking for (e.g., "A story about love and conflict")
2. **Select Category**: Choose from available book categories or select "All"
3. **Select Emotional Tone**: Pick the emotional vibe you want
4. **Click "Find Recommendations"**: View the recommended books

### Example Queries

- "A thrilling mystery novel with unexpected plot twists"
- "A heartwarming story about friendship"
- "A sci-fi adventure exploring new worlds"

## 🔧 Configuration

### Modify Recommendation Parameters

In `gradio-dashboard.py`, adjust:
```python
def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,      # Initial search results
    final_top_k: int = 16         # Final recommendations shown
) -> pd.DataFrame:
```

### Change Embedding Model
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # Change this
)
```

Popular alternatives:
- `sentence-transformers/all-mpnet-base-v2` (better quality, slower)
- `sentence-transformers/paraphrase-MiniLM-L6-v2` (balanced)

2. **Batch Processing**: For large datasets, process in batches

3. **Vector Database**: Chroma is persisted in `./chroma_db/` for faster subsequent loads

4. **Model Selection**: Use smaller models (`all-MiniLM-L6-v2`) for speed vs. accuracy trade-off

## 📊 Dataset

The system uses a books dataset with the following columns:
- `isbn13`: Book ISBN
- `title`: Book title
- `authors`: Author(s) name
- `description`: Book description
- `simple_categories`: Book category
- `thumbnail`: Cover image URL
- `joy`, `sadness`, `anger`, `fear`, `surprise`: Emotion scores (0-1)

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/feature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/feature`)
5. Open a Pull Request

- UI: [Gradio](https://gradio.app/)

---

**Made with ❤️ by [Your Name]**

⭐ If you find this project useful, please consider giving it a star!
