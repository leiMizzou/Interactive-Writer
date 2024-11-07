# InterActWriter

![InterActWriter Logo](https://github.com/yourusername/InterActWriter/raw/main/logo.png)

InterActWriter is an advanced tool designed to streamline the process of generating comprehensive academic surveys from a vast collection of arXiv papers. Leveraging state-of-the-art natural language processing techniques, including SentenceTransformers and FAISS for efficient embedding and indexing, InterActWriter automates the creation of structured and well-evaluated literature reviews on any given topic.

## Features

- **Seamless Data Import**: Efficiently import and manage large datasets of arXiv papers using SQLite.
- **Advanced Embedding Generation**: Utilize powerful SentenceTransformer models to generate meaningful embeddings for paper titles and abstracts.
- **High-Performance Indexing**: Build and manage FAISS indexes for rapid similarity searches, enabling quick retrieval of relevant papers.
- **Automated Survey Generation**: Automatically draft outlines and subsections for academic surveys based on specified topics.
- **Comprehensive Evaluation**: Assess generated surveys against predefined criteria such as Coverage, Structure, and Relevance using AI-driven evaluations.
- **Scalable and Efficient**: Designed to handle large volumes of data with optimized batch processing and multi-threading support.

## Table of Contents

- [InterActWriter](#interactwriter)
  - [Features](#features)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [1. Importing Papers into SQLite](#1-importing-papers-into-sqlite)
    - [2. Generating Embeddings and Building FAISS Indexes](#2-generating-embeddings-and-building-faiss-indexes)
    - [3. Generating and Evaluating Academic Surveys](#3-generating-and-evaluating-academic-surveys)
  - [Project Structure](#project-structure)
  - [Configuration](#configuration)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)

## Installation

### Prerequisites

- **Python 3.8+**: Ensure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).
- **pip**: Python package installer (usually comes with Python).
- **Git**: To clone the repository.

### Clone the Repository

```bash
git clone https://github.com/yourusername/InterActWriter.git
cd InterActWriter
```

### Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The `sqlite3` module is part of Python's standard library and does not require separate installation.

## Usage

InterActWriter consists of three main scripts:

1. **Importing Papers into SQLite**
2. **Generating Embeddings and Building FAISS Indexes**
3. **Generating and Evaluating Academic Surveys**

Ensure you follow the steps in the correct order to achieve optimal results.

### 1. Importing Papers into SQLite

First, import your arXiv papers into the SQLite database.

#### Prepare Your Data

Ensure your arXiv papers are stored in a JSON file (`arxiv_papers.json`). The file can be in one of the following formats:

- **JSON Lines**: Each line is a separate JSON object.
- **JSON Array**: A single JSON array containing all paper objects.

#### Run the Import Script

```bash
python import_papers_sqlite.py --input_data='arxiv_papers.json' --db_path='./database' --batch_size=1000
```

**Arguments**:

- `--input_data`: Path to your input JSON file containing arXiv papers.
- `--db_path`: Directory where the SQLite database will be stored.
- `--batch_size`: Number of papers to insert per batch (default: 1000).

**Example**:

```bash
python import_papers_sqlite.py --input_data='data/arxiv_papers.json' --db_path='./database' --batch_size=1000
```

Upon successful execution, you should see:

```
导入论文: 100%|██████████| X/Y [00:XX<00:00, XXX.XX批次/秒]
论文导入成功。
```

### 2. Generating Embeddings and Building FAISS Indexes

Next, generate embeddings for the paper titles and abstracts, and build FAISS indexes for efficient retrieval.

#### Run the Embedding and Indexing Script

```bash
python generate_embeddings_and_index.py \
    --db_path='./database' \
    --embedding_model='nomic-ai/nomic-embed-text-v1' \
    --index_path_title='./database/faiss_paper_title_embeddings.bin' \
    --index_path_abs='./database/faiss_paper_abs_embeddings.bin' \
    --id_to_index_path_title='./database/arxivid_to_index_title.json' \
    --index_to_id_path_title='./database/index_to_arxivid_title.json' \
    --id_to_index_path_abs='./database/arxivid_to_index_abs.json' \
    --index_to_id_path_abs='./database/index_to_arxivid_abs.json' \
    --batch_size=64
```

**Arguments**:

- `--db_path`: Path to the SQLite database directory.
- `--embedding_model`: SentenceTransformer model name (e.g., `'nomic-ai/nomic-embed-text-v1'`).
- `--index_path_title`: Path to save the FAISS index for titles.
- `--index_path_abs`: Path to save the FAISS index for abstracts.
- `--id_to_index_path_title`: Path to save the ID-to-index mapping for titles.
- `--index_to_id_path_title`: Path to save the index-to-ID mapping for titles.
- `--id_to_index_path_abs`: Path to save the ID-to-index mapping for abstracts.
- `--index_to_id_path_abs`: Path to save the index-to-ID mapping for abstracts.
- `--batch_size`: Batch size for embedding generation (default: 64).

**Example**:

```bash
python generate_embeddings_and_index.py \
    --db_path='./database' \
    --embedding_model='nomic-ai/nomic-embed-text-v1' \
    --index_path_title='./database/faiss_paper_title_embeddings.bin' \
    --index_path_abs='./database/faiss_paper_abs_embeddings.bin' \
    --id_to_index_path_title='./database/arxivid_to_index_title.json' \
    --index_to_id_path_title='./database/index_to_arxivid_title.json' \
    --id_to_index_path_abs='./database/arxivid_to_index_abs.json' \
    --index_to_id_path_abs='./database/index_to_arxivid_abs.json' \
    --batch_size=64
```

Upon successful execution, you should see:

```
Generating title embeddings...
生成标题嵌入: 100%|██████████| X/Y [00:XX<00:00, XXX.XX批次/秒]
Generating abstract embeddings...
生成摘要嵌入: 100%|██████████| X/Y [00:XX<00:00, XXX.XX批次/秒]
Building FAISS index for titles...
Building FAISS index for abstracts...
FAISS indexes saved.
ID to Index mappings saved.
```

### 3. Generating and Evaluating Academic Surveys

Finally, generate an academic survey on a specific topic and evaluate its quality.

#### Run the Main Program

```bash
python main.py \
    --db_path='./database' \
    --embedding_model='nomic-ai/nomic-embed-text-v1' \
    --saving_path='./output/' \
    --model='gpt-4' \
    --topic='Quantum Computing' \
    --section_num=7 \
    --subsection_len=700 \
    --outline_reference_num=1500 \
    --rag_num=60 \
    --api_url='https://api.openai.com/v1/chat/completions' \
    --api_key='YOUR_API_KEY_HERE'
```

**Arguments**:

- `--db_path`: Directory of the SQLite database.
- `--embedding_model`: Embedding model used for retrieval.
- `--saving_path`: Directory to save the generated survey and evaluations.
- `--model`: AI model to use for generation (e.g., `'gpt-4'`).
- `--topic`: Topic for which to generate the survey.
- `--section_num`: Number of sections in the survey outline (default: 7).
- `--subsection_len`: Length of each subsection in words (default: 700).
- `--outline_reference_num`: Number of references to use for outline generation (default: 1500).
- `--rag_num`: Number of references to use for Retrieval-Augmented Generation (RAG) (default: 60).
- `--api_url`: URL for the OpenAI API (default: `'https://api.openai.com/v1/chat/completions'`).
- `--api_key`: Your OpenAI API key.

**Example**:

```bash
python main.py \
    --db_path='./database' \
    --embedding_model='nomic-ai/nomic-embed-text-v1' \
    --saving_path='./output/' \
    --model='gpt-4' \
    --topic='Quantum Computing' \
    --section_num=7 \
    --subsection_len=700 \
    --outline_reference_num=1500 \
    --rag_num=60 \
    --api_url='https://api.openai.com/v1/chat/completions' \
    --api_key='sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
```

Upon successful execution, you should see:

```
Outline generated.
Subsections written and refined.
Survey saved.
评估引用质量: 100%|██████████| X/Y [00:XX<00:00, XXX.XX线程/秒]
评估引用精确性: 100%|██████████| X/Y [00:XX<00:00, XXX.XX线程/秒]
Evaluation completed and saved.
```

The generated survey will be saved in the specified `saving_path` directory as both `.md` and `.json` files, along with an evaluation `.txt` file.

## Project Structure

```
InterActWriter/
│
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── model.py
│   ├── database.py
│   ├── prompt.py
│   └── agents/
│       ├── __init__.py
│       ├── judge.py
│       ├── outline_writer.py
│       └── writer.py
│
├── import_papers_sqlite.py
├── generate_embeddings_and_index.py
├── main.py
├── requirements.txt
├── README.md
├── logo.png
└── arxiv_papers.json  # Your arXiv papers data file
```

- **src/**: Contains all source code modules.
  - **utils.py**: Utility functions, including token counting and text truncation.
  - **model.py**: Handles interactions with the OpenAI API.
  - **database.py**: Manages database connections and queries using SQLite.
  - **prompt.py**: Contains prompt templates for AI interactions.
  - **agents/**: Contains agent modules for outlining, writing, and judging.
    - **outline_writer.py**: Generates survey outlines.
    - **writer.py**: Writes and refines survey subsections.
    - **judge.py**: Evaluates the quality of generated surveys.
- **import_papers_sqlite.py**: Script to import arXiv papers into SQLite.
- **generate_embeddings_and_index.py**: Script to generate embeddings and build FAISS indexes.
- **main.py**: Main script to generate and evaluate academic surveys.
- **requirements.txt**: Lists all Python dependencies.
- **README.md**: This file.
- **logo.png**: Project logo (optional).
- **arxiv_papers.json**: Your input data file containing arXiv papers.

## Configuration

You can configure various parameters through command-line arguments when running the scripts. Below are some common parameters:

### Common Parameters

- `--db_path`: Path to the SQLite database directory.
- `--embedding_model`: Name of the SentenceTransformer model to use.
- `--api_key`: Your OpenAI API key.
- `--api_url`: URL for the OpenAI API endpoint.

### Specific Script Parameters

#### `import_papers_sqlite.py`

- `--input_data`: Path to the input JSON file containing arXiv papers.
- `--batch_size`: Number of papers to insert per batch.

#### `generate_embeddings_and_index.py`

- `--index_path_title`: Path to save the FAISS index for titles.
- `--index_path_abs`: Path to save the FAISS index for abstracts.
- `--id_to_index_path_title`: Path to save the ID-to-index mapping for titles.
- `--index_to_id_path_title`: Path to save the index-to-ID mapping for titles.
- `--id_to_index_path_abs`: Path to save the ID-to-index mapping for abstracts.
- `--index_to_id_path_abs`: Path to save the index-to-ID mapping for abstracts.
- `--batch_size`: Batch size for embedding generation.

#### `main.py`

- `--saving_path`: Directory to save the generated survey and evaluations.
- `--model`: AI model to use for generation (e.g., `'gpt-4'`).
- `--topic`: Topic for which to generate the survey.
- `--section_num`: Number of sections in the survey outline.
- `--subsection_len`: Length of each subsection in words.
- `--outline_reference_num`: Number of references to use for outline generation.
- `--rag_num`: Number of references to use for Retrieval-Augmented Generation (RAG).

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes with clear messages.
4. Push to your forked repository.
5. Open a pull request detailing your changes.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or support, please contact [your.email@example.com](mailto:your.email@example.com).

---

![InterActWriter](https://github.com/yourusername/InterActWriter/raw/main/banner.png)

*Empowering researchers to create comprehensive academic surveys effortlessly.*
