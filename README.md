# InterActWriter

InterActWriter is a comprehensive tool designed to streamline academic survey generation, particularly in AI research fields. The project includes capabilities for managing and embedding academic papers, creating outlines and subsections, and efficiently interacting with large language models for text generation.

## Features

- **Paper Database Management**: Efficiently imports academic papers into an SQLite database.
- **Embedding and Indexing**: Generates embeddings for paper titles and abstracts and builds FAISS indices for similarity search.
- **Outline and Section Generation**: Provides templates for generating comprehensive outlines and subsections tailored to AI research topics.
- **Token Counting and Cost Estimation**: Estimates token usage and API costs, helping to budget for usage with various LLMs.

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- Libraries: `sentence-transformers`, `faiss`, `tqdm`, `sqlite3`, `torch`, `transformers`

## Usage

### 1. Import Papers into Database

Run `import_papers.py` to import a JSON file of academic papers into the SQLite database. This file should include metadata like titles, abstracts, and authors.

```bash
python import_papers.py --arxiv_papers <path_to_json> --db_path <path_to_db>
```

### 2. Generate Embeddings and Build Indices

Use `generate_embeddings_and_index.py` to generate embeddings for titles and abstracts and create FAISS indices for similarity search.

```bash
python generate_embeddings_and_index.py --db_path <path_to_db> --embedding_model <model_name>
```

### 3. Outline Generation and Text Structuring

Use `temp.ipynb` to generate outlines, subsections, and refine content based on pre-defined templates. This notebook includes prompts for creating structured academic content and connecting with language models for detailed content generation.

## Examples

### Importing Papers

Example command to import data:
```bash
python import_papers.py --arxiv_papers data/papers.json --db_path data/
```

### Generating Embeddings

To build a FAISS index:
```bash
python generate_embeddings_and_index.py --db_path data/ --embedding_model sentence-transformers/all-MiniLM-L6-v2
```

---

This README provides a structured overview and usage instructions based on the provided code. If you'd like any additional sections or examples, let me know!

## Acknowledgments

InterActWriter was inspired by and built upon the foundations laid by the [AutoSurvey](https://github.com/AutoSurveys/AutoSurvey) project. We would like to express our sincere gratitude to the authors for their pioneering work in automating survey generation using large language models.

```

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
