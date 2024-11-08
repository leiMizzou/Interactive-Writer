# generate_embeddings_and_index.py

import json
import sqlite3
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
from tqdm import tqdm
import argparse
import os

def generate_embeddings_and_index(db_path, embedding_model, index_path_title, index_path_abs,
                                  id_to_index_path_title, index_to_id_path_title,
                                  id_to_index_path_abs, index_to_id_path_abs, batch_size=64):
    conn = sqlite3.connect(f'{db_path}/arxiv_paper_db.sqlite')
    cursor = conn.cursor()

    model = SentenceTransformer(embedding_model, trust_remote_code=True)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # 从数据库中读取数据
    cursor.execute("SELECT id, title, abs FROM cs_paper_info")
    rows = cursor.fetchall()
    ids = []
    titles = []
    abstracts = []
    for row in rows:
        ids.append(row[0])
        titles.append(row[1] if row[1] else '')
        abstracts.append(row[2] if row[2] else '')

    # 生成标题嵌入
    print("Generating title embeddings...")
    title_embeddings = []
    for i in tqdm(range(0, len(titles), batch_size)):
        batch_titles = titles[i:i + batch_size]
        embeddings = model.encode(batch_titles, show_progress_bar=False)
        title_embeddings.append(embeddings)
    title_embeddings = np.vstack(title_embeddings).astype('float32')

    # 生成摘要嵌入
    print("Generating abstract embeddings...")
    abs_embeddings = []
    for i in tqdm(range(0, len(abstracts), batch_size)):
        batch_abstracts = abstracts[i:i + batch_size]
        embeddings = model.encode(batch_abstracts, show_progress_bar=False)
        abs_embeddings.append(embeddings)
    abs_embeddings = np.vstack(abs_embeddings).astype('float32')

    # 构建 FAISS 索引
    dimension = title_embeddings.shape[1]
    print("Building FAISS index for titles...")
    index_title = faiss.IndexFlatL2(dimension)
    index_title.add(title_embeddings)

    print("Building FAISS index for abstracts...")
    index_abs = faiss.IndexFlatL2(dimension)
    index_abs.add(abs_embeddings)

    # 保存 FAISS 索引
    faiss.write_index(index_title, index_path_title)
    faiss.write_index(index_abs, index_path_abs)
    print("FAISS indexes saved.")

    # 创建 id2index 和 index2id 映射
    id_to_index_title = {id_: idx for idx, id_ in enumerate(ids)}
    index_to_id_title = {idx: id_ for idx, id_ in enumerate(ids)}
    id_to_index_abs = {id_: idx for idx, id_ in enumerate(ids)}
    index_to_id_abs = {idx: id_ for idx, id_ in enumerate(ids)}

    # 保存映射
    with open(id_to_index_path_title, 'w') as f:
        json.dump(id_to_index_title, f)
    with open(index_to_id_path_title, 'w') as f:
        json.dump(index_to_id_title, f)
    with open(id_to_index_path_abs, 'w') as f:
        json.dump(id_to_index_abs, f)
    with open(index_to_id_path_abs, 'w') as f:
        json.dump(index_to_id_abs, f)
    print("ID to Index mappings saved.")

    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate embeddings and build FAISS indexes.')
    parser.add_argument('--db_path', default='./database', type=str, help='Path to the database directory.')
    parser.add_argument('--embedding_model', default='nomic-ai/nomic-embed-text-v1', type=str, help='SentenceTransformer model name.')
    parser.add_argument('--index_path_title', default='./database/faiss_paper_title_embeddings.bin', type=str, help='Path to save title FAISS index.')
    parser.add_argument('--index_path_abs', default='./database/faiss_paper_abs_embeddings.bin', type=str, help='Path to save abstract FAISS index.')
    parser.add_argument('--id_to_index_path_title', default='./database/arxivid_to_index_title.json', type=str, help='Path to save ID to Index mapping for titles.')
    parser.add_argument('--index_to_id_path_title', default='./database/index_to_arxivid_title.json', type=str, help='Path to save Index to ID mapping for titles.')
    parser.add_argument('--id_to_index_path_abs', default='./database/arxivid_to_index_abs.json', type=str, help='Path to save ID to Index mapping for abstracts.')
    parser.add_argument('--index_to_id_path_abs', default='./database/index_to_arxivid_abs.json', type=str, help='Path to save Index to ID mapping for abstracts.')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for embedding generation.')

    args = parser.parse_args()

    # 创建数据库目录（如果不存在）
    if not os.path.exists(args.db_path):
        os.makedirs(args.db_path)

    generate_embeddings_and_index(
        db_path=args.db_path,
        embedding_model=args.embedding_model,
        index_path_title=args.index_path_title,
        index_path_abs=args.index_path_abs,
        id_to_index_path_title=args.id_to_index_path_title,
        index_to_id_path_title=args.index_to_id_path_title,
        id_to_index_path_abs=args.id_to_index_path_abs,
        index_to_id_path_abs=args.index_to_id_path_abs,
        batch_size=args.batch_size
    )
