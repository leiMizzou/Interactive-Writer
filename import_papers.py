import json
import argparse
import sys
import sqlite3
from tqdm import tqdm

def import_papers_sqlite(arxiv_papers, db_path, batch_size=1000):
    conn = sqlite3.connect(f'{db_path}/arxiv_paper_db.sqlite')
    cursor = conn.cursor()
    
    # 创建表（如果尚未存在）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cs_paper_info (
            id TEXT PRIMARY KEY,
            title TEXT,
            abs TEXT,
            date TEXT,
            authors TEXT,
            categories TEXT,
            doi TEXT,
            journal_ref TEXT,
            comments TEXT,
            submitter TEXT
        )
    ''')
    conn.commit()
    
    total = len(arxiv_papers)
    for i in tqdm(range(0, total, batch_size), desc="导入论文", unit="批次"):
        batch = arxiv_papers[i:i + batch_size]
        mapped_batch = [
            (
                paper.get('id'),
                paper.get('title'),
                paper.get('abstract'),
                paper.get('update_date'),
                json.dumps(paper.get('authors')),
                json.dumps(paper.get('categories')),
                paper.get('doi'),
                paper.get('journal-ref'),
                paper.get('comments'),
                paper.get('submitter')
            )
            for paper in batch
        ]
        try:
            cursor.executemany('''
                INSERT OR REPLACE INTO cs_paper_info 
                (id, title, abs, date, authors, categories, doi, journal_ref, comments, submitter) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', mapped_batch)
            conn.commit()
        except sqlite3.Error as e:
            print(f"SQLite 插入错误: {e}")
    
    conn.close()
    print("论文导入成功。")

def read_json_objects(file_path):
    """
    读取包含多个 JSON 对象的文件，可以是 JSON Lines 格式或 JSON 数组。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            # 文件是一个 JSON 数组
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    print("错误：预期一个 JSON 数组。")
                    sys.exit(1)
            except json.JSONDecodeError as e:
                print(f"JSON 解析错误：{e}")
                sys.exit(1)
        else:
            # 假设文件是 JSON Lines 格式，每行一个 JSON 对象
            arxiv_papers = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                try:
                    obj = json.loads(line)
                    arxiv_papers.append(obj)
                except json.JSONDecodeError as e:
                    print(f"第 {line_num} 行 JSON 解析错误：{e}")
                    continue  # 跳过无效的 JSON 对象
            return arxiv_papers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将 arXiv 论文导入 SQLite 数据库。')
    parser.add_argument('--input_data', required=True, type=str, help='输入 arXiv 论文 JSON 文件路径。')
    parser.add_argument('--db_path', default='./database', type=str, help='SQLite 数据库目录路径。')
    parser.add_argument('--batch_size', default=1000, type=int, help='批量插入的大小。')

    args = parser.parse_args()

    arxiv_papers = read_json_objects(args.input_data)
    if not arxiv_papers:
        print("未在输入文件中找到有效的 JSON 对象。")
        sys.exit(1)

    import_papers_sqlite(arxiv_papers, db_path=args.db_path, batch_size=args.batch_size)
