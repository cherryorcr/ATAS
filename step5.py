import json
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, models
import time
import torch
import re

# ================= ⚙️ 配置路径 =================

JSON_FILE_PATH = r"D:\predict\0.1\data\2021_tree.json"
EXTERNAL_TXT_PATH = r"D:\predict\0.1\lables"  # 代码会自动处理后缀问题
LOCAL_MODEL_PATH = r"D:\predict\models\bge-large-zh-v1.5"

# 缓存的向量文件 (必须存在)
CACHE_EMB_PATH = r"D:\predict\0.1\2021project_embeddings_cache.npy"

# 最终修复结果
OUTPUT_CSV_FIXED = r"D:\predict\data\合同信息\2021_Project_Final_Fixed.csv"


# ================= 代码 =================

def load_model_for_external(model_path):
    """只用来算外部标签，很快"""
    print(f"⬇️  加载模型(仅计算外部标签): {model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    word_embedding_model = models.Transformer(model_path, max_seq_length=512)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_cls_token=True)
    return SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)


def extract_projects(file_path):
    print(f"📂 再次读取 JSON (确保顺序一致): {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    projects = []

    def recurse(node, path):
        name = node.get("name", "Root")
        curr_path = f"{path} > {name}" if path else name
        if "projects" in node:
            for p in node["projects"]:
                if p: projects.append({"项目名称": p, "原内部路径": curr_path})
        if "children" in node:
            for c in node["children"]:
                recurse(c, curr_path)

    if isinstance(data, dict):
        recurse(data, "")
    elif isinstance(data, list):
        for item in data: recurse(item, "")
    return pd.DataFrame(projects)


def clean_text(text):
    """清洗掉可能导致 CSV/Excel 错乱的字符"""
    if not isinstance(text, str): return text
    # 去除换行符、制表符
    text = text.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
    # 去除 Excel 非法控制字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    return text.strip()


def get_real_file_path(base_path):
    """【修复】智能查找文件，不管是 lables 还是 lables.txt"""
    if os.path.exists(base_path):
        return base_path
    elif os.path.exists(base_path + ".txt"):
        return base_path + ".txt"
    else:
        raise FileNotFoundError(f"❌ 找不到外部标签文件: {base_path} 或 {base_path}.txt")


def main():
    print("=" * 50)
    print("🚀 启动修复脚本 (利用缓存秒级完成)")
    print("=" * 50)

    # 1. 读取项目列表
    df_projects = extract_projects(JSON_FILE_PATH)
    print(f"📊 项目数量: {len(df_projects)}")

    # 2. 读取缓存向量
    if not os.path.exists(CACHE_EMB_PATH):
        print(f"❌ 严重错误：找不到缓存文件 {CACHE_EMB_PATH}")
        print("   请确认上一步是否生成了 .npy 文件。")
        return

    print(f"⚡ 读取项目向量缓存: {CACHE_EMB_PATH}")
    proj_emb = np.load(CACHE_EMB_PATH)

    if len(proj_emb) != len(df_projects):
        print(f"❌ 错误：项目数量({len(df_projects)}) 与 向量数量({len(proj_emb)}) 不一致！")
        print("   这说明 JSON 文件可能被改过，或者缓存是旧的。请重新运行完整流程。")
        return

    # 3. 计算外部标签向量
    real_label_path = get_real_file_path(EXTERNAL_TXT_PATH)
    print(f"🏷️  加载外部标签文件: {real_label_path}")

    with open(real_label_path, 'r', encoding='utf-8') as f:
        ext_labels = [line.strip() for line in f if line.strip()]

    # 加载模型计算外部标签
    model = load_model_for_external(LOCAL_MODEL_PATH)
    ext_emb = model.encode(ext_labels, normalize_embeddings=True, show_progress_bar=False)

    # 4. 匹配
    print("🔍 正在执行匹配...")
    sim_matrix = np.dot(proj_emb, ext_emb.T)

    # 5. 组装结果
    print("📦 正在组装数据表...")
    results = []
    top_k = 3

    for i in range(len(df_projects)):
        scores = sim_matrix[i]
        top_idx = scores.argsort()[-top_k:][::-1]

        # 获取原始信息
        row_data = df_projects.iloc[i].to_dict()

        # 清洗原始项目名和路径 (防止里面的换行符破坏 CSV)
        row_data["项目名称"] = clean_text(row_data["项目名称"])
        row_data["原内部路径"] = clean_text(row_data["原内部路径"])

        # 填入匹配结果
        for rank, idx in enumerate(top_idx):
            row_data[f"外部标签_{rank + 1}"] = ext_labels[idx]
            row_data[f"相似度_{rank + 1}"] = round(float(scores[idx]), 4)

        results.append(row_data)

    # 6. 保存
    df_final = pd.DataFrame(results)

    print(f"\n💾 正在保存修复后的 CSV: {OUTPUT_CSV_FIXED}")
    # quoting=1 (QUOTE_ALL) 强制加引号，完美解决 CSV 错行问题
    df_final.to_csv(OUTPUT_CSV_FIXED, index=False, encoding='utf-8-sig', quoting=1)

    print("✅ 修复完成！请查看新生成的 CSV 文件。")


if __name__ == "__main__":
    main()