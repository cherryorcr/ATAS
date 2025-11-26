import json
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, models
import time
import torch

# ================= ⚙️ 配置 =================

JSON_FILE_PATH = r"D:\predict\0.1\data\2015_tree.json"
EXTERNAL_TXT_PATH = r"D:\predict\0.1\lables"
LOCAL_MODEL_PATH = r"D:\predict\models\bge-large-zh-v1.5"

# 结果文件（同时保存 Excel 和 CSV）
OUTPUT_EXCEL = r"D:\predict\0.1\2015_Project_Final_Labels_GPU.xlsx"
OUTPUT_CSV = r"D:\predict\0.1\2015_Project_Final_Labels_GPU.csv"

# 中间缓存文件（防止崩了白跑）
CACHE_EMB_PATH = r"D:\predict\0.1\2015project_embeddings_cache.npy"

BATCH_SIZE = 64


# ================= 代码 =================

def load_model_on_gpu(model_path):
    print(f"\n⬇️  正在加载模型: {model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  运行设备: {device}")

    word_embedding_model = models.Transformer(model_path, max_seq_length=512)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_cls_token=True)
    return SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)


def extract_projects(file_path):
    print(f"📂 读取 JSON: {file_path}")
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


def load_external_labels(file_path):
    if not os.path.exists(file_path): file_path += ".txt"
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def main():
    print("=" * 50)
    print("🚀 最终防崩溃版启动")
    print("=" * 50)

    # 1. 加载数据
    df = extract_projects(JSON_FILE_PATH)
    project_names = df["项目名称"].tolist()
    print(f"📊 共 {len(project_names)} 条项目")

    # 2. 检查是否有缓存向量
    if os.path.exists(CACHE_EMB_PATH):
        print(f"\n⚡ 发现已计算好的向量缓存: {CACHE_EMB_PATH}")
        print("⏩ 直接加载缓存，跳过模型计算！")
        proj_emb = np.load(CACHE_EMB_PATH)
        if len(proj_emb) != len(project_names):
            print("❌ 缓存数量与数据不一致，将重新计算...")
            need_calc = True
        else:
            need_calc = False
            # 还是需要加载模型来算一下外部标签的向量
            model = load_model_on_gpu(LOCAL_MODEL_PATH)
    else:
        need_calc = True
        model = load_model_on_gpu(LOCAL_MODEL_PATH)

    # 3. 计算项目向量 (如果没缓存)
    if need_calc:
        print(f"\n⚡ 开始计算项目向量...")
        start_t = time.time()
        proj_emb = model.encode(project_names, normalize_embeddings=True, batch_size=BATCH_SIZE, show_progress_bar=True)
        print(f"✅ 计算耗时: {time.time() - start_t:.1f}s")

        # 【关键】立即保存缓存
        print(f"💾 保存向量缓存到: {CACHE_EMB_PATH}")
        np.save(CACHE_EMB_PATH, proj_emb)

    # 4. 计算外部标签向量
    print("\n🏷️  计算外部标签向量...")
    ext_labels = load_external_labels(EXTERNAL_TXT_PATH)
    ext_emb = model.encode(ext_labels, normalize_embeddings=True, batch_size=BATCH_SIZE, show_progress_bar=False)

    # 5. 匹配
    print("\n🔍 正在匹配...")
    sim_matrix = np.dot(proj_emb, ext_emb.T)

    results = []
    top_k = 3
    for i, row in df.iterrows():
        scores = sim_matrix[i]
        top_idx = scores.argsort()[-top_k:][::-1]
        item = row.to_dict()
        for rank, idx in enumerate(top_idx):
            item[f"外部标签_{rank + 1}"] = ext_labels[idx]
            item[f"相似度_{rank + 1}"] = round(float(scores[idx]), 4)
        results.append(item)

    # 6. 保存结果 (双重保险)
    df_res = pd.DataFrame(results)

    # 优先保存 CSV (速度快，不依赖 openpyxl)
    print(f"\n💾 正在保存 CSV: {OUTPUT_CSV}")
    df_res.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')  # utf-8-sig 防止中文乱码

    # 尝试保存 Excel
    try:
        print(f"💾 正在保存 Excel: {OUTPUT_EXCEL}")
        df_res.to_excel(OUTPUT_EXCEL, index=False)
        print("✅ Excel 保存成功")
    except ImportError:
        print("⚠️ 缺少 openpyxl 库，Excel 保存失败，但 CSV 已保存成功！")
    except Exception as e:
        print(f"⚠️ Excel 保存出错: {e} (请查看 CSV 文件)")

    print("\n🎉 全部完成！")


if __name__ == "__main__":
    main()