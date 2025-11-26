import json
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, models  # <--- 注意这里引入了 models
import time

# ================= 配置路径 =================

# 1. 输入：你的 JSON 技术树文件
JSON_FILE_PATH = r"D:\predict\0.1\data\2024_tree.json"

# 2. 输入：第2步生成的外部标签数据
EMBEDDING_DIR = r"D:\predict\0.1\embeddings_output"

# 3. 模型路径
LOCAL_MODEL_PATH = r"D:\predict\models\bge-large-zh-v1.5"

# 4. 输出：最终结果 Excel
OUTPUT_EXCEL = r"D:\predict\0.1\2024_Project_Final_Labels.xlsx"

# 5. 参数
TOP_K = 3  # 每个项目匹配前3个外部标签

# ================= 核心代码 =================

def load_external_data():
    """加载第2步生成的外部向量库"""
    print(f"📂 正在加载外部标签库: {EMBEDDING_DIR}")
    try:
        ext_emb = np.load(os.path.join(EMBEDDING_DIR, "external_embeddings.npy"))
        with open(os.path.join(EMBEDDING_DIR, "external_labels_clean.txt"), 'r', encoding='utf-8') as f:
            ext_labels = [line.strip() for line in f]
        return ext_emb, ext_labels
    except Exception as e:
        print(f"❌ 加载失败，请检查第2步是否成功运行。错误: {e}")
        exit()

def extract_projects_from_json(file_path):
    """递归解析JSON树，提取所有项目及其路径"""
    print(f"📂 正在读取 JSON: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    project_list = [] 

    def recurse(node, path_str):
        current_name = node.get("name", "Root")
        new_path = f"{path_str} > {current_name}" if path_str else current_name
        
        if "projects" in node and isinstance(node["projects"], list):
            for proj in node["projects"]:
                if proj and isinstance(proj, str):
                    project_list.append({
                        "项目名称": proj,
                        "原内部路径": new_path
                    })
        
        if "children" in node and isinstance(node["children"], list):
            for child in node["children"]:
                recurse(child, new_path)

    if isinstance(data, dict):
        recurse(data, "")
    elif isinstance(data, list):
        for item in data:
            recurse(item, "")
            
    print(f"✅ JSON 解析完成，共提取到 {len(project_list)} 个项目")
    return project_list

def load_model_manually(model_path):
    """
    【核心修复】手动组装模型，解决缺失 modules.json 导致的 Pooling 错误
    """
    print(f"\n⬇️  正在手动组装 BGE 模型: {model_path}")
    try:
        # 1. 加载基础 Transformer 模型 (只读取 config.json 和 pytorch_model.bin)
        word_embedding_model = models.Transformer(model_path, max_seq_length=512)
        
        # 2. 定义 Pooling 层
        # BGE 模型通常使用 CLS 标记作为句向量 (pooling_mode_cls_token=True)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_cls_token=True,  # BGE 推荐使用 CLS
            pooling_mode_mean_tokens=False,
            pooling_mode_max_tokens=False
        )
        
        # 3. 组合成 SentenceTransformer
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        print("✅ 模型组装加载成功！")
        return model
    except Exception as e:
        print(f"❌ 模型加载依然失败: {e}")
        print("请确保文件夹里至少有 config.json 和 pytorch_model.bin (或 model.safetensors)")
        exit()

def main():
    print("="*50)
    print("🚀 开始第 4 步：项目级精准映射 (修复版)")
    print("="*50)

    # 1. 加载外部库
    ext_emb, ext_labels = load_external_data()

    # 2. 提取项目
    all_projects = extract_projects_from_json(JSON_FILE_PATH)
    if not all_projects:
        print("❌ JSON中未找到任何项目，请检查文件内容。")
        return
    
    df_projects = pd.DataFrame(all_projects)
    project_names = df_projects["项目名称"].tolist()

    # 3. 【修改】调用手动加载函数
    model = load_model_manually(LOCAL_MODEL_PATH)
    
    print(f"⚡ 正在计算 {len(project_names)} 个项目的向量...")
    start_time = time.time()
    project_embeddings = model.encode(project_names, normalize_embeddings=True, show_progress_bar=True)
    print(f"✅ 计算完成，耗时: {time.time() - start_time:.2f} 秒")

    # 4. 核心匹配
    print("\n🔍 正在进行语义匹配...")
    similarity_matrix = np.dot(project_embeddings, ext_emb.T)

    # 5. 整理结果
    final_results = []
    
    for i, row in df_projects.iterrows():
        proj_name = row["项目名称"]
        path_info = row["原内部路径"]
        
        scores = similarity_matrix[i]
        top_indices = scores.argsort()[-TOP_K:][::-1]
        
        res_item = {
            "项目名称": proj_name,
            "原内部路径": path_info
        }
        
        for rank, idx in enumerate(top_indices):
            res_item[f"外部标签_{rank+1}"] = ext_labels[idx]
            res_item[f"相似度_{rank+1}"] = round(float(scores[idx]), 4)
            
        final_results.append(res_item)

    # 6. 保存 Excel
    print(f"\n💾 正在保存最终结果到: {OUTPUT_EXCEL}")
    df_final = pd.DataFrame(final_results)
    df_final.to_excel(OUTPUT_EXCEL, index=False)
    
    print(f"🎉 全部完成！请查看结果文件：{OUTPUT_EXCEL}")

if __name__ == "__main__":
    main()