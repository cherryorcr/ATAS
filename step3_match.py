import os
import numpy as np
import pandas as pd

# ================= 配置路径 =================

# 1. 上一步生成数据的文件夹
DATA_DIR = r"D:\predict\data\合同信息\embeddings_output"

# 2. 最终映射结果保存路径 (Excel文件)
OUTPUT_EXCEL = r"D:\predict\data\合同信息\label_mapping_result.xlsx"

# 3. 配置参数
TOP_K = 3          # 每个内部标签匹配最相似的 3 个外部标签
MIN_SCORE = 0.0    # 相似度阈值 (0~1)，低于这个分数的可以忽略，设为0表示保留所有结果

# ================= 核心代码 =================

def load_data():
    """加载向量和标签文件"""
    print(f"📂 正在加载数据: {DATA_DIR}")
    
    try:
        # 加载向量
        int_emb = np.load(os.path.join(DATA_DIR, "internal_embeddings.npy"))
        ext_emb = np.load(os.path.join(DATA_DIR, "external_embeddings.npy"))
        
        # 加载标签文本
        with open(os.path.join(DATA_DIR, "internal_labels_clean.txt"), 'r', encoding='utf-8') as f:
            int_labels = [line.strip() for line in f]
            
        with open(os.path.join(DATA_DIR, "external_labels_clean.txt"), 'r', encoding='utf-8') as f:
            ext_labels = [line.strip() for line in f]
            
        print(f"✅ 数据加载成功！")
        print(f"   内部标签数: {len(int_labels)}")
        print(f"   外部标签数: {len(ext_labels)}")
        return int_emb, ext_emb, int_labels, ext_labels
        
    except FileNotFoundError as e:
        print(f"❌ 错误：找不到文件，请检查路径。详情: {e}")
        exit()

def main():
    print("="*50)
    print("🚀 开始第 3 步：计算相似度矩阵并生成映射表")
    print("="*50)

    # 1. 加载数据
    int_emb, ext_emb, int_labels, ext_labels = load_data()
    
    # 2. 计算相似度矩阵 (矩阵乘法，速度极快)
    # 形状: (内部数量, 外部数量)
    print("\n⚡ 正在计算相似度矩阵...")
    similarity_matrix = np.dot(int_emb, ext_emb.T)
    
    # 3. 寻找 Top-K 匹配
    print(f"🔍 正在为每个内部标签寻找 Top-{TOP_K} 匹配...")
    
    results = []
    
    for i, i_label in enumerate(int_labels):
        # 获取第 i 个内部标签的所有相似度分数
        scores = similarity_matrix[i]
        
        # 对分数排序，取前 Top_K 的索引 (argsort 返回的是从小到大的索引，所以要[::-1]反转)
        top_indices = scores.argsort()[-TOP_K:][::-1]
        
        # 构建一行数据
        row_data = {"内部标签": i_label}
        
        for rank, idx in enumerate(top_indices):
            score = float(scores[idx])
            matched_label = ext_labels[idx]
            
            if score >= MIN_SCORE:
                row_data[f"匹配外部标签_{rank+1}"] = matched_label
                row_data[f"相似度_{rank+1}"] = round(score, 4) # 保留4位小数
            else:
                row_data[f"匹配外部标签_{rank+1}"] = "低于阈值"
                row_data[f"相似度_{rank+1}"] = round(score, 4)
        
        results.append(row_data)

    # 4. 导出到 Excel
    print(f"\n💾 正在写入 Excel: {OUTPUT_EXCEL}")
    df = pd.DataFrame(results)
    
    # 调整列顺序，好看一点
    cols = ["内部标签"]
    for k in range(1, TOP_K + 1):
        cols.extend([f"匹配外部标签_{k}", f"相似度_{k}"])
    df = df[cols]
    
    df.to_excel(OUTPUT_EXCEL, index=False)
    
    print(f"🎉 成功！映射表已生成。\n请打开查看效果: {OUTPUT_EXCEL}")

if __name__ == "__main__":
    main()