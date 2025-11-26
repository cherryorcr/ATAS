import pandas as pd
import os
from itertools import combinations
from collections import Counter, defaultdict
from tqdm import tqdm
import re

# ================= ⚙️ 配置 =================
# 1. 项目全路径报表 (来源)
PROJECT_CSV = r"D:\predict\0.1\data\2022_Project_Flattened_Report_FullPath.csv"
# 2. 映射表 (来源)
MAPPING_FILE = r"D:\predict\0.1\label_mapping_result.xlsx"
# 3. 输出结果
OUTPUT_CSV = r"D:\predict\0.1\data\2022_External_Tech_Weighted_Graph.csv"

# 权重系数
WEIGHT_DIRECT = 1.0  # 直接共现权重
WEIGHT_INDIRECT_FACTOR = 0.3  # 间接共现系数 (内部业务出现次数 * 0.3)


# ================= 🛠️ 辅助函数 =================
def get_leaf_name(text):
    """提取叶子名"""
    if pd.isna(text) or str(text).strip() == "": return ""
    text = str(text).replace('root > ', '').replace(' > ', '-').replace('>', '-').replace('--', '-').replace('_', '-')
    parts = text.split('-')
    return parts[-1].strip()


def clean_internal_key(text):
    """清洗内部标签用于匹配 (统一格式)"""
    if pd.isna(text): return ""
    # 统一转为 A-B-C 格式
    clean = str(text).replace('root > ', '').replace(' > ', '-').replace('--', '-')
    return clean.strip()


def get_full_path_tuple(text):
    """解析外部标签的层级 (L1, L2, L3)"""
    if pd.isna(text) or str(text).strip() == "": return ("未知", "未知", "未知")
    text = str(text).replace(' > ', '-').replace('>', '-').replace('--', '-')
    parts = text.split('-')
    # 补齐
    while len(parts) < 3: parts.insert(0, "通用领域")
    return (parts[-3], parts[-2], parts[-1])  # L1, L2, L3


def main():
    print("=" * 50)
    print("🚀 开始构建混合加权外部技术图谱")
    print("=" * 50)

    # ----------------------------------------------------
    # 1. 统计内部标签在项目中出现的次数 (用于计算间接权重)
    # ----------------------------------------------------
    print("📥 正在统计内部业务活跃度...")
    project_df = pd.read_csv(PROJECT_CSV, encoding='utf-8-sig').fillna("")

    # 计数器: { "先进制造-工艺-其他": 500次 }
    internal_usage_counts = Counter()

    # 同时也统计直接共现
    direct_edge_weights = Counter()

    # 存储每个技术的层级信息 (用于后面生成节点属性)
    tech_hierarchy_map = {}

    print("⚡ 计算直接共现 & 内部统计...")
    for _, row in tqdm(project_df.iterrows(), total=len(project_df)):
        # A. 统计内部标签频率
        raw_internal = row["原内部归属(完整)"]
        clean_int = clean_internal_key(raw_internal)
        if clean_int:
            internal_usage_counts[clean_int] += 1

        # B. 统计直接共现 (AI匹配技术)
        # 提取 3 列技术
        techs = []
        for i in range(1, 4):
            full_tag = row.get(f"AI匹配技术_{i}")
            if full_tag:
                leaf = get_leaf_name(full_tag)
                techs.append(leaf)
                # 记录层级结构
                if leaf not in tech_hierarchy_map:
                    tech_hierarchy_map[leaf] = get_full_path_tuple(full_tag)

        # 两两组合，加权重
        unique_techs = sorted(list(set(techs)))
        if len(unique_techs) > 1:
            for pair in combinations(unique_techs, 2):
                direct_edge_weights[pair] += WEIGHT_DIRECT

    print(f"✅ 内部业务统计完成，共 {len(internal_usage_counts)} 个活跃部门")

    # ----------------------------------------------------
    # 2. 计算间接共现 (基于映射表)
    # ----------------------------------------------------
    print("📥 正在计算间接结构权重...")
    if os.path.exists(MAPPING_FILE):
        map_df = pd.read_excel(MAPPING_FILE).fillna("")
    else:
        map_df = pd.read_csv(MAPPING_FILE.replace(".xlsx", ".csv"), encoding='utf-8-sig').fillna("")

    indirect_edge_weights = Counter()

    for _, row in map_df.iterrows():
        # 获取该行的内部标签
        map_internal = clean_internal_key(row["内部标签"])

        # 获取该内部标签在项目中出现的次数 (活跃度)
        # 注意：映射表里的名字可能和项目表里有一点点差异，这里尽量匹配
        # 如果项目表里是 "A-B"，映射表是 "A-B-C"，可能匹配不上，暂且假设清洗后一致
        occur_count = internal_usage_counts.get(map_internal, 0)

        if occur_count > 0:
            # 提取该业务对应的 3 个标准外部技术
            std_techs = []
            for i in range(1, 4):
                full_tag = row.get(f"匹配外部标签_{i}")
                if full_tag:
                    leaf = get_leaf_name(full_tag)
                    std_techs.append(leaf)
                    if leaf not in tech_hierarchy_map:
                        tech_hierarchy_map[leaf] = get_full_path_tuple(full_tag)

            # 计算间接权重： 活跃度 * 系数
            weight_add = occur_count * WEIGHT_INDIRECT_FACTOR

            unique_std = sorted(list(set(std_techs)))
            if len(unique_std) > 1:
                for pair in combinations(unique_std, 2):
                    indirect_edge_weights[pair] += weight_add

    # ----------------------------------------------------
    # 3. 合并权重并保存
    # ----------------------------------------------------
    print("🔄 正在合并权重...")
    final_edges = {}  # Key: (A, B), Value: weight

    # 合并所有涉及的 pair
    all_pairs = set(direct_edge_weights.keys()) | set(indirect_edge_weights.keys())

    edge_list = []
    for pair in all_pairs:
        w_d = direct_edge_weights.get(pair, 0)
        w_i = indirect_edge_weights.get(pair, 0)
        total_w = w_d + w_i

        # 获取层级信息用于CSV
        l1_a, l2_a, _ = tech_hierarchy_map.get(pair[0], ("未知", "未知", "未知"))
        l1_b, l2_b, _ = tech_hierarchy_map.get(pair[1], ("未知", "未知", "未知"))

        edge_list.append({
            "Source": pair[0],
            "Target": pair[1],
            "Weight": round(total_w, 2),
            "Direct_Score": w_d,
            "Indirect_Score": round(w_i, 2),
            "Source_L1": l1_a, "Source_L2": l2_a,
            "Target_L1": l1_b, "Target_L2": l2_b
        })

    print(f"💾 正在保存 {len(edge_list)} 条边到 CSV...")
    df_out = pd.DataFrame(edge_list)

    # 按权重降序排列
    df_out = df_out.sort_values(by="Weight", ascending=False)

    df_out.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"🎉 完成！文件已保存: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()