import pandas as pd
import os
from itertools import combinations
from collections import Counter
from tqdm import tqdm
import re

# ================= ⚙️ 配置路径 =================
# 输入：必须是上一步生成的【全路径】报表
INPUT_CSV = r"D:\predict\0.1\data\2021_Project_Flattened_Report_FullPath.csv"
# 输出：共现统计结果
OUTPUT_CSV = r"D:\predict\0.1\data\2021_Internal_Cooccurrence_Stats.csv"


# ================= 🛠️ 辅助函数 =================
def get_leaf_name(text):
    """从全路径中提取最后一段业务名，用于生成第一列的组合名称"""
    if pd.isna(text) or str(text).strip() == "":
        return ""
    # 兼容各种分隔符：root >, >, --, -
    text = str(text).replace('root > ', '').replace(' > ', '-').replace('>', '-').replace('--', '-').replace('_', '-')
    parts = text.split('-')
    return parts[-1].strip()


def main():
    print("=" * 50)
    print("🚀 开始统计共现频率 (组合名简化，源数据完整)")
    print("=" * 50)

    # 1. 加载数据
    print("📥 正在加载报表数据...")
    if not os.path.exists(INPUT_CSV):
        print(f"❌ 错误：找不到文件 {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig').fillna("")
    print(f"✅ 加载完成: {len(df)} 行")

    # 2. 准备统计器
    # Key 是元组: (完整路径A, 完整路径B)
    pair_counter = Counter()

    target_cols = [
        "原内部归属(完整)",
        "反查归属_1(完整)",
        "反查归属_2(完整)",
        "反查归属_3(完整)"
    ]

    valid_rows_count = 0

    # 3. 遍历统计
    print("⚡ 正在计算共现矩阵...")
    for _, row in tqdm(df.iterrows(), total=len(df)):

        # 过滤：原内部归属必须存在
        original_path = str(row[target_cols[0]]).strip()
        if not original_path:
            continue

        valid_rows_count += 1

        # 收集该行所有不为空的归属标签（全路径）
        labels_in_row = set()
        for col in target_cols:
            val = str(row.get(col, "")).strip()
            if val:
                labels_in_row.add(val)

        # 只有1个或0个标签无法组队
        if len(labels_in_row) < 2:
            continue

        # 生成两两组合 (排序确保唯一性)
        sorted_labels = sorted(list(labels_in_row))
        for pair in combinations(sorted_labels, 2):
            pair_counter[pair] += 1

    # 4. 格式化输出
    print(f"\n📊 统计完成，正在生成 CSV...")

    result_data = []

    # most_common() 默认按次数降序排列
    for (path_a, path_b), count in pair_counter.most_common():
        # 提取叶子名用于第一列展示
        leaf_a = get_leaf_name(path_a)
        leaf_b = get_leaf_name(path_b)

        # 拼接组合名
        combo_name = f"{leaf_a} & {leaf_b}"

        result_data.append({
            "归属组合(简化)": combo_name,
            "同时出现次数": count,
            "标签_A(完整路径)": path_a,
            "标签_B(完整路径)": path_b
        })

    result_df = pd.DataFrame(result_data)

    # 保存
    print(f"💾 正在保存结果到: {OUTPUT_CSV}")
    result_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print("🎉 全部完成！")
    if not result_df.empty:
        print("\n🏆 预览前 3 条数据:")
        print(result_df.head(3).to_string())


if __name__ == "__main__":
    main()