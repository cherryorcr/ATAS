import pandas as pd
import os
from tqdm import tqdm

# ================= ⚙️ 配置路径 =================
PROJECT_CSV = r"D:\predict\data\合同信息\2021_Project_Final_Fixed.csv"
MAPPING_FILE = r"D:\predict\data\合同信息\label_mapping_result.xlsx"
OUTPUT_FLAT_CSV = r"D:\predict\0.1\data\2021_Project_Flattened_Report_FullPath.csv"


# ================= 🛠️ 辅助函数 =================
def get_leaf_name(text):
    """
    【仅用于匹配键】提取标签的最后一段
    用于把 '先进制造-增材制造' 和 '增材制造技术' 统一起来进行匹配
    """
    if pd.isna(text) or str(text).strip() == "":
        return ""
    text = str(text).replace('root > ', '').replace(' > ', '-').replace('>', '-').replace('_', '-').replace('—', '-')
    parts = text.split('-')
    return parts[-1].strip()


def clean_full_path(text):
    """
    【用于展示】保留完整路径，但清洗掉 root 前缀
    """
    if pd.isna(text) or str(text).strip() == "":
        return ""
    # 去掉 JSON 树中的 root 节点，保留后续结构
    text = str(text).replace('root > ', '').strip()
    return text


def main():
    print("=" * 50)
    print("🚀 开始生成全路径反查报表")
    print("=" * 50)

    # -------------------------------------------------------
    # 1. 构建“最强反查字典” (Value 保留完整路径)
    # -------------------------------------------------------
    print("📥 1. 加载映射表 & 构建索引...")
    if os.path.exists(MAPPING_FILE):
        map_df = pd.read_excel(MAPPING_FILE).fillna("")
    else:
        map_df = pd.read_csv(MAPPING_FILE.replace(".xlsx", ".csv"), encoding='utf-8-sig').fillna("")

    best_match_dict = {}

    for _, row in tqdm(map_df.iterrows(), total=len(map_df), desc="构建索引"):
        # 【修改点1】这里不再取 leaf，而是保留完整路径
        internal_full_path = str(row["内部标签"]).strip()

        for i in range(1, 4):
            ext_col = f"匹配外部标签_{i}"
            score_col = f"相似度_{i}"

            if ext_col in row and score_col in row:
                ext_val = row[ext_col]
                score_val = row[score_col]

                # Key 依然用叶子名，为了能和项目的技术名匹配上
                ext_key_leaf = get_leaf_name(ext_val)

                if ext_key_leaf and pd.notna(score_val):
                    try:
                        current_score = float(score_val)
                    except:
                        current_score = 0.0

                    if ext_key_leaf not in best_match_dict:
                        best_match_dict[ext_key_leaf] = {
                            "internal_full": internal_full_path,  # 存全路径
                            "score": current_score
                        }
                    else:
                        # 竞价排名：保留分数更高的那个内部完整路径
                        if current_score > best_match_dict[ext_key_leaf]["score"]:
                            best_match_dict[ext_key_leaf] = {
                                "internal_full": internal_full_path,
                                "score": current_score
                            }

    print(f"✅ 索引构建完成！")

    # -------------------------------------------------------
    # 2. 处理项目数据
    # -------------------------------------------------------
    print("\n📥 2. 加载项目数据...")
    projects_df = pd.read_csv(PROJECT_CSV, encoding='utf-8-sig').fillna("")

    print("⚡ 3. 正在匹配每一行...")
    results = []

    for _, row in tqdm(projects_df.iterrows(), total=len(projects_df), desc="生成报表"):
        p_name = row["项目名称"]

        # 【修改点2】原归属保留完整路径 (去掉 root > 即可)
        orig_internal_full = clean_full_path(row["原内部路径"])

        # 提取 AI 匹配的技术 (Key)
        # 这里的展示列，你可以选择保留全名或者叶子名。
        # 通常“匹配技术”也是带路径的，建议也保留原样或清洗后展示。
        # 这里我们展示清洗后的完整技术名（如果有路径的话），方便阅读
        tech_display_1 = clean_full_path(row.get("外部标签_1", ""))
        tech_display_2 = clean_full_path(row.get("外部标签_2", ""))
        tech_display_3 = clean_full_path(row.get("外部标签_3", ""))

        # 提取用于查找的 Key (叶子名)
        key_1 = get_leaf_name(row.get("外部标签_1", ""))
        key_2 = get_leaf_name(row.get("外部标签_2", ""))
        key_3 = get_leaf_name(row.get("外部标签_3", ""))

        # 反查
        rev_1 = best_match_dict[key_1]["internal_full"] if key_1 in best_match_dict else ""
        rev_2 = best_match_dict[key_2]["internal_full"] if key_2 in best_match_dict else ""
        rev_3 = best_match_dict[key_3]["internal_full"] if key_3 in best_match_dict else ""

        results.append({
            "项目名称": p_name,
            "原内部归属(完整)": orig_internal_full,
            "AI匹配技术_1": tech_display_1,
            "AI匹配技术_2": tech_display_2,
            "AI匹配技术_3": tech_display_3,
            "反查归属_1(完整)": rev_1,
            "反查归属_2(完整)": rev_2,
            "反查归属_3(完整)": rev_3
        })

    # -------------------------------------------------------
    # 3. 保存
    # -------------------------------------------------------
    print(f"\n💾 4. 正在保存到: {OUTPUT_FLAT_CSV}")
    final_df = pd.DataFrame(results)

    cols_order = [
        "项目名称", "原内部归属(完整)",
        "AI匹配技术_1", "AI匹配技术_2", "AI匹配技术_3",
        "反查归属_1(完整)", "反查归属_2(完整)", "反查归属_3(完整)"
    ]
    final_df = final_df[cols_order]

    final_df.to_csv(OUTPUT_FLAT_CSV, index=False, encoding='utf-8-sig')
    print("🎉 全部完成！")


if __name__ == "__main__":
    main()