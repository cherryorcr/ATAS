import pandas as pd
import os

# --- 配置路径 ---
source_file_path = r"D:\predict\0.1\data\2025.csv"
target_file_path = r"D:\predict\0.1\data\2025_Project_Flattened_Report_FullPath.csv"


def clean_text(text):
    """清理函数：去除多余的引号和首尾空格"""
    if not text:
        return ""
    # 替换掉 CSV 中常见的双引号 wrapper
    clean = text.replace('"""', '').replace('"', '').strip()
    return clean


try:
    # ---------------------------------------------------------
    # 1. 读取源文件并构建字典 (Hash Map)
    # ---------------------------------------------------------
    print(f"正在读取并解析源文件: {source_file_path}")

    # 数据字典结构: { "项目名称": ("金额", "开始时间") }
    project_data_map = {}
    dirty_lines_count = 0

    with open(source_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('###')

            # 简单的脏数据过滤：如果切分后少于4部分，说明该行格式严重错误
            if len(parts) < 4:
                dirty_lines_count += 1
                continue

            # 提取数据
            # 第1列(索引0): 项目名称 (用于匹配)
            # 第2列(索引1): 金额
            # 第4列(索引3): 开始时间
            p_name = clean_text(parts[0])
            p_amount = clean_text(parts[1])
            p_time = clean_text(parts[3])

            # 只有项目名称不为空才存入
            if p_name:
                project_data_map[p_name] = (p_amount, p_time)
            else:
                dirty_lines_count += 1

    print(f"源文件解析完成。有效项目: {len(project_data_map)} 个，忽略脏行/空名: {dirty_lines_count} 行。")

    # ---------------------------------------------------------
    # 2. 读取目标文件
    # ---------------------------------------------------------
    print(f"正在读取目标文件: {target_file_path}")
    try:
        df_target = pd.read_csv(target_file_path, encoding='gbk')
    except UnicodeDecodeError:
        df_target = pd.read_csv(target_file_path, encoding='utf-8')

    # ---------------------------------------------------------
    # 3. 准备目标文件的列 (扩充到至少10列)
    # ---------------------------------------------------------
    # Excel I列是第9列(Index 8)，J列是第10列(Index 9)
    while df_target.shape[1] < 10:
        new_col_idx = df_target.shape[1]
        # 如果是填充I列，列名暂定 Amount_Extracted，J列暂定 Time_Extracted
        if new_col_idx == 8:
            col_name = "Amount_Extracted"
        elif new_col_idx == 9:
            col_name = "Time_Extracted"
        else:
            col_name = f"Unnamed_{new_col_idx}"
        df_target[col_name] = ""

    # 获取 I列 和 J列 的列名
    col_name_I = df_target.columns[8]
    col_name_J = df_target.columns[9]

    # ---------------------------------------------------------
    # 4. 遍历匹配并更新
    # ---------------------------------------------------------
    print("正在进行项目名称匹配和数据填充...")

    matched_count = 0

    # 获取目标文件第一列的列名（假设第一列是项目名称）
    target_key_col = df_target.columns[0]

    # 为了提高效率，我们将需要更新的列转换为列表或使用 apply，但循环对于几十万行也很快且逻辑清晰
    # 这里使用逐行查找更新

    for index, row in df_target.iterrows():
        # 获取目标文件的项目名称 (清理一下空格以提高匹配率)
        target_name = str(row[target_key_col]).strip()

        if target_name in project_data_map:
            amount, start_time = project_data_map[target_name]

            # 更新 I 列 (金额)
            df_target.at[index, col_name_I] = amount
            # 更新 J 列 (时间)
            df_target.at[index, col_name_J] = start_time

            matched_count += 1

    # ---------------------------------------------------------
    # 5. 保存结果
    # ---------------------------------------------------------
    print(f"匹配完成！共成功匹配并更新了 {matched_count} 行数据。")
    print("正在保存文件...")

    df_target.to_csv(target_file_path, index=False, encoding='utf-8-sig')

    print(f"处理完毕。结果已保存至: {target_file_path}")

except FileNotFoundError:
    print("错误：找不到文件，请检查路径。")
except Exception as e:
    import traceback

    print(f"发生未知错误: {e}")
    print(traceback.format_exc())