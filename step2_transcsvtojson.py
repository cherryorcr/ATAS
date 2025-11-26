import json
import os
import time


def process_large_csv(input_path, output_path):
    print(f"开始处理文件: {input_path}")

    # 初始化根节点
    root = {
        "name": "root",
        "children": []
    }

    # 统计计数器
    line_count = 0
    start_time = time.time()

    # 打开文件
    # 注意：如果报错 UnicodeDecodeError，请将 encoding='utf-8' 改为 'gbk' 或 'gb18030'
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 1. 解析行数据
                # 原始数据格式："值"###"值"###...
                parts = line.split('"###"')

                # 简单的完整性检查，防止空行或坏数据导致索引越界
                # 根据之前的样例，分类在索引8，所以至少要有9列
                if len(parts) < 9:
                    continue

                # 清理数据：处理首尾的引号
                # 第1列：项目名称 (parts[0] 左边可能有引号)
                project_name = parts[0].lstrip('"')

                # 第9列：分类路径 (parts[8])，例如 "先进制造--工艺--其他"
                # 注意：parts[8] 可能包含后续的 "###..." 残余，因为我们split的时候可能没切干净末尾
                # 但根据 split('"###"') 的逻辑，parts[8] 应该是纯净的，或者右边带引号
                category_raw = parts[8]
                category_path = category_raw.split('"###')[0].strip('"')

                # 如果分类为空，归类到"未分类"（可选）
                if not category_path or category_path == r"\N":
                    category_path = "未分类"

                categories = category_path.split('--')

                # 2. 构建树结构
                current_level_children = root["children"]

                for index, category_name in enumerate(categories):
                    # 在当前层级查找节点
                    found_node = None
                    for child in current_level_children:
                        if child["name"] == category_name:
                            found_node = child
                            break

                    # 如果没找到，创建新节点
                    if not found_node:
                        new_node = {
                            "name": category_name,
                            "children": []
                        }
                        current_level_children.append(new_node)
                        found_node = new_node

                    # 如果是叶子分类，添加项目
                    if index == len(categories) - 1:
                        if "projects" not in found_node:
                            found_node["projects"] = []
                        found_node["projects"].append(project_name)

                    # 下沉到下一级
                    current_level_children = found_node["children"]

                # 进度条
                line_count += 1
                if line_count % 10000 == 0:
                    print(f"已处理 {line_count} 行... (耗时: {time.time() - start_time:.2f}s)")

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_path}")
        return
    except UnicodeDecodeError:
        print("错误：文件编码读取失败。请尝试将代码中的 encoding='utf-8' 修改为 'gbk' 或 'gb18030'。")
        return
    except Exception as e:
        print(f"发生未知错误: {e}")
        return

    print(f"处理完成，共 {line_count} 行。正在写入 JSON 文件...")

    # 写入结果
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(root, f_out, ensure_ascii=False, indent=2)

    print(f"文件已保存至: {output_path}")
    print(f"总耗时: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    # 配置输入输出路径
    # 使用 r"" 表示原始字符串，防止 Windows 路径中的反斜杠被转义
    input_csv = r"D:\predict\0.1\data\2014.csv"
    output_json = r"D:\predict\0.1\data\2014_tree.json"

    process_large_csv(input_csv, output_json)