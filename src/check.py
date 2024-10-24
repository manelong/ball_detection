import os

def check_image_sequence_and_log_missing(directory, log_file='missing_numbers.txt'):
    # 初始化日志内容
    log_content = ""

    # 遍历一级目录
    for root, dirs, _ in os.walk(directory):
        # 仅处理包含.jpg文件的目录
        if any(file.endswith('.jpg') for file in os.listdir(root)):
            # 对当前目录下的图片进行序号检查
            subfolder_result = check_sequence_in_subfolder(root)
            # 如果有缺失，合并到总的日志内容
            if subfolder_result:
                log_content += subfolder_result + "\n"
    
    # 写入日志文件
    with open(os.path.join(directory, log_file), 'w') as file:
        file.write(log_content)
        
    if log_content:
        print(f"已记录缺失的图片序号及对应子文件夹到 '{log_file}' 中。")
    else:
        print("所有图片序号连续，无缺失。")

import os

def check_sequence_in_subfolder(subfolder_path):
    # 获取子文件夹中所有图片文件名
    image_files = [f for f in os.listdir(subfolder_path) if f.endswith('.jpg')]
    
    # 对文件名进行排序
    image_files.sort()
    
    # 计算最大序号和图片总数
    max_number = 0
    for image_file in image_files:
        try:
            file_number = int(''.join(filter(str.isdigit, image_file)))
            max_number = max(max_number, file_number)
        except ValueError:
            print(f"警告：无法从文件名 '{image_file}' 中提取有效的序号。")
            continue
    
    # 检查图片总数是否等于最大序号，若不等则进一步检查缺失序号
    if len(image_files) < max_number:
        expected_number = 1
        missing_numbers = []

        for image_file in image_files:
            try:
                file_number = int(''.join(filter(str.isdigit, image_file)))
            except ValueError:
                continue
            
            while expected_number < file_number:
                missing_numbers.append(expected_number)
                expected_number += 1
            
            expected_number += 1
        
        # 构建日志条目，包括子文件夹路径和缺失的序号
        if missing_numbers:
            return f"子文件夹 '{os.path.relpath(subfolder_path)}': 缺失序号: {', '.join(map(str, missing_numbers))}"
    else:
        return None  # 图片数量等于最大序号，表明序列连续或完全缺失

# 使用示例
directory_path = 'D:\multi_ball\multi_ball\pic'  # 替换为你的图片文件夹路径
check_image_sequence_and_log_missing(directory_path)