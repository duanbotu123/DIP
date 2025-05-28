import numpy as np
import os

def load_npz_file(file_path):
    """
    从 .npz 文件加载数据并打印其内容信息。

    参数:
        file_path (str): .npz 文件的路径。

    返回:
        类似字典的对象: 一个 NpzFile 对象（行为类似字典），包含
                          来自 .npz 文件的数组。如果文件无法加载，则返回 None。
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误：在 '{file_path}' 未找到文件")
            return None
        
        # 加载 .npz 文件
        data = np.load(file_path)
        
        print(f"成功加载 '{file_path}'")
        print("-" * 30)
        
        # 列出存储在 .npz 文件中的所有数组（键）
        print("此文件中的数组（键）:")
        for arr_name in data.files:
            print(f"- {arr_name}")
        print("-" * 30)
        
        # 你可以使用数组的名称作为键来访问单个数组
        # 例如，如果有一个名为 'my_array' 的数组:
        # if 'my_array' in data.files:
        #     my_data_array = data['my_array']
        #     print(f"'my_array' 的形状: {my_data_array.shape}")
        #     print(f"'my_array' 的数据类型: {my_data_array.dtype}")
        #     print(f"'my_array' 的前5个元素:\n{my_data_array[:5]}")
        # else:
        #     print("在文件中未找到数组 'my_array'。")
            
        return data
        
    except Exception as e:
        print(f"加载 '{file_path}' 时发生错误: {e}")
        return None

if __name__ == "__main__":
    # --- 配置 ---
    # 将 'your_file.npz' 替换为你的 .npz 文件的实际路径
    npz_file_path = '/nas_data/dataset/4dhoi/25_04_03/person00/motion01/body_track/smpl_params.npz' 
    
    # --- 示例：如果你没有 .npz 文件，创建一个用于测试的虚拟 .npz 文件 ---
    # (如果你有自己的 .npz 文件，可以注释掉此部分)
    if not os.path.exists(npz_file_path) or npz_file_path == 'your_file.npz':
        print(f"未找到 '{npz_file_path}' 或其为默认值。正在创建一个用于演示的虚拟 .npz 文件。")
        dummy_array_a = np.arange(10)
        dummy_array_b = np.random.rand(3, 4)
        np.savez_compressed('dummy_test_file.npz', array_one=dummy_array_a, array_two=dummy_array_b)
        npz_file_path = 'dummy_test_file.npz' # 指向虚拟文件
        print(f"虚拟文件 '{npz_file_path}' 已创建，包含数组 'array_one' 和 'array_two'。")
        print("-" * 30)
    # --- 虚拟文件创建结束 ---

    # 加载 .npz 文件
    loaded_data = load_npz_file(npz_file_path)
    
    if loaded_data:
        print("-" * 30)
        print("从加载的数据中访问特定数组:")
        
        # 示例：遍历数组并打印一些信息
        for arr_name in loaded_data.files:
            array_content = loaded_data[arr_name]
            print(f"\n数组 '{arr_name}' 的详细信息:")
            print(f"  形状: {array_content.shape}")
            print(f"  数据类型: {array_content.dtype}")
            # 打印少量元素（处理非常大的数组时要小心）
            if array_content.size > 0:
                if array_content.ndim == 0: # 标量
                     print(f"  值: {array_content.item()}")
                elif array_content.ndim == 1:
                    print(f"  前最多5个元素: {array_content[:min(5, array_content.shape[0])]}")
                else: # 多维数组
                    print(f"  第一行的前最多5个元素 (如果是2D+): {array_content[0][:min(5, array_content.shape[1])] if array_content.shape[1] > 0 else 'N/A'}")
            else:
                print("  数组为空。")
        
        # 示例：如果你知道数组的名称，可以访问特定数组
        # (假设 'array_one' 存在于虚拟示例或你的文件中)
        if 'array_one' in loaded_data.files:
            specific_array = loaded_data['array_one']
            print(f"\n成功访问 'array_one'。其内容为: {specific_array}")

        # 如果不再需要文件对象，记得关闭它，
        # 尽管使用 `with` 语句（如果使用）或当对象超出作用域时，
        # 通常会自动处理。
        # 对于 NpzFile，如果长时间保持打开状态，显式关闭是个好习惯。
        loaded_data.close()
        print("-" * 30)
        print(f"已关闭 '{npz_file_path}'。")

    else:
        print("加载 .npz 文件失败。")

    # --- 如果创建了虚拟文件，则进行清理 ---
    if npz_file_path == 'dummy_test_file.npz' and os.path.exists('dummy_test_file.npz'):
        os.remove('dummy_test_file.npz')
        print(f"虚拟文件 'dummy_test_file.npz' 已被删除。")