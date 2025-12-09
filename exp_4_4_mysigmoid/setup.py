from setuptools import setup
from torch.utils import cpp_extension
import os
import shutil 
setup(
    #TODO: 给出编译后的链接库名称
    name='mysigmoid_cpp',
    ext_modules=[
        cpp_extension.CppExtension(
    #TODO：以正确的格式给出编译文件即编译函数
            name='mysigmoid_cpp',
            sources=['mysigmoid.cpp']
        )
    ],
    # 执行编译命令设置
    cmdclass={						       
        'build_ext': cpp_extension.BuildExtension
    }
)

current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 获取上一级目录 (stu_upload)
parent_dir = os.path.dirname(current_dir)

found = False
# 3. 遍历当前目录寻找生成的 .so 文件
for filename in os.listdir(current_dir):
    if filename.endswith(".so"):
        src_path = os.path.join(current_dir, filename)
        dst_path = os.path.join(parent_dir, filename)
        
        print(f"Found library: {filename}")
        
        # 如果上一级目录已经有旧文件，先删除，防止权限或覆盖报错
        if os.path.exists(dst_path):
            os.remove(dst_path)
            
        # 移动文件
        shutil.move(src_path, dst_path)
        print(f"Successfully moved to: {dst_path}")
        found = True

if not found:
    print("Warning: No .so file generated or found in current directory.")


print("generate .so PASS!\n")
