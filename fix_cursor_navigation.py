#!/usr/bin/env python3
"""
修复Cursor中Python函数跳转问题的脚本
"""

import os
import json
import subprocess
import sys

def fix_cursor_navigation():
    """修复Cursor导航问题"""
    
    print("🔧 正在修复Cursor Python函数跳转问题...")
    
    # 1. 检查当前Python解释器
    python_path = "/home/user/anaconda3/envs/env_isaaclab/bin/python"
    if not os.path.exists(python_path):
        print(f"❌ Python解释器不存在: {python_path}")
        return False
    
    # 2. 检查Isaac Lab路径
    isaaclab_paths = [
        "/home/user/IsaacLab/source/isaaclab",
        "/home/user/IsaacLab/source/isaaclab_tasks", 
        "/home/user/IsaacLab/source/isaaclab_assets",
        "/home/user/IsaacLab/source/isaaclab_rl",
        "/home/user/IsaacLab/source/isaaclab_mimic"
    ]
    
    missing_paths = []
    for path in isaaclab_paths:
        if not os.path.exists(path):
            missing_paths.append(path)
    
    if missing_paths:
        print(f"❌ 缺少以下路径: {missing_paths}")
        return False
    
    # 3. 更新VSCode设置
    vscode_settings = {
        "python.defaultInterpreterPath": python_path,
        "python.languageServer": "Pylance",
        "python.analysis.extraPaths": [
            "${workspaceFolder}/source/isaaclab",
            "${workspaceFolder}/source/isaaclab_tasks",
            "${workspaceFolder}/source/isaaclab_assets", 
            "${workspaceFolder}/source/isaaclab_rl",
            "${workspaceFolder}/source/isaaclab_mimic"
        ],
        "python.analysis.autoImportCompletions": True,
        "python.analysis.indexing": True,
        "python.analysis.completeFunctionParens": True,
        "python.analysis.typeCheckingMode": "basic"
    }
    
    settings_file = "/home/user/IsaacLab/.vscode/settings.json"
    
    try:
        # 读取现有设置
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        existing_settings = json.loads(content)
                    else:
                        existing_settings = {}
            except json.JSONDecodeError:
                print("⚠️ 现有设置文件格式有问题，将创建新文件")
                existing_settings = {}
        else:
            existing_settings = {}
        
        # 更新设置
        existing_settings.update(vscode_settings)
        
        # 写入设置
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(existing_settings, f, indent=4, ensure_ascii=False)
        
        print(f"✅ 已更新VSCode设置: {settings_file}")
        
    except Exception as e:
        print(f"❌ 更新设置失败: {e}")
        return False
    
    # 4. 创建Python路径文件
    pythonpath_file = "/home/user/IsaacLab/.pythonpath"
    try:
        with open(pythonpath_file, 'w') as f:
            for path in isaaclab_paths:
                f.write(f"{path}\n")
        print(f"✅ 已创建Python路径文件: {pythonpath_file}")
    except Exception as e:
        print(f"❌ 创建Python路径文件失败: {e}")
    
    # 5. 测试Python导入
    print("\n🧪 测试Python模块导入...")
    test_imports = [
        "isaaclab",
        "isaaclab_tasks", 
        "rsl_rl.runners"
    ]
    
    for module in test_imports:
        try:
            result = subprocess.run([
                python_path, "-c", f"import {module}; print(f'✅ {module} 导入成功')"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print(f"✅ {module} 导入成功")
            else:
                print(f"❌ {module} 导入失败: {result.stderr}")
        except Exception as e:
            print(f"❌ {module} 导入测试失败: {e}")
    
    print("\n🎯 修复完成！请执行以下步骤：")
    print("1. 重启Cursor")
    print("2. 按 Ctrl+Shift+P，输入 'Python: Select Interpreter'")
    print("3. 选择: /home/user/anaconda3/envs/env_isaaclab/bin/python")
    print("4. 按 Ctrl+Shift+P，输入 'Python: Restart Language Server'")
    print("5. 等待语言服务器重启完成")
    
    return True

if __name__ == "__main__":
    fix_cursor_navigation()
