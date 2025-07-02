#!/usr/bin/env python3
"""
RAG演示系统启动脚本
"""

import sys
import os

def main():
    print("🔍 RAG演示系统")
    print("=" * 40)
    print("请选择运行模式:")
    print("1. Web界面 (推荐)")
    print("2. 命令行演示")
    print("3. 退出")
    print("=" * 40)
    
    while True:
        try:
            choice = input("请输入选择 (1-3): ").strip()
            
            if choice == "1":
                print("\n🚀 启动Web界面...")
                print("请在浏览器中访问: http://localhost:8000")
                print("按 Ctrl+C 停止服务器")
                os.system("python app.py")
                break
                
            elif choice == "2":
                print("\n💻 启动命令行演示...")
                os.system("python demo.py")
                break
                
            elif choice == "3":
                print("\n👋 再见！")
                break
                
            else:
                print("❌ 无效选择，请输入 1、2 或 3")
                
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main() 