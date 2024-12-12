#!/usr/bin/env sh
#!/bin/bash

# 帮助函数
show_help() {
    echo "脚本使用说明:"
    echo "  ./1.sh ss       - 列出当前目录下所有文件/文件夹的大小（仅第一层）"
    echo "  ./1.sh git      - 从 https://github.com/gongshangzheng/STA 克隆代码"
    echo "  ./1.sh git history  - 克隆代码并保留完整提交历史"
    echo "  ./1.sh -h       - 显示帮助信息"
    echo "  ./1.sh help     - 显示帮助信息"
}

# 列出文件/文件夹大小的函数
list_sizes() {
    echo "当前目录下文件和文件夹大小："
    du -sh * 2>/dev/null
}

# Git克隆函数
clone_repo() {
    local REPO_URL="https://github.com/gongshangzheng/STA"

    # 检查是否需要保留完整历史
    if [ "$1" == "history" ]; then
        git clone "$REPO_URL"
    else
        git clone --depth 1 "$REPO_URL"
    fi
}

# 主逻辑
case "$1" in
    "ss")
        list_sizes
        ;;
    "git")
        clone_repo "$2"
        ;;
    "-h"|"help")
        show_help
        ;;
    *)
        echo "无效的参数。使用 -h 或 help 查看帮助。"
        exit 1
        ;;
esac

exit 0
