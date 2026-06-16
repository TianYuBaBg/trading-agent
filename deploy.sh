#!/bin/bash
# 部署脚本 — 用于 Linux 服务器
#
# 使用：
#   ./deploy.sh start    启动后台运行
#   ./deploy.sh stop     停止
#   ./deploy.sh status   查看状态
#   ./deploy.sh logs     查看日志

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$PROJECT_DIR/server.pid"
LOG_FILE="$PROJECT_DIR/data/server.log"

start() {
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "已在运行中，PID: $(cat "$PID_FILE")"
        exit 1
    fi
    cd "$PROJECT_DIR"
    nohup python run_server.py > "$LOG_FILE" 2>&1 &
    PID=$!
    echo $PID > "$PID_FILE"
    echo "已启动，PID: $PID"
    echo "日志: $LOG_FILE"
}

stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "未在运行"
        exit 0
    fi
    PID=$(cat "$PID_FILE")
    kill $PID 2>/dev/null
    rm -f "$PID_FILE"
    echo "已停止，PID: $PID"
}

status() {
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "运行中，PID: $(cat "$PID_FILE")"
    else
        echo "未运行"
    fi
}

logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -50 "$LOG_FILE"
    else
        echo "暂无日志"
    fi
}

case "${1:-}" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    logs) logs ;;
    restart)
        stop
        sleep 2
        start
        ;;
    *)
        echo "用法: $0 {start|stop|status|logs|restart}"
        exit 1
        ;;
esac
