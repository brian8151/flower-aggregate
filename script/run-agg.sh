#!/bin/bash

# Configurationi
APP_DIR_BASE=/home/ec2-user/flwr-test
APP_DIR=$APP_DIR_BASE/flower-aggregate
LOG_PATH="/home/ec2-user/flwr-test"  # Ensure this directory exists
PYTHON_BIN="/usr/local/bin/python3.9"

cd $APP_DIR
echo "Running from flower-aggregate directory: $(pwd)"

# Default to 'start' if no argument is given
cmd=${1:-start}

# Define the function to install dependencies if needed
install_deps() {
    if [ ! -f ".deps_installed" ]; then
        echo "Installing dependencies from requirements.txt..."
        $PYTHON_BIN -m pip install -r requirements.txt
        touch .deps_installed
        echo "Dependencies installed."
    else
        echo "Dependencies already installed."
    fi
}

# Define the function to start the application
start_app() {
    install_deps
    cd $APP_DIR_BASE
    echo "Starting Flower agg application..."
    nohup $PYTHON_BIN flower-aggregate 2>&1 &
    echo "Flower agg started in the background"
}

# Define the function to stop the application
stop_app() {
    local pid=$(ps aux | grep -i 'flower-aggregate' | grep -v grep | awk '{print $2}')
    if [[ ! -z "$pid" ]]; then
        echo "Found process $pid, shutting down flower-aggregate..."
        kill $pid
        sleep 2
        if ps -p $pid > /dev/null; then
            echo "Process $pid did not terminate, forcing shutdown..."
            kill -9 $pid
        fi
        echo "Flower aggregate stopped."
    else
        echo "No process found for flower-aggregate."
    fi
}

# Handle the command line argument
case "$cmd" in
    start)
        stop_app  # Ensure the application is not already running
        start_app
        ;;
    stop)
        stop_app
        ;;
    *)
        echo "Usage: $0 {start|stop}"
        exit 1
        ;;
esac