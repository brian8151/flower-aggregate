#!/bin/bash

PATH=/home/ec2-user/flwr-test/flower-aggregate
LOG_PATH=/home/ec2-user/flwr-test/
cd $PATH


# Print the starting directory
echo "Running from flower-aggregate directory: $(pwd)"
# Default to 'start' if no argument is given
cmd=${1:-start}

# Define the function to setup the virtual environment and install dependencies
setup_env() {
    if [ ! -d "venv" ]; then
        echo "Virtual environment not found. Creating one..."
        python -m venv venv
        echo "Virtual environment created."

        echo "Activating virtual environment..."
        source venv/bin/activate

        echo "Installing dependencies from requirements.txt..."
        pip install -r requirements.txt

        echo "Dependencies installed."
    else
        echo "Virtual environment already exists."
        echo "Activating virtual environment..."
        source venv/bin/activate
    fi
}


# Define the function to start the application
start_app() {
    setup_env
    echo "Starting Flower agg application..."
    /usr/bin/nohup python -m flower-aggregate > $LOG_PATH/flwr-agg.log 2>&1 &
    echo "Flower agg started in the background, logs: $LOG_PATH/flwr-agg.log"
}

# Define the function to stop the application
stop_app() {
    pid=$(ps aux | grep -i 'flower-aggregate' | grep -v grep | awk '{print $2}')
    if [ ! -z "$pid" ]; then
        echo "Found process $pid, shutting down flower-aggregate..."
        kill -9 $pid
        echo "Flower agg stopped."
    else
        echo "No process found for flower-aggregate."
    fi
    echo "Deactivating virtual environment..."
    deactivate
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

