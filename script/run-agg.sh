#!/bin/bash

PATH=/home/ec2-user/flwr-test/flower-aggregate

cd $PATH

# Print the starting directory
echo "Running from flower-aggregate directory: $(pwd)"

# Default to 'start' if no argument is given
cmd=${1:-start}

# Define the function to start the application
start_app() {
    echo "Activating virtual environment..."
    source ./venv/bin/activate
    echo "Starting Python flower agg application in the background..."
    nohup python -m flower-aggregate > flwr-agg.log 2>&1 &
    echo $! > flower-agg.pid
    echo "Flower agg started with PID $(cat flower-agg.pid)"
}

# Define the function to stop the application
stop_app() {
    echo "Stopping flower agg..."
    if [ -f flower.pid ]; then
        kill $(cat flower-agg.pid) && rm flower-agg.pid
        echo "Flower agg stopped."
    else
        echo "No PID file found. Is the application running?"
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

