#!/bin/bash

# Function to cleanup background processes on script exit
cleanup() {
    echo "Shutting down services..."
    kill $(jobs -p) 2>/dev/null
    exit
}

# Set up cleanup trap
trap cleanup EXIT INT TERM

# # Start the streaming server
# echo "Starting streaming server..."
# python3 streaming_server.py &

# Start the API server
echo "Starting API server..."
python3 api_server.py &

# Start the React client
echo "Starting web client..."
cd web-client && npm start &

# Start the Smart CCTV server
echo "Starting Smart CCTV server..."
python3 smart_cctv.py &

# Wait for all processes
wait 