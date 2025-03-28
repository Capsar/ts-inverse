#!/bin/bash
source ~/.profile

export PATH="$CONDA_PATH:$PATH"

echo $PATH

# Base name for tmux sessions
SESSION_BASE="JupyterNotebook"

# Base port number for Jupyter notebooks
BASE_PORT=8888

# Function to find the next available session number and port
find_next_available() {
    local session_num=0
    local port=$BASE_PORT
    while tmux has-session -t ${SESSION_BASE}${session_num} 2>/dev/null; do
        let session_num++
        let port=BASE_PORT+session_num
    done
    echo $session_num $port
}

# Find the next available session number and port
read session_num port <<< $(find_next_available)

# Create a new tmux session with a dynamic name based on the available session number
SESSION_NAME="${SESSION_BASE}${session_num}"

tmux new-session -d -s $SESSION_NAME

echo Session name is going to be: $SESSION_NAME

# # Sometimes the conda environment needs to be activated after starting the tmux session
tmux send-keys -t $SESSION_NAME "conda activate $CONDA_ENV_NAME" C-m

# # Start a Jupyter notebook on the determined port without opening a browser
tmux send-keys -t $SESSION_NAME "jupyter notebook --port=$port" C-m

# echo "Started Jupyter Notebook in tmux session: $SESSION_NAME on port: $port"
sleep 5

jupyter notebook list

# # Optionally, attach to the session (comment out if not desired)
# tmux attach -t $SESSION_NAME


# For the tasks disable VS code automatically enabling environment
# Also disable Conda enabling base environment: conda config --set auto_activate_base false
