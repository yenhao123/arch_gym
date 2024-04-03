#!/bin/bash

home_dir="/home/user/Desktop/oss-arch-gym/project_ssd/tuning_tools"
device="/dev/nvme0"

# Read config from "config.json" and parse to dictionary format
function read_config_json() {
    config_path="$home_dir/config.json"

    # Check if the file exists
    if [ -f "$config_path" ]; then
        # Parse JSON using jq and store it in the 'config' variable
        config=$(cat $config_path)
        echo "$config"
    else
        echo "Error: Config file not found!"
        exit 1
    fi
}

# Function to check if the values are the same
function is_same_as_current_value() {
    # Get the current value from the nvme get-feature command output
    local config_value="$1"
    local output=$(sudo nvme get-feature "$device" -f "$2")
    local current_value=$(echo "$output" | grep -oP 'Current value:0x\K[0-9a-fA-F]+' | awk '{print $1+0}')
    if [[ "$config_value" == "$current_value" ]]; then
        echo 1
    else
        echo 0
    fi
}

# Get the config content, queue weight, power state, workload type, InterruptCoalescing

config_set=$(read_config_json)

# Consider the option: queue weight
option_id="0x1"
weight=$(echo "$config_set" | jq '.configuration[0]')
burst=$(echo "$config_set" | jq '.configuration[1]')

if [ "$weight" -eq 1 ]; then
    weight_cg="0x000000"
elif [ "$weight" -eq 2 ]; then
    weight_cg="0x010000"
elif [ "$weight" -eq 3 ]; then
    weight_cg="0x0A0000"
elif [ "$weight" -eq 4 ]; then
    weight_cg="0x020100"
elif [ "$weight" -eq 5 ]; then
    weight_cg="0x0A0100"
elif [ "$weight" -eq 6 ]; then
    weight_cg="0x0A0A00"
fi

config="${weight_cg}0${burst}"

if [ "$weight" -ge 1 ] && [ "$weight" -le 6 ] && [ "$burst" -ge 0 ] && [ "$burst" -le 7 ]; then
    echo "Start setting Arbitration policy for NVMe device..."
    sudo nvme set-feature "$device" -f "$option_id" -v "$config"
else
    echo "Error: Invalid value!"
    exit 1
fi

# Consider the option: power management policy
option_id="0x2"
power_state=$(echo "$config_set" | jq '.configuration[2]')
#workload_type=$(echo "$config_set" | jq '.configuration[3]')
#config=="${workload_type}0000${power_state}"
config="${power_state}"

if [ "$power_state" -ge 0 ] && [ "$power_state" -le 4 ]; then
    echo "Start setting power management policy for NVMe device..."
    sudo nvme set-feature "$device" -f "$option_id" -v "$config"
else
    echo "Error: Invalid value!"
    exit 1
fi

# Consider the option: interrupt coalescing policy
option_id="0x8"
ic_state=$(echo "$config_set" | jq '.configuration[4]')
if [ "$ic_state" -eq 0 ]; then
    config="0x0000"
elif [ "$ic_state" -eq 1 ]; then
    config="0x0100"
elif [ "$ic_state" -eq 2 ]; then
    config="0x0200"
elif [ "$ic_state" -eq 3 ]; then
    config="0x0A00"
elif [ "$ic_state" -eq 4 ]; then
    config="0x6400"
elif [ "$ic_state" -eq 5 ]; then
    config="0xFF00"
fi

if [ "$ic_state" -ge 0 ] && [ "$ic_state" -le 5 ]; then
    echo "Start setting interrupt coalescing policy for NVMe device..."
    sudo nvme set-feature "$device" -f "$option_id" -v "$config"
else
    echo "Error: Invalid value!"
    exit 1
fi

# Consider the option: write atomicity policy
option_id="0xa"
atomicity_state=$(echo "$config_set" | jq '.configuration[6]')
config="${atomicity_state}"
if [ "$atomicity_state" -ge 0 ] && [ "$atomicity_state" -le 1 ]; then
    echo "Start setting write atomicity policy for NVMe device..."
    sudo nvme set-feature "$device" -f "$option_id" -v "$config"
else
    echo "Error: Invalid value!"
    exit 1
fi
