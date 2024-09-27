#!/bin/bash

# Define the directories
teal_dir="teal/"
black_dir="black/"

# Function to process files
process_files() {
    dir=$1
    output_dir="${dir%/}_output"  # Remove trailing slash and add _output
    mkdir -p "$output_dir"        # Create the output directory if it doesn't exist
    echo "Processing files in $dir directory, output will be saved to $output_dir..."

    for file in "$dir"*.cwa; do
        if [[ -f "$file" ]]; then
            echo "Processing $file"
            stepcount "$file" -o "$output_dir"
        else
            echo "No .cwa files found in $dir"
        fi
    done
}

# Process blue/ directory
process_files $teal_dir

# Process black/ directory
process_files $black_dir
