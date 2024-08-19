#!/bin/bash

# Define the base paths
PYTHON_SCRIPT="/data/taeho/Email-RAG/ours_benchmark.py"
INPUT_BASE="/data/taeho/email_questions_"
OUTPUT_BASE="/data/taeho/ours_"

# Define the question types
QUESTION_TYPES=("meta_data"  "longform"  "factoid" "yes_no")

# Loop through each question type
for type in "${QUESTION_TYPES[@]}"
do
    echo "Running benchmark for $type questions..."
    
    INPUT_FILE="${INPUT_BASE}${type}.json"
    OUTPUT_FILE="${OUTPUT_BASE}${type}.json"
    
    # Run the Python script
    python "$PYTHON_SCRIPT" --input_file "$INPUT_FILE" --output_file "$OUTPUT_FILE"
    
    echo "Completed benchmark for $type questions. Output saved to $OUTPUT_FILE"
    echo "----------------------------------------"
done

echo "All benchmarks completed."