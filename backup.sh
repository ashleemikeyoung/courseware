#!/usr/bin/env bash

set -euo pipefail

# Ensure exactly two arguments are provided
if [[ $# -ne 2 ]]; then
    echo "Usage: $(basename "$0") <source_file> <backup_directory>"
    exit 1
fi

SOURCE_FILE="$1"
BACKUP_DIR="$2"

# Ensure the source file exists
if [[ ! -f "$SOURCE_FILE" ]]; then
    echo "Error: Source file does not exist: $SOURCE_FILE"
    exit 1
fi

# Create the backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Generate timestamp (YYYYMMDD_HHMMSS)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Extract just the filename
FILENAME=$(basename "$SOURCE_FILE")

# Build the backup filename
BACKUP_FILE="${BACKUP_DIR}/${FILENAME}.${TIMESTAMP}"

# Copy while preserving attributes
cp -p "$SOURCE_FILE" "$BACKUP_FILE"

echo "Backup created:"
echo "  $BACKUP_FILE"
