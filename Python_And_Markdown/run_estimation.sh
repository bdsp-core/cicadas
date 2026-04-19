#!/bin/bash

# Script to run MATLAB estimation test
# Adjust the MATLAB path as needed for your system

MATLAB_CMD="/Applications/MATLAB_R2024b.app/bin/matlab"

if [ ! -f "$MATLAB_CMD" ]; then
    # Try alternative paths
    MATLAB_CMD="/Applications/MATLAB_R2023b.app/bin/matlab"
    if [ ! -f "$MATLAB_CMD" ]; then
        MATLAB_CMD="/Applications/MATLAB_R2023a.app/bin/matlab"
        if [ ! -f "$MATLAB_CMD" ]; then
            echo "MATLAB not found. Please update the path in this script."
            exit 1
        fi
    fi
fi

echo "Using MATLAB: $MATLAB_CMD"
echo "Running PKPD parameter estimation test..."

$MATLAB_CMD -batch "test_PKPD_estimation"