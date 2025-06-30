#! /bin/bash

#  Mount the HTTP server directory
rcp drive mount -d html

# Copy the data files to the HTTP server directory

# generate a directory name based on the current date and time
dir_name=$(date +%Y%m%d_%H%M%S)
echo "Copying data to $HOME/mnt/html/data/generatedata/$dir_name"
mkdir -p $HOME/mnt/html/public_html/data/generatedata/$dir_name

cp ../data/processed/* $HOME/mnt/html/public_html/data/generatedata/$dir_name

DATA_URL="http://users.wpi.edu/~rcpaffenroth/data/generatedata/$dir_name"
echo "Data files copied to $DATA_URL"

# Update the DATA_URL in the config.py file
echo "Updating DATA_URL in ../generatedata/config.py"
# Use echo to write the new DATA_URL to the config.py file
echo "DATA_URL = '$DATA_URL'" > ../generatedata/config.py

