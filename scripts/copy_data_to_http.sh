#! /bin/bash

#  Mount the HTTP server directory if not already mounted
if [ ! -d "$HOME/mnt/html/public_html" ]; then
  rcp drive mount -d html
fi

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
# Rewrite only the DATA_URL line.  config.py also holds the HuggingFace pin and
# the comments explaining how to retire this backend, so overwriting the whole
# file (as this used to do) would silently destroy them.
sed -i "s|^DATA_URL = .*|DATA_URL = '$DATA_URL'|" ../generatedata/config.py

