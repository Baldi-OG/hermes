# !/bin/bash

# Download Reuters-50-50 dataset
wget https://archive.ics.uci.edu/static/public/217/reuter+50+50.zip
unzip reuter+50+50.zip -d reuters-50-50
rm reuter+50+50.zip
mv reuters-50-50/* src/datasets/reuters_50_50/data
rm -rf reuters-50-50