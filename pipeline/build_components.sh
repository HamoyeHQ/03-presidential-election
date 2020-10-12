#!/bin/sh

echo "\nBuild and push preprocess component"
./preprocess_data/build_image.sh

echo "\nBuild and push train component"
./train_model/build_image.sh

echo "\nBuild and push test component"
./test_model/build_image.sh

echo "\nBuild and push deploy component"
./deploy_model/build_image.sh
