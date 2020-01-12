wget -r -np -nd -A .zip http://vision.middlebury.edu/stereo/data/scenes2014/zip/ -P training/
cd training/
unzip "*.zip"
rm *.zip
cd ..
