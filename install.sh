echo Installing S3FD testing components...
echo -----------------------------------------------------------------------------------------
cd ./verification/s3fd/
./get_test.sh
cd ..

echo Installing deep_privacy development components
echo -----------------------------------------------------------------------------------------
cd ./deep_privacy
echo Downloading models, this might take a while...
python3 download_files.py
cd ..

echo Installing deep_privacy testing components
echo -----------------------------------------------------------------------------------------
cd ./verification/deep_privacy
./get_test.sh
echo Downloading models, this might take a while...
python3 download_files.py
cd ..

