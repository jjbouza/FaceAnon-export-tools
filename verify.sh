echo Testing S3FD components...
echo -----------------------------------------------------------------------------------------
cd ./verification/s3fd
python3 align_test.py
cd ../..

echo Testing deep_privacy components
echo -----------------------------------------------------------------------------------------
cd ./verification/deep_privacy
python3 test_deep_privacy.py
echo Output example:
