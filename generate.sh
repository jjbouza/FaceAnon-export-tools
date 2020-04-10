echo Exporting S3FD components...
echo -----------------------------------------------------------------------------------------
cd s3fd
echo Exporting ONNX...
python3 save_onnx.py
echo Exporting CoreML...
python3 save_coreml.py
echo Exporting processing modules...
python3 ssfd.py
if [ -d "./outputs/sfd_detector.mlmodel /Users/josebouza/Projects/Face\ Anonymizer/Face\ Anonymizer/models" ]; then
    echo Copying models...
    cp ./outputs/sfd_detector.mlmodel /Users/josebouza/Projects/Face\ Anonymizer/Face\ Anonymizer/models
    cp ./outputs/*.pt /Users/josebouza/Projects/Face\ Anonymizer/Face\ Anonymizer/Modules/
else
    echo Warning: did not copy generated model files to XCode.
fi
cd ..
echo S3FD Processing Done.
echo -----------------------------------------------------------------------------------------


echo Exporting deep_privacy components...
echo -----------------------------------------------------------------------------------------
cd verification/deep_privacy
python3 inout_generate.py
cp preprocess_input.pt generator_inputs.pt postprocess_inputs.pt ../../deep_privacy/deep_privacy
cd ../../deep_privacy
echo Exporting Torchscript components...
python3 save_torchscript.py
echo Warning: did not copy generated model files to XCode.
cd ..
