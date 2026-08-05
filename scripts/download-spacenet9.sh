mkdir -p data/spacenet9
wget https://spacenet-dataset.s3.us-east-1.amazonaws.com/spacenet/SN9_cross-modal/train.zip -O data/spacenet9/train.zip
wget https://spacenet-dataset.s3.us-east-1.amazonaws.com/spacenet/SN9_cross-modal/testpublic.zip -O data/spacenet9/testpublic.zip

unzip data/spacenet9/train.zip -d data/spacenet9/
unzip data/spacenet9/testpublic.zip -d data/spacenet9/

rm data/spacenet9/train.zip
rm data/spacenet9/testpublic.zip