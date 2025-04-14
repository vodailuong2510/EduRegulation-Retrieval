sudo apt install teseract-ocr
sudo apt install teseract-ocr-vie

dvc init
dvc add data
dvc remote add -d myremote /.dvc/dvcstore -f