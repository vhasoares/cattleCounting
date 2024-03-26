
rm -rf env

virtualenv -p python3 env
source env/bin/activate

pip install -r requirement


echo 'SETUP DONE'
