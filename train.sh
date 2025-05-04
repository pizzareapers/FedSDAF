# Set root directory (Please put the root directory here)
ROOT_DIR=

# Train on PACS dataset
nohup python -u $ROOT_DIR/FedSDAF/train.py \
    --dataset pacs \
    --batch_size 128 \
    > $ROOT_DIR/FedSDAF/train_log/PACS.log 2>&1 &

# Train on OfficeHome dataset
nohup python -u $ROOT_DIR/FedSDAF/train.py \
    --dataset officehome \
    --batch_size 128 \
    > $ROOT_DIR/FedSDAF/train_log/OfficeHome.log 2>&1 &

# Train on VLCS dataset
nohup python -u $ROOT_DIR/FedSDAF/train.py \
    --dataset vlcs \
    --batch_size 64 \
    > $ROOT_DIR/FedSDAF/train_log/VLCS.log 2>&1 &

# Train on DomainNet dataset
nohup python -u $ROOT_DIR/FedSDAF/train.py \
    --dataset domainnet \
    --batch_size 1024 \
    > $ROOT_DIR/FedSDAF/train_log/DomainNet.log 2>&1 &