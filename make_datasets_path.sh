# Set root directory (Please put the root directory here)
ROOT_DIR=

# Set the dataset file path
cat > $ROOT_DIR/FedSDAF/configs/default.py <<EOF
pacs_path = '$ROOT_DIR/FedSDAF/datasets/pacs/'
officehome_path = '$ROOT_DIR/FedSDAF/datasets/office_home/'
vlcs_path = '$ROOT_DIR/FedSDAF/datasets/vlcs/'
domainnet_path = '$ROOT_DIR/FedSDAF/datasets/domainnet/'
log_count_path = '$ROOT_DIR/FedSDAF/log/'
EOF
