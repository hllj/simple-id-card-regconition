# import package
import midv500

# set directory for dataset to be downloaded
dataset_dir = 'data/midv500/'

# download and unzip midv500 dataset
midv500.download_dataset(dataset_dir)

# set directory for coco annotations to be saved
export_dir = 'data/midv500/'

# convert midv500 annotations to coco format
midv500.convert_to_coco(dataset_dir, export_dir)
