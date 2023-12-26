# create example dataset
from clearml import StorageManager, Dataset

# Create a dataset with ClearML`s Dataset class
dataset = Dataset.create(
    dataset_project="mlops2_2", dataset_name="kaggle_dataset"
)

# add the example csv
PATH_base = './data'
dataset.add_files(path=PATH_base + '/train.csv')
dataset.add_files(path=PATH_base + '/test.csv')

# Upload dataset to ClearML server (customizable)
dataset.upload()

# commit dataset changes
dataset.finalize()
