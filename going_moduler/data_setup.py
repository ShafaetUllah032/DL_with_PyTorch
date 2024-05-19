
"""
contains functionality for creating pytorch Dataloader for
image classification data.

"""
import os 
from torch.utils.data import  dataloader
from torchvision import datasets,transforms  


NUM_WORKER=os.cpu_count()

def create_dataloader(
    train_dir: str,
    test_dir:  str,
    transform:  transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKER):
    
    """ Creates training and testing Dataloaders 

    Takes in a training directories and testing directories path and turns then into 
    pytorch Datasets and then into pytorch DataLoaders.

    Args:
        train_dir: path to training directory
        test_dir: path to testing directory
        transform: torchvision transforms to perform on training and testing data
        batch_size: Number of samples per batch in each of the DataLoaders
        num_workers: An integer for number of workers per dataLoader.

    Returns:
        A tuple of (triain_dataloader,test_dataloader,class_names)
        where class_name is a list of the target classes.
        Example usages:
        train_dataloader,test_dataloader,class_names=\
            =create_dataloader(train_dir:path/to/train_dir,
                            test_dir: path/to/test_dri,
                            transform=some_transform,
                            batch_size=32,
                            num_workers=4)

    """

    # Use ImageFolder to create dataset(s)

    train_data=datasets.ImageFolder(root=train_dir,
                                    transform=transform)

    test_data=datasets.ImageFolder(root=test_dir,
                                transform=transforms)
                            
    # Get class names
    class_names=train_data.classes


    # Turn Image into dataloader

    train_dataloader=DataLoader(dataset=train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_worker=NUM_WORKER,
                                pin_memory=True)

    test_dataloader=DataLoader(dataset=test_data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_worker=NUM_WORKER,
                            pin_number=True)

    return train_dataloader,test_dataloader,class_names



