"""
containing functionality for creating data folder 
by which data we are going to experiments
"""

import torch
import torchvision
import os 
import zipfile
import requests

from typing import Tuple
from pathlib import Path

def get_data(path: str,
             sub_folder: str,
             url : str,
             ) -> Tuple[str,str]:

    """Creating a data directory and store data to train and test the model  
    
    Take the folder path and name of train and test data continer folder 
    url from which we are going to collect the data.

    Args:
        path       : directroy in where we are creating data folder to store data.
        sub_folder : name of the data folder in where we store train and test data.
        url        : Line of the url from where we are going to collect the data . 

    Return:
        A touple of (train directory and test directory)
        Which is the path of train data and test data
        Exaple usage:
          train_dir, test_dir= get_data(path="data",
                                        sub_folder="image_path",
                                        url="abc/.../...")
     
    """
    data_path=Path(path)
    image_path=data_path / sub_folder
    data_raw_url=url

    if image_path.is_dir():
        print("path_exists , skip creating .....")
    else:
        image_path.mkdir(parents=True, exist_ok=True)
        print("directory created, time to wrtie the file ...... ")

        with open(image_path/ "pizza_steak_sushi.zip","wb") as file:
            request=requests.get(data_raw_url)
            print("writing data to the path:  ",image_path)
            file.write(request.content)
            print("file written done...")
        
        with zipfile.ZipFile(image_path / "pizza_steak_sushi.zip","r") as zip_ref:
            zip_ref.extractall(image_path)
            print("extract done ...")

    if image_path/"pizza_steak_sushi.zip".is_file():
        os.remove(image_path/"pizza_steak_sushi.zip")
        print("zip file removed ...... ")

    # Setup train and testing paths
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    return train_dir, test_dir
