"""
Contains various utility functions for Pytorch model training and saving
"""
from pathlib import Path
import torch

def save_model(model:torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a pytorch model to a target directory.

    Args:
        model      : A target pytorch model to save.
        target_dir : A directory for saving the model to.
        model_name : A filename for the saved model. should include either ".pth" or
        ".pt" as the file extenstion.
    
    Example usages:
    save_model(model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(train_dir)
    train_dir_path.mkdir(parents=True,
                         exist_ok=True)

    # Create a model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth' "
    model_save_path=target_dir_path/model_name

    # Save the model  state_dict()
    print(f"[INFO] Saving model to : {mdoel_save_path}")
    torch.save(obj=model_state_dict(),
               f=model_save_path)
