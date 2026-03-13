import torch
from typing import Dict


def create_ground_truth_parameters() -> Dict[str, torch.Tensor]:
    """Create ground truth parameters for testing."""
    param_dict = {}
    
    param_dict['emissions'] = []
    param_dict["trans_logits"] = []
    param_dict["init_logits"] = []

    # State 0: Standard LV parameters (log scale)
    param_dict['emissions'] = (
        torch.log(torch.Tensor(
            [[1.0, 1.0, 1.0, 1.0, 0.05, 0.05], 
             [0.5, 0.5, 0.2, 0.2, 0.05, 0.05]]
        ))
    )
    
    # Transition matrix where delta t = dt (log scale)
    param_dict["trans_logits"] = torch.log(torch.Tensor([[0.999, 0.001], [0.001, 0.999]]))
    
    # Initial distribution (log scale)
    param_dict["init_logits"] = torch.log(torch.Tensor([0.5, 0.5]))
    
    return param_dict