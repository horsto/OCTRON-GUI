from pathlib import Path
import yaml

def check_boxmot_trackers(trackers_yaml_path):
    """
    Check the availability of boxmot trackers loaded from 
    the trackers_yaml_path .yaml file. 
    
    
    Parameters
    ----------
    trackers_yaml_path : str or Path
        Path to the trackers yaml file. 
        For example "tracking/boxmot_trackers.yaml"

    Returns
    -------
    trackers_dict : dict
        Dictionary of the models and their configurations. 
        For example:
        {
            'BaseTracker': {
                'name: 'base tracker',
                'config_path': 'configs/basetracker.yaml',
            },
            
        ...
          
    """
    trackers_yaml_path = Path(trackers_yaml_path)
    assert trackers_yaml_path.exists(), f"Path {trackers_yaml_path} does not exist"
    
    # Load the model YAML file and convert it to a dictionary
    with open(trackers_yaml_path, 'r') as file:
        trackers_dict = yaml.safe_load(file)
    return trackers_dict