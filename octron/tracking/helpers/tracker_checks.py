from pathlib import Path
import yaml

def load_boxmot_trackers(trackers_yaml_path):
    """
    Load boxmot tracker overview .yaml.
    This yaml contains info about the tracker names, and where 
    their configuration files are saved. 
    
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


def load_boxmot_tracker_config(config_yaml_path):
    
    """
    Load OCTRON boxmot tracker configuration file.
    
    Parameters
    ----------
    config_yaml_path : str or Path
        Path to the tracker config yaml file. 
        For example "configs/bytetrack.yaml"
        (This is in octron/tracking/)

    Returns
    -------
    config_dict : dict
        Config dictionary

    """
    config_yaml_path = Path(config_yaml_path)
    assert config_yaml_path.exists(), f"Path {config_yaml_path} does not exist"
    
    # Load the model YAML file and convert it to a dictionary
    with open(config_yaml_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict

    