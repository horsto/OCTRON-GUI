from pathlib import Path
import json
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional, Union, Dict, List, Any

from octron.sam2_octron.helpers.sam2_colors import (
    create_label_colors,
    sample_maximally_different,
)


class Obj(BaseModel):
    label: str
    suffix: str
    label_id: Optional[int] = None
    color: Optional[list] = None
    prediction_layer: Optional[Any] = None
    annotation_layer: Optional[Any] = None
    predicted_frames: List[Any] = Field(default_factory=list)
    
    # Exclude non-serializable fields
    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True
        json_encoders = {
            # Add custom encoders for non-serializable types
            tuple: lambda v: list(v),  # Convert tuples to lists
        }

    @field_validator("color")
    def check_color_length(cls, v):
        if v is not None and len(v) != 4:
            raise ValueError("Color must be a list of length 4")
        return v
    
    def add_predicted_frame(self, frame_index: int) -> None:
        """
        Add a frame index to predicted_frames if it is not already present.
        """
        if frame_index not in self.predicted_frames:
            self.predicted_frames.append(frame_index)

    def add_predicted_frames(self, frame_indices: list) -> None:
        for index in frame_indices:
            self.add_predicted_frame(index)
    
    def remove_predicted_frame(self, frame_index: int) -> None:
        """
        Remove a frame index from predicted_frames - only if it exists.
        """
        if frame_index in self.predicted_frames:
            self.predicted_frames.remove(frame_index)
            
    def remove_predicted_frames(self, frame_indices: list) -> None:
        for index in frame_indices:
            self.remove_predicted_frame(index)

    def reset_predicted_frames(self) -> None:
        """
        Reset the predicted_frames list.
        """
        self.predicted_frames = []
            
    def __repr__(self) -> str:
        return (
            f"Obj(label={self.label!r}, suffix={self.suffix!r}, label_id={self.label_id!r}, "
            f"color={self.color!r}, prediction_layer={self.prediction_layer!r}, annotation_layer={self.annotation_layer!r})"
        )


class ObjectOrganizer(BaseModel):
    """
    This module provides pydantic-based classes for organizing and managing tracking entries in OCTRON. 
    It is designed to help maintain a dictionary of object entries (named Obj, see above), 
    defined each by a unique ID,
    that is also the one that SAM2 internally deals with, 
    and representing attributes such as label, a unique label ID, suffix, color. 
    CAVE: label ID and object ID are not the same! Label IDs are 
    unique for each label and are used to assign colors. The object ID is unique for each object and 
    are used by the SAM2 tracking module.
    It also assigns a unique mask  (=prediction) layer and annotation layer to each object.
    This way, all information for objects flowing through OCTRON is saved in one place. 
    
    
    I am using a color picking strategy that is based on the number of labels and suffixes. 
    A colormap is created (see self.all_colors) and then maximally different colors are picked 
    for each label and suffix combination.
    Labels are assigend "slices" of a colormap and then each suffix is assigned a color from that slice.
    You CAN assign a color yourself, but I would recommend using the internal strategy.
    
    
    Example usage:
    >>> object_organizer = ObjectOrganizer()
    >>> object_organizer.add_entry(0, Obj(label='worm', suffix='0'))
    >>> object_organizer.add_entry(1, Obj(label='worm', suffix='1')) # Color is picked automatically 
    
    
    """
    entries: Dict[int, Obj] = {}
    # Mapping from label to its assigned label_id.
    label_id_map: Dict[str, int] = {}
    # The next available label_id.
    next_label_id: int = 0
    
    # Color dictionary for all labels
    n_labels_max: int = 10 # MAXIMUM ALLOWED NUMBER OF LABELS 
    n_subcolors: int  = 50


    def all_colors(self) -> tuple[list, list, list]:
        """
        Create a list of colors for all labels. This is currently capped at 10, 
        which I think is reasonable.
        Returns a list of lists:
            -> label
                -> colors for each label
        """
        label_colors = create_label_colors(cmap='cmr.tropical', 
                                           n_labels=self.n_labels_max, 
                                           n_colors_submap=self.n_subcolors,
                                           )
        # Create maximally different colors for each label and for each submap 
        indices_max_diff_labels = sample_maximally_different(list(range(self.n_labels_max)))
        indices_max_diff_subcolors = sample_maximally_different(list(range(self.n_subcolors)))
        return (label_colors, indices_max_diff_labels, indices_max_diff_subcolors)

    def exists_id(self, id_: int) -> bool:
        return id_ in self.entries
    
    def max_id(self) -> int:
        try:
            return max([e for e in self.entries])
        except ValueError:
            return 0
        
    def min_available_id(self) -> int:
        """
        Find the minimum integer >= 0 that is not yet present as an ID in the object organizer.
        """
        existing_ids = set(self.entries.keys())
        min_id = 0
        while min_id in existing_ids:
            min_id += 1
        return min_id
    
    def exists_label(self, label: str) -> bool:
        return any(entry.label == label for entry in self.entries.values())

    def exists_suffix(self, suffix: str) -> bool:
        return any(entry.suffix == suffix for entry in self.entries.values())

    def exists_combination(self, label: str, suffix: str) -> bool:
        """
        Check if the combination of label and suffix is already present.
        """
        return any(entry.label == label and entry.suffix == suffix for entry in self.entries.values())

    def get_current_labels(self) -> List[str]:
        """
        Return a list of unique labels from all entries.
        """
        return list({entry.label for entry in self.entries.values()})
    
    def get_entries_by_label(self, label: str) -> List[Obj]:
        """
        Return a list of entries (Obj) that have the given label.
        """
        return [entry for entry in self.entries.values() if entry.label == label]

    def get_suffixes_by_label(self, label: str) -> List[str]:
        """
        For a given label, return a list of all suffixes from entries matching that label.
        """
        return [entry.suffix for entry in self.entries.values() if entry.label == label]

    def get_entry_by_label_suffix(self, label: str, suffix: str) -> Optional[Obj]:
        """
        Return the entry that matches the given label and suffix combination.
        """
        for entry in self.entries.values():
            if entry.label == label and entry.suffix == suffix:
                return entry
        return None

    def add_entry(self, id_: int, entry: Obj) -> bool:
        if id_ in self.entries:
            raise ValueError(f"ID {id_} already exists.")
        if self.exists_combination(entry.label, entry.suffix):
            if entry.annotation_layer is not None:
                print(f"Combination ({entry.label}, {entry.suffix}) already exists.")
                return False
        # Check if label already exists and assign label_id accordingly.
        if entry.label in self.label_id_map:
            entry.label_id = self.label_id_map[entry.label]
        else:
            entry.label_id = self.next_label_id
            self.label_id_map[entry.label] = self.next_label_id
            self.next_label_id += 1

        # Find out which color to assign (if necessary)
        if entry.color is None:
            n_subcolors = len(self.get_suffixes_by_label(entry.label)) # These colors must exist
            label_colors, indices_max_diff_labels, indices_max_diff_subcolors = self.all_colors()
            if entry.label_id >= len(indices_max_diff_labels):
                raise ValueError(f"Label_id {entry.label_id} exceeds the maximum number of labels.")
            this_color = label_colors[indices_max_diff_labels[entry.label_id]][indices_max_diff_subcolors[n_subcolors]]
            entry.color = this_color
            
        self.entries[id_] = entry
        return True
    
    def update_entry(self, id_: int, entry: Obj) -> None:
        if id_ not in self.entries:
            raise ValueError(f"ID {id_} does not exist.")
        for eid, existing in self.entries.items():
            if eid != id_ and existing.label == entry.label and existing.suffix == entry.suffix:
                raise ValueError(f"Combination ({entry.label}, {entry.suffix}) already exists in ID {eid}.")
        self.entries[id_] = entry

    def get_entry_id(self, organizer_entry: Obj) -> Optional[int]:
        """
        Return the ID of the given organizer entry.
        """
        for id_, entry in self.entries.items():
            if entry == organizer_entry:
                return id_
        return None

    def get_entry(self, id_: int) -> Obj:
        """
        Return the entry for the given id.
        """
        if id_ not in self.entries:
            raise ValueError(f"ID {id_} does not exist.")
        return self.entries[id_]

    def remove_entry(self, id_: int) -> Obj:
        """
        Remove the entry for the given id and return it.
        Raises KeyError if id is not found.
        """
        if id_ not in self.entries:
            raise KeyError(f"ID {id_} does not exist.")
        return self.entries.pop(id_)
    
    def save_to_disk(self, file_path: Union[str, Path]):
        """
        Save the object organizer to disk as JSON
        """
        file_path = Path(file_path)
        # Create a copy of the data without non-serializable objects
        serializable_data = {
            "entries": {}
        }
        for obj_id, obj in self.entries.items():
            # Create a serializable version without the layers
            serializable_obj = obj.model_dump(exclude={"annotation_layer", "prediction_layer"})
            # Then add metadata info back in 
            # Add metadata about the annotation layer
            if obj.annotation_layer is not None:
                serializable_obj["annotation_layer_metadata"] = {
                    "name": obj.annotation_layer.name,
                    "type": obj.annotation_layer._basename(),  # 'Shapes' or 'Points'
                    "data_shape": [list(d) if isinstance(d, tuple) else d for d in obj.annotation_layer.data.shape] 
                                if hasattr(obj.annotation_layer.data, 'shape') else None,
                    "ndim": obj.annotation_layer.ndim,
                    "visible": obj.annotation_layer.visible,
                    "opacity": obj.annotation_layer.opacity,
                }
            # Add metadata about the prediction (mask, ... ) layer
            if obj.prediction_layer is not None:
                serializable_obj["prediction_layer_metadata"] = {
                    "name": obj.prediction_layer.name,
                    "type": obj.prediction_layer._basename(),  # 'Labels'
                    "data_shape": list(obj.prediction_layer.data.shape) 
                                if hasattr(obj.prediction_layer.data, 'shape') else None,
                    "ndim": obj.prediction_layer.ndim,
                    "visible": obj.prediction_layer.visible,
                    "opacity": obj.prediction_layer.opacity,
                }
                
                # Handle colormap serialization based on its type
                colormap = obj.prediction_layer.colormap
                if colormap is not None:
                    # For DirectLabelColormap (from napari utils), extract the colors
                    if hasattr(colormap, "name"):
                        serializable_obj["prediction_layer_metadata"]["colormap_name"] = colormap.name
                    # Excluding saving the color dictionary for now since it is a duplicate of the existing color
                    # if hasattr(colormap, "color_dict"):   
                    #     color_dict = {}
                    #     for k, color in colormap.color_dict.items():
                    #         color_dict[k] = color.tolist()
                    #     serializable_obj["prediction_layer_metadata"]["colormap_colors"] = color_dict
                      
                # Add zarr file path if it exists in metadata
                if hasattr(obj.prediction_layer, 'metadata') and '_zarr' in obj.prediction_layer.metadata:
                    serializable_obj["prediction_layer_metadata"]["zarr_path"] = obj.prediction_layer.metadata['_zarr'].as_posix()
            
            serializable_data["entries"][str(obj_id)] = serializable_obj  # Convert key to string for JSON compatibility
            
        # Write to file
        if file_path.exists():
            print(f"⚠️ Overwriting existing file at {file_path.as_posix()}")
            file_path.unlink()
        with open(file_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        print(f"💾 Octron object organizer saved to {file_path.as_posix()}")
        
    # TODO: Implement load_from_disk
    #       -> This needs more careful handling of the annotation / ... layers
    # @classmethod
    # def load_from_disk(cls, file_path: Union[str, Path]) -> "ObjectOrganizer":
    #     """
    #     Load object organizer from disk
    #     """
    #     file_path = Path(file_path)
    #     if not file_path.exists():
    #         print(f"No organizer file found at {file_path}")
    #         return cls()
        
    #     try:
    #         with open(file_path, 'r') as f:
    #             data = json.load(f)
    #         organizer = cls()
    #         # Reconstruct objects from serialized data
    #         for obj_id, obj_data in data.get("entries", {}).items():
    #             # Convert lists back to tuples for color
    #             if "color" in obj_data and isinstance(obj_data["color"], list):
    #                 obj_data["color"] = tuple(obj_data["color"])
                
    #             # Create Obj instance
    #             obj = Obj(**obj_data)
    #             organizer.entries[obj_id] = obj
                
    #         print(f"Object organizer loaded from {file_path}")
    #         return organizer
            
    #     except Exception as e:
    #         print(f"Error loading object organizer: {e}")
    #         return cls()    



    def __repr__(self) -> str:
        output_lines = ["OCTRON ObjectOrganizer with entries:"]
        for id_, entry in self.entries.items():
            output_lines.append(
                f"  ID {id_}: label='{entry.label}', suffix='{entry.suffix}', label_id='{entry.label_id}', color='{entry.color}'"
            )
        return "\n".join(output_lines)
 
 
 
 
def load_object_organizer(file_path):
    """
    Load object organizer .json from disk and return
    its content as dictionary.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the .json file.
    
    Returns
    -------
    dict
        Dictionary containing all object organizer data.
    
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"No organizer file found at {file_path}")
        return
    if not file_path.suffix == '.json': 
        print(f"❌ File is not a json file: {file_path}")
        return
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"📖 Octron object organizer loaded from {file_path.as_posix()}")
        return data
    except Exception as e:
        print(f"❌ Error loading json: {e}")
        return 