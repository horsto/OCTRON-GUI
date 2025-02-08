from pydantic import BaseModel, field_validator
from typing import Optional, Dict, List, Any

from octron.sam2_octron.helpers.sam2_colors import (
    create_label_colors,
)


class Obj(BaseModel):
    label: str
    suffix: str
    color: Optional[list] = None
    mask_layer: Optional[Any] = None
    annotation_layer: Optional[Any] = None
    
    @field_validator("color")
    def check_color_length(cls, v):
        if v is not None and len(v) != 4:
            raise ValueError("Color must be a list of length 4")
        return v
    
    def __repr__(self) -> str:
        return (f"Obj(label={self.label!r}, suffix={self.suffix!r}, "
                f"color={self.color!r}, mask_layer={self.mask_layer!r}, "
                f"annotation_layer={self.annotation_layer!r})")


class ObjectOrganizer(BaseModel):
    '''
    This module provides pydantic-based classes for organizing and managing tracking entries in OCTRON. 
    It is designed to help you maintain a dictionary of object entries (named Obj), defined each by a unique ID,
    that is also the one that SAM2 internally deals with, 
    and representing attributes such as label, suffix, color, and optional layers. 


    
    '''
    entries: Dict[int, Obj] = {}

    def all_colors(self) -> tuple[list, list, list]:
        '''
        Create a list of colors for all labels. This is currently capped at 10, 
        which I think is reasonable.
        Returns a list of lists:
            -> label
                -> colors for each label
        '''
        n_labels_max = 10 # MAXIMUM NUMBER OF LABELS 
        n_subcolors = 250
        label_colors = create_label_colors(cmap='cmr.tropical', 
                                           n_labels=n_labels_max, 
                                           n_colors_submap=n_subcolors,
                                           ) 
        # Create maximally different colors for each label and for each submap 
        indices_max_diff_labels = sample_maximally_different(list(range(n_labels_max)))
        indices_max_diff_subcolors = sample_maximally_different(list(range(n_subcolors)))
        return (label_colors, indices_max_diff_labels, indices_max_diff_subcolors)


    def exists_id(self, id_: int) -> bool:
        return id_ in self.entries

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
    
    def get_suffixes_by_label(self, label: str) -> List[str]:
        """
        For a given label, return a list of all suffixes from entries matching that label.
        """
        return [entry.suffix for entry in self.entries.values() if entry.label == label]

    def add_entry(self, id_: int, entry: Obj) -> None:
        if id_ in self.entries:
            raise ValueError(f"ID {id_} already exists.")
        if self.exists_combination(entry.label, entry.suffix):
            raise ValueError(f"Combination ({entry.label}, {entry.suffix}) already exists.")
        
        # Check if label already exists
        if self.exists_label(entry.label):
            print(f"Label {entry.label} already exists. ")

        
        
        # Find out which color to assign 
        if entry.color is None:
            print('No color, finding one ... ')
            
        
        
        
        self.entries[id_] = entry

    def update_entry(self, id_: int, entry: Obj) -> None:
        if id_ not in self.entries:
            raise ValueError(f"ID {id_} does not exist.")
        for eid, existing in self.entries.items():
            if eid != id_ and existing.label == entry.label and existing.suffix == entry.suffix:
                raise ValueError(f"Combination ({entry.label}, {entry.suffix}) already exists in ID {eid}.")
        self.entries[id_] = entry

    def get_entry(self, id_: int) -> Obj:
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

    def __repr__(self) -> str:
        output_lines = ["ObjectOrganizer with entries:"]
        for id_, entry in self.entries.items():
            output_lines.append(
                f"  ID {id_}: label='{entry.label}', suffix='{entry.suffix}', color='{entry.color}'"
            )
        return "\n".join(output_lines)
    
    
### Helper functions for the ObjectOrganizer class
def sample_maximally_different(seq):
    """
    Given an ascending list of numbers, return a new ordering
    where each subsequent number is chosen such that its minimum
    absolute difference to all previously picked numbers is maximized.

    Example:
        Input:  [1, 2, 3, 4, 5]
        Possible Output: [1, 5, 2, 4, 3]
    """
    if not seq:
        return []
    # Start with the first element.
    sample = [seq[0]]
    remaining = list(seq[1:])
    while remaining:
        # For each candidate, compute the minimum distance to any element in sample,
        # then select the candidate with the maximum such distance.
        candidate = max(remaining, key=lambda x: min(abs(x - s) for s in sample))
        sample.append(candidate)
        remaining.remove(candidate)
    return sample