from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List, Any

from octron.sam2_octron.helpers.sam2_colors import (
    create_label_colors,
)


class Obj(BaseModel):
    label: str
    suffix: str
    label_id: Optional[int] = None
    color: Optional[list] = None
    mask_layer: Optional[Any] = None
    annotation_layer: Optional[Any] = None
    predicted_frames: List[Any] = Field(default_factory=list)
    
    class Config:
        validate_assignment = True # To make sure that internally generated values are validated
        
    @field_validator("color")
    def check_color_length(cls, v):
        if v is not None and len(v) != 4:
            raise ValueError("Color must be a list of length 4")
        return v
    
    def add_predicted_frame(self, frame_index: int) -> None:
        '''
        Add a frame index to predicted_frames if it is not already present.
        '''
        if frame_index not in self.predicted_frames:
            self.predicted_frames.append(frame_index)

    def add_predicted_frames(self, frame_indices: list) -> None:
        for index in frame_indices:
            self.add_predicted_frame(index)
    
    def remove_predicted_frame(self, frame_index: int) -> None:
        '''
        Remove a frame index from predicted_frames - only if it exists.
        '''
        if frame_index in self.predicted_frames:
            self.predicted_frames.remove(frame_index)
            
    def remove_predicted_frames(self, frame_indices: list) -> None:
        for index in frame_indices:
            self.remove_predicted_frame(index)

    def reset_predicted_frames(self) -> None:
        '''
        Reset the predicted_frames list.
        '''
        self.predicted_frames = []
            
    def __repr__(self) -> str:
        return (
            f"Obj(label={self.label!r}, suffix={self.suffix!r}, label_id={self.label_id!r}, "
            f"color={self.color!r}, mask_layer={self.mask_layer!r}, annotation_layer={self.annotation_layer!r})"
        )


class ObjectOrganizer(BaseModel):
    '''
    This module provides pydantic-based classes for organizing and managing tracking entries in OCTRON. 
    It is designed to help maintain a dictionary of object entries (named Obj, see above), 
    defined each by a unique ID,
    that is also the one that SAM2 internally deals with, 
    and representing attributes such as label, a unique label ID, suffix, color. 
    It also assigns a unique mask layer and annotation layer to each object.
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
    
    
    '''
    entries: Dict[int, Obj] = {}
    # Mapping from label to its assigned label_id.
    label_id_map: Dict[str, int] = {}
    # The next available label_id.
    next_label_id: int = 0

    def all_colors(self) -> tuple[list, list, list]:
        '''
        Create a list of colors for all labels. This is currently capped at 10, 
        which I think is reasonable.
        Returns a list of lists:
            -> label
                -> colors for each label
        '''
        n_labels_max = 10 # MAXIMUM ALLOWED NUMBER OF LABELS 
        n_subcolors =  50
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
    
    def max_id(self) -> int:
        try:
            return max([e for e in self.entries])
        except ValueError:
            return 0

    def exists_label(self, label: str) -> bool:
        return any(entry.label == label for entry in self.entries.values())

    def exists_suffix(self, suffix: str) -> bool:
        return any(entry.suffix == suffix for entry in self.entries.values())

    def exists_combination(self, label: str, suffix: str) -> bool:
        '''
        Check if the combination of label and suffix is already present.
        '''
        return any(entry.label == label and entry.suffix == suffix for entry in self.entries.values())

    def get_current_labels(self) -> List[str]:
        '''
        Return a list of unique labels from all entries.
        '''
        return list({entry.label for entry in self.entries.values()})
    
    def get_suffixes_by_label(self, label: str) -> List[str]:
        '''
        For a given label, return a list of all suffixes from entries matching that label.
        '''
        return [entry.suffix for entry in self.entries.values() if entry.label == label]

    def add_entry(self, id_: int, entry: Obj) -> None:
        if id_ in self.entries:
            raise ValueError(f"ID {id_} already exists.")
        if self.exists_combination(entry.label, entry.suffix):
            raise ValueError(f"Combination ({entry.label}, {entry.suffix}) already exists.")
        
        # Check if label already exists and assign label_id accordingly.
        if entry.label in self.label_id_map:
            entry.label_id = self.label_id_map[entry.label]
            #print(f"Label {entry.label} already exists. Reusing label_id {entry.label_id}.")
        else:
            entry.label_id = self.next_label_id
            self.label_id_map[entry.label] = self.next_label_id
            self.next_label_id += 1
            #print(f"New label {entry.label} found. Assigning label_id {entry.label_id}.")
        
        # Find out which color to assign (if necessary)
        if entry.color is None:
            n_subcolors = len(self.get_suffixes_by_label(entry.label)) # These colors must exist
            label_colors, indices_max_diff_labels, indices_max_diff_subcolors = self.all_colors()
            if entry.label_id >= len(indices_max_diff_labels):
                raise ValueError(f"Label_id {entry.label_id} exceeds the maximum number of labels.")
            #print(f"Assigning color. Label ID: {entry.label_id} | Subcolor ID: {n_subcolors}")
            this_color = label_colors[indices_max_diff_labels[entry.label_id]][indices_max_diff_subcolors[n_subcolors]]
            #print(f"Color assigned: {this_color}")
            entry.color = this_color
            
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
        '''
        Remove the entry for the given id and return it.
        Raises KeyError if id is not found.
        '''
        if id_ not in self.entries:
            raise KeyError(f"ID {id_} does not exist.")
        return self.entries.pop(id_)

    def __repr__(self) -> str:
        output_lines = ["OCTRON ObjectOrganizer with entries:"]
        for id_, entry in self.entries.items():
            output_lines.append(
                f"  ID {id_}: label='{entry.label}', suffix='{entry.suffix}', label_id='{entry.label_id}', color='{entry.color}'"
            )
        return "\n".join(output_lines)
 
 
################################################################################################################################   
# Helper functions for the ObjectOrganizer class
def sample_maximally_different(seq):
    '''
    Given an ascending list of numbers, return a new ordering
    where each subsequent number is chosen such that its minimum
    absolute difference to all previously picked numbers is maximized.

    I added this to choose colors that are maximally different from each other,
    both for labels as well as for sub-label (same label, different suffix).

    Example:
        Input:  [1, 2, 3, 4, 5]
        Possible Output: [1, 5, 2, 4, 3]
    '''
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