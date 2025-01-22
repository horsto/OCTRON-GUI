import pathlib
import numpy as np
from pims import PyAVReaderIndexed


def get_video_toc(video_file,
                  verbose=True,
                 ):
    '''
    Build a table of contents (toc) dictionary that can be processed by 
    pims.PyAVReaderIndexed()
    https://github.com/soft-matter/pims/blob/85b1adcab4e1d2881a9915ac29da6b1556bb904f/pims/pyav_reader.py#L274
    This relies on installation of PyAV 
    https://pyav.org/docs/develop/overview/installation.html
        
    The toc enables random access to video frames. The issue is that "simpler" readers 
    like the opencv default reader seem to have issues to accurately load specific frames 
    based on index from a video file. 
    See also https://scikit-image.org/docs/stable/user_guide/video.html#adding-random-access-to-pyav
    
    I noticed skipped / duplicate frames and inaccurate loading based on indices. 
    PIMS (https://soft-matter.github.io/pims/v0.6.1/)solves this by first decoding the 
    whole video and creating an index. 
    It does this in chunks, and does not load the whole video into memory. 
    This is an extremely slow process, but I found that this is the only solution that yields a precise 
    table of content of all video frames. I am saving the toc so that next time the video is accessed, 
    it is fast. 
    I tried to engage people into discussions about speed and threading here: 
    https://github.com/soft-matter/pims/issues/442    
    
    Parameter
    ---------
    video_file : str or pathlib.Path. Path to video file
    verbose : boolean
    
    Returns
    -------
    toclog : dict : 
        toc: dict : pims.PyAVReaderIndexed.toc (can be loader by pyav to skip toc buliding next time)
        toc_no_skipped : int : number of elements in toc['lengths'] that are != 1, indicating skipped (?) frames
        no_frames : int : Total number of frames in toc. This is to compare with ffmpeg probe (sanity check)
    '''
    if isinstance(video_file, str):
        video_file = pathlib.Path(video_file)
    assert video_file.exists(), f'Could not find video file "{video_file}"'
    
    video_pims_indexed = PyAVReaderIndexed(video_file)

    toc = video_pims_indexed.toc
    toc_lengths = np.array(toc['lengths'])
    toc_no_skipped = len(toc_lengths[toc_lengths!=1])
    no_frames = video_pims_indexed._len
    
    if verbose:
        print(f'TOC built. Found {no_frames} frames, of which {toc_no_skipped} are skipped')
        
    toclog = {
        'toc' : toc,
        'toc_no_skipped' : toc_no_skipped,
        'no_frames' : no_frames
    }
    
    return toclog