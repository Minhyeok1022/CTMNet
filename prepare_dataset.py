"""
# Function: Read files and preprocess them
# Steps:
# 1. Import data from the gdf file provided before the competition, 
#    remove unwanted channels, and select required events.
# 2. Select desired time segments for slicing; treat each segment (4s) as one sample.
# 3. Import labels from the mat file provided after the competition, 
#    ensuring they correspond with epochs and their numbers match.
# 4. Save the resulting data in a new mat file, 
#    preparing it for use in the subsequent main.py.
"""

import mne
import numpy as np
import scipy.signal as signal
from scipy.io import savemat
import scipy.io as sio
import numpy as np

def changeGdf2Mat(dir_path, mode="train"):
    '''
    read data from GDF files and store as mat files

    Parameters
    ----------
    dir_path : str
        GDF file dir path.
    mode : str, optional
        change train dataset or eval dataset. The default is "train".

    Returns
    -------
    None.

    '''
    mode_str = ''
    if mode=="train":
        mode_str = 'T'
    else:
        mode_str = 'E'
    for nSub in range(1, 10):
        # Load the gdf file
        data_filename = dir_path+'BCICIV_2a_gdf/A0{}{}.gdf'.format(nSub, mode_str)
        raw = mne.io.read_raw_gdf(data_filename)  
    
        # Select the events of interest
        events, event_dict = mne.events_from_annotations(raw) 

        e_dict={'276': 1, '277': 2, '768': 3, '769': 4, '770': 5, '771': 6, '772': 7, '783': 8, '1023': 9, '1072': 10, '32766': 11}

        for i in range(len(events)):
            events[i][2] = e_dict[[k for k,v in event_dict.items() if v == events[i][2]][0]]
        if mode=="train":
            # train dataset are labeled
            event_id = {'Left': e_dict['769'],
                        'Right': e_dict['770'], 
                        'Foot': e_dict['771'],
                        'Tongue': e_dict['772']}  
        else:
            # evaluate dataset are labeled as 'Unknnow'
            event_id = {'Unknown': e_dict['783']}
            
        # Select the events corresponding to the four categories we are interested in. Here, events[:, 2] refers to the third column of the events array, which represents the event IDs.
        selected_events = events[np.isin(events[:, 2], list(event_id.values()))]  
        
        # remove EOG channels
        raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')
        # Epoch the data
        # using 4s (1000 sample point ) segmentation
        epochs = mne.Epochs(raw, selected_events, event_id, picks=picks,tmin=0, tmax=3.996, preload=True, baseline=None)
        
        filtered_data = epochs.get_data()
        label_filename = dir_path + 'true_labels/'+'A0{}{}.mat'.format(nSub, mode_str)
        mat = sio.loadmat(label_filename)  # load target mat file
        labels = mat['classlabel']
       
        # Save the data and labels to a .mat file
        result_filename = 'mymat_raw/A0{}{}.mat'.format(nSub, mode_str)
        savemat(result_filename, {'data': filtered_data, 'label': labels})



def changeGdf2Mat_2b(dir_path, mode="train"):

    mode_str = ''
    if mode=="train":
        mode_str = 'T'
        sessions = [1,2,3]
    else:
        mode_str = 'E'
        sessions = [4,5]

    for nSub in range(1, 10):
        all_data = []
        all_events = []
        cumulative_samples = 0
        # Load the gdf file
        for session in sessions:
            data_filename = data_filename = dir_path + f'BCICIV_2b_gdf/B0{nSub}0{session}{mode_str}.gdf'
            raw_session = mne.io.read_raw_gdf(data_filename)
            events_session, event_dict = mne.events_from_annotations(raw_session)

            e_dict={'1023': 12, '1077': 13, '1078': 3, '1079': 4, '1081': 5, '276': 6, '277': 7, '32766': 8, '768': 9, '781': 10, '783': 11, '769':1, '770':2}

            for i in range(len(events_session)):
                events_session[i][2] = e_dict[[k for k,v in event_dict.items() if v == events_session[i][2]][0]]
            events_session[:, 0] += cumulative_samples
            cumulative_samples += len(raw_session.times)
            all_data.append(raw_session)
            all_events.append(events_session)
        
        raw = mne.concatenate_raws(all_data)
        events = np.concatenate(all_events)
        # sort_idx = np.argsort(events[:, 0])
        # events = events[sort_idx]
        
        if mode=="train":
            # train dataset are labeled
            event_id = {'Left': e_dict['769'],
                        'Right': e_dict['770']}  
        else:
            # evaluate dataset are labeled as 'Unknown'
            event_id = {'Unknown': e_dict['783']}
            
        # Select the events corresponding to the four categories we are interested in. Here, events[:, 2] refers to the third column of the events array, which represents the event IDs.
        selected_events = events[np.isin(events[:, 2], list(event_id.values()))]  
        
        # remove EOG channels
        raw.info['bads'] += ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')
        # Epoch the data
        # using 4s (1000 sample point ) segmentation
        epochs = mne.Epochs(raw, selected_events, event_id, picks=picks,tmin=0, tmax=3.996, preload=True, baseline=None)
        
        filtered_data = epochs.get_data()
        all_labels = []
        for session in sessions:
            label_filename = dir_path + f'true_labels/B0{nSub}0{session}{mode_str}.mat'
            mat = sio.loadmat(label_filename)  # load target mat file
            all_labels.append(mat['classlabel'])
        labels = np.concatenate(all_labels)
       
        # Save the data and labels to a .mat file
        result_filename = 'D:/Cuda_Test_model/CTNET/mymat_raw/B0{}{}.mat'.format(nSub, mode_str)
        savemat(result_filename, {'data': filtered_data, 'label': labels})



dir_path = 'set_your_path'
# prepare train dataset

# changeGdf2Mat(dir_path, 'train')
# # prepare test dataset
# changeGdf2Mat(dir_path, 'eval')

changeGdf2Mat_2b(dir_path, 'train')
# prepare test dataset
changeGdf2Mat_2b(dir_path, 'eval')
