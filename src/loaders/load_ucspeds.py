"""
Module that loads the raw data from UCSPeds datasets and saves it in files to fed into the models.
This module also loads data from the previous files to be fed into the model.
"""

import os
import settings
import utils
import cv2
import joblib
import scipy.io as sio
import numpy as np
import utils.transformations

FRAMES_SPACING = 5
SIGMAX = 8
SIGMAY = 15

def parse_folder_name(folder_name):
    folder_name = folder_name[4:-2]
    splits = folder_name.split('_')
    if len(splits) != 3:
        raise ValueError(folder_name + " is incorrect!")
    vid_number = int(splits[0])
    if splits[1] != '33':
        raise ValueError(folder_name + " does not have the value 33!")
    clip_number = int(splits[2])

    return int(vid_number), int(clip_number)

def parse_image_name(image_name):
    image_name = image_name[4:-4]
    splits = image_name.split('_')
    if len(splits) != 4:
        raise ValueError(image_name + " is incorrect!")
    vid_number = int(splits[0])
    if splits[1] != '33':
        raise ValueError(image_name + " does not have the value 33!")
    clip_number = int(splits[2])
    if splits[3][0] != 'f':
        raise ValueError(image_name + " frame does not have character f!")
    frame_number = int(splits[3][1:])

    return int(vid_number), int(clip_number), int(frame_number)

def parse_gt_file_name(gt_file_name):
    gt_file_name = gt_file_name[4:-15]
    splits = gt_file_name.split('_')
    if len(splits) != 3:
        raise ValueError(gt_file_name + " is incorrect!")
    vid_number = int(splits[0])
    if splits[1] != '33':
        raise ValueError(gt_file_name + " does not have the value 33!")
    clip_number = int(splits[2])

    return int(vid_number), int(clip_number)

class VideoDataUCS:
    def __init__(self):
        self.frames = {}
        self.mask = None
    
    def compute_mask_counts(self):
         for frame_id in self.frames:
            if self.frames[frame_id].frame is not None and self.frames[frame_id].density is not None:
                self.frames[frame_id].compute_mask_count(self.mask)

class FrameDataUCS:

    def __init__(self):
        self.frame = None
        self.density = None
        self.count = None
    
    def compute_mask_count(self, mask):
        self.count_mask = np.sum(self.density*mask)


def compute_mask(domain_path, data_domain):
    subfiles = [f for f in os.listdir(domain_path)]

    for file in subfiles:
        if file[-12:] == '1_33_roi.mat' or file[-24:] == '1_33_roi_mainwalkway.mat':
            file_path = os.path.join(domain_path, file)
            mask = sio.loadmat(file_path)
            data_domain[1].mask = np.array([mask['roi'][0][0][2]], dtype=np.float)

            if settings.LOAD_DATA_AUGMENTATION:
                mask_s90 = utils.transformations.transform_matrix_channels(data_domain[1].mask, utils.transformations.symmetric, 90)
                data_domain[1].augmentation = {'s90': mask_s90}

def compute_densities(domain_path, data_domain):
    subfiles = [f for f in os.listdir(domain_path)]

    for file in subfiles:
            if file[-15:] == '_frame_full.mat':
                vid_number, clip_number = parse_gt_file_name(file)
                file_path = os.path.join(domain_path, file)
                data_matlab = sio.loadmat(file_path)
                
                for frame_number in range(len(data_matlab['fgt'][0][0][0][0])):
                    if (frame_number-1) % FRAMES_SPACING == 0:
                        centers = []
                        sigmas = []
                        real_frame_number = (clip_number*200+frame_number-1)/FRAMES_SPACING
                        for person in data_matlab['fgt'][0][0][0][0][frame_number][0][0][0]:
                            centers.append([person[0], person[1]])
                            sigmas.append([SIGMAX, SIGMAY])
                            shape = data_domain[vid_number].frames[real_frame_number].frame.shape
                        
                        data_domain[vid_number].frames[real_frame_number].density = utils.density_map((shape[1], shape[2]), centers, sigmas, mask=None).reshape((1, shape[1], shape[2]))
                        data_domain[vid_number].frames[real_frame_number].count = len(data_matlab['fgt'][0][0][0][0][frame_number][0][0][0])

                        if settings.LOAD_DATA_AUGMENTATION:
                            density_s90 = utils.transformations.transform_matrix_channels(data_domain[vid_number].frames[real_frame_number].density, utils.transformations.symmetric, 90)
                            data_domain[vid_number].frames[real_frame_number].density_augmentation = {'s90': density_s90}

    for vid_number in data_domain.keys():
        keys = list(data_domain[vid_number].frames.keys())
        for frame_number in keys:
            if data_domain[vid_number].frames[frame_number].density is None:
                data_domain[vid_number].frames.pop(frame_number)                

def save_data_multiple_files_domain(data_domain, domain_id, prefix_frames, prefix_densities):
    preprocessed_directory = os.path.join(settings.DATASET_DIRECTORY, 'Preprocessed')
    ucspeds_directory = os.path.join(preprocessed_directory, settings.UCSPEDS_PREPROCESSED_DIRECTORY)
    multiple_files_directory = os.path.join(ucspeds_directory, 'Multiple_Files')
    frames_directory = os.path.join(multiple_files_directory, 'Frames')
    densities_directory = os.path.join(multiple_files_directory, 'Densities')
    str_id = str(domain_id)
    domain_directory_frame = os.path.join(frames_directory, str_id)
    domain_directory_density = os.path.join(densities_directory, str_id)

    if not os.path.exists(domain_directory_frame):
        os.makedirs(domain_directory_frame)
    if not os.path.exists(domain_directory_density):
        os.makedirs(domain_directory_density)
    
    for video_number in data_domain.keys():
        str_id = str(video_number)
        video_directory_frame = os.path.join(domain_directory_frame, str_id)
        video_directory_density = os.path.join(domain_directory_density, str_id)

        if not os.path.exists(video_directory_frame):
            os.makedirs(video_directory_frame)
        if not os.path.exists(video_directory_density):
            os.makedirs(video_directory_density)
            
        for frame_id in data_domain[video_number].frames:
            frame = data_domain[video_number].frames[frame_id].frame 
            density = data_domain[video_number].frames[frame_id].density
            if frame is not None and density is not None:
                str_id = str(frame_id)

                frame_path = os.path.join(video_directory_frame, prefix_frames+'_'+str(str_id)+'.npy')
                joblib.dump(frame, frame_path)
                density_path = os.path.join(video_directory_density, prefix_densities+'_'+str(str_id)+'.npy')
                joblib.dump(density, density_path)

                data_domain[video_number].frames[frame_id].frame = 0
                data_domain[video_number].frames[frame_id].density = 0

                if settings.LOAD_DATA_AUGMENTATION:
                    frames_aug = data_domain[video_number].frames[frame_id].augmentation
                    densities_aug = data_domain[video_number].frames[frame_id].density_augmentation
                    for frame_aug_key in frames_aug:
                        frame_aug = frames_aug[frame_aug_key]
                        density_aug = densities_aug[frame_aug_key]
                        frame_path = os.path.join(video_directory_frame, prefix_frames+'_'+str(str_id)+'_'+frame_aug_key+'.npy')
                        joblib.dump(frame_aug, frame_path)
                        density_path = os.path.join(video_directory_density, prefix_densities+'_'+str(str_id)+'_'+frame_aug_key+'.npy')
                        joblib.dump(density_aug, density_path)
                        frames_aug[frame_aug_key] = 0
                        densities_aug[frame_aug_key] = 0 

def load_data(save_changes=True):
    uscPedsDir = os.path.join(settings.DATASET_DIRECTORY, 'UCSPeds')
    data = {}
    subdirs = [d for d in os.listdir(uscPedsDir)]

    for subdir in subdirs:
        subdir_path = os.path.join(uscPedsDir, subdir)
        
        if os.path.isdir(subdir_path) and subdir[:3]=='vid':
            domain_id = subdir
            
            data[domain_id] = {}
            subsubdirs = [d for d in os.listdir(subdir_path)]
            for subsubdir in subsubdirs:
                
                subsubdir_path = os.path.join(subdir_path, subsubdir)
                if os.path.isdir(subsubdir_path) and len(subsubdir) >= 5 and subsubdir[:4] == domain_id and subsubdir[4] == '1'  and subsubdir[-2:] == '.y':
                    vid_number, clip_number = parse_folder_name(subsubdir)
                    if vid_number not in data[domain_id].keys():
                        data[domain_id][vid_number] = VideoDataUCS()
                    
                    video_data = data[domain_id][vid_number]

                    images = [img for img in os.listdir(subsubdir_path)]
                    for image in images:
                        if image[:4] == domain_id and image[-4:] == '.png':
                            vid_number_image, clip_number_image, frame_number = parse_image_name(image)

                            if vid_number_image != vid_number or clip_number_image != clip_number:
                                raise ValueError(image + ' is in the wrong folder!')
                            
                            if (frame_number-1) % FRAMES_SPACING == 0:
                                image_path = os.path.join(subsubdir_path, image)
                                frame_data = FrameDataUCS()
                                frame_data.frame = np.moveaxis(cv2.imread(image_path), 2, 0)
                                if settings.LOAD_DATA_AUGMENTATION:
                                    frame_s90 = utils.transformations.transform_matrix_channels(frame_data.frame, utils.transformations.symmetric, 90)
                                    frame_data.augmentation = { 's90': frame_s90}
                                real_frame_number = int((clip_number*200+frame_number-1)/FRAMES_SPACING)
                                video_data.frames[real_frame_number] = frame_data
                
            compute_densities(subdir_path, data[domain_id])
            compute_mask(subdir_path, data[domain_id])
            data[domain_id][1].compute_mask_counts()

        if save_changes:
            save_data_multiple_files_domain( data[domain_id], domain_id, 'first', 'first')           
    
    if save_changes:
        preprocessed_directory = os.path.join(settings.DATASET_DIRECTORY, 'Preprocessed')
        ucspeds_directory = os.path.join(preprocessed_directory, settings.UCSPEDS_PREPROCESSED_DIRECTORY)
        multiple_files_directory = os.path.join(ucspeds_directory, 'Multiple_Files')
        data_directory = os.path.join(multiple_files_directory, 'Data')
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)

        data_path = os.path.join(data_directory, 'first'+'_'+'.npy')
        joblib.dump(data, data_path)

    return data



def load_data_structure(prefix):
    preprocessed_directory = os.path.join(settings.DATASET_DIRECTORY, 'Preprocessed')
    ucspeds_directory = os.path.join(preprocessed_directory, settings.UCSPEDS_PREPROCESSED_DIRECTORY)
    multiple_files_directory = os.path.join(ucspeds_directory, 'Multiple_Files')
    data_directory = os.path.join(multiple_files_directory, 'Data')
    data_path = os.path.join(data_directory, prefix+'_'+'.npy')
    data = joblib.load(data_path)
    return data


def load_structure(is_frame, domain_id, video_id, frame_id, prefix, data_augment = None):
    preprocessed_directory = os.path.join(settings.DATASET_DIRECTORY, 'Preprocessed')
    ucspeds_directory = os.path.join(preprocessed_directory, settings.UCSPEDS_PREPROCESSED_DIRECTORY)
    multiple_files_directory = os.path.join(ucspeds_directory, 'Multiple_Files')
    if is_frame:
        directory = os.path.join(multiple_files_directory, 'Frames')
    else:
        directory = os.path.join(multiple_files_directory, 'Densities')
    if data_augment is None or data_augment == 'None':
        aug_prefix = ''
    else:
        aug_prefix = '_'+data_augment
    domain_directory = os.path.join(directory, str(domain_id))
    time_directory = os.path.join(domain_directory, str(video_id))
    frame_path = os.path.join(time_directory, prefix+'_'+str(frame_id)+aug_prefix+'.npy')

    structure = joblib.load(frame_path)

    return structure

def load_mask(data, domain_id, video_id, data_augment = None):

    if data_augment is None or data_augment == 'None':
        return data[domain_id][int(video_id)].mask
    elif data_augment == 's90':
        return data[domain_id][int(video_id)].augmentation['s90']

def load_insts(prefix_data, max_insts_per_domain=None):
    data = load_data_structure(prefix_data)

    data_insts = []

    for domain_id in settings.UCSPEDS_DOMAINS:

        domain_insts = []
    
        new_num_insts = 0
        video_ids = list(data[domain_id].keys())
        video_ids.sort()
        for video_id in video_ids:
            if max_insts_per_domain is not None and new_num_insts >= max_insts_per_domain:
                break
            new_data_insts, new_data_densities, new_data_counts = {}, {}, {}
            frame_ids = list(data[domain_id][video_id].frames.keys())
            frame_ids.sort()
            for frame_id in frame_ids:
                if max_insts_per_domain is not None and new_num_insts >= max_insts_per_domain:
                    break
                if data[domain_id][video_id].frames[frame_id].frame is not None:
                    frame_data = data[domain_id][video_id].frames[frame_id]
                    
                    new_data_insts.setdefault('None', []).append([domain_id, video_id, frame_id, 'None'])
                    if settings.LOAD_DATA_AUGMENTATION:
                        for aug_key in frame_data.augmentation:
                            new_data_insts.setdefault(aug_key, []).append([domain_id, video_id, frame_id, aug_key])
                    
                
                    new_num_insts += 1
                else:
                    print('None')
        
            if settings.TEMPORAL:
                for key in new_data_insts:
                    domain_insts.append(new_data_insts[key])
                
            else:
                for key in new_data_insts:
                    domain_insts += new_data_insts[key]

        data_insts.append(domain_insts)
    
    return data, data_insts

if __name__ == '__main__':
    data = load_data()
