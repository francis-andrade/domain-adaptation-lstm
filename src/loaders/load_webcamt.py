"""
Module that loads the raw data from WebCamT datasets and saves it in files to fed into the models.
This module also loads data from the previous files to be fed into the model.
"""
##
## Note 1: File 410/410-20160430-12/000157.xml and other have an error Substitute & for &amp
## Note 2: Some frames id are the same for different frames: for example in 410-20160704-12 290.xml and 295.xml
## Note 3: 
#   - Some xmax/ymax are equal to width/height, for example: file:///home/francisco/Desktop/Tese/Dataset/WebCamT/511/511-20160704-15/000244.xml (vehicle 7)
#   - Some xmax/ymax are larger than width/height, for example: file:///home/francisco/Desktop/Tese/Dataset/WebCamT/164/164-20160223-09/000001.xml (vehicle 1)
#   - Some xmin/ymin are 0's, for example: file:///home/francisco/Desktop/Tese/Dataset/WebCamT/511/511-20160429-15/000139.xml (vehicle 1)
#   - Some xmin/ymin are lower than 0's, for example: file:///home/francisco/Desktop/Tese/Dataset/WebCamT/511/511-20160429-15/000125.xml (vehicle1)
#   - Some xmin/ymin are greater than xmax/ymax, for example: file:///home/francisco/Desktop/Tese/Dataset/WebCamT/410/410-20160704-12/000007.xml (vehicle 3)
import os
import settings
import xml.etree.ElementTree as ET
import utils
from scipy.ndimage import zoom
import numpy as np
import joblib
import utils.transformations
from PIL import Image, ImageDraw

class VehicleData:

    def __init__(self, xml_node):
        for child in xml_node:
            if child.tag == "id":
                self.id = int(child.text)
            elif child.tag == "bndbox":
                for grandchild in child:
                    if grandchild.tag == "xmax":
                        self.xmax = int(grandchild.text)
                    elif grandchild.tag == "xmin":
                        self.xmin = int(grandchild.text)
                    elif grandchild.tag == "ymax":
                        self.ymax = int(grandchild.text)
                    elif grandchild.tag == "ymin":
                        self.ymin = int(grandchild.text)
            elif child.tag == "type":
                self.type = int(child.text)
            elif child.tag == "direction":
                self.direction = int(child.text)
            elif child.tag == "previous":
                self.previous = int(child.text)
        
       
    
    def check_boundaries(self, frame_height, frame_width):
        if self.xmin >= self.xmax  or self.ymin >= self.ymax:
            return False
        else:
            return True
    
    def calculate_center(self):
        return [(self.xmax+self.xmin)/2, (self.ymax+self.ymin)/2]
    
    def calculate_sigma(self):
        factor = 1/1.96 # so that exactly 5% of gaussian distribution is outside the car boundaries
        #return [factor*(self.xmax-self.xmin), factor*(self.ymax-self.ymin)]
        return [4, 4]               

class FrameData:
    
    def __init__(self, root_xml):
        self.invalid = False
        self.frame = None
        self.density = None
        self.vehicles = []
        for child in root_xml:
            if child.tag == "time":
                self.year, self.month, self.day, self.hour, self.minute, self.second = utils.retrieveTimeXML(child.text)
            elif child.tag == "width":
                self.width = int(child.text)
            elif child.tag == "height":
                self.height = int(child.text)
            elif child.tag == "weather":
                self.weather = child.text
            elif child.tag == "facing":
                self.tag = int(child.text)
            elif child.tag == "flow":
                self.flow = int(child.text)
            elif child.tag == "vehicle":
                vehicle_data = VehicleData(child)
                self.vehicles.append(vehicle_data)
                if not vehicle_data.check_boundaries(self.height, self.width):
                    self.invalid = True
                    self.invalid_id = vehicle_data.id
                    break
            elif child.tag == "frame":
                self.id = int(child.text)
        
        self.count = len(self.vehicles)
    
    
    def compute_bounding_box(self, zoom_shape = settings.WEBCAMT_NEW_SHAPE, mask=None):

        if settings.USE_GAUSSIAN:
            self.draw_gaussian(zoom_shape = zoom_shape, mask=mask)
        else:
            self.draw_bounding_box()

        if zoom_shape is not None:
             if not settings.USE_GAUSSIAN:
                self.density = zoom(self.density, (zoom_shape[0]/self.density.shape[0], zoom_shape[1]/self.density.shape[1]))

        self.density = self.density.reshape(1, self.density.shape[0], self.density.shape[1])
        if settings.LOAD_DATA_AUGMENTATION:
            density_s90 = utils.transformations.transform_matrix_channels(self.density, utils.transformations.symmetric, 90)
            self.density_augmentation = {'s90': density_s90}
             
    
    def draw_gaussian(self, zoom_shape = settings.WEBCAMT_NEW_SHAPE, mask = None):
        centers = []
        sigmas = []
        for vehicle in self.vehicles:
            centers.append(vehicle.calculate_center())
            sigmas.append(vehicle.calculate_sigma())

        self.density = utils.density_map((self.original_shape[0], self.original_shape[1]), centers, sigmas, zoom_shape, mask)
        
    def draw_bounding_box(self):

        self.density = np.zeros((self.original_shape[0], self.original_shape[1]))
       
        for vehicle in self.vehicles:
            area = (vehicle.ymax - vehicle.ymin + 1)*(vehicle.xmax-vehicle.xmin+1)
            #self.density[vehicle.ymin:vehicle.ymax+1, vehicle.xmin:vehicle.xmax+1] += 1 / area
            self.density[vehicle.ymin:(vehicle.ymax+1), vehicle.xmin:(vehicle.xmax+1)] = 1

    
    def compute_mask_count(self, mask):
        self.count_mask = np.sum(self.density*mask)


class CameraData:

    def __init__(self, id):
        self.id = id
        self.camera_times = {}
    
class CameraTimeData:

    def __init__(self, time_identifier):
        self.year, self.month, self.day, self.hour, self.minute = utils.retrieveTime(time_identifier)
        self.frames = {}

    
    def find_region_of_interest(self, mask_file, zoom_shape=settings.WEBCAMT_NEW_SHAPE):
        self.points = []
        line_no = 0
        while True:
            line = mask_file.readline()
            if line == "" or line == "\n":
                break
            line_no += 1
            if line_no > 1:
                points = line.split(',')
                points[0] = points[0][1:]
                points[1] = points[1][:-1]
                if points[1][-1] == "]":
                    points[1] = points[1][:-1]
                self.points.append([int(points[0]), int(points[1])])
        

        self.mask = build_mask(self.points, zoom_shape)
        if settings.LOAD_DATA_AUGMENTATION:
            mask_s90 = utils.transformations.transform_matrix_channels(self.mask, utils.transformations.symmetric, 90)
            self.augmentation = {'s90': mask_s90}

    def extract_frames_from_video(self, filepath, zoom_shape = settings.WEBCAMT_NEW_SHAPE, max_frames_per_video=None):
        frame_images = utils.readFramesFromVideo(filepath)
        '''
        if len(self.frames) != len(frame_images):
            print(filepath)
            print(len(self.frames))
            print(len(frame_images))
        '''
        for i in range(min(len(self.frames), len(frame_images))):
            if max_frames_per_video is not None:
                if i>= max_frames_per_video:
                    break

            if self.frames[i+1].invalid:
                print("Invalid: ", filepath, " ", self.frames[i+1].id, " ", self.frames[i+1].invalid_id)
                self.frames[i+1].frame = None
            else:
                self.frames[i+1].frame = frame_images[i]
                self.frames[i+1].original_shape = frame_images[i].shape
                if zoom_shape is not None:
                    self.frames[i+1].frame = zoom(self.frames[i+1].frame, (zoom_shape[0]/self.frames[i+1].frame.shape[0], zoom_shape[1]/self.frames[i+1].frame.shape[1], 1))
                
                self.frames[i+1].frame = np.moveaxis(self.frames[i+1].frame, 2, 0)
                if settings.LOAD_DATA_AUGMENTATION:
                    frame_s90 = utils.transformations.transform_matrix_channels(self.frames[i+1].frame, utils.transformations.symmetric, 90)
                    self.frames[i+1].augmentation = { 's90': frame_s90}

    
    def compute_bounding_box(self, zoom_shape = settings.WEBCAMT_NEW_SHAPE):
        for frame_id in self.frames:
            if self.frames[frame_id].frame is not None:
                self.frames[frame_id].compute_bounding_box(zoom_shape, mask=None)
    
    def compute_mask_counts(self):
        for frame_id in self.frames:
            if self.frames[frame_id].frame is not None and self.frames[frame_id].density is not None:
                self.frames[frame_id].compute_mask_count(self.mask)

                
           


def load_data(max_videos_per_domain = None, zoom_shape = settings.WEBCAMT_NEW_SHAPE, compute_bounding_box=True, save_in_stages=False, compute_mask=True, max_frames_per_video=None):
    webcamTDir = os.path.join(settings.DATASET_DIRECTORY, 'WebCamT')
    data = {}
    subdirs = [d for d in os.listdir(webcamTDir)]
    for subdir in subdirs:
        subdir_path = os.path.join(webcamTDir, subdir)
        if (utils.isInteger(subdir) and os.path.isdir(subdir_path) and int(subdir) in settings.WEBCAMT_DOMAINS):
            domain_id = int(subdir)
            camera = CameraData(domain_id)
            data[domain_id] = camera
            subsubdirs = [d for d in os.listdir(subdir_path)]
            video_queue = []
            for subsubdir in subsubdirs:
                 if subsubdir[-4:] != ".jpg":
                    if subsubdir[0:len(subdir)+1] == subdir+"-":
                        subsubdir_path = os.path.join(subdir_path, subsubdir)
                        time_identifier = subsubdir[(len(subdir)+1):]
                        if time_identifier[0:3] != "big":
                            if time_identifier[-4] == '.':
                                time_identifier = time_identifier[:-4]
                            if time_identifier not in camera.camera_times.keys():
                                camera.camera_times[time_identifier] = CameraTimeData(time_identifier)
                    
                            subsubdir_path = os.path.join(subdir_path, subsubdir)
                            if os.path.isdir(subsubdir_path):
                                frame_xmls = [d for d in os.listdir(subsubdir_path)]
                                for frame in frame_xmls:
                                    if frame[-4:] == ".xml":
                                        frame_file = open(os.path.join(subsubdir_path, frame))
                                        xml_data = frame_file.read()
                                        xml_data = xml_data.replace('&', '&amp;')
                                        root = ET.fromstring(xml_data)
                                        frame_data = FrameData(root)
                                        frame_file_id = int(frame[:-4])
                                        camera.camera_times[time_identifier].frames[frame_file_id] = frame_data

                            elif(subsubdir[-4:] == '.avi' and os.path.isfile(subsubdir_path)):
                                camera.camera_times[time_identifier].video = subsubdir_path
                                video_queue.append([time_identifier, subsubdir_path])
                            elif(subsubdir[-4:] == '.msk' and os.path.isfile(subsubdir_path)) and compute_mask:
                                file_mask = open(subsubdir_path)
                                camera.camera_times[time_identifier].find_region_of_interest(file_mask, zoom_shape)
            
            for i, video in enumerate(video_queue):
                if max_videos_per_domain is not None:
                    if i >= max_videos_per_domain:
                        break
                print("Extracting Video: "+time_identifier)
                [time_identifier, subsubdir_path] = video
                camera.camera_times[time_identifier].extract_frames_from_video(subsubdir_path, zoom_shape, max_frames_per_video)
                if compute_bounding_box:
                    camera.camera_times[time_identifier].compute_bounding_box(zoom_shape)
                    if compute_mask:
                        print("Computing domain: ", domain_id, ' ', time_identifier)
                        camera.camera_times[time_identifier].compute_mask_counts()
            
            
            if save_in_stages:
                save_data_multiple_files_domain( data[domain_id], domain_id, settings.PREFIX_DATA, settings.PREFIX_DENSITIES)
               
    if save_in_stages:
        preprocessed_directory = os.path.join(settings.DATASET_DIRECTORY, 'Preprocessed')
        webcamt_directory = os.path.join(preprocessed_directory, settings.WEBCAMT_PREPROCESSED_DIRECTORY)
        multiple_files_directory = os.path.join(webcamt_directory, 'Multiple_Files')
        data_directory = os.path.join(multiple_files_directory, 'Data')
        data_path = os.path.join(data_directory, settings.PREFIX_DATA+'_'+'.npy')
        joblib.dump(data, data_path)

    return data

def save_data(data, prefix):
    """
    Saves data to a file
    :param data: data we want to save in the file
    """
    preprocessed_directory = os.path.join(settings.DATASET_DIRECTORY, 'Preprocessed')
    webcamt_directory = os.path.join(preprocessed_directory, settings.WEBCAMT_PREPROCESSED_DIRECTORY)
    data_directory = os.path.join(webcamt_directory, 'Data')
    for domain_id in data:
        joblib.dump(data[domain_id], os.path.join(data_directory,prefix+'_'+ str(domain_id)+'.npy'))

def save_data_multiple_files(data, prefix_data, prefix_frames, prefix_densities):

    for domain_id in data:      
        save_data_multiple_files_domain(data[domain_id], domain_id, prefix_frames, prefix_densities)
    
    preprocessed_directory = os.path.join(settings.DATASET_DIRECTORY, 'Preprocessed')
    webcamt_directory = os.path.join(preprocessed_directory, settings.WEBCAMT_PREPROCESSED_DIRECTORY)
    multiple_files_directory = os.path.join(webcamt_directory, 'Multiple_Files')
    data_directory = os.path.join(multiple_files_directory, 'Data')
    data_path = os.path.join(data_directory, prefix_data+'_'+'.npy')
    joblib.dump(data, data_path)

def save_data_multiple_files_domain(data_domain, domain_id, prefix_frames, prefix_densities):
    preprocessed_directory = os.path.join(settings.DATASET_DIRECTORY, 'Preprocessed')
    webcamt_directory = os.path.join(preprocessed_directory, settings.WEBCAMT_PREPROCESSED_DIRECTORY)
    multiple_files_directory = os.path.join(webcamt_directory, 'Multiple_Files')
    data_directory = os.path.join(multiple_files_directory, 'Data')
    frames_directory = os.path.join(multiple_files_directory, 'Frames')
    densities_directory = os.path.join(multiple_files_directory, 'Densities')
    str_id = str(domain_id)
    domain_directory_frame = os.path.join(frames_directory, str_id)
    domain_directory_density = os.path.join(densities_directory, str_id)

    if not os.path.exists(domain_directory_frame):
        os.makedirs(domain_directory_frame)
    if not os.path.exists(domain_directory_density):
        os.makedirs(domain_directory_density)
    
    for time_id in data_domain.camera_times:
        str_id = str(time_id)
        time_directory_frame = os.path.join(domain_directory_frame, str_id)
        time_directory_density = os.path.join(domain_directory_density, str_id)

        if not os.path.exists(time_directory_frame):
            os.makedirs(time_directory_frame)
        if not os.path.exists(time_directory_density):
            os.makedirs(time_directory_density)
            
        for frame_id in data_domain.camera_times[time_id].frames:
            frame = data_domain.camera_times[time_id].frames[frame_id].frame 
            density = data_domain.camera_times[time_id].frames[frame_id].density
            if frame is not None and density is not None:
                str_id = str(frame_id)
                frame_path = os.path.join(time_directory_frame, prefix_frames+'_'+str(str_id)+'.npy')
                joblib.dump(frame, frame_path)
                density_path = os.path.join(time_directory_density, prefix_densities+'_'+str(str_id)+'.npy')
                joblib.dump(density, density_path)

                data_domain.camera_times[time_id].frames[frame_id].frame = 0
                data_domain.camera_times[time_id].frames[frame_id].density = 0

                if settings.LOAD_DATA_AUGMENTATION:
                    frames_aug = data_domain.camera_times[time_id].frames[frame_id].augmentation
                    densities_aug = data_domain.camera_times[time_id].frames[frame_id].density_augmentation
                    for frame_aug_key in frames_aug:
                        frame_aug = frames_aug[frame_aug_key]
                        density_aug = densities_aug[frame_aug_key]
                        frame_path = os.path.join(time_directory_frame, prefix_frames+'_'+str(str_id)+'_'+frame_aug_key+'.npy')
                        joblib.dump(frame_aug, frame_path)
                        density_path = os.path.join(time_directory_density, prefix_densities+'_'+str(str_id)+'_'+frame_aug_key+'.npy')
                        joblib.dump(density_aug, density_path)
                        frames_aug[frame_aug_key] = 0
                        densities_aug[frame_aug_key] = 0

def load_data_structure(prefix):
    preprocessed_directory = os.path.join(settings.DATASET_DIRECTORY, 'Preprocessed')
    webcamt_directory = os.path.join(preprocessed_directory, settings.WEBCAMT_PREPROCESSED_DIRECTORY)
    multiple_files_directory = os.path.join(webcamt_directory, 'Multiple_Files')
    data_directory = os.path.join(multiple_files_directory, 'Data')
    data_path = os.path.join(data_directory, prefix+'_'+'.npy')
    data = joblib.load(data_path)
    return data



def load_structure(is_frame, domain_id, time_id, frame_id, prefix, data_augment = None):
    preprocessed_directory = os.path.join(settings.DATASET_DIRECTORY, 'Preprocessed')
    webcamt_directory = os.path.join(preprocessed_directory, settings.WEBCAMT_PREPROCESSED_DIRECTORY)
    multiple_files_directory = os.path.join(webcamt_directory, 'Multiple_Files')
    if is_frame:
        directory = os.path.join(multiple_files_directory, 'Frames')
    else:
        directory = os.path.join(multiple_files_directory, 'Densities')
    if data_augment is None or data_augment == 'None':
        aug_prefix = ''
    else:
        aug_prefix = '_'+data_augment
    domain_directory = os.path.join(directory, str(domain_id))
    time_directory = os.path.join(domain_directory, str(time_id))
    frame_path = os.path.join(time_directory, prefix+'_'+str(frame_id)+aug_prefix+'.npy')

    structure = joblib.load(frame_path)
    return structure

def build_mask(points, zoom_shape = settings.WEBCAMT_NEW_SHAPE):
    tuple_points = []
    
    for point in points:
        tuple_points.append((point[0], point[1]))
    
    img = Image.new('L', (settings.WEBCAMT_SHAPE[1], settings.WEBCAMT_SHAPE[0]), 0)
    ImageDraw.Draw(img).polygon(tuple_points, outline=1, fill=1)
    mask = np.array(img)

    if zoom_shape is not None:
        mask = zoom(mask, (zoom_shape[0] / settings.WEBCAMT_SHAPE[0], zoom_shape[1] / settings.WEBCAMT_SHAPE[1]))

    return np.array([mask], dtype=np.float)

def load_mask(data, domain_id, time_id, data_augment=None):

    if data_augment is None or data_augment == 'None':
        return data[int(domain_id)].camera_times[time_id].mask
    elif data_augment == 's90':
        return data[int(domain_id)].camera_times[time_id].augmentation['s90']


def save_densities(data, prefix):
    densities_directory = os.path.join(settings.DATASET_DIRECTORY, 'Preprocessed/'+settings.WEBCAMT_PREPROCESSED_DIRECTORY+'/Densities')
    for domain_id in data:
        dens_dict = {}
        for time_id in data[domain_id].camera_times:
            dens_dict[time_id] = {}
            for frame_id in data[domain_id].camera_times[time_id].frames:
                frame_data = data[domain_id].camera_times[time_id].frames[frame_id]
                if frame_data.frame is not None:
                    dens_dict[time_id][frame_id] = {}
                    dens_dict[time_id][frame_id]['None'] = frame_data.density 
                    if settings.LOAD_DATA_AUGMENTATION:
                        densities_aug = data[domain_id].camera_times[time_id].frames[frame_id].density_augmentation
                        for aug_key in densities_aug:
                            dens_dict[time_id][frame_id][aug_key] = frame_data.density_augmentation[aug_key]

        joblib.dump(dens_dict, os.path.join(densities_directory, prefix+'_'+str(domain_id)+'.npy'))

def load_data_from_file(prefix_data, prefix_densities):
    data = {}
    data_directory = os.path.join(settings.DATASET_DIRECTORY, 'Preprocessed/'+settings.WEBCAMT_PREPROCESSED_DIRECTORY+'/Data')
    densities_directory = os.path.join(settings.DATASET_DIRECTORY, 'Preprocessed/'+settings.WEBCAMT_PREPROCESSED_DIRECTORY+'/Densities')
    files = [d for d in os.listdir(data_directory)]
    for file in files:
        file_path = os.path.join(data_directory, file)
        if (file[0:len(prefix_data)+1] == prefix_data + '_' and not os.path.isdir(file_path)):
            domain_id = file[len(prefix_data)+1:-4]
            if utils.isInteger(domain_id) and int(domain_id) in settings.WEBCAMT_DOMAINS:
                domain_id = int(domain_id)
                data[domain_id] = joblib.load(file_path)
                densities_file = os.path.join(densities_directory, prefix_densities+'_'+str(domain_id)+'.npy')
                dens_dict = joblib.load(densities_file)
                for time_id in dens_dict:
                    for frame_id in dens_dict[time_id]:
                        frame_data = data[domain_id].camera_times[time_id].frames[frame_id]
                        frame_data.density = dens_dict[time_id][frame_id]['None']
                        if settings.LOAD_DATA_AUGMENTATION:
                            frame_data.density_augmentation = {}
                            for aug_key in dens_dict[time_id][frame_id]:
                                if aug_key != 'None':
                                    frame_data.density_augmentation[aug_key] = dens_dict[time_id][frame_id][aug_key]

    
    return data

def load_insts(prefix_data, max_insts_per_domain=None):
    data = load_data_structure(prefix_data)
    
    print('Finished loading data')

    data_insts = []

    for domain_id in settings.WEBCAMT_DOMAINS:
        domain_insts = []
    
        new_num_insts = 0
        camera_times_ids = list(data[domain_id].camera_times.keys())
        camera_times_ids.sort()
        for time_id in camera_times_ids:
            if max_insts_per_domain is not None and new_num_insts >= max_insts_per_domain:
                break
            new_data_insts, new_data_densities, new_data_counts = {}, {}, {}
            frame_ids = list(data[domain_id].camera_times[time_id].frames.keys())
            frame_ids.sort()
            for frame_id in frame_ids:
                if max_insts_per_domain is not None and new_num_insts >= max_insts_per_domain:
                    break
                if data[domain_id].camera_times[time_id].frames[frame_id].frame is not None:
                    frame_data = data[domain_id].camera_times[time_id].frames[frame_id]
                    
                    new_data_insts.setdefault('None', []).append([domain_id, time_id, frame_id, 'None'])
                    if settings.LOAD_DATA_AUGMENTATION:
                        for aug_key in frame_data.augmentation:
                            new_data_insts.setdefault(aug_key, []).append([domain_id, time_id, frame_id, aug_key])

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
   

def compute_densities(data):
    for domain_id in data:
        dens_dict = {}
        for time_id in data[domain_id].camera_times:
            data[domain_id].camera_times[time_id].compute_bounding_box(settings.WEBCAMT_NEW_SHAPE)
            
    

if __name__ == '__main__':
    
    '''
    data = load_data_from_file('first', 'first')
    save_data_multiple_files(data, 'first', 'first', 'first')
    '''
    '''
    data = load_data(compute_bounding_box=False)
    save_data(data, 'first')
    compute_densities(data)
    save_densities(data, 'first')
    '''

    '''
    data = load_data_from_file('first', 'first')
    compute_densities(data)
    save_densities(data, 'proportional')
    save_data_multiple_files(data, 'first', 'first', 'proportional')
    '''

    data = load_data(save_in_stages=True)