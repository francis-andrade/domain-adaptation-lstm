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
import transformations


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
    
    def calculateCenter(self):
        return [(self.xmax+self.xmin)/2, (self.ymax+self.ymin)/2]
    
    def calculateSigma(self):
        factor = 1/1.96 # so that exactly 5% of gaussian distribution is outside the car boundaries
        return [factor*(self.xmax-self.xmin), factor*(self.ymax-self.ymin)]
        #return [15, 15]               

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
    
    def computeBoundingBox(self, zoom_shape = settings.IMAGE_NEW_SHAPE):

        if settings.USE_GAUSSIAN:
            self.drawGaussian(zoom_shape = settings.IMAGE_NEW_SHAPE)
        else:
            self.drawBoundingBox()

        if zoom_shape is not None:
             if not settings.USE_GAUSSIAN:
                self.density = zoom(self.density, (zoom_shape[0]/self.density.shape[0], zoom_shape[1]/self.density.shape[1]))
             self.density = self.density.reshape(1, zoom_shape[0], zoom_shape[1])
             if settings.USE_DATA_AUGMENTATION:
                density_r180 = transformations.transform_matrix_channels(self.density, utils.rotate, 180)
                density_s0 = transformations.transform_matrix_channels(self.density, utils.symmetric, 0)
                density_s90 = transformations.transform_matrix_channels(self.density, utils.symmetric, 90)
                density_brightness = np.copy(self.density)
                density_contrast = np.copy(self.density)
                self.density_augmentation = {'r180': density_r180, 's0': density_s0, 's90': density_s90, 'brightness': density_brightness, 'contrast': density_contrast}
             
    
    def drawGaussian(self, zoom_shape = settings.IMAGE_NEW_SHAPE):
        centers = []
        sigmas = []
        for vehicle in self.vehicles:
            centers.append(vehicle.calculateCenter())
            sigmas.append(vehicle.calculateSigma())
        self.density = utils.density_map((self.original_shape[0], self.original_shape[1]), centers, sigmas, zoom_shape)
        
    def drawBoundingBox(self):

        self.density = np.zeros((self.frame.shape[0], self.frame.shape[1]))
       
        for vehicle in self.vehicles:
            area = (vehicle.ymax - vehicle.ymin + 1)*(vehicle.xmax-vehicle.xmin+1)
            self.density[vehicle.ymin:vehicle.ymax+1, vehicle.xmin:vehicle.xmax+1] += 1 / area
    

class CameraData:

    def __init__(self, id):
        self.id = id
        self.camera_times = {}
    
class CameraTimeData:

    def __init__(self, time_identifier):
        self.year, self.month, self.day, self.hour, self.minute = utils.retrieveTime(time_identifier)
        self.frames = {}

    
    def find_region_of_interest(self, mask):
        self.points = []
        line_no = 0
        while True:
            line = mask.readline()
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
    
    def extractFramesFromVideo(self, filepath, zoom_shape = settings.IMAGE_NEW_SHAPE):
        frame_images = utils.readFramesFromVideo(filepath)
        '''
        if len(self.frames) != len(frame_images):
            print(filepath)
            print(len(self.frames))
            print(len(frame_images))
        '''
        for i in range(min(len(self.frames), len(frame_images))):
            if self.frames[i+1].invalid:
                print("Invalid: ", filepath, " ", self.frames[i+1].id, " ", self.frames[i+1].invalid_id)
                self.frames[i+1].frame = None
            else:
                self.frames[i+1].frame = frame_images[i]
                self.frames[i+1].original_shape = frame_images[i].shape
                if zoom_shape is not None:
                    self.frames[i+1].frame = zoom(self.frames[i+1].frame, (zoom_shape[0]/self.frames[i+1].frame.shape[0], zoom_shape[1]/self.frames[i+1].frame.shape[1], 1))
                    self.frames[i+1].frame = np.moveaxis(self.frames[i+1].frame, 2, 0)
                    if settings.USE_DATA_AUGMENTATION:
                        frame_r180 = transformations.transform_matrix_channels(self.frames[i+1].frame, utils.rotate, 180)
                        frame_s0 = transformations.transform_matrix_channels(self.frames[i+1].frame, utils.symmetric, 0)
                        frame_s90 = transformations.transform_matrix_channels(self.frames[i+1].frame, utils.symmetric, 90)
                        frame_brightness = utils.change_brightness_contrast(self.frames[i+1].frame, 50, 0)
                        frame_contrast = utils.change_brightness_contrast(self.frames[i+1].frame, 0, 30)
                        self.frames[i+1].augmentation = {'r180': frame_r180, 's0': frame_s0, 's90': frame_s90, 'contrast': frame_contrast, 'brightness': frame_brightness}

    
    def computeBoundingBox(self, zoom_shape = settings.IMAGE_NEW_SHAPE):
        for frame_id in self.frames:
            if self.frames[frame_id].frame is not None:
                self.frames[frame_id].computeBoundingBox(zoom_shape)
                
           


def load_data(max_videos_per_domain = None, zoom_shape = settings.IMAGE_NEW_SHAPE, compute_bounding_box=True, save_in_stages=False):
    webcamTDir = os.path.join(settings.DATASET_DIRECTORY, 'WebCamT')
    data = {}
    subdirs = [d for d in os.listdir(webcamTDir)]
    for subdir in subdirs:
        subdir_path = os.path.join(webcamTDir, subdir)
        if (utils.isInteger(subdir) and os.path.isdir(subdir_path) and int(subdir) in settings.DATASETS):
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
                            elif(subsubdir[-4:] == '.msk' and os.path.isfile(subsubdir_path)):
                                file = open(subsubdir_path)
                                camera.camera_times[time_identifier].find_region_of_interest(file)
            
            for i, video in enumerate(video_queue):
                if max_videos_per_domain is not None:
                    if i >= max_videos_per_domain:
                        break
                [time_identifier, subsubdir_path] = video
                camera.camera_times[time_identifier].extractFramesFromVideo(subsubdir_path, zoom_shape)
                if compute_bounding_box:
                    camera.camera_times[time_identifier].computeBoundingBox(zoom_shape)
            
            if save_in_stages:
                save_data_multiple_files_domain( data[domain_id], domain_id, 'first', 'first')
               
    if save_in_stages:
        multiple_files_directory = os.path.join(settings.DATASET_DIRECTORY, 'Multiple_Files')
        data_directory = os.path.join(multiple_files_directory, 'Data')
        data_path = os.path.join(data_directory, 'first'+'_'+'.npy')
        joblib.dump(data, data_path)

    return data

def save_data(data, prefix):
    """
    Saves data to a file
    :param data: data we want to save in the file
    """
    data_directory = os.path.join(settings.DATASET_DIRECTORY, 'Data')
    for domain_id in data:
        joblib.dump(data[domain_id], os.path.join(data_directory,prefix+'_'+ str(domain_id)+'.npy'))

def save_data_multiple_files(data, prefix_data, prefix_frames, prefix_densities):

    for domain_id in data:      
        save_data_multiple_files_domain(data[domain_id], domain_id, prefix_frames, prefix_densities)
    
    multiple_files_directory = os.path.join(settings.DATASET_DIRECTORY, 'Multiple_Files')
    data_directory = os.path.join(multiple_files_directory, 'Data')
    data_path = os.path.join(data_directory, prefix_data+'_'+'.npy')
    joblib.dump(data, data_path)

def save_data_multiple_files_domain(data_domain, domain_id, prefix_frames, prefix_densities):
    multiple_files_directory = os.path.join(settings.DATASET_DIRECTORY, 'Multiple_Files')
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
    
    for time_id in data[domain_id].camera_times:
        str_id = str(time_id)
        time_directory_frame = os.path.join(domain_directory_frame, str_id)
        time_directory_density = os.path.join(domain_directory_density, str_id)

        if not os.path.exists(time_directory_frame):
            os.makedirs(time_directory_frame)
        if not os.path.exists(time_directory_density):
            os.makedirs(time_directory_density)
            
        for frame_id in data[domain_id].camera_times[time_id].frames:
            frame = data[domain_id].camera_times[time_id].frames[frame_id].frame 
            density = data[domain_id].camera_times[time_id].frames[frame_id].density
            if frame is not None and density is not None:
                str_id = str(frame_id)
                frame_path = os.path.join(time_directory_frame, prefix_frames+'_'+str(str_id)+'.npy')
                joblib.dump(frame, frame_path)
                density_path = os.path.join(time_directory_density, prefix_densities+'_'+str(str_id)+'.npy')
                joblib.dump(density, density_path)

                data[domain_id].camera_times[time_id].frames[frame_id].frame = 0
                data[domain_id].camera_times[time_id].frames[frame_id].density = 0

                if settings.USE_DATA_AUGMENTATION:
                    frames_aug = data[domain_id].camera_times[time_id].frames[frame_id].augmentation
                    densities_aug = data[domain_id].camera_times[time_id].frames[frame_id].density_augmentation
                    for frame_aug_key in frames_aug:
                        frame_aug = frames_aug[frame_aug_key]
                        density_aug = densities_aug[frame_aug_key]
                        frame_path = os.path.join(time_directory_frame, prefix_frames+'_'+str(str_id)+'_'+frame_aug_key+'.npy')
                        joblib.dump(frame, frame_path)
                        density_path = os.path.join(time_directory_density, prefix_densities+'_'+str(str_id)+'_'+frame_aug_key+'.npy')
                        joblib.dump(density, density_path)
                        frames_aug[frame_aug_key] = 0
                        densities_aug[frame_aug_key] = 0

def load_data_structure(prefix):
    multiple_files_directory = os.path.join(settings.DATASET_DIRECTORY, 'Multiple_Files')
    data_directory = os.path.join(multiple_files_directory, 'Data')
    data_path = os.path.join(data_directory, prefix+'_'+'.npy')
    data = joblib.load(data_path)
    return data



def load_structure(is_frame, domain_id, time_id, frame_id, prefix, data_augment = None):
    multiple_files_directory = os.path.join(settings.DATASET_DIRECTORY, 'Multiple_Files')
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

def save_densities(data, prefix):
    densities_directory = os.path.join(settings.DATASET_DIRECTORY, 'Densities')
    for domain_id in data:
        dens_dict = {}
        for time_id in data[domain_id].camera_times:
            dens_dict[time_id] = {}
            for frame_id in data[domain_id].camera_times[time_id].frames:
                frame_data = data[domain_id].camera_times[time_id].frames[frame_id]
                if frame_data.frame is not None:
                    dens_dict[time_id][frame_id] = {}
                    dens_dict[time_id][frame_id]['None'] = frame_data.density 
                    if settings.USE_DATA_AUGMENTATION:
                        densities_aug = data[domain_id].camera_times[time_id].frames[frame_id].density_augmentation
                        for aug_key in densities_aug:
                            dens_dict[time_id][frame_id][aug_key] = frame_data.density_augmentation[aug_key]

        joblib.dump(dens_dict, os.path.join(densities_directory, prefix+'_'+str(domain_id)+'.npy'))

def load_data_from_file(prefix_data, prefix_densities):
    data = {}
    data_directory = os.path.join(settings.DATASET_DIRECTORY, 'Data')
    densities_directory = os.path.join(settings.DATASET_DIRECTORY, 'Densities')
    files = [d for d in os.listdir(data_directory)]
    for file in files:
        file_path = os.path.join(data_directory, file)
        if (file[0:len(prefix_data)+1] == prefix_data + '_' and not os.path.isdir(file_path)):
            domain_id = file[len(prefix_data)+1:-4]
            if utils.isInteger(domain_id) and int(domain_id) in settings.DATASETS:
                domain_id = int(domain_id)
                data[domain_id] = joblib.load(file_path)
                densities_file = os.path.join(densities_directory, prefix_densities+'_'+str(domain_id)+'.npy')
                dens_dict = joblib.load(densities_file)
                for time_id in dens_dict:
                    for frame_id in dens_dict[time_id]:
                        frame_data = data[domain_id].camera_times[time_id].frames[frame_id]
                        frame_data.density = dens_dict[time_id][frame_id]['None']
                        if settings.USE_DATA_AUGMENTATION:
                            frame_data.density_augmentation = {}
                            for aug_key in dens_dict[time_id][frame_id]:
                                if aug_key != 'None':
                                    frame_data.density_augmentation[aug_key] = dens_dict[time_id][frame_id][aug_key]

    
    return data

def load_insts(prefix_data, max_insts_per_domain=None):
    data = load_data_structure(prefix_data)
    
    print('Finished loading data')

    data_insts, data_counts = [], []

    for domain_id in settings.DATASETS:
        #print(domain_id)
        domain_insts, domain_counts = [], []
    
        new_num_insts = 0
        camera_times_ids = list(data[domain_id].camera_times.keys())
        camera_times_ids.sort()
        for time_id in camera_times_ids:
            #print('\t', time_id)
            if max_insts_per_domain is not None and new_num_insts > max_insts_per_domain:
                break
            new_data_insts, new_data_densities, new_data_counts = {}, {}, {}
            frame_ids = list(data[domain_id].camera_times[time_id].frames.keys())
            frame_ids.sort()
            for frame_id in frame_ids:
                if max_insts_per_domain is not None and new_num_insts > max_insts_per_domain:
                    break
                if data[domain_id].camera_times[time_id].frames[frame_id].frame is not None:
                    frame_data = data[domain_id].camera_times[time_id].frames[frame_id]
                    
                    new_data_insts.setdefault('None', []).append([domain_id, time_id, frame_id, 'None'])
                    if settings.USE_DATA_AUGMENTATION:
                        for aug_key in frame_data.augmentation:
                            new_data_insts.setdefault(aug_key, []).append([domain_id, time_id, frame_id, aug_key])
                    

                    no_vehicles = len(frame_data.vehicles)
                    new_data_counts.setdefault('None', []).append(no_vehicles)
                    if settings.USE_DATA_AUGMENTATION:
                        for aug_key in frame_data.augmentation:
                            new_data_counts.setdefault(aug_key, []).append(no_vehicles)
                
                    new_num_insts += 1
                else:
                    print('None')
        
            if settings.TEMPORAL:
                for key in new_data_insts:
                    domain_insts.append(new_data_insts[key])
                    domain_counts.append(new_data_counts[key])
                
            else:
                for key in new_data_insts:
                    domain_insts += new_data_insts[key]
                    domain_counts += new_data_counts[key]

        data_insts.append(domain_insts)
        data_counts.append(domain_counts)
 

    for domain_id in range(len(data_insts)):

        data_counts[domain_id] = np.array(data_counts[domain_id]) 
        data_insts[domain_id] = np.array(data_insts[domain_id]) 
    
    return data_insts, data_counts
   

def compute_densities(data):
    for domain_id in data:
        dens_dict = {}
        for time_id in data[domain_id].camera_times:
            data[domain_id].camera_times[time_id].computeBoundingBox(settings.IMAGE_NEW_SHAPE)
            
    

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
    data = load_data_from_file('first', 'first')
    compute_densities(data)
    save_densities(data, 'proportional')
    save_data_multiple_files(data, 'first', 'first', 'proportional')