##
## Note 1: File 410/410-20160430-12/000157.xml and other have an error Substitute & for &amp
## Note 2: Some frames id are the same for different frames: for example in 410-20160704-12 290.xml and 295.xml
import os
import settings
import xml.etree.ElementTree as ET
import utils
from scipy.ndimage import zoom
import numpy as np
import joblib


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
    
    def calculateCenter(self):
        return [(self.xmax+self.xmin)/2, (self.ymax+self.ymin)/2]
    
    def calculateSigma(self):
        #factor = 1/1.96 # so that exactly 5% of gaussian distribution is outside the car boundaries
        #return [factor*(self.xmax-self.xmin), factor*(self.ymax-self.ymin)]
        return [15, 15]               

class FrameData:
    
    def __init__(self, root_xml):
        self.frame = None
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
                self.vehicles.append(VehicleData(child))
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
             self.frame = zoom(self.frame, (zoom_shape[0]/self.frame.shape[0], zoom_shape[1]/self.frame.shape[1], 1))
             self.frame = self.frame.reshape(3, zoom_shape[0], zoom_shape[1])
    
    def drawGaussian(self, zoom_shape = settings.IMAGE_NEW_SHAPE):
        centers = []
        sigmas = []
        for vehicle in self.vehicles:
            centers.append(vehicle.calculateCenter())
            sigmas.append(vehicle.calculateSigma())
        #self.centers = centers
        #self.sigmas = sigmas
        self.density = utils.density_map((self.frame.shape[0], self.frame.shape[1]), centers, sigmas, zoom_shape)
        
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
    
    def extractFramesFromVideo(self, filepath):
        frame_images = utils.readFramesFromVideo(filepath)
        '''
        if len(self.frames) != len(frame_images):
            print(filepath)
            print(len(self.frames))
            print(len(frame_images))
        '''
        for i in range(min(len(self.frames), len(frame_images))):
            self.frames[i+1].frame = frame_images[i]
    
    def computeBoundingBox(self, zoom_shape = settings.IMAGE_NEW_SHAPE):
        for frame_id in self.frames:
            if self.frames[frame_id].frame is not None:
                self.frames[frame_id].computeBoundingBox(zoom_shape)
           


def load_data(max_videos_per_dataset = None, zoom_shape = settings.IMAGE_NEW_SHAPE, compute_bounding_box=True):
    webcamTDir = os.path.join(settings.DATASET_DIRECTORY, 'WebCamT')
    data = {}
    subdirs = [d for d in os.listdir(webcamTDir)]
    for subdir in subdirs:
        subdir_path = os.path.join(webcamTDir, subdir)
        if (utils.isInteger(subdir) and os.path.isdir(subdir_path) and int(subdir) in settings.DATASETS):
            camera = CameraData(int(subdir))
            data[int(subdir)] = camera
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
                if max_videos_per_dataset is not None:
                    if i >= max_videos_per_dataset:
                        break
                [time_identifier, subsubdir_path] = video
                camera.camera_times[time_identifier].extractFramesFromVideo(subsubdir_path)
                if compute_bounding_box:
                    camera.camera_times[time_identifier].computeBoundingBox(zoom_shape)
    return data

def save_data(data, prefix):
    """
    Saves data to a file
    :param data: data we want to save in the file
    """
    frame_directory = os.path.join(settings.DATASET_DIRECTORY, '/Frames')
    for domain_id in data:
        joblib.dump(data[domain_id], os.path.join(frame_directory,prefix+'_'+ str(domain_id)+'.npy'))


def save_densities(data, prefix):
    densities_directory = os.path.join(settings.DATASET_DIRECTORY, '/Densities')
    for domain_id in data:
        dens_dict = {}
        for time_id in data[domain_id].camera_times:
            dens_dict[time_id] = {}
            for frame_id in data[domain_id].camera_times[time_id].frames:
                dens_dict[time_id][frame_id] = data[domain_id].camera_times[time_id].frames[frame_id].density 
        joblib.dump(dens_dict, os.path.join(densities_directory, prefix+'_'+str(domain_id)+'.npy'))

def load_data_densities(prefix_data, prefix_densities):
    data = {}
    frame_directory = os.path.join(settings.DATASET_DIRECTORY, '/Frames')
    densities_directory = os.path.join(settings.DATASET_DIRECTORY, '/Densities')
    files = [d for d in os.listdir(frame_directory)]
    for file in files:
        file_path = os.path.join(frame_directory, file)
        if (file[0:len(prefix_data)+1] == prefix_data + '_' and not os.path.isdir(file_path)):
            domain_id = file[len(prefix_data)+1:-4]
            if utils.isInteger(domain_id) and domain_id in settings.DATASETS:
                domain_id = int(domain_id)
                data[domain_id] = joblib.load(file_path)
                densities_file = os.path.join(densities_directory, prefix_densities+'_'+str(domain_id)+'.npy')
                dens_dict = joblib.load(densities_file)
                for time_id in data[domain_id].camera_times:
                    for frame_id in data[domain_id].camera_times[time_id].frames:
                        data[domain_id].camera_times[time_id].frames[frame_id].density = dens_dict[time_id][frame_id]

def compute_densities(data):
    for domain_id in data:
        dens_dict = {}
        for time_id in data[domain_id].camera_times:
            data[domain_id].camera_times[time_id].computeBoundingBox(settings.IMAGE_NEW_SHAPE)
            
    

if __name__ == '__main__':
    data = load_data(compute_bounding_box=False)
    save_data(data, 'first')
    compute_densities(data)
    save_densities(data, 'first')
    #save_data_file('Frames_constant_variance')