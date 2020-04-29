##
## Note 1: File 410/410-20160430-12/000157.xml and other have an error Substitute & for &amp
## Note 2: Some frames id are the same for different frames: for example in 410-20160704-12 290.xml and 295.xml
import os
import settings
import xml.etree.ElementTree as ET
import utils
from scipy.ndimage import zoom
import numpy as np


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
            self.drawGaussian()
        else:
            self.drawBoundingBox()

        if zoom_shape is not None:
             self.density = zoom(self.density, (zoom_shape[0]/self.density.shape[0], zoom_shape[1]/self.density.shape[1]))
             self.density = self.density.reshape(1, zoom_shape[0], zoom_shape[1])
             self.frame = zoom(self.frame, (zoom_shape[0]/self.frame.shape[0], zoom_shape[1]/self.frame.shape[1], 1))
             self.frame = self.frame.reshape(3, zoom_shape[0], zoom_shape[1])
    
    def drawGaussian(self):
        centers = []
        sigmas = []
        for vehicle in self.vehicles:
            centers.append(vehicle.calculateCenter())
            sigmas.append(vehicle.calculateSigma())
        #self.centers = centers
        #self.sigmas = sigmas
        self.density = utils.density_map((self.frame.shape[0], self.frame.shape[1]), centers, sigmas)
        
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
           


def load_data(max_videos_per_dataset = None, zoom_shape = settings.IMAGE_NEW_SHAPE):
    data = {}
    subdirs = [d for d in os.listdir(settings.DATASET_DIRECTORY)]
    for subdir in subdirs:
        subdir_path = os.path.join(settings.DATASET_DIRECTORY, subdir)
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
                camera.camera_times[time_identifier].computeBoundingBox(zoom_shape)
    return data

def save_data_file(filename, data):
    """
    Saves data to a file
    :param filename: name of the file to where the data should be saved
    :param data: data we want to save in the file
    """
    filepath = os.path.join(settings.DATASET_DIRECTORY, '../Frames/'+filename)
    np.save(filepath, data)

if __name__ == '__main__':
    pass
    #data = load_data()
    #save_data_file('Frames_constant_variance')