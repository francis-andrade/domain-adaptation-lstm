##
## Note 1: File 410/410-20160430-12/000157.xml and other have an error Substitute & for &amp
## Note 2: Some frames id are the same for different frames: for example in 410-20160704-12 290.xml and 295.xml
import os
import settings
import xml.etree.ElementTree as ET
import utils
from scipy.ndimage import zoom
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

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
    
    def calculateGamma(self):
        return [(self.xmax-self.xmin), (self.ymax-self.ymin)]
                        

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
    
    def computeGaussian(self, zoom_shape = settings.IMAGE_ZOOM_SHAPE):
        centers = []
        gammas = []
        for vehicle in self.vehicles:
            centers.append(vehicle.calculateCenter())
            gammas.append(vehicle.calculateGamma())
        self.density = utils.density_map((self.height, self.width), centers, gammas)
        
        if zoom_shape is not None:
             self.density = zoom(self.density, (zoom_shape[0]/self.density.shape[0], zoom_shape[1]/self.density.shape[1]))
             self.density = self.density.reshape(1, zoom_shape[0], zoom_shape[1])
             self.frame = zoom(self.frame, (zoom_shape[0]/self.frame.shape[0], zoom_shape[1]/self.frame.shape[1], 1))
             self.frame = self.frame.reshape(3, zoom_shape[0], zoom_shape[1])

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
    
    def computeGaussian(self):
        for frame_id in self.frames:
            if self.frames[frame_id].frame is None:
                #print("None")
                pass
            else:
                #print("Not None")
                self.frames[frame_id].computeGaussian()
   


def load_data():
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
                if i > 10:
                    break
                [time_identifier, subsubdir_path] = video
                camera.camera_times[time_identifier].extractFramesFromVideo(subsubdir_path)
                camera.camera_times[time_identifier].computeGaussian()

    return data

if __name__ == '__main__':
    data = load_data()
    for domain in data:
        for ct in domain[data].camera_times:
            for id in domain[data].camera_times[ct].frames:
                if domain[data].camera_times[ct].frames[id].frame is not None:
                    X = domain[data].camera_times[ct].frames[id].frame
                    cid = str(domain)+'/'+ str(ct)+'/'+str(domain[data].camera_times[ct].frames[id].id)
                    count = len(domain[data].camera_times[ct].frames[id].vehicles)
                    density = domain[data].camera_times[ct].frames[id].density
                    print('Image {}: cid={}, count={}, density_sum={:.3f}'.format(i, cid, count, np.sum(density)))
                    gs = gridspec.GridSpec(2, 2)
                    fig = plt.figure()
                    ax1 = fig.add_subplot(gs[0, 0])
                    ax1.imshow(X/255.)
                    ax1.set_title('Masked image')
                    ax2 = fig.add_subplot(gs[0, 1])
                    density = density.squeeze()
                    ax2.imshow(density, cmap='gray')
                    ax2.set_title('Density map')
                    ax3 = fig.add_subplot(gs[1, :])
                    Xh = np.tile(np.mean(X, axis=2, keepdims=True), (1, 1, 3))
                    Xh[:, :, 1] *= (1-density/np.max(density))
                    Xh[:, :, 2] *= (1-density/np.max(density))
                    ax3.imshow(Xh.astype('uint8'))
                    ax3.set_title('Highlighted vehicles')
                    plt.show()