import os
import settings
import xml.etree.ElementTree as ET
import utils

class VehicleData:

    def __init__(self):
        pass
    
    def parseXMLNode(self, node):
        pass

class FrameData:

    def __init__(self):
        pass

    def parseXML(self, node):
        pass

class CameraData:

    def __init__(self, id):
        self.id = id
    
class CameraTimeData:

    def __init__(self, time_identifier):
        self.year, self.month, self.day, self.hour, self.minute = utils.retrieveTime(time_identifier)
        self.points = []
    
    def find_region_of_interest(self, mask):
        
        line_no = 0
        while True:
            line = mask.readline()
            line_no += 1
            if line_no > 1:
                points = line.split(',')
                points[0] = points[0][1:]
                points[1] = points[1][:-1]
                self.points.append([int(points[0]), int(points[1])])
    
    def parse_xml(self, xml):
        pass

def load_data():
    data = {}
    subdirs = [d for d in os.listdir(settings.DATASET_DIRECTORY)]
    for subdir in subdirs:
        subdir_path = os.path.join(settings.DATASET_DIRECTORY, subdir)
        if (utils.isInteger(subdir) and os.path.isdir(subdir_path)):
            camera = CameraData(int(subdir))
            data[int(subdir)] = camera
            camera_times = {}
            subsubdirs = [d for d in os.listdir(subdir_path)]
            for subsubdir in subsubdirs:
                 if subsubdir[0:len(subdir)+1] == subdir+"-":
                    subsubdir_path = os.path.join(subdir_path, subsubdir)
                    time_identifier = subsubdir[(len(subdir)+1):]
                    if time_identifier[-4] == '.':
                        time_identifier = time_identifier[:-4]
                    if time_identifier not in camera_times.keys():
                        camera_times[time_identifier] = CameraTimeData(time_identifier)
                    if os.path.isdir(subsubdir_path):
                        frame_xmls = [d for d in os.listdir(subdir_path)]
                        for frame in frame_xmls:
                            pass

                    elif(subsubdir[-4:] == '.avi' and os.path.isfile(subsubdir_path)):
                        camera_times[time_identifier].video = os.path.join(subdir_path, subsubdir)
                    elif(subsubdir[-4:] == '.msk' and os.path.isfile(subsubdir_path)):
                        file = open(os.path.join(subdir_path, subsubdir))
                        camera_times[time_identifier].find_region_of_interest(file)


if __name__ == '__main__':
    load_data()