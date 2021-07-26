import cv2.cv2 as cv2
import os
import json
from lxml.etree import Element, SubElement, ElementTree

folder = "day1"
images_day1 = os.listdir(folder)
jsons_day1 = os.listdir("day_1JSON")

classes = ['sign_pole', 'street_light_pole', 'sign', 'hanging_sign', 'traffic_light_pole', 'bus_stop',
           'road_marking', 'traffic_light', 'p_frame', 'speed_bump', 'temp_frame', 'console_frame',
           'g_frame', 'info_banner', 'info_display', 'double_pole', 't_frame', 'frame']

errorFiles=[]
for im in images_day1:
    im_path = os.path.join(folder, im)
    json_path = os.path.join("day_1JSON", im[:-4]+".json")
    try:
        with open(json_path) as json_file:
            data = json.load(json_file)
    except Exception:
        errorFiles.append({im_path: json_path})
    print(data)
    break
#     image = cv2.imread(im_path)
#     width, height, depth = image.shape
#     # print(width, height, depth)
#
#     node_root = Element('annotation')
#
#     node_folder = SubElement(node_root, 'folder')
#     node_folder.text = folder
#
#     node_filename = SubElement(node_root, 'filename')
#     node_filename.text = im
#
#     node_size = SubElement(node_root, 'size')
#     node_width = SubElement(node_size, 'width')
#     node_width.text = str(width)
#
#     node_height = SubElement(node_size, 'height')
#     node_height.text = str(height)
#
#     node_depth = SubElement(node_size, 'depth')
#     node_depth.text = str(depth)
#     for obj in data:
#
#         node_object = SubElement(node_root, 'object')
#         node_name = SubElement(node_object, 'name')
#         node_name.text = obj["class"]
#
#         node_difficult = SubElement(node_object, 'score')
#         node_difficult.text = str(obj["score"])
#         node_bndbox = SubElement(node_object, 'bndbox')
#         node_xmin = SubElement(node_bndbox, 'xmin')
#         node_xmin.text = str(obj["bbox"][0])
#         node_ymin = SubElement(node_bndbox, 'ymin')
#         node_ymin.text = str(obj["bbox"][1])
#         node_xmax = SubElement(node_bndbox, 'xmax')
#         node_xmax.text = str(obj["bbox"][2])
#         node_ymax = SubElement(node_bndbox, 'ymax')
#         node_ymax.text = str(obj["bbox"][3])
#
#     tree = ElementTree(node_root)
#     tree.write(folder+'_PascalVoc/'+im[:-4]+'.xml', pretty_print=True, xml_declaration=True, encoding="utf-8")
#
# print(errorFiles)
# print(len(errorFiles))