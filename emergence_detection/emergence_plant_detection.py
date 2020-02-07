'''
Created on Sep 16, 2019

@author: zli
'''
import cv2
import sys, os, json, argparse, shutil, math, csv, terra_common
from glob import glob
import numpy as np
from PIL import Image
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.utils.timer import Timer
from scipy.ndimage.filters import convolve


def options():
    
    parser = argparse.ArgumentParser(description='Terra-ref emergence plant detection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    #parser.add_argument("-i", "--input_path", help="full path of input image")
    #parser.add_argument("-o", "--out_dir", help="output directory")
    parser.add_argument("-m", "--module_path", help="R-CNN trained module path")

    args = parser.parse_args()

    return args


def main():
    
    args = options()
    
    detector = load_model(args.module_path)

    in_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/raw_data/stereoTop/2018-05-03'
    out_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/emergence_detection/2018-05-03'
    full_day_process(in_dir, out_dir, detector)
    
    in_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/raw_data/stereoTop/2018-05-04'
    out_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/emergence_detection/2018-05-04'
    full_day_process(in_dir, out_dir, detector)
    
    in_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/raw_data/stereoTop/2018-05-05'
    out_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/emergence_detection/2018-05-05'
    full_day_process(in_dir, out_dir, detector)
    
    in_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/raw_data/stereoTop/2018-05-06'
    out_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/emergence_detection/2018-05-06'
    full_day_process(in_dir, out_dir, detector)
    
    return

def test():
    
    in_dir = '/media/zli/Elements/terra_evaluation/emp/'
    out_dir = '/media/zli/Elements/terra_evaluation/emp/outputs/'
    bety_dir = '/media/zli/Elements/terra_evaluation/emp/outputs/BETY'
    
    convt = terra_common.CoordinateConverter()
    convt.bety_query('2017-04-27', False)
    str_date = '2017-04-27'
    
    full_day_aggregate(in_dir, out_dir, bety_dir, str_date, 6, convt)
    
    str_date = '2017-04-28'
    
    full_day_aggregate(in_dir, out_dir, bety_dir, str_date, 6, convt)
    
    str_date = '2017-04-29'
    
    full_day_aggregate(in_dir, out_dir, bety_dir, str_date, 6, convt)
    
    
    '''
    in_1 = '/media/zli/Elements/terra_evaluation/emp/2017-04-27/2017-04-27__09-43-14-707'
    in_2 = '/media/zli/Elements/terra_evaluation/emp/2017-04-27/2017-04-27__09-43-16-229'
    
    convt = terra_common.CoordinateConverter()
    convt.bety_query('2017-04-27', False)
    
    boxList = []
    
    pl1, emp1 = parse_data_from_emp_csv(in_1, convt)
    for emp in emp1:
        boxList.append(emp)
    pl2, emp2 = parse_data_from_emp_csv(in_2, convt)
    for emp in emp2:
        saveFlag = True
        for savedItem in boxList:
            center_dist = math.sqrt((savedItem[0]-emp[0])**2+(savedItem[1]-emp[1])**2)
            radiusAdd = savedItem[2]+emp[2]
            if center_dist < radiusAdd:
                saveFlag = False
                break
        
        if saveFlag:
            boxList.append(emp)
    '''
    return

def full_day_process(in_dir, out_dir, detector):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    list_dirs = os.listdir(in_dir)
    
    for d in list_dirs:
        in_path = os.path.join(in_dir, d)
        out_path = os.path.join(out_dir, d)
        
        if not os.path.isdir(in_dir):
            continue
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
            
        singe_image_process(in_path, out_path, detector)
    
    return

def full_day_aggregate(in_path, out_path, bety_dir, str_date, seasonNum, convt):
    
    if not os.path.isdir(bety_dir):
        os.makedirs(bety_dir)
    
    out_dir = os.path.join(in_path, str_date)
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        
    in_dir = os.path.join(in_path, str_date)
        
    list_dirs = os.listdir(in_dir)
    
    boxList = [[] for i in range(864)]
    
    for d in list_dirs:
        in_path = os.path.join(in_dir, d)
        
        if not os.path.isdir(in_path):
            continue
        
        plotNumList, empList = parse_data_from_emp_csv(in_path, convt)
    
        for plotNum, emp in zip(plotNumList, empList):
            if plotNum == 0:
                continue
            saveFlag = True
            onePlotList = boxList[plotNum-1]
            for savedItem in onePlotList:
                center_dist = math.sqrt((savedItem[0]-emp[0])**2+(savedItem[1]-emp[1])**2)
                radiusAdd = savedItem[2]+emp[2]
                if center_dist < radiusAdd:
                    saveFlag = False
            
            if saveFlag:
                boxList[plotNum-1].append(emp)
                
    
    # save details
    fields = ('plotNum', 'range', 'column', 'X', 'Y', 'Radius')
    all_emp_file = os.path.join(out_dir, 'empDetail.csv')
    csvHandle = open(all_emp_file, 'w')
    csvHandle.write(','.join(map(str, fields)) + '\n')
    for i in range(len(boxList)):
        plotNum = i+1
        plot_row, plot_col = convt.plotNum_to_fieldPartition(plotNum)
        tarList = boxList[i]
        for j in range(len(tarList)):
            saveItem = tarList[j]
            print_line = '{},{},{},{},{},{}\n'.format(plotNum, plot_row, plot_col, saveItem[0], saveItem[1], saveItem[2])
            csvHandle.write(print_line)
            
    csvHandle.close()
    
    # save count per plot
    fields = ('plotNum', 'range', 'column', 'emergence_count')
    count_file = os.path.join(out_dir, 'cnn_emergence_count.csv')
    csvHandle = open(count_file, 'w')
    csvHandle.write(','.join(map(str, fields)) + '\n')
    for i in range(len(boxList)):
        plotNum = i+1
        plot_row, plot_col = convt.plotNum_to_fieldPartition(plotNum)
        plot_count = len(boxList[i])
        print_line = '{},{},{},{}\n'.format(plotNum, plot_row, plot_col, plot_count)
        csvHandle.write(print_line)
        
    csvHandle.close()
    
    # save emp data in bety form
    save_data_to_bety_form(boxList, bety_dir, str_date, seasonNum, convt)
    
    return

def save_data_to_bety_form(emp_count, csv_dir, str_date, seasonNum, convt):
    
    out_file = os.path.join(csv_dir, '{}_emergenceCount.csv'.format(str_date))
    csv = open(out_file, 'w')
    
    (fields, traits) = get_traits_table_emp()
    
    csv.write(','.join(map(str, fields)) + '\n')
    
    for i in range(len(emp_count)):
        plotNum = i+1
        
        plot_count = len(emp_count[i])
        if plot_count == 0:
            continue
        
        str_time = str_date+'T12:00:00'
        traits['local_datetime'] = str_time
        traits['emergence_counting'] = plot_count
        traits['site'] = parse_site_from_plotNum(plotNum, seasonNum, convt)
        trait_list = generate_traits_list_emp(traits)
        csv.write(','.join(map(str, trait_list)) + '\n')
    
    csv.close()
    
    return

def parse_site_from_plotNum(plotNum, seasonNum, convt):
    
    plot_row, plot_col = convt.plotNum_to_fieldPartition(plotNum)
    
    rel = 'MAC Field Scanner Season {} Range {} Column {}'.format(str(seasonNum), str(plot_row), str(plot_col))
    
    return rel

def get_traits_table_emp():
    
    fields = ('local_datetime', 'access_level', 'species', 'site',
        'citation_author', 'citation_year', 'citation_title', 'method', 'emergence_counting')
    
    traits = {'local_datetime' : '',
              'access_level' : '2',
              'species' : 'Sorghum bicolor',
              'site': [],
              'citation_author': 'ZongyangLi',
              'citation_year' : '2020',
              'citation_title' : 'Maricopa Field Station Data and Metadata',
              'method' : 'Stereo RGB data to emergence counting',
              'emergence_counting' : []
        }
    
    return (fields, traits)

def generate_traits_list_emp(traits):
    
    trait_list = [  traits['local_datetime'],
                    traits['access_level'],
                    traits['species'],
                    traits['site'],
                    traits['citation_author'],
                    traits['citation_year'],
                    traits['citation_title'],
                    traits['method'],
                    traits['emergence_counting']
                  ]
    
    return trait_list
    

def parse_data_from_emp_csv(in_dir, convt):
    
    plotNumList = []
    empList = []
    image_shape = (3296, 2472)
    # find csv file and json file
    csv_suffix = os.path.join(in_dir, '*.csv')    
    csvs = glob(csv_suffix)
    json_suffix = os.path.join(in_dir, '*_metadata.json')    
    jsons = glob(json_suffix)
    if len(jsons) == 0 or len(csvs)==0:
        return plotNumList, empList
    
    # load csv file to roi list
    with open(csvs[0], 'r') as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',')
        roiList = map(tuple, csvReader)
    
    # roi list to emp list and plotNum list
    metadata = lower_keys(load_json(jsons[0]))
    center_position = parse_metadata(metadata)
    fov = get_fov(metadata, center_position[2])
    
    # make fov y bigger to fit
    fov_adj = 1.04
    fov[0] = fov[0]*fov_adj
    fov[1] = fov[1]*fov_adj
    
    for strItem in roiList:
        roiItem = [float(i) for i in strItem]
        roiCx = (roiItem[2]+roiItem[0])/2
        roiCy = (roiItem[3]+roiItem[1])/2
        
        emp_position = pixel_to_field_position(center_position, fov, image_shape, (roiCy, roiCx))
        r = box_to_radius(fov, image_shape, roiItem)
        plot_row, plot_col = convt.fieldPosition_to_fieldPartition(emp_position[0], emp_position[1])
        plotNum = convt.fieldPartition_to_plotNum(plot_row, plot_col)
        plotNumList.append(plotNum)
        empList.append([emp_position[0], emp_position[1], r])
    
    return plotNumList, empList

def parse_metadata(metadata):
    
    try:
        gantry_meta = metadata['lemnatec_measurement_metadata']['gantry_system_variable_metadata']
        gantry_x = gantry_meta["position x [m]"]
        gantry_y = gantry_meta["position y [m]"]
        gantry_z = gantry_meta["position z [m]"]
        
        cam_meta = metadata['lemnatec_measurement_metadata']['sensor_fixed_metadata']
        cam_x = cam_meta["location in camera box x [m]"]
        cam_y = cam_meta["location in camera box y [m]"]

        
        if "location in camera box z [m]" in cam_meta: # this may not be in older data
            cam_z = cam_meta["location in camera box z [m]"]
        else:
            cam_z = 0

    except KeyError as err:
        fail('Metadata file missing key: ' + err.args[0])
        
    position = [float(gantry_x), float(gantry_y), float(gantry_z)]
    center_position = [position[0]+float(cam_x), position[1]+float(cam_y), position[2]+float(cam_z)]
    
    return center_position

def get_fov(metadata, camHeight):
    try:
        cam_meta = metadata['lemnatec_measurement_metadata']['sensor_fixed_metadata']
        fov = cam_meta["field of view at 2m in x- y- direction [m]"]
    except KeyError as err:
        fail('Metadata file missing key: ' + err.args[0])

    try:
        fov_list = fov.replace("[","").replace("]","").split()
        fov_x = float(fov_list[0])
        fov_y = float(fov_list[1])

        # given fov is at 2m, so need to convert for actual height
        fov_x = (camHeight * (fov_x))/2
        fov_y = (camHeight * (fov_y))/2

    except ValueError as err:
        fail('Corrupt FOV inputs, ' + err.args[0])
    return [fov_x, fov_y]

def pixel_to_field_position(center_position, fov, imgSize, pixel):
    
    s_y0 = center_position[1]-fov[1]/2
    s_x0 = center_position[0]-fov[0]/2
    
    px = fov[0]/imgSize[0]
    py = fov[1]/imgSize[1]
    
    x = s_x0 + (imgSize[0]-pixel[0])*px
    y = s_y0 + (imgSize[1]-pixel[1])*py
    
    x0 = s_x0 - pixel[0]*px
    y0 = s_y0 - pixel[0]*py
    
    return (x, y)

def box_to_radius(fov, imgSize, roi_box):
    
    py = fov[1]/imgSize[1]
    px = fov[0]/imgSize[0]
    
    y_r = (roi_box[2] - roi_box[0])*py
    x_r = (roi_box[3] - roi_box[1])*px
    
    box_radius = math.sqrt(x_r*x_r+y_r*y_r)/2
    
    return box_radius
    

def singe_image_process(in_dir, out_dir, detector):
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        
    metas, ims_left, ims_right = find_input_files(in_dir)
        
    json_path = os.path.join(in_dir, metas[0])
    json_basename = os.path.basename(metas[0])
    json_dst = os.path.join(out_dir, json_basename)
    if os.path.exists(json_dst):
        return
    
    # bin to image
    image_shape = (3296, 2472)
    
    base_name = os.path.basename(ims_left[0])[:-4]
    
    raw_data_to_image(ims_left[0], image_shape, in_dir, out_dir, base_name, 'left')
    
    input_image_path = os.path.join(out_dir, '{}.jpg'.format(base_name))
    # extract panicle bounding boxes
    input_image = cv2.imread(input_image_path)
    
    x_inds, y_inds, img_vec = crop_reflectance_image(input_image)
    
    saved_boxes = []
    ind = 0
    for x_ind, y_ind, img in zip(x_inds, y_inds, img_vec):
        ind += 1
        if ind > 29:
            break
        dets, scores, im2show = object_detection(detector, img, 0.8)
        if len(dets)==0:
            continue

        new_boxes = remap_box_coordinates(x_ind, y_ind, dets)
        for box in new_boxes:
            saved_boxes.append(box)
        
    init_boxes = box_integrate(saved_boxes)
    
    save_box_image(input_image, init_boxes, out_dir, base_name)
    
    # save boxes position to out_dir
    out_csv_path = os.path.join(out_dir, '{}.csv'.format(base_name[:-5]))
    with open(out_csv_path, 'w') as f:
        for each_box in init_boxes:
            print_line = ','.join(map(str,each_box))
            f.write(print_line+'\n')
    
    shutil.copyfile(json_path, json_dst)
    
    return

def save_box_image(gImage, merged_boxes, out_dir, base_name):
    
    im2show = np.copy(gImage)
    for box in merged_boxes:
        box = [int(i) for i in box]
        cv2.rectangle(im2show, (box[0],box[1]), (box[2],box[3]), (255, 205, 51), 3)
        
    cv2.imwrite(os.path.join(out_dir, 'labeled_'+base_name+'.jpg'), im2show)
    
    return

def box_integrate(all_boxes):
    
    new_boxes = []
    
    deled_ind = []
    
    for i in range(len(all_boxes)):
        if i in deled_ind:
            continue
        curr_box = all_boxes[i]
        for j in range(len(all_boxes)):
            if j in deled_ind:
                continue
            if i == j:
                continue
            
            pair_box = all_boxes[j]
            overlapping_ratio = compute_box_overlap(curr_box, pair_box)
            if overlapping_ratio > 0.3:
                deled_ind.append(j)
                add_box = combine_boxes(curr_box, pair_box)
                if max(add_box[2]-add_box[0], add_box[3]-add_box[1])>80:
                    continue
                else:
                    curr_box = add_box
                break
            
        new_boxes.append(curr_box)
    
    return new_boxes

def combine_boxes(box, pair_box):
    
    box[0] = min(box[0], pair_box[0])
    box[1] = min(box[1], pair_box[1])
    box[2] = max(box[2], pair_box[2])
    box[3] = max(box[3], pair_box[3])
    
    return box

def compute_box_overlap(box, box_pair):
    
    si = max(0, min(box[2], box_pair[2])-max(box[0], box_pair[0]))*max(0, min(box[3], box_pair[3])-max(box[1], box_pair[1]))
    
    sa = (box[2]-box[0])*(box[3]-box[1])
    sb = (box_pair[2]-box_pair[0])*(box_pair[3]-box_pair[1])
    if sa < sb:
        ret = si/sa
    else:
        ret = si/sb
    
    return max(0, ret)

def object_detection(detector, image, score_threshold=0.9):
    
    dets, scores, classes = detector.detect(image, score_threshold)
        
    im2show = np.copy(image)
    for i, det in enumerate(dets):
        det = tuple(int(x) for x in det)
        cv2.rectangle(im2show, det[0:2], det[2:4], (255, 205, 51), 2)
        cv2.putText(im2show, '%s: %.3f-%d' % (classes[i], scores[i], i), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness=2)
    
    return dets, scores, im2show

def remap_box_coordinates(fileIndX, fileIndY, boxes, out_img_size=(600, 600)):
    
    x_offset = fileIndX*out_img_size[0]
    y_offset = fileIndY*out_img_size[1]
    
    new_boxes = []
    
    for box in boxes:
        new_box = [box[0]+y_offset, box[1]+x_offset, box[2]+y_offset, box[3]+x_offset]
        new_boxes.append(new_box)
    
    return new_boxes

def crop_reflectance_image(img, out_img_size=(600,600)):
    
    width, height, channels = img.shape
    
    i_wid_max = int(round(width/out_img_size[0]))+1
    i_hei_max = int(round(height/out_img_size[1]))+1
    
    x_ind = []
    y_ind = []
    img_vec = []
            
            
    for i in range(i_wid_max):
        for j in range(i_hei_max):
            cropped_img = img[i*out_img_size[1]:(i+1)*out_img_size[1], j*out_img_size[0]:(j+1)*out_img_size[0]]
            #img_path = os.path.join(out_dir, base_name+'_'+str(i)+'_'+str(j)+'.jpg')
            #cv2.imwrite(img_path, crop_img)
            x_ind.append(i)
            y_ind.append(j)
            img_vec.append(cropped_img)
    
    return x_ind, y_ind, img_vec


def raw_data_to_image(im_name, shape, in_dir, out_dir, path_name, side):
    
    baseName = im_name[:-4]
    try:
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        im = np.fromfile(os.path.join(in_dir, im_name), dtype='uint8').reshape(shape[::-1])
        im_color = demosaic(im)
        im_color = (np.rot90(im_color))
        out_path = os.path.join(out_dir, '{}.jpg'.format(baseName))
        Image.fromarray(im_color).save(out_path)
        return im_color
    except Exception as ex:
        fail('Error processing image "%s": %s' % (im_name, str(ex))) 
        
def demosaic(im):
    # Assuming GBRG ordering.
    B = np.zeros_like(im)
    R = np.zeros_like(im)
    G = np.zeros_like(im)
    R[0::2, 1::2] = im[0::2, 1::2]
    B[1::2, 0::2] = im[1::2, 0::2]
    G[0::2, 0::2] = im[0::2, 0::2]
    G[1::2, 1::2] = im[1::2, 1::2]

    fG = np.asarray(
            [[0, 1, 0],
             [1, 4, 1],
             [0, 1, 0]]) / 4.0
    fRB = np.asarray(
            [[1, 2, 1],
             [2, 4, 2],
             [1, 2, 1]]) / 4.0

    im_color = np.zeros(im.shape+(3,), dtype='uint8') #RGB
    im_color[:, :, 0] = convolve(R, fRB)
    im_color[:, :, 1] = convolve(G, fG)
    im_color[:, :, 2] = convolve(B, fRB)
    return im_color


def find_input_files(in_dir):
    metadata_suffix = '_metadata.json'
    metas = [os.path.basename(meta) for meta in glob(os.path.join(in_dir, '*' + metadata_suffix))]
    if len(metas) == 0:
        fail('No metadata file found in input directory.')

    guids = [meta[:-len(metadata_suffix)] for meta in metas]
    ims_left = [guid + '_left.bin' for guid in guids]
    ims_right = [guid + '_right.bin' for guid in guids]

    return metas, ims_left, ims_right

def load_json(meta_path):
    try:
        with open(meta_path, 'r') as fin:
            return json.load(fin)
    except Exception as ex:
        fail('Corrupt metadata file, ' + str(ex))
    
    
def lower_keys(in_dict):
    if type(in_dict) is dict:
        out_dict = {}
        for key, item in in_dict.items():
            out_dict[key.lower()] = lower_keys(item)
        return out_dict
    elif type(in_dict) is list:
        return [lower_keys(obj) for obj in in_dict]
    else:
        return in_dict

def fail(reason):
    print >> sys.stderr, reason

def load_model(model_file_path):
    
    detector = FasterRCNN()
    network.load_net(model_file_path, detector)
    detector.cuda()
    detector.eval()
    print('load model successfully!')
    
    return detector

if __name__ == '__main__':
    
    #main()
    test()
    
    
    
    
    
    
    
    
    
    
    
    