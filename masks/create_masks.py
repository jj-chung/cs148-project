# import cv2
import os
import numpy as np
import json
from PIL import Image, ImageDraw
import random

def create_box_masks(img_id, img_name, img_path, visited):
    '''
    Create box masks given bounding boxes.
    '''
    # make new image with black where there is no bounding box and white 
    # where there is,
    # save resulting image
    
    # Note: COCO annotation bboxes have form
    # [xtl, ytl, width, height]
    bbox = ann_dict["bbox"]
    get_path = os.path.join(img_path + '/luderick-seagrass/', img_name)
    
    # If the image has already been drawn on, add another box to that
    # existing image
    if img_id in visited:
        # If the box has already been visited, we can use the same get
        # and save image names
        img_name_sep = img_name.split('.')
        get_path = os.path.join(img_path + 'box_masks/', img_name_sep[0] + '.' 
                               + img_name_sep[1] + '_box_mask.jpg')
        f_name = (img_path + box_path + img_name_sep[0] + '.' 
                               + img_name_sep[1] + '_box_mask.jpg')
        
        
    with Image.open(get_path) as im:
        draw = ImageDraw.Draw(im)
        
        width = 1920
        height = 1080
        
        # If not visited before, black out image first, and update
        # the save name
        if img_id not in visited:
            draw.rectangle([0, 0, width + 1, height + 1], outline='black',
                          fill='black', width=1)
            f_name = img_path + box_path + ''.join(img_name.split('.')[0] + '.' 
                                               + img_name.split('.')[1]
                                                + '_box_mask.jpg')

        box_width = bbox[2]
        box_height = bbox[3]
        
        # For ImageDraw.rectangle(xy):
        # xy - Two points to define the bounding box. Sequence of either 
        # [(x0, y0), (x1, y1)] or [x0, y0, x1, y1]. The second point is just 
        # outside the drawn rectangle.
        
        bbox = [bbox[0], bbox[1], bbox[0] + box_width, bbox[1] + box_height]
        
        # Draw a white rectangle on top of the image
        draw.rectangle(bbox, outline='white', fill='white', width=1)
        
        visited.add(img_id)
        
        # Uncomment these lines if want to save the masks as images
        #im = im.convert('RGB')
        #im.save(f_name, 'JPEG')
        np_arr = np.asarray(im)
        
        return np.shape(np_arr[:, :, 0])
        
 
def two_dim_gaussian(pt, mu, sigma):
    '''
    Given pt = (x, y), mu = (mu_x, mu_y), and sigma = (sigma_x, sigma_y), 
    compute value of 2D Gaussian with parameters
    mu and sigma at (x, y).
    '''
    x, y = pt
    mu_x, mu_y = mu
    sig_x, sig_y = sigma
    
    # Setting amplitude for RGB
    amp = 255
    inner_term = (((x - mu_x) ** 2) / (2 * sig_x ** 2) 
                  + ((y - mu_y) ** 2) / (2 * sig_y ** 2))
    
    return amp * np.exp(-1 * inner_term)


def create_gaussian_masks(img_id, img_name, img_path, visited):
    '''
    Create gaussian masks given bounding boxes.
    Follows same format as create_box_masks, but applies ellipse with
    center at box's center.
    '''
    # make new image with black where there is no bounding box and white 
    # where there is,
    # save resulting image
    
    # Note: COCO annotation bboxes have form
    # [xtl, ytl, width, height]
    bbox = ann_dict["bbox"]
    get_path = os.path.join(img_path + '/luderick-seagrass/', img_name)
    
    # If the image has already been drawn on, add another box to that
    # existing image
    if img_id in visited:
        # If the box has already been visited, we can use the same get
        # and save image names
        img_name_sep = img_name.split('.')
        get_path = os.path.join(img_path + 'gaussian_masks/', img_name_sep[0] + '.' 
                               + img_name_sep[1] + '_gauss_mask.jpg')
        f_name = (img_path + gauss_path + img_name_sep[0] + '.' 
                               + img_name_sep[1] + '_gauss_mask.jpg')
        
        
    with Image.open(get_path) as im:
        draw = ImageDraw.Draw(im)
        
        width = 1920
        height = 1080
        
        # If not visited before, black out image first, and update
        # the save name
        if img_id not in visited:
            draw.rectangle([0, 0, width + 1, height + 1], outline='black',
                          fill='black', width=1)
            f_name = img_path + gauss_path + ''.join(img_name.split('.')[0] + '.' 
                                               + img_name.split('.')[1]
                                                + '_gauss_mask.jpg')

        box_width = bbox[2]
        box_height = bbox[3]
        
        # For ImageDraw.rectangle(xy):
        # xy - Two points to define the bounding box. Sequence of either 
        # [(x0, y0), (x1, y1)] or [x0, y0, x1, y1]. The second point is just 
        # outside the drawn rectangle.
        
        bbox = [bbox[0], bbox[1], bbox[0] + box_width, bbox[1] + box_height]
        
        # Draw a white/grey/black ellipse with gradation based on the
        # 2D Gaussian (calling random.gauss b/c independent axes).
        
        # Let the 2D mean (point) be the center of the box.
        mu_x = bbox[0] + box_width / 2
        mu_y = bbox[1] + box_height / 2
        
        # Let sigma_x and sigma_y, i.e. the standard deviations in each
        # direction, be one sixth of the length of the box in that dimension.
        # This way, the function will disappear approximately at the edges
        # of the box (center + 3x the standard deviation for that direction).
        sigma_x = box_width / 6
        sigma_y = box_height / 6
        
        # For each coordinate in the box, determine it's black/white color
        for x_coord in range(bbox[0], bbox[0] + box_width):
            for y_coord in range(bbox[1], bbox[1] + box_height):
                pt = (x_coord, y_coord)
                mu = (mu_x, mu_y)
                sigma = (sigma_x, sigma_y)
                c_val = int(two_dim_gaussian(pt, mu, sigma))
                draw.point([x_coord, y_coord], fill=(c_val, c_val, c_val))
        
        visited.add(img_id)
        # im = im.convert('RGB')
        # im.save(f_name, 'JPEG')
        
        np_arr = np.asarray(im)
        return np.shape(np_arr[:, :, 0])
        
def main():
	img_path = './Fish_automated_identification_and_counting/'
	box_path = './box_masks/'
	gauss_path = './gaussian_masks/'

	train_path = './luderick_seagrass_jack_evans_a.json'
	val_path = './luderick_seagrass_jack_evans_b.json'
	all_path = './luderick_seagrass_all.json'

	paths = [all_path]

	# Set of visited images (drawn on)
	visited = set()

	# dictionary from image ids to image filenames
	ids_to_names = {}
	    
	# using each path for annotations, create ids to filenames dict
	for ann_path in paths:
	    with open(img_path + ann_path) as f:
	        # dictionary containing COCO format annotations
	        COCO_ann = f.read()

	    COCO_ann = json.loads(COCO_ann)
	    
	    # Lists of dictionaries with image/annotation information
	    img_info = COCO_ann["images"]
	    annotations = COCO_ann["annotations"]

	    for img_dict in img_info:
	        img_id = img_dict["id"]
	        img_name = img_dict["file_name"]
	        ids_to_names[img_id] = img_name

	    # For each annotation, get id, filename, and create a black-white 
	    # and gaussian mask.
	    for ann_dict in annotations:
	        img_id = ann_dict["image_id"]

	        if img_id in ids_to_names:
	            img_name = ids_to_names[img_id]
	            create_box_masks(img_id, img_name, img_path, visited)
	            # create_gaussian_masks(img_id, img_name, img_path, visited)
	        else:
	            print(img_id)

if __name__ == '__main__':
	main()
