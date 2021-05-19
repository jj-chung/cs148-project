import os 
import json

def create_curr_to_prev():
    '''
    Creates a json file containing a dictionary of current-frame filenames
    to previous-frame filenamm
    '''
    img_path = './Fish_automated_identification_and_counting/'
    box_path = './box_masks/'
    gauss_path = './gaussian_masks/'

    train_path = './luderick_seagrass_jack_evans_a.json'
    val_path = './luderick_seagrass_jack_evans_b.json'
    all_path = './luderick_seagrass_all.json'
    
    paths = [all_path]
    
    # dictionary from filenames to filenames, dict from img_ids to img_ids
    curr_to_prev_filename = {}
    curr_to_prev_img_id = {}

    # using each path for annotations, create ids to filenames dict
    for ann_path in paths:
        with open(img_path + ann_path) as f:
            # dictionary containing COCO format annotations
            COCO_ann = f.read()

        COCO_ann = json.loads(COCO_ann)

        # Lists of dictionaries with image/annotation information
        img_info = COCO_ann["images"]

        # For the first image, there is clearly no previous frame
        curr_to_prev_filename[img_info[0]["file_name"]] = None
        curr_to_prev_img_id[img_info[0]["id"]] = None
        num_no_prev = 0
            
        for i in range(1, len(img_info)):
            curr_img_dict = img_info[i]
            prev_img_dict = img_info[i - 1]
            
            curr_img_id, curr_img_name = curr_img_dict["id"], curr_img_dict["file_name"]
            prev_img_id, prev_img_name = prev_img_dict["id"], prev_img_dict["file_name"]
            
            # Check if the timestamps are spaced correctly -- if not, assume
            # first frame in new clip 
            sep_curr = curr_img_name.split('.')
            sep_prev = prev_img_name.split('.')
            
            curr_six_digits = int(sep_curr[len(sep_curr) - 2][-6:])
            prev_six_digits = int(sep_prev[len(sep_prev) - 2][-6:])
            
            if curr_six_digits - prev_six_digits not in [200, 1]:
                curr_to_prev_filename[curr_img_name] = None
                curr_to_prev_img_id[curr_img_name] = None
                num_no_prev += 1
            else:
                curr_to_prev_filename[curr_img_name] = prev_img_name
                curr_to_prev_img_id[curr_img_id] = prev_img_id
                
    # Save dictionaries as .json files
    with open("curr_to_prev_filename.json", "w") as write_file:
        json.dump(curr_to_prev_filename, write_file, indent=4)
        
    with open("curr_to_prev_img_id.json", "w") as write_file:
        json.dump(curr_to_prev_img_id, write_file, indent=4)
        
    print(num_no_prev)
            
def main():
	create_curr_to_prev()
	
if __name__ == '__main__':
	main()