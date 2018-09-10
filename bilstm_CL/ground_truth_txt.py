import sys
import numpy as np

def ground_truth_txt(mdtm_file):
    """Create ground truth txt for AMI Corpus .mdtm files. 
    It will save the txt file in the txt_files folder.
    
    Arguments:
    mdtm_file: Which mdtm file will be considered."""

    change_time_array = []
    
    with open(main_set) as f:
        content = f.readlines()
    content = [x.strip() for x in content] 
    tmp_filename = content[0].split(' ')[0]

    for single_line in content:
        filename = single_line.split(' ')[0]

        if (filename != tmp_filename):

            np.savetxt(fname="./txt_files/" + tmp_filename + "_change_time.txt", X=change_time_array, delimiter=' ', fmt='%1.3f')
            change_time_array = []
            
        tmp_filename = single_line.split(' ')[0]
        offset = float(single_line.split(' ')[2])
        duration = float(single_line.split(' ')[3])
        end_time = offset+duration

        change_time_array.append(offset)
        change_time_array.append(end_time)


if __name__ == "__main__":
    ground_truth_txt(sys.argv[1])
