import os
import glob
import sys

def txt_checker(feature_dir, txt_dir):

    numpy_files = glob.glob(os.path.join(feature_dir, '*npy'))
    txt_files = glob.glob(os.path.join(txt_dir, '*txt'))

    for numpy_file in numpy_files:
        numpy_filename = numpy_file.split('/')[-1].split('.')[0]
        if any(numpy_filename in txt_filename for txt_filename in txt_files):
            pass

        else:
            os.remove(numpy_file)
            print (numpy_file + " has been deleted.")



if __name__ == "__main__":
    txt_checker(sys.argv[1], sys.argv[2])

