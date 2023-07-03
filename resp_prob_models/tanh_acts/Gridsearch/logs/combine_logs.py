import sys, os
import glob
import pandas as pd
import time

#combine all the logs for easier analysis
os.chdir(sys.path[0])
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
timestr = time.strftime("%m%d-%H%M")

combined_csv.to_csv( f'gridsearch_{timestr}.csv', index=False, encoding='utf-8-sig')