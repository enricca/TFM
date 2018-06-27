"""
Quickfix solution for x
"""
import pandas
import numpy as np
import argparse
import os

def get_label_from_roi(mask, x, y, d):
    """
    Gets the label from a a mask
    """
    d = int(d)
    return np.max(mask[x-d/2:x +d/2, y -d/2: y + d/2])

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Load the labels in acsv.')
    parser.add_argument('-input_csv')
    parser.add_argument('-output_csv')
    parser.add_argument('-npz_dir')

    parser.add_argument('-channel', default = '3', help = 'Channel to load the label from. Default (3 = malignancy)')

    args = parser.parse_args()

    df = pandas.read_csv(args.input_csv)
    df['label'] = -1
    for p in set(df.patientid):
        print p
        data = np.load(os.path.join(args.npz_dir, p + '.npz'))['arr_0']
        for i,row in df[df['patientid'] == p].iterrows():
            df.iloc[i].label = get_label_from_roi(data[int(args.channel), row.nslice,:,:], row.x, row.y, row.diameter)
        
    df.to_csv(args.output_csv)