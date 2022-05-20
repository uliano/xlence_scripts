#!/usr/bin/env python

import os.path
import argparse
import pandas as pd


def main(directory, outfile):

    lista = os.listdir(directory)
    if directory is None:
        directory = ''

    files = []
    for f in lista:
        _, ext = os.path.splitext(f)
        if ext == '.mdout':
            files.append(f)

    files.sort()

    all_lines = []

    for f in files:
        with open(os.path.join(directory, f), 'rt') as textfile:
            lines = textfile.readlines()
        lines = [l.strip() for l in lines]

        stato = 'fuori'
        my_lines = []
        for line in lines:
            if stato == 'fuori':
                if line[:11] == '4.  RESULTS':
                    stato = 'dentro'
                continue
            elif stato == 'dentro':
                if line[:15] == 'A V E R A G E S':
                    stato = 'fuori'
                    continue
                if line == '':
                    continue
                if line[:3] == '---':
                    continue
                if 'wrapping' in line:  # some random line here and there started with this
                    continue
                my_lines.append(line)
            else:
                print('non deve succedere')
                exit(0)
        all_lines += my_lines

    results = dict()
    for line in all_lines:
        the_line = line.replace('1-4 NB', '1-4_NB').replace('1-4 EEL', '1-4_EEL')
        elements = the_line.split()
        while len(elements) > 0:
            label, _, value, *elements = elements
            if label in results:
                results[label].append(float(value))
            else:
                results[label] = [float(value)]

    results['TIME(PS)'] = [element - results['TIME(PS)'][0] for element in results['TIME(PS)']]
    df = pd.DataFrame(results)
    df.to_csv(outfile, index=False)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", help="directory with all the '.mdout' file")
    parser.add_argument("-o", "--output", help="output csv file", default='mdout.csv')
    args = parser.parse_args()
    main(args.directory, args.output)