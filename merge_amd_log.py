#!/usr/bin/env python

import argparse
import os.path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Merge AMD logfiles',
                                     description='Merges multiple logfiles and generates a .dat file for reweighting.')
    parser.add_argument("-b", "--base_name", required=True, help='Common part of the name of files to be merged')
    parser.add_argument("-o", "--output", required=True, help='Output base name')
    parser.add_argument("-s", "--skip", default='0:1000000000:1', help='first:last:skip (python style numbering)')
    parser.add_argument("-t", "--temperature", type=float, default=300.0, help='Temperature in Kelvin')
    args = parser.parse_args()
    directory, basein = os.path.split(args.base_name)
    start, stop, step = args.skip.split(':')
    filelist = os.listdir(directory)
    if start == '':
        start = 0
    else:
        start = int(start)
    if stop == '':
        stop = 1000000000
    else:
        stop = int(stop)
    if step == '':
        step = 1
    else:
        step = int(step)
    logs = sorted([log for log in filelist if '.log' in log and basein in log])
    with open(os.path.join(directory, logs[0]), 'rt') as inp:
        lines = inp.readlines()
        header = lines[:3]
        fr_step = int(lines[4].split()[0])
    logfile = open(args.output + '.log', 'wt')
    for line in header:
        logfile.write(line)
    datfile = open(args.output + '.dat', 'wt')
    count = -1
    for log in logs:
        with open(os.path.join(directory, log), 'rt') as inp:
            lines = inp.readlines()
            for line in lines:
                if '#' in line:
                    continue
                count += 1
                if count % step != 0:
                    continue
                if count < start:
                    continue
                if count >= stop:
                    continue
                values = line.split()[2:]
                v = [float(value) for value in values]
                tot = (count + 1) * fr_step
                l1 = f'{fr_step:10}{tot:12}{v[0]:22.12}{v[1]:22.12}{v[2]:22.12}{v[3]:22.12}{v[4]:22.12}{v[5]:22.12}\n'
                logfile.write(l1)
                en = v[4] + v[5]
                en_kt = en / (0.001987 * args.temperature)
                l2 = f'{en_kt} {tot} {en}\n'
                datfile.write(l2)
    datfile.close()
    logfile.close()
