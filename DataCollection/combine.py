"""
Reference: 
http://blog.csdn.net/bytxl/article/details/23372405
"""

import csv
import os
allFileNum = 0

csv_head = ['make and model', 'year', 'VIN', 'condition', 'cylinders', 'drive', 'fuel', 'color', 'odometer', 'size', 'title', 'transmission', 'type', 'price']


def printPath(level, path):
    global allFileNum

    dirList = []
    fileList = []
    files = os.listdir(path)
    dirList.append(str(level))
    for f in files:
        if(os.path.isdir(path + '/' + f)):
            # hidden folder will not be checked
            if(f[0] == '.'):
                pass
            else:
                dirList.append(f)
        if(os.path.isfile(path + '/' + f)):
            if(f[0] == '.'):
                pass
            else:
                fileList.append(f)

    i_dl = 0
    for dl in dirList:
        if(i_dl == 0):
            i_dl = i_dl + 1
        else:
            print ('-' * (int(dirList[0])), dl)
            printPath((int(dirList[0]) + 1), path + '/' + dl)

    for fl in fileList:

        print ('-' * (int(dirList[0])), fl)
        allFileNum = allFileNum + 1
        flList = []
        with open(path + '/'  + fl, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
               flList.append(row)
        f.close()
        with open("./all.csv", 'a+') as ff:
            writer = csv.writer(ff)
            writer.writerows(flList[1:])
        ff.close()

if __name__ == '__main__':

    with open("./all.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(csv_head)
    f.close()

    printPath(1, './Data')
    print ('total files =', allFileNum)
