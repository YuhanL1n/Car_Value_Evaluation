import csv

csv_head = ['make and model', 'year', 'VIN', 'condition', 'cylinders', 'drive', 'fuel', 'color', 'odometer', 'size', 'title', 'transmission', 'type', 'price']

dict = {}

ls = []

def cal(filename):

    sum = 0
    count = 0
    cnt = 1
    with open(filename, 'r') as f:
        reader = csv.reader(f)

        for row in reader:
            ls.append(row)

        for r in ls[1:]:
            if cnt == 1:
                cnt += 1
                pass
            if r[8] == 'None':
                r[8] = 0
            else:
                r[8] = int(r[8])
        for r in ls[1:]:
            if r[8] != 0 :
                year = 2017 - int(r[1])
                if year != 0:
                    avg = r[8]/year
                    sum += avg
                    count += 1
        f.close()
        print(sum/count)

cal('all.csv')
