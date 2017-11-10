import csv
import collections

def most_frequent_model(dataFile):
    #print the most frequent make-model pairs in dataset
    models=[]
    f = open(dataFile)
    next(f) # ignore first line
    reader = csv.reader(f)
    counts=0
    for row in reader:
        models.append((row[14],row[10]))
        counts+=1
        if counts%10000==0:
            print counts

    counter=collections.Counter(models)
    print counter.most_common(10)

if __name__ == "__main__":
    #most_frequent_model('data/autos.csv')