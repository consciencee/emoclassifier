import model.params as params

def loadTimings(csv_name, ignoredIDs = ()):
    csv_file = open(csv_name, "r")

    if(not csv_file):
        print("error loading file")
        return

    contents = csv_file.readlines()
    dataset = []

    for line in contents:
        line = line[:-2]
        if line == "":
            continue
        if line in params.emotionLabels:
            continue
        if line[0:7] == "Session":
            print(line)
            currentIgnore = line in ignoredIDs
            continue

        if not currentIgnore:
            lineSplit = line.split(",")
            # lineSplit[0] is time
            dataset.append(float(lineSplit[0]))

    return dataset

def countSamplingsRate():
    millis = loadTimings("../samples/Alexey/2/Alexey5_2_eeg_log.csv")

    begin1 = millis[0]
    end = begin1 + 1000
    itemsPerSec = 0
    sec = 1
    avg = 0

    #offset = 742


    for i in range(0,1000):
        begin = begin1 + i
        end = begin + 1000
        res = []
        for item in millis:
            if item < begin:
                continue
            if item < end:
                itemsPerSec += 1
                avg += 1
            else:
                res.append(itemsPerSec)
                end += 1000
                sec += 1
                itemsPerSec = 1

        #res.append(itemsPerSec)
        print(i, " ", res)
