data = [0,0.1, 3, 3.1, 3.2, 5, 5.5, 5.6, 5.7, 10]
clust = {}
index = 0
clust_threshold = 0.2
for i in range(len(data)):
    if i == 0:
        clust[index] = [data[i]]
    else:
        dis = data[i] - data[i - 1]
        if dis <= clust_threshold:
            clust[index].append(data[i])
        else:
            index += 1
            clust[index] = [data[i]]

print(clust)


