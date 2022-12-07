import numpy as np
import matplotlib.pyplot as plt
import random

def sum_squared_distance(a,b):
    return(sum((a[1:]-b[1:])**2))

def leave_one_out_accuracy(data,features,feature_to_add):
    data_copy = data.copy()
    remove = []
    for i in range(len(data_copy[0])):
        if i not in features and i != feature_to_add and i != 0:
            remove.append(i)
    data_copy[:,remove] = 0

    for i in range(len(data_copy[0])):
        pass
    score = 0
    for i,object_to_classify in enumerate(data_copy):
        label = object_to_classify[0]
        nn_label = 0
        nn_dist = np.inf
        nn_loc = np.inf
        for k,object_to_compare in enumerate(data_copy):
            if (i != k):
                dist = sum_squared_distance(object_to_classify, object_to_compare)
                if (dist < nn_dist):
                    nn_dist = dist
                    nn_loc = k
                    nn_label = object_to_compare[0]
        if nn_label == label:
            score+=1
    return (score / len(data))

def leave_one_out_accuracy_TEST(data,current_set,feature_to_add):
    return (random.random())

def feature_search(data):
    features = []
    
    for i in range(1,len(data[0])):
        print(f"On level {i} of the tree")
        feature_to_add = 0
        best_so_far_accuracy = 0


        for k in range(1,len(data[0])):
            if k not in features:
                print(f"Considering feature {k}")
                accuracy = leave_one_out_accuracy(data, features, k)

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add = k
        features.append(feature_to_add)
        print(f"On level {i} feature {feature_to_add} was added to set of features")

    return features
    


arr = np.loadtxt("CS170_Small_Data__96.txt")
features = feature_search(arr)
print(f"Features chosen: {features}")
