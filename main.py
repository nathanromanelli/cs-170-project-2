import numpy as np
import time
import matplotlib.pyplot as plt

def sum_squared_distance(a,b):
    return(sum((a[1:]-b[1:])**2))

def sum_squared_distance_fast(a,b):
    subt = a[1:] - b[1:]
    return(np.dot(subt,subt))

def precision(n,precision):
    a : float = (n*precision//1)/precision
    return (a)

def leave_one_out_accuracy(data,features,feature_to_add):
    data_copy = data.copy()
    remove = []
    for i in range(len(data_copy[0])):
        if i not in features and i != feature_to_add and i != 0:
            remove.append(i)
    data_copy[:,remove] = 0

    score = 0
    for i,object_to_classify in enumerate(data_copy):
        label = object_to_classify[0]
        nn_label = 0
        nn_dist = np.inf
        nn_loc = np.inf
        for k,object_to_compare in enumerate(data_copy):
            if (i != k):
                dist = sum_squared_distance_fast(object_to_classify, object_to_compare)
                #print(dist)
                #print(sum_squared_distance_fast(object_to_classify, object_to_compare))
                if (dist < nn_dist):
                    nn_dist = dist
                    nn_loc = k
                    nn_label = object_to_compare[0]
        if nn_label == label:
            score+=1
    return (float(score / len(data)))

def leave_one_out_accuracy_fast(data,features,feature_to_add):
    data_copy = data.copy()
    remove = []
    for i in range(len(data_copy[0])):
        if i not in features and i != feature_to_add and i != 0:
            remove.append(i)
    data_copy[:,remove] = 0

    score = 0
    for i,object_to_classify in enumerate(data_copy):
        label = object_to_classify[0]

        diff_matrix = data_copy[:,1:] - object_to_classify[1:]
        diff_matrix[i] = np.inf
        diff_matrix = np.einsum('ij,ij->i',diff_matrix,diff_matrix)
        index_min = np.argmin(diff_matrix)
        nn_dist = sum_squared_distance_fast(object_to_classify, data_copy[index_min])
        nn_loc = index_min
        nn_label = data_copy[index_min,0]
        if nn_label == label:
            score+=1
    return (float(score/len(data)))

def feature_search_forward(data,epsilon):
    best_features = []
    best_accuracy: float = 0
    search_features = []
    print("Starting Search \n")
    best_last_loop = 0
    default_rate = 0
    features_all = []
    results_all = []
    features_all.append([])
    features_all.append([])

    for i in data:
        if i[0] == 2:
            default_rate+=1
    default_rate = precision(max(default_rate/len(data),1-default_rate/len(data)),100)
    print(f"The default rate is {default_rate}\n")
    results_all.append(default_rate)

    for i in range(1,len(data[0])):
        feature_to_add = 0
        best_so_far_accuracy = 0

        for k in range(1,len(data[0])):
            if k not in search_features:
                accuracy = leave_one_out_accuracy_fast(data, search_features, k)
                print(f"    Using feature(s) {search_features} and [{k}] accuracy is {accuracy}")

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add = k
        results_all.append(best_so_far_accuracy)
        if best_so_far_accuracy >= best_accuracy - epsilon:
            best_accuracy = best_so_far_accuracy
            best_features.append(feature_to_add)
        search_features.append(feature_to_add)
        features_all.append(search_features.copy())
        if (best_so_far_accuracy < best_last_loop):
            print(f"Best accuracy decrease from {best_last_loop} to {best_so_far_accuracy}")
        best_last_loop = best_so_far_accuracy
        print(f"Feature {feature_to_add} was added to set of features\n")

    return (best_features,best_accuracy,results_all,features_all)

def feature_search_backward(data,epsilon):
    best_features = []
    best_accuracy: float = 0
    search_features = []
    for i in range(1,len(data[0])):
        search_features.append(i)
    print("Starting Search \n")
    best_last_loop = 0
    results_all = []
    features_all = []
    features_all.append(search_features.copy())
    
    for i in range(1,len(data[0])-1):
        feature_to_remove = 0
        best_so_far_accuracy = 0

        for k in search_features:
            temp = search_features.copy()
            temp.remove(k)
            accuracy = leave_one_out_accuracy_fast(data, temp, np.inf)
            print(f"    Using feature(s) {search_features} and removing [{k}] accuracy is {accuracy}")
            if accuracy > best_so_far_accuracy:
                best_so_far_accuracy = accuracy
                feature_to_remove = k
        results_all.append(best_so_far_accuracy)
        search_features.remove(feature_to_remove)
        features_all.append(search_features.copy())
        if best_so_far_accuracy >= best_accuracy - epsilon:
            best_accuracy = best_so_far_accuracy
            best_features = search_features.copy()
        print("")
        if (best_so_far_accuracy < best_last_loop):
            print(f"Best accuracy decrease from {best_last_loop} to {best_so_far_accuracy}\n")
        best_last_loop = best_so_far_accuracy
        if i < len(data[0])-2:
            print(f"Removed feature {feature_to_remove}, current set is {search_features}\n")
    default_rate = 0
    for i in data:
        if i[0] == 2:
            default_rate+=1
    default_rate = precision(max(default_rate/len(data),1-default_rate/len(data)),100)
    print(f"The default rate is {default_rate}\n")
    results_all.append(default_rate)
    features_all.append([])

    return (best_features,best_accuracy,results_all,features_all)

print("What data set would you like to run?")
print("    1) Small set 96")
print("    2) Large set 21")
print("    3) Small set 6")
print("    4) Large data set 96")
print("    5) Small data set 88")
print("    6) Large data set 6")
print("    7) Small data set 100")
print("    8) Large data set 85")
choice = input("Input 1-8: ")
if choice == "1":
    string = "CS170_Small_Data__96.txt"
elif choice == "2":
    string = "CS170_Large_Data__21.txt"
elif choice == "3":
    string = "CS170_Small_Data__6.txt"
elif choice == "4":
    string = "CS170_Large_Data__96.txt"
elif choice == "5":
    string = "CS170_Small_Data__88.txt"
elif choice == "6":
    string = "CS170_Large_Data__6.txt"
elif choice == "7":
    string = "CS170_Small_Data__100.txt"
elif choice == "8":
    string = "CS170_Large_Data__85.txt"

else:
    print("bruh! that isn't a choice")
    exit()
print("Which algorithm would you like to run?")
print("    1) Forward Selection")
print("    2) Backward Elimination")
choice = input("Input 1 or 2: ")

arr = np.loadtxt(string)
time1 = time.time()
if choice == "1":
    results = feature_search_forward(arr, 0)
else:
    results = feature_search_backward(arr,0)
    pass
time2 = time.time()
time = time2 - time1

print(f"The best set of features is {results[0]} with accuracy {results[1]}")
if time/60 < 1:
    print(f"Took {precision(time, 1000)} seconds to converge")
else:
    print(f"Took {precision(time/60, 1000)} minutes to converge")

accuracies = results[2]
features = results[3]
plt.plot(accuracies,'o-r',)
plt.title('Accuracy of features for large set 85')
plt.xlabel('Features removed')
plt.ylabel('Accuracy')
#plt.gca().set_xticklabels(features)
plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()]) 
plt.show()