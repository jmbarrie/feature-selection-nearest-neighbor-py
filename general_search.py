import pandas as pd
import numpy as np
import timeit
from copy import deepcopy

def feature_search_demo(data_list):
    best_accuracy_so_far = 0
    current_set_of_features = set()
    current_dict_of_features = {}
    num_lines = len(data_list)
    num_features = len(data_list[0])

    for i in range(0, num_lines):
        print('-' * 25)
        print(f'On the %d level of the search tree' % (i + 1))
        feature_to_add_at_this_level = None

        for k in range(1, num_features):
            if k not in current_set_of_features:
                print(f'--Considering adding the %s feature' % k)
            
                accuracy = leave_one_out_cross_validation(data_list, current_set_of_features, k, num_lines, num_features)

                if accuracy > best_accuracy_so_far:
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = k
        
        if feature_to_add_at_this_level is not None:
            print(f'On level %s i added feature %s to current set' % (i + 1, feature_to_add_at_this_level))
            current_set_of_features.add(feature_to_add_at_this_level)
            print(f'Accuracy: %f' % best_accuracy_so_far)
            current_dict_of_features[feature_to_add_at_this_level] = best_accuracy_so_far
        else:
            print('Accuracy is decreasing, stopping the search here.')
            break
    
    print('-' * 25)

    print(f'Best set of features: %s, with accuracy: {best_accuracy_so_far * 100:.2f}' % (current_set_of_features))

    return current_dict_of_features

def leave_one_out_cross_validation(data, current_set, feature_to_add, num_lines, num_features):
    cross_data = deepcopy(data)
    cross_check = deepcopy(list(current_set))
    cross_check.append(feature_to_add)

    for i in range(0, num_lines):
        for k in range(1, num_features):
            if k not in cross_check:
                cross_data[i][k] = 0

    number_correctly_classified = 0

    for i in range(0, num_lines):
        object_to_classify = cross_data[i][1:]
        label_object_to_classify = cross_data[i][0]

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')

        for k in range(0, num_lines):
            if k != i:
                # print(f'Ask if %s is nearest neighbor with %s' % (i, k))
                distance = np.sqrt(sum([(a - b) * (a - b) for a, b in zip(object_to_classify, cross_data[k][1:])]))

                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = cross_data[nearest_neighbor_location][0]

        # print(f'Object %s is class %s' % (i + 1, label_object_to_classify))
        # print(f'Its nearest neighbor is %s which is in class %s' % (nearest_neighbor_location, nearest_neighbor_label))

        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified = number_correctly_classified + 1

    accuracy = number_correctly_classified / len(cross_data)

    return accuracy 

def main():
    # file_name = 'data/CS170_small_special_testdata__95.txt'
    # file_name = 'data/CS170_small_special_testdata__96.txt'
    # file_name = 'data/CS170_small_special_testdata__97.txt'
    # file_name = 'data/CS170_small_special_testdata__98.txt'
    file_name = 'data/CS170_small_special_testdata__99.txt'
    # file_name = 'data/CS170_SMALLtestdata__33.txt'
    # file_name = 'data/CS170_largetestdata__38.txt'
    df = pd.read_csv(file_name, delimiter='\s+', header=None)
    df_list = df.values.tolist()
    start = timeit.default_timer()
    features = feature_search_demo(df_list[:50])
    end = timeit.default_timer()
    print(f'Results: %s' % features)
    print(f'Time to run: %s' % (end - start))

if __name__ == '__main__':
    main()