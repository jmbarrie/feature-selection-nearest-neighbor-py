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

    print('Beginning search.\n')

    for i in range(0, num_lines):
        feature_to_add_at_this_level = None

        for k in range(1, num_features):
            if k not in current_set_of_features:
                accuracy = leave_one_out_cross_validation(data_list, current_set_of_features, k, num_lines, num_features)
                print_set = deepcopy(current_set_of_features)
                print_set.add(k)
                print(f'Using feature(s) %s accuracy is {accuracy * 100:.2f}%%' % (print_set))

                if accuracy > best_accuracy_so_far:
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = k
        
        if feature_to_add_at_this_level is not None:
            current_set_of_features.add(feature_to_add_at_this_level)
            current_dict_of_features[feature_to_add_at_this_level] = best_accuracy_so_far
            print(f'\nFeature set %s was best, accuracy is {best_accuracy_so_far * 100:.2f}%%\n' % current_set_of_features)
        elif len(current_set_of_features) == num_features:
            print('All features added')
            break
        else:
            print('\nWarning: Accuracy is decreasing, stopping the search here.')
            break
    
    print(f'\nFinished search!! The best feature subset is %s, which has an accuracy of {best_accuracy_so_far * 100:.2f}%%' % current_set_of_features)

    return current_dict_of_features

def backward_search(data_list):
    num_lines = len(data_list)
    num_features = len(data_list[0])
    current_set_of_features = set(range(1, num_features))
    current_dict_of_features = {}
    best_accuracy_so_far = 0

    print('Beginning search.\n')

    for i in range(0, num_lines):
        feature_to_remove_at_this_level = None

        for k in range(1, num_features):
            if k in current_set_of_features:
                accuracy = leave_one_out_cross_validation(data_list, current_set_of_features, (-1 * k), num_lines, num_features)
                print_set = deepcopy(current_set_of_features)
                print_set.remove(k)
                print(f'Using feature(s) %s accuracy is {accuracy * 100:.2f}%%' % (print_set))

                if accuracy > best_accuracy_so_far:
                    best_accuracy_so_far = accuracy
                    feature_to_remove_at_this_level = k
        
        if feature_to_remove_at_this_level is not None:
            current_set_of_features.remove(feature_to_remove_at_this_level)
            current_dict_of_features[feature_to_remove_at_this_level] = best_accuracy_so_far
            print(f'\nFeature set %s was best, accuracy is {best_accuracy_so_far * 100:.2f}%%\n' % current_set_of_features)
        elif len(current_set_of_features) == num_features:
            print('All features added')
            break
        else:
            print('\nWarning: Accuracy is decreasing, stopping the search here.')            
            break

    print(f'\nFinished search!! The best feature subset is %s, which has an accuracy of {best_accuracy_so_far * 100:.2f}%%' % current_set_of_features)

    return current_dict_of_features
    

def leave_one_out_cross_validation(data, current_set, feature, num_lines, num_features):
    # If forward search, append the feature to add
    cross_check = deepcopy(list(current_set))
    if feature > 0:
        cross_check.append(feature)
    else:
        cross_check.remove(abs(feature))

    cross_data = deepcopy(data)
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

def selection():
    while True:
        file_selection = input('Type in the name of the file to test: ')
        break

    while True:
        print('Type the number of the algorithm you want to run.')
        print('\n1) Forward Selection\n2) Backward Elimination')
        algo_selection = input('\n')

        if algo_selection == '1':
            print('\nSelected the Forward Selection algorithm.')
            break
        elif algo_selection == '2':
            print('\nSelected the Backward Elimination algorithm.')
            break
        else:
            print('Please select either "1" or "2".')

    return file_selection, algo_selection 

def main():
    print("Welcome to Juan's Feature Selection Algorithm.")
    file_name, user_selection = selection()
    df = pd.read_csv(file_name, delimiter='\s+', header=None)
    df_list = df.values.tolist()

    print(f'\nThe data has %d features (not including the class attribute), with %d instances\n' % (len(df_list[0]) - 1, len(df_list)))
    start = timeit.default_timer()
    if user_selection == '1':
        features = feature_search_demo(df_list[:])
    elif user_selection == '2':
        features = backward_search(df_list[:])
    end = timeit.default_timer()
    print(f'Time to complete the search was {end - start:.2f} seconds')

if __name__ == '__main__':
    main()