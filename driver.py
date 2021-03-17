import pandas as pd
import numpy as np
import timeit
from copy import deepcopy

def feature_search_demo(data_list):
    """
    Forward selection. Starts with empty set, and progressively adds more features
    to the set as they provide an increased accuracy.
    """
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
                # deepcopy to print this out, can be optimized a bit better
                print_set = deepcopy(current_set_of_features)
                print_set.add(k)
                print(f'Using feature(s) %s accuracy is {accuracy * 100:.2f}%%' % (print_set))

                # If the accuracy from cross validation is better, update the 
                # feature to add and its accuracy
                if accuracy > best_accuracy_so_far:
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = k
        
        # If the feature provides an increase in accuracy, add it to the set
        # Otherwise stop because all features are added or accuracy is 
        # decreasing
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
    """
    Backward elimination. Start with a set of all possible features and remove
    a feature at every level based on the overall accuracy.
    """
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
                # We use -1 * k to denote backward elimination 
                accuracy = leave_one_out_cross_validation(data_list, current_set_of_features, (-1 * k), num_lines, num_features)
                # deepcopy to print this out, can be optimized a bit better
                print_set = deepcopy(current_set_of_features)
                print_set.remove(k)
                print(f'Using feature(s) %s accuracy is {accuracy * 100:.2f}%%' % (print_set))
                # If the accuracy from cross validation is better, update the 
                # feature to add and its accuracy
                if accuracy > best_accuracy_so_far:
                    best_accuracy_so_far = accuracy
                    feature_to_remove_at_this_level = k
        # If the feature provides an increase in accuracy, add it to the set
        # Otherwise stop because all features are added or accuracy is 
        # decreasing        
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
    """
    Performing the nearest neighbor search. Uses the features provided to 
    determine an accuracy.
    """
    # If feature is a positive value, it is a forward search so we append the 
    # feature
    # Otherwise, if feature is a negative value, it is a backward elimination so
    # we remove the feature
    cross_check = deepcopy(list(current_set))
    if feature > 0:
        cross_check.append(feature)
    else:
        cross_check.remove(abs(feature))

    # Create a deepcopy of the data and then set all unnecessary features to 0
    cross_data = deepcopy(data)
    for i in range(0, num_lines):
        for k in range(1, num_features):
            if k not in cross_check:
                cross_data[i][k] = 0

    number_correctly_classified = 0

    # Calculating the nearest neighbor distance and classifying each instance
    # then using the correctly classified values to calculate accuracy
    for i in range(0, num_lines):
        object_to_classify = cross_data[i][1:]
        label_object_to_classify = cross_data[i][0]

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')

        for k in range(0, num_lines):
            if k != i:
                distance = np.sqrt(sum([(a - b) * (a - b) for a, b in zip(object_to_classify, cross_data[k][1:])]))

                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = cross_data[nearest_neighbor_location][0]

        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified = number_correctly_classified + 1

    accuracy = number_correctly_classified / len(cross_data)

    return accuracy 

def selection():
    """
    This gathers the user input such as file name and algorithm selection.
    """
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