import pandas as pd
import numpy as np

def general_search(data_list):
    current_set_of_features = []

    for i in range(0, len(data_list)):
        if i not in current_set_of_features:
            # print(f'On the %sth level of the search tree' % (i + 1))
            feature_to_add_at_this_level = 0
            best_accuracy_so_far = 0

            for j in range(1, len(data_list[i])):
                # print(f'--Considering adding the %s feature' % j)
            
                accuracy = leave_one_out_cross_validation(data_list, current_set_of_features, j+1)

                if accuracy > best_accuracy_so_far:
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = j
        
            current_set_of_features.append(feature_to_add_at_this_level)
            # print(f'On level %s i added feature %s to current set' % (i, feature_to_add_at_this_level))

def leave_one_out_cross_validation(data, current_set, feature_to_add):
    number_correctly_classified = 0
    nearest_neighbor_label = 0

    for i in range(0, len(data)):
        object_to_classify = data[i][1:]
        label_object_to_classify = data[i][0]

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')

        for j in range(0, len(data)):
            # print(f'Ask if %s is nearest neighbor with %s' % (i, j))
            if j != i:
                # distance = sum([a - b for a, b in zip(object_to_classify, data[j][1:])])
                distance = np.sqrt(np.sum([(a - b) * (a - b) for a, b in zip (object_to_classify, data[j][1:])]))

                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = j
                    nearest_neighbor_label = data[nearest_neighbor_location][0]

        # print(f'Object %s is class %s' % (i + 1, label_object_to_classify))
        print(f'Its nearest neighbor is %s which is in class %s' % (nearest_neighbor_location, nearest_neighbor_label))

        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1

    accuracy = number_correctly_classified / len(data)
    # print(f'Classifed %s correctly' % number_correctly_classified)
    print(f'Accuracy: %s' % accuracy)

    return accuracy 

def main():
    df = pd.read_csv(file_name, delimiter='\s+', header=None)
    df_list = df.values.tolist()
    general_search(df_list)

if __name__ == '__main__':
    main()