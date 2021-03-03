import pandas as pd

def general_search(data_list):
    current_set_of_features = []

    for i in range(0, len(data_list)):
        if i not in current_set_of_features:
            print(f'On the %sth level of the search tree' % (i + 1))
            feature_to_add_at_this_level = 0
            best_accuracy_so_far = 0

            for j in range(1, len(data_list[i])):
                print(f'--Considering adding the %s feature' % j)
            
                accuracy = leave_one_out_cross_validation(data_list, current_set_of_features, j+1)

                if accuracy > best_accuracy_so_far:
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = j
        
            current_set_of_features.append(feature_to_add_at_this_level)
            print(f'On level %s i added feature %s to current set' % (i, feature_to_add_at_this_level))

def main():
    file_name = 'data/test_features.csv'
    df = pd.read_csv(file_name, delimiter=' ')
    df_list = df.values.tolist()
    general_search(df_list)

if __name__ == '__main__':
    main()