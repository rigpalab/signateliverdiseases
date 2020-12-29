import pandas as pd
import numpy as np

from sklearn import metrics, preprocessing, linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, explained_variance_score  
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, f1_score

from catboost import CatBoostClassifier
from catboost import Pool

def main():
    # Set seed for reproducibility
    np.random.seed(8814)
    
    print("Loading data...")
    # Load the data from the CSV files
    training_data = pd.read_csv('train.csv', header=0)
    prediction_data = pd.read_csv('test.csv', header=0)

    print(training_data.head(16))
    
    features = ['Age','Gender','T_Bil','D_Bil','ALP','ALT_GPT','AST_GOT','TP','Alb','AG_ratio']
    print(f"Loaded {len(features)} features")

    train = training_data[features]
    target = training_data["disease"]
    test = prediction_data[features]

    test['Gender'] = test['Gender'].map( {'Male':1, 'Female':0} )
    print(test[['Gender']])

    train['Gender'] = train['Gender'].map( {'Male':1, 'Female':0} )
    print(train[['Gender']])   

    # step 1 hyperparameters optimization
    # model = CatBoostClassifier(iterations=5000, loss_function='Logloss', logging_level='Verbose',  task_type = 'GPU')

    # grid = {
    #      'learning_rate': [0.0003, 0.0019, 0.0017, 0.04, 0.1],
    #      'depth': [4, 5, 6, 8, 7, 9, 10, 11, 12],
    #      'l2_leaf_reg': [1, 3, 5, 7, 9],
    #      'bagging_temperature' : [0.1, 0.26, 0.54, 0.67, 0.86, 0.97],
    #      }

    # randomized_search_result = model.randomized_search(grid, X=train, y=target, plot=True)\
    
    # print(randomized_search_result)
 
    # results = pd.DataFrame(randomized_search_result.items())

    # results.to_csv("parameters_liver009.csv", index=True)

    # step 2 when parameters known from step 1 (file parameters_liver009.csv)
    # parameres found {'params': {'depth': 9, 'l2_leaf_reg': 7, 'bagging_temperature': 0.86, 'learning_rate': 0.0017}, 
    model = CatBoostClassifier(depth=9, l2_leaf_reg=7, bagging_temperature=0.86, learning_rate=0.0017,
        iterations=4894, loss_function='Logloss', logging_level='Verbose',  task_type = 'GPU')

    # step 1 and 2 kfolding
    n_split = 12 
    kf = KFold(n_splits=n_split, random_state=883, shuffle=True)
    y_valid_pred = 0 * target
    y_test_pred = 0

    print("# Training...")

    for idx, (train_index, valid_index) in enumerate(kf.split(train)):
        y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
        X_train, X_valid = train.iloc[train_index,:], train.iloc[valid_index,:]
        _train = Pool(X_train, label=y_train)
        _valid = Pool(X_valid, label=y_valid)
        print( "\nFold ", idx)
        fit_model = model.fit(_train,
                              eval_set=_valid,
                              use_best_model=True,
                              verbose=500,
                              plot=True
                             )
        pred = fit_model.predict_proba(X_valid)[:, 1]
        # just for fun, these metrics most for regression tasks
        print( "  mean_absolute_error = ", mean_absolute_error(y_valid, pred) )
        print( "  mean_squared_error = ", mean_squared_error(y_valid, pred) )
        print( "  mean_squared_log_error = ", mean_squared_log_error(y_valid, pred) )
        print( "  explained_variance_score = ", explained_variance_score(y_valid, pred, sample_weight=None, multioutput='uniform_average') )
        y_valid_pred.iloc[valid_index] = pred
        y_test_pred += fit_model.predict_proba(test)[:, 1]

    print(y_test_pred)
    y_test_pred /= n_split
    print(y_test_pred)

    print("# Creating submission probabilities...")
    # Create your submission
    results_df = pd.DataFrame(data={'prediction':y_test_pred})
    print("# Writing predictions to submission_kfold009.csv...")
    # Save the predictions out to a CSV file.
    results_df.to_csv("submission_kfold009.csv", index=True)
    # to upload to competition site is need to remove header from submission file

if __name__ == '__main__': 
    main()
