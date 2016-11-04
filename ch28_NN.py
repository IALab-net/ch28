"""DRAFT NN dsc28
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor # Keras wrapper object for use in scikit-learn as a regression estimator
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X = patients[predictors].values
features_num = X.shape[1]
Y = patients['cible'].values

### Parameters

# Neural Networks training parameters
nb_folds = 10
nb_epoch = 10 # try up to 50
batch_size = 5 # try down to 5. On baseline model : ==5 => 90 sec per epoch ; ==50 => 10 sec
verbose = 1 # => 1: progression bars

# For automation ease
want_to_ask = False # True : will ask you for choices. False : will take default decisions.
def ask_me_maybe(message, default='yes'):
    '''If choice given, asks for choice.
    Returns default choice if no choice given, or enter pressed.
    Default = 'yes' or 'no', if not will return error'''
    default = {'yes': True, 'no':False}[default] # replacing 'yes'/'no' by boolean
    if not want_to_ask :
        return default
    else :
        yes_values = ['y', 'Y', 'yes', 'YES']
        if default == 'yes':
            yes_values.append('') # pressing enter will answer 'yes' by default
        return input(message) in yes_values

# Tester sur extrait de la BDD ?
extrait = 80000
want_sample = ask_me_maybe('Want to try NN first on a smaller sample '
    '(%i observations) ? (y/n) >> ' % extrait, default = 'no')
if want_sample:
    X = X[:extrait, :]
    Y = Y[:extrait]
print('Training models on %i examples' % len(Y))
print("Predictor features : " + str(predictors))

### fix random seed for reproducibility
seed = 7
np.random.seed(seed)

### helper functions
# print RMSE
def print_results(message, results):
    print("\n- " + message + ": %.2f%% average RMSE on %i folds (with %.2f%% std deviation)"
        % (results.mean() * 100, nb_folds, results.std() * 100))
# evaluate model with standardized dataset
def evaluate_stded_model(model, message):
    ''' creates a scikit-learn Pipeline
    that first standardizes the dataset then
    creates and evaluate the baseline neural network model.'''
    np.random.seed(seed)
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=model,
        nb_epoch = nb_epoch, batch_size = batch_size, verbose = verbose)))
    # NOTE: The Keras wrappers require a function as an argument --> ici, model = baseline_model
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=nb_folds, random_state=seed)
    results = np.sqrt(cross_val_score(pipeline, X, Y, cv=kfold))
    print_results(message, results)
    return results



''' ------------------------------------------------------------------
1. Modeling a Baseline Neural Network '''

print('\n# Modeling a Baseline Neural Network\n')
# define base model
def baseline_model():
    '''returns a basic neural network'''
    # create model
    model = Sequential()
    model.add(Dense(features_num, input_dim=features_num,
        init='normal', activation='relu')) # relu = good practice
    model.add(Dense(1, init='normal')) # w/o activation => regression
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

want_base = ask_me_maybe('Want to try again with standardisation of features ? (y/n) >> ',
    default = 'yes')
if want_base :
    start_timer()
    results_base = evaluate_stded_model(baseline_model, "Baseline") # NOTE: function passed as argument
    time_to('train "baseline" NN, with standardized features')

'''------------------------------------------------------------------
2. Evaluate a Deeper Network Topology'''

print('\n# Evaluating a Deeper Network Topology\n')

def larger_model_2l():
	# create model
	model = Sequential()
	model.add(Dense(features_num, input_dim = features_num, init='normal', activation='relu'))
	model.add(Dense(features_num // 2, init='normal', activation='relu'))
	model.add(Dense(1, init='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
want_2l = ask_me_maybe('Want to train and test a deeper neural network (2 layers) ? (y/n) >> ',
    default = 'yes')
if want_2l :
    start_timer()
    results_larger_2l = evaluate_stded_model(larger_model_2l, "2 layers")
    time_to('train 2-layers NN')

def larger_model_3l():
	# create model
	model = Sequential()
	model.add(Dense(features_num, input_dim = features_num, init='normal', activation='relu'))
	model.add(Dense(features_num, init='normal', activation='relu'))
	model.add(Dense(features_num // 2, init='normal', activation='relu'))
	model.add(Dense(1, init='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
want_3l = ask_me_maybe('Want to train and test a deeper neural network (3 layers) ? (y/n) >> ',
    default = 'yes')
if want_3l :
    start_timer()
    results_larger_3l = evaluate_stded_model(larger_model_3l, "3 layers")
    time_to('train 3-layers NN')


'''------------------------------------------------------------------
3. Evaluate a Wider Network Topology'''

print('\n# Evaluating a Wider Network Topology\n')
def wider_deeper_model():
	# create model
	model = Sequential()
	model.add(Dense(int(features_num // 0.6), input_dim = features_num,
        init='normal', activation='relu'))
	model.add(Dense(1, init='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
want_wider = ask_me_maybe('Want to train and test a deeper neural network (3 layers) ? (y/n) >> ',
    default = 'yes')
if want_wider :
    start_timer()
    results_wider = evaluate_stded_model(wider_deeper_model, "Wider & Deeper")
    time_to('train wider NN')


'''------------------------------------------------------------------
3. Deep & Wide & Dropout'''

print('\n# Evaluating a Wider, Deeper Network Topology\n')
def wider_model():
    # create model
    model = Sequential()
    model.add(Dropout(0.2))
    model.add(Dense(int(features_num // 0.6), input_dim = features_num,
        init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(features_num, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(features_num // 2, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, init='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
want_wider = ask_me_maybe('Want to train and test a deeper neural network (3 layers) ? (y/n) >> ',
    default = 'yes')
if want_wider :
    start_timer()
    results_wider = evaluate_stded_model(wider_model, "Wider")
    time_to('train wider NN')
