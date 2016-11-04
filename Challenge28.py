
''' Datascience.net challenge 28
2016 oct, FZ
indentation convention : tabs
------------------------------------------------------------------------------
PART 1 - PREPARING DATA
'''

print('\n## IMPORTING DATA')
# setting timer
import time
def start_timer():
	global start
	start = time.clock()
def current_timer():
	global start
	return str(round(time.clock() - start,1)) + " sec"
def time_to(do_this):
	'''measures and tells you what'''
	print("\n... time to " + do_this + " : " + current_timer())

# importing useful libraries
start_timer()
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
import datetime
time_to("import libraries : ")
matplotlib.interactive(True)	# debugging plt.show()

# loading datasets
start_timer()
finess_type = {'Finess': 'object'} # to prevent mixed types bugs on hospital codes
patients  = pd.read_csv('data2.csv', sep=';', dtype = finess_type)
time_to("open data2.csv")
print("\nSize of training set : " + str(patients.shape[0]) + ' observations, with '
	+ str(patients.shape[1]) + ' features.')

start_timer()
patients_test  = pd.read_csv('test2.csv', sep=';', dtype = finess_type)
time_to("open test2.csv : ")
print("\nSize of test set : " + str(patients_test.shape[0]) + ' observations, with '
	+ str(patients_test.shape[1]) + ' features.')

''' ------------------------------------------------------------------------------
# # 1 - Basic data treatment
'''
print('\n\n## TREATING DATA, ON BOTH TRAIN AND TEST SETS')

'''### Simplifying column names'''
print('\n- Simplifying columns labels :')

print('\nTraining set original columns :')
print(patients.columns)
full_names = patients.columns
simple_names = ['code', 'nom', 'dept', 'domaine', 'classe_age', 'nb_cmo',
	'nb_total', 'annee', 'cible']
# keeping track (for better graph labels)
simple_to_full = {simple_names[i]: full_names[i] for i in range(0,len(simple_names))}
# full_to_simple = {full_names[i]: simple_names[i] for i in range(0,len(simple_names))}
def full_feature(feature):
	try :
		return simple_to_full[feature]
	except :
		return feature
# converting
patients.columns = simple_names
print('\nNew columns :')
print(patients.columns)
# same same for test data
simple_names_test = simple_names[0:8]
simple_names_test.insert(0, 'id')
patients_test.columns = simple_names_test

'''### Cleaning features: finess, dept, classe_age, domaines d'activité'''
# helper function for corsica dept (removing letters A or B)
def convert_dept(str):
	if str[0] == "2":
		return 2
	try :
		return int(str)
	except :
		return 0
# helper function for corsica hospitals codes (removing letters A or B)
def convert_finess(str):
	if str[0:2] == '2A' :
		return '201' + str[2:]
	if str[0:2] == '2B' :
		return '202' + str[2:]
	else :
		return str
# cleaner function
def clean(bdd) :
	# converting hospital code --> int (non numeric as NaN)
	bdd['code_num'] = bdd['code'].apply(lambda x: convert_finess(x))
	bdd['code_num'] = pd.to_numeric(bdd['code_num'], errors = 'coerce')
	# converting 'dept' --> numero 'dept_num' as int
	bdd['dept_num'] = bdd['dept'].apply(lambda x: convert_dept(x[0:2]))
	simple_to_full['dept_num'] = simple_to_full['dept'] # label for charts
	# classe_age
	bdd['classe_age_0_1'] = bdd['classe_age'].map({'<=75 ans': 0, '>75 ans': 1}).astype(int)
	simple_to_full['classe_age_0_1'] = "Classe d'age (supérieur ou non à 75 ans)" # label
	# domaines d'activité
	bdd['domaine_num'] = bdd['domaine'].apply(lambda x: convert_dept(x[1:3]))
# cleaning sets
print('\n- Converting categorical labels to numeric values (into new columns)')
clean(patients)
clean(patients_test)

'''### Feature engineering : '''

def feature_engineer(df):
	''' Centralizes all transformations and new variables,
	to be applied to both train and test set '''
	# boolean tracker for nb_cmo
	print('... Creating a boolean tracker of nb_cmo')
	df['cmo_or_not'] = df['nb_cmo']
	df.loc[df['cmo_or_not'] > 0, 'cmo_or_not'] = 1 # if != 0, then = 1

	# logarithmic transformations
	print('... Logarithmic transformations')
	for feature in ['nb_cmo', 'nb_total', 'cible']:
		if feature in df.columns : # applicable to train and test sets
			start_timer()
			feature_log = feature + '_log'
			df[feature_log] = df[feature].apply(lambda x: math.log(x+1))
			simple_to_full[feature_log] = simple_to_full[feature] + ' (logarithmic transformation)'

print('\n- Feature engineering')
for df in [patients, patients_test]:
	feature_engineer(df)


'''### Data checks '''

# the following lines allow to check that ( cmo = 0 <=> cible = 0 )
# patients['cible_or_not'] = patients['cible'].apply(lambda x: one_or_zero(x))
# patients['cible_n_cmo'] = patients['cmo_or_not'] + patients['cible_or_not']

print('\n- Simplifying the problem : removing observations where (nb_cmo==0)')
if patients.loc[patients['nb_cmo']!=0].shape[0] == patients.shape[0] :
	print('zeros already removed')
else :
	print("Since (nb_cmo==0)<=>(target==0), we can simplify the problem, and restrict the machine learning on cases where (nb_cmo!=0).")
	print('\nDistribution of nb_cmo :')
	train_0 = str(round(patients.loc[patients['nb_cmo']==0].shape[0]
		/ patients.shape[0] * 100)) + '%'
	train_1 = str(round(patients.loc[patients['nb_cmo']!=0].shape[0]
		/ patients.shape[0] * 100)) + '%'
	test_0 = str(round(patients_test.loc[patients['nb_cmo']==0].shape[0]
		/ patients_test.shape[0] * 100)) + '%'
	test_1 = str(round(patients_test.loc[patients['nb_cmo']!=0].shape[0]
		/ patients_test.shape[0] * 100)) + '%'
	print(pd.DataFrame(
		data = {'train': [ train_0, train_1], 'test':[test_0, test_1]} ,
		index = ['zeros', 'non-zeros']
		) . sort_index(axis=1, ascending = False) )

	# Removing values = 0, if the user wishes so
	want_less = input("\nDo you want to remove observations where 'nb_cmo' is equal to zero ? \n(y/n) >> ")
	if want_less in ['y', 'Y', 'yes', 'YES', ''] :  # default (enter) = yes
		patients = patients[patients['nb_cmo'] > 0]
		print('Zeros removed, size of the training set : ' + str(patients.shape[0]) + " observations")



# checking NA values
print("\n- Checking number of NA's : ")
df_NA = {}
count = 0
data = [('train set', patients), ('test set', patients_test)]
for bdd in data:
	NA_count = {}
	for label in bdd[1].columns:
		NA_count[label] = bdd[1].shape[0] - bdd[1][label].dropna().shape[0]
		count += NA_count[label]
	NA_values = [NA_count[label] for label in NA_count]
	NA_index = [label for label in NA_count]
	df_NA[bdd[0]] = pd.DataFrame(NA_values, index = NA_index, columns= [bdd[0]])
if count == 0 :
	print('data clean, no NA found')
else :
	print(pd.DataFrame(df_NA[data[0][0]].join(df_NA[data[1][0]], how= 'outer')))

# checking consistency between train and test
print('\n- Checking number of unique values in both sets. Consistency report : ')
values = {'train set':[], 'test set':[], 'common values':[], 'differences':[]}
values_index = []
for truc in patients_test.columns:
	try :
		a, b = list(patients[truc].unique()), list(patients_test[truc].unique())
		c = pd.Series(list(set(a) & set(b))).shape[0]
		a, b = len(a), len(b)
		values['train set'].append(a)
		values['test set'].append(b)
		values['common values'].append(c)
		values['differences'].append(a + b - 2 * c)
		values_index.append(truc)
	except KeyError as error:
		print('NB: ' + truc + ' not found in train set.')
print(pd.DataFrame(values, index = values_index)
	.sort_index(axis = 1, ascending = False).sort_values(by = 'differences', axis=0))


# ### feature engineering: dummy variables (binarisation des var)
# 18 domaines, 108 (-1) depts, éventuellement clusters d'établissements
''' Utiliser le code qui suit pour créer les variables binaires'''
'''def dummy(x,y):
	if str(x) == str(y):
		return 1
	else:
		return 0
def dummize(df, label):
	start_timer()
	count = 0
	for value in df[label].unique():
		count += 1
		label_value = label + '_' + str(value)
		if label_value in df.columns:
			print(str(count) + '/' + str(len(df[label].unique())) + ' - '
				+ label_value + ' passed')
			pass
		else:
			print(str(count) + '/' + str(len(df[label].unique())) + ' - '
				+ 'processing ' + label_value + ' column ')
			df[label_value] = df[label].apply(lambda x: int(str(x) == str(value)) )
			print('...' + str(round(time.clock() - start,1)) + " sec")
	time_to('make dummy variable from ' + label)
print('\n- Making dummy variables')
for df in [patients, patients_test]:
	for variable in ['domaine_num', 'dept_num']:
		dummize(df, variable)'''


''' Saving dummy VAR into separate csv files'''
''' Non exploitable tel quel (nécessite de nommer les listes de colonne pour chaque sauvegarde)
start_timer()
patients[domaine].to_csv("data_domaine.csv", index=False)
time_to('save')
start_timer()
patients[dept].to_csv("data_dept.csv", index=False)
time_to('save')
start_timer()
patients_test[domaine].to_csv("test_domaine.csv", index=False)
time_to('save')
start_timer()
patients_test[dept].to_csv("test_dept.csv", index=False)
time_to('save')'''

''' Loading dummy variables from csv's
Utiliser le code qui suit pour charger les variables binaires préalablement sauvegardées'''
"""def append_csv(df, csv_names):
	''' loads data stored in separate csv files, appends it to the working dataframe '''
	print("\n- Loading csv's")
	for csv_file in csv_names:
		start_timer()
		data = pd.read_csv(csv_file)
		df[list(data.columns)] = data[list(data.columns)]
		del data
		time_to('load ' + csv_file)
append_csv(patients, ['data_dept.csv', 'data_domaine.csv'])
append_csv(patients_test, ['test_dept.csv', 'test_domaine.csv'])"""



''' ------------------------------------------------------------------------------
# # 2 - Exploratory graphs
'''
print('\n\n## EXPLORATORY GRAPHS')

# Helper functions
def bin_it(param):
	''' function to make bins from continuous variables, for boxplots purpose
	param : list [feature, size of bins] '''
	start_timer()
	feature = param[0]
	bin_size = param[1]
	bin_feature = feature + '_bins'
	patients[bin_feature] = patients[feature].apply(
		lambda x: np.round( (x - bin_size / 2) / bin_size) * bin_size) #.astype(int)
	time_to("convert " + feature + " into bins : ")
	global simple_to_full
	bin_full_feature = full_feature(feature) + ' (feature grouped into ' + str(bin_size) + '-sized bins)'
	simple_to_full[bin_feature] = bin_full_feature
def bin_it_log(param):
	''' function to make logarithmic bins'''
	start_timer()
	feature = param[0]
	bin_size = param[1]
	bin_feature = feature + '_log_bins'
	patients[bin_feature] = patients[feature].apply(lambda x:
		np.round( (math.log(x + 1) - bin_size / 2) / bin_size) * bin_size)
	time_to("convert " + feature + " into bins : ")
	global simple_to_full
	bin_full_feature = simple_to_full[feature] + ' (logarithmic feature grouped into ' + str(bin_size) + '-sized bins)'
	simple_to_full[bin_feature] = bin_full_feature

def show_graph():
	'''shows, pauses, closes'''
	plt.show()
	time_to("compute graph : ")
	input("Press Enter to continue...")
	plt.close()
def histo(df, name):
	"""Simple histograms to visualize variable's distribution"""
	start_timer()
	df[name].hist()
	plt.title(name)
	show_graph()
def plot_it_nice(data, labels):
	''' makes a plot with one or two series, in red and blue dots, with labels
    data = list of 2 or 4 dataframes : (x,y) or (x1, y1, x2, y2),
	labels = list of 3 strings (x, y labels, title)
	'''
	# print('(using "plot_it_nice" function...)') # debug
	start_timer()
	if len(data) == 2 :
		plt.plot(data[0], data[1], 'r.')
	elif len(data) == 4 :
		plt.plot(data[0], data[1], 'r.', data[2], data[3], 'b.')
	plt.xlabel(labels[0])
	plt.ylabel(labels[1])
	plt.title(labels[2])
	show_graph()
def boxplote(data_columns, labels=None) : #, x_ticks=None):
	''' makes a boxplot with labels
	data_columns = list of 2 dataframes : (x, y)
	labels = list of 2 strings (x, y labels)
	'''
	start_timer()
	ax = sns.boxplot(x = data_columns[0] , y = data_columns[1] , data = patients, showmeans = True)
	if labels != None:
		plt.xlabel(labels[0])
		plt.ylabel(labels[1])
	#if x_ticks != None:
	#	plt.xticks(x_ticks[0], x_ticks[1])		# I eventually did not use this features
	show_graph()


# Graphs

want_graphs = input("\nDo you want to observe variables' distribution (histograms) ? (y/n) >> ")
if want_graphs in ['y', 'Y', 'yes', 'YES'] : # default (enter) = no
	start_timer()
	# find numerical variables
	numerical_variables = []
	for d_type in ['float64', 'int64']:
		for variable in patients.dtypes.index :
			if patients.dtypes[variable] == d_type :
				numerical_variables.append(variable)
	numerical_variables.sort()
	# plot them
	for variable in numerical_variables :
		print('... timer : ' + current_timer())
		histo(patients, variable)

want_graphs = input('\nDo you want to print simple plots of target VS. features ? (y/n) >> ')
if want_graphs in ['y', 'Y', 'yes', 'YES'] : # default (enter) = no
	start_timer()
	for feature in ['nb_cmo', 'nb_total', 'nb_cmo_log', 'nb_total_log', 'classe_age_0_1'] :
		print('... timer : ' + current_timer())
		plot_it_nice([patients[feature], patients['cible']], [full_feature(feature), 'cible', ''])

want_graphs = input('\nDo you want to print Box-plots of target VS. features ? (y/n) >> ')
if want_graphs in ['y', 'Y', 'yes', 'YES'] : # default (enter) = no
	print('\n- Box plots')
	print("Making log-bins with continuous features ...")
	for feature in ['nb_cmo', 'nb_total']:
		log_step = 0.5       # CHOOSE THE STEP. 0.5 is fine compared to the higher values (~5)
		bin_it_log([feature, 0.5])
	for feature in ['nb_cmo_log_bins', 'nb_total_log_bins', 'annee', 'dept_num',
		'code', 'domaine'] :
		# 'classe_age_0_1' exclu car fait exploser la mémoire (60Go de RAM virtuelle avant plantage)
		print('... timer : ' + current_timer())
		boxplote([patients[feature], patients['cible']], [full_feature(feature), 'cible'])



'''------------------------------------------------------------------------------
# # 3 - Feature engineering and further graphs
'''
print('\n\n## ADDITIONAL FEATURE ENGINEERING')

### Analysons les hopitaux par leur "note moyenne"
print('\n- Making clusters of hospitals based on their average target')
# compute the mean scores
start_timer()
mean_for_hospital = {}
for value in patients['code'].unique():
	mean_for_hospital[value] = math.log(patients['cible'][patients['code']== value].mean() * 100 +1)
patients['cible_moy_par_code'] = patients['code'].map(mean_for_hospital)
time_to('compute means')
# creating categories (bins) of hospitals, based on their score
nb_bins = 37
step = round(max(patients['cible_moy_par_code']) / nb_bins, 1)
bin_it(['cible_moy_par_code', step])
# extrapolating to test set
start_timer()
cluster_imputation = patients['cible_moy_par_code_bins'].mean()
def extrapolate_code_bin(code, bin_size) :
	if code in mean_for_hospital:
		x = mean_for_hospital[code]
		return np.round( (x - bin_size / 2) / bin_size) * bin_size
	else :
		return cluster_imputation
patients_test['cible_moy_par_code_bins'] = patients_test['code'].apply(lambda
	x: extrapolate_code_bin(x, step))
time_to('extrapolate clusters to the test set')
print('Default value inputed to new hospitals : ' + str(round(cluster_imputation, 3)))
# visualising distribution of target over categories
boxplote(('cible_moy_par_code_bins', 'cible_log'), ('cible_moy_par_code_bins', 'cible'))


### Créons des variables de croissance (vs une référence), à vocation de remplacer les années
print('\n- Transforming year into growth')
mean_per_year={}
# Show us the numbers
print('\nAverage values of target for each year :')
for annee in patients['annee'].unique():
	mean_per_year[annee] = patients.loc[ patients['annee']== annee, 'cible'].mean()
	print(str(annee) + " : " + str(round(mean_per_year[annee], 2)))
	if annee - 1 in patients['annee'].unique():
		croissance = (mean_per_year[annee] - mean_per_year[annee - 1])
		croissance_pourcent = (croissance / mean_per_year[annee - 1]) * 100
		print("Growth vs. previous year : +"
			+ str(round(croissance, 3)) + ' / +'
			+ str(round(croissance_pourcent, 2)) + '%')
# Show us how it looks like
print('\nVisualising data on plot')
plot_it_nice([pd.Series(mean_per_year).index, pd.Series(mean_per_year)],
	['year (written in a strange way... please add 2.008e3)', 'mean target',
	'mean value of target for each year'])
# Let's guess for 2014 & 2015
'''SET THE GROWTH HERE '''
growth_coeff = {2014: 12, 2015: 12.36} # NOTE: A MàJ
print("\nValues guessed for 2014 & 2015 : \n"
	+ str(growth_coeff[2014]) + '% (2013-2014), \n'
	+ str(round(((growth_coeff[2015] + 100) / (growth_coeff[2014] + 100) - 1) * 100, 2))
	+ '% (2014-2015), thus ' + str(growth_coeff[2015])
	+ '% growth applied from 2013 to 2015')
# Compute coefficient for any year
growth_coeff[2013] = 0
for year in [2012, 2011, 2010, 2009, 2008]:
	growth_coeff[year] = (mean_per_year[year] - mean_per_year[2013]) / mean_per_year[2013] * 100
# Make it a new feature in both sets
print('\nCreating the new feature, equals to the "growth" vs 2013')
for df in [patients, patients_test]:
	df['percent_vs_2013'] = df['annee'].map(growth_coeff)


'''------------------------------------------------------------------------------
# # 4 - ML
'''
want_continue = input('\n\nDo you want to go on with the machine learning ? (y/n) >> ')
if want_continue not in ['y', 'Y', 'yes', 'YES', ''] : # default (enter) = yes
	import sys
	sys.exit()

print('\n\n## MACHINE LEARNING')


def rmse(y_true, y_pred):
	return np.sqrt( metrics.mean_squared_error(y_true, y_pred) )


'''0. predicteur idiot'''

'''Pour calculer le score le plus mauvais que l'on peut obtenir en predisant la
moyenne (cas nuls exclus)'''
'''cible_moy_cmo = patients['cible'][patients['nb_cmo'] != 0].mean()
patients['dumb_prediction'] = cible_moy_cmo
patients['dumb_prediction'][patients['nb_cmo'] == 0] = 0
print(rmse(patients['dumb_prediction'], patients['cible']))
patients_test['dumb_prediction'] = cible_moy_cmo
patients_test['dumb_prediction'][patients_test['nb_cmo'] == 0] = 0
submission = patients_test[['id', 'dumb_prediction']]
submission.columns = ['id', 'cible']
submission.to_csv("dumb_cmo_predictor.csv", index=False, sep=';')
'''

''' 1. Some preliminary treatments '''

# Cutting the problem in two, since (nb_cmo==0)<=>(target==0)
print('\n- Simplifying the problem : removing observations where (nb_cmo==0)')
# Removing target values = 0
if patients.loc[patients['nb_cmo']!=0].shape[0] == patients.shape[0] :
	print('zeros already removed previously')
else :
	patients = patients[patients['nb_cmo'] > 0]
	print('Zeros removed, size of the training set : ' + str(patients.shape[0]) + " observations")


# Testing some trick on years

want_year_trick = False # NOTE: A NETTOYER (avec la suite dans la fonction de soumission)
"""# saving actul year and related coefficients, setting years to 2013 for the ML
want_year_trick = input("\nDo you want to treat the variable 'annee' apart from the "
	"ML problem ? (y/n) >> ") in ['y', 'Y', 'yes', 'YES', '']  # default (enter) = yes
# '''SET THE GROWTH HERE : '''
yearly_growth = [0.08, 0.1664]
def set_year_coeff(yearly_growth):
	coefficients = {2014: round(1 + yearly_growth[0],4),
		2015: round(1 + yearly_growth[1], 4)}
	if 'annee_sauv' not in patients_test.columns :
		patients_test['annee_sauv'] = patients_test['annee']
	patients_test['annee'] = 2013
	patients_test['coef_annee'] = patients_test['annee_sauv'].apply(lambda
		x: coefficients[x])
	print("\nThe variable 'annee' has been set to 2013 in all the test set. \n"
		"The target will be extrapolated by the following year-to-year growths : \n"
		+ str(round(yearly_growth[0] * 100, 2)) + '% (2013-2014), \n'
		+ str(round((coefficients[2015] / coefficients[2014] * 100) - 100, 2))
		+ '% (2014-2015), thus ' + str(round(coefficients[2015] * 100 - 100, 2))
		+ '% growth applied from 2013 to 2015')
if want_year_trick :
	set_year_coeff(yearly_growth)
"""

'''2. ML'''
# Setting the predictors, based on the training set, as configured in the previous file
non_predictors = ['code', 'nom', 'dept', 'domaine', 'classe_age'
	, 'cible', 'codes_noms_nums', 'cible_log'
	, 'nb_cmo_log_bins', 'nb_total_log_bins'
	, 'cible_moy_par_code'
	,'nb_cmo', 'nb_total', 'annee' # variables remplacées
	]
# 	'dept_num', 'domaine_num'] # DO NOT USE IT IF MADE DUMMIES
#	   , 'age_75+',
#       'cible_moy_par_code', 'cible_moy_par_code_bins']
predictors = [labels for labels in patients.columns if labels not in non_predictors]
print('\n- Predictors features : ')
print(predictors)


# Gestion des modeles
model_lr = LinearRegression()
model_rf = RandomForestRegressor(random_state=1, n_estimators=150, min_samples_split=8, min_samples_leaf=4)
# min_samples_leaf and min_samples_split enhanced to reduce overfitting
model_gbm1 = GradientBoostingRegressor(random_state=1, n_estimators=25, max_depth=3)
model_gbm2 = GradientBoostingRegressor(random_state=1, n_estimators=100, max_depth=3)
model_gbm3 = GradientBoostingRegressor(random_state=1, n_estimators=25, max_depth=6)

models = {'rf': model_rf}
''' {'rf': model_rf, 'gbm1': model_gbm1, 'gbm2': model_gbm2,  }'''


# model trainer on kfolds
def train_on_3fold(model, predictors):
	'''trains and returns a given model, after testing it on 3-folds and printing RMSE
	model : sklearn model instance
	predictors : list'''
	print(model)
	predictions = []
	count = 0
	start_timer()
	# Training and testing on folds
	kf = KFold(patients.shape[0], n_folds=3, random_state=1)
	for train, test in kf:
		count +=1
		# train on train fold
		df_predictors = (patients[predictors].iloc[train,:])
		df_target = patients['cible'].iloc[train]
		model.fit(df_predictors, df_target)
		# predict on test fold
		df_predictions = model.predict(patients[predictors].iloc[test,:])
		predictions.append(df_predictions)
		print('fold #' + str(count) + ' : ' + current_timer())
	time_to('train model')
	predictions = np.concatenate(predictions, axis=0)
	# Printing results report
	print('Model trained on ' + str(patients.shape[0]) + ' observations')
	print('RMSE : ' + str(rmse(predictions, patients['cible']))
		+ ', on ' + str(len(predictors)) + ' predictors.')
	return model

# for submission
def today_now():
	return str(datetime.datetime.now().year) +'-' + str(datetime.datetime.now()
		.month) + '-' + str(datetime.datetime.now().day) + '-' + str(datetime.datetime.now()
		.hour) + '-' + str(datetime.datetime.now().minute)
def make_submission(trained_model, csvfilename):
	'''Makes csv submission at the right format, you choose the name '''
	start_timer()
	predictions = trained_model.predict(patients_test[predictors])
	submission = pd.DataFrame( {"id": patients_test['id'],
		"cible": predictions
		}).sort_index(axis=1, ascending=False)
	submission.loc[patients_test['nb_cmo'] == 0, 'cible'] = 0
	#submission.to_csv(csvfilename + '.csv', index=False, sep=';')
	#print('submitted as : ' + csvfilename + '.csv')
	if want_year_trick :
		print("applying coefficients to target, to inferentiate from years")
		submission['cible'] = submission['cible'] * patients_test['coef_annee']
	submission.to_csv(csvfilename + '.csv', index=False, sep=';')
	print('submitted as : ' + csvfilename + '.csv')
	time_to('predict & save submission')

# action
print('\n- Training & testing models')
print('Parametered models : ' + str(list(models)))
want_train = input('Do you want to train the models ? (y/n) >> '
	) in ['y', 'Y', 'yes', 'YES'] # default : no
if want_train:
	for model_name in models:
		print('\n> Model name : ' + model_name)
		trained_model = train_on_3fold(models[model_name], predictors)
		make_submission(trained_model, today_now() + '-' + model_name)

'''
set_year_coeff([0.07, 0.1449])
make_submission(model_rf, today_now() + '-' + 'rf')
'''

'''# PICKLE (SAVE) models to the disk:
from sklearn.externals import joblib
joblib.dump(model_rf, '/prog/#dsc/kag/dsc28/dump/rf_1_150_8.pkl') '''
# LOAD MODEL BACK
want_load = input('Do you want to load back the previous model ? (y/n) >> '
	) in ['y', 'Y', 'yes', 'YES'] # default : no
if want_load:
	try:
		start_timer()
		model = '/prog/#dsc/kag/dsc28/dump/rf_1_150_8.pkl'
		model_rf = joblib.load(model)
		time_to('load model ' + model)
	except Exception:
		print('No model found in the specified adress')
