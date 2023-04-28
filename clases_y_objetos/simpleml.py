# basic
import time
import os

# ds
import pandas as pd
import numpy as np

# for plots
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef as mcc
# import the f1 score
from sklearn.metrics import f1_score, make_scorer
# import the recall score
from sklearn.metrics import recall_score, precision_score
# import CalibratedClassifierCV for continuous probability
from sklearn.calibration import CalibratedClassifierCV
# import proportions_ztest
from statsmodels.stats.proportion import proportions_ztest
# import scipy
import scipy
# to specify a function
from typing import Callable, List
# import ColumnTransformer and FunctionTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from IPython.display import display
import category_encoders as ce
import inspect
from sklearn.base import TransformerMixin, BaseEstimator
from termcolor import colored
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression, LassoCV, LinearRegression
from sklearn.ensemble import RandomForestClassifier



from sklearn.preprocessing import StandardScaler, RobustScaler





class CustomEncoders():
	class IntegerTargetEncoder():
		def __init__(self,col_names = None, model = None):
			self.col_names = col_names
			if model == None:
				self.model = LinearRegression()	
			else:
				self.model = model
				
			self.avg_value_dic = {}
			self.fitted_model_dic = {}
		def fit_column(self,x:pd.core.frame.DataFrame,y:pd.core.series.Series):
			# get the name of the x column
			col_name = list(x.columns)[0]
			# copy the dataframe so that we can modify it without problem
			full_df = x.copy()
			# pass the target variable to the full dataframe
			full_df['y'] = y
			# get the average value of the target column, when the x column is null
			avg_value = full_df[full_df[col_name].isnull()].y.mean()
			# get the dataframe of non null values so that we can fit the model
			non_null = full_df[full_df[col_name].notnull()]
			# get the class model
			tmp_model = self.model
			# get x to train
			x = non_null[[col_name]]
			y = non_null[['y']]
			# fit the model
			tmp_model = tmp_model.fit(x,y)
			# add the values to the class dictionaries
			self.avg_value_dic[col_name] = avg_value
			self.fitted_model_dic[col_name] = tmp_model
		def fit(self,X,y):
			if self.col_names == None:
				self.col_names = list(X.columns)
			for col_name in self.col_names:
				x = X[[col_name]]
				self.fit_column(x,y)
		def transform_col(self,x:pd.core.frame.DataFrame):
			col_name = list(x.columns)[0]
			# we will separate x into nan and non nan, for nan values we will impute the average value
			# and for non nan values we will predict using
			avg_value = self.avg_value_dic[col_name]
			model = self.fitted_model_dic[col_name]
			def predict_value(age):
				if pd.isna(age):
					ans = avg_value
				else:
					ans = model.predict([[age]])[0][0]
				return ans
			ans_col = x[col_name].apply(lambda x: predict_value(x))
			# now we will scale using standard scaler
			scaler = StandardScaler()
			ans_col = scaler.fit_transform(ans_col.to_numpy().reshape(-1, 1))
			ans_col = pd.Series(ans_col.flatten())
			return ans_col
			
		def transform(self, X:pd.core.frame.DataFrame):
			for col_name in self.col_names:
				x = X[[col_name]]
				ans = self.transform_col(x)
				X.loc[:, col_name] = ans
			return X
		
		def fit_transform(self,X,y):
			self.fit(X,y)
			ans = self.transform(X)
			return ans
				
			
		def debug(self):
			print(self.col_names)
			print(self.model)
			print(self.avg_value_dic)
			print(self.fitted_model_dic)








class pipeline():





	def __init__(self,train,target):
		"""
		train: the dataframe
		target: the name of the target column
		"""

		X_train, X_test, y_train, y_test = train_test_split(train.drop(columns = [target]), train[target])
		# save the variables
		self.xr =X_train
		self.xe = X_test
		self.yr =y_train
		self.ye = y_test

		self.steps = []
		self.commits = [[self.xr.copy(),self.xe.copy(), self.yr.copy(), self.ye.copy()]]
		self.unique_counter = 0

		self.target_col = y_train.name

		self.pdf = train.copy()
		self.feature_columns = self.pdf.columns.tolist()
		self.cols_prev = []







	class ColumnTransformerDf(TransformerMixin, BaseEstimator):
		def __init__(self,scaler, columns=None):
			self.columns = columns
			self.scaler = scaler
	
		def fit(self, X, y=None):
			if self.columns:
				if y is not None:
					self.scaler.fit(X[self.columns], y)
				else:
					self.scaler.fit(X[self.columns])
			else:
				if y is not None:
					self.scaler.fit(X,y)
				else:
					self.scaler.fit(X)
			return self
	
		def transform(self, X, y=None):
			if self.columns:
				X[self.columns] = self.scaler.transform(X[self.columns])
			else:
				X = self.scaler.transform(X)
			return pd.DataFrame(X, columns=X.columns)



	def _colTransformer(self,function:Callable,columns:list):
		Name = type(function).__name__
		for col in columns:
			Name += "_" + col
		step = (Name,self.ColumnTransformerDf(function,columns = columns))
		self.steps.append(step)





	def _funTransformer(self, function:Callable):
		Name = function.__name__
		ft = FunctionTransformer(function)
		step = (Name, ft)
		self.steps.append(step)

	def _apply_function(self,function:Callable):
		self.xr = function(self.xr)
		self.xe = function(self.xe)


	def _apply_column_transformer(self,function:Callable,cols):
		""" we will trainsform the columns based on a sklearn.preprocessing class or based in a category_encoders class"""
		encoder = function
		#if inspect.getmodule(function) == inspect.getmodule(FunctionTransformer):
		self.xr.loc[:, cols] = encoder.fit_transform(self.xr[cols], self.yr)
		self.xe.loc[:, cols] = encoder.transform(self.xe[cols])
	
	def _check_function(self,function, variable_name):
		function_source = inspect.getsource(function)
		if variable_name in function_source:
			print(colored("WARNING: you are using " + variable_name + " the pipeline might fail", "red"))	



	def _check_return(self,function, variable_name):
		function_source = inspect.getsource(function)
		if variable_name in function_source:
			print(colored("WARNING: you are using " + variable_name + " the pipeline might fail", "red"))	

	def _update_pdf(self,X_train,X_test,y_train,y_test):
		X_df = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_test)], axis=0).reset_index(drop=True)
		# Combine y_train and y_test into a single dataframe
		y_df = pd.concat([pd.DataFrame(y_train), pd.DataFrame(y_test)], axis=0).reset_index(drop=True)
		# Combine X_df and y_df column-wise
		self.pdf = pd.concat([X_df, y_df], axis=1)

	def wr(self, function:Callable,columns = None):
		"""
		funtion: the function to append to the pipeline
		columns: the list of columns to apply the transformation if you are using a pipeline
		"""
		self.cols_prev = self.pdf.columns.tolist().copy()
		if columns == None:
			self._check_function(function, "p.xr")
			self._check_function(function, "p.xe")
			self._check_function(function, "p.yr")
			self._check_function(function, "p.ye")
			self._check_function(function, "p.pdf")
			self._check_return(function, "return X")
			self._apply_function(function)
			self._funTransformer(function)
		else:
			self._apply_column_transformer(function, columns)
			self._colTransformer(function, columns)

		# to update the plotting pdf
		self._update_pdf(self.xr,self.xe,self.yr,self.ye)
		self.commits.append([self.xr.copy(), self.xe.copy(), self.yr.copy(), self.ye.copy()])
		self.feature_colums = self.pdf.columns.tolist().copy()
		self.report_status()


	def __str__(self):
		ans= ""
		ans += " =============== STEPS ============= :\n"
		for step in self.steps:
			ans += step[0] + "\n"
		ans += " =============== END STEPS ============= :\n"


		deleted_cols = set(self.cols_prev) - set(self.pdf.columns)
		added_cols = set(self.pdf.columns) - set(self.cols_prev)

		tmpstr = ""
		ans += "current columns:\n"
		for col in self.pdf.columns:
			ans += col + " "
			tmpstr += col + " "
			if len(tmpstr) >60:
				tmpstr = ""
				ans += "\n"


		tmpstr = ""
		if added_cols:
			ans += "\nadded columns: \n"
			for col in added_cols:
				ans += colored(col, "green") + " "
				tmpstr += col + " "
				if len(tmpstr) >60:
					tmpstr = ""
					ans += "\n"
		ans += "\n"

		tmpstr = ""
		# Print deleted columns in red on a new line
		if deleted_cols:
			ans += "\nDeleted columns: \n"
			for col in deleted_cols:
				ans += colored(col, "red") + " "
				tmpstr += col + " "
				if len(tmpstr) >60:
					tmpstr = ""
					ans += "\n"
		ans += "\n"
		return ans

	def report_status(self):
		print(self.__str__())
		display(self.pdf.head(2))

	def getSteps(self):
		return self.steps

	def pop(self):
		""" pop a step from the steps """
		if len(self.steps) == 0:
			print("nothing to pop")
			return
		stepname = self.steps[-1]
		self.steps = self.steps[:-1]
		self.commits = self.commits[:-1]
		self.xr = self.commits[-1][0].copy()
		self.xe = self.commits[-1][1].copy()
		self.yr = self.commits[-1][2].copy()
		self.ye = self.commits[-1][3].copy()
		self._update_pdf(self.xr,self.xe,self.yr,self.ye)
		print("popped " + stepname[0])



	def getPipeline(self):
		""" get the pipeline """
		return Pipeline(steps = self.steps)

	def getCommits(self):
		""" to get the commits list"""
		return self.commits


























class plot():
	def stacked_bar(self,df, pivot, target, horizontal = False):
		"""
		Plots a stacked bar chart of the target variable and a pivot columns
		df: the dataframe with the target column and the pivot column
		pivot: str, name of pivot column
		target: str, name of target column
		horizontal: bool, if True, the plot will be horizontal
		"""
		pivot_column = pivot
		target = target
		tmp_df = df[[pivot_column,target]]
		# now create the counts dataframe
		columns = tmp_df[pivot_column].unique()
		rows = tmp_df[target].unique()
		counts = pd.DataFrame(columns = columns)

		# append row by row the necessary values
		for row in rows:
			arr = []
			for col in columns:
				value = len(tmp_df[(tmp_df[pivot_column] == col) & (tmp_df[target] == row)])
				arr.append(value)
			counts.loc[row] = arr
		counts = counts.T
		pro = counts.div(counts.sum(axis=1), axis=0)
		# plot
		if horizontal:
			ax = pro.plot(kind='barh', figsize=(len(rows)//2 + 4,max(4,len(columns) - len(columns)*7 // 20)), stacked=True)
		else:
			ax = pro.plot(kind='bar', figsize=(max(4,len(columns) - len(columns)*7 // 20),len(rows) // 2 + 4), stacked=True)
		# move legend
		ax.legend(bbox_to_anchor=(1, 1.01), loc='upper left')
		# column names from per used to get the column values from df
		cols = pro.columns
		# iterate through each group of containers and the corresponding column name
		for c, col in zip(ax.containers, cols):
			# get the values for the column from df
			vals = counts[col]
			# create a custom label for bar_label
			labels = []
			for v, val in zip(c,vals):
				label = ""
				w = v.get_width() if horizontal else v.get_height()
				if w > 0:
					label = f'{val}--{w*100:.1f}%'
				labels.append(label)
			# annotate each section with the custom labels
			ax.bar_label(c, labels=labels, fontsize = 7, label_type='center', rotation = 40,weight = 'bold')
		if not horizontal:
			plt.xticks(rotation = 90)
		else:
			plt.yticks(rotation = 30)
		plt.title(pivot + " values with target:" + target)
		plt.show()
		# box plot

	def boxplots(self,arr,labels):
		"""
		arr: an array of the data
		labels: an array of the labels
		"""
		fig, ax = plt.subplots(figsize = (10,10))
		ax.boxplot(arr, labels = labels)
		plt.show()
	def confussion_matrix(self,y_test, y_pred):
		"""
		plot the confussion matrix
		y_test: the test data
		y_pred: the predicted data
		"""
		cm = confusion_matrix(y_test, y_pred)
		text = [['true positive', 'false positive'], ['false negative', 'true negative']]
		data = np.array([[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]])
		# combining text with values
		formatted_text = (np.asarray([["{0}: {1}".format(text, data) for text, data in zip(row, col)] for row, col in zip(text, data)]))
		sns.heatmap(data, annot = formatted_text, fmt = '', cmap = 'Blues')

	def _absolute_value_and_percentage(self,val,sum_vals):
		abs_val = np.round(val/100*sum_vals,0)
		return "{:.1f}%--({:d})".format(val, int(abs_val))
	def pie_plot(self,data,target, pivot,ncols = 3, nrows = -1):
		"""
		data: the data
		target: the target
		pivot: the column we will be visualizing
		"""
		# get a list of all the unique values in the pivot column
		unique = data[pivot].unique()
		# order the unique array from most common to least common
		commonality = []
		for i in unique:
			commonality.append([i,len(data[data[pivot] == i])])
		commonality = sorted(commonality, key = lambda x: x[1], reverse = True)
		unique = [i[0] for i in commonality]

		lu = len(unique)
		# get the number of rows and columns
		if nrows == -1:
			nrows = int(np.ceil(lu/ncols))
		# create a figure
		fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize = (ncols*5,nrows*5))
		# loop through the unique values
		for i in range(lu):
			target_unique = data[target].unique()
			target_counts = {}
			for j in range(len(target_unique)):
				target_counts[target_unique[j]] = len(data[(data[pivot] == unique[i]) & (data[target] == target_unique[j])])
			values = target_counts.values()
			keys = target_counts.keys()
			if max(values) == 0:
				values = [1]
				keys = ['No data']
			# get the total number of samples
			total = sum(values)

			# plot the pie chart
			ax.flat[i].pie(values, labels = keys, autopct = lambda val: self._absolute_value_and_percentage(val,total))
			ax.flat[i].set_title(str(unique[i]) + ' (samples:' + str(total) + ')')
		# delete non used axes
		for i in range(lu, nrows*ncols):
			fig.delaxes(ax.flat[i])

		plt.tight_layout()
		plt.show()


	def pair_grid(self,data, hue = None, regplot = False):
		# measure the time it takes to run the code
		prev = time.time()
		column_names = data.columns.to_list()
	
		# plot the data using seaborn
		if hue == None:
			g = sns.PairGrid(data)
		else:
			if data[hue].dtype == 'object' or data[hue].dtype == "category":
				# Use a larger palette of colors for categorical variables
				n_categories = len(data[hue].unique())
				palette = sns.color_palette("husl", n_colors=n_categories)
			else:
				data["tmpcat123456"] = pd.cut(data[hue], bins=np.linspace(data[hue].min(), data[hue].max(), len(data[hue].unique())), labels=False)
				# Use blue as the lower limit and red as the upper limit for numerical variables
				n_colors = len(data["tmpcat123456"].unique())
				palette = sns.color_palette("Set1", n_colors=n_colors)
				if (1, 1, 1) in palette:
					palette.remove((1, 1, 1))  # Remove white if present
					palette.insert((0,0,0))
				data = data.drop(columns = ['tmpcat123456'],axis = 1)


			g = sns.PairGrid(data, hue = hue,palette = palette)
	
		g.map_upper(sns.scatterplot, s = 6)
		g.map_lower(sns.kdeplot, alpha = 0.5)
		g.map_diag(sns.histplot, kde = True, alpha = 0.5)
	
		xlabels,ylabels = [],[]
	
		# get the x labels
		for ax in g.axes[-1,:]:
			xlabel = ax.xaxis.get_label_text()
			xlabels.append(xlabel)
	
		# get the y labels
		for ax in g.axes[:,0]:
			ylabel = ax.yaxis.get_label_text()
			ylabels.append(ylabel)
	
		# set all the axes
		for i in range(len(xlabels)):
			for j in range(len(ylabels)):
				g.axes[j,i].xaxis.set_label_text(xlabels[i], visible = True)
				g.axes[j,i].yaxis.set_label_text(ylabels[j],visible = True)
	
		for ax in g.axes.flat:
			# get the ax position
			pos = ax.get_position()
			# check if the ax is below the main diagonal
			# example
			# 0 1 2
			# 3 4 5
			# 6 7 8
			# if the ax is 3, 6, 7 then it is below the main diagonal
	
	
			xlabel = ax.xaxis.get_label_text()
			ylabel = ax.yaxis.get_label_text()
			xindex = column_names.index(xlabel)
			yindex = column_names.index(ylabel)
	
			below = False
			if xindex < yindex:
				below = True
	
	
			# set xlabel and y label text visible
			ax.tick_params(labelleft = True, labelbottom = True)
			ax.xaxis.label.set_visible(True)
			ax.yaxis.label.set_visible(True)
			ax.grid(True)
	
			color_palette = ["blue","orange","green"]
			# create a regplot for the current ax
			if hue != None and regplot == True and below == False:
				colors = {}
				for i in range(len(data[hue].unique())):
					colors[data[hue].unique()[i]] = color_palette[i%len(color_palette)]
				if ax.get_xlabel() != ax.get_ylabel():
					tcounter = 0
					for cat in data[hue].unique():
						tmp_data = data[data[hue] == cat]
						sns.regplot(x = ax.get_xlabel(), y = ax.get_ylabel(), data = tmp_data, ax = ax, color = colors[cat], scatter = False, line_kws = {"alpha":0.5})
						tcounter+=1
	
		g.tight_layout()
		# show hue legend
		g.add_legend()
		print( "time(s) : "  + str(time.time() - prev))
	
	
	

class SeabornFig2Grid():

	def __init__(self, seaborngrid, fig,  subplot_spec):
		self.fig = fig
		self.sg = seaborngrid
		self.subplot = subplot_spec
		if isinstance(self.sg, sns.axisgrid.FacetGrid) or isinstance(self.sg, sns.axisgrid.PairGrid):
			self._movegrid()
		elif isinstance(self.sg, sns.axisgrid.JointGrid):
			self._movejointgrid()
		self._finalize()

	def _movegrid(self):
		""" Move PairGrid or Facetgrid """
		self._resize()
		n = self.sg.axes.shape[0]
		m = self.sg.axes.shape[1]
		self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
		for i in range(n):
			for j in range(m):
				self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

	def _movejointgrid(self):
		""" Move Jointgrid """
		h= self.sg.ax_joint.get_position().height
		h2= self.sg.ax_marg_x.get_position().height
		r = int(np.round(h/h2))
		self._resize()
		self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

		self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
		self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
		self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

	def _moveaxes(self, ax, gs):
		#https://stackoverflow.com/a/46906599/4124317
		ax.remove()
		ax.figure=self.fig
		self.fig.axes.append(ax)
		self.fig.add_axes(ax)
		ax._subplotspec = gs
		ax.set_position(gs.get_position(self.fig))
		ax.set_subplotspec(gs)

	def _finalize(self):
		plt.close(self.sg.fig)
		self.fig.canvas.mpl_connect("resize_event", self._resize)
		self.fig.canvas.draw()

	def _resize(self, evt=None):
		self.sg.fig.set_size_inches(self.fig.get_size_inches())






























class feature_selection():
	# =============
	# ====  for a target get the kendall values =====
	# =============
	# get_kendall function
	def get_kendall(self,data,target,proportion):
		"""
		gets the kendall correlation given a cleaned dataset
		# get_kendall(data,target)
		# data: the cleaned dataset
		# target: the target column
		# proportion: the proportion of coluns that you want to get
	
		# returns a list of column names
		"""
		kendall = data.corr(method = 'kendall')
		# apply absolute value to the kendall correlation
		kendall = kendall.abs()
		kendall = kendall[target].sort_values(ascending = False)
		print(kendall)
		# get the top 30% of the columns
		kendall = kendall[:int(len(kendall) * proportion)]
		# get the columns
		kendall = list(kendall.index)
		return kendall

	# =============
	# ==== for categorical values to get the values in a column that are statistically significant =====
	# =============

	def _get_relevant_column_values_with_categorical_target(self,pandas_dataframe, column_name, target_name,target_value, looking = 'two-sided', significance = 0.05, minsamples_per_cat = 100):
		"""
		pandas_dataframe: a pandas dataframe containing the column_name and the target_val_name
		column_name: the name of the column
		target_name: the name of the target column
		looking: two-sided( default ) , larger ( get values that predict the value ), smaller ( get the values that predict smaller values for the target)
		target_value: the value in the target that you are trying to predict
		"""
	
		relevant_values = []
		significance_values = []
		total_num_rows = len(pandas_dataframe)
		ratio = len(pandas_dataframe[pandas_dataframe[target_name] == target_value]) / total_num_rows
		# get the array of unique values
		array_unique_values = pandas_dataframe[column_name].unique()
		for val in array_unique_values:
			target_val_num_rows = len(pandas_dataframe[(pandas_dataframe[column_name] == val) & (pandas_dataframe[target_name] == target_value)])
			non_target_num_rows = len(pandas_dataframe[(pandas_dataframe[column_name] == val) & (pandas_dataframe[target_name] != target_value)])
			rows_total = target_val_num_rows + non_target_num_rows
			if rows_total < minsamples_per_cat:
				continue
			# we perform a ttest
			stat, pvalue = proportions_ztest(count = target_val_num_rows, nobs = rows_total, value = ratio, alternative = looking)
			if pvalue <= significance:
				significance_values.append(pvalue)
				relevant_values.append(val)
		return relevant_values, significance_values

	def _get_relevant_column_values_with_numerical_target(self,pandas_dataframe, column_name, target_name, looking = 'two-sided', significance = 0.05,minsamples_per_cat = 100):
		"""
		pandas_dataframe: a pandas dataframe containing the column_name and the target_val_name
		column_name: the name of the column
		target_name: the name of the target column
		looking: two-sided( default ) , larger ( get values that predict the value ), smaller ( get the values that predict smaller values for the target)
		significance: the significance level
		"""
		relevant_values = []
		significance_values = []

		# get the array of unique values
		array_unique_values = pandas_dataframe[column_name].unique()


		for val in array_unique_values:

			# now get the mean of the unique value 
			sample = np.array(pandas_dataframe[target_name].values)
			subsample = np.array(pandas_dataframe[pandas_dataframe[column_name] == val][target_name].values)
			if len(subsample) < minsamples_per_cat:
				continue

			# we will test if they have equal variances 
			stat, tpvalue = scipy.stats.levene(sample, subsample)
			pvalue = 10
			if tpvalue <= 0.05:
				# we perform a ttest
				stat, pvalue = scipy.stats.ttest_ind(sample, subsample, equal_var = False, alternative = looking)
			else:
				# we perform a ttest
				stat, pvalue = scipy.stats.ttest_ind(sample, subsample, equal_var = True, alternative = looking)
			if pvalue <= significance:

				relevant_values.append(val)
				significance_values.append(pvalue)


		return relevant_values, significance_values

	def getColumnRelevantVals(self,ctype, pandas_dataframe, column_name, target_name, looking = 'two-sided', target_value = "none", significance = 0.05, ploting = True, log = False,minsamples_per_cat = 100):
		"""
		ctype: 'cat','num' the type of the target column
		pandas_dataframe: a pandas dataframe containing the column_name and the target_val_name
		column_name: the name of the column
		target_name: the name of the target column
		looking: two-sided( default ) , larger ( get values that predict the value ), smaller ( get the values that predict smaller values for the target)
		"""
		vals = []




		if ctype == 'num':
			vals, signif =  self._get_relevant_column_values_with_numerical_target(pandas_dataframe, column_name, target_name, looking, significance = significance, minsamples_per_cat = minsamples_per_cat)
		else:
			vals, signif = self._get_relevant_column_values_with_categorical_target(pandas_dataframe, column_name, target_name,target_value, looking = looking, significance = significance, minsamples_per_cat = minsamples_per_cat)

		if ploting:
			if len(vals) == 0:
				print("we got no values fro plotting from " + str(column_name))
				return vals


		# Create a figure and axes
		fig, ax = plt.subplots(1, len(vals), figsize=(4 * len(vals), 5))
		
		# Ensure ax is always a list even if there's only one axis
		if len(vals) == 1:
			ax = [ax]
		
		# Iterate through unique values and plot boxplots
		for i, val in enumerate(vals):
			if log:
				tmp_vals = np.log(pandas_dataframe[pandas_dataframe[column_name] == val][target_name].values + 1)
				whole_vals = np.log(pandas_dataframe[target_name].values + 1)
			else:
				tmp_vals = pandas_dataframe[pandas_dataframe[column_name] == val][target_name].values
				whole_vals = pandas_dataframe[target_name].values
		
			# Plot the boxplot
			ax[i].boxplot([whole_vals, tmp_vals])
			ax[vals.index(val)].set_title(str(val) + "\n" + str(signif[vals.index(val)]))	
			
			# Set xtick labels
			ax[i].set_xticklabels(['sample', 'subsample'])
		fig.suptitle('Boxplots for each value in ' + str(column_name), fontsize=16)
		plt.tight_layout()
		plt.show()

		return vals




	def correlation_selector(self,X, y, threshold=0.5):
		selected_features = []
		for column in X.columns:
			corr, _ = pearsonr(X[column], y)
			if abs(corr) >= threshold:
				selected_features.append((column, 1))
			else:
				selected_features.append((column, 0))
		return pd.DataFrame(selected_features, columns=["feature_name", "selected"])
	
	def chi_square_selector(self,X, y, k=10):
		selector = SelectKBest(chi2, k=k)
		selector.fit(X, y)
		selected_features = [(column, int(included)) for column, included in zip(X.columns, selector.get_support())]
		return pd.DataFrame(selected_features, columns=["feature_name", "selected"])
	
	def recursive_feature_elimination(self,X, y, n_features_to_select=10):
		estimator = LogisticRegression(solver="liblinear")
		selector = RFE(estimator, n_features_to_select=n_features_to_select)
		selector.fit(X, y)
		selected_features = [(column, int(included)) for column, included in zip(X.columns, selector.get_support())]
		return pd.DataFrame(selected_features, columns=["feature_name", "selected"])
	
	def lasso_selection(self,X, y, threshold=1e-5):
		lasso = LassoCV(cv=5).fit(X, y)
		selected_features = [(column, int(abs(coef) > threshold)) for column, coef in zip(X.columns, lasso.coef_)]
		return pd.DataFrame(selected_features, columns=["feature_name", "selected"])
	
	def select_from_model(self,X, y):
		selector = SelectFromModel(RandomForestClassifier(n_estimators=100))
		selector.fit(X, y)
		selected_features = [(column, int(included)) for column, included in zip(X.columns, selector.get_support())]
		return pd.DataFrame(selected_features, columns=["feature_name", "selected"])
	
	def feature_selection_report(self,X, y):
		corr_results = self.correlation_selector(X, y)
		chi2_results = self.chi_square_selector(X, y)
		rfe_results = self.recursive_feature_elimination(X, y)
		lasso_results = self.lasso_selection(X, y)
		sfm_results = self.select_from_model(X, y)
	
		report = pd.DataFrame(X.columns, columns=["feature_name"])
		report = report.merge(corr_results, on="feature_name", how="left", suffixes=("", "_corr"))
		report = report.merge(chi2_results, on="feature_name", how="left", suffixes=("_corr", "_chi2"))
		report = report.merge(rfe_results, on="feature_name", how="left", suffixes=("_chi2", "_rfe"))
		report = report.merge(lasso_results, on="feature_name", how="left", suffixes=("_rfe", "_lasso"))
		report = report.merge(sfm_results, on="feature_name", how="left", suffixes=("_lasso", "_sfm"))
	
		report.columns = [
			"feature_name",
			"selected_corr",
			"selected_chi2",
			"selected_rfe",
			"selected_lasso",
			"selected_sfm",
		]
		return report



















# create a class called test
class test():
	def _sampler(self, tr, target, balancer = None, X_train = None, y_train = None):
		if balancer != None:
			# check if it is an array of balancers or a simple balancer
			if type(balancer) == list:
				for i in range(len(balancer)):
					X_train, y_train = balancer[i].fit_resample(X_train, y_train)
			else:
				# balance the data
				X_train, y_train = balancer.fit_resample(X_train, y_train)
		return X_train, y_train

	def _accuracy(self, y_test, y_pred, acc):
		if acc == 'accuracy':
			return accuracy_score(y_test, y_pred)
		elif acc == 'f1':
			return f1_score(y_test, y_pred)
		elif acc == 'mcc':
			return mcc(y_test, y_pred)
		elif acc == 'recall':
			return recall_score(y_test, y_pred)
		# else raise an error
		else:
			raise ValueError('The accuracy method is not valid')
	

	def test_xgboost(self,tr,target, balancer = None, plot = False, acc = None):
		train = tr.copy()
		# get a random number to set the random state
		rand = np.random.randint(0,1000)
		# the accuracy will be a dictionary of each accuracy method
		accuracy = {}
		# now we will split the data to train and test a xgboost model
		X_train, X_test, y_train, y_test = train_test_split(train.drop(columns = [target]), train[target], test_size = 0.2, random_state = 42)

		X_train, y_train = self._sampler(tr, target, balancer, X_train, y_train)

		# we will create a random forest model
		model = XGBClassifier()
		model.fit(X_train, y_train)
		# we will test the accuracy of the model
		y_pred = model.predict(X_test)
		# get the accuracy of the model
		accuracy['accuracy'] = accuracy_score(y_test, y_pred)
		accuracy['mcc'] = mcc(y_test, y_pred)
		accuracy['f1'] = f1_score(y_test, y_pred)
		accuracy['recall'] = recall_score(y_test, y_pred)

		# create a matrix of the confussion matrix for each different possible value in target
		#conf_matrix = confusion_matrix(y_test, y_pred)
		conf_matrix = pd.crosstab(y_test, y_pred, rownames = ['Actual'], colnames = ['Predicted'])
		# plot the confusion matrix
		#sns.heatmap(conf_matrix, annot = True)
		#plt.show()
		if plot:
			sns.heatmap(conf_matrix, annot = True, fmt = 'g')
			plt.title('xgboost')
			plt.show()


		if acc is not None:
			accuracy = accuracy[acc]
		return accuracy, conf_matrix,model
	
	def test_kneighbors(self,tr, target,balancer = None, plot = False, acc = 'accuracy'):
		train = tr.copy()
	
		# split the data
		X_train, X_test, y_train, y_test = train_test_split(train.drop(columns = [target]), train[target], test_size = 0.2, random_state = 42)


		X_train, y_train = self._sampler(tr, target, balancer, X_train, y_train)

		# test accuracy
		acc_arr = []
		max_accuracy = -1
		model = None
		for i in range(1,20):
			knn = KNeighborsClassifier(n_neighbors = i)
			knn.fit(X_train, y_train)
			tmp_y_pred = knn.predict(X_test)
			accuracy = self._accuracy(y_test, tmp_y_pred, acc)
			acc_arr.append(accuracy)
			if accuracy > max_accuracy:
				max_accuracy = accuracy
				model = knn
				y_pred = tmp_y_pred

		conf_matrix = pd.crosstab(y_test, y_pred, rownames = ['Actual'], colnames = ['Predicted'])
		# plot the confusion matrix
		#sns.heatmap(conf_matrix, annot = True)
		#plt.show()
		if plot:
			sns.heatmap(conf_matrix, annot = True, fmt = 'g')
			plt.title('kneighbors')
			plt.show()

		# returns an array of the accuracy of each k
		return max_accuracy, conf_matrix,model
	
	def test_descision_tree(self,tr, target, balancer = None, plot = False, acc = None):
		train = tr.copy()
		# split the data
		X_train, X_test, y_train, y_test = train_test_split(train.drop(columns = [target]), train[target], test_size = 0.2, random_state = 42)

		X_train, y_train = self._sampler(tr, target, balancer, X_train, y_train)

		# test accuracy
		acc_arr = []
		model = None
		best_accuracy = -1
		for i in range(1,8):
			dt = DecisionTreeClassifier(max_depth = i)
			dt.fit(X_train, y_train)
			tmp_y_pred = dt.predict(X_test)
			accuracy = self._accuracy(y_test, tmp_y_pred, acc)
			acc_arr.append(accuracy)
			if accuracy > best_accuracy:
				best_accuracy = accuracy
				model = dt
				y_pred = tmp_y_pred




		conf_matrix = pd.crosstab(y_test, y_pred, rownames = ['Actual'], colnames = ['Predicted'])
		# plot the confusion matrix
		#sns.heatmap(conf_matrix, annot = True)
		#plt.show()
		if plot:
			sns.heatmap(conf_matrix, annot = True, fmt = 'g')
			plt.title('descision_tree')
			plt.show()

		# returns an array of the accuracy of each max_depth
		return best_accuracy, conf_matrix,model
	
	
	def test_svc(self,tr, target, balancer = None, plot = False, acc = None):
		train = tr.copy()
		# split the data
		X_train, X_test, y_train, y_test = train_test_split(train.drop(columns = [target]), train[target], test_size = 0.2, random_state = 42)



		X_train, y_train = self._sampler(tr, target, balancer, X_train, y_train)

		rand = np.random.randint(0,1000)
		# train the model
		svc = SVC(gamma = 'auto', random_state = rand)


		model_calibrated = CalibratedClassifierCV(svc, method='sigmoid')

		model_calibrated.fit(X_train, y_train)
		y_pred = model_calibrated.predict(X_test)

		accuracy = {}
		accuracy['accuracy'] = accuracy_score(y_test, y_pred)
		accuracy['mcc'] = mcc(y_test, y_pred)
		accuracy['f1'] = f1_score(y_test, y_pred)
		accuracy['recall'] = recall_score(y_test, y_pred)


		conf_matrix = pd.crosstab(y_test, y_pred, rownames = ['Actual'], colnames = ['Predicted'])
		# plot the confusion matrix
		#sns.heatmap(conf_matrix, annot = True)
		#plt.show()
		if plot:
			sns.heatmap(conf_matrix, annot = True, fmt = 'g')
			plt.title('SVC')
			plt.show()

		model = model_calibrated
		if acc is not None:
			accuracy = accuracy[acc]

		return accuracy, conf_matrix,model
	
	def test_stocastic_gradient_descent(self,tr, target, balancer = None, plot = False, acc = None):
		train = tr.copy()
		rand = np.random.randint(0,1000)
		# split the data
		X_train, X_test, y_train, y_test = train_test_split(train.drop(columns = [target]), train[target], test_size = 0.2, random_state = 42)


		X_train, y_train = self._sampler(tr, target, balancer, X_train, y_train)

		sgd = SGDClassifier(penalty = 'l2',max_iter = 4500, tol = 1e-3, random_state = rand)
		model_calibrated = CalibratedClassifierCV(sgd, method='sigmoid')

		model_calibrated.fit(X_train, y_train)
		y_pred = model_calibrated.predict(X_test)


		accuracy = {}
		accuracy['accuracy'] = accuracy_score(y_test, y_pred)
		accuracy['mcc'] = mcc(y_test, y_pred)
		accuracy['f1'] = f1_score(y_test, y_pred)
		accuracy['recall'] = recall_score(y_test, y_pred)


		conf_matrix = pd.crosstab(y_test, y_pred, rownames = ['Actual'], colnames = ['Predicted'])
		# plot the confusion matrix
		#sns.heatmap(conf_matrix, annot = True)
		#plt.show()
		if plot:
			sns.heatmap(conf_matrix, annot = True,fmt = 'g')
			plt.title('SGD')
			plt.show()

		model = model_calibrated


		if acc is not None:
			accuracy = accuracy[acc]

		return accuracy, conf_matrix,model

	def test_random_forest(self, tr, target, balancer = None, plot = False, acc = None):
		train = tr.copy()
		rand = np.random.randint(0,1000)
		# split the data
		X_train, X_test, y_train, y_test = train_test_split(train.drop(columns = [target]), train[target], test_size = 0.2, random_state = 42)


		X_train, y_train = self._sampler(tr, target, balancer, X_train, y_train)
		
		rf = RandomForestClassifier(n_estimators = 1000, random_state = rand)
		rf.fit(X_train, y_train)
		y_pred = rf.predict(X_test)

		accuracy = {}
		accuracy['accuracy'] = accuracy_score(y_test, y_pred)
		accuracy['mcc'] = mcc(y_test, y_pred)
		accuracy['f1'] = f1_score(y_test, y_pred)
		accuracy['recall'] = recall_score(y_test, y_pred)

		conf_matrix = pd.crosstab(y_test, y_pred, rownames = ['Actual'], colnames = ['Predicted'])
		# plot the confusion matrix
		#sns.heatmap(conf_matrix, annot = True)
		#plt.show()
		if plot:
			sns.heatmap(conf_matrix, annot = True, fmt = 'g')
			plt.title('RF')
			plt.show()

		model = rf



		if acc is not None:
			accuracy = accuracy[acc]

		return accuracy, conf_matrix,model



	def _get_scorer(self):
		scoring = {}
		scoring['accuracy'] = make_scorer(accuracy_score)
		scoring['mcc'] = make_scorer(mcc)
		scoring['f1'] = make_scorer(f1_score, average = 'macro')
		scoring['recall'] = make_scorer(recall_score, average = 'macro')
		return scoring





	def grid_params(self,train, target,paramgrid,model, metric = 'accuracy', balancer = None, plot = True, acc = None ):
		"""
		train: the training data containing the target
		target: the target column name
		paramgrid: the parameters to be tested
		model: the model to be tested ex: SVC()
		metric: the metric to be used for the grid search ex: f1,accuracy,recall
		balancer: the balancer to be used ex: SMOTE()
		plot: if the confusion matrix should be plotted
		"""
		train = train.copy()
		rand = np.random.randint(0,1000)
		# split the data
		X_train, X_test, y_train, y_test = train_test_split(train.drop(columns = [target]), train[target], test_size = 0.2, random_state = rand)
		X_train, y_train = self._sampler(train, target, balancer, X_train, y_train)

		scoring = self._get_scorer()





		gridSerach = GridSearchCV(estimator = model, param_grid = paramgrid, scoring = scoring, n_jobs = -1, verbose = 10, refit = metric)

		# now perform the whole fit on the pipeline
		gridSerach.fit(X_train, y_train)
		y_pred = gridSerach.predict(X_test)

		accuracy = {}
		accuracy['accuracy'] = accuracy_score(y_test, y_pred)
		accuracy['mcc'] = mcc(y_test, y_pred)
		accuracy['f1'] = f1_score(y_test, y_pred)
		accuracy['recall'] = recall_score(y_test, y_pred)

		conf_matrix = pd.crosstab(y_test, y_pred, rownames = ['Actual'], colnames = ['Predicted'])

		best_model = gridSerach.best_estimator_

		if plot:
			sns.heatmap(conf_matrix, annot = True, fmt = 'g')
			plt.title('Grid Search' + str(gridSerach.best_params_) + " " + str(model))
			plt.show()

		return accuracy, conf_matrix, best_model


