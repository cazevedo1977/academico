# Load libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from pandas import DataFrame
from imblearn.over_sampling import SMOTE
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
import matplotlib.pyplot as plt 
import itertools
from matplotlib import rcParams

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import make_scorer

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.externals.joblib import dump, load 
import pydotplus

#RandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#SVM
from sklearn import svm

#ANN
from sklearn.neural_network import MLPClassifier
import seaborn as sns

import matplotlib.cm as cm
roc_curve_values=[]
confusion_matrix_values=[]


def feature_plot(classifier, feature_names, model_name, top_features=5):
 coef = classifier.coef_.ravel()
 top_positive_coefficients = np.argsort(coef)[-top_features:]
 top_negative_coefficients = np.argsort(coef)[:top_features]
 top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
 plt.figure(figsize=(10, 7))
 colors = ['#3e5c71' if c < 0 else '#7ea9c7' for c in coef[top_coefficients]]
 plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
 feature_names = np.array(feature_names)
 plt.xticks(np.arange(1 + 2 * top_features), feature_names[top_coefficients], rotation=45, ha='right')
 plt.xlabel('Importância da variável')
 plt.ylabel('Variáveis')
 plt.title("Importância das Variáveis - "+ model_name)
 plt.legend()
 plt.savefig('img/ImportanciaVariaveis'+model_name +'.png', dpi=300)
 plt.clf()
 plt.cla()
 plt.close()


def buildLR(df, cluster):
	#variavel dependente
	label_cols=['declined']
	#variavel independente
	quatitative_cols=['valueClient', 'extraCost', 'valueResources', 'net',
		       'processDuration', 'daysOnContact', 'daysOnInterview', 'daysOnSendCV',
		       'daysOnReturn', 'daysOnCSchedule', 'daysOnCRealized']

	for i in quatitative_cols: 
		df[i] = df[i].astype(float)

	#filtrando as amostras do conglomerado
	df=df[df["cluster"]==cluster]

	quantitative_data=DataFrame(df, columns=quatitative_cols)
	labels_data=DataFrame(df, columns=label_cols)
	selected_columns = quantitative_data.columns
	data = quantitative_data.copy()
	quantitative_data=DataFrame(quantitative_data, columns=selected_columns)
	quantitative_data.head()

		
	os = SMOTE(random_state=0)
	X_train, X_test, y_train, y_test = train_test_split(quantitative_data, labels_data, test_size=0.2, random_state=0)
	#Frequencia de amostras de treinamento no cluster antes do SMOTE
	ax = sns.countplot(x="declined", data=y_train, palette="Blues_d")
	plt.subplots_adjust(bottom=0.4)
	for p in ax.patches:
		ax.annotate(str(p.get_height()), (p.get_x()+0.35, p.get_height()+2))
	plt.ylabel('Frequência')
	plt.savefig('img/FrequenciaTreinamentoCluster'+str(cluster)+'.png', dpi=300)
	plt.clf()
	plt.cla()
	plt.close()

	# Balanceamento do conjunto de treinamento
	os_data_X,os_data_y=os.fit_sample(X_train, y_train)
	os_data_X = pd.DataFrame(data=os_data_X,columns=selected_columns )
	os_data_y= pd.DataFrame(data=os_data_y,columns=['declined'])

	X=os_data_X[selected_columns]
	y=os_data_y['declined']

	#Frequencia de amostras de treinamento no cluster depois do Smote
	ax = sns.countplot(x="declined", data=os_data_y, palette="Blues_d")
	plt.subplots_adjust(bottom=0.4)
	for p in ax.patches:
		ax.annotate(str(p.get_height()), (p.get_x()+0.35, p.get_height()+2))
	plt.ylabel('Frequência')
	plt.savefig('img/FrequenciaTreinamentoDepoisSmoteCluster'+str(cluster)+'.png', dpi=300)
	plt.clf()
	plt.cla()
	plt.close()


	#selecao das variaveis em acordo com o p-value
	logit_model=sm.Logit(y,X)
	result=logit_model.fit(method='bfgs')
	print(result.summary2())

	if cluster==0:
		quatitative_cols=['valueClient', 'extraCost', 'valueResources', 'net',
		       'processDuration', 'daysOnSendCV',
		       'daysOnCRealized']
		X=os_data_X[quatitative_cols]
		y=os_data_y['declined']

	elif cluster==1:
		quatitative_cols=['valueClient',  'processDuration']
		X=os_data_X[quatitative_cols]
		y=os_data_y['declined']

	logit_model=sm.Logit(y,X)
	result=logit_model.fit(method='bfgs')
	print(result.summary2())

	hyperparameters=[{'solver': ['liblinear'], 'penalty': ['l1'], 'C': np.logspace(0, 4, 10)}, {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'], 'penalty': ['l2'], 'C': np.logspace(0, 4, 10)}]
	#Contrucao do modelo depois de selecionar as variaveis
	model_measures = buildModel('accuracy', LogisticRegression(max_iter=10000, tol=0.001), hyperparameters, "Regressão Logística (Cluster"+str(cluster)+")" , os_data_X[quatitative_cols], os_data_y, X_test[quatitative_cols], y_test, quatitative_cols)
	return model_measures


def buildModels(df):

	#variavel dependente
	label_cols=['declined']

	#variavel independente
	quatitative_cols=['valueClient', 'extraCost', 'valueResources', 'net',
		       'processDuration', 'daysOnContact', 'daysOnInterview', 'daysOnSendCV',
		       'daysOnReturn', 'daysOnCSchedule', 'daysOnCRealized']

	for i in quatitative_cols: 
		df[i] = df[i].astype(float)

	quantitative_data=DataFrame(df, columns=quatitative_cols)
	labels_data=DataFrame(df, columns=label_cols)
	
	#Separando o conjunto de treinamento e de teste
	X_train, X_test, y_train, y_test = train_test_split(quantitative_data, labels_data, test_size=0.2, random_state=0)
	
	#Frequencia de amostras de treinamento no cluster antes do SMOTE
	ax = sns.countplot(x="declined", data=y_train, palette="Blues_d")
	plt.subplots_adjust(bottom=0.4)
	#Imprimindo as frequencias no grafico
	for p in ax.patches:
		ax.annotate(str(p.get_height()), (p.get_x()+0.35, p.get_height()+10))
	plt.ylabel('Frequência')
	plt.savefig('img/FrequenciaTreinamentoBaseCompleta.png', dpi=300)
	plt.clf()
	plt.cla()
	plt.close()

	#Balanceamento do conjunto de treinamento
	os = SMOTE(random_state=0)
	os_data_X,os_data_y=os.fit_sample(X_train, y_train)
	#Frequencia de amostras de treinamento em cada classe depois do SMOTE 
	ax = sns.countplot(x="declined", data=os_data_y, palette="Blues_d")
	plt.subplots_adjust(bottom=0.4)
	#Imprimindo as frequencias no grafico
	for p in ax.patches:
		ax.annotate(str(p.get_height()), (p.get_x()+0.35, p.get_height()+10))
	plt.ylabel('Frequência')
	plt.savefig('img/FrequenciaTreinamentoBaseCompletaDepoisSmote.png', dpi=300)
	plt.clf()
	plt.cla()
	plt.close()

	#Estabelecimento dos conjuntos de parâmetros para cada modelo
	estimators = [10, 50, 100, 200, 500, 1000, 2000] 
	max_depths = [3, 6, 10, 15, 20, 30]

	tuned_parameters_mlp = [{'hidden_layer_sizes': [(1,), (2,), (3,), (4,), (5,), (1,1,), (1,2,), (1,3,), (1,4,), (1,5,), (2,1,), (2,2,), (2,3,), (2,4,), (2,5,), (3,1,), (3,2,), (3,3,), (3,4,), (3,5,), (4,1,), (4,2,), (4,3,), (4,4,), (4,5,), (5,1,), (5,2,), (5,3,), (5,4,), (5,5,)],
                      'activation' : ['identity', 'logistic', 'tanh', 'relu'],
                    'learning_rate': ['constant', 'adaptive'],
                    'alpha': [0.0001, 0.001, 0.01, 0.1, 1]}]

	tuned_parameters_svm = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]
	
	#Para diminuir o tempo de execucao do codigo, os arrays acima devem ser comentados e então descomentar as duas linhas abaixo, as quais ja contem os melhores parametros
	tuned_parameters_mlp = {'activation': ['tanh'], 'alpha': [0.1], 'hidden_layer_sizes': [(5, 2)], 'learning_rate': ['adaptive']}
	tuned_parameters_svm = [{'kernel': ['rbf'], 'gamma': [0.001], 'C': [1000]}]

	tuned_parameters_rf = {'n_estimators': estimators, 'max_depth':max_depths}
	tuned_parameters_dt = {'criterion':['gini','entropy'],'max_depth':[2,3,4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150,200,300]}

	tuned_parameters=[tuned_parameters_dt, tuned_parameters_svm, tuned_parameters_rf, tuned_parameters_mlp]

	names = ["Árvore de Decisão", "SVM", "Floresta Aleatória", "Rede Neural"]

	classifiers = [DecisionTreeClassifier(), SVC(probability=True), RandomForestClassifier(), MLPClassifier()]

	#Definindo acuracia como sendo a medida de qualidade do modelo no processo de otimizaçao
	scores = ['accuracy']
	models_measures = pd.DataFrame(columns = ['Model', 'AUC', 'Acurácia', 'Precisão', 'Revocação', 'Medida-F'])

	#Contrucao dos modelos
	for score in scores:
		for classifier, tuned_parameter, name in zip (classifiers, tuned_parameters, names):
			model_measures = buildModel(score, classifier, tuned_parameter, name, os_data_X, os_data_y, X_test, y_test, quatitative_cols)
			models_measures=models_measures.append(model_measures, ignore_index=True)
			
	return models_measures


def buildModel(score, classifier, tuned_parameter, name, os_data_X, os_data_y, X_test, y_test, quatitative_cols):
			print("# Ajustando os hyper-parametros para  o modelo "+ name + ", considerendo a métrica: " + score)
			print()

			clf = GridSearchCV(classifier, tuned_parameter, scoring=score, cv=5)

			print("Construindo o modelo " + name)

			clf.fit(os_data_X, os_data_y.values.ravel())

			print("Melhores parametros obtidos com o conjunto de treinamento para o modelo "+ name)
			print()
			print(clf.best_params_)
			print()
			print("Tabela de resultados para cada combinação de parâmetros: "+ name)
			print()
			means = clf.cv_results_['mean_test_score']
			stds = clf.cv_results_['std_test_score']
			for mean, std, params in zip(means, stds, clf.cv_results_['params']):
				print("%0.3f (+/-%0.03f) para os parametros %r" % (mean, std * 2, params))
			print()

			print("Relatorio de medidas do modelo  "+ name)
			print()
			y_true, y_pred = y_test, clf.predict(X_test)
			print(classification_report(y_true, y_pred))
			print()

			confusion_matrix_obj= {'Model': name, 'y_test': y_test, 'y_pred': y_pred}
			confusion_matrix_values.append(confusion_matrix_obj)

			sns.set()
			fpr, tpr, thresholds = roc_curve(y_true, y_pred)
			roc_auc = auc(fpr,tpr)
			if 'Regressão' not in name:
				roc_object = {'Model': name, 'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}
				roc_curve_values.append(roc_object)
			plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
			plt.legend(loc='lower right')
			plt.plot([0,1],[0,1],'r--')
			plt.xlim([-0.1,1.0])
			plt.ylim([-0.1,1.01])
			plt.ylabel('Taxa de Verdadeiro Positivo')
			plt.xlabel('Taxa de Falso Positivo')
			plt.savefig('img/CurvaROC'+name +'.png', dpi=300)
			plt.clf()
			plt.cla()
			plt.close()

			auc_lr = roc_auc_score(y_true, y_pred)
			accuracy_lr = accuracy_score(y_true, y_pred)
			precision_lr = precision_score(y_true, y_pred)
			recall_lr = recall_score(y_true, y_pred)
			f1_lr = f1_score(y_true, y_pred)

			print ("Métricas para "+ name)
			print ("AUC = %.2f" % auc_lr)
			print ("Accuracy = %.2f" % accuracy_lr)
			print ("Precision = %.2f" % precision_lr)
			print ("Recall = %.2f" % recall_lr)
			print ("F1 Score = %.2f" % f1_lr)
	
			model_measures=pd.DataFrame(data = {'Model':name, 'AUC': [auc_lr], 'Acurácia' : [accuracy_lr], 'Precisão': [precision_lr], 'Revocação': [recall_lr], 'Medida-F': [f1_lr]})

			#Investigacao da importancia das variaveis
			feature_imp= pd.Series([])
			try:
				feature_imp = pd.Series(clf.best_estimator_.feature_importances_,index=quatitative_cols).sort_values(ascending=False)
			except:
				try:
					feature_plot(classifier, quatitative_cols, name, top_features=5)
				except Exception as inst:
					print(type(inst))
					print("O classificador não possui valores para a importância das variáveis")
				
			#Graficos de importancia das variaveis para os modelos
			if feature_imp.shape[0]>0:
				fig = plt.figure(figsize = (15,15))
				ax = fig.gca()
				rcParams.update({'figure.autolayout': True})
				rcParams.update({'font.size': 20})
				sns.set(font_scale=2)
				sns.barplot(x=feature_imp, y=feature_imp.index, ax=ax, palette="Blues_d")
				plt.xlabel('Importância da variável', fontsize=20)
				plt.ylabel('Variáveis',  fontsize=20)
				ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
				plt.title("Importância das Variáveis - "+ name,  fontsize=20)
				plt.savefig('img/ImportanciaVariaveis'+name +'.png', dpi=300)
				plt.clf()
				plt.cla()
				plt.close()

			if 'Decisão' in name:
				#Desenho da arvore
				clfTree = DecisionTreeClassifier(criterion=clf.best_params_['criterion'], max_depth=clf.best_params_['max_depth'])
				clfTree = clfTree.fit(os_data_X,os_data_y)
				dot_data = StringIO()
				export_graphviz(clfTree, out_file=dot_data,  
								filled=True, rounded=True,
								special_characters=True,feature_names = quatitative_cols,class_names=['0','1'])
				graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
				graph.write_png('img/Arvore.png')
				Image(graph.create_png())
		
			elif "SVM" in name and "linear" in clf.best_params_['kernel']:
				#Importancia das variaveis para o SVM, caso tenha nucleo linear
				feature_plot(clf, quatitative_cols, name, top_features=5)
	
			return model_measures

#Inicio da execucao
#lendo os dados
df = pd.read_csv("data/kmeans_result_zscore.csv")
measures = pd.DataFrame(columns = ['Model', 'AUC', 'Acurácia', 'Precisão', 'Revocação', 'Medida-F'])

#Implementando os modelos de regressão Logística para cada cluster
model_measuresRL0=buildLR(df, 0)
model_measuresRL1=buildLR(df, 1)

#Concatenando os resultados dos clusters
y_test_new=np.append (confusion_matrix_values[0]['y_test'], confusion_matrix_values[1]['y_test'])
y_pred_new=np.append (confusion_matrix_values[0]['y_pred'], confusion_matrix_values[1]['y_pred'])

fpr, tpr, thresholds = roc_curve(y_test_new, y_pred_new)
roc_auc = auc(fpr,tpr)
roc_object = {'Model': 'Regressão Logística', 'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}
roc_curve_values.append(roc_object)
confusion_matrix_obj= {'Model': 'Regressão Logística', 'y_test': y_test_new, 'y_pred': y_pred_new}
confusion_matrix_values=[]
confusion_matrix_values.append(confusion_matrix_obj)

auc_lr = roc_auc_score(y_test_new, y_pred_new)
accuracy_lr = accuracy_score(y_test_new, y_pred_new)
precision_lr = precision_score(y_test_new, y_pred_new)
recall_lr = recall_score(y_test_new, y_pred_new)
f1_lr = f1_score(y_test_new, y_pred_new)

#Armazenando os resultados
model_measures=pd.DataFrame(data = {'Model':'Regressão \nLogística', 'AUC': [auc_lr], 'Acurácia' : [accuracy_lr], 'Precisão': [precision_lr], 'Revocação': [recall_lr], 'Medida-F': [f1_lr]})
measures=measures.append(model_measures, ignore_index=True)

#Contrucao dos demais modelos
model_measures=buildModels(df)
measures=measures.append(model_measures, ignore_index=True)

print("Consolidando os resultados dos modelos... ")
print()
#Matriz Confusao
for cm_obj in confusion_matrix_values:
	conf_matrix = confusion_matrix(cm_obj['y_test'], cm_obj['y_pred'])
	thresh = (conf_matrix.max() / 1.5)
	rcParams.update({'figure.autolayout': True})
	rcParams['axes.titlepad'] = 40 
	cmap = plt.get_cmap('Blues')
	fig = plt.figure(figsize=(16,10))
	ax = fig.add_subplot(111)
	# Hide grid lines
	ax.grid(False)
	sns.set(font_scale=2)
	cax = ax.matshow(conf_matrix, interpolation='nearest', cmap=cmap)
	plt.title('Matriz Confusão do Classificador '+ cm_obj['Model'], pad=35)
	for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
		plt.text(j, i, "{:,}".format(conf_matrix[i, j]), horizontalalignment="center", color="white" if conf_matrix[i, j] > thresh else "black")
	labels = ['Declinou (declined=1)', 'Continou no Processo \n Seletivo (declined=0)']
	fig.colorbar(cax)
	ax.set_xticklabels([''] + labels, fontsize= 20)
	ax.set_yticklabels([''] + labels, fontsize= 20)
	plt.xlabel('Predito', fontsize= 20)
	plt.savefig('img/MatrizConfusao'+cm_obj['Model']+'.png', dpi=300)
	plt.clf()
	plt.cla()
	plt.close()

#Graficos de resultados
fig, axs = plt.subplots(ncols=1, nrows=5, constrained_layout=True, figsize=(15,20))
fig.tight_layout(pad=1)
sns.set(font_scale=1.5)
axs[0].plot(measures['Model'].values, measures['AUC'].values)
axs[0].set_title('AUC', fontsize=20, pad=-0.5)
axs[1].plot(measures['Model'].values, measures['Acurácia'].values)
axs[1].set_title('Acurácia', fontsize=20)
axs[2].plot(measures['Model'].values, measures['Precisão'].values)
axs[2].set_title('Precisão', fontsize=20)
axs[3].plot(measures['Model'].values, measures['Revocação'].values)
axs[3].set_title('Revocação', fontsize=20)
axs[4].plot(measures['Model'].values, measures['Medida-F'].values)
axs[4].set_title('Precisão', fontsize=20)
axs[4].set_title('Medida-F', fontsize=20)
plt.savefig('img/Results.png', dpi=300)
plt.clf()
plt.cla()
plt.close()

#Curvas ROCs de todos os modelos em um grafico
sns.set(font_scale=1)
cmap = cm.get_cmap("rainbow")
colors = cmap(np.linspace(0, 1, len(roc_curve_values)))
for roc_obj, c in zip(roc_curve_values, colors):
	plt.plot(roc_obj['fpr'], roc_obj['tpr'], 'b', label=roc_obj['Model']+' (AUC = %0.3f)'% roc_obj['roc_auc'], color=c)

plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'b--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('Taxa de Verdadeiro Positivo')
plt.xlabel('Taxa de Falso Positivo')
plt.savefig('img/CurvaROC.png', dpi=300)
plt.clf()
plt.cla()
plt.close()

