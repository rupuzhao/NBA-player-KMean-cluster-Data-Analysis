import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#print("hello")
all_players = pd.read_csv('nba_2018.csv')
print(all_players.head(7))

#get all number of rows and columns
print(all_players.shape)

#average numerical value for all players
print(all_players.mean())

#average value for specific column (average point players scored)
print(all_players.loc[:, 'PTS'].mean())

#pairwise scatter plot
sns.pairplot(all_players[['PTS', 'TRB', 'AST']])
plt.show()

#find the correlations between those 3 values using heat map
corr = all_players[['PTS', 'TRB', 'AST']].corr()
sns.heatmap(corr, annot = True)
plt.show()



#make cluster of players by KMeans
#build the kmeans model
kmeans_model = KMeans(n_clusters = 20, random_state = 1)

#clean data (remove missing value or N/A value)
clean_data = all_players._get_numeric_data().dropna(axis = 1)

#train the kmeans model
kmeans_model.fit(clean_data)

#cluster label for each player
labels = kmeans_model.labels_

# print(labels)
#plot players by cluster
pca2 = PCA(2)
plot_columns = pca2.fit_transform((clean_data))    #transfer and get x and y coordinates
# print(plot_columns)
plt.scatter(x = plot_columns[:, 0], y = plot_columns[:, 1], c = labels)
plt.show()


#find some specific famous players and their data

LBJ = clean_data.loc[all_players['Player'] == 'LeBron James']
# print(LBJ)
KI = clean_data.loc[all_players['Player'] == 'Kyrie Irving']
# print(KI)
SC = clean_data.loc[all_players['Player'] == 'Stephen Curry']
# print(SC)
KK = clean_data.loc[all_players['Player'] == 'Kyle Kuzma']
# print(KK)
JT = clean_data.loc[all_players['Player'] == 'Jayson Tatum']
# print(JT)


# print(clean_data)
#make prediction using kmeans model
#transfer data to list for prediction
lbj_list = LBJ.values.tolist()
ki_list = KI.values.tolist()
sc_list = SC.values.tolist()
kk_list = KK.values.tolist()
jt_list = JT.values.tolist()

#predict and see what cluster does the player belongs to using kmeans model
lbj_predict = kmeans_model.predict(lbj_list)
ki_predict = kmeans_model.predict(ki_list)
sc_predict = kmeans_model.predict(sc_list)
kk_predict = kmeans_model.predict(kk_list)
jt_predict = kmeans_model.predict(jt_list)


#print the result and see those players' cluster
#Lebron James, Kyrie Irving, Stephen Curry, Kyle Kuzma, Jayson Tatum
print(lbj_predict)
print(ki_predict)
print(sc_predict)
print(kk_predict)
print(jt_predict)


# print(all_players.corr())


#linear regression
#split data, 80% training and 20% testing
x1_train, x1_test, y1_train, y1_test = train_test_split(all_players[['PTS']], all_players[['FG']], test_size = 0.2, random_state = 1)
x2_train, x2_test, y2_train, y2_test = train_test_split(all_players[['Age']], all_players[['AST']], test_size = 0.2, random_state = 1)


#build linear regression model
linearR1 = LinearRegression()
linearR2 = LinearRegression()

#training the linear regression model
linearR1.fit(x1_train, y1_train)
linearR2.fit(x2_train, y2_train)

#predict the test data
predict_y1 = linearR1.predict(x1_test)
predict_y2 = linearR2.predict(x2_test)


#print(predict_y)    #predicted value
#print(y_test)       #actual value

#score returns the coefficient of determination R^2 of the prediction
pst_fg_confidence = linearR1.score(x1_test, y1_test)
age_ast_confidence = linearR2.score(x2_test, y2_test)

#the coefficient of field goal and points
print("PST & FG: " + str(pst_fg_confidence))

#the coeffcient of assist and age
print("AGE & AST: " + str(age_ast_confidence))

