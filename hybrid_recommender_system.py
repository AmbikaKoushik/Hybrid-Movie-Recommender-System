import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from math import sqrt

movies = pd.read_csv('E:\\UC CS\\Machine Learning\\Recommender Systems\\ml-latest-small\\ml-latest-small\\movies.csv')
ratings = pd.read_csv('E:\\UC CS\\Machine Learning\\Recommender Systems\\ml-latest-small\\ml-latest-small\\ratings.csv')
movieData = pd.read_csv('E:\\UC CS\\Machine Learning\\Recommender Systems\\ml-latest-small\\ml-latest-small\\scrapped.csv')
links = pd.read_csv('E:\\UC CS\\Machine Learning\\Recommender Systems\\ml-latest-small\\ml-latest-small\\links.csv')

ratings['frequency'] = ratings.groupby(by = 'movieId')['movieId'].transform('count')

mergedFrame = pd.merge(ratings, movies, on = 'movieId')
mergedFrame = pd.merge(mergedFrame, links, on = 'movieId')
mergedFrame = pd.merge(mergedFrame, movieData, on = 'imdbId')

mergedFrame.drop('tmdbId', axis = 1, inplace=True)
slicedMergedFrame = mergedFrame[mergedFrame['frequency'] >= 1]

genreList = []
translator = str.maketrans('', '', '[]')
for i in slicedMergedFrame['Genre']:
    genres = i.translate(translator).split(',')
    for x in genres:
        if x not in genreList:
            genreList.append(x)

#User based collaborative system

userPivotedData = slicedMergedFrame.pivot(index='userId', columns='movieId', values='rating').fillna(0)
userTrainData, userTestData = train_test_split(userPivotedData, test_size=0.25)

sparseUserTrainData = csr_matrix(userTrainData.values)
sparseUserTestData = csr_matrix(userTestData.values)

userModel = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
userModel.fit(sparseUserTrainData)


#Content based system
uniqueMovieData = slicedMergedFrame.drop_duplicates(subset='movieId')
translator2 = str.maketrans('', '', string.punctuation)

for index, genre in enumerate(genreList):
    x = []
    for i in uniqueMovieData['genreArray']:
        if i[0, index] == 1:
            x.append(1)
        else:
            x.append(0)
    uniqueMovieData[genre.translate(translator2)] = x

select_columns = ['movieId', 'frequency', 'Rating']
for i in uniqueMovieData.columns.values:
    if i not in ['userId', 'title', 'rating', 'timestamp', 'genres', 'imdbId', 'Actors', 'Directors',
                'movieId', 'frequency', 'Rating', 'Genre', 'Release',
       'genreArray', 'directorArrays']:
        select_columns.append(i)
uniqueMovieData = uniqueMovieData[select_columns]

uniqueMovieData.set_index('movieId', inplace= True)
uniqueMovieData.fillna(0, inplace=True)

contentModel = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
contentModel.fit(uniqueMovieData)

#Hybrid System

def findSimilarUsers(userId):
    similarUsers = []
    query_index = userPivotedData.index.get_loc(userId)
    distances, indices = userModel.kneighbors(userPivotedData.iloc[query_index, :].reshape(1, -1), n_neighbors = 5)    

    for i in range(0, len(indices.flatten())):
        similarUsers.append(userPivotedData.index[indices.flatten()[i]])
    return similarUsers

def findSimilarMovies(movieIdList):
    similarMovies = []
    for movieId in movieIdList:        
        query_index = uniqueMovieData.index.get_loc(movieId)
        distances, indices = contentModel.kneighbors(uniqueMovieData.iloc[query_index, :].reshape(1, -1), n_neighbors = 5) 
        for i in range(0, len(indices.flatten())):
            recommendedId = uniqueMovieData.index[indices.flatten()[i]]
            if recommendedId not in movieIdList:
                similarMovies.append(recommendedId)
    return similarMovies

def getMoviesSeenbyUser(userId):
    mov = np.ndarray.tolist(sparseUserTestData[userTestData.index.get_loc(userId)].nonzero()[1])
    for i, x in enumerate(mov):
        mov[i] = userTestData.columns[x]
    return mov

def recommendMoviesForUser(userId):
    similarUsers = findSimilarUsers(userId)
    moviesSeenByUser = getMoviesSeenbyUser(userId)
    similarMovies = findSimilarMovies(moviesSeenByUser)
    userId = []
    movieId = []
    title = []
    rating = []
    for user in similarUsers:
        hybridFrame = slicedMergedFrame[(slicedMergedFrame['userId'] == user) & (slicedMergedFrame['movieId'].isin(similarMovies))][['userId', 'movieId', 'title', 'rating']]
        for index, row in hybridFrame.iterrows():
            userId.append(row['userId'])
            movieId.append(row['movieId'])
            title.append(row['title'])
            rating.append(row['rating'])
            
    columns = ['userId', 'movieId', 'title', 'rating']
    hybridFrame = pd.DataFrame(np.column_stack([userId, movieId, title, rating]), columns=columns)
    return hybridFrame


originalRating = []
predictions = []
def evaluate(userId):
    recommendMoviesForUser(userId)
    for recommendedMovie in hybridFrame['movieId'].unique():
        if int(recommendedMovie) in getMoviesSeenbyUser(userId):
            original = userPivotedData.loc[userId, int(recommendedMovie)]
            originalRating.append(original)
            pred = 0
            for i in hybridFrame['userId'].unique():
                ratingToCompare = userPivotedData.loc[int(i), int(recommendedMovie)]
                if ratingToCompare > pred:
                    pred = ratingToCompare
            predictions.append(pred)

for i in userTestData.index:
    evaluate(i)

rmse = sqrt(mean_squared_error(originalRating, predictions))
normalized_rmse = rmse/5

print('Normalized RMSE: ')
print (normalized_rmse)

