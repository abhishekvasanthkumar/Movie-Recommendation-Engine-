import numpy as np

def  vecNor(vec):
    return np.sqrt(np.sum(np.square(vec)))


def vecFil(vec1, vec2):
    vec1Mod = []
    vec2Mod = []
    if (type(vec1) is np.ndarray) or (type(vec1) is list):
        for x, i in enumerate(vec1):
            y = vec2[x]
            if y > 0 and x > 0:
                vec1Mod.append(x)
                vec2Mod.append(y)
    else:
        for x, i in enumerate(vec1):
            y = vec2[i]
            x = vec1[i]
            if y > 0 and x > 0:
                vec1Mod.append(x)
                vec2Mod.append(y)

    return np.array(vec1Mod), np.array(vec2Mod)


def cosineSimilarity(vecA, vecB):
    aModified, bModified = vecFil(vecA, vecB)

    similar = np.dot(aModified, bModified)

    normVecA =  vecNor(aModified)
    normVecB =  vecNor(bModified)
    if normVecA != 0 and normVecB != 0:
        similar /= (normVecA * normVecB)
    else:
        similar = 0

    if similar > 1:
        similar = 1
    elif similar < -1:
        similar = -1
    ratedA = len([r for r in vecA if r > 0])
    ratedB = len([r for r in vecB if r > 0])
    if ratedA > 0 and ratedB > 0:
        similar *= ((ratedA + ratedB) / (ratedA * ratedB)) * (len(aModified) / 2)
    return similar


def adjustedCosineSimilarity(vecA, vecB, users):
    if not hasattr(adjustedCosineSimilarity, 'avgs'):
        refinedUsers = [[x for x in u if x > 0] for u in users]
        adjustedCosineSimilarity.avgs = [np.mean(u) for u in refinedUsers]

    avgs = adjustedCosineSimilarity.avgs
    adVecA = np.subtract(vecA, np.rint(avgs))
    adVecB = np.subtract(vecB, np.rint(avgs))
    temp = 0
    for val in adVecA:
        if val == -(np.rint(avgs[temp])):
            adVecA[temp] = 0
            temp += 1

    temp = 0
    for val in adVecA:
        if val == -(np.rint(avgs[temp])):
            adVecB[temp] = 0
            temp += 1

    return cosineSimilarity(adVecA, adVecB)


def pearsonCorrelation(vecA, vecB):
    # This function is used to calculate the pearson correlation between vec A and vec B.

    filtA, filtB = vecFil(vecA, vecB)
    meanA = filtA.mean()
    meanB = filtB.mean()
    adjA = np.subtract(filtA, meanA[np.newaxis])
    adjB = np.subtract(filtB, meanB[np.newaxis])
    numerator = np.dot(adjA, adjB)
    sumOfSquaresA = np.dot(adjA, adjA)
    sumOfSquaresB = np.dot(adjB, adjB)
    denominator = np.sqrt(sumOfSquaresA) * np.sqrt(sumOfSquaresB)
    if denominator == 0:
        return 0
    corr = numerator/denominator
    if -1 > corr > 1:
        print("Yes ", corr)
    # Adding weight-age of their individually rated movies also.
    ratedA = len([r for r in vecA if r > 0])
    ratedB = len([r for r in vecB if r > 0])
    corr *= ((ratedA + ratedB) / (ratedA * ratedB)) * (len(filtA) / 2)
    return corr


delta = 5

def ipTrain(inputData, train_file='train.txt'):
    dataTraining = open(train_file, 'r')
    dataTraining = dataTraining.read().strip().split('\n')
    for i, line in enumerate(dataTraining):
        inputData[i] = [int(scoring) for scoring in line.split()]

#  making sure that all the scorings are between 1 and 5
def fixScorings(scorings):
    temp = 0
    for scoring in scorings:
        scoring = int(np.rint(scoring))
        if scoring > 5:
            scoring = 5
        elif scoring < 1:
            scoring = 1
        scorings[temp] = scoring
        temp += 1
    return scorings


def scoringsPearsonC(inputData, currUsr, currUsrId, movieIds, rHo=None):
    weights = [pearsonCorrelation(currUsr, u) for
               u in inputData]
    temp = 0
    for weight, userOther in zip(weights, inputData):
        common = 0
        for userGivenRating, otherGivenRating in zip(currUsr, userOther):
            if userGivenRating > 0 and otherGivenRating > 0:
                common += 1
        weight *= min(common, 5) / 5
        weights[temp] = weight
        temp += 1

    if rHo is not None:
        weights = [w * np.abs(w) ** (rHo - 1) for w in weights]
    user_averages = [np.average([r for r in u if r > 0]) for u in inputData]
    scorings = []
    r_avg = np.average([x for x in currUsr.values() if x > 0])
    for movie_id in movieIds:
        sum_w = 0
        scoring = 0
        for w, u_other, user_avg in zip(weights, inputData, user_averages):
            u_rating = u_other[movie_id]
            if u_rating == 0:
                continue
            sum_w += np.abs(w)
            scoring += (w * (u_rating - user_avg))

        if sum_w != 0:
            scoring = r_avg + (scoring/sum_w)
        else:
            scoring = r_avg

        scorings.append(np.rint(scoring))

    return fixScorings(scorings)


def scoringsPearsonIUF(inputData, currUsr, currUsrId, movieIds):
    if not hasattr(scoringsPearsonIUF, 'run'):
        scoringsPearsonIUF.run = True
        maximumUsers = len(inputData)
        for i in range(1000):
            userRatedMovies = len([0 for u in inputData if u[i] != 0])
            if userRatedMovies == 0:
                continue
            IUF = np.log(maximumUsers/userRatedMovies)
            for trainUsers in inputData:
                trainUsers[i] *= IUF

    scorings = scoringsPearsonC(inputData, currUsr, currUsrId, movieIds)
    return fixScorings(scorings)


def scoringsPearsonCaseModification(users, user, user_id, movie_ids):
    scorings = scoringsPearsonC(users, user, user_id, movie_ids, rHo=2.5)
    return scorings


def scoringsCosine(inputData, currUsr, currUsrId, movieIds):
    weights = [cosineSimilarity(currUsr, u) for
               u in inputData]

    scorings = []
    for movie_id in movieIds:
        sum_w = 0
        scoring = 0

        for w, u_other in zip(weights, inputData):
            u_rating = u_other[movie_id]
            if u_rating == 0:
                continue
            if w > 0.3:
                sum_w += w
                scoring += (w * u_rating)

        if sum_w != 0:
            scoring /= sum_w
        else:
            scoring = 3

        scoring = int(np.rint(scoring))
        scorings.append(scoring)
    return fixScorings(scorings)


def scoringsItemBased(inputData, currUsr, currUsrId, movieIds):
    items = np.array(inputData).T
    scorings = []
    user_items = list(currUsr.keys())
    for movie_id in movieIds:
        item = items[movie_id]
        weights = [adjustedCosineSimilarity(items[i], item, inputData)
                   for i in user_items]
        sum_w = 0
        scoring = 0
        for weight, userOther in zip(weights, inputData):
            common = 0
            for userGivenRating, otherGivenRating in zip(currUsr, userOther):
                if userGivenRating > 0 and otherGivenRating > 0:
                    common += 1
            weight *= min(common, delta) / delta
        for w, i in zip(weights, user_items):
            u_rating = currUsr[i]

            sum_w += np.abs(w)
            scoring += (w * u_rating)

        if sum_w != 0:
            scoring /= sum_w
        else:
            scoring = 3

        scoring = int(np.rint(scoring))
        scorings.append(scoring)

    return fixScorings(scorings)


#Own Implementation Alogrithm
def scoringsItemBasedRelative(inputData, currUsr, currUsrId, movieIds):
    items = np.array(inputData).T
    scorings = []
    user_items = list(currUsr.keys())
    user_averages = [np.average([r for r in u if r > 0]) for u in inputData]

    for movie_id in movieIds:
        item = items[movie_id]
        i_ratings = [r for r in item if r > 0]
        if len(i_ratings) > 0:
            ratingsAvg = np.average(i_ratings)
        else:
            ratingsAvg = 3

        weights = [adjustedCosineSimilarity(items[i], item, inputData)
                   for i in user_items]
        sumWeights = 0
        scoring = 0

        for w, i, user_avg in zip(weights, user_items, user_averages):
            userGivenRating = currUsr[i]
            sumWeights += np.abs(w)
            scoring += (w * (userGivenRating - user_avg))

        if sumWeights != 0:
            scoring = ratingsAvg + (scoring/sumWeights)
        else:
            scoring = ratingsAvg

        scoring = int(np.rint(scoring))
        scorings.append(scoring)

    cosineRatings = ratingsCosine(inputData, currUsr, currUsrId, movieIds)
    hybridRatings = [(i + j) / 2 for i, j in zip(scorings, cosineRatings)]
    return fixScorings(hybridRatings)


def findScorings(inputData, currUsr, currUsrId, movieIds, results):
    if len(movieIds) > 0:

        #Cosine Similarity
        scorings = scoringsCosine(inputData, currUsr, currUsrId, movieIds)
        
        
        #Pearson Coefficient
        #scorings = scoringsPearsonC(inputData, currUsr, currUsrId, movieIds)
        
        
        #Pearson IUF
        #scorings = scoringsPearsonIUF(inputData, currUsr, currUsrId, movieIds)
        
        
        #Pearson Case Modification
        #scorings = scoringsPearsonCaseModification(inputData, currUsr, currUsrId, movieIds)
        
        
        #Item Based Collaborative Algorithm
        #scorings = scoringsItemBased(inputData, currUsr, currUsrId, movieIds)
        
        #Own Implementation
        #scorings = scoringsItemBasedRelative(inputData, currUsr, currUsrId, movieIds)
        

        for m_id, r in zip(movieIds, scorings):
            if r < 1 or r > 5:
                raise Exception('Rating %d' % r)
            results.append((currUsrId + 1, m_id + 1, r))


def testDataSet(inputData, trainFile):
    dataset = open(trainFile, 'r').read().strip().split('\n')
    dataset = [data.split() for data in dataset]
    dataset = [[int(e) for e in data] for data in dataset]
    currUsrId = dataset[0][0] - 1
    currUsr = {}
    movieIds = []
    results = []
    for userId, movieId, scoring in dataset:
        userId -= 1
        movieId -= 1
        print('User %d' % userId, end='\r')

        if userId != currUsrId:
            findScorings(inputData, currUsr, currUsrId, movieIds, results)
            currUsrId = userId
            currUsr = {}
            movieIds = []

        if scoring == 0:
            movieIds.append(movieId)
        else:
            currUsr[movieId] = scoring

    findScorings(inputData, currUsr, currUsrId, movieIds, results)
    return results


def outputResults(results, fileName):
    fout = open(fileName, 'w')
    for result in results:
        fout.write(' '.join(str(x) for x in result) + '\n')


def genResultsFile(inputData):
    print('Scoring predictions for file: test5')
    results5 = testDataSet(inputData, 'test5.txt')
    outputResults(results5, 'result5.txt')
    print('Scoring predictions for file: test10')
    results10 = testDataSet(inputData, 'test10.txt')
    outputResults(results10, 'result10.txt')
    print('Scoring predictions for file: test20')
    results20 = testDataSet(inputData, 'test20.txt')
    outputResults(results20, 'result20.txt')


def beginRating():
    totalNumOfUsers = 200
    totalNumOfMovies = 1000
    inputData = [[0] * totalNumOfMovies] * totalNumOfUsers
    ipTrain(inputData, 'train.txt')
    genResultsFile(inputData)

beginRating()