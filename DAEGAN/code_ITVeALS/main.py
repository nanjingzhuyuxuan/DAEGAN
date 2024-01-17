import numpy as np
import pandas as pd
import math
from collections import defaultdict
from ITVeALS import ITVeALS
from sklearn.model_selection import train_test_split
from surprise import Reader, Dataset
from surprise import SVD


def split_data(fliepath, sep, test_size):
    data = pd.read_csv(fliepath, sep=sep, header=0,names=['user_id', 'item_id', 'rating', 'timestamp'])
    train_data, test_data = train_test_split(data, test_size=test_size)
    train_data = train_data.sort_values(by=['user_id', 'item_id'], ascending=True)
    test_data = test_data.sort_values(by=['user_id', 'item_id'], ascending=True)
    train_data.to_csv('./data/movielens 100k/train1.csv', sep='\t', index=False, header=True)
    test_data.to_csv('./data/movielens 100k/test1.csv', sep='\t', index=False, header=False)


def get_pre_use_preference(ratingMatrix, F, λ, max_iter, Wui):
    ITV_eALS = ITVeALS(ratingMatrix, F, λ, Wui)
    predictRating = ITV_eALS.iteration_train(max_iter)
    return predictRating


'''
ε:Weight coefficient
days:Control time period size；The number of ratings in each time period needs to be controlled to reflect the difference in popularity
'''
def get_Wui(train_data, ε, days):
    Wui = np.zeros((row, col))
    time_min = train_data['timestamp'].min()
    matrix = np.zeros((row, col))
    for line in train_data.itertuples():
        matrix[line[1] - 1, line[2] - 1] = 1
    user_popularity = np.sum(matrix, axis=1)
    user_popularity_max = np.max(user_popularity)
    user_popularitylist = list(
        map(lambda x: round((math.log(x + 1)) / math.log(user_popularity_max + 1), 2), user_popularity))
    for user in range(row):
        max_time = train_data[train_data['user_id'] == (user + 1)]['timestamp'].max()

        min_time = train_data[train_data['user_id'] == (user + 1)]['timestamp'].min()
        time_r = train_data[(train_data['timestamp'] >= min_time) & (train_data['timestamp'] <= max_time)]

        matrix = np.zeros((row, col))
        for line in time_r.itertuples():
            matrix[line[1] - 1, line[2] - 1] = 1
        item_popularity = np.sum(matrix, axis=0)

        item_popularity_max = np.max(item_popularity)
        if len(time_r) == 0:
            item_popularitylist =  list(
            map(lambda x: 0, item_popularity))
        else:
            item_popularitylist = list(
                map(lambda x: round(math.log(x + 1) / math.log(item_popularity_max + 1), 2), item_popularity))
        for item in range(col):
            Wui[user, item] = ε * user_popularitylist[user] + (1 - ε) * item_popularitylist[item]
    for i in range(row):
        for j in range(col):
            if data_matrix[i, j] == 1:
                Wui[i, j] = 1
    return Wui


def evaluate(pui_matrix, topN):
    for n in topN:
        pre_preferences = pd.DataFrame(pui_matrix, index=range(1, row + 1), columns=range(1, col + 1))
        all_hit = 0
        pre = 0
        rec = 0
        ndcg = 0
        user_count = 0
        for user in range(row):
            u_r = pre_preferences.iloc[user]
            test_r = test_data[test_data['user_id'] == (user + 1)]
            rellist = test_r['item_id'].unique().tolist()
            reclist = list(u_r.sort_values(ascending=False).index)[:n]
            if len(rellist) > 0:
                user_count += 1
                hit = len(list(set(rellist) & set(reclist)))
                all_hit += hit
                pre += hit / n
                rec += hit / len(rellist)
                ndcg += get_ndcg(reclist, rellist)
        precision = pre / user_count
        recall = rec / user_count
        ndcg = ndcg / user_count
        print(f'Precision@{n} :{precision}\t'
              f'Recall@{n} :{recall}\t'
              f'NDCG@{n} :{ndcg}')


def get_ndcg(l1, l2):
    hit = []
    dcg = 0
    idcg = 0
    for i in l1:
        if i in l2:
            hit.append(1)
        else:
            hit.append(0)
    if len(l2) >= len(l1):
        ihit = len(l1)
    else:
        ihit = len(l2)
    for i in range(len(hit)):
        dcg += np.divide(np.power(2, hit[i]) - 1, np.log2(i + 2))
    for i in range(ihit):
        idcg += np.divide(np.power(2, 1) - 1, np.log2(i + 2))
    ndcg = dcg / idcg
    return ndcg


def fill_rating(pui_matrix, θ, ratio, σ):
    filled_rating = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'timestamp'])
    user_list = []
    item_list = []
    for i in range(row):
        pui_u = pui_matrix[i]
        pui_u_nan = [x for x in pui_u if np.isnan(x)]
        pui_u_notnan = [x for x in pui_u if not np.isnan(x)]
        print(len(pui_u_nan))
        print(len(pui_u_notnan))
        index = int(len(pui_u_notnan) * θ)
        print(index)
        uninteresting_items_candidate = pui_u.argsort()[:index]
        pui_u = np.array(pui_u)
        fill_count = int(len(pui_u_nan) * ratio)
        print(fill_count)
        print('-----')
        uninteresting_items = pui_u.argsort()[:fill_count]
        for j in uninteresting_items:
            if j in uninteresting_items_candidate:
                user_list.append(i + 1)
                item_list.append(j + 1)
    filled_rating['user_id'] = user_list
    filled_rating['item_id'] = item_list

    filled_rating['rating'] = 1

    return filled_rating


def topn_svd(train_fill):
    reader = Reader(rating_scale=(0, 5))
    train_set_fill = Dataset.load_from_df(train_fill[['user_id', 'item_id', 'rating']], reader)
    trainset_fill = train_set_fill.build_full_trainset()
    algo = SVD()
    algo.fit(trainset_fill)
    testset = trainset_fill.build_anti_testset()
    predictions = algo.test(testset)
    return predictions


def evaluate_svd(test_data, predictions, topN):
    print('svd algorithm')
    for n in topN:
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]
        count = 0
        pre = 0
        rec = 0
        ndcg = 0
        all_hit = 0
        for user in range(row):
            u_r = top_n[user + 1]
            rec_list = [i[0] for i in u_r]
            test_r = test_data[test_data['user_id'] == (user + 1)]
            rel_list = test_r['item_id'].unique().tolist()
            if len(rel_list) > 0:
                count += 1
                hit = len(list(set(rec_list) & (set(rel_list))))
                all_hit += hit
                pre += hit / n
                rec += hit / len(rel_list)
                ndcg += get_ndcg(rec_list, rel_list)
        precision = pre / count
        recall = rec / count
        ndcg = ndcg / count
        print(f'Precision@{n} :{precision}\t'
              f'Recall@{n} :{recall}\t'
              f'NDCG@{n} :{ndcg}')

def evaluate_error(testdata,filldata):

    filldict =  {str(i + 1): [] for i in range(data_config['user'] + 1)}
    testdict = {str(i + 1): [] for i in range(data_config['user'] + 1)}
    for line in testdata.iterrows():
        testdict[line[0]].append(line[1])
    for line in filldata.iterrows():
        filldict[line[0]].append(line[1])

    err = 0
    num = 0
    for u, i_list in testdict.items():
        fill_list = filldict[u]
        if len(i_list)==0:
            continue
        hit = len(list(set(i_list) & set(fill_list)))
        num+=1
        err += hit / len(i_list)
    err_ave = err/num
    print(err_ave)

if __name__ == '__main__':

    ε =0.5
    days = 31
    F=20
    λ=0.01
    max_iter=20
    θ=0.7
    ratio = 4
    σ=1
    config = {
        '100k':{
            'train_data':'data/movielens 100k/train.csv',
            'test_data':'data/movielens 100k/test.csv',
            'pui_matrix':'data/movielens 100k/pui_matrix.txt',
            'split_flag':'\t',
            'user':943,
            'item':1682,
            'pui_matrix':'data/movielens 100k/pui_matrix.txt',
            'filled_rating':'data/movielens 100k/'
        },
        'ml1m': {
            'train_data': 'data/ml1m/train2.csv',
            'test_data': 'data/ml1m/test2.csv',
            'pui_matrix': 'data/ml1m/pui_matrix2.txt',
            'split_flag': '\t',
            'user': 1000,
            'item': 1606,

            'filled_rating': 'data/ml1m/'
        },
        'am_movie': {
            'train_data': 'data/cd_movies/train27.csv',
            'test_data': 'data/cd_movies/test27.csv',
            'pui_matrix': 'data/cd_movies/pui_matrix27.txt',
            'split_flag': '\t',
            'user': 1000,
            'item': 2004,

            'filled_rating': 'data/cd_movies/'
        },
    }
    data_name = 'ml1m'
    data_config = config[data_name]

    train_data = pd.read_csv(data_config['train_data'], sep=data_config['split_flag'],
                             names=['user_id', 'item_id', 'rating', 'timestamp'])
    test_data = pd.read_csv(data_config['test_data'], sep=data_config['split_flag'],
                            names=['user_id', 'item_id', 'rating', 'timestamp'])
    row = data_config['user']
    col = data_config['item']
    data_matrix = np.zeros((row, col))
    for line in train_data.itertuples():
        data_matrix[line[1] - 1, line[2] - 1] = 1

    Wui = get_Wui(train_data=train_data, ε=ε, days=days)
    pui_matrix = get_pre_use_preference(ratingMatrix=data_matrix, F=F, λ=λ, max_iter=max_iter, Wui=Wui)
    np.savetxt(data_config['pui_matrix'], pui_matrix, fmt='%.4f')
    for i in range(row):
        for j in range(col):
            if data_matrix[i, j] == 1:
                pui_matrix[i, j] = np.nan
    topN = [5, 10, 20]
    print('ITV-eALS algorithm')
    evaluate(pui_matrix, topN)
    filled_rating = fill_rating(pui_matrix, θ=θ, ratio=ratio, σ=σ)
    filled_rating.to_csv(data_config['filled_rating']+'fill{}_2.csv'.format(ratio), sep='\t', index=False,header=False)  # Save filled ratings
