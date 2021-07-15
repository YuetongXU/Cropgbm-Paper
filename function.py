# -*- coding: utf-8 -*-
import random
import seaborn
import numpy as np
import pandas as pd
from pandas import read_csv
import scipy.stats as stats
from multiprocessing import Pool
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import lightgbm as lgb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


colorlist = ['#B22222', '#F08080', '#FF0000', '#006400', '#3CB371', '#2E8B57', '#00FF7F', '#00FF00', '#7FFF00',
             '#FFFF00', '#EEB422', '#FF6666', '#FF9900', '#996633', '#836FFF', '#4876FF', '#0000FF', '#00008B',
             '#1C86EE', '#63B8FF', '#00BFFF', '#87CEFF', '#00688B', '#996666', '#FF00FF', '#8B008B', '#BF3EFF',
             '#912CEE', '#FF83FA', '#551A8B']
colorarray = np.array(colorlist)


def prepare_cv_data(phe_data, save_path, cv_times, cvfold):
    sample_block = int(phe_data.shape[0] / 5)
    phe_index = phe_data.index.to_numpy(copy=True)
    for cvi in range(cv_times):
        random.shuffle(phe_index)
        for i in range(cvfold):
            if i == cvfold - 1:
                phe_data.loc[phe_index[sample_block * i:], 'cv'+str(cvi)] = i
            else:
                phe_data.loc[phe_index[sample_block * i: sample_block * (i + 1)], 'cv'+str(cvi)] = i
    phe_data.sort_index(inplace=True)
    phe_data.to_csv(save_path, header=True, index=True)
    return phe_data


def prepare_params(method, grid_search_params_dict):
    param_list = []

    for learning_rate in grid_search_params_dict['learning_rate_list']:
        for max_depth in grid_search_params_dict['max_depth_list']:
            for n_estimators in grid_search_params_dict['n_estimators_list']:
                for min_data_in_leaf in grid_search_params_dict['min_data_in_leaf_list']:
                    for num_leaves in grid_search_params_dict['num_leaves_list']:

                        params_dict = {
                            'learning_rate': learning_rate,
                            'max_depth': max_depth,
                            'n_estimators': n_estimators}

                        if method == 'gb':
                            params_dict.update({'min_samples_leaf': min_data_in_leaf,
                                                'max_leaf_nodes': num_leaves})
                        elif method == 'lgb':
                            params_dict.update({'min_child_samples': min_data_in_leaf,
                                                'num_leaves': num_leaves})
                        elif method == 'xgb':
                            pass
                        elif method == 'cb':
                            params_dict.update({'min_data_in_leaf': min_data_in_leaf,
                                                'num_leaves': num_leaves})
                        else:
                            raise ValueError('method == [gb, lgb, xgb, cb]')

                        param_list.append(params_dict)

    return param_list


def prepare_header(method):
    header = ['learning_rate', 'max_depth', 'n_estimators']
    if method == 'gb':
        header.extend(['min_samples_leaf', 'max_leaf_nodes'])
    elif method == 'lgb':
        header.extend(['min_child_samples', 'num_leaves'])
    elif method == 'xgb':
        pass
    else:
        header.extend(['min_data_in_leaf', 'num_leaves'])
    header.append('pearson')
    return header


def train_predict(geno_data, phe_data, method, params_dict, phe_name: str, cv_time: str, cvfold: int):

    n_job = 10
    pearson_list = []

    for i in range(cvfold):
        train_phe = phe_data[phe_data[cv_time] != i][phe_name].dropna(axis=0)
        test_phe = phe_data[phe_data[cv_time] == i][phe_name].dropna(axis=0)
        train_geno = geno_data.loc[train_phe.index.values, :]
        test_geno = geno_data.loc[test_phe.index.values, :]

        if method == 'lgb':
            reg = LGBMRegressor(**params_dict, random_state=2, n_jobs=n_job, silent=-1)
        elif method == 'gb':
            reg = GradientBoostingRegressor(**params_dict, random_state=2, verbose=0, max_features='sqrt')
        elif method == 'xgb':
            reg = XGBRegressor(**params_dict, booster='gbtree', random_state=2, verbosity=0, n_jobs=n_job)
        else:
            reg = CatBoostRegressor(**params_dict, grow_policy='Lossguide', random_state=2,
                                    verbose=0, thread_count=n_job)

        reg.fit(train_geno, train_phe)
        predict_phe = reg.predict(test_geno)
        pearson_i, p_value_i = stats.pearsonr(predict_phe, test_phe)
        pearson_list.append(pearson_i)

    pearson = np.mean(pearson_list)

    return pearson


def grid_search(geno_data, phe_data, save_path, phe_name: str, cv_time: str, cvfold: int, pool_max: int,
                method: str, grid_search_params_dict: dict):

    params_dict_list = prepare_params(method, grid_search_params_dict)

    pool = Pool(pool_max)

    pearson_list = []
    for params_dict in params_dict_list:
        pearson_list.append(pool.apply_async(
            train_predict, [geno_data, phe_data, method, params_dict, phe_name, cv_time, cvfold]))

    pool.close()
    pool.join()

    header = prepare_header(method)

    for i, pearson in enumerate(pearson_list):
        params_dict_list[i]['pearson'] = pearson.get()

    with open(save_path, 'w+') as file:
        print('\t'.join(header))
        file.write('\t'.join(header) + '\n')

        for params_dict in params_dict_list:
            params_values = []
            for i in header:
                params_values.append(str(params_dict[i]))

            print('\t'.join(params_values))
            file.write('\t'.join(params_values) + '\n')


def cv(geno_data, phe_data, params_dict: dict):

    cv_times = params_dict['cv_times']
    cvfold = params_dict['cvfold']
    pool_max = params_dict['pool_max']
    method = params_dict['method']
    phe_name = params_dict['phe_name']

    train_params = {
        'learning_rate': params_dict['learning_rate'],
        'max_depth': params_dict['max_depth'],
        'n_estimators': params_dict['n_estimators']
    }

    if method == 'gb':
        train_params.update({'min_samples_leaf': params_dict['min_data_in_leaf'],
                             'max_leaf_nodes': params_dict['num_leaves']})
    elif method == 'lgb':
        train_params.update({'min_child_samples': params_dict['min_data_in_leaf'],
                            'num_leaves': params_dict['num_leaves']})
    elif method == 'cb':
        train_params.update({'min_data_in_leaf': params_dict['min_data_in_leaf'],
                            'num_leaves': params_dict['num_leaves']})

    pool_list = []
    pool = Pool(pool_max)
    for cv_time in range(cv_times):
        cv_time = 'cv' + str(cv_time)
        pool_list.append(pool.apply_async(
            train_predict, [geno_data, phe_data, method, train_params, phe_name, cv_time, cvfold]))

    pool.close()
    pool.join()

    pearson_list = []
    for pearson in pool_list:
        pearson_list.append(pearson.get())

    return pearson_list


def train_cv_tree_model(geno_data, phe_data, model_savepath, params_dict, n_estimators,
                        phe_name: str, cv_time: str, cvfold: int):

    for i in range(cvfold):
        print(cv_time, i, phe_name)
        train_phe = phe_data[phe_data[cv_time] != i][phe_name].dropna(axis=0)
        train_geno = geno_data.loc[train_phe.index.values, :]
        train_set = lgb.Dataset(train_geno, label=train_phe)
        train_boost = lgb.train(params_dict, train_set, n_estimators)
        train_boost.save_model(model_savepath + 'fold' + str(i) + '.lgb_model')


def lgb_cv_tree_model(geno_data, phe_data, savepath, params_dict: dict):

    cv_times = params_dict['cv_times']
    cvfold = params_dict['cvfold']
    pool_max = params_dict['pool_max']
    phe_name = params_dict['phe_name']

    train_params = {
        'learning_rate': params_dict['learning_rate'],
        'max_depth': params_dict['max_depth'],
        'min_data_in_leaf': params_dict['min_data_in_leaf'],
        'num_leaves': params_dict['num_leaves'],
        'objective': 'regression',
        'num_threads': 10
    }
    n_estimators = params_dict['n_estimators']

    pool_list = []
    pool = Pool(pool_max)
    for cv_time in range(cv_times):
        cv_time = 'cv' + str(cv_time)
        savepath_model = savepath + cv_time + '_'
        pool_list.append(pool.apply_async(
            train_cv_tree_model, [geno_data, phe_data, savepath_model, train_params,
                                  n_estimators, phe_name, cv_time, cvfold]))

    pool.close()
    pool.join()


def labcolor_dict(labels_set):

    labels_num = len(labels_set)

    if labels_num > 30:
        # Put back sampling
        color_index = np.random.randint(0, 30, size=labels_num)
    else:
        # No return sampling
        color_index = np.random.choice(range(0, 30), labels_num, replace=False)

    labelscolor_array = colorarray[color_index]
    labelscolor_dict = dict(zip(labels_set, labelscolor_array))

    return labelscolor_dict


def phe_plot(dir_path, species):
    phe_path = dir_path + species + '_pheno.csv'
    phe_data = read_csv(phe_path, header=0, index_col=0)

    phe_name_list = phe_data.columns
    for phe_name in phe_name_list:
        save_plot = dir_path + species + '_' + phe_name + '_plotdd.pdf'
        pdf = PdfPages(save_plot)
        plt.figure(figsize=(15, 15))
        phe_data[phe_name].plot(kind='hist', bins=30)
        plt.tick_params(labelsize=20)
        plt.ylabel('')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        pdf.close()


def geno_plot(dir_path, species):
    geno_path = dir_path + species + '_geno.csv'
    geno_data = read_csv(geno_path, header=0, index_col=0)

    pca = PCA(n_components=0.95)
    redim_array = pca.fit_transform(geno_data)

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(redim_array)

    save_path = dir_path + species
    with PdfPages(save_path + '_redim.pdf') as pdf:
        plt.figure(figsize=(9, 9))
        plt.scatter(redim_array[:, 0], redim_array[:, 1])
        plt.title('Dimensionality reduction')

        plt.xlabel('PC1')
        plt.ylabel('PC2')

        plt.tight_layout()
        pdf.savefig()
        plt.close()

    with PdfPages(save_path + '_cluster.pdf') as pdf:
        predictgroup_set = set(kmeans.labels_)
        labelscolor_dict = labcolor_dict(predictgroup_set)
        plt.figure(figsize=(9, 9))

        for ilabel in labelscolor_dict:
            idata = redim_array[kmeans.labels_ == ilabel]
            plt.scatter(idata[:, 0], idata[:, 1], c=labelscolor_dict[ilabel])
        plt.scatter(redim_array[kmeans.labels_ == -1, 0], redim_array[kmeans.labels_ == -1, 1],
                    c='#CCCCCC', marker='v', alpha=0.6, label='no_cluster')
        plt.title('Cluster')
        plt.legend()

        plt.xlabel('PC1')
        plt.ylabel('PC2')

        plt.tight_layout()
        pdf.savefig()
        plt.close()


def plot_heatmap(heatmap_data, save_path, vmax=None, vmin=None, cmp='YlOrRd'):
    heatmap_data = heatmap_data.T
    pdf = PdfPages(save_path)
    plt.figure(figsize=(30, 15))
    seaborn.heatmap(heatmap_data, vmax=vmax, vmin=vmin, cmap=cmp)
    plt.xticks(size=10)
    plt.yticks(size=10)
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    pdf.close()


def extree_info(model_path, num_boost_round, tree_info_dict=None):
    if tree_info_dict is None:
        tree_info_dict = {}
        for tree_index in range(0, num_boost_round):
            tree_info_dict['tree_' + str(tree_index)] = {}

    for i, row in enumerate(open(model_path)):
        if row.find('feature_names') != -1:
            features_name_list = row.strip().split('=')[-1].split(' ')
            continue
        if row.find('Tree=') != -1:
            tree_index = row.strip().split('=')[-1]
            continue
        if row.find('split_feature') != -1:
            features_index_list = row.strip().split('=')[-1].split(' ')
            features_index_list = [int(i) for i in features_index_list]
            continue
        if row.find('split_gain') != -1:
            features_gain_list = row.strip().split('=')[-1].split(' ')
            features_gain_list = [float(i) for i in features_gain_list]
            seq_index = 0
            for index in features_index_list:
                feature_name = features_name_list[index]
                feature_gain = features_gain_list[seq_index]
                try:
                    tree_info_dict['tree_' + tree_index][feature_name].append(feature_gain)
                except KeyError:
                    tree_info_dict['tree_' + tree_index][feature_name] = [feature_gain]
                seq_index += 1

    return tree_info_dict


def lgb_iter_feature(bygain_feature_array, trainfile_data, trainphe_data, params_dict,
                     cv_times, num_boost_round, savepath_prefix):
    print('lgb_iter_feature_cv_all is in progress. This process will take a long time, please wait patiently')

    exfeature_num = bygain_feature_array.shape[0]
    params_dict['verbosity'] = -1

    # bygain
    bygain_mse = []
    for i in range(1, exfeature_num + 1):
        features = bygain_feature_array[:i]
        features_gene = trainfile_data.loc[:, features]
        train_set = lgb.Dataset(features_gene, label=trainphe_data)
        cv_result = lgb.cv(params_dict, train_set, num_boost_round=num_boost_round, stratified=False)
        imse = cv_result['l2-mean'][-1]
        bygain_mse.append(imse)

    with PdfPages(savepath_prefix + '_bygain.pdf') as pdf:
        plt.figure(figsize=(9, 9))
        plt.scatter(range(1, exfeature_num + 1), bygain_mse)
        plt.xticks(range(1, len(bygain_feature_array) + 1), bygain_feature_array, rotation=90, size=6)
        plt.title('ByGain')
        plt.ylabel('MSE')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # random
    random_imse = []
    for i in range(cv_times):
        snpid_array = trainfile_data.columns.values
        random_array = np.random.choice(range(trainfile_data.shape[1]), exfeature_num, replace=False)
        random_feature_array = snpid_array[random_array]
        random_jmse = []
        for j in range(1, exfeature_num + 1):
            features = random_feature_array[:j]
            features_gene = trainfile_data.loc[:, features]
            train_set = lgb.Dataset(features_gene, label=trainphe_data)
            cv_result = lgb.cv(params_dict, train_set, num_boost_round=num_boost_round, stratified=False)
            if params_dict['objective'] == 'regression':
                jmse = cv_result['l2-mean'][-1]
            else:
                jmse = cv_result['multi_logloss-mean'][-1]
            random_jmse.append(jmse)
        random_imse.append(random_jmse)
    random_mse_data = pd.DataFrame(random_imse)

    with PdfPages(savepath_prefix + '_random.pdf') as pdf:
        fig, axes = plt.subplots(figsize=(9, 9))
        random_mse_data.boxplot(ax=axes, rot=90, fontsize=6, grid=False)
        plt.xticks(range(1, random_mse_data.shape[1] + 1), range(1, random_mse_data.shape[1] + 1))
        plt.title('Random')
        plt.xlabel('Snp Number')
        plt.ylabel('MSE')
        plt.tight_layout()
        pdf.savefig()
        plt.close()


def exfeature_by_regression(tree_info_dict, num_boost_round, save_path=None):
    alltree = pd.DataFrame()
    for itree in range(0, num_boost_round):
        feature_id, feature_gain = [], []
        itree_info_dict = tree_info_dict['tree_' + str(itree)]
        for feature_name in itree_info_dict:
            feature_id.append(feature_name)
            feature_gain.append(sum(itree_info_dict[feature_name]))
        itree_df = pd.DataFrame({'featureid': feature_id, 'tree_' + str(itree): feature_gain})
        try:
            alltree = alltree.merge(itree_df, how='outer', on='featureid')
        except KeyError:
            alltree = itree_df

    alltree.fillna(0, inplace=True)
    feature_gain_sum = alltree.sum(axis=1)
    alltree.insert(1, 'featureGain_sum', feature_gain_sum)
    alltree = alltree.sort_values(by='featureGain_sum', axis=0, ascending=False)
    alltree.to_csv(save_path, index=None)


def summery_cv_feature(model_prefix, cv_times, cvfold, n_estimators):

    for cv_time in range(cv_times):
        model_cv_path = model_prefix + 'cv' + str(cv_time) + '_'
        for cv_foldi in range(cvfold):
            model_path = model_cv_path + 'fold' + str(cv_foldi) + '.lgb_model'
            if (cv_time == 0) and (cv_foldi == 0):
                tree_info_dict = extree_info(model_path, n_estimators)
            else:
                tree_info_dict = extree_info(model_path, n_estimators, tree_info_dict)

    feature_save = model_prefix + 'cv' + str(cv_times)
    exfeature_by_regression(tree_info_dict, n_estimators, feature_save + '.feature')





