# -*- coding: utf-8 -*-
import numpy as np
from pandas import read_csv
import function as f
import time


dir_path = '/data4/xyt/G3_6species/'
species_list = ['rice', 'sorghum', 'soy', 'spruce', 'switchgrass', 'maize']
cv_times = 30
cvfold = 5
pool_max = 80

params_file = dir_path + 'summery.csv'
params_data = read_csv(params_file, header=0, index_col=None)

cv_pearson_mean = []
cv_pearson_std = []

for i, row in enumerate(params_data.values):
    method, species, phe_namei, learning_rate, max_depth, n_estimators, min_data_in_leaf, num_leaves = row[:-1]

    params_dict = {
        'method': method,
        'species': species,
        'phe_name': phe_namei,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'n_estimators': n_estimators,
        'min_data_in_leaf': min_data_in_leaf,
        'num_leaves': num_leaves,
        'cv_times': cv_times,
        'cvfold': cvfold,
        'pool_max': pool_max
    }

    geno_path = dir_path + species + '_geno.csv'
    phe_path = dir_path + species + '_pheno.csv'
    cv_path = dir_path + species + '_CVFs.csv'
    geno = read_csv(geno_path, header=0, index_col=0)
    phe_cvid = read_csv(phe_path, header=0, index_col=0)

    start = time.time()
    # CV
    cv_pearson_listi = f.cv(geno, phe_cvid, params_dict)
    cv_pearson_mean_i = np.mean(cv_pearson_listi)
    cv_pearson_std_i = np.std(cv_pearson_listi)
    cv_pearson_mean.append(cv_pearson_mean_i)
    cv_pearson_std.append(cv_pearson_std_i)

    end = time.time()
    run_time = end - start

    print(species, phe_namei, method)
    print(cv_pearson_listi)
    print(cv_pearson_mean_i, cv_pearson_std_i)
    print(run_time)

params_data['cv'+str(cv_times)+'_mean'] = cv_pearson_mean
params_data['cv'+str(cv_times)+'_std'] = cv_pearson_std
save_path = dir_path + 'summery_cv' + str(cv_times) + '_pearson.csv'
params_data.to_csv(save_path, header=True, index=False)















