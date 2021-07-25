#  -*- coding: utf-8 -*-
from pandas import read_csv
import numpy as np
import Female_1428_function as f


cv_times = 30
cvfold = 5
pool_max = 30
n_estimators = 160

dir_path = '/data/xyt/data_female_1428/'

# prepare CV data
phe_path = dir_path + 'Phenotype.female_1428_values.txt'
cvid_path = dir_path + 'Phenotype.female_1428_values.cvid.csv'

phe = read_csv(phe_path, header=0, index_col=0)
phe_name_list = phe.columns.values.copy()
f.prepare_cv_data(phe, cvid_path, cv_times, cvfold)

# load data
data_path = dir_path + 'Genotype.female_1428_012.txt'
geno_data = read_csv(data_path, index_col=0, header=0)
phe_data = read_csv(cvid_path, index_col=0, header=0)

for phe_namei in phe_name_list:

    params_dict = {
                'phe_name': phe_namei,
                'learning_rate': 0.05,
                'max_depth': 5,
                'n_estimators': n_estimators,
                'min_data_in_leaf': 20,
                'num_leaves': 10,
                'cv_times': cv_times,
                'cvfold': cvfold,
                'pool_max': pool_max
            }

    model_save = dir_path + 'female_1428_' + phe_namei
    cv_pearson_list = f.cv(geno_data, phe_data, params_dict, model_save)
    cv_pearson_mean = np.mean(cv_pearson_list)
    cv_pearson_std = np.std(cv_pearson_list)

    print(phe_namei, cv_pearson_mean, cv_pearson_std)

    f.summery_cv_feature(model_save, cv_times, cvfold, n_estimators)
