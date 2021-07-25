#  -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pandas import read_csv
import Female_1428_function as f


dir_path = '/data/xyt/data_female_1428/'
phe_list = ['DTT', 'PH', 'EW']
fnumber_list = [12, 24, 48, 96, 192, 384, 1000, 2000, 3000, 4000]
cv_times = 30
random_times = 10
cvfold = 5
pool_max = 30
n_estimators = 160


geno_path = dir_path + 'Genotype.female_1428_012.txt'
geno_data = read_csv(geno_path, header=0, index_col=0)
all_snpid = geno_data.columns.values.copy()

for phe_namei in phe_list:

    phe_path = dir_path + 'Phenotype.female_1428_values.cvid.csv'
    phe_data = read_csv(phe_path, header=0, index_col=0)

    # feature_path = dir_path + 'female_1428_' + phe_namei + '_cv' + str(cv_times) + '.feature'
    # feature_data = read_csv(feature_path, header=0, index_col=0)
    # snpid = feature_data.index.values.copy()

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

    # for fnumber in fnumber_list:

        # # Select SNP by IG
        # ig_snpid = snpid[: fnumber]
        # ig_geno = geno_data.loc[:, ig_snpid]
        #
        # ig_model = dir_path + 'female_1428_' + phe_namei + '_' + str(fnumber) + 'SNP_ig'
        # ig_pearson_list = f.cv(ig_geno, phe_data, params_dict, ig_model)
        # ig_pearson_mean = np.mean(ig_pearson_list)
        # ig_pearson_std = np.std(ig_pearson_list)
        #
        # print(phe_namei, 'igSNP', fnumber, ig_pearson_mean, ig_pearson_std)
        #
        # f.summery_cv_feature(ig_model, cv_times, cvfold, n_estimators)
        #
        # # Select SNP by uniform
        # all_snpid = geno_data.columns.values.copy()
        # uniform_index = np.linspace(0, len(all_snpid), fnumber, endpoint=False)
        # uniform_index = [int(x) for x in uniform_index]
        # uniform_snpid = all_snpid[uniform_index]
        # uniform_geno = geno_data.loc[:, uniform_snpid]
        #
        # uniform_model = dir_path + 'female_1428_' + phe_namei + '_' + str(fnumber) + 'SNP_uniform'
        # uniform_pearson_list = f.cv(uniform_geno, phe_data, params_dict, uniform_model)
        # uniform_pearson_mean = np.mean(uniform_pearson_list)
        # uniform_pearson_std = np.std(uniform_pearson_list)
        #
        # print(phe_namei, 'uniformSNP', fnumber, uniform_pearson_mean, uniform_pearson_std)
        #
        # f.summery_cv_feature(uniform_model, cv_times, cvfold, n_estimators)

    # Select SNP by random

    for r in range(random_times):
        random_index = np.random.permutation(32559)
        random_4000snpid = all_snpid[random_index[: 4000]]
        random_4000snpid = pd.DataFrame({'snpid': random_4000snpid})
        random_save = dir_path + 'female_1428_' + phe_namei + '_random' + str(r)
        random_4000snpid.to_csv(random_save + '.4000snpid', header=True, index=False)

        for fnumber in fnumber_list:
            random_index_fnum = random_index[: fnumber]
            random_snpid = all_snpid[random_index_fnum]

            print('random_times: %d\tfnumber: %d\trandom_snpid: %s' % (r, fnumber, ' '.join(random_snpid)))

            random_geno = geno_data.loc[:, random_snpid]

            random_pearson_list = f.cv(random_geno, phe_data, params_dict)
            random_pearson_mean = np.mean(random_pearson_list)
            random_pearson_std = np.std(random_pearson_list)

            print('randomSNP', phe_namei, r, fnumber, random_pearson_mean, random_pearson_std)





