import numpy as np
import pandas as pd
# from plotnine import *
import gc
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from plotnine import (element_blank, scale_color_manual, scale_x_continuous, ggplot, aes, geom_line,
                      geom_bar, geom_point, theme, element_text, labs, ggtitle, scale_y_continuous, coord_flip, ggsave)

# load numeric data


def load_data(filepath: str, chunksize: int, data_size_to_load: int, usecols: list) -> pd.DataFrame:

    number = 0
    data = pd.DataFrame()

    with pd.read_csv(filepath, chunksize=chunksize, usecols=usecols) as reader:

        for chunk in reader:

            data = pd.concat([data, chunk])

            number += chunk.shape[0]
            del chunk
            gc.collect()
            if number >= data_size_to_load:
                break

    return data

# build a numeric dataframe for heatmap


def data_for_hmplot(data: pd.DataFrame, row: int) -> pd.DataFrame:

    hm_data = data.replace(np.nan, -50000)
    hm_data['Id'] = range(data.shape[0])
    hm_data = hm_data.iloc[0:(row + 1), :].melt(id_vars=['Id', 'Response'])
    hm_data['value'] = np.where(hm_data.value != -50000, 50000, hm_data.value)
    hm_data['value'] = hm_data['value'].astype('category')
    hm_data['line'] = hm_data['variable'].apply(lambda x: x[0:2])
    hm_data['line'] = hm_data['line'].astype('category')

    return hm_data

# draw a heatmap


def heatmap(data: pd.DataFrame, num_of_features: dict, y_lab: str, title='Missing/ Non-missing of numeric data location distribution base on good products') -> ggplot:
    '''
    num_of_features = {'L0': "L0: 168 features",
                       'L1': "L1: 513 features",
                       'L2': "L2: 42 features",
                       'L3': "L3: 245 features"}

    '''

    g = (
        ggplot(data, aes(x='variable', y="Id", fill='value')) +
        geom_tile() +
        facet_wrap('~line', nrow=1, ncol=4, scales='free_x', labeller=num_of_features) +
        theme(axis_ticks=element_blank(),
              axis_text_x=element_blank(),
              axis_text_y=element_text(size=13),
              axis_title=element_text(size=14),
              legend_text=element_text(size=14),
              plot_title=element_text(size=18),
              strip_text=element_text(),
              panel_grid_major=element_blank(),
              panel_grid_minor=element_blank()) +
        scale_fill_hue(h=0.8, l=0.5, s=0.6, name=" ", labels=["Missing", "Non-missing"]) +
        ggtitle(title) +
        labs(x='Features', y=y_lab)
    )
    return g

# draw_feature_distribution


def draw_feature_distribution(data: pd.DataFrame, line_information: dict, data_format: str) -> ggplot:
    '''
    example:
    line_infomation = {'L0': "L0: 24 sta. & 184 f.",
                       'L1': "L1: 2 sta. & 621 f.",
                       'L2': "L2: 3 sta. & 78 f.",
                       'L3': "L3: 23 sta. & 273 f."}
    '''
    g = (
        ggplot(data.sort_values('station'), aes(x='station'))
        + geom_bar(aes(fill='line'))
        + theme(axis_text_x=element_text(angle=90, size=6),
                axis_text_y=element_text(size=13),
                axis_title=element_text(size=14),
                legend_text=element_text(size=14),
                legend_title=element_text(size=14),
                plot_title=element_text(size=18))
        + scale_y_continuous(breaks=range(0, 1000, 25))
        + facet_wrap('~line', scales='free_x', nrow=1,
                     ncol=4, labeller=line_information)
        + guides(fill=guide_legend(title="Production line"))
        + labs(x='Station', y='Features numbers')
        + ggtitle('Features distribution {} based on lines and stations'.format(data_format))
        + scale_fill_brewer(type="qual", palette="Set1")
    )
    return g


def get_line_information(feat: list) -> pd.DataFrame:
    '''
    transform and order features into and by lines, stations, features as a dataframe
    '''

    features = pd.DataFrame({'feature': feat})
    features['line'] = features['feature'].apply(lambda x: x[0:2])
    features['station'] = features.apply(
        lambda x: x.feature[3: x.feature.find('_', 3, 8)], axis=1)
    features['station'] = features.apply(
        lambda x: 'S' + ('00' + x.station.lstrip('S'))[-2:], axis=1)
    features['feat'] = features.apply(
        lambda x: x.feature[(x.feature.find('_', 3, 10) + 2): 12], axis=1)
    features['feat'] = features.apply(
        lambda x: 'F' + ('00000' + x.feat)[-4:], axis=1)
    return features


def features_by_line(filepath: str, line: str, drop_var=['Id']) -> pd.Series:
    '''
    get features by line or all features
    '''

    feats = pd.read_csv(filepath, nrows=1).columns.to_list()
    feats = list(set(feats) - set(drop_var))
    feat = []
    feat_all = []

    for col in feats:
        if col.split('_')[0] == line:
            feat.append(col)
        else:
            feat_all.append(col)

    if line in ['L0', 'L1', 'L2', 'L3']:
        features = get_line_information(feat)
        return features.sort_values(['station', 'feat'])['feature']
    else:
        features = get_line_information(feat_all)
        return features.sort_values(['station', 'feat'])['feature']


def load_agg_date_data(filepath: str, chunksize: int, data_size_to_load: int, load_agg_data=True, load_data=False) -> pd.DataFrame:
    '''
    1. load date dat
    2. cauclate min, max, mean, middle, std, etc. based on each line and all lines
    '''

    number = 0
    data = pd.DataFrame()
    agg_data = pd.DataFrame()
    feats = pd.read_csv(filepath, nrows=1).columns.to_list()
    feats = list(set(feats) - set('Id'))
    lines = ['All', 'L0', 'L1', 'L2', 'L3']

    with pd.read_csv(filepath, chunksize=chunksize) as reader:
        for chunk in reader:

            temp = pd.DataFrame()
            if load_agg_data:
                temp = chunk.Id.to_frame()
                for line in lines:
                    # selelct features by lines or overall lines
                    chunk_temp = chunk[features_by_line(filepath, line)]
                    temp[line + '_' +
                         'min_date'] = np.nanmin(chunk_temp.to_numpy(), axis=1)
                    temp[line + '_' +
                         'max_date'] = np.nanmax(chunk_temp.to_numpy(), axis=1)
                    temp[line + '_' + 'avg_date'] = np.nanmean(
                        chunk_temp.to_numpy(), axis=1).round(3)
                    temp[line + '_' +
                         'mid_date'] = np.nanmedian(chunk_temp.to_numpy(), axis=1)
                    temp[line + '_' + 'std_date'] = np.nanstd(
                        chunk_temp.to_numpy(), axis=1).round(3)
                    temp[line + '_' + 'na_count'] = np.count_nonzero(
                        np.isnan(chunk_temp.to_numpy()), axis=1)
                    temp[line + '_' +
                         'unique_val'] = chunk.nunique(axis=1, dropna=True)
                    temp[line + '_' + 'duration'] = temp[line + '_' +
                                                         'max_date'] - temp[line + '_' + 'min_date']

                    del chunk_temp
                    gc.collect()

            if load_agg_data:
                temp['All_kurtosis'] = chunk.kurtosis(axis=1)
                temp['All_skewness'] = chunk.skew(axis=1)
                agg_data = pd.concat([agg_data, temp])

            if load_data:
                data = pd.concat([data, chunk])

            number += chunk.shape[0]
            del chunk, temp
            gc.collect()
            if number >= data_size_to_load:
                break

    return agg_data, data

# To fill 0 value for that some features have no corresponding label of 0 or 1


def modified_groupby_data(temp_cate):
    temp = pd.DataFrame({'Response': [np.nan], 'counts': [0]})
    if temp_cate.empty:
        temp_cate = pd.DataFrame({'Response': [0, 1], 'counts': [0, 0]})
    elif (temp_cate['Response'].sum())/temp_cate.shape[0] == 0:
        temp['Response'] = 1
        temp_cate = pd.concat([temp_cate, temp], ignore_index=True)
    elif (temp_cate['Response'].sum())/temp_cate.shape[0] == 1:
        temp['Response'] = 0
        temp_cate = pd.concat(
            [temp_cate, temp], ignore_index=True).sort_values(['Response'])

    return temp_cate

# calculate the number of factors occurred by each observation (row)


def cate_factor_count(filepath: str, chunksize: int, data_size_to_load: int) -> pd.DataFrame:

    number = 0
    data = pd.DataFrame()

    with pd.read_csv(filepath, chunksize=chunksize) as reader:

        for chunk in reader:

            id = pd.DataFrame()
            temp_data = pd.DataFrame()

            id = pd.concat([id, pd.DataFrame({'Id': chunk['Id']})])
            temp_data = pd.concat([id, chunk.iloc[:, 1:].apply(
                lambda x: x.value_counts(), axis=1)], axis=1)
            data = pd.concat([data, temp_data])

            number += chunk.shape[0]
            del chunk, id, temp_data
            gc.collect()
            if number >= data_size_to_load:
                break

    return data

# select top kth factors occurred in each row and output a dataframe for barplot


def rowwise_feature_eng(data: pd.DataFrame, select_top: int) -> pd.DataFrame:
    cate_train_factors = pd.DataFrame({'count': data.iloc[:, 2:len(
        data.columns) - 1].notnull().apply(lambda x: sum(x), axis=0)})
    cate_train_factors['Index'] = cate_train_factors.index

    top_factor = cate_train_factors.sort_values(
        'count', ascending=False).iloc[:select_top, :]
    others_factor = pd.DataFrame({'count': cate_train_factors.sort_values(
        'count', ascending=False).iloc[select_top:, 0].sum(), 'Index': pd.Series(['others'], index=['others'])})
    new_factor = pd.concat([top_factor, others_factor])
    new_factor = new_factor.sort_values('count', ascending=False)
    new_factor['Index'] = pd.Categorical(
        new_factor.Index, categories=pd.unique(new_factor.Index))

    top_feats = cate_train_factors.sort_values(
        'count', ascending=False).iloc[:select_top, 1]
    cate_train_data = data[top_feats]

    other_feats = cate_train_factors.sort_values(
        'count', ascending=False).iloc[select_top:, 1]
    cate_train_data['others'] = np.nansum(data[other_feats].to_numpy(), axis=1)
    cate_train_data.replace(np.nan, 0, inplace=True)
    cate_train_data = pd.concat(
        [data['Id'], cate_train_data, data['unique']], axis=1)

    return cate_train_data, new_factor


# return the best threshold and mcc

def best_thr_mcc(preds, dtrain):
    labels = dtrain.get_label()
    thresholds = np.linspace(0.30, 0.55, 7)
    mcc = np.array([matthews_corrcoef(labels, preds > thr)
                   for thr in thresholds])
    best_thr = thresholds[np.argmax(mcc)]
    best_score = mcc.max()
    return best_thr, best_score

# customed eval_metric


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    thresholds = np.linspace(0.30, 0.55, 7)
    mcc = np.array([matthews_corrcoef(labels, preds > thr)
                   for thr in thresholds])
    best_score = mcc.max()
    return 'error', -best_score


def submit_and_layer_merge(csv_name, te_stacking, tr_stacking, te, scores, tr_layer, te_layer, output):
    te_stacking = pd.concat(
        [te['Id'], pd.DataFrame({'Response': te_stacking})], axis=1)
    tr_stacking = tr_stacking.sort_values('index')
    submission = pd.DataFrame(
        {'Id': te['Id'], 'Response': te_stacking['Response'] > np.mean(scores['threshold'])})
    submission['Response'] = submission['Response'].astype('int')
    submission.to_csv(output + csv_name + '_submission.csv', index=0)

    tr_layer = pd.merge(tr_layer, tr_stacking, how='left', on='index')
    te_layer = pd.merge(te_layer, te_stacking.rename(
        columns={"Response": csv_name}), how='left', on='Id')

    return tr_layer, te_layer


def train_model(var_name: str, n_fold: int, n_tree: int, early_stopping_rounds: int, x, y, te, params=None):

    kf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=123)

    scores = {'fold': [], 'mcc': [], 'g_means': [],
              'auc_scores': [], 'f1_scores': [], 'threshold': []}

    evals_result = {}
    loss_data = pd.DataFrame()
    te_stacking = np.zeros(len(te))
    tr_stacking = pd.DataFrame()

    for fold_i, (tr_idx, va_idx) in enumerate(kf.split(x, y)):

        temp_loss_data = pd.DataFrame()
        temp_tr = pd.DataFrame()

        tr_x, va_x = x.iloc[tr_idx], x.iloc[va_idx]
        tr_y, va_y = y.iloc[tr_idx], y.iloc[va_idx]

        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        dvalid = xgb.DMatrix(va_x, label=va_y)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

        model = xgb.train(params=params,
                          dtrain=dtrain,
                          num_boost_round=n_tree,
                          evals=watchlist,
                          verbose_eval=1,
                          early_stopping_rounds=early_stopping_rounds,
                          evals_result=evals_result,
                          feval=evalerror
                          )

        va_pred = model.predict(dvalid)

        temp_tr = pd.DataFrame({'index': va_idx, var_name: va_pred})
        tr_stacking = pd.concat([tr_stacking, temp_tr],
                                axis=0).reset_index(drop=True)

        prediction = model.predict(xgb.DMatrix(te.drop(['Id'], axis=1)))
        te_stacking += prediction

        if (fold_i + 1) == n_fold:
            te_stacking /= (fold_i + 1)

        best_thresh, mcc_value = best_thr_mcc(va_pred, dvalid)
        y_pred = np.array(
            [1 if y_pro > best_thresh else 0 for y_pro in va_pred])

        tn, fp, fn, tp = confusion_matrix(va_y, y_pred).ravel()
        spec = tn / (tn + fp)
        sens = tp / (tp + fn)
        g_means = np.sqrt(spec * sens)
        fmeasure = f1_score(va_y, y_pred)
        auc_score = roc_auc_score(va_y, y_pred)

        temp_loss_data = pd.DataFrame({'train': list(evals_result['train'].values())[0],
                                       'test':  list(evals_result['eval'].values())[0],
                                       'tree':  [i+1 for i in range(len(list(evals_result['train'].values())[0]))]})

        print('mcc value: {}'.format(round(mcc_value, 3)))
        print('g_means value: {}'.format(round(g_means, 3)))
        print('auc value: {}'.format(round(auc_score, 3)))
        print('f1_score value: {}'.format(round(fmeasure, 3)))
        print('threshold: {}'.format(round(best_thresh, 3)))

        loss_data = pd.concat([loss_data, temp_loss_data],
                              axis=0).reset_index(drop=True)
        scores['fold'].append(fold_i + 1)
        scores['mcc'].append(mcc_value)
        scores['g_means'].append(g_means)
        scores['f1_scores'].append(fmeasure)
        scores['auc_scores'].append(auc_score)
        scores['threshold'].append(best_thresh)

        gc.collect()

    return model, scores, loss_data, tr_stacking, te_stacking


def xgb_model(params):

    var_name = params['var_name']
    n_fold = 5
    n_tree = int(params['n_tree'])
    early_stopping_rounds = int(round(5 / (1/params['eta']), 0))
    x = params['x']
    y = params['y']
    te = params['te']

    xgb_params = {'colsample_bytree': params['colsample_bytree'],
                  'eta': 1/params['eta'],
                  'max_depth': int(params['max_depth']),
                  'subsample': params['subsample'],
                  'min_child_weight': int(params['min_child_weight']),
                  'gamma': params['gamma'] / 10,
                  'ntree': n_tree,
                  'early_stopping_rounds': early_stopping_rounds,
                  'objective': 'binary:logistic',
                  'random_state': 123,
                  'disable_default_eval_metric': 1,
                  'var_name': params['var_name']
                  }

    model, scores, loss_data, tr_stacking, te_stacking = train_model(x=x, y=y, te=te, var_name=var_name, n_fold=n_fold, n_tree=n_tree,
                                                                     early_stopping_rounds=early_stopping_rounds,
                                                                     params=xgb_params)
    return model, scores, loss_data, tr_stacking, te_stacking


def loss(loss_: pd.DataFrame, plot_title: str, start, end, gap):

    loss_[['train', 'test']] = loss_[['train', 'test']] * -1
    loss_ = loss_.groupby('tree')['train', 'test'].mean().reset_index()
    loss_ = pd.melt(loss_, id_vars=[
        'tree'], value_vars=['train', 'test'])
    loss_['variable'] = loss_['variable'].astype('category')

    g = (ggplot(loss_, aes(x='tree', y='value', group='variable', color='variable', linetype='variable'))
         + geom_line(size=1.5)
         + theme(axis_text_x=element_text(size=13),
                 axis_text_y=element_text(size=13),
                 axis_title=element_text(size=14),
                 legend_text=element_text(size=14),
                 legend_title=element_blank())
         + scale_x_continuous(breaks=range(start, end, gap))
         + scale_y_continuous(breaks=np.linspace(0, 1.0, 21, endpoint=True))
         + scale_color_manual(values=["#619ED6", "#6BA547"])
         + labs(x='number of rounds', y='MCC')
         + ggtitle(plot_title)
         )
    return g


# customed eval_metric
def lgb_evalerror(preds, dtrain):
    labels = dtrain.get_label()
    thresholds = np.linspace(0.30, 0.55, 7)
    mcc = np.array([matthews_corrcoef(labels, preds > thr)
                   for thr in thresholds])
    best_score = mcc.max()
    return 'MCC', best_score, True


def lgb_train_model(var_name: str, te: pd.DataFrame, x: pd.DataFrame, y: pd.DataFrame, n_fold: int, early_stopping_rounds: int, params=None):

    kf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=123)

    scores = {'fold': [], 'mcc': [], 'g_means': [],
              'auc_scores': [], 'f1_scores': [], 'threshold': []}

    evals_result = {}
    loss_data = pd.DataFrame()
    te_stacking = np.zeros(len(te))
    tr_stacking = pd.DataFrame()

    for fold_i, (tr_idx, va_idx) in enumerate(kf.split(x, y)):

        temp_loss_data = pd.DataFrame()
        temp_tr = pd.DataFrame()

        tr_x, va_x = x.iloc[tr_idx], x.iloc[va_idx]
        tr_y, va_y = y.iloc[tr_idx], y.iloc[va_idx]

        dtrain = lgb.Dataset(data=tr_x, label=tr_y)
        dvalid = lgb.Dataset(data=va_x, label=va_y, reference=dtrain)
        watchlist = [dtrain, dvalid]

        model = lgb.train(params=params,
                          train_set=dtrain,
                          valid_sets=watchlist,
                          valid_names=['train', 'eval'],
                          verbose_eval=1,
                          early_stopping_rounds=early_stopping_rounds,
                          evals_result=evals_result,
                          feval=lgb_evalerror
                          )

        va_pred = model.predict(va_x)

        temp_tr = pd.DataFrame({'index': va_idx, var_name: va_pred})
        tr_stacking = pd.concat([tr_stacking, temp_tr],
                                axis=0).reset_index(drop=True)

        prediction = model.predict(te.drop(['Id'], axis=1))
        te_stacking += prediction

        if (fold_i + 1) == n_fold:
            te_stacking /= (fold_i + 1)

        best_thresh, mcc_value = best_thr_mcc(va_pred, dvalid)
        y_pred = np.array(
            [1 if y_pro > best_thresh else 0 for y_pro in va_pred])

        tn, fp, fn, tp = confusion_matrix(va_y, y_pred).ravel()
        spec = tn / (tn + fp)
        sens = tp / (tp + fn)
        g_means = np.sqrt(spec * sens)
        fmeasure = f1_score(va_y, y_pred)
        auc_score = roc_auc_score(va_y, y_pred)

        temp_loss_data = pd.DataFrame({'train': list(evals_result['train'].values())[0],
                                       'test': list(evals_result['eval'].values())[0],
                                       'tree': [i+1 for i in range(len(list(evals_result['train'].values())[0]))]})

        print('mcc value: {}'.format(round(mcc_value, 3)))
        print('g_means value: {}'.format(round(g_means, 3)))
        print('auc value: {}'.format(round(auc_score, 3)))
        print('f1_score value: {}'.format(round(fmeasure, 3)))
        print('threshold: {}'.format(round(best_thresh, 3)))

        loss_data = pd.concat([loss_data, temp_loss_data],
                              axis=0).reset_index(drop=True)
        scores['fold'].append(fold_i + 1)
        scores['mcc'].append(mcc_value)
        scores['g_means'].append(g_means)
        scores['f1_scores'].append(fmeasure)
        scores['auc_scores'].append(auc_score)
        scores['threshold'].append(best_thresh)

        gc.collect()

    return model, scores, loss_data, tr_stacking, te_stacking


def lgb_model(params):

    n_fold = 5
    early_stopping_rounds = int(round(5 / (1/params['eta']), 0))
    x = params['x']
    y = params['y']
    te = params['te']
    var_name = params['var_name']

    lgb_params = {'feature_fraction': params['feature_fraction'],
                  'num_iteration': int(params['num_iteration']),
                  'eta': 1/params['eta'],
                  'max_depth': int(params['max_depth']),
                  'bagging_fraction': params['bagging_fraction'],
                  'min_child_weight': int(params['min_child_weight']),
                  'num_leaves': int(params['num_leaves']),
                  'min_data_in_leaf': int(params['min_data_in_leaf']),
                  'objective': 'binary',
                  'random_state': 123,
                  'metric': 'None',
                  'early_stopping_rounds': early_stopping_rounds
                  }

    model, scores, loss_data, tr_stacking, te_stacking = lgb_train_model(var_name=var_name, te=te, x=x, y=y, n_fold=n_fold,
                                                                         early_stopping_rounds=early_stopping_rounds,
                                                                         params=lgb_params)
    return model, scores, loss_data, tr_stacking, te_stacking


def rf_best_thr_mcc(preds, va_y):
    thresholds = np.linspace(0.25, 0.55, 8)
    mcc = np.array([matthews_corrcoef(va_y, preds > thr)
                   for thr in thresholds])
    best_thr = thresholds[np.argmax(mcc)]
    best_score = mcc.max()
    return best_thr, best_score


def rf_train_model(var_name: int, x: pd.DataFrame, y: pd.DataFrame, te: pd.DataFrame, params=None, n_fold=2):

    kf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=123)

    scores = {'fold': [], 'mcc': [], 'g_means': [],
              'auc_scores': [], 'f1_scores': [], 'threshold': []}

    te_stacking = np.zeros(len(te))
    tr_stacking = pd.DataFrame()

    for fold_i, (tr_idx, va_idx) in enumerate(kf.split(x, y)):

        temp_tr = pd.DataFrame()

        tr_x, va_x = x.iloc[tr_idx], x.iloc[va_idx]
        tr_y, va_y = y.iloc[tr_idx], y.iloc[va_idx]

        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                     max_depth=params['max_depth'],
                                     min_samples_leaf=params['min_samples_leaf'],
                                     min_samples_split=params['min_samples_split'],
                                     n_jobs=-1, random_state=123)

        clf.fit(tr_x, tr_y)
        va_pred = clf.predict_proba(va_x)[:, 1]

        temp_tr = pd.DataFrame({'index': va_idx, var_name: va_pred})
        tr_stacking = pd.concat([tr_stacking, temp_tr],
                                axis=0).reset_index(drop=True)

        prediction = clf.predict_proba(te.drop(['Id'], axis=1))[:, 1]
        te_stacking += prediction

        if (fold_i + 1) == n_fold:
            te_stacking /= (fold_i + 1)

        best_thresh, mcc_value = rf_best_thr_mcc(va_pred, va_y)
        y_pred = np.array(
            [1 if y_pro > best_thresh else 0 for y_pro in va_pred])

        tn, fp, fn, tp = confusion_matrix(va_y, y_pred).ravel()
        spec = tn / (tn + fp)
        sens = tp / (tp + fn)
        g_means = np.sqrt(spec * sens)
        fmeasure = f1_score(va_y, y_pred)
        auc_score = roc_auc_score(va_y, y_pred)

        print('fold {}'.format(fold_i + 1))
        print('mcc value: {}'.format(round(mcc_value, 3)))
        print('g_means value: {}'.format(round(g_means, 3)))
        print('auc value: {}'.format(round(auc_score, 3)))
        print('f1_score value: {}'.format(round(fmeasure, 3)))
        print('threshold: {}'.format(round(best_thresh, 3)))
        print('='*20)

        scores['fold'].append(fold_i + 1)
        scores['mcc'].append(mcc_value)
        scores['g_means'].append(g_means)
        scores['f1_scores'].append(fmeasure)
        scores['auc_scores'].append(auc_score)
        scores['threshold'].append(best_thresh)

        gc.collect()

    return clf, scores, tr_stacking, te_stacking


def rf_model(params):

    x = params['x']
    y = params['y']
    te = params['te']
    var_name = params['var_name']

    rf_params = {'n_estimators': int(params['n_estimators']),
                 'max_depth': int(params['max_depth']),
                 'min_samples_leaf': int(params['min_samples_leaf']),
                 'min_samples_split': int(params['min_samples_split'])
                 }

    clf, scores, tr_stacking, te_stacking = rf_train_model(
        params=rf_params, x=x, y=y, te=te, var_name=var_name)
    return clf, scores, tr_stacking, te_stacking


def lt_train_model(var_name: int, x: pd.DataFrame, y: pd.DataFrame, te: pd.DataFrame, params=None, n_fold=5):

    kf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=123)

    scores = {'fold': [], 'mcc': [], 'g_means': [],
              'auc_scores': [], 'f1_scores': [], 'threshold': []}

    te_stacking = np.zeros(len(te))
    tr_stacking = pd.DataFrame()

    for fold_i, (tr_idx, va_idx) in enumerate(kf.split(x, y)):

        temp_tr = pd.DataFrame()

        tr_x, va_x = x.iloc[tr_idx], x.iloc[va_idx]
        tr_y, va_y = y.iloc[tr_idx], y.iloc[va_idx]

        model = LogisticRegression(random_state=123,
                                   solver=params['solver'],
                                   C=params['C'])
        model.fit(tr_x, tr_y)

        va_pred = model.predict_proba(va_x)[:, 1]

        temp_tr = pd.DataFrame({'index': va_idx, var_name: va_pred})
        tr_stacking = pd.concat([tr_stacking, temp_tr],
                                axis=0).reset_index(drop=True)

        prediction = model.predict_proba(te.drop(['Id'], axis=1))[:, 1]
        te_stacking += prediction

        if (fold_i + 1) == n_fold:
            te_stacking /= (fold_i + 1)

        best_thresh, mcc_value = rf_best_thr_mcc(va_pred, va_y)
        y_pred = np.array(
            [1 if y_pro > best_thresh else 0 for y_pro in va_pred])

        tn, fp, fn, tp = confusion_matrix(va_y, y_pred).ravel()
        spec = tn / (tn + fp)
        sens = tp / (tp + fn)
        g_means = np.sqrt(spec * sens)
        fmeasure = f1_score(va_y, y_pred)
        auc_score = roc_auc_score(va_y, y_pred)

        print('fold {}'.format(fold_i + 1))
        print('mcc value: {}'.format(round(mcc_value, 3)))
        print('g_means value: {}'.format(round(g_means, 3)))
        print('auc value: {}'.format(round(auc_score, 3)))
        print('f1_score value: {}'.format(round(fmeasure, 3)))
        print('threshold: {}'.format(round(best_thresh, 3)))
        print('='*20)

        scores['fold'].append(fold_i + 1)
        scores['mcc'].append(mcc_value)
        scores['g_means'].append(g_means)
        scores['f1_scores'].append(fmeasure)
        scores['auc_scores'].append(auc_score)
        scores['threshold'].append(best_thresh)

        gc.collect()

    return model, scores, tr_stacking, te_stacking


def lt_model(params):

    x = params['x']
    y = params['y']
    te = params['te']
    var_name = params['var_name']

    lt_params = {'C': params['C'],
                 'solver': params['solver']
                 }

    model, scores, tr_stacking, te_stacking = lt_train_model(
        params=lt_params, x=x, y=y, te=te, var_name=var_name)
    return model, scores, tr_stacking, te_stacking
