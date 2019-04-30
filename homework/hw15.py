# ���J�ݭn���M��
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # �t�@��ø��-�˦��M��
plt.style.use('ggplot')

# ����ĵ�i�T��
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')



dir_data = 'E:\\python\\ML-day100\\data\\'
# Ū���ɮ�
f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
app_train.head()



# ���X EXT_SOURCE ���X���ܼƨ�����������
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
ext_data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
ext_data_corrs = ext_data.corr()
ext_data_corrs



plt.figure(figsize = (8, 6))
# ø�s�����Y�� (correlations) �� Heatmap
sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');



# �i�@�B�ڭ��ˬd�o�T���ܼƦb Target �W�������O�_���P
plt.figure(figsize = (24, 8))

# �̤��P EXT_SOURCE �v��ø�s KDE �ϧ�
for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
    # �� subplot
    plt.subplot(1, 3, i + 1)
    
    # KDE �ϧ�
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, source], label = 'target == 0')
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, source], label = 'target == 1')
    
    # �[�W�U���ϧμ���
    plt.title('Distribution of %s by Target Value' % source)
    plt.xlabel('%s' % source); plt.ylabel('Density');
    
plt.tight_layout(h_pad = 2.5)




# �ন�H�~�O�֫�A�N�H��O�֥ᱼ
plot_data = ext_data.copy()
plot_data['YEARS_BIRTH'] = plot_data['DAYS_BIRTH'] / 365
plot_data.drop(['DAYS_BIRTH'],axis = 1, inplace=True)

# �w�q��� : �p���� column �����������Y��
def corr_func(x, y, **kwargs):
    r = np.corrcoef(x, y)[0][1]
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.2, .8), xycoords=ax.transAxes,
                size = 20)
				
				
				
				
				
N_sample = 100000
# �� NaN �ƭȧR�h, �í����ƤW���� 100000 : �]���n�e�I��, �p�G�I�Ӧh�A�|�e�ܤ[!
plot_data = plot_data.dropna().sample(n = N_sample)
# �إ� pairgrid ����
grid = sns.PairGrid(data = plot_data, size = 3, diag_sharey=False,
                    hue = 'TARGET', 
                    vars = [x for x in list(plot_data.columns) if x != 'TARGET'])

# �W�b���� scatter
grid.map_upper(plt.scatter, alpha = 0.2)

# �﨤�u�e histogram
grid.map_diag(sns.kdeplot)

# �U�b���� density plot
grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r)

plt.suptitle('Ext Source and Age Features Pairs Plot', size = 32, y = 1.05)
plt.show()





# �յۥH���P size �e�ݬݡA�[����̤��Ӥ@��
N_sample = 1000

plot_data = plot_data.dropna().sample(n = N_sample)
# �إ� pairgrid ����
grid = sns.PairGrid(data = plot_data, size = 3, diag_sharey=False,
                    hue = 'TARGET', 
                    vars = [x for x in list(plot_data.columns) if x != 'TARGET'])

# �W�b���� scatter
grid.map_upper(plt.scatter, alpha = 0.2)

# �﨤�u�e histogram
grid.map_diag(sns.kdeplot)

# �U�b���� density plot
grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r)

plt.suptitle('Ext Source and Age Features Pairs Plot', size = 32, y = 1.05)
plt.show()




#�@�~
matrix = np.random.uniform(-1,1,(10,10))
plt.figure(figsize=(10,10))
heatmap = sns.heatmap(matrix, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.show()




nrow = 1000
ncol = 3

matrix = np.random.uniform(0, 1, (nrow, ncol))

# �H������ 0, 1, 2 �T�ؼ���
indice = np.random.choice([0,1,2], size=nrow)
plot_data = pd.DataFrame(matrix, indice)

# ø�s seborn �i�� Heatmap
grid = sns.PairGrid(data = plot_data, size = 3, diag_sharey=False)

grid.map_upper(plt.scatter , alpha = 0.2)
grid.map_diag(sns.distplot)
grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r)

plt.show()