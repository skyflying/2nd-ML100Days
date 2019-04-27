import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline



# ��l�]�w Ages �����
ages = pd.DataFrame({"age": [18,22,25,27,7,21,23,37,30,61,45,41,9,18,80,100]})







# �s�W��� "equal_width_age", ��~�ְ����e����
ages["equal_width_age"] = pd.cut(ages["age"], 4)


# �[��W�����U, �C�ӺزնZ�U�X�{�X��
ages["equal_freq_age"].value_counts() # �C�� bin ����Ƶ��ƬO�@�˪�



# �s�W��� "equal_freq_age", ��~�ְ����W����
ages["equal_freq_age"] = pd.qcut(ages["age"], 4)



# �[��W�����U, �C�ӺزնZ�U�X�{�X��
ages["equal_freq_age"].value_counts() # �C�� bin ����Ƶ��ƬO�@�˪�






ages["customized_age_grp"] = pd.cut(ages["age"],[0,10,20,30,50,100])
ages["customized_age_grp"].value_counts().sort_index()

