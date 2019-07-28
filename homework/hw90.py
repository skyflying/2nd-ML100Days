import os
import keras
os.environ["CUDA_VISIBLE_DEVICES"] = "" # �ϥ� CPU

import cv2 # ���J cv2 �M��
import matplotlib.pyplot as plt

train, test = keras.datasets.cifar10.load_data()





image = train[0][0] # Ū���Ϥ�
plt.imshow(image)
plt.show()



gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(gray)
plt.show()





# �ե� cv2.calcHist ��ơA�^�ǭȴN�O histogram
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()


print("hist shape:", hist.shape, "\n����Ϥ��e��ӭ�:", hist[:2]) # 1 ��ܸӦǫ׹Ϥ��A�u�� 1 �� pixel ���ȬO 0�A0 �� pixel ���ȬO 1




chans = cv2.split(image) # ��Ϲ��� 3 �� channel �����X��
colors = ("r", "g", "b")
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

# ���Ҧ� channel
for (chan, color) in zip(chans, colors):
    # �p��� channel �������
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
 
    # �e�X�� channel �������
    plt.plot(hist, color = color)
    plt.xlim([0, 256])
plt.show()





chans = cv2.split(image) # ��Ϲ��� 3 �� channel �����X��
colors = ("r", "g", "b")
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

# ���Ҧ� channel
for (chan, color) in zip(chans, colors):
    # �p��� channel �������
    hist = cv2.calcHist([chan], [0], None, [16], [0, 256])
    print("�C��",color, "�b [16, 32] �o�� bin �����G", hist[1], "��")
 
    # �e�X�� channel �������
    plt.plot(hist, color = color)
    plt.xlim([0, 16])
plt.show()