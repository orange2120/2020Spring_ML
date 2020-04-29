import sys, os
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import slic

"""## Lime

Lime 的部分因為有現成的套件可以使用，因此下方直接 demo 如何使用該套件
其實非常的簡單，只需要 implement 兩個 function 即可
"""

# model_path = './data/model/model_20200410-02-35-27.pkl'
model_path = './data/model/model_20200402-22-21-42.pkl'

model = torch.load(model_path)
train_set = torch.load('./data/train_set.npy')

def predict(input):
    # input: numpy array, (batches, height, width, channels)

    model.eval()
    input = torch.FloatTensor(input).permute(0, 3, 1, 2)
    # 需要先將 input 轉成 pytorch tensor，且符合 pytorch 習慣的 dimension 定義
    # 也就是 (batches, channels, height, width)

    output = model(input.cuda())
    return output.detach().cpu().numpy()

def segmentation(input):
    # 利用 skimage 提供的 segmentation 將圖片分成 100 塊
    return slic(input, n_segments=100, compactness=1, sigma=1)

img_indices = [83, 4218, 4707, 8598]
images, labels = train_set.getbatch(img_indices)
fig, axs = plt.subplots(1, 4, figsize=(15, 8))
np.random.seed(16)
# 讓實驗 reproducible
for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):
    x = image.astype(np.double)
    # lime 這個套件要吃 numpy array

    explainer = lime_image.LimeImageExplainer()
    explaination = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation, top_labels=11)
    # 基本上只要提供給 lime explainer 兩個關鍵的 function，事情就結束了
    # classifier_fn 定義圖片如何經過 model 得到 prediction
    # segmentation_fn 定義如何把圖片做 segmentation

    lime_img, mask = explaination.get_image_and_mask(
                                label=label.item(),
                                positive_only=False,
                                hide_rest=False,
                                num_features=11,
                                min_weight=0.05
                            )
    # 把 explainer 解釋的結果轉成圖片

    axs[idx].imshow(lime_img)

plt.title('lime')
plt.savefig('./output/lime.png')
# 從以下前三章圖可以看到，model 有認出食物的位置，並以該位置為主要的判斷依據
# 唯一例外是第四張圖，看起來 model 似乎比較喜歡直接去認「碗」的形狀，來判斷該圖中屬於 soup 這個 class
