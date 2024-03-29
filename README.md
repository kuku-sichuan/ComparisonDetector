# NOTICE!!!!
you can get code and dataset in new [address](https://github.com/CVIU-CSU/ComparisonDetector)
# NEW VERSION WILL UPDATE!

## Comparison-Based Convolutional Neural Networks for Cervical Cell/Clumps Detection in the Limited Data Scenario

### abstract
Automated detection of cervical cancer cells or cell clumps has the potential to significantly
 reduce error rate and increase productivity in cervical cancer screening. However, most traditional 
 methods rely on the success of accurate cell segmentation and discriminative hand-crafted features 
 extraction. Recently there are emerging deep learning-based methods which train convolutional neural 
 networks to classify image patches, but they are computationally expensive. In this paper we 
 propose to an end-to-end object detection methods for cervical cancer detection. More importantly, we develop the Comparison detector based on Faster-RCNN with Feature Pyramid Network(baseline model) to deal with 
 the limited-data problem. Actually, the key idea is that classify the region proposals by comparising with the prototype representations of each category which learn from reference images. In addition, we propose to learn the prototype representations of the background
 from the data instead of reference images manually choosing by some heuristic rules. Comparison detector shows significant improvement for small dataset, achieving a mean Average Precision (mAP) __26.3%__ and an Average Recall (AR) __35.7%__,
 both improving about __20__ points compared to baseline model. Moreover, Comparison detector achieves better performance on mAP compared with baseline model when training on the medium dataset, and improves AR by __4.6__ points. Our method is promising for the development of automation-assisted cervical cancer screening systems.

### Environment
* CUDA==9.1
* cuDNN==7.0
* tensorflow==1.8.0

### Downloading Data and Weight
If you want to check the effect, you can download the test set in [here](https://pan.baidu.com/s/1BYU3DsX8J8AiaKbE43Iqgw) and put it under the `tfdata/tct`. As same time, you must download the [weight](https://pan.baidu.com/s/1fC3fsKzwfGxq7BxvMjzC1Q) of model and unzip in the home directory.

### Evaluation and Prediction

We provide `evaluate_network.ipynb` to verify our results. We also provide `predict.ipynb` to predict results of a single picture.

### Dataset
The dataset consists of 7410 cervical microscopical images which are cropped from the whole slide images (WSIs) obtained by Pannoramic MIDI II digital slide scanner. In the dataset, there are totally 48,587 instances belonging to 11 categories. We randomly divide the dataset into training set D<sub>f</sub> which contains 6666 images and test set which contains 744 images. The small training set D<sub>s</sub> contains 762 images randomly chosen from D<sub>f</sub>.

__Original image cropped from WSI__
<p align="center">
  <img width="450" src="https://github.com/kuku-sichuan/ComparisonDetector/blob/master/images/README/orig.jpg" />
</p>

__Some instances in 11 categories__
<p align="center">
  <img width="500" src="https://github.com/kuku-sichuan/ComparisonDetector/blob/master/images/README/categories.png" />
</p>

The dataset is available on Google driver [here](https://drive.google.com/drive/folders/1YzPkv6rHLNQXA6QmEUoCl9mWV9fQFsik).
