 Problem Statement:
 In the realm of computer vision and artificial intelligence, the Remote Sensing Image Captioning Dataset (RSICD) provides a unique challenge for participants to develop state-of-the-art models capable of generating accurate and descriptive captions for remote sensing images.The dataset consists of over 10,000 diverse remotely sensed images from satellites. This dataset presents a variety of scenarios and resolutions, all standardized to 224x224 pixels.. Each image comes with five sentences of descriptive captions, making it a robust foundation for training and evaluating image captioning models.
 
 
 Objective:
 Develop an image captioning model that can analyze and comprehend the content of remote sensing images, generating coherent and contextually relevant textual descriptions. The focus is on leveraging the RSICD dataset, with special attention to handling the unique characteristics of remote sensing images

 References:
 1. https://developers.arcgis.com/python/guide/how-image-captioning-works/
 2. https://github.com/openai/CLIP
 3. https://arxiv.org/abs/2103.00020
 4. Xiaoqiang Lu, Binqiang Wang, Xiangtao Zheng, Xuelong Li: “Exploring Models and Data for Remote Sensing Image Caption Generation”, 2017; arXiv:1712.07835. DOI: 10.1109/TGRS.2017.2776321.


Output:

Train Data:

 
![Screenshot (354)](https://github.com/Its-Shreya/TRINIT_AccessDenied_ML/assets/139217213/6496d8b1-1513-4d57-8221-5c0b1ab3508f)

![Screenshot (355) val](https://github.com/Its-Shreya/TRINIT_AccessDenied_ML/assets/139217213/ed80ad76-3d49-449d-b070-f0ac476889ac)

![Screenshot (355)](https://github.com/Its-Shreya/TRINIT_AccessDenied_ML/assets/139217213/d077125e-5264-49f4-916f-9082986fcff7)

Test Data Prediction output:

![Screenshot (356)](https://github.com/Its-Shreya/TRINIT_AccessDenied_ML/assets/139217213/0f4a1870-c567-4cf1-804a-0a4cd7af97fa)


Evaluation:
Bilingual Evaluation Understudy Score(BLEU’s): is a popular metric that measures the number of sequential words that match between the predicted and the ground truth caption. It compares n-grams of various lengths from 1 through 4 to do this. A perfect match results in a score of 1.0, whereas a perfect mismatch results in a score of 0.0. summarizes how close the generated text is to the expected text.



