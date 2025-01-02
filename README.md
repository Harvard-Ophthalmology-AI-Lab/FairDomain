# FairDomain
> [**ECCV 2024**] [**FairDomain: Achieving Fairness in Cross-Domain Medical Image Segmentation and Classification**](https://arxiv.org/pdf/2407.08813)
>
> by [Yu Tian*](https://yutianyt.com/), [Congcong Wen*](https://wencc.xyz/), [Min Shi*](https://shiminxst.github.io/index.html), Muhammad Muneeb Afzal, Hao Huang, Muhammad Osama Khan, [Yan Luo](https://luoyan407.github.io/), [Yi Fang](https://engineering.nyu.edu/faculty/yi-fang), and [Mengyu Wang](https://ophai.hms.harvard.edu/team/dr-wang/).
>
![intro](https://github.com/Harvard-Ophthalmology-AI-Lab/FairDomain/assets/19222962/e6d5afe0-8262-473a-83e3-381b3f51cbbd)




## FairDomain Dataset
* Harvard FairDomain (Harvard-FairDomain): This Harvard-FairDomain dataset includes data for both segmentation and classification tasks for studying fairness in domain shift. For the segmentation task, 10,000 samples from 10,000 patients are included. For the classification task, 10,000 samples from 10,000 patients are included. The samples from Harvard-FairDomain dataset are derived from [**Harvard-FairSeg**](https://github.com/Harvard-Ophthalmology-AI-Lab/FairSeg) and [**Fair-FairVLMed**](https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP) with an added imaging modality of en-face fundus image in addition to the imaging modality of scanning laser ophthalmoscopy (SLO) fundus image originally in the two datasets. The data use license is [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). If you have any questions, please email <harvardophai@gmail.com> and <harvardairobotics@gmail.com>.
  
* The dataset can be accessed via this [link](https://drive.google.com/drive/folders/1huH93JVeXMj9rK6p1OZRub868vv0UK0O?usp=drive_link).

* If you cannot directly download the FairDomain dataset, please request access using the above Google Drive link, we will make sure to grant you access within 3-5 days.

* Note that, the modifier word “Harvard” only indicates that our dataset is from the Department of Ophthalmology of Harvard Medical School and does not imply an endorsement, sponsorship, or assumption of responsibility by either Harvard University or Harvard Medical School as a legal identity. 


## Please refer to each folder (Dataset and Code) for classification and segmentation tasks, respectively.

## Acknowledgement and Citation


If you find this repository useful for your research, please consider citing our [paper](https://arxiv.org/pdf/2407.08813):

```bibtex
@article{tian2024fairdomain,
  title={FairDomain: Achieving Fairness in Cross-Domain Medical Image Segmentation and Classification},
  author={Tian, Yu and Wen, Congcong and Shi, Min and Afzal, Muhammad Muneeb and Huang, Hao and Khan, Muhammad Osama and Luo, Yan and Fang, Yi and Wang, Mengyu},
  journal={arXiv preprint arXiv:2407.08813},
  year={2024}
}

```
