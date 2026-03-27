# 🏠 SpatialSeg

> This is the official implementation of the TRO paper [Spatial Balancing for RGB-Thermal Semantic Segmentation in Autonomous Driving: A Study from Analysis to Improvement](https://doi.org/10.1109/TRO.2026.3677009).

[![Demonstration Video](https://img.youtube.com/vi/p3TGbufsDhY/0.jpg)](https://www.youtube.com/watch?v=p3TGbufsDhY)

*Click the image above to watch the demonstration video.*

## 📖 Overview
We propose a **Gaussian-guided regional balancing masking** method to balance segmentation performance across different image regions. Moreover, we introduce a **spatial-weighted loss** to further enhance the overall segmentation performance. Experimental results on MFNet dataset and KP dataset demonstrate the effectiveness of our method in mitigating spatial bias and improving balanced performance.

## 📂 Dataset
* Download the MF dataset from the original [website](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/), or our pre-processed dataset from [here](https://nas.labsun.org/downloads/2026_tro_spatial/MF_Dataset.zip).
* Download our pre-processed KP dataset from [here](https://nas.labsun.org/downloads/2026_tro_spatial/MF_Dataset.zip).

Place them in the 'datasets' folder in the following structure:

```shell
<datasets>
|-- <MFdataset>
    |-- <RGB>
    |-- <Thermal>
    |-- <Label>
    |-- train.txt
    |-- val.txt
    |-- test.txt

|-- <KPdataset>
    |-- <images>
        |-- set00
        |-- set01
        ...
    |-- <labels>
    |-- train.txt
    |-- val.txt
    |-- test.txt
```

## 🚀 Usage
For usage instructions, please refer to [CRM](https://github.com/UkcheolShin/CRM_RGBTSeg?tab=readme-ov-file).

## 📚 Results
We offer the pre-trained weights on two RGB-T semantic segmentation dataset.

### MFNet dataset (9 classes)
| Architecture | Backbone | mIOU | Weight (Google Drive) | Weight (NAS) |
|:---:|:---:|:---:|:---:|:---:|
| Ours | Swin-T | 59.4% | [MF_swin_T](https://drive.google.com/file/d/1Sj8CXz0-YrN3lA5y70QhBuxqgj5XGHv-/view?usp=drive_link) | [MF_swin_T](https://nas.labsun.org/downloads/2026_tro_spatial/weights/MF_swin_T.ckpt) |
| Ours | Swin-S | 62.1% | [MF_swin_S](https://drive.google.com/file/d/1-33xxEAoS1L0eEoJkE063oWXQt_STjbb/view?usp=drive_link) | [MF_swin_S](https://nas.labsun.org/downloads/2026_tro_spatial/weights/MF_swin_S.ckpt) |
| Ours | Swin-B | 64.6% | [MF_swin_B](https://drive.google.com/file/d/11vPxrrIiSqKY9WzAOb1Kup0BwrHQH5-Z/view?usp=drive_link) | [MF_swin_B](https://nas.labsun.org/downloads/2026_tro_spatial/weights/MF_swin_B.ckpt) |

### KP dataset (19 classes)
| Architecture | Backbone | mIOU | Weight (Google Drive) | Weight (NAS) |
|:---:|:---:|:---:|:---:|:---:|
| Ours | Swin-T | 52.3% | [KP_swin_T](https://drive.google.com/file/d/1QARuZUREhx68CMxu1LkGPxpK-eG4qygt/view?usp=drive_link) | [KP_swin_T](https://nas.labsun.org/downloads/2026_tro_spatial/weights/KP_swin_T.ckpt) |
| Ours | Swin-S | 54.9% | [KP_swin_S](https://drive.google.com/file/d/1fM3TtXaQ3OxXklduM2w-8Vf9Hcg3MDHy/view?usp=drive_link) | [KP_swin_S](https://nas.labsun.org/downloads/2026_tro_spatial/weights/KP_swin_S.ckpt) |
| Ours | Swin-B | 56.8% | [KP_swin_B](https://drive.google.com/file/d/16JXuZ-5rcHouISP-gWHfylDz1XR0J4CT/view?usp=drive_link) | [KP_swin_B](https://nas.labsun.org/downloads/2026_tro_spatial/weights/KP_swin_B.ckpt) |

## 🔗 Citation
If you use our work in your research, please cite:

```
    @article{li2026spatialseg,
      title={Spatial Balancing for RGB-Thermal Semantic Segmentation in Autonomous Driving: A Study from Analysis to Improvement},
      author={Li, Haotian and Chu, Henry K and Sun, Yuxiang},
      journal={IEEE Transactions on Robotics},
      year={2026},
      publisher={IEEE}
    }
```

## Acknowledgement
Our network architecture and codebase are built upon [CRM](https://github.com/UkcheolShin/CRM_RGBTSeg?tab=readme-ov-file).

The inspiration and analytical approach of this paper draw from [ZoneEval](https://github.com/Zzh-tju/ZoneEval).
