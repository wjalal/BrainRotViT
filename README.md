# BrainRotViT: Transformer-ResNet Hybrid for Explainable Modeling of Brain Aging from 3D sMRI

## Abstract

Accurate brain-age estimation from structural MRI provides a valuable biomarker
for studying healthy aging and neurodegenerative disease. Conventional CNN-based
approaches are constrained by limited receptive fields and susceptibility to
overfitting on heterogeneous multi-site data, whereas pure transformer
architectures are computationally demanding and require large-scale training
data. We propose the Brain ResNet over trained Vision Transformer (BrainRotViT),
a two-stage, slice-based (2.5D) framework that approximates volumetric context.
A Vision Transformer is first pretrained on an auxiliary age-sex composite
classification task and subsequently frozen; its per-slice embeddings are
stacked into a 160 x 768 feature matrix and processed as a pseudo-image by a
lightweight 2D residual CNN regressor, with subject sex incorporated at the
final prediction layer. Trained and validated on 11 multi-site datasets (6,056
unique subjects from more than 130 imaging sites), the model achieves a mean
absolute error (MAE) of 3.43 years and generalizes to four fully held-out
cohorts (MAE 4.79 to 5.35 years), outperforming retrained 3D-ResNet, SFCN,
3D-ViT, Global-Local Transformer, and TSAN baselines under identical
preprocessing. Following age-bias correction and covariate adjustment, the
brain-age-gap (BAG) is significantly associated with Alzheimer's disease, mild
cognitive impairment, and autism spectrum disorder. An integrated
interpretability pipeline combining ViT attention maps with guided
backpropagation, validated through saliency sanity checks and anatomical
localization using the AAL3 atlas, identifies aging-related regions consistent
with existing literature.

## Method Overview

The framework decouples representation learning from volumetric aggregation
through two stages (see Figure 1 of the paper).

1. **Stage 1: Vision Transformer representation learning.** A ViT is trained on
   an auxiliary age-sex composite classification task using Weight-Decomposed
   Low-Rank Adaptation (DoRA) for parameter-efficient fine-tuning. The
   classification head is then discarded and the encoder is frozen.
2. **Stage 2: Residual CNN regression.** The frozen encoder produces a
   768-dimensional embedding for each of the 160 sagittal slices. These
   embeddings are stacked in spatial order into a 160 x 768 feature matrix, which
   is treated as a single-channel pseudo-image and regressed to a scalar brain
   age by a lightweight residual CNN, with biological sex fused at the final
   layer.
3. **Interpretability.** ViT patch attention and CNN guided backpropagation are
   fused per subject into slice-level saliency maps, aggregated across subjects
   into a 3D attention volume, and mapped onto AAL3 regions. Saliency
   credibility is established through model-parameter randomization sanity
   checks and split-half reliability analysis.

## Datasets

The study uses fifteen publicly available structural MRI datasets with a
subject-level split. Eleven cohorts are pooled for training and validation
(ADNI, IXI, ABIDE-II, DLBS, COBRE, FCON1000, CORR, OASIS-1, Cam-CAN, NIMH, and
BOLD variability), and four fully held-out cohorts (SALD, SUDMEX-CONN, AgeRisk,
and TrueCrime) are reserved for zero-shot cross-cohort evaluation. Refer to
Table 1 of the paper for per-dataset demographics. The datasets are publicly
available from their respective providers and are not redistributed in this
repository.

## Repository Organization

The repository is organized into a main pipeline at the root and several
supporting directories. Unless noted otherwise, Python scripts inside
subdirectories are intended to be run from the repository root; the
subdirectories exist to reduce clutter.

```
BrainRotViT/
|-- root pipeline scripts        Main training, inference, and interpretability
|-- best_checkpoints/            Split archive of trained ViT and CNN weights
|-- preproc_scripts/             Per-dataset MRI preprocessing (bash)
|-- skullstrip_scripts/          Per-dataset skull stripping (Python)
|-- comparison_methods/          Baseline models for benchmarking
|-- adni_analysis/               Alzheimer's disease / MCI brain-age-gap study
|-- abide_analysis/              Autism spectrum disorder brain-age-gap study
|-- maps_out_dora/               Attention-map post-processing and AAL mapping
```

### Root pipeline

| File | Description |
| --- | --- |
| [vit_dora_train_feature_cnn_main_mix_roi.py](vit_dora_train_feature_cnn_main_mix_roi.py) | Main training pipeline. Trains the ViT feature extractor with DoRA on the age-sex classification task, extracts per-slice embeddings, and trains the residual CNN regressor for brain-age prediction. |
| [vit_dora_train_feature_cnn_main_mix_roi_test.py](vit_dora_train_feature_cnn_main_mix_roi_test.py) | Inference and evaluation counterpart of the training script. Runs the trained model on the validation and held-out cohorts and reports metrics. |
| [cnn_mx_bigdo_ch_sw_res.py](cnn_mx_bigdo_ch_sw_res.py) | Definition of the residual CNN regression head (`AgePredictionCNN`), including the SiLU-activated residual convolutional blocks and fully connected layers with late sex fusion. |
| [dataset_cls.py](dataset_cls.py) | PyTorch `Dataset` classes for loading slice embeddings, sex, age, and domain labels. |
| [3dmap_grad_vit_cnn_main_mix_roi_dora.py](3dmap_grad_vit_cnn_main_mix_roi_dora.py) | Interpretability pipeline. Fuses ViT patch attention with CNN guided backpropagation per subject and aggregates the result into a 3D attention volume. |
| [stability_sanity_dora.py](stability_sanity_dora.py) | Saliency-map credibility analysis: inter-subject stability, split-half reliability, and model-parameter randomization sanity checks (Adebayo et al.). |
| [run_age_range_maps.sh](run_age_range_maps.sh) | Generates per-age-range 3D attention maps and reduces each to its AAL-atlas-fit NIfTI and region ranking. |
| [map_slice_compare.sh](map_slice_compare.sh) | Builds side-by-side comparisons of attention-map slices across output folders. |

### Supporting directories

- **[best_checkpoints/](best_checkpoints/)** A multi-part 7-Zip archive of the
  trained ViT and CNN weights. Extract with `7z x best_checkpoints.7z.001` and
  place the results in `model_dumps/` (ViT) and `model_dumps/mix/` (CNN) in the
  repository root. See [best_checkpoints/README.md](best_checkpoints/README.md).
- **[preproc_scripts/](preproc_scripts/)** Per-dataset bash scripts implementing
  the harmonization pipeline: skull stripping, N4 bias-field correction, affine
  registration to a common template, cropping, resampling to 160 sagittal
  slices, and intensity normalization. One script per cohort.
- **[skullstrip_scripts/](skullstrip_scripts/)** Per-dataset Python scripts for
  non-brain tissue removal using the DeepBrain U-Net skull stripping tool.
- **[comparison_methods/](comparison_methods/)** Baseline architectures used to
  benchmark BrainRotViT under identical preprocessing: SFCN
  ([sfcn_run.py](comparison_methods/sfcn_run.py)), 3D-ResNet
  ([3dresnet.py](comparison_methods/3dresnet.py)), 3D-ViT
  ([3dvit.py](comparison_methods/3dvit.py)), the Global-Local Transformer
  ([globallocal.py](comparison_methods/globallocal.py),
  [GlobalLocalTransformer.py](comparison_methods/GlobalLocalTransformer.py)),
  Triamese-ViT ([triamese.py](comparison_methods/triamese.py)), and TSAN
  adjuster utilities.
- **[adni_analysis/](adni_analysis/)** Downstream brain-age-gap study on the ADNI
  cohort, including diagnosis classification
  ([ADNI_vit_cnn_diagnosis_cls.py](adni_analysis/ADNI_vit_cnn_diagnosis_cls.py))
  and crude, bias-corrected, and covariate-adjusted BAG statistics for
  Alzheimer's disease and mild cognitive impairment groups.
- **[abide_analysis/](abide_analysis/)** Parallel downstream BAG study on the
  ABIDE-II cohort for the autism spectrum disorder group.
- **[maps_out_dora/](maps_out_dora/)** Post-processing utilities for the 3D
  attention volume. [center.py](maps_out_dora/center.py) crops, centers, and
  resizes the attention map, and
  [intense_regions_max.py](maps_out_dora/intense_regions_max.py) ranks AAL3
  regions by weighted attention intensity. The AAL crop template
  `aal_crop_centered.nii` is tracked here.

## Installation

The code was developed with Python 3.11 and PyTorch 2.4. Install the
dependencies into a fresh environment:

```
pip install -r requirements.txt
```

Key dependencies include PyTorch and torchvision, `transformers` and `timm`
(ViT and DoRA adaptation), `nibabel` and `SimpleITK` (NIfTI handling and N4
correction), `deepbrain` (skull stripping), and the standard scientific Python
stack.

## Reproducing the Results

1. **Preprocessing.** Skull-strip each dataset with the matching script in
   [skullstrip_scripts/](skullstrip_scripts/), then run the harmonization
   pipeline with the matching script in [preproc_scripts/](preproc_scripts/).
2. **Training.** Train the model with
   [vit_dora_train_feature_cnn_main_mix_roi.py](vit_dora_train_feature_cnn_main_mix_roi.py),
   or extract the released weights from [best_checkpoints/](best_checkpoints/).
3. **Evaluation.** Evaluate on the validation and held-out cohorts with
   [vit_dora_train_feature_cnn_main_mix_roi_test.py](vit_dora_train_feature_cnn_main_mix_roi_test.py).
4. **Interpretability.** Generate and validate attention maps with
   [3dmap_grad_vit_cnn_main_mix_roi_dora.py](3dmap_grad_vit_cnn_main_mix_roi_dora.py)
   and [stability_sanity_dora.py](stability_sanity_dora.py).
5. **Clinical analysis.** Run the cohort-specific brain-age-gap analyses in
   [adni_analysis/](adni_analysis/) and [abide_analysis/](abide_analysis/).

A fixed random seed is set across the training and interpretability scripts so
that the data split and results are reproducible.

## Results

On the pooled validation set (subject-level split), BrainRotViT attains a mean
absolute error of 3.43 years (Pearson r = 0.98, R^2 = 0.96, Spearman rho = 0.97)
and generalizes to the four held-out cohorts with an MAE of 4.79 to 5.35 years,
outperforming the retrained 3D-ResNet, SFCN, 3D-ViT, Global-Local Transformer,
and TSAN baselines under identical preprocessing. Full quantitative tables and
ablations are reported in the paper.

## Citation

If you use BrainRotViT in your research, please cite:

```bibtex
@misc{jalal2025brainrotvittransformerresnethybridexplainable,
      title={BrainRotViT: Transformer-ResNet Hybrid for Explainable Modeling of Brain Aging from 3D sMRI}, 
      author={Wasif Jalal and Md Nafiu Rahman and Atif Hasan Rahman and M. Sohel Rahman},
      year={2025},
      eprint={2511.15188},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.15188}, 
}
```
