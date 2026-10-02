
## 1. Open Knee MRI & Ligament Injury Datasets (For AI & Computer Vision)
Data to train models to recognize ACL/PCL tears or knee structure, several major institutions provide open-source repositories:

* 
* [Stanford MRNet Dataset](https://aimi.stanford.edu/datasets/mrnet-knee-mris): A highly popular dataset containing 1,370 knee MRI exams from the Stanford University Medical Center. It includes explicitly labeled clinical data for ACL tears, meniscal tears, and other abnormalities. [1] 
* [KneeMRI Dataset on Zenodo](https://zenodo.org/records/14789903): A public repository featuring 917 retrospective grayscale volumes of knee scans. Each is manually and double-blind annotated into three categories: healthy, partially injured, or completely ruptured ACLs. [2, 3, 4] 
* [Mendeley Data ACL-PCL Dataset](https://data.mendeley.com/datasets/fjns2xzhrr): A curated, preprocessed collection of knee MRI scans explicitly annotated across multiple views (sagittal, axial, coronal) for normal vs. torn ACL and PCL conditions. [5] 
* [Roboflow Universe ACL Datasets](https://universe.roboflow.com/search?q=class%3Aacl): Publicly accessible, user-contributed computer vision datasets (like skm-mri or DeepKnee) that feature annotated knee joint anatomy and localized ligament damage. [6] 
* 

## 2. Public Arthroplasty & Orthopedic Clinical Databases
Clinical outcome metrics, patient-reported questionnaires, or general orthopedic data rather than imagery:

* 
* The ACORN Registry (Arthroplasty Clinical Outcomes Registry National): While focused on total hip and knee replacements rather than isolated ligament reconstructions, ACORN has made an extensive dataset available for public research use. It details pre-operative patient expectations and 6-month post-operative outcomes.
* [OEHR (Orthopedic Electronic Health Record Dataset)](https://dl.acm.org/doi/10.1145/3626772.3657885): A public, multimodal benchmark database combining orthopedic electronic health records, text summaries, and medical imaging to assist clinical and data science research.
* [Mendeley Ortho Trauma Dataset](https://data.mendeley.com/datasets/zgxwbhsc3d): An open spreadsheet resource compiling clinical data points specifically targeted at orthopedic trauma research. [7, 8, 9, 10] 
* 

## 3. Aggregated International Registry Reports
Not raw patient-level data matrices but statistical insights:

* 
* Platforms affiliated with the [International Society of Arthroplasty Registries (ISAR)](https://actaorthop.org/actao/article/view/42706) frequently share open-access, standardized annual reports mapping data quality, revision rates, and methodology summaries across global knee and joint tracking initiatives. [11] 

## 4. Sarcoma Subtype / Radiographic Morphology:
* Radiogenomic signature discovery for sarcoma subtypes — Correlate radiomic features (texture, shape, intensity heterogeneity) extracted from CT/MR with specific mutations/CNV/expression clusters (e.g., MDM2 amplification in liposarcoma, TP53 status in leiomyosarcoma). 
* TCGA-SARC (via TCIA)
* GDC project page (to cross-reference genomic data per case ID): portal.gdc.cancer.gov/projects/TCGA-SARC

* Proof-of-concept / feasibility pipeline study — Frame it as "we built and demonstrated an end-to-end radiogenomic pipeline (segmentation → radiomics extraction → genomic correlation) on available cases," not as a validated predictive model. This is legitimate and publishable as a methods/feasibility paper, explicitly scoped as hypothesis-generating.

* Combine cohorts to get real n — As discussed, merge TCGA-SARC with other TCIA sarcoma collections (Soft-tissue-Sarcoma, WORC, etc.) to reach a workable sample size (ideally 50+ for any defensible ML, ~100+ better). This is the actual path to a viable pilot classifier study — not using TCGA-SARC alone.

[1] [https://aimi.stanford.edu](https://aimi.stanford.edu/datasets/mrnet-knee-mris)
[2] [https://zenodo.org](https://zenodo.org/records/14789903)
[3] [https://www.kaggle.com](https://www.kaggle.com/datasets/sohaibanwaar1203/kneemridataset)
[4] [https://zenodo.org](https://zenodo.org/records/14789903)
[5] [https://data.mendeley.com](https://data.mendeley.com/datasets/fjns2xzhrr)
[6] [https://universe.roboflow.com](https://universe.roboflow.com/search?q=class%3Aacl)
[7] [https://dl.acm.org](https://dl.acm.org/doi/10.1145/3626772.3657885)
[8] [https://unsworks.unsw.edu.au](https://unsworks.unsw.edu.au/entities/dataset/564380e3-8f87-40a9-98c6-f14e9dc61ec8)
[9] https://www.acornregistry.org
[10] [https://data.mendeley.com](https://data.mendeley.com/datasets/zgxwbhsc3d)
[11] [https://actaorthop.org](https://actaorthop.org/actao/article/view/42706)

