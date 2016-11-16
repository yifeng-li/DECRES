This package is named DECRES (DEep learning for identifying Cis-Regulatory ElementS). DECRES is an extension of the Deep Learning Tutorials developped by LISA lab (www.deeplearning.net/tutorial/). Although DECRES is developped for the identification of CREs, it can also be used for other applications. 

DECRES is composed of 7 parts as follows.

1. Modified methods from Deep Learning Tutorials:
Multi-class logistic/softmax regression: logistic_sgd.py
Multilayer perceptrons (MLP): mlp.py
Denoising autoencoder (dA): dA.py
Contractive autoencoder (cA): cA.py
Stacked denoising autoencoder (SdA): SdA.py
Stacked contractive autoencoder (ScA): ScA.py
Restricted Boltzman machine (RBM): rbm.py
Deep belief network (DBN, stacked restricted Boltzman machine): DBN.py
Convolutional neural network (CNN): convolutional_mlp.py

2. Our deep-feature-selection (DFS) models:
Deep feature selection based on MLP: deep_feat_select_mlp.py
Deep feature selection based on ScA: deep_feat_select_ScA.py
Deep feature selection based on DBN: deep_feat_select_DBN.py
Randomized deep feature selection: randomized_dfs.py

4. New convolutional neural network for integrating multiple sources of data:
Integrative convolutional neural network (iCNN): icnn.py

5. A utility module for classification is included. 
This module is named classification.py. It includes normlization methods, class and feature pre-processing functions, post-processing functions, and visualizations of classification results. See the beginning of this module for usage information.

6. Examples:
For every methods in 1 and 2, an example is provided to demonstrate how to use it.
The file names of these examples are main_[module_name].py

7. Data:
We include our data sets for the cis-regulatory element classifications, so that our results can be reproduced and the new methods. Your new methods can be conveniently tested on these data set. These data are for 8 cell lines including four cell lines with sufficient samples and features: (GM12878, HelaS3, HepG2, K562), and four cell lines with limited number of enhancers or/and features: HUVEC, A549, HMEC, MCF7. Taking GM12878 as an example, the data are explained as below.
GM12878_200bp_Data.txt : the feature values, each row is a sample/instance/example/feature_vector, each column corresponds to a feature.
GM12878_200bp_Classes.txt : the class labels corresponding to the samples in GM12878_200bp_Data.txt.
GM12878_200bp_Regions.bed : the fixed-length (200bp) regions corresponding to the samples in GM12878_200bp_Data.txt.
GM12878_Features.txt : the list of feature names.
GM12878_Regions.bed : the original regions of variable length corresponding to the samples in GM12878_200bp_Data.txt.

Installation:
Very easy!
1. Download the code, save it in your local directory.
2. Add the directory to your Pyhton path, and you are ready to use it.

Citation:

@INPROCEEDINGS{LiRECOMB2015,
    AUTHOR = "Y. Li and C. Chen and W. Wasserman",
    TITLE = "Deep feature selection: {T}heory and application to identify enhancers and promoters",
    BOOKTITLE = "2015 Annual International Conference on Research in Computational Molecular Biology",
    VOLUME = {LNBI 9029},
    PAGES = "21-33",
    ADDRESS= "",
    ORGANIZATION = "",
    PUBLISHER="",
    MONTH = "April",
    YEAR = {2015}}

@ARTICLE{Li2016a,
    AUTHOR = "Y. Li and C. Chen and W.W. Wasserman",
    TITLE = "Deep feature selection: {T}heory and application to identify enhancers and promoters",
    JOURNAL = "Journal of Computational Biology",
    VOLUME = {23},
    NUMBER = {5},
    PAGES = {322-336},
    MONTH = "",
    YEAR = {2016}}

@ARTICLE{Li2015a,
    AUTHOR = "Y. Li and C. Chen and A.M. Kaye and W.W. Wasserman",
    TITLE = "The Identification of cis-Regulatory Elements: A Review from a Machine Learning Perspective",
    JOURNAL = "BioSystems",
    VOLUME = {138},
    NUMBER = {},
    PAGES = {6-17},
    MONTH = "",
    YEAR = {2015}}

@ARTICLE{Li2016b,
    AUTHOR = "Y. Li and W. Shi and W.W. Wasserman",
    TITLE = "Genome-Wide Prediction of \textit{cis}-Regulatory Regions Using Supervised Deep Learning Methods",
    JOURNAL = "bioRxiv",
    VOLUME = {},
    NUMBER = {},
    PAGES = {doi: http://dx.doi.org/10.1101/041616},
    MONTH = "",
    YEAR = {2016}}

License:
See LICENSE_Original_Deep_Learning_Tutorials.txt
Note, we also reserve the copyright on the part we contributed in DECRES.

Other Useful Information:
[1] Deep Learning Tutorials (www.deeplearning.net/tutorial/).
[2] http://deeplearning.cs.toronto.edu/
[3] UFLDL Tutorial: http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial

===========================================================
Contact:

Wyeth W. Wasserman, Ph.D
Executive Director, Child and Family Research Institute (CFRI)
Associate Dean (Research), Faculty of Medicine, UBC
Senior Scientist, CMMT/CFRI, UBC
Professor, Department of Medical Genetics, UBC
Principal Investigator
http://www.cmmt.ubc.ca/research/investigators/wasserman
http://www.cmmt.ubc.ca/directory/faculty/wyeth-wasserman
Email: wyeth@cmmt.ubc.ca

Yifeng Li, Ph.D
Post-Doctoral Research Fellow
Wasserman Lab
Centre for Molecular Medicine and Therapeutics
Department of Medical Genetics
University of British Columbia
Child and Family Research Institute
Vancouver, BC, Canada
Email: yifeng@cmmt.ubc.ca, yifeng.li.cn@gmail.com
Home Page: http://www.cmmt.ubc.ca/directory/faculty/yifeng-li
NMF Toolbox: https://sites.google.com/site/nmftool
SR Toolbox:    https://sites.google.com/site/sparsereptool
RLMK Toolbox: https://sites.google.com/site/rlmktool
PGM Toolbox:   https://sites.google.com/site/pgmtool
Spectral Clustering Toolbox: https://sites.google.com/site/speclust
===========================================================
