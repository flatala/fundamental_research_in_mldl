# fundamental_research_in_ml_dl

## General Info
The updated version of the repo for the [Why Can't I Dance in the Mall? Learning to Mitigate Scene Bias in Action Recognition](https://arxiv.org/abs/1912.05534) paper.
The updates include the following:
- training / validation / testing files updated to support latest versions of PyTorch and cuda. Original code supports pytorch < 1.0 with breaking changes introduced since
- code for generating human masks for ucf101
- code for generating human + randomised masks for ucf101
- code for generating pseudolabels using a resnet model, as well as using an llm
- additional support for scene classification accuracy testing on ucf101
- improved logging
- clenead up training and evaluation commands

## Useful links:
- Original repo of the authors': [https://github.com/vt-vl-lab/SDN](https://github.com/vt-vl-lab/SDN) 
- Docker image with updated dependencies: **flatala/mldl:latest** on dockerhub
- Docker image with original (outdated) dependencies: **flatala/mldl:fl_public** on dockerhub
