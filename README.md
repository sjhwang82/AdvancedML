# Advanced Deep Learning @ KAIST

## Course Information
**Instructor:** Sung Ju Hwang (sjhwang82@kaist.ac.kr)  

**Teaching Assistants:**  
Haebeom Lee (haebeom.lee@kaist.ac.kr)   
Moonsu Han (mshan92@kaist.ac.kr)  
Sunghyun Park (psh01087@kaist.ac.kr)

**Office:** E3-1, Room 1427 (Instructor) Room 1435 (TAs)  
Office hours: By appointment only.

### Grading Policy
* **Absolute Grading**
* Paper Presentation: 20%
* Attendance and Participation: 20%
* Project: 60% 

## Tentative Schedule

| Dates | Topic | 
|---|:---|
|3/17| Course Introduction |
|3/19| Review of Deep Learning Basics |
|3/24| Review of Deep Learning Basics |
|3/26| Bayesian Deep Learning (Lecture) | 
|3/31| Bayesian Deep Learning (Lecture) | 
|4/2| Bayesian Deep Learning (Presentation) |
|4/7| Generative Adversarial Networks (Lecture) | 
|4/9| Generative Adversarial Networks (Presentation) | 
|4/14| Autoregressive and Flow-Based Generative Models (Lecture) | 
|4/16| Autoregressive and Flow-Based Generative Models (Presentation), **Project Proposal Due 4/17** |
|4/21| Deep Reinforcement Learning (Lecture) | 
|4/23| Deep Reinforcement Learning (Presentation) | 
|4/28| Memory- and Computation-Efficient Deep Learning (Lecture) | 
|5/7| **Mid-term Presentation** 
|5/12| Memory- and Computation-Efficient Deep Learning (Presentation) | 
|5/14| Meta-Learning (Lecture) | 
|5/19| Meta-Learning (Presentation) | 
|5/21| Continual Learning (Lecture) | 
|5/26| Continual Learning (Presentation) | 
|5/28| Interpretable Deep Learning (Lecture) | 
|6/2| Interpretable Deep Learning (Presentation) | 
|6/4| Reliable Deep Learning (Lecture) | 
|6/9| Reliable Deep Learning (Presentation) |
|6/11| Adversarial Deep Learning (Lecture), **Final Paper Due 6/12** |
|6/16| Adversarial Deep Learning (Presentation) |  
|6/18| Graph Neural Networks (Lecture) | 
|6/23| Graph Neural Networks (Presentation) | 
|6/26| **Half-day Workshop**


## Reading List

### Bayesian Deep Learning
[[Kingma and Welling 14]](https://arxiv.org/pdf/1312.6114.pdf) Auto-Encoding Variational Bayes, ICLR 2014.   
[[Kingma et al. 15]](https://arxiv.org/pdf/1506.02557.pdf) Variational Dropout and the Local Reparameterization Trick, NIPS 2015.   
[[Blundell et al. 15]](https://arxiv.org/pdf/1505.05424.pdf) Weight Uncertainty in Neural Networks, ICML 2015.   
[[Gal and Ghahramani 16]](http://proceedings.mlr.press/v48/gal16.pdf) Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning, ICML 2016.   
[[Gal et al. 17]](https://papers.nips.cc/paper/6949-concrete-dropout.pdf) Concrete Dropout, NIPS 2017.  
[[Gal et al. 17]](http://proceedings.mlr.press/v70/gal17a/gal17a.pdf) Deep Bayesian Active Learning with Image Data, ICML 2017.  
[[Kendal and Gal 17]](https://papers.nips.cc/paper/7141-what-uncertainties-do-we-need-in-bayesian-deep-learning-for-computer-vision.pdf) What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?, ICML 2017.  
[[Teye et al. 18]](http://proceedings.mlr.press/v80/teye18a/teye18a.pdf) Bayesian Uncertainty Estimation for Batch Normalized Deep Networks, ICML 2018.  
[[Garnelo et al. 18]](http://proceedings.mlr.press/v80/garnelo18a/garnelo18a.pdf) Conditional Neural Process, ICML 2018.  
[[Kim et al. 19]](http://https://arxiv.org/pdf/1901.05761.pdf) Attentive Neural Processes, ICLR 2019.  
***
[[Sun et al. 19]](https://arxiv.org/pdf/1903.05779.pdf) Functional Variational Bayesian Neural Networks, ICLR 2019.  
[[Louizos et al. 19]](http://papers.nips.cc/paper/9079-the-functional-neural-process.pdf) The Functional Neural Process, NeurIPS 2019.  
[[Maddox et al. 19]](https://papers.nips.cc/paper/9472-a-simple-baseline-for-bayesian-uncertainty-in-deep-learning.pdf) A Simple Baseline for Bayesian Uncertainty in Deep Learning, NeurIPS 2019. 


### Generative Adversarial Networks
[[Goodfellow et al. 14]](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) Generative Adversarial Nets, NIPS 2014.   
[[Radford et al. 15]](https://arxiv.org/abs/1511.06434) Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, ICLR 2016.   
[[Chen et al. 16]](https://papers.nips.cc/paper/6399-infogan-interpretable-representation-learning-by-information-maximizing-generative-adversarial-nets.pdf) InfoGAN: Interpreting Representation Learning by Information Maximizing Generative Adversarial Nets, NIPS 2016.   
[[Arjovsky et al. 17]](http://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf) Wasserstein Generative Adversarial Networks, ICML 2017.  
[[Zhu et al. 17]](https://arxiv.org/pdf/1703.10593.pdf) Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks, ICCV 2017.  
[[Zhang et al. 17]](https://arxiv.org/pdf/1706.03850.pdf) Adversarial Feature Matching for Text Generation, ICML 2017.  
[[Karras et al. 18]](https://openreview.net/forum?id=Hk99zCeAb) Progressive Growing of GANs for Improved Quality, Stability, and Variation, ICLR 2018.  
***
[[Brock et al. 19]](https://openreview.net/pdf?id=B1xsqj09Fm) Large Scale GAN Training for High-Fidelity Natural Image Synthesis, ICLR 2019.  
[[Karras et al. 19]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.pdf) A Style-Based Generator Architecture for Generative Adversarial Networks, CVPR 2019.  
[[Xu et al. 19]](https://papers.nips.cc/paper/8953-modeling-tabular-data-using-conditional-gan.pdf) Modeling Tabular Data using Conditional GAN, NeurIPS 2019.    



### Autoregressive and Flow-Based Generative Models
[[Rezende and Mohamed 15]](http://proceedings.mlr.press/v37/rezende15.pdf) Variational Inference with Normalizing Flows, ICML 2015.   
[[Germain et al. 15]](http://proceedings.mlr.press/v37/germain15.pdf) MADE: Masked Autoencoder for Distribution Estimation, ICML 2015.  
[[Kingma et al. 16]](https://papers.nips.cc/paper/6581-improved-variational-inference-with-inverse-autoregressive-flow.pdf) Improved Variational Inference with Inverse Autoregressive Flow, NIPS 2016.  
[[Oord et al. 16]](http://proceedings.mlr.press/v48/oord16.pdf) Pixel Recurrent Neural Networks, ICML 2016.  
[[Dinh et al. 17]](https://openreview.net/pdf?id=HkpbnH9lx) Density Estimation Using Real NVP, ICLR 2017.  
[[Papamakarios et al. 17](https://papers.nips.cc/paper/6828-masked-autoregressive-flow-for-density-estimation.pdf) Masked Autoregressive Flow for Density Estimation, NIPS 2017.  
[[Huang et al.18]](http://proceedings.mlr.press/v80/huang18d/huang18d.pdf) Neural Autoregressive Flows, ICML 2018.  
[[Kingma and Dhariwal 18]](http://papers.nips.cc/paper/8224-glow-generative-flow-with-invertible-1x1-convolutions.pdf) Glow: Generative Flow with Invertible 1x1 Convolutions, NeurIPS 2018.  
***
[[Ho et al. 19]](http://proceedings.mlr.press/v97/ho19a/ho19a.pdf) Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design, ICML 2019.    
[[Chen et al. 19]](https://papers.nips.cc/paper/9183-residual-flows-for-invertible-generative-modeling.pdf) Residual Flows for Invertible Generative Modeling, NeurIPS 2019.   
[[Kumar et al. 20]](https://openreview.net/pdf?id=rJgUfTEYvH) VideoFlow: A Conditional Flow-Based Model for Stochastic Video Generation, ICLR 2020.    


### Deep Reinforcement Learning 
[[Mnih et al. 13]](https://arxiv.org/pdf/1312.5602.pdf) Playing Atari with Deep Reinforcement Learning, NIPS Deep Learning Workshop 2013.  
[[Silver et al. 14]](http://proceedings.mlr.press/v32/silver14.pdf) Deterministic Policy Gradient Algorithms, ICML 2014.      
[[Schulman et al. 15]](https://arxiv.org/pdf/1502.05477.pdf) Trust Region Policy Optimization, ICML 2015.  
[[Lillicrap et al. 16]](https://arxiv.org/pdf/1509.02971.pdf) Continuous Control with Deep Reinforcement Learning, ICLR 2016.    
[[Schaul et al. 16]](https://arxiv.org/pdf/1511.05952.pdf) Prioritized Experience Replay, ICLR 2016.  
[[Wang et al. 16]](http://proceedings.mlr.press/v48/wangf16.pdf) Dueling Network Architectures for Deep Reinforcement Learning, ICML 2016.    
[[Mnih et al. 16]](http://proceedings.mlr.press/v48/mniha16.pdf) Asynchronous Methods for Deep Reinforcement Learning, ICML 2016.  
[[Schulman et al. 17]](https://arxiv.org/pdf/1707.06347.pdf) Proximal Policy Optimization Algorithms, arXiv preprint, 2017.  
[[Nachum et al. 18]](https://papers.nips.cc/paper/7591-data-efficient-hierarchical-reinforcement-learning.pdf) Data-Efficient Hierarchical Reinforcement Learning, NeurIPS 2018.  
[[Ha et al. 18]](https://papers.nips.cc/paper/7512-recurrent-world-models-facilitate-policy-evolution.pdf) Recurrent World Models Facilitate Policy Evolution, NeurIPS 2018.  
***
[[Burda et al. 19]](https://openreview.net/forum?id=rJNwDjAqYX) Large-Scale Study of Curiosity-Driven Learning, ICLR 2019.  
[[Vinyals et al. 19]](https://www.nature.com/articles/s41586-019-1724-z) Grandmaster level in StarCraft II using multi-agent reinforcement learning, Nature, 2019.  
[[Bellemare et al. 19]](https://papers.nips.cc/paper/8687-a-geometric-perspective-on-optimal-representations-for-reinforcement-learning.pdf) A Geometric Perspective on Optimal Representations for Reinforcement Learning, NeurIPS 2019.  
[[Janner et al. 19]](http://papers.nips.cc/paper/9416-when-to-trust-your-model-model-based-policy-optimization.pdf) When to Trust Your Model: Model-Based Policy Optimization, NeurIPS 2019.  
[[Fellows et al. 19]](https://papers.nips.cc/paper/8934-virel-a-variational-inference-framework-for-reinforcement-learning.pdf) VIREL: A Variational Inference Framework for Reinforcement Learning, NeurIPS 2019.  
[[Hafner et al. 20]](https://openreview.net/pdf?id=S1lOTC4tDS) Dream to Control: Learning Behaviors by Latent Imagination, ICLR 2020.  
[[Kaiser et al. 20]](https://openreview.net/pdf?id=S1xCPJHtDB) Model Based Reinforcement Learning for Atari, ICLR 2020.  

### Memory and Computation-Efficient Deep Learning
[[Han et al. 15]](https://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network.pdf) Learning both Weights and Connections for Efficient Neural Networks, NIPS 2015.  
[[Wen et al. 16]](https://papers.nips.cc/paper/6504-learning-structured-sparsity-in-deep-neural-networks.pdf) Learning Structured Sparsity in Deep Neural Networks, NIPS 2016  
[[Han et al. 16]](https://arxiv.org/pdf/1510.00149.pdf) Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding, ICLR 2016  
[[Molchanov et al. 17]](https://arxiv.org/pdf/1701.05369.pdf) Variational Dropout Sparsifies Deep Neural Networks, ICML 2017  
[[Luizos et al. 17]](https://papers.nips.cc/paper/6921-bayesian-compression-for-deep-learning.pdf) Bayesian Compression for Deep Learning, NIPS 2017.  
[[Luizos et al. 18]](https://openreview.net/pdf?id=H1Y8hhg0b) Learning Sparse Neural Networks Through L0 Regularization, ICLR 2018.    
[[Howard et al. 18]](https://arxiv.org/pdf/1704.04861.pdf) MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications, CVPR 2018.    
[[Frankle and Carbin 19]](https://https://openreview.net/pdf?id=rJl-b3RcF7) The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks, ICLR 2019.  
***
[[Lee et al. 19]](https://openreview.net/pdf?id=B1VZqjAcYX) SNIP: Single-Shot Network Pruning Based On Connection Sensitivity, ICLR 2019.  
[[Liu et al. 19]](https://openreview.net/pdf?id=rJlnB3C5Ym) Rethinking the Value of Network Pruning, ICLR 2019.  
[[Jung et al. 19]](https://arxiv.org/abs/1808.05779) Learning to Quantize Deep Networks by Optimizing Quantization Intervals with Task Loss, CVPR 2019.  
[[Morcos et al. 19]](https://papers.nips.cc/paper/8739-one-ticket-to-win-them-all-generalizing-lottery-ticket-initializations-across-datasets-and-optimizers.pdf) One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers, NeurIPS 2019.  
[[Renda et al. 20]](https://openreview.net/pdf?id=S1gSj0NKvB) Comparing Rewinding and Fine-tuning in Neural Network Pruning, ICLR 2020.  

### Meta Learning
[[Santoro et al. 16]](http://proceedings.mlr.press/v48/santoro16.pdf) Meta-Learning with Memory-Augmented Neural Networks, ICML 2016  
[[Vinyals et al. 16]](https://arxiv.org/pdf/1606.04080.pdf) Matching Networks for One Shot Learning, NIPS 2016    
[[Edwards and Storkey 17]](https://arxiv.org/pdf/1606.02185.pdf) Towards a Neural Statistician, ICLR 2017  
[[Finn et al. 17]](https://arxiv.org/pdf/1703.03400.pdf) Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks, ICML 2017  
[[Snell et al. 17]](https://arxiv.org/pdf/1703.05175.pdf) Prototypical Networks for Few-shot Learning, NIPS 2017  
[[Lee and Choi 18]](https://arxiv.org/abs/1801.05558) Gradient-Based Meta-Learning with Learned Layerwise Metric and Subspace, ICML 2018  
[[Liu et al. 19]](https://openreview.net/pdf?id=SyVuRiC5K7) Learning to Propagate Labels: Transductive Propagation Network for Few-shot Learning, ICLR 2019 
***

[[Gordon et al. 19]](https://openreview.net/pdf?id=HkxStoC5F7) Meta-Learning Probabilistic Inference for Prediction, ICLR 2019  
[[Ravi and Beatson 19]](https://openreview.net/pdf?id=rkgpy3C5tX) Amortized Bayesian Meta-Learning, ICLR 2019.    
[[Rakelly et al. 19]](http://proceedings.mlr.press/v97/rakelly19a/rakelly19a.pdf) Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables, ICML 2019.  
[[Lee et al. 20]](https://openreview.net/pdf?id=rkeZIJBYvr) Learning to Balance: Bayesian Meta-Learning for Imbalanced and Out-of-distribution Tasks, ICLR 2020

### Continual Learning
[[Rusu et al. 16]](https://arxiv.org/pdf/1606.04671.pdf) Progressive Neural Networks, arXiv preprint, 2016  
[[Kirkpatrick et al. 17]](https://arxiv.org/pdf/1612.00796.pdf) Overcoming catastrophic forgetting in neural networks, PNAS 2017  
[[Lee et al. 17]](https://papers.nips.cc/paper/7051-overcoming-catastrophic-forgetting-by-incremental-moment-matching.pdf) Overcoming Catastrophic Forgetting by Incremental Moment Matching, NIPS 2017  
[[Shin et al. 17]](https://papers.nips.cc/paper/6892-continual-learning-with-deep-generative-replay.pdf) Continual Learning with Deep Generative Replay, NIPS 2017.  
[[Lopez-Paz and Ranzato 17]](https://papers.nips.cc/paper/7225-gradient-episodic-memory-for-continual-learning.pdf) Gradient Episodic Memory for Continual Learning, NIPS 2017.  
[[Yoon et al. 18]](https://openreview.net/pdf?id=Sk7KsfW0-) Lifelong Learning with Dynamically Expandable Networks, ICLR 2018.  
[[Nguyen et al. 18]](https://arxiv.org/pdf/1710.10628.pdf) Variational Continual Learning, ICLR 2018.  
[[Schwarz et al. 18]](http://proceedings.mlr.press/v80/schwarz18a/schwarz18a.pdf) Progress & Compress: A Scalable Framework for Continual Learning, ICML 2018.   
***
[[Chaudhry et al. 19]](https://openreview.net/pdf?id=Hkf2_sC5FX) Efficient Lifelong Learning with A-GEM, ICLR 2019.  
[[Rao et al. 19]](https://papers.nips.cc/paper/8981-continual-unsupervised-representation-learning.pdf) Continual Unsupervised Representation Learning, NeurIPS 2019.  
[[Rolnick et al. 19]](https://papers.nips.cc/paper/8327-experience-replay-for-continual-learning.pdf) Experience Replay for Continual Learning, NeurIPS 2019.   
[[Yoon et al. 20]](https://openreview.net/forum?id=r1gdj2EKPB) Scalable and Order-robust Continual Learning with Additive Parameter Decomposition, ICLR 2020.  

### Interpretable Deep Learning
[[Ribeiro et al. 16]](https://arxiv.org/pdf/1602.04938.pdf) "Why Should I Trust You?" Explaining the Predictions of Any Classifier, KDD 2016  
[[Kim et al. 16]](https://people.csail.mit.edu/beenkim/papers/KIM2016NIPS_MMD.pdf) Examples are not Enough, Learn to Criticize! Criticism for Interpretability, NIPS 2016  
[[Choi et al. 16]](https://papers.nips.cc/paper/6321-retain-an-interpretable-predictive-model-for-healthcare-using-reverse-time-attention-mechanism.pdf) RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism, NIPS 2016  
[[Koh et al. 17]](https://arxiv.org/pdf/1703.04730.pdf) Understanding Black-box Predictions via Influence Functions, ICML 2017  
[[Bau et al. 17]](https://arxiv.org/pdf/1704.05796.pdf) Network Dissection: Quantifying Interpretability of Deep Visual Representations, CVPR 2017  
[[Selvaraju et al. 17]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf) Grad-CAM: Visual Explanation from Deep Networks via Gradient-based Localization, ICCV 2017.  
[[Kim et al. 18]](http://proceedings.mlr.press/v80/kim18d/kim18d.pdf) Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV), ICML 2018.  
[[Heo et al. 18]](http://papers.nips.cc/paper/7370-uncertainty-aware-attention-for-reliable-interpretation-and-prediction.pdf) Uncertainty-Aware Attention for Reliable Interpretation and Prediction, NeurIPS 2018.   
***

[[Bau et al. 19]](https://openreview.net/pdf?id=Hyg_X2C5FX) GAN Dissection: Visualizing and Understanding Generative Adversarial Networks, ICLR 2019.   
[[Guan et al. 19]](http://proceedings.mlr.press/v97/guan19a/guan19a.pdf) Towards a Deep and Unified Understanding of Deep Neural Models in NLP, ICML 2019.  
[[Ghorbani et al. 19]](https://papers.nips.cc/paper/9126-towards-automatic-concept-based-explanations.pdf) Towards Automatic Concept-based Explanations, NeurIPS 2019.  
[[Chen et al. 19]](https://papers.nips.cc/paper/9095-this-looks-like-that-deep-learning-for-interpretable-image-recognition.pdf) This Looks Like That: Deep Learning for Interpretable Image Recognition, NeurIPS 2019.  



### Reliable Deep Learning
[[Guo et al. 17]](https://arxiv.org/pdf/1706.04599.pdf) On Calibration of Modern Neural Networks, ICML 2017.   
[[Lakshminarayanan et al. 17]](http://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf) Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles, NIPS 2017.  
[[Liang et al. 18]](https://openreview.net/pdf?id=H1VGkIxRZ) Enhancing the Reliability of Out-of-distrubition Image Detection in Neural Networks, ICLR 2018.  
[[Lee et al. 18]](https://openreview.net/pdf?id=ryiAv2xAZ) Training Confidence-calibrated Classifiers for Detecting Out-of-Distribution Samples, ICLR 2018.  
[[Kuleshov et al. 18]](https://arxiv.org/pdf/1807.00263.pdf) Accurate Uncertainties for Deep Learning Using Calibrated Regression, ICML 2018.    
[[Jiang et al. 18]](https://papers.nips.cc/paper/7798-to-trust-or-not-to-trust-a-classifier.pdf) To Trust Or Not To Trust A Classifier, NeurIPS 2018.  
[[Madras et al. 18]](https://papers.nips.cc/paper/7853-predict-responsibly-improving-fairness-and-accuracy-by-learning-to-defer.pdf) Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer, NeurIPS 2018.  
***
[[Kull et al. 19]](https://papers.nips.cc/paper/9397-beyond-temperature-scaling-obtaining-well-calibrated-multi-class-probabilities-with-dirichlet-calibration.pdf) Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration, NeurIPS 2019.  
[[Thulasidasan et al. 19]](https://papers.nips.cc/paper/9540-on-mixup-training-improved-calibration-and-predictive-uncertainty-for-deep-neural-networks.pdf) On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks, NeurIPS 2019.  
[[Ovadia et al. 19]](https://papers.nips.cc/paper/9547-can-you-trust-your-models-uncertainty-evaluating-predictive-uncertainty-under-dataset-shift.pdf) Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift, NeurIPS 2019.  
[[Hendrycks et al. 20]](https://openreview.net/pdf?id=S1gmrxHFvB) AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty, ICLR 2020.  


### Deep Adversarial Learning
[[Szegedy et al. 14]](https://arxiv.org/pdf/1312.6199.pdf) Intriguing Properties of Neural Networks, ICLR 2014.    
[[Goodfellow et al. 15]](https://arxiv.org/pdf/1412.6572.pdf) Explaining and Harnessing Adversarial Examples, ICLR 2015.    
[[Kurakin et al. 17]](https://openreview.net/pdf?id=BJm4T4Kgx) Adversarial Machine Learning at Scale, ICLR 2017.    
[[Madry et al. 18]](https://openreview.net/pdf?id=rJzIBfZAb) Toward Deep Learning Models Resistant to Adversarial Attacks, ICLR 2018.    
[[Eykholt et al. 18]](https://arxiv.org/pdf/1707.08945.pdf) Robust Physical-World Attacks on Deep Learning Visual Classification.  
[[Athalye et al. 18]](https://arxiv.org/pdf/1802.00420.pdf) Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples, ICML 2018.   
***
[[Zhang et al. 19]](http://proceedings.mlr.press/v97/zhang19p/zhang19p.pdf) Theoretically Principled Trade-off between Robustness and Accuracy, ICML 2019.  
[[Ilyas et al. 19]](https://papers.nips.cc/paper/8307-adversarial-examples-are-not-bugs-they-are-features) Adversarial Examples are not Bugs, They Are Features, NeurIPS 2019.  
[[Li et al. 19]](https://papers.nips.cc/paper/9143-certified-adversarial-robustness-with-additive-noise.pdf) Certified Adversarial Robustness with Additive Noise, NeurIPS 2019.  
[[Tramèr and Boneh]](https://papers.nips.cc/paper/8821-adversarial-training-and-robustness-for-multiple-perturbations.pdf) Adversarial Training and Robustness for Multiple Perturbations, NeurIPS 2019.  
[[Shafahi et al. 19]](https://papers.nips.cc/paper/8597-adversarial-training-for-free.pdf) Adversarial Training for Free!, NeurIPS 2019.  
[[Wong et al. 20]](https://openreview.net/pdf?id=BJx040EFvH) Fast is Better Than Free: Revisiting Adversarial Training, ICLR 2020.  

### Graph Neural Networks
[[Li et al. 16]](https://arxiv.org/pdf/1511.05493.pdf) Gated Graph Sequence Neural Networks, ICLR 2016.  
[[Hamilton et al. 17]](https://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf) Inductive Representation Learning on Large Graphs, NIPS 2017.  
[[Kipf and Welling 17]](https://openreview.net/pdf?id=SJU4ayYgl) Semi-Supervised Classification with Graph Convolutional Networks, ICLR 2017.  
[[Velickovic et al. 18]](https://openreview.net/pdf?id=rJXMpikCZ) Graph Attention Networks, ICLR 2018.   
[[Ying et al. 18]](https://papers.nips.cc/paper/7729-hierarchical-graph-representation-learning-with-differentiable-pooling.pdf) Hierarchical Graph Representation Learning with Differentiable Pooling, NeurIPS 2018.  
***
[[Yun et al. 19]](https://papers.nips.cc/paper/9367-graph-transformer-networks.pdf) Graph Transformer Neteworks, NeurIPS 2019.    
[[Hu et al. 20]](https://openreview.net/forum?id=HJlWWJSFDH) Strategies for Pre-training Graph Neural Networks, ICLR 2020.  
[[Vashishth et al. 20]](https://openreview.net/pdf?id=BylA_C4tPr) Composition-based Multi-Relational Graph Convolutional Networks, ICLR 2020.   


### Neural Architecture Search 
[[Zoph and Le 17]](https://arxiv.org/abs/1611.01578) Neural Architecture Search with Reinforcement Learning, ICLR 2017.  
[[Baker et al. 17]](https://openreview.net/pdf?id=S1c2cvqee) Designing Neural Network Architectures using Reinforcement Learning, ICLR 2017.  
[[Real et al. 17]](http://proceedings.mlr.press/v70/real17a/real17a.pdf) Large-Scale Evolution of Image Classifiers, ICML 2017.  
[[Liu et al. 18]](https://openreview.net/pdf?id=BJQRKzbA-) Hierarchical Representations for Efficient Architecture Search, ICLR 2018.  
[[Pham et al. 18]](http://proceedings.mlr.press/v80/pham18a.html) Efficient Neural Architecture Search via Parameters Sharing, ICML 2018.  
[[Luo et al. 18]](https://papers.nips.cc/paper/8007-neural-architecture-optimization.pdf) Neural Architecture Optimization, NeurIPS 2018.  
[[Liu et al. 19]](https://openreview.net/pdf?id=S1eYHoC5FX) DARTS: Differentiable Architecture Search, ICLR 2019.  
[[Tan et al. 19]](https://arxiv.org/abs/1807.11626) MnasNet: Platform-Aware Neural Architecture Search for Mobile, CVPR 2019.  
***
[[Cai et al. 19]](https://openreview.net/pdf?id=HylVB3AqYm) ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware, ICLR 2019.  
[[Zhou et al. 19]](http://proceedings.mlr.press/v97/zhou19e/zhou19e.pdf) BayesNAS: A Bayesian Approach for Neural Architecture Search, ICML 2019.  
[[Tan and Le 19]](http://proceedings.mlr.press/v97/tan19a/tan19a.pdf) EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks, ICML 2019.  
[[Guo et al. 19]](https://papers.nips.cc/paper/8362-nat-neural-architecture-transformer-for-accurate-and-compact-architectures.pdf) NAT: Neural Architecture Transformer for Accurate and Compact Architectures, NeurIPS 2019.  
[[Chen et al. 19]](https://papers.nips.cc/paper/8890-detnas-backbone-search-for-object-detection.pdf) DetNAS: Backbone Search for Object Detection, NeurIPS 2019.  
[[Dong and Yang 20]](https://openreview.net/forum?id=HJxyZkBKDr) NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search, ICLR 2020.  
[[Zela et al. 20]](https://openreview.net/pdf?id=H1gDNyrKDS) Understanding and Robustifying Differentiable Architecture Search, ICLR 2020.  


### Federated Learning
[[Konečný et al. 16]](https://arxiv.org/pdf/1610.02527.pdf) Federated Optimization: Distributed Machine Learning for On-Device Intelligence, arXiv Preprint, 2016.  
[[Konečný et al. 16]](https://arxiv.org/abs/1610.054920) Federated Learning: Strategies for Improving Communication Efficiency, NIPS Workshop on Private Multi-Party Machine Learning 2016.  
[[McMahan et al. 17]](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf) Communication-Efficient Learning of Deep Networks from Decentralized Data, AISTATS 2017.   
[[Smith et al. 17]](https://papers.nips.cc/paper/7029-federated-multi-task-learning.pdf) Federated Multi-Task Learning, NIPS 2017.  
[[Li et al. 20]](https://proceedings.mlsys.org/static/paper_files/mlsys/2020/176-Paper.pdf) Federated Optimization in Heterogeneous Networks, MLSys 2020.   
***
[[Mohri et al. 19]](http://proceedings.mlr.press/v97/mohri19a/mohri19a.pdf) Agnostic Federated Learning, ICML 2019.  
[[Yurochkin et al. 19]](http://proceedings.mlr.press/v97/yurochkin19a/yurochkin19a.pdf) Bayesian Nonparametric Federated Learning of Neural Networks, ICML 2019.  
[[Bonawitz et al. 19]](https://arxiv.org/pdf/1902.01046.pdf) Towards Federated Learning at Scale: System Design, MLSys 2019.  
[[Wang et al. 20]](https://openreview.net/forum?id=BkluqlSFDS) Federated Learning with Matched Averaging, ICLR 2020.  
[[Li et al. 20]](https://openreview.net/pdf?id=HJxNAnVtDS) On the Convergence of FedAvg on Non-IID data, ICLR 2020.  





<!--
## Deep Learning for Time-series Analysis
[[Che et al. 18]](http://proceedings.mlr.press/v80/che18a/che18a.pdf) Hierarchical Deep Generative Models for Multi-Rate Multivariate Time Series, ICML 2018  
[[Campos et al. 18]](https://openreview.net/pdf?id=HkwVAXyCW) Skip RNN: Learning to Skip State Updates in Recurrent Neural Networks, ICLR 2018  
[[Tatbul et al. 18]](https://papers.nips.cc/paper/7462-precision-and-recall-for-time-series.pdf) Precision and Recall for Time Series, NeurIPS 2018  
[[Rangapuram et al. 18]](https://papers.nips.cc/paper/8004-deep-state-space-models-for-time-series-forecasting.pdf) Deep State Space Models for Time Series Forecasting, NeurIPS 2018  
## Deep Learning for Computer Vision
[[Ren et al. 15]](https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf_) Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, NIPS 2015  
[[Liu et al. 16]](https://arxiv.org/pdf/1512.02325.pdf) SSD: Single Shot MultiBox Detector, ECCV 2016  
[[Long et al. 15]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf) Fully Convolutional Networks for Semantic Segmentation, CVPR 2015  
[[Xu et al. 15]](http://proceedings.mlr.press/v37/xuc15.pdf) Show, Attend and Tell: Neural Image Caption Generation with Visual Attention, ICML 2015  
[[Held et al. 16]](http://davheld.github.io/GOTURN/GOTURN.pdf) Learning to Track at 100 FPS with Deep Regression Networks, ECCV 2016  
***
[[Lin et al. 17]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf) Focal Loss for Dense Object Detection, ICCV 2017  
[[Chen et al. 18]](https://arxiv.org/pdf/1802.02611.pdf) Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation, ECCV 2018  
[[Girdhar et al. 18]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Girdhar_Detect-and-Track_Efficient_Pose_CVPR_2018_paper.pdf) Detect-and-Track: Efficient Pose Estimation in Videos, CVPR 2018
-->  





