### CNN Image Classification 

Last update: May 2020

---

Starter code for (robust) image classification with deep residual networks.

Contains implementations of the following models, for CIFAR-10 and ImageNet:

- ResNet [1]
- ResNet V2, often called "pre-activation" [2]
- Wide ResNet [3]
- Squeeze and Excitation ResNet [4]

**Robustness**

We are interested in a *robust* model, which is to say: although we train on a distribution <img alt="$P$" src="svgs/df5a289587a2f0247a5b97c1e8ac58ca.svg" align="middle" width="12.83677559999999pt" height="22.465723500000017pt"/>, we still want good performance for a worst-case distribution <img alt="$Q$" src="svgs/1afcdb0f704394b16fe85fb40c45ca7a.svg" align="middle" width="12.99542474999999pt" height="22.465723500000017pt"/> within some uncertainty set. Implicitly this relies on a definition of distance between distributions. Below we consider two examples.

For <img alt="$\ell_p$" src="svgs/ca185a0f63add2baa6fe729fd1cfef60.svg" align="middle" width="13.625845199999988pt" height="22.831056599999986pt"/> adversarial robustness, we implement empirical attacks:

- PGD for L-2 and L-inf robustness [5]
- FGSM for L-inf robustness [6]

For semantic robustness, the state-of-the-art defense is advanced data augmentation techniques such as such as AugMix or Expectation over Transformation. This is still todo.

**Calibration**

Beyond discrimination, we are interested in the *calibration* of a model. It turns out that deep classifiers are not calibrated by default, despite log-likelihood being a proper scoring rule [7]. This is a result of over-fitting to the training data [8]. Mixup [9] is implemented because it's been shown to improve calibration by regularizing the model [10]. We evaluate with the de-biased <img alt="$\ell_2$" src="svgs/336fefe2418749fabf50594e52f7b776.svg" align="middle" width="13.40191379999999pt" height="22.831056599999986pt"/> calibration metric over a pre-specified number of bins, following [11]. 

We remark that *adversarial training* without mixup makes it easier to yield well-calibrated models because it takes longer (more iterations) to overfit to the training data. Yet adversarial training does eventually overfit [12], so calibration eventually suffers unless further regularization is applied.

---

#### References

[1] K. He, X. Zhang, S. Ren, & J. Sun, Deep Residual Learning for Image Recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016), pp. 770–778. https://doi.org/10.1109/CVPR.2016.90.

[2] K. He, X. Zhang, S. Ren, & J. Sun, Identity Mappings in Deep Residual Networks. In B. Leibe, J. Matas, N. Sebe, & M. Welling,eds., Computer Vision – ECCV 2016 (Cham: Springer International Publishing, 2016), pp. 630–645. https://doi.org/10.1007/978-3-319-46493-0_38.

[3] S. Zagoruyko & N. Komodakis, Wide Residual Networks. Procedings of the British Machine Vision Conference 2016 (York, UK: British Machine Vision Association, 2016), pp. 87.1-87.12. https://doi.org/10.5244/C.30.87.

[4] J. Hu, L. Shen, & G. Sun, Squeeze-and-Excitation Networks. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (Salt Lake City, UT: IEEE, 2018), pp. 7132–7141. https://doi.org/10.1109/CVPR.2018.00745.

[5] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, & A. Vladu, Towards Deep Learning Models Resistant to Adversarial Attacks. International Conference on Learning Representations (2018).

[6] E. Wong, L. Rice, & J. Z. Kolter, Fast is better than free: Revisiting adversarial training. International Conference on Learning Representations (2020).

[7] J. Bröcker, Estimating reliability and resolution of probability forecasts through decomposition of the empirical score. *Climate Dynamics*, **39** (2012) 655–667. https://doi.org/10.1007/s00382-011-1191-1.

[8] C. Guo, G. Pleiss, Y. Sun, & K. Q. Weinberger, On Calibration of Modern Neural Networks. International Conference on Machine Learning (2017), pp. 1321–1330.

[9] H. Zhang, M. Cisse, Y. N. Dauphin, & D. Lopez-Paz, mixup: Beyond Empirical Risk Minimization. International Conference on Learning Representations (2018).

[10] S. Thulasidasan, G. Chennupati, J. A. Bilmes, T. Bhattacharya, & S. Michalak, On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d\textquotesingle Alché-Buc, E. Fox, & R. Garnett,eds., Advances in Neural Information Processing Systems 32 (Curran Associates, Inc., 2019), pp. 13888–13899.

[11] A. Kumar, P. S. Liang, & T. Ma, Verified Uncertainty Calibration. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d\textquotesingle Alché-Buc, E. Fox, & R. Garnett,eds., *Advances in Neural Information Processing Systems 32* (Curran Associates, Inc., 2019), pp. 3787–3798.

[12] L. Rice, E. Wong, & J. Z. Kolter, *Overfitting in adversarially robust deep learning* (2020).