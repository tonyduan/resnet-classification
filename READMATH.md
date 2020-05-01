### Image Classification 

Starter code for (robust) image classification with deep residual networks.

Contains implementations of the following models, for CIFAR-10 and ImageNet:

- ResNet [1]
- ResNet V2, often called "pre-activation" [2]
- Wide ResNet [3]
- Squeeze and Excitation ResNet [4]

We are interested in a *robust* model, which is to say: even though we train on a distribution P,
we still want good performance for a worst-case distribution Q within some uncertainty set.

For $\ell_p$ adversarial robustness, we implement empirical attacks:

- PGD for L-2 and L-inf robustness [5]
- FGSM for L-inf robustness [6]

For semantic robustness, we still need to implement advanced data augmentation 
such as AugMix or Expectation over Transformation.

Beyond discrimination, we are interested in the *calibration* of a model. 
It turns out that deep classifiers are not calibrated by default [7].
Mixup [8] is implemented because it's been shown to improve calibration by regularizing labels
in the presence of high dimensional inputs [9].

#### References

[1] K. He, X. Zhang, S. Ren, & J. Sun, Deep Residual Learning for Image Recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016), pp. 770–778. https://doi.org/10.1109/CVPR.2016.90.
[2] K. He, X. Zhang, S. Ren, & J. Sun, Identity Mappings in Deep Residual Networks. In B. Leibe, J. Matas, N. Sebe, & M. Welling,eds., Computer Vision – ECCV 2016 (Cham: Springer International Publishing, 2016), pp. 630–645. https://doi.org/10.1007/978-3-319-46493-0_38.
[3] S. Zagoruyko & N. Komodakis, Wide Residual Networks. Procedings of the British Machine Vision Conference 2016 (York, UK: British Machine Vision Association, 2016), pp. 87.1-87.12. https://doi.org/10.5244/C.30.87.
[4] J. Hu, L. Shen, & G. Sun, Squeeze-and-Excitation Networks. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (Salt Lake City, UT: IEEE, 2018), pp. 7132–7141. https://doi.org/10.1109/CVPR.2018.00745.
[5] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, & A. Vladu, Towards Deep Learning Models Resistant to Adversarial Attacks. International Conference on Learning Representations (2018).
[6] E. Wong, L. Rice, & J. Z. Kolter, Fast is better than free: Revisiting adversarial training. International Conference on Learning Representations (2020).
[7] C. Guo, G. Pleiss, Y. Sun, & K. Q. Weinberger, On Calibration of Modern Neural Networks. International Conference on Machine Learning (2017), pp. 1321–1330.
[8] H. Zhang, M. Cisse, Y. N. Dauphin, & D. Lopez-Paz, mixup: Beyond Empirical Risk Minimization. International Conference on Learning Representations (2018).
[9] S. Thulasidasan, G. Chennupati, J. A. Bilmes, T. Bhattacharya, & S. Michalak, On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d\textquotesingle Alché-Buc, E. Fox, & R. Garnett,eds., Advances in Neural Information Processing Systems 32 (Curran Associates, Inc., 2019), pp. 13888–13899.

