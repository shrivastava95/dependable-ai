# files made:
1. `train_cel.py` 
2. `main.py`
3. `data_new.py` (contains the new dataset wrapper that reads attacked samples from `dataset/pgd_samples_CIFAR10.pt`)
4. took `dataset` folder from JS2
5. took `Exps/sample/model.py` and subsequently `Exps/sample/inception.py` (loads the pretrained network from `Exps/sample/inceptionv3_state.pth`, which is the classifier model to be defended)
6. took and modified `resnext.py` and subsequently `resnext101.py` - the models are U-Net decoders and both have a denoising module (`net.denoise`) and a classifier module (`net.net`). The classifier module has `net.denoise` inbuilt into it, which is used to denoise the input image before feeding it to the classifier in case `defense=True` is specified in the `forward()` call of `net.net`.
7. `net.net` returns a list of length 2. the first element is the features, the second element is the logits. Refer to Figure 4 (a) and (b) in the original paper to see why this is done.
8. the code we typed uses Figure4 (b) which is Logits Guided Denoiser (minimizes the loss between logits of denoised images versus logits of original images)
9. `net(orig_x, adv_x)` does:
    - `orig_outputs = net.net(orig_x)`                                           - classifies unperturbed image
    - `adv_outputs = net.net(adv_x, defense=True)`                     - denoises the adv image and classifies the denoised image
    - `loss = net.loss(adv_outputs, orig_outputs)`                        - loss between denoised and unperturbed
    - `control_outputs = net.net(adv_x)`                                       - classifies adv image
    - `control_loss = net.loss(control_outputs, orig_outputs)`      - loss between adv and unperturbed
10. The `train` function logs the training metrics in output and saves the original image, the adv image, the predicted noise, and the label in `noise/noise_cifar10_resnet_pgd.pt`
11. The `train` function in `train_cel.py` runs and logs the outputs of the training and testing data for one epoch. it saves to the `noise` folder.

Script for running code in the repository for 10 epochs with lr 3e-4 and batch size 16, storing the inference on test and train along the way:- 
`python main.py --exp sample --workers 1 --epochs 10 --start-epoch 0 --batch-size 16 --learning-rate 0.0003 --weight-decay 0 --save-freq 1 --print-iter 1 --save-dir noise  --test 0 --test_e4 0 --defense 1 --optimizer adam`

The gitignore ignores the logs outputted in `dataset/*` and `noise/*`


12. Add `eval.py`, `train_eval.py`, `eval.sh` and `eval_windows.sh` (equivalent to eval.sh, for use on lower resource laptop). Use `eval.sh` to run the evaluation script on the train and test dataset. The shell script runs `eval.py` with the appropriate flags, some of which are redundant and may be removed in the future. `eval.py` internally calls `train_eval.py` to run the model on the train and test loader. Add `make_datasets` folder to store the pretraining scripts.


# TODO:
13. (incomplete) add the MNIST dataset wala part and run it. (files added: `data_new_mnist.py`, `datasets/MNIST` folder)
14. (incomplete) add `make_datasets_MNIST.py`, finish and run it to create the attacked samples for MNIST dataset for ResNet18.
15. (yet to start) add the unsupervised training edit from ishaan's minor 1 paper and run it.
16. (yet to start) add the intuitive explanations of why the modelling noise is better than modelling cleaned images. 
17. (yet to start) start adding the parts related to your minor 1 and explain how this can offer a univerally defensed classifier.
