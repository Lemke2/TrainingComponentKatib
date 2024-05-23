# TrainingComponentKatib

Katib is Google Kubeflow's automl tool, used for finding the optimal hyperparameters for a model based on a variety of chosen algorithms.

Using Kubeflow's automl format, with pyhton's arg-parser to parse input arguments, and yaml files to provide the arguments, we train a given segmentation model (HDCNet, PSPNet, ResNet, SegNet, Unet, UperNet...)

We train all of them in a given hyperparameter space in order to quickly and efficiently find the optimal model/hyperparams for a specific problem.

We then further train a model with these specifications via Kubeflow Pipelines, and provide it in production using Kubeflow inference tools.
