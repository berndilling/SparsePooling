# Taken from:
# https://github.com/loeweX/Greedy_InfoMax

from optparse import OptionGroup

def parse_args(parser):
    group = OptionGroup(parser, "training options")
    group.add_option(
        "--num_epochs", type="int", default=10, help="Number of Epochs for Training"
    )
    group.add_option(
        "--start_epoch",
        type="int",
        default=0,
        help="Epoch to start GIM training from: "
        "v=0 - start training from scratch, "
        "v>0 - load pre-trained model that was trained for v epochs and continue training "
        "(path to pre-trained model needs to be specified in opt.model_path)",
    )
    group.add_option("--batch_size", type="int", default=32, help="Batchsize")
    group.add_option(
        "--in_channels_input",
        type=int,
        default=1,
        help="Number of in_channels to the SparsePooling network",
    )
    group.add_option(
        "--learning_rate", 
        type="float", 
        default=5e-3, #5e-3
        help="Learning rate for feedforward weights (W_ff). Learning rates for W_rec and thresholds are scaled accordingly"
    )
    group.add_option(
        "--learning_rate_decay", 
        type="float", 
        default=1., # 
        help="Learning rate decay for feedforward weights (W_ff). Learning rates for W_rec and thresholds are scaled accordingly."
    )
    group.add_option(
        "--weight_decay", 
        type="float", 
        default=0., 
        help="weight decay or l2-penalty on weights (for ADAM optimiser, default = 0., i.e. no l2-penalty)"
    )
    parser.add_option(
        "--dataset_type",
        type="string",
        default="static", # "moving"
        help="Can be static (for SC training) or moving (for SFA training); for moving, a sequence of moving image patches will be extracted instead of a single patch. The time dimension will be appended to batch dimension",
    )
    parser.add_option(
        "--sequence_length",
        type="int",
        default=10,
        help="Length (number of frames) of image patch sequences for SFA learning",
    )
    parser.add_option(
        "--dataset",
        type="string",
        default="olshausen", # "stl10"
        help="Dataset to use for training, default: olshausen",
    )
    group.add_option(
        "--preprocess",
        type="string",
        default="subtractmean", # whiten
        help="Preprocessing of extracted image patches (subtract pixel-wise mean or whitening)",
    )
    group.add_option(
        "--patch_size",
        type="int",
        default=20, 
        help="Size (number of pixels) of image patches"
    )
    group.add_option(
        "--N_patches_per_image",
        type="int",
        default=1000, 
        help="Number of image patches extracted from each image"
    )
    group.add_option(
        "--epsilon", 
        type="float", 
        default=1e-3, 
        help="(normalized) convergence threshold for SC/SFA lateral recurrence loop"
    )
    group.add_option(
        "--tau", 
        type="float", 
        default=.1, 
        help="time constant for lateral recurrence update (1=immediate update)"
    )
    group.add_option(
        "--enc_patch_size",
        type="int",
        default=16,
        help="Encoder patch size. Use single integer for same encoding size for all modules (default=16)",
    )
    group.add_option(
        "--random_crop_size",
        type="int",
        default=64,
        help="Size of the random crop window. Use single integer for same size for all modules (default=64)",
    )
    group.add_option(
        "--inference_recurrence",
        type="int",
        default=1,
        help="recurrence (on the layer level) during inference."
        "0 - no recurrence"
        "1 - lateral recurrence within layer"
    )
    group.add_option(
        "--train_module",
        type="int",
        default=3,
        help="Index of the module to be trained individually (0-2), "
        "or training network as one (3)",
    )
    group.add_option(
        "--save_dir",
        type="string",
        default="",
        help="If given, uses this string to create directory to save results in "
        "(be careful, this can overwrite previous results); "
        "otherwise saves logs according to time-stamp",
    )
    group.add_option(
        "--model_path",
        type="string",
        default=".",
        help="Directory of the saved model (path within --data_input_dir)",
    )
    group.add_option(
        "--model_num",
        type=int,
        default=10,
        help="(epoch) Number of model to classify from. ",
    )
    group.add_option(
        "--data_input_dir",
        type="string",
        default="./datasets",
        help="Directory to store bigger datafiles (dataset and models)",
    )
    group.add_option(
        "--data_output_dir",
        type="string",
        default=".",
        help="Directory to store bigger datafiles (dataset and models)",
    )
    group.add_option(
        "--classifying",
        action="store_true",
        default=False,
        help="Boolean to communicate classification mode",
    )
    group.add_option(
        "--dataset_class",
        type="string",
        default="stl10",
        help="Dataset to use for training, default: stl10",
    )
    group.add_option(
        "--class_from_layer",
        type=int,
        default=None,
        help="Layer number to classify from. Default None corresponds to the last layer. Keep the 0-indexing in mind!"
        "-1 means direct classification on the (flattened) input image",
    )
    group.add_option(
        "--create_hidden_representation",
        action="store_true",
        default=False,
        help="Boolean whether to create and save hidden representations (output of context model) of train set and train classifier on these."
        "CAREFUL: This is much faster but leads to overfitting on one of the random crops. Better test performance without that option!",
    )    
    group.add_option(
        "--download_dataset",
        action="store_true",
        default=False,
        help="Boolean to decide whether to download the dataset to train on (only tested for STL-10)",
    )
    group.add_option(
        "--validate",
        action="store_true",
        default=False,
        help="Boolean to decide whether to split train dataset into train/val and plot validation loss (True) or combine train+validation set for final testing (False)",
    )
    group.add_option(
        "--grayscale",
        action="store_true",
        default=True,
        help="[stl-10] Boolean to decide whether to convert images to grayscale (default: true)",
    )
    group.add_option(
        "--end_to_end_supervised",
        action="store_true",
        default=False,
        help="[stl-10] Boolean to decide whether full model should be trained end-to-end. CAREFUL: in this case the model should be standard CNN, NOT SparsePoolingModel",
    )
    group.add_option(
        "--in_channels",
        type=int,
        default=None,
        help="Option to explicitly specify the number of input channels for the linear classifier."
        "If None, the default options for resnet output is taken",
    )

    parser.add_option_group(group)
    return parser
