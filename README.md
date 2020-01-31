## Transfer learning / Few-shot learning

Some transfer learning and few-shot learning experiments on the [Fashion Product Images dataset](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset/version/1).

### Setup

Download the [dataset](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset/version/1) and extract it to a desired location. Run the `create_splits.sh` script:

`$ sh ./create_splits.sh path/to/styles.csv output_dir`

The script will create four files in `output_dir`:

 - `train_top20.csv`, which contains a dataset of fashion product from even years which belong to the top-20 most frequent categories
 - `train_other.csv`, which contains a dataset of products from even years that don't belong to the top-20 categories
 - `test.csv`, containing a testing dataset of all products from odd years
 - `classes.txt`, which contains all the unique class names (in alphabetical order)

Next, install Pip requirements:

`$ pip install -r requirements.txt`

The path to the dataset root folder should be specified in

### Usage

Experiments are defined with [Gin configuration files](https://github.com/google/gin-config), located in the `configs` directory. To run an experiment, pass the desired configuration file to `experiment.py`, e.g.

`$ python experiment.py -c configs/transfer.gin`

An `experiments` folder will be created in the current directory, which will be used to store experiment artifacts.

To override configuration parameters from command line, use the `-p/--parameter` option:

`$ python experiment.py -c configs/transfers.gin -p "main.num_epochs=10" -p "torch.optim.Adam.lr=1e-3"`

The configuration options for the dataset root folder and CSV file should be set accordingly, either directly in the configuration files or from command line:

`FashionDataset.root_dir = '/path/to/fashion-dataset'`

`FashionDataset.csv_file = '/path/to/train_top20.csv'`

#### Running fine-tuning on a pre-trained model

`$ python experiment.py -c configs/finetune.gin -p "TransferNet.saved_model_path='path/to/model.mdl'"`

#### Running evaluation on test set

`$ python experiment.py -c configs/transfer_evaluate.gin -p "TransferNet.saved_model_path='path/to/model.mdl'"`

### Results

#### Transfer learning

##### Top-20 classes

A ResNet-50 model pre-trained with ImageNet was used to learn the classification of top-20 classes. The specific hyperparameters can be seen in the [configuration file](/configs/transfer.gin).

![Training progress](/images/train_val_top20.png?raw=true)

The model at around ~700 optimization steps was chosen to avoid overfitting.

![Confusion matrix](/images/cm_top20.png?raw=true)


###### Evaluations on test set
**Top-1 Accuracy:** 0.8771

**Top-5 Accuracy:** 0.9566

| Class        | Top-1 Accuracy | Top-5 Accuracy |
| -------------|---------------:|---------------:|
|Jeans|0.9852|0.9963|
|Perfume and Body Mist|0.0000|0.0000|
|Formal Shoes|0.8813|0.9892|
|Socks|0.9171|0.9724|
|Backpacks|0.9308|0.9958|
|Belts|0.9813|0.9907|
|Briefs|0.8663|1.0000|
|Sandals|0.8071|1.0000|
|Flip Flops|0.8971|0.9755|
|Wallets|0.9673|1.0000|
|Sunglasses|1.0000|1.0000|
|Heels|0.8084|0.9945|
|Handbags|0.9801|0.9960|
|Tops|0.6600|1.0000|
|Kurtas|0.9815|0.9986|
|Sports Shoes|0.8874|0.9957|
|Watches|1.0000|1.0000|
|Casual Shoes|0.7946|0.9969|
|Shirts|0.9919|1.0000|
|Tshirts|0.9509|0.9993|
