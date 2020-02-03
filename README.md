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

### Transfer learning

#### Top-20 classes

A ResNet-50 model pre-trained with ImageNet was used to learn the classification of top-20 classes. The specific hyperparameters can be seen in the [configuration file](/configs/transfer.gin).

Run with:

`$ python experiment.py -c configs/transfer.gin`

![Training progress](/images/train_val_top20.png?raw=true)

The model at around ~700 optimization steps was chosen to avoid overfitting.

![Confusion matrix](/images/cm_top20.png?raw=true)


#### Evaluations on test set
Run with:

`$ python experiment.py -c configs/transfer_evaluate.gin -p "TransferNet.saved_model_path='saved/model/path.mdl'"`

*Note that the class "Perfume and Body Mist" does not have any examples in the training set.*

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

#### Rare classes

The previous model was finetuned on the rare classes, i.e. the `train_other.csv` dataset. The hyperparameters can be found in the [finetune.gin](/configs/finetune.gin) configuration file.

Run with:

`$ python experiment.py -c configs/finetune.gin -p "TransferNet.saved_model_path='saved/model/path.mdl'"`

![Training progress](/images/train_val_finetune.png?raw=true)

![Confusion matrix](/images/cm_finetune.png?raw=true)

#### Evaluations on test set

Run with:

`$ python experiment.py -c configs/transfer_evaluate.py -p "TransferNet.saved_model_path='saved/model/path.mdl'"`

*Note that roughly 33.3% of datapoints in the test set are of classes which have no examples in training set.*

**Top-1 Accuracy:** 0.4941

**Top-5 Accuracy:** 0.6298

| Class        | Top-1 Accuracy | Top-5 Accuracy |
| -------------|---------------:|---------------:|
|Accessory Gift Set|0.9897|1.0000|
|Baby Dolls\*|0.0000|0.0000|
|Bangle|0.0870|0.8696|
|Basketballs|0.0000|1.0000|
|Bath Robe\*|0.0000|0.0000|
|Beauty Accessory\*|0.0000|0.0000|
|Body Lotion\*|0.0000|0.0000|
|Body Wash and Scrub\*|0.0000|0.0000|
|Bra|0.9939|1.0000|
|Bracelet|0.8571|1.0000|
|Camisoles|0.0294|0.8529|
|Capris|0.7885|0.9808|
|Caps|0.9466|0.9847|
|Churidar|0.5000|1.0000|
|Clutches|0.9861|1.0000|
|Compact\*|0.0000|0.0000|
|Concealer\*|0.0000|0.0000|
|Cufflinks|1.0000|1.0000|
|Deodorant\*|0.0000|0.0000|
|Dresses|0.8495|0.9462|
|Duffel Bag|0.6957|0.9130|
|Dupatta|0.8750|1.0000|
|Earrings|1.0000|1.0000|
|Eye Cream\*|0.0000|0.0000|
|Eyeshadow\*|0.0000|0.0000|
|Face Moisturisers\*|0.0000|0.0000|
|Face Scrub and Exfoliator\*|0.0000|0.0000|
|Face Serum and Gel\*|0.0000|0.0000|
|Face Wash and Cleanser\*|0.0000|0.0000|
|Flats|0.9747|1.0000|
|Footballs|1.0000|1.0000|
|Foundation and Primer\*|0.0000|0.0000|
|Fragrance Gift Set\*|0.0000|0.0000|
|Free Gifts|0.0833|0.4167|
|Gloves|0.6667|1.0000|
|Hair Colour\*|0.0000|0.0000|
|Hat\*|0.0000|0.0000|
|Headband|0.0000|1.0000|
|Highlighter and Blush\*|0.0000|0.0000|
|Innerwear Vests|0.0000|0.0000|
|Jackets|0.6330|0.9255|
|Jewellery Set|1.0000|1.0000|
|Jumpsuit|0.0000|1.0000|
|Kajal and Eyeliner\*|0.0000|0.0000|
|Key chain|0.0000|0.0000|
|Kurta Sets|1.0000|1.0000|
|Kurtis|0.6818|0.9886|
|Laptop Bag|0.7714|1.0000|
|Leggings|0.6458|0.8958|
|Lip Care\*|0.0000|0.0000|
|Lip Gloss\*|0.0000|0.0000|
|Lip Liner\*|0.0000|0.0000|
|Lip Plumper\*|0.0000|0.0000|
|Lipstick\*|0.0000|0.0000|
|Lounge Pants|0.0000|0.5263|
|Lounge Shorts|0.0000|0.7778|
|Makeup Remover|0.0000|0.6667|
|Mascara\*|0.0000|0.0000|
|Mask and Peel\*|0.0000|0.0000|
|Mens Grooming Kit\*|0.0000|0.0000|
|Messenger Bag|0.5000|0.9583|
|Mobile Pouch|0.3333|0.9444|
|Mufflers|0.0690|0.9655|
|Nail Essentials\*|0.0000|0.0000|
|Nail Polish\*|0.0000|0.0000|
|Necklace and Chains|0.9250|1.0000|
|Nightdress|0.1688|0.8438|
|Night suits|0.3143|0.8143|
|Pendant|0.8519|0.9630|
|Rain Jacket|1.0000|1.0000|
|Rain Trousers|0.0000|0.0000|
|Ring|0.9483|1.0000|
|Robe|0.0000|0.0000|
|Rucksacks|0.8750|1.0000|
|Scarves|0.7174|0.7826|
|Shapewear|1.0000|1.0000|
|Shoe Accessories|0.0000|0.0526|
|Shoe Laces\*|0.0000|0.0000|
|Shorts|0.8436|0.9526|
|Shrug|0.0000|1.0000|
|Skirts|0.6981|0.8302|
|Sports Sandals|0.7667|1.0000|
|Stockings|0.1765|0.7059|
|Stoles|0.4000|0.9000|
|Sunscreen\*|0.0000|0.0000|
|Sweaters|0.6667|0.9412|
|Sweatshirts|0.6498|0.9689|
|Swimwear|0.0000|0.5000|
|Ties|0.9880|0.9880|
|Ties and Cufflinks\*|0.0000|0.0000|
|Tights|0.0000|1.0000|
|Toner\*|0.0000|0.0000|
|Track Pants|0.8902|0.9595|
|Tracksuits|0.7000|1.0000|
|Travel Accessory|0.1000|0.7000|
|Trousers|0.9157|0.9885|
|Tunics|0.4000|0.9778|
|Umbrellas|0.0000|0.0000|
|Waistcoat|0.1667|1.0000|
|Waist Pouch|0.0000|1.0000|
|Water Bottle|1.0000|1.0000|
|Wristbands|0.2500|1.0000|

*\*: no examples of these classes in training set*

### Image similarity embedding with triplet loss

A pre-trained Resnet-50 was trained to embed the product images in to a vector space such that visually similar images are closer to each other, and vice versa.
This is done using a so called [triplet loss](https://arxiv.org/abs/1503.03832), which takes as input the network outputs for three sample images from the dataset:
 - an *anchor* image
 - a random *positive* image which belongs to the same class as the anchor
 - a random *negative* image which belongs to a different class

The loss function then encourages the network to keep the distance between the embeddings of the anchor and the positive sample smaller than between the anchor and the negative sample. The loss is defined as

**L(a,p,n) = max(d(f(a),f(p)) - d(f(a),f(n)) + margin, 0)**

where **f(x)** is the embedding function learned by the model, and **d(x,y)** is some distance metric on the embedding space. Here L2-distance was used.
A simple non-parametric k-nearest neighbors algorithm is then used to classify the embeddings. The intuition is that KNN should perform well with unbalanced classes, given that the embedding is sufficient.

The datasets used are the same as above, with the exception of classes with fewer than 4 samples being removed.

Run with:
`$ python experiment.py -c configs/triplet.gin`

#### Evaluations on test set

![t-sne projection](/images/tsne.png?raw=true)

*[Click here for a higher quality interactive plot](https://users.aalto.fi/~tellaa2/fashion_scatter.html)*

The plot above shows a t-sne projection of the learned embeddings evaluated on the test set. Despite some overlapping and outliers, the data points are relatively distinctly clustered, and similar clusters such as different kinds of shoes and trousers are close to each other.


**Accuracy (top-20 classes):** 0.8672

**Accuracy (all classes):** 0.7466

**Accuracy (rare classes):** 0.4916


### Possible improvements
 - improve embedding accuracy:
     - use smarter sampling method for the negative samples in triplet loss
