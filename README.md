# hiking-trail-ai
Following hiking trails with a small AI (suits edge computing). W/ Self-recorded dataset.

## TLDR of Report
We train an AI for hiking trail directional estimation. There are two main goals: recording data to build a dataset, and training a light-weight model.

* **Dataset**: hiking trail directional datasets are sparse, and lack enough trail variety for good deep learning generalization. So, we record our own data to use alongside the already existing data. This is recorded on a couple trails and are categorized based on which of the "left", "right" and "forward" cameras records the footage.

* **Model**: Convolutional Neural Networks (CNNs) tend to be heavy and slow, making them unsuited for edge computing. Since we want a lightweight AI that can be deployed in the wild. The MobileNetV2 architecture is thus employed, an architecture which attempts to reduce CNN size while remaining performant. 

* **Results**: The model was small enough to load on a Raspeberry Pi 3, with a processing rate of 2 imgs per sec. The AI's performance was decent when tested on a trail not in the training set, but not good enough for practical use. More testing would need to be done to check whether that is a fault of the architecture or insufficient training data variety.

## Where's the Dataset
The dataset we recorded ourselves is available for download [HERE](https://www.kaggle.com/dataset/c89aa7511b75aeb9b8a2ed35b43f449d35ad7b788d645a57153c14e0b52937d1). Before augmentations contains >70k images, divided across 3 directional categories. 
For other data used for training, refer to the report.

## Where's the AI
While the model itself isn't published for now, you can find here all resources to replicate the results: the code, dataset, train/test methodology and logs.
