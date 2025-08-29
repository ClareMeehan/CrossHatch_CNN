# CrossHatch_CNN

## Introduction:
This project aims to develop a Convolutional Neural Network (CNN) that can take images from the TVAC testing of the Nancy Grace Roman Space Telescope at Goddard Space Flight Center and identify instances of the detector effect known as crosshatching. Crosshatching is a type of dark current detector effect that arises in H4RG detectors. These are small-scale (subpixel) QE fluctuations seen in HgCdTe detectors, and these patterns have been observed in some of the detectors for Roman. Crosshatching arises along the three crystal axes in the HgCdTe and has the potential to create issues in various data analyses conducted on the data. As such, it is helpful to know where the crosshatching arises in order to mask it. Some examples of crosshatching are shown below.

<img width="1203" height="395" alt="Screenshot 2025-08-29 at 5 57 52 PM" src="https://github.com/user-attachments/assets/c056465f-8a22-4fe9-9493-5770f76d380f" />

The current map of crosshatching location is shown below, as well as an updated version where the author of this project noticed crosshatching in the classification of the data.

<img width="1326" height="437" alt="Screenshot 2025-08-29 at 5 58 31 PM" src="https://github.com/user-attachments/assets/4817479b-8db9-4da0-9fae-7ff91bca6571" />

For this project, approximately 1,212 whole images were examined for crosshatching, with roughly 80% used for training and 20% for validation.

## Methods:

The method of reference pixel corrections follows from the WFI triplet test (https://archive.stsci.edu/missions-and-data/roman/wfi-triplet-test-data). This involves a standard row correction, then (rather than a slope correction) a simple background subtraction of the second measure of the exposure is used, followed by a normalization without outliers for the CNN. This is the point at which the training data was classified, meaning the images were examined for crosshatching, and if crosshatching was present, the “patch” of the image containing it was identified. This model is patch-based, meaning the images are divided into a 4x4 grid with IDs 1 through 16.

Through initial testing, I found that since the crosshatching pattern is so small/fine, the model often had difficulty identifying it. However, when the images were divided into smaller patches, results improved, and the filters/model were better able to find the pattern. The patches and their labels (patch data, label, filepath, patch ID) were saved as tensors to be loaded into the model.

The first step of this model is that Gabor filters are applied to the image data. As crosshatching results from the crystal structure in the HgCdTe layer of the chips, the crosshatching lines are always at the same angles from the horizontal. Gabor filters are a kind of linear filter with a reference direction, meaning one can input the angles from the horizontal that these lines fall on (I measured 68, 114, 34, and 170 degrees from the horizontal) and generate output that emphasizes these regions. After running these images through the Gabor filters, they are saved in large tensor chunks for faster data loading.

For the model, these large tensor chunks are loaded and put into data loaders with a batch size of 32. However, one recurring issue is that the data is highly unbalanced: in the patches, about 1 in every 57 images is positive for crosshatching, while the others are negative. This can result in the model essentially becoming a print("false") model, as that would still typically result in low loss. The first mitigation step is done in the dataloader, where we use a weighted random sampler to sample the data with replacement so that the positive class is more likely to be chosen. This helps reduce the imbalance and ensures the model sees and trains on more positive cases.

The overall goal of this model is to confidently discriminate definite yes’s and no’s regarding the presence of crosshatching in the data. Ideally, we want to reduce the number of images that a person needs to examine. The general structure of this model is a flexible CNN with a loss function tuned for the model’s goals, including an objective function for Optuna hyperparameter tuning. The CNN itself had the kernel sizes and number of layers optimized, with each layer consisting of a convolution, a ReLU, and a 2D max pooling operation. Another method used to address the skewed data was to use BCE with logistic loss and assign the positive class a weight equal to the inverse of its proportion in the data. Additionally, to maximize the number of positive cases the model sees, all labeled data was split into only training and validation groups (no test group). Optuna was run using this loss to find local minima of the loss function, and ROC AUC was used as a measure of how well the model could reduce the proportion of “maybes” in its labels. The optimized model was found to be a CNN with 5 layers, a kernel size of 5, a dropout rate of about 0.4, a learning rate of about 1.06e-4, a weight decay of about 2.3e-4, and 16 base channels.

## Results:

After the optimized model was saved, the CNN’s hyperparameters and parameters were set to those of the optimized version. The CNN was run on the validation loader, and various evaluation metrics were calculated. A ROC curve was plotted and is shown below, demonstrating great discriminatory power and probabilistic interpretation. The AUC ROC of this model is about 0.95, meaning it can distinguish between positive and negative cases 95% of the time.

<img width="461" height="372" alt="Screenshot 2025-08-29 at 5 59 49 PM" src="https://github.com/user-attachments/assets/449947ac-67ef-4382-b80c-79ae3ab06a0c" />

The goal of this model is to divide the data into definite yes’s, no’s, and a “maybe” group. Thus, we need lower and upper thresholds on the probabilities to divide them (previously, anything higher than 0.5 was a yes, and lower was a no). To find these thresholds, a histogram number line was created to examine the probability distributions of yes’s and no’s.

<img width="1240" height="485" alt="Screenshot 2025-08-29 at 6 00 27 PM" src="https://github.com/user-attachments/assets/709849ab-865f-415a-a870-2600814c0548" />

As shown in the graph, the selection that gives full confidence in the no’s and yes’s while still reducing the number of maybe’s in the “maybe” category ranges from 0.2 to 0.8. Implementing this and testing on the validation loader results in about 26% of the images falling into the maybe category. Additionally, since some patches never have crosshatching, the simple tweak of assigning all of those “never crosshatching” patches to the “no” category (optional, and not recommended in some cases) reduces the maybe category to about 15%.
This is somewhat successful in reducing the number of images in the maybe category, but it still presents a challenge, as this is after increasing the number of images by a factor of 16. While this may seem like an increase in the work required to search for crosshatching, based on my experience classifying crosshatching data for this project, I can say that zooming in to look for crosshatching takes considerable time. It may be that, since each image is already zoomed in and is only 1/16th of the area to search, there is an overall time reduction, but it is difficult to draw any definitive conclusions on that without testing.





