# Shape Dataset

Contains
- a Dataset of shapes for Machine Learning Classification.
- a Notebook to read the images and preppare data
- a Notebook to do shape classification

I have created this set of data for my app **[Mix on Pix](https://apps.apple.com/us/app/mix-on-pix-text-on-photos/id633281586)**.

The Dataset is in the directory: classify/data

### Images 
Images exists in 4 shapes
- Ellipse
- Rectangle
- Triangle
- Other

It contains images (70px x 70 px x 1 gray channel) separated in:
- Training set. 21393 images. Other: 5316, Ellipse: 5025, Rectangle: 5740, Triangle: 5312
- Validation Set. 3983 images. Other: 1130, Ellipse: 1069, Rectangle: 860, Triangle: 924
- Test set. 1923 images. Other: 841, Ellipse: 360, Rectangle: 359, Triangle: 363

Example of images from the Training set:
![examples](images/train_images.png)

### Notes
- When training for [Mix on Pix](https://apps.apple.com/us/app/mix-on-pix-text-on-photos/id633281586) using a GPU over 300 epochs, I get a validation accuracu around 0.9980


### TODO
General
- Make a smaller dataset?
- Describe how the data was generated. Tools and People.
- Create nbviewer links for the notebooks.
- Describe the Obvious vs Interesting. Easy vs casesd where a human can hardly decide between 2 shapes. Also Rotation.
- Describe the benefit of *other* shape.
- Improve this Readme.

Preparation
- Improve ETL documentation

Classification
- Rename variables like X_train
- Indicate CPU (150s) vs GPU (6s) per epoch, so 25 times faster...
- Explain main use of Augmentation and how this augmentation helps overfit
- Explain Data is all generated from the same tool -> some characteristics are constant
- Explain lr reducing strategy
- Explain conclusion from Confusion matrix
- Final Conclusion

Later
- Show model to calculate Vertices

---
by Francois Robert 

