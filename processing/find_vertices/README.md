# Regression to find vertices

Regression can be used to determine the exact **intended size and angle** of each drawn shape.

The vertices are the oranges points superimposed on the ellipses in the examples below.   

![examples](readme_images/vertices_ell.png)


For each shape, the vertices of the shape are expressed as x,y coordinates.  
Vertices are the extremities that define the shape.


## Finding the vertices
- The labeled vertices are the oranges points superimposed on the ellipses in the examples below. 
- The predicted vertices are the green points superimposed on the ellipses in the examples below.

![examples](readme_images/predictions_ell.png)

Note: The accuracy of the predictions can be improved by training with more epochs.
- Once the model was trained, I generated a TensorFlow Lite model that I then use in [Mix on Pix](https://apps.apple.com/us/app/mix-on-pix-text-on-photos/id633281586).

---

## Notebooks
- a [Notebook](notebooks/a_etl_ellipse.ipynb) to read the images and vertices. Then prepare data (ETL).
- a [Notebook](notebooks/b_regression_ellipse.ipynb) to find vertices for Ellipses. 

## Images
Images are (70px x 70 px x 1 gray channel). In the ETL phase, I separated the data in:

| Set | Ellipse |
| :--------------|---------------: |
| Training set |  4828  |
| Validation set | 1446  |
| Test set | 180  |

---
## First vertex problematic
One of the diffuculties with vertices is to determine **what will be considered the first point (or vertex)**.

### Solution
The solution was to set an anchor point.  
From the way the images were generated, I knew that the shape would be centered.  
The anchor is an arbitrary location from which we can draw a line to the center of the image.

From that line, we navigation clockwise until we reach a first point. This point will be considered the first vertex.  
Then we continue clockwise for all vertices.  

![first_point_anchor](readme_images/first_point_anchor.PNG)

For the Ellipses, I tried various locations for the **anchor**. The location **(0.0, 0.65)** seemed to give the best result.
  
### The problem with that solution
When a vertex is very close to the line betwwen the anchor and the center, the training can get confused.  

I am illustrating below what I mean.


![first_point_issue](readme_images/first_point_issue.PNG)

In a clockwise navigation:
- if a point is right after the anchor line, it will be set like Point 1 in Turquoise
- if a point is right before the anchor line, it will be set like Point 4 in Yellow as the Point 1 will have been found about 90 degres of the anchor line.

The net result is that over a lot of this kind of samples, the prediction will end up being an average betwwen the Point 1 in Turquoise and the Point 1 in Yellow, resulting in a Point P1 in Magenta, which is bad.

### TODO:
- Indicates possibles solutions

---
by Francois Robert 

