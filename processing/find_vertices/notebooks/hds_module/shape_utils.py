import numpy as np
import imageio
import os
from shapely.geometry import Polygon

from scipy import ndimage
from skimage.transform import resize
from tensorflow.keras.utils import Sequence

from hds_module.utils import get_file_paths, path_leaf

# Normalize image by pixel depth by making it white on black instead of black on white
def normalize_image(image_file, pixel_depth):
    try:
        array = imageio.imread(image_file)
    except ValueError:
        raise

    return 1.0 - (array.astype(float))/pixel_depth  # (1 - x) will make it white on black

# Restore original image from normalized image
def unnormalize_image(image, pixel_depth):
    return (pixel_depth*image).astype(np.uint8)

def is_correct_shape_to_process(shapeDir, correct_shape):
    shape = os.path.basename(shapeDir)
    if shape == correct_shape:
        return True
    return False

def replace_last(source_string, replace_what, replace_with):
    head, _sep, tail = source_string.rpartition(replace_what)
    return head + replace_with + tail

def find_nearest_index(vertices, x_from, y_from, vertice_count):
    """
    Find the index of the vertex that is the closest to specified 
    coordinate (x_from, y_from).
    """

    min_distance_squared  = 1000000
    nearest_index = 0

    x_from *= 1000
    y_from *= 1000

    for index, vertice in enumerate(vertices):
        dx = (vertice[0] * 1000) - x_from
        dy = (vertice[1] * 1000) - y_from
        square_dist = dx*dx + dy*dy
        
        if square_dist < min_distance_squared and index < vertice_count:
            nearest_index = index
            min_distance_squared = square_dist

    return nearest_index


def select_first_vertice_index(vertices, vertice_count, x_pos, y_pos):
    """
    The first vertex will be the vertex that is closest to the anchor point
    specified as (x_pos, y_pos)
    """
    nearest_index = find_nearest_index(vertices, x_pos, y_pos, vertice_count)

    return nearest_index

def sort_vertices_clockwize(vertices, first_vertice_index, vertice_count):
    """
    Sort the vertices by navigating clockwise starting with the first vertex
    """
    vertices_sorted = np.zeros(vertices.shape)

    first_vertice_angle = 0
    smaller_vertices = []   # contains an aray of tuple (index, angle) where the angle is smaller to the 1st vertice
    bigger_vertices  = []   # contains an aray of tuple (index, angle) where the angle is bigger  to the 1st vertice
    for index, vertice in enumerate(vertices):
        if index < vertice_count:
            vertice_angle = np.arctan2(0.5 - vertice[1], vertice[0] - 0.5) * 180 / np.pi
            if vertice_angle < 0:
                vertice_angle = 360 + vertice_angle
            if index == first_vertice_index:
                first_vertice_angle = vertice_angle
                break

    for index, vertice in enumerate(vertices):
        if index < vertice_count:
            vertice_angle = np.arctan2(0.5 - vertice[1], vertice[0] - 0.5) * 180 / np.pi
            if vertice_angle < 0:
                vertice_angle = 360 + vertice_angle
            if index != first_vertice_index:
                if vertice_angle < first_vertice_angle:
                    smaller_vertices.append((index, vertice_angle))
                else:
                    bigger_vertices.append((index, vertice_angle))

    # Ordered (clockwise) vertices that we need will be composed of:
    # 1. The first vertice
    # 2. The smaller vertice from the biggest angle to the smallest angle (0)
    # 2. The bigger  vertice from the biggest angle to the smallest angle (first_vertice_angle)
    vertices_ordered = [vertices[first_vertice_index]]
    smaller_vertices.sort(key=lambda x: x[1], reverse=True)
    bigger_vertices.sort( key=lambda x: x[1], reverse=True)
    for index, vertice_angle in smaller_vertices:
        vertices_ordered.append(vertices[index])
    for index, vertice_angle in bigger_vertices:
        vertices_ordered.append(vertices[index])

    for i in range(0, vertice_count):
        vertices_sorted[i] = vertices_ordered[i]
    return vertices_sorted

def load_images_for_shape(root, pixel_depth, user_images,
                          user_images_labels, user_images_paths, 
                          min_nimages=1, 
                          vertice_count=4, 
                          x_pos=0.2, y_pos=1.0,
                          verbose=False):
    """
    Load images and vertices for a specific user and shape.
    """

    if verbose:
        print("root for load_images_for_shape: ", root)

    image_files = get_file_paths(root)
    image_index = 0

    for image_file in image_files:
        try:
            if path_leaf(image_file).startswith('.'):  # skip files like .DSStore
                continue

            # Make sure that the corresponding vertice file exists
            vertice_file = replace_last(image_file, "/images/", "/vertices/")
            vertice_file = replace_last(vertice_file, ".png", ".csv")

            if os.path.exists(vertice_file) == False:
                raise FileNotFoundError(vertice_file)

            # Load Vertices file as points
            vertices = np.loadtxt(vertice_file, delimiter=",") #, max_rows=3)

            # Re-order the vertices
            first_vertice_index = select_first_vertice_index(vertices, vertice_count=vertice_count, x_pos=x_pos, y_pos=y_pos)
            vertices_sorted     = sort_vertices_clockwize(vertices, first_vertice_index=first_vertice_index, vertice_count=vertice_count)

            vertices = vertices_sorted.ravel()
            vertices = vertices.reshape(-1)
            vertices = vertices[:vertice_count*2] # *2 because x and y are separate

            image_data_all_channels = normalize_image(image_file, pixel_depth)
            image_data = image_data_all_channels[:, :, 0]

            user_images.append(image_data)
            user_images_labels.append(vertices)

            image_index += 1
        except Exception as error:
            print(error)
            print('Skipping because of not being able to read: ', image_file)

    if image_index < min_nimages:
        raise Exception('Fewer images than expected: %d < %d' % (image_index, min_nimages))

# Coordinates translations
def cartesian_to_polar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def polar_to_cartesian(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

# Calculate Accuracy
def calculate_IOU(label, pred, nb_vertices=4):
    
    y_polygon   = Polygon(label.reshape(nb_vertices, 2))
    pred_polygon= Polygon(pred.reshape(nb_vertices, 2))

    I = y_polygon.intersection(pred_polygon).area
    U = y_polygon.union(pred_polygon).area
    IOU = I / U
    return IOU  

def calculate_Dice(label, pred, nb_vertices=4):
    
    y_polygon   = Polygon(label.reshape(nb_vertices, 2))
    pred_polygon= Polygon(pred.reshape(nb_vertices, 2))

    I = y_polygon.intersection(pred_polygon).area
    U = y_polygon.union(pred_polygon).area
    dice = 2 * I / (y_polygon.area + pred_polygon.area) 
    return dice 

def calculate_Dice_for_set(Y, Y_pred, nb_vertices):

    nb_samples = Y.shape[0]
    dice_sum = 0.0
    valid_shapes_count = 0
    for i in range(nb_samples):
        try:
            dice = calculate_Dice(Y[i], Y_pred[i], nb_vertices=nb_vertices)
            dice_sum += dice
            valid_shapes_count += 1
        except Exception as e:
            print(f"Cannot compute Dice for shape: {i}.")    

    return dice_sum / valid_shapes_count

class HdsDataGenerator(Sequence):
    """
    Generates data for Keras while modifying the labels
    as the images are flipped or rotated.
    """

    def __init__(self, list_IDs, x_set, y_set, batch_size=32, dim=(70, 70), n_channels=1, n_vertices=4, x_pos=0.0, y_pos=0.65, shuffle=True):
        """
        Initialization
        """
        self.dim = dim
        self.im_size = dim[0]
        self.batch_size = batch_size
        self.x_set = x_set
        self.y_set = y_set
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_vertices = n_vertices
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def rotate_vertices(self, label, angle, padding_is_used, padding):  
        """
        Rotate vertices counter-clockwize in degres
        """
        nb_label = len(label)
        label_new = np.array(label, copy=True)

        # rotate each point by angle
        vertices = label_new.reshape((-1, 2))

        for index, vertice in enumerate(vertices):
            x = vertice[0]
            y = vertice[1]

            dx = x - 0.5
            dy = y - 0.5
            current_angle  = np.arctan2(-dy, dx) * 180 / np.pi # Note Y is reversed
            current_length = np.sqrt(dx*dx + dy*dy)

            new_angle = current_angle + angle
            new_rad_angle = new_angle * np.pi / 180
            new_length = current_length
            if padding_is_used:
                new_length  *= (1 - 2*padding/self.im_size)

            new_dx, new_dy = polar_to_cartesian(new_length, new_rad_angle)

            new_x = 0.5 + new_dx
            new_y = 0.5 - new_dy  # Note Y is de-reversed
            label_new[2*index + 0] = new_x
            label_new[2*index + 1] = new_y

        return label_new

    def shift_vertices(self, label, offset_h_px, offset_v_px):

        label_new = np.array(label, copy=True)
        vertices = label_new.reshape((-1, 2))

        for index, vertice in enumerate(vertices):
            x = vertice[0]
            y = vertice[1]

            new_x = x + offset_h_px/self.im_size
            new_y = y + offset_v_px/self.im_size

            label_new[2*index + 0] = new_x
            label_new[2*index + 1] = new_y

        return label_new

    def get_margin_values(self, image_data):
        image_size = self.im_size

        # TODO Perform an analysis on the number of padding pixel.
        # All black on the left, top, right or bottom.
        # For each image Determine the smallest black margin

        im = image_data.reshape((image_size, image_size))

        mat_rows = np.all(im == 0, axis=1)
        mat_cols = np.all(im == 0, axis=0)

        rows = np.argwhere(mat_rows == False)
        cols = np.argwhere(mat_cols == False)
        top_margin = 0
        bottom_margin = 0
        left_margin = 0
        right_margin = 0

        if len(rows) > 0 and len(cols) > 0:
            top_margin = rows[0]
            bottom_margin = image_size - 1 - rows[-1]

            left_margin = cols[0]
            right_margin = image_size - 1 - cols[-1]

        return top_margin, right_margin, bottom_margin, left_margin

    def get_margin(self, image_data):
        top_margin, right_margin, bottom_margin, left_margin = self.get_margin_values(image_data)

        margin = [0]

        if top_margin != 0 or right_margin != 0 or bottom_margin != 0 or left_margin != 0:
            margin = min(top_margin + bottom_margin, left_margin + right_margin)

        return margin[0]

    def rotate_image(self, image, angle):
        rotated_image = ndimage.rotate(image, angle, reshape=False)
        cleaned_image = self.cleanup_image(rotated_image)
        return cleaned_image

    def center_image(self, image):
        top_margin, right_margin, bottom_margin, left_margin = self.get_margin_values(image)
        target_h_margin = (left_margin + right_margin) // 2
        target_v_margin = (top_margin + bottom_margin) // 2

        offset_h = target_h_margin - left_margin
        offset_v = target_v_margin - top_margin
        
        cleaned_image = image
        
        if offset_h != 0 or offset_v != 0:
            centered_image = ndimage.shift(image, (offset_v,offset_h))
            cleaned_image  = self.cleanup_image(centered_image)

        return cleaned_image, offset_h, offset_v

    def cleanup_image(self, image):
        img = image.reshape(self.dim)
        img[img > 1] = 1

        img[img < 0.1] = 0

        return img

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_vertices * 2))
        indices_lr = np.random.choice(1000, self.batch_size, replace=False)
        indices_ud = np.random.choice(1000, self.batch_size, replace=False)
        indices_ro = np.random.choice(380, self.batch_size, replace=False)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            image = np.array(self.x_set[ID], copy=True)
            label = np.array(self.y_set[ID], copy=True)

            is_modified = False

            # Perform modification on both X (the image) and y (the vertices)
            if (indices_lr[i] < 450):   # Flip Left - Right with probability 450/1000
                image = np.fliplr(image)
                for m in range(len(label)):
                    if m % 2 == 0:
                        label[m] = 1 - label[m]
                is_modified = True
            
            if (indices_ud[i] < 450):   # Flip Up - Down with probability 450/1000
                image = np.flipud(image)
                nb_label = len(label)
                for m in range(nb_label):
                    if m % 2 == 1:
                        label[m] = 1 - label[m]
                is_modified = True

            # Rotate counter-clockwize
            if indices_ro[i] < 360:     # Rotate with probability 360/380
                angle = indices_ro[i]

                margin_before = self.get_margin(image)

                image = self.rotate_image(image, angle)

                margin_rotated = self.get_margin(image)

                padding = int((margin_before - margin_rotated) / 2)
                padding_is_used = False

                image = image.reshape(self.dim)
                if margin_rotated < 10 and padding > 0:
                    image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant') 
                    image = resize(image, self.dim, anti_aliasing=True)
                    padding_is_used = True
                margin_padded = self.get_margin(image)

                label = self.rotate_vertices(label, angle, padding_is_used, padding)

                image, offset_h_px, offset_v_px = self.center_image(image)
                if offset_h_px != 0 or offset_v_px != 0:
                    label = self.shift_vertices(label, offset_h_px, offset_v_px)

                image = image.reshape((self.im_size, self.im_size, 1))

                is_modified = True

            if is_modified:
                # re-order the vertices in the labels
                vertices = label.reshape((self.n_vertices, 2))
                first_vertice_index = select_first_vertice_index(vertices, vertice_count=self.n_vertices, x_pos=self.x_pos, y_pos=self.y_pos)
                vertices_sorted = sort_vertices_clockwize(vertices, first_vertice_index=first_vertice_index,
                                                          vertice_count=self.n_vertices)
                label = vertices_sorted.reshape((self.n_vertices * 2,))

            X[i,] = image
            y[i] = label

        return X, y
