import numpy as np
import imageio
import os
import ntpath
from six.moves import cPickle as pickle

### Functions for getting array of directory paths and array of file paths
def get_dir_paths(root):
    return [os.path.join(root, n) for n in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, n))]

def get_file_paths(root):
    return [os.path.join(root, n) for n in sorted(os.listdir(root)) if os.path.isfile(os.path.join(root, n))]

def path_leaf(path):  
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

## Function for saving an object to a pickle file
def save_to_pickle(pickle_file, object, force=False):
    if os.path.exists(pickle_file) and not force:
        print('%s already present, skipping pickling' % pickle_file)
    else:
        try:
            f = open(pickle_file, 'wb')
            pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save object to', pickle_file, ':', e)
            raise

# Normalize image by pixel depth by making it white on black instead of black on white
def normalize_image(image_file, pixel_depth):
    try:
        array = imageio.imread(image_file)
    except ValueError:
        raise

    return 1.0 - (array.astype(float))/pixel_depth  # (1 - x) will make it white on black

# Retrieve original image from normalized image
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

def find_lefmost_index_in_subrect(vertices, x_from, y_from, x_to_excluded,y_to_excluded, vertice_count):
    count = 0
    min_x = 1
    leftmost_index = 0
    very_close_range = 0.03

    for index, vertice in enumerate(vertices):
        if vertice[0] >= x_from and vertice[0] < x_to_excluded and \
        vertice[1] >= y_from and vertice[1] < y_to_excluded and \
        index < vertice_count:
            count += 1
            if vertice[0] < min_x:
                min_x = vertice[0]
                leftmost_index = index

    # If another vertices is very close to the leftmost one, pick the lowest one in the image (higher y)
    for index, vertice in enumerate(vertices):
        if index != leftmost_index and abs(vertice[0] - min_x) <= very_close_range and vertice[1] > y_of_min_x:
            min_x = vertice[0]
            y_of_min_x = vertice[1]
            leftmost_index = index

    return (count, leftmost_index)

def find_nearest_index(vertices, x_from, y_from, vertice_count):
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

def select_first_vertice_index(vertices, vertice_count):
    count_bottom_left, leftmost_index_bottom_left = find_lefmost_index_in_subrect(vertices, 0, 0.5, 0.5, 1.0, vertice_count)
    if count_bottom_left == 1:
        return leftmost_index_bottom_left

    count_top_left, leftmost_index_top_left = find_lefmost_index_in_subrect(vertices, 0, 0, 0.5, 0.5, vertice_count)
    if count_top_left == 1:
        return leftmost_index_top_left

    count, leftmost_index = find_lefmost_index_in_subrect(vertices, 0, 0, 1.0, 1.0, vertice_count)
    if count < 1:
        raise ValueError("No Vertices found")
    return leftmost_index

def select_first_vertice_index2(vertices, vertice_count, x_pos, y_pos):
    nearest_index = find_nearest_index(vertices, x_pos, y_pos, vertice_count)

    return nearest_index

def sort_vertices_clockwize(vertices, first_vertice_index, vertice_count):
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

    # ordered (clockwise) vertices that we need will be composed of:
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

# Load data for a single user.
def load_images_for_shape(root, pixel_depth, user_images,
                          user_images_labels, user_images_paths, 
                          min_nimages=1, 
                          vertice_count=4, 
                          x_pos=0.2, y_pos=1.0,
                          verbose=False):

    if verbose:
        print("root for load_images_for_shape: ", root)

    image_files = get_file_paths(root)
    image_index = 0

    for image_file in image_files:
        try:
            if path_leaf(image_file).startswith('.'):  # skip file like .DSStore
                continue

            # Make sure that the corresponding vertice file exists
            vertice_file = replace_last(image_file, "/images/", "/vertices/")
            vertice_file = replace_last(vertice_file, ".png", ".csv")

            if os.path.exists(vertice_file) == False:
                raise FileNotFoundError(vertice_file)

            # Load Vertices file as points
            vertices = np.loadtxt(vertice_file, delimiter=",") #, max_rows=3)

            # Re-order the vertices
            first_vertice_index = select_first_vertice_index2(vertices, vertice_count=vertice_count, x_pos=x_pos, y_pos=y_pos)
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

    # if verbose:
    #     print('Full dataset tensor: ', dataset.shape)
    #     print('Mean: ', np.mean(dataset))
    #     print('Standard deviation: ', np.std(dataset))