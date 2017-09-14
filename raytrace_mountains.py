from __future__ import division
from collections import namedtuple
from geopy.distance import vincenty, great_circle
import math
import urllib
import zipfile
import struct
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#from cartopy.io.img_tiles import GoogleTiles # todo add a google map image, currently crashing


# Motivation: To be able to programmatically visualize the view from the top of a mountain or ridgeline.
#
# Usage: Set the bounds, eye location, and direction in main.
# From the command line run this module. It will take a few minutes, and will produce 4 charts.
# Click any point on the 'View' chart to see that location plotted on the elevation and relief maps.
#
# Approach: The current approach is inspired by the computer graphics technique raytracing. Basically we define
# the location of an 'eye' and we decide on a canvas of actual locations. We then draw a ray from the
# eye to each point and follow it until it intersects with the terrain. We say thats what would be seen
# if a person were to look in that direction from that location.
#
# Definitions: As I built the raytracing I thought in terms of x, y, and z coordinates, but the elevation data
# is based on latitude longitude and height in meters. Unfortunately this has yielded some tricky and messy code.
# If I were to invest more time in this project I would want to simply this. As of now the conventions are that
# latitude increases as you go North, longitude increases as you go West (sorry, this too is silly, but I was thinking
# in terms of North America and started dropping the negative signs on the longtitude). Also worth checking the
# order of arguments to functions, sometimes its lat, long, sometimes long, lat depending on context. Anytime x
# is inappropriately used to refer to location its long and y is lat. When indicies are discussed however, due
# to the way the elevation data is loaded (Here I probably could have just rotated the array, again apologies)
# x increases as you go East and y increases as you go South. The arrays are indexed 
# [row][column] -> [y_index][x_index] yet another potential source of confusion.
#
# Issues/todos: As of now this assumes a flat earth (http://rol.st/2xI43qR), but subtracts a certain amount from
# each point depending on how far it is from the 'eye'. This is an approximation and I haven't done much to test
# this against real world observations.
# The biggest source of error here is all distance <-> lat/long conversions are done at the eye. This means that
# there is actually a slight curve to the rays as they get far from the eye. Hopefully this isn't too major a
# distortion.
# The proper solution to both would be to rethink everything in terms of three dimensional space, probably
# declaring the origin as the center of the earth. This is a major overhaul and would really just entail starting
# over. Hopefully this is still a decent approximation in most cases.

Location = namedtuple('Vector', 'long lat z')
Box = namedtuple('Box', 'layer_index x_index y_index min_long max_long min_lat max_lat height')

## Functions

def retriveSrtm(min_long, max_long, min_lat, max_lat):
  # First we download all squares that we need
  data = None
  for longitude in range(int(min_long), int(math.ceil(max_long))):
    column = None
    for latitude in range(int(min_lat), int(math.ceil(max_lat))):
      square = retriveSingleSrtmSquare('N' + str(latitude) + 'W' + str(longitude))
      if column is None:
        column = square
      else:
        column = np.vstack([square, column])
    if data is None:
      data = column
    else:
      data = np.hstack([column, data])

  # Then we chop off the excess
  data_span_longitude = math.ceil(max_long) - int(min_long)
  data_span_latitude = math.ceil(max_lat) - int(min_lat)
  (rows, columns) = data.shape
  start_col_index = columns * (math.ceil(max_long) - max_long) / data_span_longitude
  end_col_index = columns - columns * (min_long - int(min_long)) / data_span_longitude
  start_row_index = rows * (math.ceil(max_lat) - max_lat) / data_span_latitude  
  end_row_index = rows - rows * (min_lat - int(min_lat)) / data_span_latitude
  return data[int(start_row_index):int(end_row_index), int(start_col_index):int(end_col_index)]

def retriveSingleSrtmSquare(location_string): # location_string is N37W122 for example
  filename = 'https://dds.cr.usgs.gov/srtm/version2_1/SRTM3/North_America/'
  filename += location_string + '.hgt.zip'
  filehandle, _ = urllib.urlretrieve(filename)
  zip_file = zipfile.ZipFile(filehandle, 'r')
  # Source: https://stevendkay.wordpress.com/2009/09/05/beginning-digital-elevation-model-work-with-python/
  one_dimensional_array = struct.unpack(">1442401H", zip_file.read(location_string + '.hgt'))
  elevations = np.zeros((1201, 1201))
  for r in range(0, 1201):
    for c in range(0, 1201):
      value = one_dimensional_array[(1201 * r) + c]
      if (value == 65535 or value < 0 or value > 10000):
        value = 0.0
      elevations[r][c] = float(value)
  return elevations

# Obtain numpy array of values for a shaded relief map
# Source: http://www.geophysique.be/2014/02/25/shaded-relief-map-in-python/
def shadedReliefMap(elevations):
  x, y = np.gradient(elevations)
  slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
  # -x here because of pixel orders in the SRTM tile
  aspect = np.arctan2(-x, y)
 
  altitude = np.pi / 4.
  azimuth = np.pi / 2.
  return (np.sin(altitude) * np.sin(slope) +
          (np.cos(altitude) * np.cos(slope) * 
           np.cos((azimuth - np.pi / 2.0) - aspect)))


def distanceToBox(starting_location, box):
  return vincenty(((box.min_lat + box.max_lat) / 2.0, (box.min_long + box.max_long) / 2.0),
                  (starting_location.lat, starting_location.long)).kilometers

# Returns delta kilometers / delta latitude at that location
def kilometerLatitudeRatio(lat, long):
  return vincenty((lat, long), (lat + 0.01, long)).kilometers * 100.0

# Returns delta kilometers / delta longitude at that location
def kilometerLongitudeRatio(lat, long):
  return vincenty((lat, long), (lat, long + 0.01)).kilometers * 100.0


# A class to obtain information about the elevation data. This class contains the optional ability
# to construct a bounding box tree to speed up processing. Note this optimization is currently not
# used or needed in the below raytrace function.
class Elevations(object):
  
  def __init__(self,
               elevations, 
               min_long, max_long, min_lat, max_lat, 
               eye_location_to_approximate_curvature_of_earth = None,
               construct_bounding_volume_heirachy = False):
    self.min_long = min_long
    self.min_lat = min_lat
    self.elevations = elevations
    self.long_delta = (max_long - min_long) / elevations.shape[1]
    self.lat_delta = (max_lat - min_lat) / elevations.shape[0]

    if eye_location_to_approximate_curvature_of_earth is not None:
      self.discountElevationsForEarthsCurvature(eye_location_to_approximate_curvature_of_earth)

    self.sizes = [(self.long_delta, self.lat_delta)]
    self.tree = [self.elevations]

    if construct_bounding_volume_heirachy:
      current_layer = heights
      next_layer = self.constructNextLayer(current_layer)
      while next_layer is not None:
        self.tree.append(next_layer)
        self.sizes.append((self.sizes[-1][0] * 2, self.sizes[-1][1] * 2))
        current_layer = next_layer
        next_layer = self.constructNextLayerOfBoundingTree(current_layer)

  # Used to create the higher levels in the bounding volume tree. In each higher layer
  # a cell represents 4 (2 x 2) from the layer below. To create the bounding
  # box we max the height of each cell in the layer below.
  #
  # If there is a single row or column left in the lower layer it will be
  # skipped in constructing the layer above
  def constructNextLayerOfBoundingTree(self, layer):
    if len(layer.shape) != 2:
      raise Exception(layer.shape, "layer must be a 2d numpy array")
    if layer.shape[0] < 2 or layer.shape[1] < 2:
      return None

    new_layer = []
    row_index = 0
    while row_index < len(layer) - 1:
      new_row = []
      col_index = 0
      while col_index < len(layer[0]) - 1:
        new_row.append(max(layer[row_index][col_index],
                           layer[row_index][col_index + 1],
                           layer[row_index + 1][col_index],
                           layer[row_index + 1][col_index + 1]))
        col_index += 2
      new_layer.append(new_row)
      row_index += 2
    return np.array(new_layer)

  # This attempts to deal with the fact that the earth falls away as you move forward toward the horizon
  # For all points it treats the eye as ground zero and gives an array of values
  # that are subtracted from the elevation data to compensate for the curvature. These are in meters.
  # This approximates the earth as a perfect sphere. Probably an ok assumption for this.
  def discountElevationsForEarthsCurvature(self, eye, print_progress = True):
    earth_radius = 6371.0
    discounts = np.zeros(self.elevations.shape)
    (rows, columns) = self.elevations.shape

    progress_increment = int(rows / 10)

    for row in range(rows):
      if print_progress and row % progress_increment == 0:
        print 'Discount for curvature: ' + str(100.0 * row / rows) + "%\n"

      for column in range(columns):
        long = self.min_long + (columns - column) * self.long_delta
        lat = self.min_lat + (rows - row) * self.lat_delta
        horizontal_distance = great_circle((eye.lat, eye.long), (lat, long)).kilometers
        discount_km = math.sqrt(earth_radius ** 2 + horizontal_distance ** 2) - earth_radius
        discounts[row][column] = discount_km * 1000.0
    self.elevations -= discounts

  def siblingBoxInDirection(self, box, direction):
    # Direction is based on lat and long which are opposite the indicies of the array
    x_index = box.x_index - direction[0]
    y_index = box.y_index - direction[1]

    layer_shape = self.tree[box.layer_index].shape
    if (x_index < 0 or x_index >= layer_shape[1] or 
        y_index < 0 or y_index >= layer_shape[0]):
      return None

    return self.boxForIndicies(box.layer_index, x_index, y_index)

  # Indicies of a point in the most fine grained (the first) layer of elevation boxes
  def indiciesForLocation(self, longitude, latitude):
    x_index = self.elevations.shape[1] - 1 - int((longitude - self.min_long) / self.long_delta)
    y_index = self.elevations.shape[0] - 1 - int((latitude - self.min_lat) / self.lat_delta)
    return (x_index, y_index)

  def boxForLocation(self, longitude, latitude):
    (x_index, y_index) = self.indiciesForLocation(longitude, latitude)
    return self.boxForIndicies(0, x_index, y_index)

  # Returns a box composed of a min and max lat and long as well as a height
  def boxForIndicies(self, layer_index, x_index, y_index):
    return Box(layer_index, x_index, y_index,
               self.min_long + (self.tree[layer_index].shape[1] - x_index - 1) * self.sizes[layer_index][0],
               self.min_long + (self.tree[layer_index].shape[1] - x_index) * self.sizes[layer_index][0],
               self.min_lat + (self.tree[layer_index].shape[0] - y_index - 1) * self.sizes[layer_index][1],
               self.min_lat + (self.tree[layer_index].shape[0] - y_index) * self.sizes[layer_index][1],
               self.tree[layer_index][y_index][x_index])


# This class follows a single ray from a starting location along a delta until it finds a point that the ray
# intersects with.
class Raytrace(object):
  MIN_DISTANCE = 2.5

  def __init__(self, ray_start, ray_delta, terrain):
    # first two arguments are 3 item tuples, next is an instance of Elevations
    if (len(ray_start) != 3):
      raise Exception(ray_start, "ray_start must be a 3d vector of the long, lat, and height")
    if (len(ray_delta) != 3):
      raise Exception(ray_delta, "ray_delta must be a 3d vector of the long, lat, and height difference as the ray moves forward")

    self.terrain = terrain
    self.ray_start = ray_start
    self.ray_delta = ray_delta

  def intersect(self, box):
    # We don't know if the ray points in positive or negative x (or y)
    # so we find the t values at which the ray crosses into the correct
    # x (or y) range and then find the smaller and larger or those t values
    if self.ray_delta.long == 0.0:
      if self.ray_start.long >= box.min_long and self.ray_start.long <= box.max_long:
        t_at_x_min = 0
        t_at_x_max = float('inf')
      else:
        t_at_x_min = float('inf')
        t_at_x_max = float('inf')
    else:
      t_at_x_min = (box.min_long - self.ray_start.long) / self.ray_delta.long
      t_at_x_max = (box.max_long - self.ray_start.long) / self.ray_delta.long
       
    min_t_for_valid_x = max(min(t_at_x_min, t_at_x_max), 0)
    max_t_for_valid_x = max(t_at_x_min, t_at_x_max)
    if (max_t_for_valid_x < 0):
      return False
    
    if self.ray_delta.lat == 0.0:
      if self.ray_start.lat >= box.min_lat and self.ray_start.lat <= box.max_lat:
        t_at_y_min = 0
        t_at_y_max = float('inf')
      else:
        t_at_y_min = float('inf')
        t_at_y_max = float('inf')
    else: 
      t_at_y_min = (box.min_lat - self.ray_start.lat) / self.ray_delta.lat
      t_at_y_max = (box.max_lat - self.ray_start.lat) / self.ray_delta.lat

    min_t_for_valid_y = max(min(t_at_y_min, t_at_y_max), 0)
    max_t_for_valid_y = max(t_at_y_min, t_at_y_max)
    if (max_t_for_valid_y < 0):
      return False
   
    # check if ray passes through box by checking if there is overlap in
    # the range of t's for which the ray is in the correct x and correct y
    if (min_t_for_valid_y > max_t_for_valid_x or max_t_for_valid_y < min_t_for_valid_x):
      return False

    # given ray passes through box,check if ray goes low enough to touch box
    min_t_in_box = max(min_t_for_valid_y, min_t_for_valid_x)
    max_t_in_box = min(max_t_for_valid_y, max_t_for_valid_x)
    z_at_min_t = self.ray_start.z + min_t_in_box * self.ray_delta.z
    z_at_max_t = self.ray_start.z + max_t_in_box * self.ray_delta.z
    return min(z_at_min_t, z_at_max_t) <= box.height

  # This function assumes that the ray passes through the x, y projection of
  # box and returns the direction of the next box it will pass through
  #
  # Direction is one of North: (0, 1) South: (0, -1) West: (1, 0) East: (-1, 0)
  def directionOfNextSiblingBoxAlongRay(self, box):
    # Deal with the division by zero cases up front
    if self.ray_delta.long == 0:
      return (0, 1) if self.ray_delta.lat > 0 else (0, -1)
    if self.ray_delta.lat == 0:
      return (1, 0) if self.ray_delta.long > 0 else (-1, 0)

    t_at_x_min = (box.min_long - self.ray_start.long) / self.ray_delta.long
    t_at_x_max = (box.max_long - self.ray_start.long) / self.ray_delta.long

    max_t_for_valid_x = max(t_at_x_min, t_at_x_max)

    t_at_y_min = (box.min_lat - self.ray_start.lat) / self.ray_delta.lat
    t_at_y_max = (box.max_lat - self.ray_start.lat) / self.ray_delta.lat

    max_t_for_valid_y = max(t_at_y_min, t_at_y_max)

    # If the smaller max t is on an x intercept then the next box the next
    # box in the x direction (positive or negative based on ray_delta.long)
    if max_t_for_valid_x < max_t_for_valid_y:
      return (1, 0) if self.ray_delta.long > 0 else (-1, 0)
    return (0, 1) if self.ray_delta.lat > 0 else (0, -1)

  def findFirstIntersectingBox(self, starting_box = None):
    if starting_box is not None:
      box = starting_box
    else:
      box = self.terrain.boxForLocation(self.ray_start.long, self.ray_start.lat)

    while (box is not None and 
           not (self.intersect(box) and distanceToBox(self.ray_start, box) > self.MIN_DISTANCE)):
      box = self.terrain.siblingBoxInDirection(box, self.directionOfNextSiblingBoxAlongRay(box))

    return box


class RenderView(object):
  MAX_DISTANCE = 100.0 # Since even on a super clear day its nearly impossible to see past 100 km

  def __init__(self, 
               elevations, eye_location, 
               min_long, max_long, min_lat, max_lat, 
               approximate_earth_curvature = True):
    self.eye_location = eye_location
    eye_location_arg = eye_location if approximate_earth_curvature else None
    self.elevation_tree = Elevations(elevations,
                                     min_long, max_long, min_lat, max_lat,
                                     eye_location_to_approximate_curvature_of_earth = eye_location_arg)

  # The function returns three 2d numpy arrays.
  #
  # The first, the view is what is seen from the location
  # of the 'eye' as the specificed angle with the specificed width and height.
  #
  # The second is the map
  # which is where each of these visible points are as seen from above. The points correspond to the
  # smallest (first) grid in the elevation_tree.
  #
  # The third, the view_locations, are the corresponding locations for each of the pixels in the view
  # they are stored as (average_longitude, average_latitude, height in meters)
  def getViewAndMap(self, 
                    horizontal_eye_angle, # measured as degrees clockwise from due north
                    verticle_eye_angle = 0.0,
                    width_angle = 90.0, 
                    height_angle = 12.0, 
                    width_resolution = 2000, 
                    height_resolution = 1000,
                    print_progress = False):

    view = np.zeros((height_resolution, width_resolution))
    view_locations = np.empty((height_resolution, width_resolution), dtype = object)
    map = np.zeros(self.elevation_tree.elevations.shape)

    horizontal_angle_delta = width_angle / width_resolution
    vertical_angle_delta = height_angle / height_resolution

    progress_increment = int(width_resolution / 10)

    # We define each point on the canvas to lie 1 km from the eye. We calculate the long_delta, lat_delta
    # and z_delta (in meters) based on this
    for column_index in range(width_resolution):
      if print_progress and column_index % progress_increment == 0:
        print str(100.0 * column_index / width_resolution) + "%\n"
        
      horizontal_angle = horizontal_eye_angle - width_angle / 2.0 + column_index * horizontal_angle_delta
      x_delta = math.sin(horizontal_angle * math.pi / 180)
      y_delta = math.cos(horizontal_angle * math.pi / 180)
      long_delta = x_delta / kilometerLongitudeRatio(self.eye_location.lat, self.eye_location.long)
      lat_delta = y_delta / kilometerLatitudeRatio(self.eye_location.lat, self.eye_location.long)

      # We can speed things up be recognizing that for a given x_delta, y_delta
      # a larger z_delta will have to be equal or further than the last one
      starting_box = None
      for row_index in range(height_resolution):
        verticle_angle = verticle_eye_angle - height_angle / 2.0 + row_index * vertical_angle_delta
        z_delta = math.sin(verticle_angle * math.pi / 180) * 1000.0

        raytrace = Raytrace(self.eye_location, Location(long_delta, lat_delta, z_delta), self.elevation_tree)
        box = raytrace.findFirstIntersectingBox(starting_box)
        if box is None:
          break

        distance_km = distanceToBox(eye_location, box)
        if distance_km > self.MAX_DISTANCE:
          break

        view[row_index][column_index] = distance_km
        view_locations[row_index][column_index] = ((box.min_long + box.max_long) / 2.0,
                                                   (box.min_lat + box.max_lat) / 2.0,
                                                   box.height)

        if not box.layer_index == 0:
          raise Exception(box.layer_index, "box.layer_index should be equal to 0 as we should have found the smallest interesecting box")
        map[box.y_index][box.x_index] = distance_km

        starting_box = box

    # Because raytrace treats x as positive to the right and we use long as positive to the left we need to
    # horizontally flip the view
    return (np.flip(view, 1), map, np.flip(view_locations, 1))

# Sample with mt diablo
if __name__ == "__main__":

  (min_long, max_long, min_lat, max_lat) = (122.0, 124.0, 37.0, 39.0)
  elevations = retriveSrtm(min_long, max_long, min_lat, max_lat)
  #eye_location = Location(122.83, 37.67, 900.0) # A location a bit south of Mt Diablo
  eye_location = Location(122.9142, 37.8817, 951) # The peak of Mt Diablo
  
  # Render the view and map data in numpy arrays
  renderer = RenderView(np.copy(elevations), eye_location, 
                        min_long, max_long, min_lat, max_lat,
  #                      approximate_earth_curvature = False # for debugging
                        )
  (view, map, view_location) = renderer.getViewAndMap(
                                       90.0, 
  #                                     width_resolution = 16, height_resolution = 8, # for debugging
                                       print_progress = True)
  (eye_x_index, eye_y_index) = renderer.elevation_tree.indiciesForLocation(eye_location.long, eye_location.lat)

  # Plot everything
  plt.ion()
  figure = plt.figure()

  ## Show elevation heatmap
  heatmap_subplot = figure.add_subplot(2, 2, 1)
  plt.imshow(elevations, cmap = cm.spectral, alpha = 1.0)
  plt.grid(False)
  plt.colorbar(ax = heatmap_subplot)
  heatmap_subplot.scatter(eye_x_index, eye_y_index, c = 'w')
  
  ## Show shaded relief map
  relief_subplot = figure.add_subplot(2, 2, 2)
  shaded = shadedReliefMap(elevations)
  plt.imshow(shaded, cmap = 'Greys')
  plt.grid(False)
  relief_subplot.scatter(eye_x_index, eye_y_index)

  ## Show view
  view_subplot = figure.add_subplot(2, 2, 3)
  plt.imshow(view, cmap = cm.spectral, alpha = 1.0, origin = 'lower')
  view_subplot.set_title('View')
  plt.grid(False)

  ## Show map
  map_subplot = figure.add_subplot(2, 2, 4)
  plt.imshow(map, cmap = cm.spectral, alpha = 1.0)
  map_subplot.set_title('Map')
  plt.grid(False)
  plt.colorbar()
  map_subplot.scatter(eye_x_index, eye_y_index, c = 'w')
  
  def onclick(event):
    try:
      column_index = int(round(event.xdata))
      row_index = int(round(event.ydata))
      location = view_location[row_index][column_index]
      (click_x_index, click_y_index) = renderer.elevation_tree.indiciesForLocation(location[0], location[1])
    except:
      return
    
    print(location)
    relief_subplot.scatter(click_x_index, click_y_index)
    heatmap_subplot.scatter(click_x_index, click_y_index)
    plt.show(block = True)

  figure.canvas.mpl_connect('button_release_event', onclick)

  plt.show(block = True)