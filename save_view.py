from raytrace_mountains import *

# This script can be run on a server without windowing capability.
# It uses matplotlib to create and save a png, but doesn't attempt to display it.
#
# https://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
import matplotlib
matplotlib.use('Agg')

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

if __name__ == "__main__":

  #eye_location = Location(122.9142, 37.8817, 951) # The peak of Mt Diablo
  eye_location = Location(123.262209, 37.919241, 371) # The rotary peace grove lookout in tilden park
  mode = 'debugging' # 'run', 'debugging'

  # Render the view and map data in numpy arrays
  renderer = RenderView(eye_location, debugging = mode == 'debugging')

  if mode == 'run':
    (view, map, view_location) = renderer.loadOrConstructAndSaveView()
  elif mode == 'debugging':
    (view, map, view_location) = renderer.loadOrConstructAndSaveView(width_resolution = 36, height_resolution = 10)

  ## Save png of view
  figure = plt.figure(1, figsize = (12, 4))
  plt.subplots_adjust(left = 0.04, right = 1.0)
  view_subplot = figure.add_subplot(1, 1, 1)
  plt.imshow(view, cmap = cm.spectral, alpha = 1.0, origin = 'lower')
  plt.colorbar()
  plt.grid(False)

  image_path = renderer.pathForEye() + '/view.png'
  figure.savefig(image_path, bbox_inches = 'tight')
  print 'Figure saved as ' + image_path
