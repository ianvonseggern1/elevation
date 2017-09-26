from raytrace_mountains import *

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches


if __name__ == "__main__":

  (min_long, max_long, min_lat, max_lat) = (122.0, 124.0, 37.0, 39.0)
  elevations = retriveSrtm(min_long, max_long, min_lat, max_lat)
  #eye_location = Location(122.83, 37.67, 900.0) # A location a bit south of Mt Diablo
  #eye_location = Location(122.9142, 37.8817, 951) # The peak of Mt Diablo
  #eye_location = Location(123.262209, 37.919241, 370) # The rotary peace grove lookout in tilden park
  eye_location = Location(123.262209, 37.919241, 371) # Debugging only
  mode = 'save-debugging' # 'load', 'save', 'save-debugging'

  # Render the view and map data in numpy arrays
  renderer = RenderView(np.copy(elevations), eye_location,
                        min_long, max_long, min_lat, max_lat,
                        approximate_earth_curvature = mode == 'save' # don't do this for debugging or loading
                        )

  if mode == 'save':
    (view, map, view_location) = renderer.save360View()
  elif mode == 'save-debugging':
    (view, map, view_location) = renderer.save360View(width_resolution = 36, height_resolution = 10)
  elif mode == 'load':
    (view, map, view_location) = renderer.loadView()

  (eye_x_index, eye_y_index) = renderer.elevation_tree.indiciesForLocation(eye_location.long, eye_location.lat)

  # Plot everything
  plt.ion()
  figure = plt.figure(1, figsize = (12, 4))

  ## Show elevation heatmap
  heatmap_subplot = figure.add_subplot(1, 3, 1)
  plt.imshow(elevations, cmap = cm.spectral, alpha = 1.0)
  heatmap_subplot.set_title('Elevation')
  plt.grid(False)
  heatmap_subplot.scatter(eye_x_index, eye_y_index, c = 'w')

  ## Show shaded relief map
  relief_subplot = figure.add_subplot(1, 3, 2)
  shaded = shadedReliefMap(elevations)
  relief_subplot.set_title('Shaded Relief Map')
  plt.imshow(shaded, cmap = 'Greys')
  plt.grid(False)
  relief_subplot.scatter(eye_x_index, eye_y_index)

  ## Show map
  map_subplot = figure.add_subplot(1, 3, 3)
  plt.imshow(map, cmap = cm.spectral, alpha = 1.0)
  map_subplot.set_title('View')
  plt.grid(False)
  map_subplot.scatter(eye_x_index, eye_y_index, c = 'w')

  ## Show view
  figure = plt.figure(2, figsize = (12, 4))
  plt.subplots_adjust(left = 0.04, right = 1.0)
  view_subplot = figure.add_subplot(1, 1, 1)
  plt.imshow(view, cmap = cm.spectral, alpha = 1.0, origin = 'lower')
  plt.colorbar()
  plt.grid(False)

  # nonlocal hack - https://stackoverflow.com/questions/2609518/python-nested-function-scopes
  scatterplot_colors = [['b', 'g', 'r', 'c', 'm', 'y']]

  def onclick(event):
    try:
      column_index = int(round(event.xdata))
      row_index = int(round(event.ydata))
      location = view_location[row_index][column_index]
      (click_x_index, click_y_index) = renderer.elevation_tree.indiciesForLocation(location[0], location[1])
    except:
      return

    print(location)
    color = scatterplot_colors[0][0]
    relief_subplot.scatter(click_x_index, click_y_index, c = color)
    heatmap_subplot.scatter(click_x_index, click_y_index, c = color)
    map_subplot.scatter(click_x_index, click_y_index, c = color)
    view_subplot.scatter(column_index, row_index, c = color)

    # Rotate colors
    scatterplot_colors[0] = scatterplot_colors[0][1:]
    scatterplot_colors[0].append(color)
    plt.show(block = True)

  figure.canvas.mpl_connect('button_release_event', onclick)

  plt.show(block = True)
