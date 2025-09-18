beam_angle = 15 #sensor beam field of view (FoV). In degrees
beam_max_range = 2 #max range allowed for the beam. In meters
sensors = 3 #number of transmiters/receivers
propagation_speed = 340 #wave propagation speeed over air. In m/s
initial_amplitude = 1 #initial amplitude of the wave
wave_frequency = 40e3 #Wave frequency, in Hertz
sampling_rate = 1/6800 #Sampling rate in seconds

main_sensor_coordinates = [0,0] #Origin Coordinates of the main sensor [X,Y}


import shapely
from itertools import product
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns

from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import MultiPolygon
from shapely.geometry import LineString
from shapely import affinity

plt.rcParams["figure.figsize"] = (6,6)


#function that creates a equaly space grid for a given geometry
def make_grid(polygon, edge_size):
    """
    polygon : shapely.geometry
    edge_size : length of the grid cell
    """
    bounds = polygon.bounds
    x_coords = np.arange(bounds[0] + edge_size/2, bounds[2], edge_size)
    y_coords = np.arange(bounds[1] + edge_size/2, bounds[3], edge_size)
    combinations = np.array(list(product(x_coords, y_coords)))
    squares = gpd.points_from_xy(combinations[:, 0], combinations[:, 1]).buffer(edge_size / 2, cap_style=3)

    return gpd.GeoSeries(squares[squares.intersects(polygon)])

def get_distance(p1, p2):
    """
    Description: Function to return the Eucledian distance between 2 points
    Input: p1: First point with coordinates x1,y1
            p2: Second point with coordinates x2,y2
    Output: Eucleadian distance between p1 and p2
    """
    x2 = p2[0]
    x1 = p1[0]
    y2 = p2[1]
    y1 = p1[1]
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    return dist


def test(real_obj, predicted_obj):
    #Geometry of the main beam (sensor 2)
    #create a triangle with coordinates points
    p1 = Point(0,0)
    p2 = Point(0.26, 1.983)
    p3 = Point(-0.26, 1.983)

    triangle1 = Polygon([p1, p2,p3])

    #create a circle
    diameter = 1
    circle1 = Point(0, 1).buffer(diameter, resolution= 50000)

    #intersect both geometries
    geometry_1 = triangle1.intersection(circle1)

    #calculate the origin coordinates for the beam sensor
    sensor_2_coordinates = np.dstack(p1.coords.xy).tolist()[0][0]

    x, y = geometry_1.exterior.xy

    #Geometry of the secondary left beam (sensor 1)

    #rotate by the main beam by the upper-right point
    rotate_point = (-0.26, 1.983)
    rotation_angle = -5
    geometry_2 = affinity.rotate(geometry_1,
                                 rotation_angle,
                                 rotate_point)
    #calculate the origin coordinates for the beam sensor
    sensor_1_coordinates = np.dstack(affinity.rotate(triangle1,
                                     rotation_angle,
                                     rotate_point).exterior.coords.xy).tolist()[0][0]


    #Geometry of the third right beam (sensor 3)

    #rotate by the main beam by the upper-right point
    rotate_point = (0.26, 1.983)
    rotation_angle = +5
    geometry_3 = affinity.rotate(geometry_1,
                                 rotation_angle,
                                 rotate_point)
    #calculate the origin coordinates for the beam sensor
    sensor_3_coordinates = np.dstack(affinity.rotate(triangle1,
                                     rotation_angle,
                                     rotate_point).exterior.coords.xy).tolist()[0][0]


    #Unify all the geometries
    int_1 = geometry_1.intersection(geometry_2)
    int_2 = geometry_1.intersection(geometry_3)
    int_3 = int_1.union(int_2)

    
    #divide the final geometry into equally size quadrants
    quadrant_size = 0.06
    grid_geometry = make_grid(int_3,
                              quadrant_size)

    #plot the grid 
    #grid_geometry.boundary.plot(label = 'Quadrant grid')

    #plot the composed beam area
    x, y = int_3.exterior.xy

    #plot the composed beam area
    x, y = int_3.exterior.xy
    plt.plot(x, y, label = 'Beam Area', color= 'black')

    #plot the objects
    for object_coordinates in real_obj:
        coords = np.dstack(grid_geometry[object_coordinates].centroid.coords.xy).tolist()[0][0]
        temp_obj = Point(coords)
        x,y = temp_obj.buffer(0.03).exterior.xy
        plt.plot(x,y, color = 'red', label = 'Real object')

    for object_coordinates in predicted_obj:
        coords = np.dstack(grid_geometry[object_coordinates].centroid.coords.xy).tolist()[0][0]
        temp_obj = Point(coords)
        x,y = temp_obj.buffer(0.021).exterior.xy
        plt.plot(x,y, color = 'blue', label = 'Predicted object')

    plt.xlim([-1,1])
    plt.ylim([0, 2])
    plt.legend()
    plt.title('Predicted objects')
    plt.show()
    
    return