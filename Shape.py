import numpy as np
from shapely.geometry import Point,MultiPolygon,LineString
from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union
from math import asin, atan2, cos, degrees, radians, sin
import matplotlib.pyplot as plt
import os
import random
import datetime
import copy
import pandas as pd
import seaborn as sns

class System:
  def __init__(self,steer_angle,max_dist=10,opening_angle=90,location=(0,0),min_dist=0):
    self.min_dist = min_dist
    self.location = location
    self.opening_angle = opening_angle
    self.steer_angle = steer_angle
    self.max_dist = max_dist
    self.optimized = False
    self.shape = None

  def copy(self):
        # Using deepcopy to create a deep copy of the object
      return copy.deepcopy(self)
  def build_shape(self,obstacles=None):
    left_border_angle = radians(self.steer_angle + self.opening_angle/2)
    right_border_angle = radians(self.steer_angle - self.opening_angle/2)

    outer_cicrle = Point(self.location).buffer(self.max_dist)
    inner_cicrle = Point(self.location).buffer(self.min_dist)

    C = Polygon([self.location,
                (self.location[0]+20*self.max_dist*cos(left_border_angle),
                 self.location[0]+20*self.max_dist*sin(left_border_angle)),
                (self.location[1]+20*self.max_dist*cos(right_border_angle),
                 self.location[1]+20*self.max_dist*sin(right_border_angle))])
    self.shape = outer_cicrle.difference(inner_cicrle).intersection(C)
    #if there is only one blockade
    if obstacles is not None:
      self.shape = Obstacle.calc_shadow(obstacles=obstacles,system=self)
   #  if isinstance(obstacles.shape, Polygon):
   #      self.shape = obstacles.calc_shadow(system=self)
   # #if there are many blockades
   #  elif isinstance(obstacles.shape, MultiPolygon):
   #     for obstacle in obstacles.shape.geoms:
   #       uncovered_shape = Obstacle.calc_shadow(system=self)
   #       self.shape = uncovered_shape

  def update_values(self, **kwargs):
   for key, value in kwargs.items():

      if hasattr(self, key):
         setattr(self, key, value)
   self.build_shape(kwargs['obstacles'])



class Area:
   def __init__(self,width_range=(1, 5), height_range=(1, 5), x_range=(-10, 10), y_range=(-10, 10)):
        if len(width_range)==1:
           width = width_range
        else:
           width = random.uniform(*width_range)

        if len(height_range)==1:
           height = height_range 
        else:
          height = random.uniform(*height_range)

        if len(x_range)==1:
           x = x_range 
        else:
          x = random.uniform(*x_range)

        if len(y_range)==1:
           y = y_range 
        else:
          y = random.uniform(*y_range)

        # Create a rectangle using Shapely's Polygon
        self.shape = Polygon([(x, y), (x + width, y), (x + width, y + height), (x, y + height)])

class Obstacle:
   def __init__(self,shape=None):
      if shape is None:
         self.shape = self.generate_random_polygon(5)
      else:
         self.shape = shape
   def generate_random_polygon(self,num_points, sigma=4):

      # Define a bounding box (xmin, ymin, xmax, ymax)
      # xmin, ymin, xmax, ymax = bbox
      
      # Generate random points within the bounding box
      points = [Point(random.gauss(0, sigma), random.gauss(0, sigma)) for _ in range(num_points)]
      
      # Create a convex hull from the random points
      convex_hull = Polygon([point.coords[0] for point in points]).convex_hull
      
      # Optionally, check if the generated polygon is valid
      if convex_hull.is_valid:
         return convex_hull
      else:
         # Retry if the generated polygon is not valid
         return self.generate_random_polygon(num_points, sigma)
   
   @staticmethod
   def check_line_polygon_intersection(obstacle,line):
    # Check if the line intersects the polygon
    if line.intersects(obstacle.shape):
        # Find intersection points
        intersection = line.intersection(obstacle.shape)
        
        # If the intersection is a Point, convert it to a list
        if isinstance(intersection, Point):
            intersection = [intersection]
        elif isinstance(intersection, LineString):
            # If the intersection is a LineString, convert it to a list of points
            intersection = list(intersection.coords)
        
        return intersection
    else:
        return None

   @classmethod
   def find_borders(cls,obstacle,system,precision=1e-2):

    check_num = 3
    start = radians(system.steer_angle - system.opening_angle/2)
    end = radians(system.steer_angle + system.opening_angle/2)
    found = 0
    start_system_intersection = None
    end_system_intersection = None
    start_obstacle_intersection = None
    end_obstacle_intersection = None
    start_intersects = 0
    line = LineString([system.location, (system.location[0] + system.max_dist * 2 * sin(0),
                                                   system.location[1] + system.max_dist * 2 * cos(0))])
    intersection = cls.check_line_polygon_intersection(obstacle,line)
    if intersection is not None:
       start_intersects = 1
   
    #what happens when theta=0 is not None?
    while True:
        thetas = np.linspace(start,end,check_num)

        for i in range(1,len(thetas)-2):
            line = LineString([system.location, (system.location[0] + system.max_dist * 2 * sin(thetas[i]),
                                                   system.location[1] + system.max_dist * 2 * cos(thetas[i]))])
            intersection = cls.check_line_polygon_intersection(obstacle,line)
            if intersection is not None:
                start = thetas[i-1]
                start_system_intersection = line.coords[-1]
                found = 1
                start_obstacle_intersection = intersection[0]
                #if we have found that for some theta there is an intersection,
               #   we go backward to find the start theta
                print(f'start is:{degrees(start)}')
                break

                
        if thetas[1]-thetas[0]<precision:
            if found:
                break
            else:
                return None

        check_num *= 2
    
    check_num = 3
    found = 0
    while True:
        thetas = np.linspace(start,end,check_num)
        for i in range(len(thetas) - 2, 0, -1):
            line = LineString([system.location, (system.location[0] + system.max_dist * 2 * sin(thetas[i]),
                                                   system.location[1] + system.max_dist * 2 * cos(thetas[i]))])
            intersection = cls.check_line_polygon_intersection(obstacle,line)
            if intersection is not None:
                end = thetas[i+1]
                end_system_intersection = line.coords[-1]
                found = 1
                end_obstacle_intersection = intersection[0]
                print(f'end is:{degrees(end)}')
                break

        if thetas[1]-thetas[0]<precision:
            if found:
                start_length = LineString([system.location,start_obstacle_intersection]).length
                end_length = LineString([system.location,end_obstacle_intersection]).length
                bigger_length = max(start_length,end_length)
                steer_angle=np.mod(degrees((end+start)/2),360)
                print(f'steer_angle:{steer_angle}')
                opening_angle = degrees(abs(end-start))
                print(f'opening_angle:{opening_angle}')

                temp_sys = System(steer_angle=steer_angle,max_dist=system.max_dist,
                                  opening_angle=opening_angle,location=system.location,
                                  min_dist = system.min_dist)
                temp_sys.build_shape(obstacles=None)
                slice = temp_sys.shape
                triangle = Polygon([system.location,start_obstacle_intersection,end_obstacle_intersection])
               #  covered = system.shape.difference(slice.union(obstacle.shape))
                covered = system.shape.difference(slice.difference(triangle))

                return covered


        check_num *= 2

   @classmethod
   def polygon_shadow(cls,obstacle,system):
      borders = None
      if system.shape.intersects(obstacle.shape):
         borders = cls.find_borders(obstacle=obstacle,system=system)
      if borders is None:
         return system.shape
      return borders
      # poly = Polygon(borders)
      # rough_shadow = Polygon(borders).union(obstacle.shape).intersection(system.shape)
      # if rough_shadow is None:
      #    return system.shape
      # precise_shadow = rough_shadow.convex_hull
      # blocked_system = system.shape.difference(rough_shadow)
      # new = system.shape.difference(Polygon(borders).union(obstacle.shape))
      # return rough_shadow
   
   @classmethod 
   def calc_shadow(cls,obstacles,system):

      covered_system = system.copy()
      if isinstance(obstacles.shape, Polygon):
         #system cannot be in an obstacle
         if obstacles.shape.contains(Point(system.location)):
            return None
         covered_system.shape = cls.polygon_shadow(obstacles,covered_system)
      elif isinstance(obstacles.shape, MultiPolygon):
         for obstacle in obstacles.shape.geoms:
            #system cannot be in an obstacle
            if obstacle.contains(Point(system.location)):
               return None
            covered_system.shape = cls.polygon_shadow(obstacle,covered_system)
      return covered_system.shape

   
class Enviroment:
   def __init__(self):
      self.areas = []
      self.obstacles = []
      self.systems = []
      self.areas_multipolygon = None
      self.obstacles_multipolygon = None
      # self.blocked_systems = []
   def build_env(self):
      systems_num = 3
      areas_num = 15
      obstacles_num = 0
      systems_locations = np.random.uniform(-10,10,size=(systems_num,2))
      steer_angles = np.random.uniform(0,360,size=(systems_num))
      self.obstacles = [Obstacle() for _ in range(obstacles_num)]
      self.obstacles_multipolygon = Obstacle(unary_union([obstacle.shape for obstacle in self.obstacles]))

      for system_i in range(systems_num):
         self.systems.append(System(location=(systems_locations[system_i,0],systems_locations[system_i,1]),
                             steer_angle=steer_angles[system_i]))
         self.systems[system_i].build_shape(self.obstacles_multipolygon)
         # blocked_system = self.obstacles.calc_shadow(self.systems[system_i])
         # self.blocked_systems.append()

      self.areas = [Area() for _ in range(areas_num)]
      self.areas_multipolygon = unary_union([area.shape for area in self.areas])


      # self.united_area = cascaded_union([area.rectangle for area in self.areas])

   def draw(self,path,optimized='location',mode=''):
      fig, ax = plt.subplots(figsize=(8, 8))
      for poly in self.areas_multipolygon.geoms:
         x, y = poly.exterior.xy
         ax.fill(x, y, hatch='/', color='lightgray', edgecolor='black')
      for system in self.systems:
         if system.shape is None:
            continue
         elif isinstance(system.shape, Polygon):
            ax.plot(*system.shape.exterior.xy)
         elif isinstance(system.shape, MultiPolygon):
            for geom in system.shape.geoms:
               ax.plot(*geom.exterior.xy)
         #draw obstacles
      if isinstance(self.obstacles_multipolygon.shape, Polygon):
         x, y = self.obstacles_multipolygon.shape.exterior.xy
         ax.fill(x, y, hatch='.', color='peachpuff', edgecolor='black')
      elif isinstance(self.obstacles_multipolygon.shape, MultiPolygon):
         for obstacle in self.obstacles_multipolygon.shape.geoms:
            x, y = obstacle.exterior.xy
            ax.fill(x, y, hatch='.', color='peachpuff', edgecolor='black')

      coverage = self.compute_coverage()
      title = mode+' '
      if optimized=='steering':
            title += 'Optimized steering'
      elif optimized == 'location':
            title += 'Optimized steering and location'
      else:
            title += 'Random Rectangles and systems'
      now = datetime.datetime.now() 
      ax.set_title(title+ ' total coverage {:.4f}'.format(coverage))
      ax.set_xlim((-20,20))
      ax.set_ylim((-20,20))
      if path:
         fig.savefig(os.path.join(path,now.strftime("%m%d-%H%M") + ' ' + title +'.png')  ,bbox_inches='tight')
      plt.show()

   def compute_coverage(self):
      total_intersection_area = 0
      for system_i,system in enumerate(self.systems):
         for part in self.areas:
            if system.shape is None:
               continue
            intersection_result = system.shape.intersection(part.shape)
            total_intersection_area += intersection_result.area
      return total_intersection_area

   def optimize(self,optimized='location'):
      #optimizing steering angle for each system
      # areas_multipolygon = unary_union([area.shape for area in self.areas])
      rotation_angles = np.linspace(0,360,100)
      integer_range = range(-10, 11,4)
      locations = [(x, y) for x in integer_range for y in integer_range]

      for _ in range(len(self.systems)):
         chosen_system = 0
         chosen_rotation = 0
         chosen_location = (0,0)
         max_intersection_area = 0
         # looking for system and angle that maximizes the intersection area
         for system_i,system in enumerate(self.systems):
            if self.systems[system_i].optimized == True:
               continue
            for rotation in rotation_angles:
               for location in locations:
                  if optimized=='steering':
                     location = system.location

                  system.update_values(steer_angle=rotation,location=location,obstacles=None)
                  total_intersection_area = 0
                  systems_multipolygon = unary_union([sys.shape for sys in self.systems])
                  
                  # compute the coverage for certain steering angle and system

                  intersection_result = systems_multipolygon.intersection(self.areas_multipolygon)
                  total_intersection_area = intersection_result.area
                  if total_intersection_area > max_intersection_area:

                     max_intersection_area = total_intersection_area
                     chosen_system = system_i
                     chosen_rotation = rotation
                     chosen_location = location

                  if optimized=='steering':
                     break

         self.systems[chosen_system].update_values(steer_angle=chosen_rotation,
                                                   location=chosen_location,obstacles=None)
         self.systems[chosen_system].optimized = True
         # print(f'max_intersection_area:{max_intersection_area}')
         # print(f'chosen_system:{chosen_system}')
         # print(f'chosen_rotation:{chosen_rotation}')
   def Adam_optimize(self, optimized='location', learning_rate=0.1, beta1=0.9, beta2=0.999,delta=1e-2, epsilon=1e-8, max_iter=1000):
      """
    Adam optimizer for finding local maxima of a black-box function.
    """
      
      def objective_function(steers):
         for system_i,system in enumerate(self.systems):
            system.update_values(steer_angle=steers[system_i],obstacles=None)
         systems_multipolygon = unary_union([sys.shape for sys in self.systems])
         intersection_result = systems_multipolygon.intersection(self.areas_multipolygon)
         return intersection_result.area
      
      position = np.array([sys.steer_angle for sys in self.systems])
      velocity = 0
      m = 0
      v = 0

      for t in range(1, max_iter + 1):
         gradient = (objective_function(position + delta) - objective_function(position - delta)) / (2 * delta)
         m = beta1 * m + (1 - beta1) * gradient
         v = beta2 * v + (1 - beta2) * (gradient ** 2)
         m_hat = m / (1 - beta1 ** t)
         v_hat = v / (1 - beta2 ** t)
         velocity = beta1 * velocity + (1 - beta1) * gradient
         position += learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
         # if t%100==0:
         #    print(f'step {t} position: {position}')
         #    print(f'step {t}: {objective_function(position)}')
      return position
