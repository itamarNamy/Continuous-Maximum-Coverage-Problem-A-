from shapely.geometry.polygon import Polygon
import random
import numpy as np

class Convex_polygon:
    def __init__(self):
        pass
    def random_convex_polygon(self,n_points=5, center=(0, 0), radius=1):
        """
        Generate a random convex polygon with n_points around a center point within a given radius.
        """
        angle_step = 360 / n_points
        points = []
        for i in range(n_points):
            angle = random.uniform(i * angle_step, (i + 1) * angle_step)
            r = random.uniform(0.5 * radius, radius)
            x = center[0] + r * np.cos(np.radians(angle))
            y = center[1] + r * np.sin(np.radians(angle))
            points.append((x, y))
        
        # Return the convex hull of the points
        polygon = Polygon(points).convex_hull
        return polygon

    def generate_non_intersecting_polygons(self,n_polygons=100, n_points=5, radius=5):
        polygons = []
        for i in range(n_polygons):
            while True:
                center = (random.uniform(-10, 10), random.uniform(-10, 10))  # Generate random center point
                polygon = self.random_convex_polygon(n_points=n_points, center=center, radius=radius)
                
                # Check if the new polygon intersects with any existing ones
                if not any(polygon.intersects(p) for p in polygons):
                    polygons.append(polygon)
                    break
        return polygons