import numpy as np
from tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import Point, Pose

def get_min_max_points(points):
    ps = np.abs(points).sum(axis=2)
    xs, ys = np.where(ps > 0)
    
    out = list()
    i = xs.argmax()
    x = xs[i]
    y = ys[i]
    out.append(points[x][y])
    i = xs.argmin()
    x = xs[i]
    y = ys[i]
    out.append(points[x][y])
    i = ys.argmax()
    x = xs[i]
    y = ys[i]
    out.append(points[x][y])
    i = ys.argmin()
    x = xs[i]
    y = ys[i]
    out.append(points[x][y])
    return out

def get_dist_yz(p1: Pose, p2: Pose):
    p1 = p1.position
    a = np.array([p1.y,p1.z])
    p2 = p2.position
    b = np.array([p2.y,p2.z])

    dist = np.linalg.norm(a-b)
    return dist

def get_bbox_dims(depth_image, intrinsic, transform):
    fx, fy, cx, cy = intrinsic[0:4]
    height, width = depth_image.shape

    # Create grid of pixel coordinates
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)

    # Calculate 3D coordinates
    z = depth_image
    x = (x_grid - cx) * z / fx
    y = (y_grid - cy) * z / fy
    
    # Stack the coordinates
    points = np.stack((x, y, z), axis=-1)

    minmax_points = get_min_max_points(points)
    for p in minmax_points:
        print(p)

    minmax_points = [Pose(position=Point(x=x, y=y, z=z)) for x,y,z in minmax_points]
    minmax_points = [do_transform_pose(p, transform) for p in minmax_points]
    for p in minmax_points:
        print(p)

    height = get_dist_yz(minmax_points[0], minmax_points[1])
    width = get_dist_yz(minmax_points[2], minmax_points[3])

    return height, width 
