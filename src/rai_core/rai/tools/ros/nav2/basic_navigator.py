# Copyright (C) 2024 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from geometry_msgs.msg import Point, Quaternion
from langchain.tools import tool
import numpy as np

from nav2_simple_commander.robot_navigator import TaskResult
from rai.tools.ros.nav2.navigator import RaiNavigator
from tf2_ros import PoseStamped
from tf_transformations import quaternion_from_euler
import rclpy

rclpy.init()
navigator = RaiNavigator()

@tool
def spin_robot(degrees: float, time_allowance_seconds: int=20) -> str:
    """
    Use this tool to spin the robot.
    
    Example usage:
    spin_robot args: {"degrees": 90}
    -> returns: "Robot spinning.", comment: "Robot will spin 90 degrees right"
    spin_robot args: {"degrees": -90, "time_allowance_seconds": 15}
    -> returns: "Robot spinning.", comment: "Robot will spin 90 degrees left for max 15 seconds. If spin is still not complated after 15s, robot will stop spinning, but will not go back to original orientation.
    """
    radians = np.radians(degrees)
    navigator.spin(spin_dist=radians, time_allowance=time_allowance_seconds)
    return "Robot spinning."


@tool
def drive_forward(distance_m: float) -> str:
    """
    Use this tool to drive the robot forward for certain distance.
    
    Example usage:
    drive_forward args: {"distance_m": 1.0}
    -> returns: "Robot driving forward."
    """
    p = Point()
    p.x = distance_m

    navigator.drive_on_heading(p, 0.5, 10)
    return "Robot driving forward."

@tool
def go_to_pose(x: float, y: float, yaw: float) -> str:
    """ 
    Use this tool to start navigating to a specific pose 
    Example usage:
    go_to_pose args: {"x": 1.0, "y": 2.0, "yaw": 0.0}
    -> returns: "Robot navigating to pose."
    """
    quat = quaternion_from_euler(0, 0, yaw)
    goal_pose = PoseStamped()
    goal_pose.header.frame_id = 'map'
    goal_pose.header.stamp = navigator.get_clock().now().to_msg()
    goal_pose.pose.position.x = x
    goal_pose.pose.position.y = y
    goal_pose.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]) 

    navigator.goToPose(goal_pose)

    return "Robot navigating to pose."

@tool
def check_nav_status() -> str:
    """
    General tool to check if any type of navigation task is completed.
    
    Example usage:
    check_nav_status args: {}
    -> returns: "Navigation Succeeded"
    check_nav_status args: {}
    -> returns: "Navigation failed", comment: Possible reasons: too small time allowance in spin, collision during spin, navigation goal pose cannot be reached. 
    check_nav_status args: {}
    -> returns: "Navigation in progress", comment: robot is still driving
    """
    if not navigator.isTaskComplete():
        return "Navigation in progress"
    else:
        result = navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            return "Navigation Succeeded"
        else:
            navigator.cancelTask()
            return "Navigation Failed"
@tool
def cancel_navigation_task():
    """
    Tool to cancel currently running navigation task

    Example usage:
    cancel_navigation_task args: {}
    """
    navigator.cancelTask()
    return "Task cancelled successfully"

tools = [
        check_nav_status,
        go_to_pose,
        spin_robot,
        drive_forward
]

import math

def quat2eulers(q0:float, q1:float, q2:float, q3:float) -> tuple:
    """
    Compute yaw-pitch-roll Euler angles from a quaternion.
    
    Args
    ----
        q0: Scalar component of quaternion.
        q1, q2, q3: Vector components of quaternion.
    
    Returns
    -------
        (roll, pitch, yaw) (tuple): 321 Euler angles in radians
    """
    roll = math.atan2(
        2 * ((q2 * q3) + (q0 * q1)),
        q0**2 - q1**2 - q2**2 + q3**2
    )  # radians
    pitch = math.asin(2 * ((q1 * q3) - (q0 * q2)))
    yaw = math.atan2(
        2 * ((q1 * q2) + (q0 * q3)),
        q0**2 + q1**2 - q2**2 - q3**2
    )
    return (roll, pitch, yaw)
