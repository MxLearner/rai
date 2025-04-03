# Copyright (C) 2025 Robotec.AI
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

from typing import cast

from langchain.tools import BaseTool
import rclpy
import rclpy.executors
import rclpy.logging

import logging
from typing import Literal, Type

import numpy as np
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from tf2_geometry_msgs import do_transform_pose

from rai.communication.ros2.connectors import ROS2ARIConnector
from rai.tools.utils import TF2TransformFetcher
from rai.utils.ros_async import get_future_result
from tf2_geometry_msgs import do_transform_pose
# import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import render_text_description_and_args, tool
from rai.agents.conversational_agent import create_conversational_agent
from rai.agents.integrations.streamlit import get_streamlit_cb, streamlit_invoke
from rai.communication.ros2 import ROS2ARIConnector
from rai.messages import HumanMultimodalMessage
from rai.tools.ros2 import (
    GetROS2ImageTool,
    GetROS2TransformTool,
)
from rai_open_set_vision.tools import GetGrabbingPointTool
from rai_open_set_vision.tools.segmentation_tools import depth_to_point_cloud
from rai.tools.ros.manipulation import GetObjectPositionsTool
from rai.tools.ros2.actions import (
    GetROS2ActionFeedbackTool,
    GetROS2ActionResultTool,
    ROS2ActionToolkit,
    StartROS2ActionTool,
)
from rai.tools.time import WaitForSecondsTool
from rai.utils.model_initialization import get_llm_model, get_tracing_callbacks
from rai_open_set_vision.tools.gdino_tools import *

from typing import Any, List, Optional, Sequence, Type

import cv2
import numpy as np
import rclpy
import sensor_msgs.msg
from langchain_core.tools import BaseTool
from pydantic import Field
from rai.communication.ros2.connectors import ROS2ARIConnector
from rai.tools.ros import Ros2BaseInput
from rai.tools.ros.utils import convert_ros_img_to_base64, convert_ros_img_to_ndarray
from rai.utils.ros_async import get_future_result
from rclpy import Future
from rclpy.exceptions import (
    ParameterNotDeclaredException,
    ParameterUninitializedException,
)

from rai_interfaces.srv import RAIGroundedSam, RAIGroundingDino
from rai_open_set_vision import GDINO_SERVICE_NAME

from rai_open_set_vision.tools.utils import get_bbox_dims

class GetBoundingBoxToolInput(BaseModel):
    object_name: str = Field(
        ..., description="The name of the object to get the positions of"
    )
from typing import Optional

class GetBoundingBox(BaseTool):
    connector: ROS2ARIConnector = Field(..., exclude=True)

    target_frame: str
    source_frame: str
    camera_topic: str  # rgb camera topic
    depth_topic: str
    camera_info_topic: str  # rgb camera info topic
    connector: ROS2ARIConnector = Field(..., exclude=True)

    name: str = "GetGrabbingPointTool"
    description: str = "Get the grabbing point of an object"
    pcd: List[Any] = []

    args_schema: Type[GetBoundingBoxToolInput] = GetBoundingBoxToolInput
    box_threshold: float = Field(default=0.35, description="Box threshold for GDINO")
    text_threshold: float = Field(default=0.45, description="Text threshold for GDINO")

    def _get_gdino_response(
        self, future: Future
    ) -> Optional[RAIGroundingDino.Response]:
        return get_future_result(future)

    def _get_gsam_response(self, future: Future) -> Optional[RAIGroundedSam.Response]:
        return get_future_result(future)

    def _get_image_message(self, topic: str) -> sensor_msgs.msg.Image:
        msg = self.connector.receive_message(topic).payload
        if type(msg) is sensor_msgs.msg.Image:
            return msg
        else:
            raise Exception("Received wrong message")

    def _call_gdino_node(
        self, camera_img_message: sensor_msgs.msg.Image, object_name: str
    ) -> Future:
        cli = self.connector.node.create_client(RAIGroundingDino, GDINO_SERVICE_NAME)
        while not cli.wait_for_service(timeout_sec=1.0):
            self.connector.node.get_logger().info(
                "service not available, waiting again..."
            )
        req = RAIGroundingDino.Request()
        req.source_img = camera_img_message
        req.classes = object_name
        req.box_threshold = self.box_threshold
        req.text_threshold = self.text_threshold

        future = cli.call_async(req)
        return future

    def _call_gsam_node(
        self, camera_img_message: sensor_msgs.msg.Image, data: RAIGroundingDino.Response
    ):
        cli = self.connector.node.create_client(RAIGroundedSam, "grounded_sam_segment")
        while not cli.wait_for_service(timeout_sec=1.0):
            self.connector.node.get_logger().info(
                "service not available, waiting again..."
            )
        req = RAIGroundedSam.Request()
        req.detections = data.detections
        req.source_img = camera_img_message
        future = cli.call_async(req)

        return future

    def _get_camera_info_message(self, topic: str) -> sensor_msgs.msg.CameraInfo:
        for _ in range(3):
            msg = self.connector.receive_message(topic, timeout_sec=3.0).payload
            if isinstance(msg, sensor_msgs.msg.CameraInfo):
                return msg
            self.connector.node.get_logger().warn(
                "Received wrong message type. Retrying..."
            )

        raise Exception("Failed to receive correct CameraInfo message after 3 attempts")

    def _get_intrinsic_from_camera_info(self, camera_info: sensor_msgs.msg.CameraInfo):
        """Extract camera intrinsic parameters from the CameraInfo message."""

        fx = camera_info.k[0]  # Focal length in x-axis
        fy = camera_info.k[4]  # Focal length in y-axis
        cx = camera_info.k[2]  # Principal point x
        cy = camera_info.k[5]  # Principal point y

        return fx, fy, cx, cy

    def _process_mask(
        self,
        mask_msg: sensor_msgs.msg.Image,
        depth_msg: sensor_msgs.msg.Image,
        intrinsic: Sequence[float],
        depth_to_meters_ratio: float,
    ):
        mask = convert_ros_img_to_ndarray(mask_msg)
        binary_mask = np.where(mask == 255, 1, 0)
        depth = convert_ros_img_to_ndarray(depth_msg)
        masked_depth_image = np.zeros_like(depth, dtype=np.float32)
        masked_depth_image[binary_mask == 1] = depth[binary_mask == 1]
        masked_depth_image = masked_depth_image * depth_to_meters_ratio

        print(f'{masked_depth_image.shape=}')
        import pickle as pkl 
        with open('mdi.pkl', 'wb') as f:
            pkl.dump(masked_depth_image, f)
        with open('intrinsic.pkl', 'wb') as f:
            pkl.dump(intrinsic, f)


        pcd = depth_to_point_cloud(
            masked_depth_image, intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        )
        
        print(f'{pcd.shape=}')

        # TODO: Filter out outliers
        points = pcd

        # https://github.com/ycheng517/tabletop-handybot/blob/6d401e577e41ea86529d091b406fbfc936f37a8d/tabletop_handybot/tabletop_handybot/tabletop_handybot_node.py#L413-L424
        grasp_z = points[:, 2].max()
        near_grasp_z_points = points[points[:, 2] > grasp_z - 0.008]
        xy_points = near_grasp_z_points[:, :2]
        xy_points = xy_points.astype(np.float32)
        _, dimensions, theta = cv2.minAreaRect(xy_points)

        gripper_rotation = theta
        # NOTE  - estimated dimentsion from the RGBDCamera5 not very precise, what may cause not desired rotation
        if dimensions[0] > dimensions[1]:
            gripper_rotation -= 90
        if gripper_rotation < -90:
            gripper_rotation += 180
        elif gripper_rotation > 90:
            gripper_rotation -= 180

        # Calculate full 3D centroid for OBJECT
        centroid = np.mean(points, axis=0)
        return centroid, gripper_rotation, masked_depth_image, intrinsic
    
    def _parse_detection_array(
        self, detection_response: RAIGroundingDino.Response
    ) -> list[DetectionData]:
        detected = []
        for detection in detection_response.detections.detections:
            class_name = detection.results[0].hypothesis.class_id
            confidence = detection.results[0].hypothesis.score
            bbox = BoundingBox(
                x_center=detection.bbox.center.position.x,
                y_center=detection.bbox.center.position.y,
                width=detection.bbox.size_x,
                height=detection.bbox.size_y,
            )
            detected.append(
                DetectionData(class_name=class_name, confidence=confidence, bbox=bbox)
            )
        return detected

    def _run(
        self,
        object_name: str,
    ):
        camera_img_msg = self.connector.receive_message(self.camera_topic).payload
        depth_msg = self.connector.receive_message(self.depth_topic).payload
        camera_info = self._get_camera_info_message(self.camera_info_topic)

        intrinsic = self._get_intrinsic_from_camera_info(camera_info)
        transform = TF2TransformFetcher(
            target_frame=self.target_frame, source_frame=self.source_frame
        ).get_data()

        import pickle as pkl
        with open('transform.pkl', 'wb') as f:
            pkl.dump(transform, f)

        future = self._call_gdino_node(camera_img_msg, object_name)
        logger = self.connector.node.get_logger()
        try:
            conversion_ratio = self.connector.node.get_parameter(
                "conversion_ratio"
            ).value
            if not isinstance(conversion_ratio, float):
                logger.error(
                    f"Parameter conversion_ratio was set badly: {type(conversion_ratio)}: {conversion_ratio} expected float. Using default value 0.001"
                )
                conversion_ratio = 0.001
        except (ParameterUninitializedException, ParameterNotDeclaredException):
            logger.warning(
                "Parameter conversion_ratio not found in node, using default value: 0.001"
            )
            conversion_ratio = 0.001
        resolved = None

        resolved = get_future_result(future)

        assert resolved is not None

        detected = self._parse_detection_array(resolved)
        
        future = self._call_gsam_node(camera_img_msg, resolved)

        ret = []
        resolved = get_future_result(future)
        if resolved is not None:
            for img_msg in resolved.masks:
                ret.append(convert_ros_img_to_base64(img_msg))
        assert resolved is not None
        rets = []
        for mask_msg in resolved.masks:
            rets.append(
                self._process_mask(
                    mask_msg,
                    depth_msg,
                    intrinsic,
                    depth_to_meters_ratio=conversion_ratio,
                )
            )

        results = rets

        poses = []
        boxes_dims = []

        for result in results:
            masked_depth_image = result[2]
            intrinsic = result[3]
            h, w = get_bbox_dims(masked_depth_image, intrinsic, transform)
            boxes_dims.append((h,w))
            cam_pose = result[0]
            poses.append(
                Pose(position=Point(x=cam_pose[0], y=cam_pose[1], z=cam_pose[2]))
            )

        target_frame_poses = []
        for pose in poses:
            target_frame_pose = do_transform_pose(pose, transform)
            target_frame_poses.append(target_frame_pose)

        return list(zip(detected, target_frame_poses, boxes_dims))


# @st.cache_resource
def initialize_agent():
    rclpy.init()
    connector = ROS2ARIConnector()
    connector.node.declare_parameter("conversion_ratio", 1.0)

    transform_tool = GetROS2TransformTool(connector=connector)

    @tool
    def get_object_position(object_name: str):
        """
        Get positions of an object of a specified type in the target frame.
        """
        box, centroid, (h,w) = GetBoundingBox(
            connector=connector,
            target_frame="map",
            source_frame="sensor_frame",
            camera_topic="/camera/camera/color/image_raw",
            depth_topic="/camera/camera/depth/image_rect_raw",
            camera_info_topic="/camera/camera/color/camera_info",
        )._run(object_name=object_name)[0]
        current_pose = transform_tool._run(source_frame="base_link", target_frame="map", timeout_sec=10)

        print(f'{box=}')
        print(f'{centroid=}')
        print(f'{current_pose=}')
        print(f'Height: {h}')
        print(f'Width: {w}')

        return box, centroid, dims

    get_object_position.invoke(dict(object_name="human"))
    return 

    tools = [
        get_object_position
    ]

    SYSTEM_PROMPT = (
        """You are an autonomous robot connected to ros2 environment. Your main goal is to fulfill the user's requests.
    Do not make assumptions about the environment you are currently in.

    Here are the tools you can use:
    """
        + f"{render_text_description_and_args(tools)}"
    ) + """
    Here are some examples of how to use the tools:
    get_object_position, args: {'object_name': 'apple'}
    get_object_position, args: {'object_name': 'chair'}
"""

    agent = create_conversational_agent(
        llm=get_llm_model("complex_model", streaming=True),
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
    )
    connector.node.declare_parameter("conversion_ratio", 1.0)
    callbacks = get_tracing_callbacks()
    return agent, callbacks


def main2():
    agent, callbacks = initialize_agent()
    state = {"messages": []}
    while True:
        inp = input("Enter your message: ")
        state["messages"].append(HumanMessage(content=inp))
        l_before = len(state["messages"])
        response = agent.invoke(state, config={"callbacks": callbacks, 'recursion_limit': 2000})
        l_after = len(state["messages"])
        for msg in state["messages"][l_before:l_after]:
            msg.pretty_print()


if __name__ == "__main__":
    main2()
