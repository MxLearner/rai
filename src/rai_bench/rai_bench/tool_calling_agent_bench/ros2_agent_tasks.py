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

import copy
import json
import logging
from itertools import permutations
from typing import Any, Dict, List, Sequence

import inflect
from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool
from rai.tools.ros.manipulation import MoveToPointToolInput
from rai_open_set_vision.tools.gdino_tools import DistanceMeasurement

from rai_bench.tool_calling_agent_bench.actions import (
    ActionBaseModel,
    DriveOnHeadingAction,
    NavigateToPoseAction,
    SpinAction,
)
from rai_bench.tool_calling_agent_bench.agent_tasks_interfaces import (
    ANY_VALUE,
    ROS2ToolCallingAgentTask,
)
from rai_bench.tool_calling_agent_bench.mocked_tools import (
    MockCallROS2ServiceTool,
    MockGetDistanceToObjectsTool,
    MockGetObjectPositionsTool,
    MockGetROS2ActionFeedbackTool,
    MockGetROS2ActionResultTool,
    MockGetROS2ActionsNamesAndTypesTool,
    MockGetROS2ImageTool,
    MockGetROS2MessageInterfaceTool,
    MockGetROS2ServicesNamesAndTypesTool,
    MockGetROS2TopicsNamesAndTypesTool,
    MockMoveToPointTool,
    MockPublishROS2MessageTool,
    MockReceiveROS2MessageTool,
    MockStartROS2ActionTool,
)

loggers_type = logging.Logger


PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT = """You are a ROS 2 expert helping a user with their ROS 2 questions. You have access to various tools that allow you to query the ROS 2 system.
                Be proactive and use the tools to answer questions.
                """

INTERFACES: Dict[str, str] = {
    "nav2_msgs/action/Spin": """
#goal definition
float32 target_yaw
builtin_interfaces/Duration time_allowance
	int32 sec
	uint32 nanosec
---
#result definition
builtin_interfaces/Duration total_elapsed_time
	int32 sec
	uint32 nanosec
---
#feedback definition
float32 angular_distance_traveled
""",
    "nav2_msgs/action/NavigateToPose": """
#goal definition
geometry_msgs/PoseStamped pose
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	Pose pose
		Point position
			float64 x
			float64 y
			float64 z
		Quaternion orientation
			float64 x 0
			float64 y 0
			float64 z 0
			float64 w 1
string behavior_tree
---
#result definition
std_msgs/Empty result
---
#feedback definition
geometry_msgs/PoseStamped current_pose
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	Pose pose
		Point position
			float64 x
			float64 y
			float64 z
		Quaternion orientation
			float64 x 0
			float64 y 0
			float64 z 0
			float64 w 1
builtin_interfaces/Duration navigation_time
	int32 sec
	uint32 nanosec
builtin_interfaces/Duration estimated_time_remaining
	int32 sec
	uint32 nanosec
int16 number_of_recoveries
float32 distance_remaining
""",
    "nav2_msgs/action/DriveOnHeading": """
#goal definition
geometry_msgs/Point target
	float64 x
	float64 y
	float64 z
float32 speed
builtin_interfaces/Duration time_allowance
	int32 sec
	uint32 nanosec
---
#result definition
builtin_interfaces/Duration total_elapsed_time
	int32 sec
	uint32 nanosec
---
#feedback definition
float32 distance_traveled
""",
}


class TaskParametrizationError(Exception):
    """Exception raised when the task parameters are not valid."""

    pass


class GetROS2TopicsTask(ROS2ToolCallingAgentTask):
    complexity = "easy"

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.expected_tools = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=[
                    "topic: /attached_collision_object\ntype: moveit_msgs/msg/AttachedCollisionObject\n",
                    "topic: /camera_image_color\ntype: sensor_msgs/msg/Image\n",
                    "topic: /camera_image_depth\ntype: sensor_msgs/msg/Image\n",
                    "topic: /clock\ntype: rosgraph_msgs/msg/Clock\n",
                    "topic: /collision_object\ntype: moveit_msgs/msg/CollisionObject\n",
                    "topic: /color_camera_info\ntype: sensor_msgs/msg/CameraInfo\n",
                    "topic: /color_camera_info5\ntype: sensor_msgs/msg/CameraInfo\n",
                    "topic: /color_image5\ntype: sensor_msgs/msg/Image\n",
                    "topic: /depth_camera_info5\ntype: sensor_msgs/msg/CameraInfo\n",
                    "topic: /depth_image5\ntype: sensor_msgs/msg/Image\n",
                    "topic: /display_contacts\ntype: visualization_msgs/msg/MarkerArray\n",
                    "topic: /display_planned_path\ntype: moveit_msgs/msg/DisplayTrajectory\n",
                    "topic: /execute_trajectory/_action/feedback\ntype: moveit_msgs/action/ExecuteTrajectory_FeedbackMessage\n",
                    "topic: /execute_trajectory/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
                    "topic: /joint_states\ntype: sensor_msgs/msg/JointState\n",
                    "topic: /monitored_planning_scene\ntype: moveit_msgs/msg/PlanningScene\n",
                    "topic: /motion_plan_request\ntype: moveit_msgs/msg/MotionPlanRequest\n",
                    "topic: /move_action/_action/feedback\ntype: moveit_msgs/action/MoveGroup_FeedbackMessage\n",
                    "topic: /move_action/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
                    "topic: /panda_arm_controller/follow_joint_trajectory/_action/feedback\ntype: control_msgs/action/FollowJointTrajectory_FeedbackMessage\n",
                    "topic: /panda_arm_controller/follow_joint_trajectory/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
                    "topic: /panda_hand_controller/gripper_cmd/_action/feedback\ntype: control_msgs/action/GripperCommand_FeedbackMessage\n",
                    "topic: /panda_hand_controller/gripper_cmd/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
                    "topic: /parameter_events\ntype: rcl_interfaces/msg/ParameterEvent\n",
                    "topic: /planning_scene\ntype: moveit_msgs/msg/PlanningScene\n",
                    "topic: /planning_scene_world\ntype: moveit_msgs/msg/PlanningSceneWorld\n",
                    "topic: /pointcloud\ntype: sensor_msgs/msg/PointCloud2\n",
                    "topic: /robot_description\ntype: std_msgs/msg/String\n",
                    "topic: /robot_description_semantic\ntype: std_msgs/msg/String\n",
                    "topic: /rosout\ntype: rcl_interfaces/msg/Log\n",
                    "topic: /tf\ntype: tf2_msgs/msg/TFMessage\n",
                    "topic: /tf_static\ntype: tf2_msgs/msg/TFMessage\n",
                    "topic: /trajectory_execution_event\ntype: std_msgs/msg/String\n",
                ]
            )
        ]

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return "Get the names and types of all ROS2 topics"

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request only the tool that retrieves the ROS2 topics names and types

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if not ai_messages:
            self.log_error(msg="No AI messages found in the response.")

        self._is_ai_message_requesting_get_ros2_topics_and_types(ai_messages[0])

        total_tool_calls = sum(len(message.tool_calls) for message in ai_messages)
        if total_tool_calls != 1:
            self.log_error(
                msg=f"Total number of tool calls across all AI messages should be 1, but got {total_tool_calls}."
            )

        if not self.result.errors:
            self.result.success = True


class GetROS2TopicsTask2(ROS2ToolCallingAgentTask):
    complexity = "easy"

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.expected_tools = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=[
                    "topic: /attached_collision_object\ntype: moveit_msgs/msg/AttachedCollisionObject\n",
                    "topic: /camera_image_color\ntype: sensor_msgs/msg/Image\n",
                    "topic: /camera_image_depth\ntype: sensor_msgs/msg/Image\n",
                    "topic: /clock\ntype: rosgraph_msgs/msg/Clock\n",
                ]
            )
        ]

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return "What is in the ROS2 network?"

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request only the tool that retrieves the ROS2 topics names and types

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if not ai_messages:
            self.log_error(msg="No AI messages found in the response.")

        self._is_ai_message_requesting_get_ros2_topics_and_types(ai_messages[0])

        total_tool_calls = sum(len(message.tool_calls) for message in ai_messages)
        if total_tool_calls != 1:
            self.log_error(
                msg=f"Total number of tool calls across all AI messages should be 1, but got {total_tool_calls}."
            )

        if not self.result.errors:
            self.result.success = True


class GetROS2RGBCameraTask(ROS2ToolCallingAgentTask):
    complexity = "easy"

    topics_and_types: Dict[str, str] = {
        "/attached_collision_object": "moveit_msgs/msg/AttachedCollisionObject",
        "/camera_image_color": "sensor_msgs/msg/Image",
        "/camera_image_depth": "sensor_msgs/msg/Image",
        "/clock": "rosgraph_msgs/msg/Clock",
        "/collision_object": "moveit_msgs/msg/CollisionObject",
        "/color_camera_info": "sensor_msgs/msg/CameraInfo",
        "/color_camera_info5": "sensor_msgs/msg/CameraInfo",
        "/depth_camera_info5": "sensor_msgs/msg/CameraInfo",
        "/depth_image5": "sensor_msgs/msg/Image",
    }

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)

        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]

        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
            MockGetROS2ImageTool(available_topics=list(self.topics_and_types.keys())),
        ]

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return "Get the RGB image from the camera."

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request:
        1. The tool that retrieves the ROS2 topics names and types to recognize the RGB image topic
        2. The tool that retrieves the RGB image from the /camera_image_color topic

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            self.log_error(
                msg=f"Expected at least 3 AI messages, but got {len(ai_messages)}."
            )

        if ai_messages:
            if not self._is_ai_message_requesting_get_ros2_topics_and_types(
                ai_messages[0]
            ):
                self.log_error(
                    msg="First AI message did not request ROS2 topics and types correctly."
                )
        if len(ai_messages) > 1:
            if self._check_tool_calls_num_in_ai_message(ai_messages[1], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[1].tool_calls[0],
                    expected_name="get_ros2_image",
                    expected_args={"topic": "/camera_image_color"},
                    expected_optional_args={"timeout_sec": None},
                )
        if not self.result.errors:
            self.result.success = True


class GetROS2DepthCameraTask(ROS2ToolCallingAgentTask):
    complexity = "easy"

    topics_and_types: Dict[str, str] = {
        "/attached_collision_object": "moveit_msgs/msg/AttachedCollisionObject",
        "/camera_image_color": "sensor_msgs/msg/Image",
        "/camera_image_depth": "sensor_msgs/msg/Image",
        "/clock": "rosgraph_msgs/msg/Clock",
        "/collision_object": "moveit_msgs/msg/CollisionObject",
        "/color_camera_info": "sensor_msgs/msg/CameraInfo",
        "/color_camera_info5": "sensor_msgs/msg/CameraInfo",
        "/depth_camera_info5": "sensor_msgs/msg/CameraInfo",
    }

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
            MockGetROS2ImageTool(available_topics=list(self.topics_and_types.keys())),
        ]

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return "Get the depth image from the camera."

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request:
        1. The tool that retrieves the ROS2 topic names and types to identify the depth image topic.
        2. The tool that retrieves the depth image from the /camera_image_depth topic

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            self.log_error(
                msg=f"Expected at least 3 AI messages, but got {len(ai_messages)}."
            )

        if ai_messages:
            if not self._is_ai_message_requesting_get_ros2_topics_and_types(
                ai_messages[0]
            ):
                self.log_error(
                    msg="First AI message did not request ROS2 topics and types correctly."
                )

        if len(ai_messages) > 1:
            if self._check_tool_calls_num_in_ai_message(ai_messages[1], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[1].tool_calls[0],
                    expected_name="get_ros2_image",
                    expected_args={"topic": "/camera_image_depth"},
                    expected_optional_args={"timeout_sec": None},
                )
        if not self.result.errors:
            self.result.success = True


class GetAllROS2RGBCamerasTask(ROS2ToolCallingAgentTask):
    complexity = "easy"

    topics_and_types: Dict[str, str] = {
        "/attached_collision_object": "moveit_msgs/msg/AttachedCollisionObject",
        "/camera_image_color": "sensor_msgs/msg/Image",
        "/camera_image_depth": "sensor_msgs/msg/Image",
        "/clock": "rosgraph_msgs/msg/Clock",
        "/collision_object": "moveit_msgs/msg/CollisionObject",
        "/color_camera_info": "sensor_msgs/msg/CameraInfo",
        "/color_camera_info5": "sensor_msgs/msg/CameraInfo",
        "/color_image5": "sensor_msgs/msg/Image",
        "/depth_camera_info5": "sensor_msgs/msg/CameraInfo",
        "/depth_image5": "sensor_msgs/msg/Image",
    }

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
            MockGetROS2ImageTool(available_topics=list(self.topics_and_types.keys())),
        ]

    def get_prompt(self) -> str:
        return "Get RGB images from all of the available cameras."

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request:
        1. The tool that retrieves the ROS2 topic names and types to identify the topics with RGB images.
        2. The tool that retrieves the RGB images - from the /camera_image_color and from the /color_image5 topic

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            self.log_error(
                msg=f"Expected at least 3 AI messages, but got {len(ai_messages)}."
            )
        if ai_messages:
            if not self._is_ai_message_requesting_get_ros2_topics_and_types(
                ai_messages[0]
            ):
                self.log_error(
                    msg="First AI message did not request ROS2 topics and types correctly."
                )

        if len(ai_messages) > 1:
            expected_tool_calls: list[dict[str, Any]] = [
                {
                    "name": "get_ros2_image",
                    "args": {"topic": "/camera_image_color"},
                    "optional_args": {"timeout_sec": None},
                },
                {
                    "name": "get_ros2_image",
                    "args": {"topic": "/color_image5"},
                    "optional_args": {"timeout_sec": None},
                },
            ]
            self._check_multiple_tool_calls(
                message=ai_messages[1], expected_tool_calls=expected_tool_calls
            )
        if not self.result.errors:
            self.result.success = True


class GetAllROS2DepthCamerasTask(ROS2ToolCallingAgentTask):
    complexity = "easy"

    topics_and_types: Dict[str, str] = {
        "/camera_image_color": "sensor_msgs/msg/Image",
        "/camera_image_depth": "sensor_msgs/msg/Image",
        "/clock": "rosgraph_msgs/msg/Clock",
        "/collision_object": "moveit_msgs/msg/CollisionObject",
        "/color_camera_info": "sensor_msgs/msg/CameraInfo",
        "/color_camera_info5": "sensor_msgs/msg/CameraInfo",
        "/color_image5": "sensor_msgs/msg/Image",
        "/depth_camera_info5": "sensor_msgs/msg/CameraInfo",
        "/depth_image5": "sensor_msgs/msg/Image",
    }

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
            MockGetROS2ImageTool(available_topics=list(self.topics_and_types.keys())),
        ]

    def get_prompt(self) -> str:
        return "Get depth images from all of the available cameras."

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request:
        1. The tool that retrieves the ROS2 topic names and types to identify the topics with depth images.
        2. The tool that retrieves the depth images - from the /camera_image_depth and from the /depth_image5 topic

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            self.log_error(
                msg=f"Expected at least 3 AI messages, but got {len(ai_messages)}."
            )
        if ai_messages:
            if not self._is_ai_message_requesting_get_ros2_topics_and_types(
                ai_messages[0]
            ):
                self.log_error(
                    msg="First AI message did not request ROS2 topics and types correctly."
                )

        if len(ai_messages) > 1:
            expected_tool_calls: list[dict[str, Any]] = [
                {
                    "name": "get_ros2_image",
                    "args": {"topic": "/camera_image_depth"},
                    "optional_args": {"timeout_sec": None},
                },
                {
                    "name": "get_ros2_image",
                    "args": {"topic": "/depth_image5"},
                    "optional_args": {"timeout_sec": None},
                },
            ]
            self._check_multiple_tool_calls(
                message=ai_messages[1], expected_tool_calls=expected_tool_calls
            )
        if not self.result.errors:
            self.result.success = True


class GetROS2MessageTask(ROS2ToolCallingAgentTask):
    complexity = "easy"

    topics_and_types: Dict[str, str] = {
        "/attached_collision_object": "moveit_msgs/msg/AttachedCollisionObject",
        "/camera_image_color": "sensor_msgs/msg/Image",
        "/camera_image_depth": "sensor_msgs/msg/Image",
        "/clock": "rosgraph_msgs/msg/Clock",
        "/collision_object": "moveit_msgs/msg/CollisionObject",
        "/color_camera_info": "sensor_msgs/msg/CameraInfo",
        "/color_camera_info5": "sensor_msgs/msg/CameraInfo",
        "/depth_camera_info5": "sensor_msgs/msg/CameraInfo",
        "/depth_image5": "sensor_msgs/msg/Image",
    }

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)

        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]

        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
            MockReceiveROS2MessageTool(
                available_topics=list(self.topics_and_types.keys())
            ),
        ]

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return "Get RGB image."

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request:
        1. The tool that retrieves the ROS2 topics names and types to recognize the RGB image topic.
        2. The tool that retrieves the RGB image from the /camera_image_color topic

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            self.log_error(
                msg=f"Expected at least 3 AI messages, but got {len(ai_messages)}."
            )

        if ai_messages:
            if not self._is_ai_message_requesting_get_ros2_topics_and_types(
                ai_messages[0]
            ):
                self.log_error(
                    msg="First AI message did not request ROS2 topics and types correctly."
                )

        if len(ai_messages) > 1:
            if self._check_tool_calls_num_in_ai_message(ai_messages[1], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[1].tool_calls[0],
                    expected_name="receive_ros2_message",
                    expected_args={"topic": "/camera_image_color"},
                )

        if not self.result.errors:
            self.result.success = True


class GetRobotDescriptionTask(ROS2ToolCallingAgentTask):
    complexity = "easy"

    topics_and_types: Dict[str, str] = {
        "/pointcloud": "sensor_msgs/msg/PointCloud2",
        "/robot_description": "std_msgs/msg/String",
        "/rosout": "rcl_interfaces/msg/Log",
        "/tf": "tf2_msgs/msg/TFMessage",
        "/tf_static": "tf2_msgs/msg/TFMessage",
        "/trajectory_execution_event": "std_msgs/msg/String",
    }

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)

        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]

        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
            MockReceiveROS2MessageTool(
                available_topics=list(self.topics_and_types.keys())
            ),
        ]

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return "Give me description of the robot."

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request:
        1. The tool that retrieves the ROS2 topics names and types to recognize the topic with the robot description
        2. The tool that retrieves the message from the /robot_description topic

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            self.log_error(
                msg=f"Expected at least 3 AI messages, but got {len(ai_messages)}."
            )
        if ai_messages:
            if not self._is_ai_message_requesting_get_ros2_topics_and_types(
                ai_messages[0]
            ):
                self.log_error(
                    msg="First AI message did not request ROS2 topics and types correctly."
                )
        if len(ai_messages) > 1:
            if self._check_tool_calls_num_in_ai_message(ai_messages[1], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[1].tool_calls[0],
                    expected_name="receive_ros2_message",
                    expected_args={"topic": "/robot_description"},
                )

        if not self.result.errors:
            self.result.success = True


class GetPointcloudTask(ROS2ToolCallingAgentTask):
    complexity = "easy"

    topics_and_types: Dict[str, str] = {
        "/pointcloud": "sensor_msgs/msg/PointCloud2",
        "/robot_description": "std_msgs/msg/String",
        "/rosout": "rcl_interfaces/msg/Log",
        "/tf": "tf2_msgs/msg/TFMessage",
        "/tf_static": "tf2_msgs/msg/TFMessage",
        "/trajectory_execution_event": "std_msgs/msg/String",
    }

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)

        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]

        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
            MockReceiveROS2MessageTool(
                available_topics=list(self.topics_and_types.keys())
            ),
        ]

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return "Get the pointcloud."

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request:
        1. The tool that retrieves the ROS2 topics names and types to recognize the topic with the pointcloud
        2. The tool that retrieves the message from the /pointcloud topic

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            self.log_error(
                msg=f"Expected at least 3 AI messages, but got {len(ai_messages)}."
            )

        if ai_messages:
            if not self._is_ai_message_requesting_get_ros2_topics_and_types(
                ai_messages[0]
            ):
                self.log_error(
                    msg="First AI message did not request ROS2 topics and types correctly."
                )

        if len(ai_messages) > 1:
            if self._check_tool_calls_num_in_ai_message(ai_messages[1], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[1].tool_calls[0],
                    expected_name="receive_ros2_message",
                    expected_args={"topic": "/pointcloud"},
                )

        if not self.result.errors:
            self.result.success = True


class MoveToPointTask(ROS2ToolCallingAgentTask):
    complexity = "easy"

    topics_and_types: Dict[str, str] = {
        "/pointcloud": "sensor_msgs/msg/PointCloud2",
        "/robot_description": "std_msgs/msg/String",
        "/rosout": "rcl_interfaces/msg/Log",
        "/tf": "tf2_msgs/msg/TFMessage",
    }

    def __init__(
        self, args: Dict[str, Any], logger: loggers_type | None = None
    ) -> None:
        super().__init__(logger=logger)

        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]

        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
            MockMoveToPointTool(manipulator_frame="base_link"),
        ]

        self.args = MoveToPointToolInput(**args)

    def get_system_prompt(self) -> str:
        return """You are a ROS 2 expert helping the user to manipulate the robotic arm. You have access to various tools that allow you to query the ROS 2 system.
                Be proactive and use the tools to answer questions.
                """

    def get_prompt(self) -> str:
        return f"Move the arm to a point x={self.args.x}, y={self.args.y}, z={self.args.z} to {self.args.task} an object."

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request the tool that moves the arm to a point specified in the prompt with requested task (grab or drop)"

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if not ai_messages:
            self.log_error(msg="No AI messages found in the response.")
        else:
            total_tool_calls = sum(len(message.tool_calls) for message in ai_messages)
            if total_tool_calls != 1:
                self.log_error(
                    msg=f"Total number of tool calls across all AI messages should be 1, but got {total_tool_calls}."
                )
            else:
                self._check_tool_call(
                    tool_call=ai_messages[0].tool_calls[0],
                    expected_name="move_to_point",
                    expected_args={
                        "x": self.args.x,
                        "y": self.args.y,
                        "z": self.args.z,
                        "task": self.args.task,
                    },
                )

        if not self.result.errors:
            self.result.success = True


class GetObjectPositionsTask(ROS2ToolCallingAgentTask):
    complexity = "easy"

    topics_and_types: Dict[str, str] = {
        "/pointcloud": "sensor_msgs/msg/PointCloud2",
        "/robot_description": "std_msgs/msg/String",
        "/rosout": "rcl_interfaces/msg/Log",
        "/tf": "tf2_msgs/msg/TFMessage",
    }

    def __init__(
        self,
        objects: Dict[str, List[dict[str, float]]],
        logger: loggers_type | None = None,
    ) -> None:
        """Task to get the positions of the objects

        Parameters
        ----------
        objects : Dict[str, List[dict[str, float]]]
            Dictionary containing the object types and their positions. Object type should be passed as singular.
        logger : loggers_type | None, optional
            Logger, by default None

        Examples
        --------
        objects = {
            "banana": [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)],
            "cube": [(0.7, 0.8, 0.9)],
        }
        """
        super().__init__(logger=logger)

        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]

        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
            MockGetObjectPositionsTool(mock_objects=objects),
        ]

        self.objects = objects

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        """Generates a prompt based on the objects provided in the task. If there is more than one object, the object in the prompt will be pluralized.
        Returns:
            str: Formatted prompt for the task
        """
        inflector = inflect.engine()
        object_counts = {obj: len(positions) for obj, positions in self.objects.items()}
        formatted_objects = [
            inflector.plural(obj) if count > 1 else obj
            for obj, count in object_counts.items()
        ]
        if len(formatted_objects) > 1:
            objects_list = (
                ", ".join(formatted_objects[:-1]) + f", and {formatted_objects[-1]}"
            )
        else:
            objects_list = formatted_objects[0]
        return f"Get the {objects_list} positions."

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request the tool for each object type to get its positions.

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if not ai_messages:
            self.log_error(msg="No AI messages found in the response.")
        else:
            ai_message = ai_messages[0]
            self._check_multiple_tool_calls(
                message=ai_message,
                expected_tool_calls=[
                    {
                        "name": "get_object_positions",
                        "args": {"object_name": object_type},
                    }
                    for object_type in self.objects
                ],
            )

        if not self.result.errors:
            self.result.success = True


class GrabExistingObjectTask(ROS2ToolCallingAgentTask):
    complexity = "medium"

    """Task to grab an object

    Parameters
    ----------
    objects : Dict[str, List[dict[str, float]]]
        Dictionary containing the object types and their positions. Object type should be passed as singular.
    object_to_grab : str
        Object to grab. Object type should be passed as singular. Object to be grabbed should be defined in the objects argument with only one instance (one position).
    logger : loggers_type | None, optional
        Logger, by default None

    Examples
    --------
    objects = {
        "banana": [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)],
        "cube": [(0.7, 0.8, 0.9)],
    }
    object_to_grab = "cube"
    """
    topics_and_types: Dict[str, str] = {
        "/attached_collision_object": "moveit_msgs/msg/AttachedCollisionObject",
        "/camera_image_color": "sensor_msgs/msg/Image",
        "/camera_image_depth": "sensor_msgs/msg/Image",
        "/clock": "rosgraph_msgs/msg/Clock",
        "/collision_object": "moveit_msgs/msg/CollisionObject",
        "/color_camera_info": "sensor_msgs/msg/CameraInfo",
    }

    def __init__(
        self,
        objects: Dict[str, List[dict[str, float]]],
        object_to_grab: str,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(logger=logger)

        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]

        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
            MockGetObjectPositionsTool(
                target_frame="panda_link0",
                source_frame="RGBDCamera5",
                camera_topic="/color_image5",
                depth_topic="/depth_image5",
                camera_info_topic="/color_camera_info5",
                mock_objects=objects,
            ),
            MockMoveToPointTool(manipulator_frame="panda_link0"),
        ]

        self.objects = objects
        self.object_to_grab = object_to_grab
        self._verify_args()

    def get_system_prompt(self) -> str:
        return """
        You are a robotic arm with interfaces to detect and manipulate objects.
        Here are the coordinates information:
        x - front to back (positive is forward)
        y - left to right (positive is right)
        z - up to down (positive is up).
        """

    def get_prompt(self) -> str:
        return f"Grab {self.object_to_grab}."

    def _verify_args(self):
        if self.object_to_grab not in self.objects:
            error_message = f"Requested object to grab {self.object_to_grab} is not present in defined objects: {self.objects}."
            self.log_error(msg=error_message)
            raise TaskParametrizationError(error_message)

        if len(self.objects[self.object_to_grab]) > 1:
            error_message = f"Requested object to grab {self.object_to_grab} has more than one position in defined objects: {self.objects[self.object_to_grab]}."
            self.log_error(msg=error_message)
            raise TaskParametrizationError(error_message)

    def verify_tool_calls(self, response: Dict[str, Any]):
        """It is expected that the agent will request:
        1. The tool get_object_positions to get the position of the object to grab.
        2. The tool move_to_point to move to the position of the object to grab.

        Parameters
        ----------
        response : Dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        expected_num_ai_messages = 3
        if len(ai_messages) != expected_num_ai_messages:
            self.log_error(
                msg=f"Expected {expected_num_ai_messages} AI messages, but got {len(ai_messages)}."
            )

        if ai_messages:
            if self._check_tool_calls_num_in_ai_message(ai_messages[0], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[0].tool_calls[0],
                    expected_name="get_object_positions",
                    expected_args={"object_name": self.object_to_grab},
                )

        if len(ai_messages) > 1:
            if self._check_tool_calls_num_in_ai_message(ai_messages[1], expected_num=1):
                obj_to_grab: dict[str, Any] = copy.deepcopy(
                    self.objects[self.object_to_grab][0]
                )
                obj_to_grab.update({"task": "grab"})
                self._check_tool_call(
                    tool_call=ai_messages[1].tool_calls[0],
                    expected_name="move_to_point",
                    expected_args=obj_to_grab,
                )

        if not self.result.errors:
            self.result.success = True


class GrabNotExistingObjectTask(ROS2ToolCallingAgentTask):
    """Task to grab an object that does not exist

    Parameters
    ----------
    objects : Dict[str, List[dict[str, float]]]
        Dictionary containing the object types and their positions. Object type should be passed as singular.
    object_to_grab : str
        Object to grab. Object type should be passed as singular. Object to be grabbed should NOT be defined in the objects argument.
    logger : loggers_type | None, optional
        Logger, by default None

    Examples
    --------
    objects = {
        "banana": [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)],
        "cube": [(0.7, 0.8, 0.9)],
    }
    object_to_grab = "apple"
    """

    complexity = "medium"

    topics_and_types: Dict[str, str] = {
        "/attached_collision_object": "moveit_msgs/msg/AttachedCollisionObject",
        "/camera_image_color": "sensor_msgs/msg/Image",
        "/camera_image_depth": "sensor_msgs/msg/Image",
        "/clock": "rosgraph_msgs/msg/Clock",
        "/collision_object": "moveit_msgs/msg/CollisionObject",
        "/color_camera_info": "sensor_msgs/msg/CameraInfo",
    }

    def __init__(
        self,
        objects: Dict[str, List[dict[str, float]]],
        object_to_grab: str,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(logger=logger)

        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]

        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
            MockGetObjectPositionsTool(
                target_frame="panda_link0",
                source_frame="RGBDCamera5",
                camera_topic="/color_image5",
                depth_topic="/depth_image5",
                camera_info_topic="/color_camera_info5",
                mock_objects=objects,
            ),
            MockMoveToPointTool(manipulator_frame="panda_link0"),
        ]

        self.objects = objects
        self.object_to_grab = object_to_grab
        self._verify_args()

    def get_system_prompt(self) -> str:
        return """
        You are a robotic arm with interfaces to detect and manipulate objects.
        Here are the coordinates information:
        x - front to back (positive is forward)
        y - left to right (positive is right)
        z - up to down (positive is up).
        """

    def get_prompt(self) -> str:
        return f"Grab {self.object_to_grab}."

    def _verify_args(self):
        if self.object_to_grab in self.objects:
            error_message = f"Requested object to grab {self.object_to_grab} is present in defined objects: {self.objects} but should not be."
            self.log_error(msg=error_message)
            raise TaskParametrizationError(error_message)

    def verify_tool_calls(self, response: Dict[str, Any]):
        """It is expected that the agent will request the tool get_object_positions to get the position of the object to grab.
        It is expected that no positions are returned and agent will not request any more tool.

        Parameters
        ----------
        response : Dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        expected_num_ai_messages = 2
        if len(ai_messages) != expected_num_ai_messages:
            self.log_error(
                msg=f"Expected {expected_num_ai_messages} AI messages, but got {len(ai_messages)}."
            )

        if ai_messages:
            if self._check_tool_calls_num_in_ai_message(ai_messages[0], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[0].tool_calls[0],
                    expected_name="get_object_positions",
                    expected_args={"object_name": self.object_to_grab},
                )

        if not self.result.errors:
            self.result.success = True


class MoveExistingObjectLeftTask(ROS2ToolCallingAgentTask):
    """Task to move an existing object to the left.

    Parameters
    ----------
    objects : Dict[str, List[dict[str, float]]]
        Dictionary containing the object types and their positions. Object type should be passed as singular.
    object_to_grab : str
        Object type should be passed as singular. Object to be grabbed should be defined in the objects argument with only one instance (one position).
    logger : loggers_type | None, optional
        Logger, by default None

    Examples
    --------
    objects = {
        "banana": [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)],
        "cube": [(0.7, 0.8, 0.9)],
    }
    object_to_grab = "cube"
    """

    complexity = "medium"

    topics_and_types: Dict[str, str] = {
        "/attached_collision_object": "moveit_msgs/msg/AttachedCollisionObject",
        "/camera_image_color": "sensor_msgs/msg/Image",
        "/camera_image_depth": "sensor_msgs/msg/Image",
        "/clock": "rosgraph_msgs/msg/Clock",
        "/collision_object": "moveit_msgs/msg/CollisionObject",
    }

    def __init__(
        self,
        objects: Dict[str, List[dict[str, float]]],
        object_to_grab: str,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(logger=logger)

        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]

        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
            MockGetObjectPositionsTool(
                target_frame="panda_link0",
                source_frame="RGBDCamera5",
                camera_topic="/color_image5",
                depth_topic="/depth_image5",
                camera_info_topic="/color_camera_info5",
                mock_objects=objects,
            ),
            MockMoveToPointTool(manipulator_frame="panda_link0"),
        ]

        self.objects = objects
        self.object_to_grab = object_to_grab
        self._verify_args()

    def get_system_prompt(self) -> str:
        return """
        You are a robotic arm with interfaces to detect and manipulate objects.
        Here are the coordinates information:
        x - front to back (positive is forward)
        y - left to right (positive is right)
        z - up to down (positive is up).
        Coordinates are in meters.
        """

    def get_prompt(self) -> str:
        return f"Move {self.object_to_grab} 20 cm to the left."

    def _verify_args(self):
        if self.object_to_grab not in self.objects:
            error_message = f"Requested object to grab {self.object_to_grab} is not present in defined objects: {self.objects}."
            self.log_error(msg=error_message)
            raise TaskParametrizationError(error_message)

        if len(self.objects[self.object_to_grab]) > 1:
            error_message = f"Requested object to grab {self.object_to_grab} has more than one position in defined objects: {self.objects[self.object_to_grab]}."
            self.log_error(msg=error_message)
            raise TaskParametrizationError(error_message)

    def verify_tool_calls(self, response: Dict[str, Any]):
        """It is expected that the agent will request:
        1. get_object_positions for the object to grab
        2. move_to_point for the object to grab with the coordinates of the object to grab specified in the task
        3. move_to_point for the the same object but with the task set to "drop" and y coordinate smaller by 0.6

        Parameters
        ----------
        response : Dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        expected_num_ai_messages = 4
        if len(ai_messages) != expected_num_ai_messages:
            self.log_error(
                msg=f"Expected {expected_num_ai_messages} AI messages, but got {len(ai_messages)}."
            )

        if ai_messages:
            if self._check_tool_calls_num_in_ai_message(ai_messages[0], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[0].tool_calls[0],
                    expected_name="get_object_positions",
                    expected_args={"object_name": self.object_to_grab},
                )

        if len(ai_messages) > 1:
            if self._check_tool_calls_num_in_ai_message(ai_messages[1], expected_num=1):
                obj_to_grab: dict[str, Any] = copy.deepcopy(
                    self.objects[self.object_to_grab][0]
                )
                obj_to_grab.update({"task": "grab"})
                self._check_tool_call(
                    tool_call=ai_messages[1].tool_calls[0],
                    expected_name="move_to_point",
                    expected_args=obj_to_grab,
                )

        if len(ai_messages) > 2:
            if self._check_tool_calls_num_in_ai_message(ai_messages[2], expected_num=1):
                obj_to_drop: dict[str, Any] = copy.deepcopy(
                    self.objects[self.object_to_grab][0]
                )
                obj_to_drop.update({"task": "drop"})
                obj_to_drop["y"] = obj_to_drop["y"] - 0.2
                self._check_tool_call(
                    tool_call=ai_messages[2].tool_calls[0],
                    expected_name="move_to_point",
                    expected_args=obj_to_drop,
                )

        if not self.result.errors:
            self.result.success = True


class MoveExistingObjectFrontTask(ROS2ToolCallingAgentTask):
    """Task to move an existing object to the front

    Parameters
    ----------
    objects : Dict[str, List[dict[str, float]]]
        Dictionary containing the object types and their positions. Object type should be passed as singular.
    object_to_grab : str
        Object to grab. Object type should be passed as singular. Object to be grabbed should be defined in the objects argument with only one instance (one position).
    logger : loggers_type | None, optional
        Logger, by default None
    """

    complexity = "medium"

    topics_and_types: Dict[str, str] = {
        "/attached_collision_object": "moveit_msgs/msg/AttachedCollisionObject",
        "/camera_image_color": "sensor_msgs/msg/Image",
        "/camera_image_depth": "sensor_msgs/msg/Image",
    }

    def __init__(
        self,
        objects: Dict[str, List[dict[str, float]]],
        object_to_grab: str,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(logger=logger)

        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]

        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
            MockGetObjectPositionsTool(
                target_frame="panda_link0",
                source_frame="RGBDCamera5",
                camera_topic="/color_image5",
                depth_topic="/depth_image5",
                camera_info_topic="/color_camera_info5",
                mock_objects=objects,
            ),
            MockMoveToPointTool(manipulator_frame="panda_link0"),
        ]

        self.objects = objects
        self.object_to_grab = object_to_grab
        self._verify_args()

    def get_system_prompt(self) -> str:
        return """
        You are a robotic arm with interfaces to detect and manipulate objects.
        Here are the coordinates information:
        x - front to back (positive is forward)
        y - left to right (positive is right)
        z - up to down (positive is up).
        Coordinates are in meters.
        """

    def get_prompt(self) -> str:
        return f"Move {self.object_to_grab} 60 cm to the front."

    def _verify_args(self):
        if self.object_to_grab not in self.objects:
            error_message = f"Requested object to grab {self.object_to_grab} is not present in defined objects: {self.objects}."
            self.log_error(msg=error_message)
            raise TaskParametrizationError(error_message)

        if len(self.objects[self.object_to_grab]) > 1:
            error_message = f"Requested object to grab {self.object_to_grab} has more than one position in defined objects: {self.objects[self.object_to_grab]}."
            self.log_error(msg=error_message)
            raise TaskParametrizationError(error_message)

    def verify_tool_calls(self, response: Dict[str, Any]):
        """It is expected that the agent will request:
        1. get_object_positions for the object to grab
        2. move_to_point for the object to grab with the coordinates of the object to grab specified in the task
        3. move_to_point for the the same object but with the task set to "drop" and x coordinate bigger by 0.6

        Parameters
        ----------
        response : Dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        expected_num_ai_messages = 4
        if len(ai_messages) != expected_num_ai_messages:
            self.log_error(
                msg=f"Expected {expected_num_ai_messages} AI messages, but got {len(ai_messages)}."
            )

        if ai_messages:
            if self._check_tool_calls_num_in_ai_message(ai_messages[0], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[0].tool_calls[0],
                    expected_name="get_object_positions",
                    expected_args={"object_name": self.object_to_grab},
                )

        if len(ai_messages) > 1:
            if self._check_tool_calls_num_in_ai_message(ai_messages[1], expected_num=1):
                obj_to_grab: dict[str, Any] = copy.deepcopy(
                    self.objects[self.object_to_grab][0]
                )
                obj_to_grab.update({"task": "grab"})
                self._check_tool_call(
                    tool_call=ai_messages[1].tool_calls[0],
                    expected_name="move_to_point",
                    expected_args=obj_to_grab,
                )

        if len(ai_messages) > 2:
            if self._check_tool_calls_num_in_ai_message(ai_messages[2], expected_num=1):
                obj_to_drop: dict[str, Any] = copy.deepcopy(
                    self.objects[self.object_to_grab][0]
                )
                obj_to_drop.update({"task": "drop"})
                obj_to_drop["x"] = obj_to_drop["x"] + 0.6
                self._check_tool_call(
                    tool_call=ai_messages[2].tool_calls[0],
                    expected_name="move_to_point",
                    expected_args=obj_to_drop,
                )

        if not self.result.errors:
            self.result.success = True


class SwapObjectsTask(ROS2ToolCallingAgentTask):
    """Task to swap objects

    Parameters
    ----------
    objects : Dict[str, List[Dict[str, float]]]
        Dictionary containing the object types and their positions. Object type should be passed as singular.
    objects_to_swap : List[str]
        Objects to be swapped. Object type should be passed as singular. Objects to be swapped should be defined in the objects argument with only one instance (one position).
    logger : loggers_type | None, optional
        Logger, by default None

    Examples
    --------
    objects = {
        "banana": [(0.1, 0.2, 0.1)],
        "cube": [(0.7, 0.8, 0.1)],
        "apple": [(0.3, 0.4, 0.1), (0.5, 0.6, 0.1)],

    }
    objects_to_swap = ["cube", "banana"]
    """

    complexity = "hard"

    topics_and_types: Dict[str, str] = {
        "/attached_collision_object": "moveit_msgs/msg/AttachedCollisionObject",
        "/camera_image_color": "sensor_msgs/msg/Image",
        "/camera_image_depth": "sensor_msgs/msg/Image",
    }

    def __init__(
        self,
        objects: Dict[str, List[Dict[str, float]]],
        objects_to_swap: List[str],
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(logger=logger)

        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]

        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
            MockGetObjectPositionsTool(
                target_frame="panda_link0",
                source_frame="RGBDCamera5",
                camera_topic="/color_image5",
                depth_topic="/depth_image5",
                camera_info_topic="/color_camera_info5",
                mock_objects=objects,
            ),
            MockMoveToPointTool(manipulator_frame="panda_link0"),
        ]

        self.objects = objects
        self.objects_to_swap = objects_to_swap
        self._verify_args()

    def _verify_args(self):
        for obj in self.objects_to_swap:
            if obj not in self.objects:
                error_message = f"Requested object to swap {obj} is not present in defined objects: {self.objects}."
                self.log_error(msg=error_message)
                raise TaskParametrizationError(error_message)
            if len(self.objects[obj]) != 1:
                error_message = f"Number of positions for object to swap ({obj}) should be equal to 1."
                self.log_error(msg=error_message)
                raise TaskParametrizationError(error_message)
        if len(self.objects_to_swap) != 2:
            error_message = f"Number of requested objects to swap {len(self.objects_to_swap)} should be equal to 2."
            self.log_error(msg=error_message)
            raise TaskParametrizationError(error_message)

    def get_system_prompt(self) -> str:
        return """
        You are a robotic arm with interfaces to detect and manipulate objects in physical environment.
        Here are the coordinates information:
        x - front to back (positive is forward)
        y - left to right (positive is right)
        z - up to down (positive is up).
        Coordinates are in meters.
        """

    def get_prompt(self) -> str:
        return f"Move {self.objects_to_swap[0]} to the initial position of {self.objects_to_swap[1]}, and move {self.objects_to_swap[1]} to the initial position of {self.objects_to_swap[0]}."

    def verify_tool_calls(self, response: Dict[str, Any]):
        """It is expected that the agent will request:
        1. get_object_positions for both objects to be swapped
        2. move_to_point for one object to some temporary position to make place to second object
        3. move_to_point for the second object to the position of the first object
        4. move_to_point for the first object to the position of the second object

        Parameters
        ----------
        response : Dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        actual_tool_calls = [
            tool_call for msg in ai_messages for tool_call in msg.tool_calls
        ]

        expected_num_tool_calls = 8
        if len(actual_tool_calls) < expected_num_tool_calls:
            self.log_error(
                msg=f"Expected at least {expected_num_tool_calls} tool calls, but got {len(actual_tool_calls)}."
            )
            return None

        obj1, obj2 = self.objects_to_swap
        obj1_pos, obj2_pos = self.objects[obj1][0], self.objects[obj2][0]

        # find a temporary position if exists
        move_to_point_args: Sequence[Dict[str, Any]] = [
            call["args"]
            for call in actual_tool_calls
            if call["name"] == "move_to_point"
        ]
        positions = copy.deepcopy(move_to_point_args)
        for arg in positions:
            arg.pop("task")
        temp_position = None
        for position in positions:
            if position != obj1_pos and position != obj2_pos:
                temp_position = position
                break

        if temp_position is None:
            self.log_error(msg="No temporary position found.")
        else:
            get_position_permutations = list(
                permutations(
                    [
                        {"name": "get_object_positions", "args": {"object_name": obj1}},
                        {"name": "get_object_positions", "args": {"object_name": obj2}},
                    ]
                )
            )

            obj_moves_options: List[List[dict[str, Any]]] = [
                [
                    {"name": "move_to_point", "args": {**obj1_pos, "task": "grab"}},
                    {
                        "name": "move_to_point",
                        "args": {**temp_position, "task": "drop"},
                    },
                    {"name": "move_to_point", "args": {**obj2_pos, "task": "grab"}},
                    {"name": "move_to_point", "args": {**obj1_pos, "task": "drop"}},
                    {
                        "name": "move_to_point",
                        "args": {**temp_position, "task": "grab"},
                    },
                    {"name": "move_to_point", "args": {**obj2_pos, "task": "drop"}},
                ],
                [
                    {"name": "move_to_point", "args": {**obj2_pos, "task": "grab"}},
                    {
                        "name": "move_to_point",
                        "args": {**temp_position, "task": "drop"},
                    },
                    {"name": "move_to_point", "args": {**obj1_pos, "task": "grab"}},
                    {"name": "move_to_point", "args": {**obj2_pos, "task": "drop"}},
                    {
                        "name": "move_to_point",
                        "args": {**temp_position, "task": "grab"},
                    },
                    {"name": "move_to_point", "args": {**obj1_pos, "task": "drop"}},
                ],
            ]

            valid_sequences: List[List[dict[str, Any]]] = []
            for get_positions in get_position_permutations:
                for obj_moves in obj_moves_options:
                    valid_sequences.append(list(get_positions) + obj_moves)

            if not any(
                self._matches_sequence(actual_tool_calls, seq)
                for seq in valid_sequences
            ):
                self.log_error(
                    msg="The tool calls are in an invalid sequence for object swapping."
                )

        if not self.result.errors:
            self.result.success = True

    def _matches_sequence(
        self,
        actual_tool_calls_seq: Sequence[ToolCall],
        expected_tool_calls_seq: Sequence[dict[str, Any]],
    ) -> bool:
        """
        Helper method to check if actual tool calls sequence match expected tool calls in terms of sequence and arguments.

        Parameters
        ----------
        actual_tool_calls_seq : Sequence[ToolCall]
            Sequence of tool calls requested by agent.
        expected_tool_calls_seq : Sequence[dict[str, Any]]
            Sequence of expected tool calls.

        Returns
        -------
        bool
            True if actual tool calls sequence matches expected tool calls sequence, False otherwise
        """
        if len(actual_tool_calls_seq) < len(expected_tool_calls_seq):
            return False
        it = iter(actual_tool_calls_seq)
        return all(
            any(call["name"] == e["name"] and call["args"] == e["args"] for call in it)
            for e in expected_tool_calls_seq
        )


class PublishROS2CustomMessageTask(ROS2ToolCallingAgentTask):
    complexity = "easy"
    topics_and_types: Dict[str, str] = {
        "/attached_collision_object": "moveit_msgs/msg/AttachedCollisionObject",
        "/camera_image_color": "sensor_msgs/msg/Image",
        "/camera_image_depth": "sensor_msgs/msg/Image",
        "/clock": "rosgraph_msgs/msg/Clock",
        "/collision_object": "moveit_msgs/msg/CollisionObject",
        "/color_camera_info": "sensor_msgs/msg/CameraInfo",
        "/color_camera_info5": "sensor_msgs/msg/CameraInfo",
        "/depth_camera_info5": "sensor_msgs/msg/CameraInfo",
        "/depth_image5": "sensor_msgs/msg/Image",
        "/to_human": "rai_interfaces/msg/HRIMessage",
    }

    interfaces: Dict[str, Dict[str, Any]] = {
        "moveit_msgs/msg/AttachedCollisionObject": {
            "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
            "link_name": "",
            "object": {
                "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
                "id": "",
                "primitives": [],
                "primitive_poses": [],
                "meshes": [],
                "mesh_poses": [],
                "planes": [],
                "plane_poses": [],
                "operation": 0,
            },
            "touch_links": [],
        },
        "sensor_msgs/msg/Image": {
            "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
            "height": 0,
            "width": 0,
            "encoding": "",
            "is_bigendian": 0,
            "step": 0,
            "data": [],
        },
        "rosgraph_msgs/msg/Clock": {"clock": {"sec": 0, "nanosec": 0}},
        "moveit_msgs/msg/CollisionObject": {
            "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
            "id": "",
            "primitives": [],
            "primitive_poses": [],
            "meshes": [],
            "mesh_poses": [],
            "planes": [],
            "plane_poses": [],
            "operation": 0,
        },
        "sensor_msgs/msg/CameraInfo": {
            "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
            "height": 0,
            "width": 0,
            "distortion_model": "",
            "D": [],
            "K": [0.0] * 9,
            "R": [0.0] * 9,
            "P": [0.0] * 12,
            "binning_x": 0,
            "binning_y": 0,
            "roi": {
                "x_offset": 0,
                "y_offset": 0,
                "height": 0,
                "width": 0,
                "do_rectify": False,
            },
        },
        "rai_interfaces/msg/HRIMessage": {
            "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
            "text": "",
            "images": [],
            "audios": [],
        },
        "rai_interfaces/msg/AudioMessage": {
            "audio": [],
            "sample_rate": 0,
            "channels": 0,
        },
    }

    expected_text = "Hello!"

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]
        interface_strings = {
            msg_type: json.dumps(interface)
            for msg_type, interface in self.interfaces.items()
        }
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=interface_strings),
            MockPublishROS2MessageTool(
                available_topics=list(self.topics_and_types.keys()),
                available_message_types=list(self.topics_and_types.values()),
            ),
        ]

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return "Publish message with text value: 'Hello!' to the /to_human topic. Before publishing check the message type of this topic and it's interface."

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request:
        1. The tool that retrieves the ROS2 topics names and types to recognize what type of message to_human topic has
        2. The tool that retrieves interfaces to check HRIMessage type
        3. The tool to publish message with proper topic, message type and content

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        self.logger.debug(ai_messages)
        if len(ai_messages) != 4:
            self.log_error(
                msg=f"Expected exactly 4 AI messages, but got {len(ai_messages)}."
            )
        if ai_messages:
            if not self._is_ai_message_requesting_get_ros2_topics_and_types(
                ai_messages[0]
            ):
                self.log_error(
                    msg="First AI message did not request ROS2 topics and types correctly."
                )
        if len(ai_messages) > 1:
            if self._check_tool_calls_num_in_ai_message(ai_messages[1], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[1].tool_calls[0],
                    expected_name="get_ros2_message_interface",
                    expected_args={"msg_type": "rai_interfaces/msg/HRIMessage"},
                )

        if len(ai_messages) > 2:
            if self._check_tool_calls_num_in_ai_message(ai_messages[2], expected_num=1):
                expected_message = self.interfaces[
                    "rai_interfaces/msg/HRIMessage"
                ].copy()
                expected_message["text"] = self.expected_text
                self._check_tool_call(
                    tool_call=ai_messages[2].tool_calls[0],
                    expected_name="publish_ros2_message",
                    expected_args={
                        "topic": "/to_human",
                        "message": expected_message,
                        "message_type": "rai_interfaces/msg/HRIMessage",
                    },
                )
        if not self.result.errors:
            self.result.success = True


class PublishROS2AudioMessageTask(ROS2ToolCallingAgentTask):
    complexity = "easy"
    topics_and_types: Dict[str, str] = {
        "/send_audio": "rai_interfaces/msg/AudioMessage",
    }

    interfaces: Dict[str, Dict[str, Any]] = {
        "rai_interfaces/msg/AudioMessage": {
            "audio": [],
            "sample_rate": 0,
            "channels": 0,
        }
    }

    expected_audio = [123, 456, 789]
    expected_sample_rate = 44100
    expected_channels = 2

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]
        interface_strings = {
            msg_type: json.dumps(interface)
            for msg_type, interface in self.interfaces.items()
        }
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=interface_strings),
            MockPublishROS2MessageTool(
                available_topics=list(self.topics_and_types.keys()),
                available_message_types=list(self.topics_and_types.values()),
            ),
        ]

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return (
            "Publish message to the /send_audio topic with audio samples [123, 456, 789], sample rate 44100, "
            "and 2 channels  Before publishing, check the "
            "message type of this topic and its interface."
        )

    def verify_tool_calls(self, response: dict[str, Any]):
        """
        It is expected that the agent will:
        1. Request the tool that retrieves ROS2 topics names and types to determine
           the message type of the /send_audio topic.
        2. Request the tool that retrieves message interfaces to check the AudioMessage type.
        3. Request the tool to publish the message with the correct topic, message type,
           and content.
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        self.logger.debug(ai_messages)
        if len(ai_messages) != 4:
            self.log_error(
                msg=f"Expected exactly 4 AI messages, but got {len(ai_messages)}."
            )
        if ai_messages:
            if not self._is_ai_message_requesting_get_ros2_topics_and_types(
                ai_messages[0]
            ):
                self.log_error(
                    msg="First AI message did not request ROS2 topics and types correctly."
                )
        if len(ai_messages) > 1:
            if self._check_tool_calls_num_in_ai_message(ai_messages[1], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[1].tool_calls[0],
                    expected_name="get_ros2_message_interface",
                    expected_args={"msg_type": "rai_interfaces/msg/AudioMessage"},
                )
        if len(ai_messages) > 2:
            if self._check_tool_calls_num_in_ai_message(ai_messages[2], expected_num=1):
                expected_message = self.interfaces[
                    "rai_interfaces/msg/AudioMessage"
                ].copy()
                expected_message["audio"] = self.expected_audio
                expected_message["sample_rate"] = self.expected_sample_rate
                expected_message["channels"] = self.expected_channels
                self._check_tool_call(
                    tool_call=ai_messages[2].tool_calls[0],
                    expected_name="publish_ros2_message",
                    expected_args={
                        "topic": "/send_audio",
                        "message": expected_message,
                        "message_type": "rai_interfaces/msg/AudioMessage",
                    },
                )
        if not self.result.errors:
            self.result.success = True


class PublishROS2DetectionArrayTask(ROS2ToolCallingAgentTask):
    complexity = "easy"
    topics_and_types: Dict[str, str] = {
        "/send_detections": "rai_interfaces/msg/RAIDetectionArray",
    }
    interfaces: Dict[str, Dict[str, Any]] = {
        "rai_interfaces/msg/RAIDetectionArray": {
            "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
            "detections": [],
            "detection_classes": [],
        }
    }

    expected_header: Dict[str, Any] = {
        "stamp": {"sec": 0, "nanosec": 0},
        "frame_id": "camera",
    }
    expected_detections: List[Any] = [
        {
            "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "camera"},
            "results": [],
            "bbox": {
                "center": {"x": 320.0, "y": 240.0},
                "size": {"x": 50.0, "y": 50.0},
            },
        }
    ]
    expected_detection_classes: List[str] = ["person", "car"]

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]
        interface_strings = {
            msg_type: json.dumps(interface)
            for msg_type, interface in self.interfaces.items()
        }
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=interface_strings),
            MockPublishROS2MessageTool(
                available_topics=list(self.topics_and_types.keys()),
                available_message_types=list(self.topics_and_types.values()),
            ),
        ]

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return (
            "Publish a detection message to the /send_detections topic. The message should have a header "
            f"with frame_id 'camera', one detection: {self.expected_detections}, and detection classes ['person', 'car']. "
            "Before publishing, check the message type of this topic and its interface."
        )

    def verify_tool_calls(self, response: dict[str, Any]):
        """
        Expected behavior:
        1. The agent should request the tool that retrieves ROS2 topics names and types to determine
           the message type of /send_detections.
        2. The agent should request the tool that retrieves the message interface for 'rai_interfaces/msg/RAIDetectionArray'.
        3. The agent should call the publish tool with the correct topic, message type, and content.
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            msg for msg in messages if isinstance(msg, AIMessage)
        ]
        self.logger.debug(ai_messages)
        if len(ai_messages) != 4:
            self.log_error(
                msg=f"Expected exactly 4 AI messages, but got {len(ai_messages)}."
            )
        if ai_messages:
            if not self._is_ai_message_requesting_get_ros2_topics_and_types(
                ai_messages[0]
            ):
                self.log_error(
                    msg="First AI message did not request ROS2 topics and types correctly."
                )
        if len(ai_messages) > 1:
            if self._check_tool_calls_num_in_ai_message(ai_messages[1], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[1].tool_calls[0],
                    expected_name="get_ros2_message_interface",
                    expected_args={"msg_type": "rai_interfaces/msg/RAIDetectionArray"},
                )
        if len(ai_messages) > 2:
            if self._check_tool_calls_num_in_ai_message(ai_messages[2], expected_num=1):
                expected_message = self.interfaces[
                    "rai_interfaces/msg/RAIDetectionArray"
                ].copy()
                expected_message["header"] = self.expected_header
                expected_message["detections"] = self.expected_detections
                expected_message["detection_classes"] = self.expected_detection_classes
                self._check_tool_call(
                    tool_call=ai_messages[2].tool_calls[0],
                    expected_name="publish_ros2_message",
                    expected_args={
                        "topic": "/send_detections",
                        "message": expected_message,
                        "message_type": "rai_interfaces/msg/RAIDetectionArray",
                    },
                )
        if not self.result.errors:
            self.result.success = True


class CallROS2ManipulatorMoveToServiceTask(ROS2ToolCallingAgentTask):
    complexity = "easy"
    services_and_types: Dict[str, str] = {
        "/manipulator_move_to": "rai_interfaces/srv/ManipulatorMoveTo",
        "/rai_ros2_ari_connector_46b5f901d7cc/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
        "/rai_ros2_ari_connector_46b5f901d7cc/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
        "/rai_ros2_ari_connector_46b5f901d7cc/get_parameters": "rcl_interfaces/srv/GetParameters",
        "/rai_ros2_ari_connector_46b5f901d7cc/list_parameters": "rcl_interfaces/srv/ListParameters",
        "/rai_ros2_ari_connector_46b5f901d7cc/set_parameters": "rcl_interfaces/srv/SetParameters",
        "/rai_ros2_ari_connector_46b5f901d7cc/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
        "/tf2_frames": "tf2_msgs/srv/FrameGraph",
    }
    interfaces: Dict[str, Dict[str, Any]] = {
        "rai_interfaces/srv/ManipulatorMoveTo": {
            "request": {
                "initial_gripper_state": False,
                "final_gripper_state": False,
                "target_pose": {
                    "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
                    "pose": {
                        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    },
                },
            },
            "response": {"success": False},
        },
        "rcl_interfaces/srv/DescribeParameters": {
            "request": {
                "names": [],
                "include_default_value": False,
            },
            "response": {
                "descriptors": [],
            },
        },
        "rcl_interfaces/srv/GetParameterTypes": {
            "request": {"names": []},
            "response": {"types": []},
        },
        "rcl_interfaces/srv/GetParameters": {
            "request": {"names": []},
            "response": {"values": []},
        },
        "rcl_interfaces/srv/ListParameters": {
            "request": {
                "prefixes": [],
                "depth": 0,
            },
            "response": {
                "names": [],
                "prefixes": [],
            },
        },
        "rcl_interfaces/srv/SetParameters": {
            "request": {"parameters": []},
            "response": {"results": []},
        },
        "rcl_interfaces/srv/SetParametersAtomically": {
            "request": {"parameters": []},
            "response": {"result": {"successful": False, "reason": ""}},
        },
        "tf2_msgs/srv/FrameGraph": {
            "request": {},
            "response": {"frame_yaml": ""},
        },
    }

    expected_initial_gripper_state = True
    expected_final_gripper_state = False
    expected_target_pose: Dict[str, Dict[str, Any]] = {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "world"},
        "pose": {
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        service_strings = [
            f"service: {service}\ntype: {srv_type}\n"
            for service, srv_type in self.services_and_types.items()
        ]
        interface_strings = {
            srv_type: json.dumps(interface)
            for srv_type, interface in self.interfaces.items()
        }
        self.expected_tools: List[BaseTool] = [
            MockGetROS2ServicesNamesAndTypesTool(
                mock_service_names_and_types=service_strings
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=interface_strings),
            MockCallROS2ServiceTool(
                available_services=list(self.services_and_types.keys()),
                available_service_types=list(self.services_and_types.values()),
            ),
        ]

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return f"Call service /manipulator_move_to with a target_pose: {self.expected_target_pose}. Before publishing, check the message type of this topic and its interface."

    def verify_tool_calls(self, response: dict[str, Any]):
        """
        It is expected that the agent will request:
        1. The tool to call the ROS2 service with proper service name, service type, and request arguments.

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent.
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        self.logger.debug(ai_messages)
        if len(ai_messages) != 2:
            self.log_error(
                msg=f"Expected exactly 2 AI messages, but got {len(ai_messages)}."
            )

        if ai_messages:
            if self._check_tool_calls_num_in_ai_message(ai_messages[0], expected_num=1):
                expected_request = self.interfaces[
                    "rai_interfaces/srv/ManipulatorMoveTo"
                ]["request"].copy()
                expected_request["initial_gripper_state"] = (
                    self.expected_initial_gripper_state
                )
                expected_request["final_gripper_state"] = (
                    self.expected_final_gripper_state
                )
                expected_request["target_pose"] = self.expected_target_pose

                self._check_tool_call(
                    tool_call=ai_messages[0].tool_calls[0],
                    expected_name="call_ros2_service",
                    expected_args={
                        "service_name": "/manipulator_move_to",
                        "service_type": "rai_interfaces/srv/ManipulatorMoveTo",
                        "service_args": expected_request,
                    },
                )

        if not self.result.errors:
            self.result.success = True


class CallGroundedSAMSegmentTask(ROS2ToolCallingAgentTask):
    complexity = "easy"
    services_and_types: Dict[str, str] = {
        "/grounded_sam_segment": "rai_interfaces/srv/RAIGroundedSam",
    }
    interfaces: Dict[str, Dict[str, Any]] = {
        "rai_interfaces/srv/RAIGroundedSam": {
            "request": {
                "detections": {
                    "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
                    "detections": [],
                },
                "source_img": {
                    "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
                    "height": 0,
                    "width": 0,
                    "encoding": "",
                    "is_bigendian": 0,
                    "step": 0,
                    "data": [],
                },
            },
            "response": {
                "masks": [],
            },
        }
    }

    expected_detections: Dict[str, Any] = {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "camera_frame"},
        "detections": [],
    }
    expected_source_img: Dict[str, Any] = {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "camera_frame"},
        "height": 480,
        "width": 640,
        "encoding": "rgb8",
        "is_bigendian": 0,
        "step": 1920,
        "data": [],
    }

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        service_strings = [
            f"service: {service}\ntype: {srv_type}\n"
            for service, srv_type in self.services_and_types.items()
        ]
        interface_strings = {
            srv_type: json.dumps(interface)
            for srv_type, interface in self.interfaces.items()
        }
        self.expected_tools: List[BaseTool] = [
            MockGetROS2ServicesNamesAndTypesTool(
                mock_service_names_and_types=service_strings
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=interface_strings),
            MockCallROS2ServiceTool(
                available_services=list(self.services_and_types.keys()),
                available_service_types=list(self.services_and_types.values()),
            ),
        ]

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return (
            "Call service /grounded_sam_segment with detections from frame 'camera_frame' and "
            "an RGB image of size 640x480. Before calling, look up the service type and its message structure."
        )

    def verify_tool_calls(self, response: dict[str, Any]):
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        self.logger.debug(ai_messages)
        if len(ai_messages) != 2:
            self.log_error(
                msg=f"Expected exactly 2 AI messages, but got {len(ai_messages)}."
            )

        if ai_messages:
            if self._check_tool_calls_num_in_ai_message(ai_messages[0], expected_num=1):
                expected_request = self.interfaces["rai_interfaces/srv/RAIGroundedSam"][
                    "request"
                ].copy()
                expected_request["detections"] = self.expected_detections
                expected_request["source_img"] = self.expected_source_img

                self._check_tool_call(
                    tool_call=ai_messages[0].tool_calls[0],
                    expected_name="call_ros2_service",
                    expected_args={
                        "service_name": "/grounded_sam_segment",
                        "service_type": "rai_interfaces/srv/RAIGroundedSam",
                        "service_args": expected_request,
                    },
                )

        if not self.result.errors:
            self.result.success = True


class CallGroundingDinoClassifyTask(ROS2ToolCallingAgentTask):
    complexity = "easy"
    services_and_types: Dict[str, str] = {
        "/grounding_dino_classify": "rai_interfaces/srv/RAIGroundingDino",
    }
    interfaces: Dict[str, Dict[str, Any]] = {
        "rai_interfaces/srv/RAIGroundingDino": {
            "request": {
                "classes": "",
                "box_threshold": 0.0,
                "text_threshold": 0.0,
                "source_img": {
                    "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
                    "height": 0,
                    "width": 0,
                    "encoding": "",
                    "is_bigendian": 0,
                    "step": 0,
                    "data": [],
                },
            },
            "response": {
                "detections": {
                    "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
                    "detections": [],
                },
            },
        }
    }

    expected_classes = "bottle, book, chair"
    expected_box_threshold = 0.4
    expected_text_threshold = 0.25
    expected_source_img: Dict[str, Any] = {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "camera_frame"},
        "height": 480,
        "width": 640,
        "encoding": "rgb8",
        "is_bigendian": 0,
        "step": 1920,
        "data": [],
    }

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        service_strings = [
            f"service: {service}\ntype: {srv_type}\n"
            for service, srv_type in self.services_and_types.items()
        ]
        interface_strings = {
            srv_type: json.dumps(interface)
            for srv_type, interface in self.interfaces.items()
        }
        self.expected_tools: List[BaseTool] = [
            MockGetROS2ServicesNamesAndTypesTool(
                mock_service_names_and_types=service_strings
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=interface_strings),
            MockCallROS2ServiceTool(
                available_services=list(self.services_and_types.keys()),
                available_service_types=list(self.services_and_types.values()),
            ),
        ]

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return (
            f"Call the service /grounding_dino_classify with the following arguments: "
            f"classes='{self.expected_classes}', box_threshold={self.expected_box_threshold}, "
            f"text_threshold={self.expected_text_threshold}, and a 640x480 RGB image from frame 'camera_frame'. "
            f"Before calling, look up the service type and its message structure."
        )

    def verify_tool_calls(self, response: dict[str, Any]):
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        self.logger.debug(ai_messages)
        if len(ai_messages) != 2:
            self.log_error(
                msg=f"Expected exactly 2 AI messages, but got {len(ai_messages)}."
            )

        if ai_messages:
            if self._check_tool_calls_num_in_ai_message(ai_messages[0], expected_num=1):
                expected_request = self.interfaces[
                    "rai_interfaces/srv/RAIGroundingDino"
                ]["request"].copy()
                expected_request["classes"] = self.expected_classes
                expected_request["box_threshold"] = self.expected_box_threshold
                expected_request["text_threshold"] = self.expected_text_threshold
                expected_request["source_img"] = self.expected_source_img

                self._check_tool_call(
                    tool_call=ai_messages[0].tool_calls[0],
                    expected_name="call_ros2_service",
                    expected_args={
                        "service_name": "/grounding_dino_classify",
                        "service_type": "rai_interfaces/srv/RAIGroundingDino",
                        "service_args": expected_request,
                    },
                )

        if not self.result.errors:
            self.result.success = True


class CallGetLogDigestTask(ROS2ToolCallingAgentTask):
    complexity = "easy"
    services_and_types: Dict[str, str] = {
        "/get_log_digest": "rai_interfaces/srv/StringList",
    }
    interfaces: Dict[str, Dict[str, Any]] = {
        "rai_interfaces/srv/StringList": {
            "request": {},
            "response": {
                "success": False,
                "string_list": [],
            },
        }
    }

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        service_strings = [
            f"service: {service}\ntype: {srv_type}\n"
            for service, srv_type in self.services_and_types.items()
        ]
        interface_strings = {
            srv_type: json.dumps(interface)
            for srv_type, interface in self.interfaces.items()
        }
        self.expected_tools: List[BaseTool] = [
            MockGetROS2ServicesNamesAndTypesTool(
                mock_service_names_and_types=service_strings
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=interface_strings),
            MockCallROS2ServiceTool(
                available_services=list(self.services_and_types.keys()),
                available_service_types=list(self.services_and_types.values()),
            ),
        ]

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return (
            "Call the service /get_log_digest to retrieve a list of log strings. "
            "Before calling, look up the service type and its message structure."
            "No request arguments are needed."
        )

    def verify_tool_calls(self, response: dict[str, Any]):
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        self.logger.debug(ai_messages)
        if len(ai_messages) != 2:
            self.log_error(
                msg=f"Expected exactly 2 AI messages, but got {len(ai_messages)}."
            )

        if ai_messages:
            if self._check_tool_calls_num_in_ai_message(ai_messages[0], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[0].tool_calls[0],
                    expected_name="call_ros2_service",
                    expected_args={
                        "service_name": "/get_log_digest",
                        "service_type": "rai_interfaces/srv/StringList",
                        "service_args": {},
                    },
                )

        if not self.result.errors:
            self.result.success = True


class CallVectorStoreRetrievalTask(ROS2ToolCallingAgentTask):
    complexity = "easy"
    services_and_types: Dict[str, str] = {
        "rai_whoami_documentation_service": "rai_interfaces/srv/VectorStoreRetrieval",
    }
    interfaces: Dict[str, Dict[str, Any]] = {
        "rai_interfaces/srv/VectorStoreRetrieval": {
            "request": {
                "query": "",
            },
            "response": {
                "success": False,
                "message": "",
                "documents": [],
                "scores": [],
            },
        }
    }

    expected_query = "What is the purpose of this robot?"

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        service_strings = [
            f"service: {service}\ntype: {srv_type}\n"
            for service, srv_type in self.services_and_types.items()
        ]
        interface_strings = {
            srv_type: json.dumps(interface)
            for srv_type, interface in self.interfaces.items()
        }
        self.expected_tools: List[BaseTool] = [
            MockGetROS2ServicesNamesAndTypesTool(
                mock_service_names_and_types=service_strings
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=interface_strings),
            MockCallROS2ServiceTool(
                available_services=list(self.services_and_types.keys()),
                available_service_types=list(self.services_and_types.values()),
            ),
        ]

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return (
            f"Call the service rai_whoami_documentation_service with the query: '{self.expected_query}'. "
            "Before calling, look up the service type and its message structure."
        )

    def verify_tool_calls(self, response: dict[str, Any]):
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        self.logger.debug(ai_messages)
        if len(ai_messages) != 2:
            self.log_error(
                msg=f"Expected exactly 2 AI messages, but got {len(ai_messages)}."
            )

        if ai_messages:
            if self._check_tool_calls_num_in_ai_message(ai_messages[0], expected_num=1):
                expected_request = {
                    "query": self.expected_query,
                }

                self._check_tool_call(
                    tool_call=ai_messages[0].tool_calls[0],
                    expected_name="call_ros2_service",
                    expected_args={
                        "service_name": "rai_whoami_documentation_service",
                        "service_type": "rai_interfaces/srv/VectorStoreRetrieval",
                        "service_args": expected_request,
                    },
                )

        if not self.result.errors:
            self.result.success = True


class CallWhatISeeTask(ROS2ToolCallingAgentTask):
    complexity = "easy"
    services_and_types: Dict[str, str] = {
        "rai/whatisee/get": "rai_interfaces/srv/WhatISee",
    }
    interfaces: Dict[str, Dict[str, Any]] = {
        "rai_interfaces/srv/WhatISee": {
            "request": {
                "observations": [],
                "perception_source": "",
                "image": {
                    "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
                    "height": 0,
                    "width": 0,
                    "encoding": "",
                    "is_bigendian": 0,
                    "step": 0,
                    "data": [],
                },
                "pose": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            },
            "response": {},  # Assuming empty response structure
        }
    }

    expected_observations = ["table", "cup", "notebook"]
    expected_perception_source = "front_camera"
    expected_image: Dict[str, Any] = {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "camera_frame"},
        "height": 480,
        "width": 640,
        "encoding": "rgb8",
        "is_bigendian": 0,
        "step": 1920,
        "data": [],
    }
    expected_pose = {
        "position": {"x": 1.0, "y": 2.0, "z": 0.5},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        service_strings = [
            f"service: {service}\ntype: {srv_type}\n"
            for service, srv_type in self.services_and_types.items()
        ]
        interface_strings = {
            srv_type: json.dumps(interface)
            for srv_type, interface in self.interfaces.items()
        }
        self.expected_tools: List[BaseTool] = [
            MockGetROS2ServicesNamesAndTypesTool(
                mock_service_names_and_types=service_strings
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=interface_strings),
            MockCallROS2ServiceTool(
                available_services=list(self.services_and_types.keys()),
                available_service_types=list(self.services_and_types.values()),
            ),
        ]

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return (
            f"Call the service rai/whatisee/get using the WhatISee interface. "
            f"Pass in observations {self.expected_observations}, source '{self.expected_perception_source}', "
            f"a 640x480 RGB image from 'camera_frame', and a pose at position {self.expected_pose['position']}."
            "Before calling, look up the service type and its message structure."
        )

    def verify_tool_calls(self, response: dict[str, Any]):
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        self.logger.debug(ai_messages)
        if len(ai_messages) != 2:
            self.log_error(
                msg=f"Expected exactly 2 AI messages, but got {len(ai_messages)}."
            )

        if ai_messages:
            if self._check_tool_calls_num_in_ai_message(ai_messages[0], expected_num=1):
                expected_request: Dict[str, Any] = {
                    "observations": self.expected_observations,
                    "perception_source": self.expected_perception_source,
                    "image": self.expected_image,
                    "pose": self.expected_pose,
                }

                self._check_tool_call(
                    tool_call=ai_messages[0].tool_calls[0],
                    expected_name="call_ros2_service",
                    expected_args={
                        "service_name": "rai/whatisee/get",
                        "service_type": "rai_interfaces/srv/WhatISee",
                        "service_args": expected_request,
                    },
                )

        if not self.result.errors:
            self.result.success = True


class CallROS2CustomActionTask(ROS2ToolCallingAgentTask):
    complexity = "easy"
    actions_and_types: Dict[str, str] = {
        # custom action
        "/perform_task": "rai_interfaces/action/Task",
        # some sample actions
        "/execute_trajectory": "moveit_msgs/action/ExecuteTrajectory",
        "/move_action": "moveit_msgs/action/MoveGroup",
        "/follow_joint_trajectory": "control_msgs/action/FollowJointTrajectory",
        "/gripper_cmd": "control_msgs/action/GripperCommand",
    }
    interfaces: Dict[str, Dict[str, Any]] = {
        "rai_interfaces/action/Task": {
            "goal": {
                "task": "",
                "description": "",
                "priority": "",
            },
            "result": {
                "success": False,
                "report": "",
            },
            "feedback": {
                "current_status": "",
            },
        },
        "moveit_msgs/action/ExecuteTrajectory": {
            "goal": {
                "trajectory": {},
            },
            "result": {
                "error_code": 0,
                "error_message": "",
            },
            "feedback": {
                "state": "",
            },
        },
        "moveit_msgs/action/MoveGroup": {
            "goal": {
                "planning_options": {},
                "request": {},
            },
            "result": {
                "error_code": 0,
                "trajectory": {},
            },
            "feedback": {
                "state": "",
            },
        },
        "control_msgs/action/FollowJointTrajectory": {
            "goal": {
                "trajectory": {},
            },
            "result": {
                "error_code": 0,
                "error_string": "",
            },
            "feedback": {
                "joint_names": [],
                "actual": {},
                "desired": {},
                "error": {},
            },
        },
        "control_msgs/action/GripperCommand": {
            "goal": {
                "command": {
                    "position": 0.0,
                    "max_effort": 0.0,
                },
            },
            "result": {
                "position": 0.0,
                "effort": 0.0,
                "reached_goal": False,
            },
            "feedback": {
                "position": 0.0,
                "effort": 0.0,
            },
        },
    }
    expected_task = "Where are you?"
    expected_description = ""
    expected_priority = "10"

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        action_strings = [
            f"action: {action}\ntype: {act_type}\n"
            for action, act_type in self.actions_and_types.items()
        ]
        interface_strings = {
            act_type: json.dumps(interface)
            for act_type, interface in self.interfaces.items()
        }
        self.expected_tools: List[BaseTool] = [
            MockGetROS2ActionsNamesAndTypesTool(
                mock_actions_names_and_types=action_strings
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=interface_strings),
            MockStartROS2ActionTool(
                available_actions=list(self.actions_and_types.keys()),
                available_action_types=list(self.actions_and_types.values()),
            ),
        ]

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return "Call action /perform_task with the provided goal values: {priority: 10, description: '', task: 'Where are you?'}"

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request:
        1. The tool to start action with proper action name, message type and content"
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) != 2:
            self.log_error(
                msg=f"Expected exactly 2 AI messages, but got {len(ai_messages)}."
            )

        if ai_messages:
            if self._check_tool_calls_num_in_ai_message(ai_messages[0], expected_num=1):
                expected_goal = self.interfaces["rai_interfaces/action/Task"][
                    "goal"
                ].copy()
                expected_goal["task"] = self.expected_task
                expected_goal["description"] = self.expected_description
                expected_goal["priority"] = self.expected_priority

                self._check_tool_call(
                    tool_call=ai_messages[0].tool_calls[0],
                    expected_name="start_ros2_action",
                    expected_args={
                        "action_name": "/perform_task",
                        "action_type": "rai_interfaces/action/Task",
                        "action_args": expected_goal,
                    },
                )

        if not self.result.errors:
            self.result.success = True


ROBOT_NAVIGATION_SYSTEM_PROMPT = """You are an autonomous robot connected to ros2 environment. Your main goal is to fulfill the user's requests.
    Do not make assumptions about the environment you are currently in.
    You can use ros2 topics, services and actions to operate.

    <rule> As a first step check transforms by getting 1 message from /tf topic </rule>
    <rule> use /cmd_vel topic very carefully. Obstacle detection works only with nav2 stack, so be careful when it is not used. </rule>>
    <rule> be patient with running ros2 actions. usually the take some time to run. </rule>
    <rule> Always check your transform before and after you perform ros2 actions, so that you can verify if it worked. </rule>

    Navigation tips:
    - it's good to start finding objects by rotating, then navigating to some diverse location with occasional rotations. Remember to frequency detect objects.
    - for driving forward/backward or to some coordinates, ros2 actions are better.
    - for driving for some specific time or in specific manner (like shaper or turns) it good to use /cmd_vel topic
    - you are currently unable to read map or point-cloud, so please avoid subscribing to such topics.
    - if you are asked to drive towards some object, it's good to:
        1. check the camera image and verify if objects can be seen
        2. if only driving forward is required, do it
        3. if obstacle avoidance might be required, use ros2 actions navigate_*, but first check your current position, then very accurately estimate the goal pose.
    - it is good to verify using given information if the robot is not stuck
    - navigation actions sometimes fail. Their output can be read from rosout. You can also tell if they partially worked by checking the robot position and rotation.
    - before using any ros2 interfaces, always make sure to check you are using the right interface
    - processing camera image takes 5-10s. Take it into account that if the robot is moving, the information can be outdated. Handle it by good planning of your movements.
    - you are encouraged to use wait tool in between checking the status of actions
    - to find some object navigate around and check the surrounding area
    - when the goal is accomplished please make sure to cancel running actions
    - when you reach the navigation goal - double check if you reached it by checking the current position
    - if you detect collision, please stop operation

    - you will be given your camera image description. Based on this information you can reason about positions of objects.
    - be careful and aboid obstacles

    Here are the corners of your environment:
    (-2.76,9.04, 0.0),
    (4.62, 9.07, 0.0),
    (-2.79, -3.83, 0.0),
    (4.59, -3.81, 0.0)

    This is location of places:
    Kitchen:
    (2.06, -0.23, 0.0),
    (2.07, -1.43, 0.0),
    (-2.44, -0.38, 0.0),
    (-2.56, -1.47, 0.0)

    # Living room:
    (-2.49, 1.87, 0.0),
    (-2.50, 5.49, 0.0),
    (0.79, 5.73, 0.0),
    (0.92, 1.01, 0.0)

    Before starting anything, make sure to load available topics, services and actions.
    """
NAVIGATION_SERVICES_AND_TYPES: Dict[str, str] = {
    "/assisted_teleop/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/assisted_teleop/_action/get_result": "nav2_msgs/action/AssistedTeleop_GetResult",
    "/assisted_teleop/_action/send_goal": "nav2_msgs/action/AssistedTeleop_SendGoal",
    "/backup/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/backup/_action/get_result": "nav2_msgs/action/BackUp_GetResult",
    "/backup/_action/send_goal": "nav2_msgs/action/BackUp_SendGoal",
    "/behavior_server/change_state": "lifecycle_msgs/srv/ChangeState",
    "/behavior_server/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/behavior_server/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/behavior_server/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/behavior_server/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/behavior_server/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/behavior_server/get_state": "lifecycle_msgs/srv/GetState",
    "/behavior_server/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/behavior_server/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/behavior_server/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/behavior_server/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/bt_navigator/change_state": "lifecycle_msgs/srv/ChangeState",
    "/bt_navigator/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/bt_navigator/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/bt_navigator/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/bt_navigator/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/bt_navigator/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/bt_navigator/get_state": "lifecycle_msgs/srv/GetState",
    "/bt_navigator/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/bt_navigator/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/bt_navigator/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/bt_navigator/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/bt_navigator_navigate_through_poses_rclcpp_node/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/bt_navigator_navigate_through_poses_rclcpp_node/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/bt_navigator_navigate_through_poses_rclcpp_node/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/bt_navigator_navigate_through_poses_rclcpp_node/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/bt_navigator_navigate_through_poses_rclcpp_node/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/bt_navigator_navigate_through_poses_rclcpp_node/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/bt_navigator_navigate_to_pose_rclcpp_node/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/bt_navigator_navigate_to_pose_rclcpp_node/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/bt_navigator_navigate_to_pose_rclcpp_node/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/bt_navigator_navigate_to_pose_rclcpp_node/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/bt_navigator_navigate_to_pose_rclcpp_node/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/bt_navigator_navigate_to_pose_rclcpp_node/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/compute_path_through_poses/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/compute_path_through_poses/_action/get_result": "nav2_msgs/action/ComputePathThroughPoses_GetResult",
    "/compute_path_through_poses/_action/send_goal": "nav2_msgs/action/ComputePathThroughPoses_SendGoal",
    "/compute_path_to_pose/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/compute_path_to_pose/_action/get_result": "nav2_msgs/action/ComputePathToPose_GetResult",
    "/compute_path_to_pose/_action/send_goal": "nav2_msgs/action/ComputePathToPose_SendGoal",
    "/controller_server/change_state": "lifecycle_msgs/srv/ChangeState",
    "/controller_server/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/controller_server/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/controller_server/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/controller_server/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/controller_server/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/controller_server/get_state": "lifecycle_msgs/srv/GetState",
    "/controller_server/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/controller_server/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/controller_server/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/controller_server/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/drive_on_heading/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/drive_on_heading/_action/get_result": "nav2_msgs/action/DriveOnHeading_GetResult",
    "/drive_on_heading/_action/send_goal": "nav2_msgs/action/DriveOnHeading_SendGoal",
    "/follow_path/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/follow_path/_action/get_result": "nav2_msgs/action/FollowPath_GetResult",
    "/follow_path/_action/send_goal": "nav2_msgs/action/FollowPath_SendGoal",
    "/follow_waypoints/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/follow_waypoints/_action/get_result": "nav2_msgs/action/FollowWaypoints_GetResult",
    "/follow_waypoints/_action/send_goal": "nav2_msgs/action/FollowWaypoints_SendGoal",
    "/global_costmap/clear_around_global_costmap": "nav2_msgs/srv/ClearCostmapAroundRobot",
    "/global_costmap/clear_entirely_global_costmap": "nav2_msgs/srv/ClearEntireCostmap",
    "/global_costmap/clear_except_global_costmap": "nav2_msgs/srv/ClearCostmapExceptRegion",
    "/global_costmap/get_costmap": "nav2_msgs/srv/GetCostmap",
    "/global_costmap/global_costmap/change_state": "lifecycle_msgs/srv/ChangeState",
    "/global_costmap/global_costmap/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/global_costmap/global_costmap/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/global_costmap/global_costmap/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/global_costmap/global_costmap/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/global_costmap/global_costmap/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/global_costmap/global_costmap/get_state": "lifecycle_msgs/srv/GetState",
    "/global_costmap/global_costmap/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/global_costmap/global_costmap/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/global_costmap/global_costmap/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/global_costmap/global_costmap/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/grounded_sam/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/grounded_sam/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/grounded_sam/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/grounded_sam/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/grounded_sam/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/grounded_sam/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/grounded_sam_segment": "rai_interfaces/srv/RAIGroundedSam",
    "/grounding_dino/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/grounding_dino/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/grounding_dino/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/grounding_dino/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/grounding_dino/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/grounding_dino/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/grounding_dino_classify": "rai_interfaces/srv/RAIGroundingDino",
    "/is_path_valid": "nav2_msgs/srv/IsPathValid",
    "/launch_ros_138640/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/launch_ros_138640/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/launch_ros_138640/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/launch_ros_138640/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/launch_ros_138640/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/launch_ros_138640/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/lifecycle_manager_navigation/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/lifecycle_manager_navigation/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/lifecycle_manager_navigation/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/lifecycle_manager_navigation/is_active": "std_srvs/srv/Trigger",
    "/lifecycle_manager_navigation/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/lifecycle_manager_navigation/manage_nodes": "nav2_msgs/srv/ManageLifecycleNodes",
    "/lifecycle_manager_navigation/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/lifecycle_manager_navigation/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/lifecycle_manager_slam/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/lifecycle_manager_slam/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/lifecycle_manager_slam/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/lifecycle_manager_slam/is_active": "std_srvs/srv/Trigger",
    "/lifecycle_manager_slam/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/lifecycle_manager_slam/manage_nodes": "nav2_msgs/srv/ManageLifecycleNodes",
    "/lifecycle_manager_slam/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/lifecycle_manager_slam/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/local_costmap/clear_around_local_costmap": "nav2_msgs/srv/ClearCostmapAroundRobot",
    "/local_costmap/clear_entirely_local_costmap": "nav2_msgs/srv/ClearEntireCostmap",
    "/local_costmap/clear_except_local_costmap": "nav2_msgs/srv/ClearCostmapExceptRegion",
    "/local_costmap/get_costmap": "nav2_msgs/srv/GetCostmap",
    "/local_costmap/local_costmap/change_state": "lifecycle_msgs/srv/ChangeState",
    "/local_costmap/local_costmap/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/local_costmap/local_costmap/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/local_costmap/local_costmap/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/local_costmap/local_costmap/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/local_costmap/local_costmap/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/local_costmap/local_costmap/get_state": "lifecycle_msgs/srv/GetState",
    "/local_costmap/local_costmap/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/local_costmap/local_costmap/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/local_costmap/local_costmap/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/local_costmap/local_costmap/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/map_saver/change_state": "lifecycle_msgs/srv/ChangeState",
    "/map_saver/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/map_saver/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/map_saver/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/map_saver/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/map_saver/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/map_saver/get_state": "lifecycle_msgs/srv/GetState",
    "/map_saver/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/map_saver/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/map_saver/save_map": "nav2_msgs/srv/SaveMap",
    "/map_saver/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/map_saver/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/nav2_container/_container/list_nodes": "composition_interfaces/srv/ListNodes",
    "/nav2_container/_container/load_node": "composition_interfaces/srv/LoadNode",
    "/nav2_container/_container/unload_node": "composition_interfaces/srv/UnloadNode",
    "/navigate_through_poses/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/navigate_through_poses/_action/get_result": "nav2_msgs/action/NavigateThroughPoses_GetResult",
    "/navigate_through_poses/_action/send_goal": "nav2_msgs/action/NavigateThroughPoses_SendGoal",
    "/navigate_to_pose/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/navigate_to_pose/_action/get_result": "nav2_msgs/action/NavigateToPose_GetResult",
    "/navigate_to_pose/_action/send_goal": "nav2_msgs/action/NavigateToPose_SendGoal",
    "/o3de_ros2_node/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/o3de_ros2_node/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/o3de_ros2_node/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/o3de_ros2_node/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/o3de_ros2_node/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/o3de_ros2_node/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/planner_server/change_state": "lifecycle_msgs/srv/ChangeState",
    "/planner_server/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/planner_server/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/planner_server/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/planner_server/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/planner_server/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/planner_server/get_state": "lifecycle_msgs/srv/GetState",
    "/planner_server/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/planner_server/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/planner_server/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/planner_server/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/rai_ros2_ari_connector_b6ed00ab6356/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/rai_ros2_ari_connector_b6ed00ab6356/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/rai_ros2_ari_connector_b6ed00ab6356/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/rai_ros2_ari_connector_b6ed00ab6356/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/rai_ros2_ari_connector_b6ed00ab6356/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/rai_ros2_ari_connector_b6ed00ab6356/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/slam_toolbox/clear_changes": "slam_toolbox/srv/Clear",
    "/slam_toolbox/clear_queue": "slam_toolbox/srv/ClearQueue",
    "/slam_toolbox/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/slam_toolbox/deserialize_map": "slam_toolbox/srv/DeserializePoseGraph",
    "/slam_toolbox/dynamic_map": "nav_msgs/srv/GetMap",
    "/slam_toolbox/get_interactive_markers": "visualization_msgs/srv/GetInteractiveMarkers",
    "/slam_toolbox/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/slam_toolbox/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/slam_toolbox/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/slam_toolbox/manual_loop_closure": "slam_toolbox/srv/LoopClosure",
    "/slam_toolbox/pause_new_measurements": "slam_toolbox/srv/Pause",
    "/slam_toolbox/save_map": "slam_toolbox/srv/SaveMap",
    "/slam_toolbox/serialize_map": "slam_toolbox/srv/SerializePoseGraph",
    "/slam_toolbox/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/slam_toolbox/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/slam_toolbox/toggle_interactive_mode": "slam_toolbox/srv/ToggleInteractive",
    "/smooth_path/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/smooth_path/_action/get_result": "nav2_msgs/action/SmoothPath_GetResult",
    "/smooth_path/_action/send_goal": "nav2_msgs/action/SmoothPath_SendGoal",
    "/smoother_server/change_state": "lifecycle_msgs/srv/ChangeState",
    "/smoother_server/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/smoother_server/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/smoother_server/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/smoother_server/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/smoother_server/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/smoother_server/get_state": "lifecycle_msgs/srv/GetState",
    "/smoother_server/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/smoother_server/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/smoother_server/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/smoother_server/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/spin/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/spin/_action/get_result": "nav2_msgs/action/Spin_GetResult",
    "/spin/_action/send_goal": "nav2_msgs/action/Spin_SendGoal",
    "/tf2_frames": "tf2_msgs/srv/FrameGraph",
    "/velocity_smoother/change_state": "lifecycle_msgs/srv/ChangeState",
    "/velocity_smoother/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/velocity_smoother/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/velocity_smoother/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/velocity_smoother/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/velocity_smoother/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/velocity_smoother/get_state": "lifecycle_msgs/srv/GetState",
    "/velocity_smoother/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/velocity_smoother/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/velocity_smoother/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/velocity_smoother/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/wait/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/wait/_action/get_result": "nav2_msgs/action/Wait_GetResult",
    "/wait/_action/send_goal": "nav2_msgs/action/Wait_SendGoal",
    "/waypoint_follower/change_state": "lifecycle_msgs/srv/ChangeState",
    "/waypoint_follower/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/waypoint_follower/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/waypoint_follower/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/waypoint_follower/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/waypoint_follower/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/waypoint_follower/get_state": "lifecycle_msgs/srv/GetState",
    "/waypoint_follower/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/waypoint_follower/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/waypoint_follower/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/waypoint_follower/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
}


class NavigateToPointTask(ROS2ToolCallingAgentTask):
    recursion_limit = 50
    complexity = "medium"
    actions_and_types: Dict[str, str] = {
        "/assisted_teleop": "nav2_msgs/action/AssistedTeleop",
        "/backup": "nav2_msgs/action/BackUp",
        "/compute_path_through_poses": "nav2_msgs/action/ComputePathThroughPoses",
        "/compute_path_to_pose": "nav2_msgs/action/ComputePathToPose",
        "/drive_on_heading": "nav2_msgs/action/DriveOnHeading",
        "/follow_path": "nav2_msgs/action/FollowPath",
        "/follow_waypoints": "nav2_msgs/action/FollowWaypoints",
        "/navigate_through_poses": "nav2_msgs/action/NavigateThroughPoses",
        "/navigate_to_pose": "nav2_msgs/action/NavigateToPose",
        "/smooth_path": "nav2_msgs/action/SmoothPath",
        "/spin": "nav2_msgs/action/Spin",
        "/wait": "nav2_msgs/action/Wait",
    }
    services_and_types: Dict[str, str] = NAVIGATION_SERVICES_AND_TYPES
    action_models: List[type[ActionBaseModel]] = [NavigateToPoseAction]

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        action_strings = [
            f"action: {action}\ntype: {act_type}\n"
            for action, act_type in self.actions_and_types.items()
        ]
        service_strings = [
            f"service: {service}\ntype: {srv_type}\n"
            for service, srv_type in self.services_and_types.items()
        ]

        self.expected_tools: List[BaseTool] = [
            MockGetROS2ActionsNamesAndTypesTool(
                mock_actions_names_and_types=action_strings
            ),
            MockStartROS2ActionTool(
                available_actions=list(self.actions_and_types.keys()),
                available_action_types=list(self.actions_and_types.values()),
                available_action_models=self.action_models,
            ),
            MockGetROS2ActionFeedbackTool(),
            MockGetROS2ActionResultTool(),
            MockGetROS2ServicesNamesAndTypesTool(
                mock_service_names_and_types=service_strings
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=INTERFACES),
        ]

    def get_system_prompt(self) -> str:
        return ROBOT_NAVIGATION_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return "Navigate to the point (2.0, 2.0, 0.0)."

    def verify_tool_calls(self, response: dict[str, Any]):
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        tool_calls = [
            tool_call for message in ai_messages for tool_call in message.tool_calls
        ]
        expected_tool_calls: list[dict[str, Any]] = [
            {"name": "get_ros2_actions_names_and_types", "args": {}},
            {
                "name": "start_ros2_action",
                "args": {
                    "action_name": "/navigate_to_pose",
                    "action_type": "nav2_msgs/action/NavigateToPose",
                    "action_args": {
                        "pose": {
                            "header": {"frame_id": "map"},
                            "pose": {
                                "position": {"x": 2.0, "y": 2.0, "z": 0.0},
                            },
                        },
                    },
                },
                "optional_args": {
                    "action_args": {
                        "pose": {
                            "header": {"stamp": {"sec": 0, "nanosec": 0}},
                            "pose": {
                                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                            },
                        },
                        "behavior_tree": "",
                    }
                },
            },
            {"name": "get_ros2_action_feedback", "args": {"action_id": ANY_VALUE}},
            {"name": "get_ros2_action_result", "args": {"action_id": ANY_VALUE}},
        ]
        self._check_multiple_tool_calls_from_list(
            tool_calls=tool_calls, expected_tool_calls=expected_tool_calls
        )

        if not self.result.errors:
            self.result.success = True


class SpinAroundTask(ROS2ToolCallingAgentTask):
    recursion_limit = 50
    complexity = "medium"
    actions_and_types: Dict[str, str] = {
        "/assisted_teleop": "nav2_msgs/action/AssistedTeleop",
        "/backup": "nav2_msgs/action/BackUp",
        "/compute_path_through_poses": "nav2_msgs/action/ComputePathThroughPoses",
        "/compute_path_to_pose": "nav2_msgs/action/ComputePathToPose",
        "/drive_on_heading": "nav2_msgs/action/DriveOnHeading",
        "/follow_path": "nav2_msgs/action/FollowPath",
        "/follow_waypoints": "nav2_msgs/action/FollowWaypoints",
        "/navigate_through_poses": "nav2_msgs/action/NavigateThroughPoses",
        "/navigate_to_pose": "nav2_msgs/action/NavigateToPose",
        "/smooth_path": "nav2_msgs/action/SmoothPath",
        "/spin": "nav2_msgs/action/Spin",
        "/wait": "nav2_msgs/action/Wait",
    }
    action_models: List[type[ActionBaseModel]] = [SpinAction]

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        action_strings = [
            f"action: {action}\ntype: {act_type}\n"
            for action, act_type in self.actions_and_types.items()
        ]
        self.expected_tools: List[BaseTool] = [
            MockGetROS2ActionsNamesAndTypesTool(
                mock_actions_names_and_types=action_strings
            ),
            MockStartROS2ActionTool(
                available_actions=list(self.actions_and_types.keys()),
                available_action_types=list(self.actions_and_types.values()),
                available_action_models=self.action_models,
            ),
            MockGetROS2ActionFeedbackTool(),
            MockGetROS2ActionResultTool(),
            MockGetROS2MessageInterfaceTool(mock_interfaces=INTERFACES),
        ]

    def get_system_prompt(self) -> str:
        return ROBOT_NAVIGATION_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return "Spin around by 3 radians."

    def verify_tool_calls(self, response: dict[str, Any]):
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        tool_calls = [
            tool_call for message in ai_messages for tool_call in message.tool_calls
        ]
        expected_tool_calls: list[dict[str, Any]] = [
            {"name": "get_ros2_actions_names_and_types", "args": {}},
            {
                "name": "start_ros2_action",
                "args": {
                    "action_name": "/spin",
                    "action_type": "nav2_msgs/action/Spin",
                    "action_args": {"target_yaw": 3},
                },
                "optional_args": {
                    "action_args": {
                        "time_allowance": {"sec": ANY_VALUE, "nanosec": ANY_VALUE}
                    }
                },
            },
            {"name": "get_ros2_action_feedback", "args": {"action_id": ANY_VALUE}},
            {"name": "get_ros2_action_result", "args": {"action_id": ANY_VALUE}},
        ]
        self._check_multiple_tool_calls_from_list(
            tool_calls=tool_calls, expected_tool_calls=expected_tool_calls
        )
        if not self.result.errors:
            self.result.success = True


class MoveToFrontTask(ROS2ToolCallingAgentTask):
    recursion_limit = 50
    complexity = "medium"
    actions_and_types: Dict[str, str] = {
        "/assisted_teleop": "nav2_msgs/action/AssistedTeleop",
        "/backup": "nav2_msgs/action/BackUp",
        "/compute_path_through_poses": "nav2_msgs/action/ComputePathThroughPoses",
        "/compute_path_to_pose": "nav2_msgs/action/ComputePathToPose",
        "/drive_on_heading": "nav2_msgs/action/DriveOnHeading",
        "/follow_path": "nav2_msgs/action/FollowPath",
        "/follow_waypoints": "nav2_msgs/action/FollowWaypoints",
        "/navigate_through_poses": "nav2_msgs/action/NavigateThroughPoses",
        "/navigate_to_pose": "nav2_msgs/action/NavigateToPose",
        "/smooth_path": "nav2_msgs/action/SmoothPath",
        "/spin": "nav2_msgs/action/Spin",
        "/wait": "nav2_msgs/action/Wait",
    }
    action_models: List[type[ActionBaseModel]] = [DriveOnHeadingAction]
    services_and_types: Dict[str, str] = NAVIGATION_SERVICES_AND_TYPES

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        action_strings = [
            f"action: {action}\ntype: {act_type}\n"
            for action, act_type in self.actions_and_types.items()
        ]
        self.expected_tools: List[BaseTool] = [
            MockGetROS2ActionsNamesAndTypesTool(
                mock_actions_names_and_types=action_strings
            ),
            MockStartROS2ActionTool(
                available_actions=list(self.actions_and_types.keys()),
                available_action_types=list(self.actions_and_types.values()),
                available_action_models=self.action_models,
            ),
            MockGetROS2ActionFeedbackTool(),
            MockGetROS2ActionResultTool(),
            MockGetROS2MessageInterfaceTool(mock_interfaces=INTERFACES),
        ]

    def get_system_prompt(self) -> str:
        base_prompt = ROBOT_NAVIGATION_SYSTEM_PROMPT
        tools_description = self.get_tools_description()
        return base_prompt + tools_description

    def get_prompt(self) -> str:
        return "Move 2 meters to the front."

    def verify_tool_calls(self, response: dict[str, Any]):
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        tool_calls = [
            tool_call for message in ai_messages for tool_call in message.tool_calls
        ]
        expected_tool_calls: list[dict[str, Any]] = [
            {"name": "get_ros2_actions_names_and_types", "args": {}},
            {
                "name": "start_ros2_action",
                "args": {
                    "action_name": "/drive_on_heading",
                    "action_type": "nav2_msgs/action/DriveOnHeading",
                    "action_args": {
                        "target": {"x": 2.0},
                        "speed": ANY_VALUE,
                    },
                    # TODO (mkotynia): add support for ranges of allowed values
                },
                "optional_args": {
                    "action_name": "/drive_on_heading",
                    "action_type": "nav2_msgs/action/DriveOnHeading",
                    "action_args": {
                        "target": {"y": 0.0, "z": 0.0},
                        "time_allowance": {"sec": ANY_VALUE, "nanosec": ANY_VALUE},
                    },
                },
            },
            {"name": "get_ros2_action_feedback", "args": {"action_id": ANY_VALUE}},
            {"name": "get_ros2_action_result", "args": {"action_id": ANY_VALUE}},
        ]
        self._check_multiple_tool_calls_from_list(
            tool_calls=tool_calls, expected_tool_calls=expected_tool_calls
        )
        if not self.result.errors:
            self.result.success = True


class MoveToBedTask(ROS2ToolCallingAgentTask):
    recursion_limit = 50
    complexity = "medium"
    actions_and_types: Dict[str, str] = {
        "/assisted_teleop": "nav2_msgs/action/AssistedTeleop",
        "/backup": "nav2_msgs/action/BackUp",
        "/compute_path_through_poses": "nav2_msgs/action/ComputePathThroughPoses",
        "/compute_path_to_pose": "nav2_msgs/action/ComputePathToPose",
        "/drive_on_heading": "nav2_msgs/action/DriveOnHeading",
        "/follow_path": "nav2_msgs/action/FollowPath",
        "/follow_waypoints": "nav2_msgs/action/FollowWaypoints",
        "/navigate_through_poses": "nav2_msgs/action/NavigateThroughPoses",
        "/navigate_to_pose": "nav2_msgs/action/NavigateToPose",
        "/smooth_path": "nav2_msgs/action/SmoothPath",
        "/spin": "nav2_msgs/action/Spin",
        "/wait": "nav2_msgs/action/Wait",
    }
    topics_names_and_types = [
        "topic: /assisted_teleop/_action/feedback\ntype: nav2_msgs/action/AssistedTeleop_FeedbackMessage\n",
        "topic: /assisted_teleop/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
        "topic: /backup/_action/feedback\ntype: nav2_msgs/action/BackUp_FeedbackMessage\n",
        "topic: /backup/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
        "topic: /behavior_server/transition_event\ntype: lifecycle_msgs/msg/TransitionEvent\n",
        "topic: /behavior_tree_log\ntype: nav2_msgs/msg/BehaviorTreeLog\n",
        "topic: /bond\ntype: bond/msg/Status\n",
        "topic: /bt_navigator/transition_event\ntype: lifecycle_msgs/msg/TransitionEvent\n",
        "topic: /camera/camera/color/camera_info\ntype: sensor_msgs/msg/CameraInfo\n",
        "topic: /camera/camera/color/image_raw\ntype: sensor_msgs/msg/Image\n",
        "topic: /camera/camera/depth/camera_info\ntype: sensor_msgs/msg/CameraInfo\n",
        "topic: /camera/camera/depth/image_rect_raw\ntype: sensor_msgs/msg/Image\n",
        "topic: /clock\ntype: rosgraph_msgs/msg/Clock\n",
        "topic: /cmd_vel_nav\ntype: geometry_msgs/msg/Twist\n",
        "topic: /cmd_vel_teleop\ntype: geometry_msgs/msg/Twist\n",
        "topic: /compute_path_through_poses/_action/feedback\ntype: nav2_msgs/action/ComputePathThroughPoses_FeedbackMessage\n",
        "topic: /compute_path_through_poses/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
        "topic: /compute_path_to_pose/_action/feedback\ntype: nav2_msgs/action/ComputePathToPose_FeedbackMessage\n",
        "topic: /compute_path_to_pose/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
        "topic: /controller_server/transition_event\ntype: lifecycle_msgs/msg/TransitionEvent\n",
        "topic: /diagnostics\ntype: diagnostic_msgs/msg/DiagnosticArray\n",
        "topic: /drive_on_heading/_action/feedback\ntype: nav2_msgs/action/DriveOnHeading_FeedbackMessage\n",
        "topic: /drive_on_heading/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
        "topic: /follow_path/_action/feedback\ntype: nav2_msgs/action/FollowPath_FeedbackMessage\n",
        "topic: /follow_path/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
        "topic: /follow_waypoints/_action/feedback\ntype: nav2_msgs/action/FollowWaypoints_FeedbackMessage\n",
        "topic: /follow_waypoints/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
        "topic: /global_costmap/costmap\ntype: nav_msgs/msg/OccupancyGrid\n",
        "topic: /global_costmap/costmap_raw\ntype: nav2_msgs/msg/Costmap\n",
        "topic: /global_costmap/costmap_updates\ntype: map_msgs/msg/OccupancyGridUpdate\n",
        "topic: /global_costmap/footprint\ntype: geometry_msgs/msg/Polygon\n",
        "topic: /global_costmap/global_costmap/transition_event\ntype: lifecycle_msgs/msg/TransitionEvent\n",
        "topic: /global_costmap/published_footprint\ntype: geometry_msgs/msg/PolygonStamped\n",
        "topic: /global_costmap/scan\ntype: sensor_msgs/msg/LaserScan\n",
        "topic: /goal_pose\ntype: geometry_msgs/msg/PoseStamped\n",
        "topic: /led_strip\ntype: sensor_msgs/msg/Image\n",
        "topic: /local_costmap/costmap\ntype: nav_msgs/msg/OccupancyGrid\n",
        "topic: /local_costmap/costmap_raw\ntype: nav2_msgs/msg/Costmap\n",
        "topic: /local_costmap/costmap_updates\ntype: map_msgs/msg/OccupancyGridUpdate\n",
        "topic: /local_costmap/footprint\ntype: geometry_msgs/msg/Polygon\n",
        "topic: /local_costmap/local_costmap/transition_event\ntype: lifecycle_msgs/msg/TransitionEvent\n",
        "topic: /local_costmap/published_footprint\ntype: geometry_msgs/msg/PolygonStamped\n",
        "topic: /local_costmap/scan\ntype: sensor_msgs/msg/LaserScan\n",
        "topic: /map\ntype: nav_msgs/msg/OccupancyGrid\n",
        "topic: /map_metadata\ntype: nav_msgs/msg/MapMetaData\n",
        "topic: /map_saver/transition_event\ntype: lifecycle_msgs/msg/TransitionEvent\n",
        "topic: /navigate_through_poses/_action/feedback\ntype: nav2_msgs/action/NavigateThroughPoses_FeedbackMessage\n",
        "topic: /navigate_through_poses/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
        "topic: /navigate_to_pose/_action/feedback\ntype: nav2_msgs/action/NavigateToPose_FeedbackMessage\n",
        "topic: /navigate_to_pose/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
        "topic: /odom\ntype: nav_msgs/msg/Odometry\n",
        "topic: /odometry/filtered\ntype: nav_msgs/msg/Odometry\n",
        "topic: /parameter_events\ntype: rcl_interfaces/msg/ParameterEvent\n",
        "topic: /plan\ntype: nav_msgs/msg/Path\n",
        "topic: /plan_smoothed\ntype: nav_msgs/msg/Path\n",
        "topic: /planner_server/transition_event\ntype: lifecycle_msgs/msg/TransitionEvent\n",
        "topic: /pose\ntype: geometry_msgs/msg/PoseWithCovarianceStamped\n",
        "topic: /preempt_teleop\ntype: std_msgs/msg/Empty\n",
        "topic: /rosout\ntype: rcl_interfaces/msg/Log\n",
        "topic: /scan\ntype: sensor_msgs/msg/LaserScan\n",
        "topic: /slam_toolbox/feedback\ntype: visualization_msgs/msg/InteractiveMarkerFeedback\n",
        "topic: /slam_toolbox/graph_visualization\ntype: visualization_msgs/msg/MarkerArray\n",
        "topic: /slam_toolbox/scan_visualization\ntype: sensor_msgs/msg/LaserScan\n",
        "topic: /slam_toolbox/update\ntype: visualization_msgs/msg/InteractiveMarkerUpdate\n",
        "topic: /smooth_path/_action/feedback\ntype: nav2_msgs/action/SmoothPath_FeedbackMessage\n",
        "topic: /smooth_path/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
        "topic: /smoother_server/transition_event\ntype: lifecycle_msgs/msg/TransitionEvent\n",
        "topic: /speed_limit\ntype: nav2_msgs/msg/SpeedLimit\n",
        "topic: /spin/_action/feedback\ntype: nav2_msgs/action/Spin_FeedbackMessage\n",
        "topic: /spin/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
        "topic: /tf_static\ntype: tf2_msgs/msg/TFMessage\n",
        "topic: /trajectories\ntype: visualization_msgs/msg/MarkerArray\n",
        "topic: /transformed_global_plan\ntype: nav_msgs/msg/Path\n",
        "topic: /unsmoothed_plan\ntype: nav_msgs/msg/Path\n",
        "topic: /velocity_smoother/transition_event\ntype: lifecycle_msgs/msg/TransitionEvent\n",
        "topic: /wait/_action/feedback\ntype: nav2_msgs/action/Wait_FeedbackMessage\n",
        "topic: /wait/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
        "topic: /waypoint_follower/transition_event\ntype: lifecycle_msgs/msg/TransitionEvent\n",
    ]
    action_models: List[type[ActionBaseModel]] = [DriveOnHeadingAction]
    services_and_types: Dict[str, str] = NAVIGATION_SERVICES_AND_TYPES

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        action_strings = [
            f"action: {action}\ntype: {act_type}\n"
            for action, act_type in self.actions_and_types.items()
        ]
        self.expected_tools: List[BaseTool] = [
            MockGetROS2ActionsNamesAndTypesTool(
                mock_actions_names_and_types=action_strings
            ),
            MockStartROS2ActionTool(
                available_actions=list(self.actions_and_types.keys()),
                available_action_types=list(self.actions_and_types.values()),
                available_action_models=self.action_models,
            ),
            MockGetROS2ActionFeedbackTool(),
            MockGetROS2ActionResultTool(),
            MockGetROS2MessageInterfaceTool(mock_interfaces=INTERFACES),
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=self.topics_names_and_types
            ),
            MockGetDistanceToObjectsTool(
                available_topics=[
                    "/camera/camera/color/image_raw",
                    "/camera/camera/depth/image_rect_raw",
                ],
                mock_distance_measurements=[
                    DistanceMeasurement(name="bed", distance=5.0)
                ],
            ),
        ]

    def get_system_prompt(self) -> str:
        return ROBOT_NAVIGATION_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return "Move closer to the to the bed. Leave 1 meter of space between the bed and you."

    def verify_tool_calls(self, response: dict[str, Any]):
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        tool_calls = [
            tool_call for message in ai_messages for tool_call in message.tool_calls
        ]
        expected_tool_calls: list[dict[str, Any]] = [
            {"name": "get_ros2_actions_names_and_types", "args": {}},
            {"name": "get_ros2_topics_names_and_types", "args": {}},
            {
                "name": "GetDistanceToObjectsTool",
                "args": {
                    "camera_topic": "/camera/camera/color/image_raw",
                    "depth_topic": "/camera/camera/depth/image_rect_raw",
                    "object_names": ["bed"],
                },
            },
            {
                "name": "start_ros2_action",
                "args": {
                    "action_name": "/drive_on_heading",
                    "action_type": "nav2_msgs/action/DriveOnHeading",
                    "action_args": {
                        "target": {"x": 4.0},
                        "speed": ANY_VALUE,
                    },
                    # TODO (mkotynia): add support for ranges of allowed values
                },
                "optional_args": {
                    "action_name": "/drive_on_heading",
                    "action_type": "nav2_msgs/action/DriveOnHeading",
                    "action_args": {
                        "target": {"y": 0.0, "z": 0.0},
                        "time_allowance": {"sec": ANY_VALUE, "nanosec": ANY_VALUE},
                    },
                },
            },
            {"name": "get_ros2_action_feedback", "args": {"action_id": ANY_VALUE}},
            {"name": "get_ros2_action_result", "args": {"action_id": ANY_VALUE}},
        ]
        self._check_multiple_tool_calls_from_list(
            tool_calls=tool_calls,
            expected_tool_calls=expected_tool_calls,
        )
        if not self.result.errors:
            self.result.success = True
