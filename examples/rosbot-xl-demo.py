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

import rclpy
import rclpy.executors
import rclpy.logging

from pydantic import BaseModel, Field
# import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import render_text_description_and_args, tool
from rai.agents.conversational_agent import create_conversational_agent
from rai.communication.ros2 import ROS2ARIConnector
from rai.messages import HumanMultimodalMessage
from rai.tools.ros.manipulation import GetObjectPositionsTool
from rai.tools.ros2 import (
    GetROS2ImageTool,
    GetROS2TransformTool,
)
from rai.tools.ros2.nav2 import (
    CancelNavigateToPoseTool,
    GetNavigateToPoseFeedbackTool,
    GetNavigateToPoseResultTool,
    NavigateToPoseTool,
)
from rai.utils.model_initialization import get_llm_model, get_tracing_callbacks
from rai_open_set_vision.tools import GetGrabbingPointTool
from rai.tools.ros.nav2.basic_navigator import navigator, spin_robot, drive_forward, go_to_pose, check_nav_status, cancel_navigation_task, quat2eulers
from tf2_ros import TransformStamped


# @st.cache_resource
def initialize_agent():
    connector = ROS2ARIConnector()
    transform_tool = GetROS2TransformTool(connector=connector)
    image_tool = GetROS2ImageTool(connector=connector)

    @tool
    def estimate_object_size(object_name:str):
        """ Returns esitmated object width and height. You can estimate humans sizes as well """
        tool = GetObjectPositionsTool(
            connector=connector,
            target_frame="map",
            source_frame="sensor_frame",
            camera_topic="/camera/camera/color/image_raw",
            depth_topic="/camera/camera/depth/image_rect_raw",
            camera_info_topic="/camera/camera/color/camera_info",
            get_grabbing_point_tool=GetGrabbingPointTool(
                connector=connector,
            ),
        )
        poses = tool._run(object_name=object_name)
        if poses is None:
            return f"{object_name} is not visible"

        print(f'Poses: {poses}')
        
        results = []
        for p in poses:
            r = {'x': round(p.position.x, 2), 'y': round(p.position.y, 2)}
            results.append(r)
        if results:
            obj_location = results[0]
        else:
            obj_location = None

        tf: TransformStamped = transform_tool._run(
            source_frame="base_link", target_frame="map", timeout_sec=5.0
        )
        x = round(tf.transform.translation.x, 2)
        y = round(tf.transform.translation.y, 2)

        q = tf.transform.rotation
        _,_,yaw = quat2eulers(q.x,q.y,q.z,q.w)
        yaw = round(yaw, 2)
        cur_location = {"x": x, "y": y, "yaw": yaw}

        llm = get_llm_model("simple_model")
        _, artifact = image_tool._run(
            topic="/camera/camera/color/image_raw", timeout_sec=5.0
        )
        system_prompt = "You are an expert in image analysis and your speciality is estimation of object sizes based on your current location and objects location. You can caluclate the distance between location of camera image and the object and based on the given image you can estimate the object dimensions very well. If object location is None, try to guess the distance based on the image"
        task = [
            SystemMessage(content=system_prompt),
            HumanMultimodalMessage(
                content=f"Please return the object dimensions. camera positon: {cur_location}. Object position: {obj_location}",
                images=artifact["images"],
            ),
        ]
        class ResponseFormatter(BaseModel):
            """Always use this tool to structure your response to the user."""
            can_see: bool = Field(description="If the object is visible or not")
            width: float = Field(description="The width of the object in meters")
            height: float = Field(description="The height of the object in meters")

        llm = llm.with_structured_output(ResponseFormatter)
        output: ResponseFormatter = llm.invoke(task, config={"callbacks": []})
        return {"width" : output.width, "height": output.height}

    @tool
    def where_am_i():
        """
        Get your current position
        """
        tf: TransformStamped = transform_tool._run(
            source_frame="base_link", target_frame="map", timeout_sec=5.0
        )
        x = round(tf.transform.translation.x, 2)
        y = round(tf.transform.translation.y, 2)

        q = tf.transform.rotation
        _,_,yaw = quat2eulers(q.x,q.y,q.z,q.w)
        yaw = round(yaw, 2)
        return {"x": x, "y": y, "yaw": yaw}

    @tool
    def what_do_i_see(object_name: str="") -> str:
        """
        Get what you see. You can pass specific object name to get it's location.

        Example usages:
        what_do_i_see args: {"object_name": ""} # just describe the environment
        -> returs: "I see a table and a chair. 
        what_do_i_see args: {"object_name": "cup"}
        -> returs: "I see a table and a chair. I can see 1 cup. Their centroids are: [{'x': 1.0, 'y': 1.0}]" 
        what_do_i_see args: {"object_name": "cup"}
        -> returs: "I see a table and a chair. I cannot see a cup. 
        """
        llm = get_llm_model("simple_model")
        _, artifact = image_tool._run(
            topic="/camera/camera/color/image_raw", timeout_sec=5.0
        )
        system_prompt = "You are an expert in image analysis. You are given an image and you need to describe what you see in it. Reply with I see..."
        task = [
            SystemMessage(content=system_prompt),
            HumanMultimodalMessage(
                content="Please describe what you see in the image. Reply with I see...",
                images=artifact["images"],
            ),
        ]
        response = llm.invoke(task, config={"callbacks": []})
        img_desc = cast(str, response.content)
        class ResponseFormatter(BaseModel):
            """Always use this tool to structure your response to the user."""
            object_visible: bool = Field(description="Whether the object is visible or not")
            count: int = Field(description="The number of objects visible")

        resp = img_desc

        if object_name:
            print("Checking for object name")
            llm = llm.with_structured_output(ResponseFormatter)

            system_prompt = "You are an expert in image analysis. You are given an image and you need to check if given object is visible in the image. If multiple objects are visible, please count them. For example if you are asked for a cube and you see 4 cubes, return True and count of 4."
            task = [
                SystemMessage(content=system_prompt),
                HumanMultimodalMessage(
                    content=f"Please check if {object_name} is visible in the image",
                    images=artifact["images"],
                ),
            ]

            output: ResponseFormatter = llm.invoke(task, config={"callbacks": []})
            print(f'output: {output}')
            if output.object_visible:
                tool = GetObjectPositionsTool(
                    connector=connector,
                    target_frame="map",
                    source_frame="sensor_frame",
                    camera_topic="/camera/camera/color/image_raw",
                    depth_topic="/camera/camera/depth/image_rect_raw",
                    camera_info_topic="/camera/camera/color/camera_info",
                    get_grabbing_point_tool=GetGrabbingPointTool(
                        connector=connector,
                    ),
                )
                poses = tool._run(object_name=object_name)
                print(f'Poses: {poses}')
                
                results = []
                if poses is None:
                    resp = f"{img_desc}. I can see {object_name}, but I can't detect it's position"
                else:
                    for p in poses:
                        r = {'x': round(p.position.x, 2), 'y': round(p.position.y, 2)}
                        results.append(r)
                    resp = f'{img_desc}. I can see {output.count} {object_name}s.'
                    resp = f'{resp}. Their centroids are: {results}'
            else:
                resp = f'{img_desc}. I cannot see a {object_name}.'

        return resp

    @tool
    def wait_for_n_seconds(n: int):
        """ Wait for n seconds """
        import time

        time.sleep(n)
        return "Done"

    tools = [
        where_am_i,
        what_do_i_see,
        go_to_pose,
        spin_robot,
        check_nav_status,
        cancel_navigation_task,
        wait_for_n_seconds,
        estimate_object_size
    ]

    SYSTEM_PROMPT = """You are an autonomous robot connected to ros2 environment. Your main goal is to fulfill the user's requests.
    Do not make assumptions about the environment you are currently in.

    You always respond in first person (as a robot). You always treat data gathered by the tools as a description of your current state coming from within.

    Here are the locations of places in your environment:

    Kitchen:
    (0.3, -0.85, 0.0),

    # Living room:
    (-0.82, 3.5, 0.0)

    Available tools:
    """ + render_text_description_and_args(tools) + """

    Examples of successful tasks:
    **Task 1**
    User: "How are you"
    - no tool calls 
    example response: "I'm fine, thank you"
    
    Important: not every user query required tool calls. Use them when necessary
    
    **Task 1**
    User: "Drive to the left of the cube"
    Sequence of tool calls:
    what_do_i_see args: {"object_name": "cube"}
    -> returs: "I see a table and a chair. I can see 1 cube. Their centroids are: [{'x': 2.0, 'y': 0.0}]" 
    where_am_i args: {}
    -> return (x: 0.0, y: 0.0, yaw: 0.9)
    estimate_object_size args: {"object_name" : "cube"}
    -> return (width: 1.0, height: 1.0)
    comment: it is important to set the navigation goal so that it is outside of the object boundaries with some margin. (20 cm is fine)
    go_to_pose args: (x: 1.0, y:0.7, yaw: 0.0)
    -> returned: Robot navigating to pose, comment: navigation is started and running in the background. Status of navigation (is it in progress/done/failed) can be checked with `get_navigate_to_pose_result` tool or canceled with `cancel_navigate_to_pose`
    wait_for_n_seconds args: {'n': 10}
    -> returned: Done
    check_nav_status args: {}
    -> returned: navigation in progress 
    wait_for_n_seconds args: {'n': 10}
    
    Two options of task continuation:
    1: 
    check_nav_status() args: {}
    -> returned: navigation succeeded
    where_am_i args: {}
    -> return (x: 0.3, y: -1.0, yaw: 0.01), comment: navigation goal is very close, so it's fine.

    2:
    wait_for_n_seconds args: {'n': 5}
    check_nav_status() args: {}
    -> returned: navigation failed
    where_am_i args: {}
    -> return (x: 1.3, y: -3.0, yaw: 0.5), comment: navigation failed and goal is far, navigate once again 
    go_to_pose args: (x: 0.3, y: -0.85, yaw: 0.0)
    -> returned: Robot navigating to pose
    wait_for_n_seconds args: {'n': 10}
    -> returned: navigation succeeded
    where_am_i args: {}
    -> return (x: 0.3, y: -1.0, yaw: 0.01), comment: navigation goal is very close, so it's fine.

    ---
    **Task 2**
    User: "Get the cube location"
    Sequence of tool calls:
    what_do_i_see args: {"object_name": "cube"}
    -> returs: "I see a table and a chair. I cannot see a cube. 
    or 
    what_do_i_see args: {"object_name": "cube"}
    -> returs: "I see a table and a chair. I can see 1 cube. Their centroids are: [{'x'0.3, 'y': -0.85}]" 
"""
    print(SYSTEM_PROMPT)

    agent = create_conversational_agent(
        llm=get_llm_model("complex_model", streaming=True),
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
    )
    connector.node.declare_parameter("conversion_ratio", 1.0)
    callbacks = get_tracing_callbacks()
    return agent, callbacks


def main():
    agent, callbacks = initialize_agent()
    state = {"messages": []}
    while True:
        inp = input("Enter your message: ")
        state["messages"].append(HumanMessage(content=inp))
        l_before = len(state["messages"])
        agent.invoke(state, config={"callbacks": callbacks, "recursion_limit": 2000})
        l_after = len(state["messages"])
        for msg in state["messages"][l_before:l_after]:
            msg.pretty_print()


if __name__ == "__main__":
    main()
