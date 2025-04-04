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

# import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from rai.agents.conversational_agent import create_conversational_agent
from rai.communication.ros2 import ROS2ARIConnector
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


# @st.cache_resource
def initialize_agent():
    rclpy.init()
    connector = ROS2ARIConnector()
    transform_tool = GetROS2TransformTool(connector=connector)
    image_tool = GetROS2ImageTool(connector=connector)

    @tool
    def where_am_i():
        """
        Get your current position
        """
        return transform_tool._run(
            source_frame="base_link", target_frame="map", timeout_sec=5.0
        )

    @tool(response_format="content_and_artifact")
    def what_do_i_see():
        """
        Get what you see
        """
        # llm = get_llm_model("simple_model")
        response, artifact = image_tool._run(
            topic="/camera/camera/color/image_raw", timeout_sec=5.0
        )
        return response, artifact
        # system_prompt = "You are an expert in image analysis. You are given an image and you need to describe what you see in it. Reply with I see..."
        # task = [
        #     SystemMessage(content=system_prompt),
        #     HumanMultimodalMessage(
        #         content="Please describe what you see in the image. Reply with I see...",
        #         images=artifact["images"],
        #     ),
        # ]
        # response = llm.invoke(task, config={"callbacks": []})
        return cast(str, response.content)

    tools = [
        where_am_i,
        # what_do_i_see,
        NavigateToPoseTool(
            connector=connector,
            action_name="navigate_to_pose",
            frame_id="map",
        ),
        GetNavigateToPoseFeedbackTool(
            connector=connector,
        ),
        GetNavigateToPoseResultTool(
            connector=connector,
        ),
        CancelNavigateToPoseTool(
            connector=connector,
        ),
        GetObjectPositionsTool(
            connector=connector,
            target_frame="map",
            source_frame="sensor_frame",
            camera_topic="/camera/camera/color/image_raw",
            depth_topic="/camera/camera/depth/image_rect_raw",
            camera_info_topic="/camera/camera/color/camera_info",
            get_grabbing_point_tool=GetGrabbingPointTool(
                connector=connector,
            ),
        ),
    ]
    SYSTEM_PROMPT = """You are an autonomous robot connected to ros2 environment. Your main goal is to fulfill the user's requests.
    Do not make assumptions about the environment you are currently in.

    You always respond in first person (as a robot). You always treat data gathered by the tools as a description of your current state coming from within.

    Here are the locations of places in your environment:

    Kitchen:
    (0.3, -0.85, 0.0),

    # Living room:
    (-0.82, 3.5, 0.0)

   """

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
