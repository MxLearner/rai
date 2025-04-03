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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import render_text_description_and_args, tool
from rai.agents.conversational_agent import create_conversational_agent
from rai.agents.integrations.streamlit import get_streamlit_cb, streamlit_invoke
from rai.communication.ros2 import ROS2ARIConnector
from rai.messages import HumanMultimodalMessage
from rai.tools.ros2 import (
    GetROS2ImageTool,
    GetROS2MessageInterfaceTool,
    GetROS2TransformTool,
)
from rai.tools.ros2.actions import (
    GetROS2ActionFeedbackTool,
    GetROS2ActionResultTool,
    ROS2ActionToolkit,
    StartROS2ActionTool,
)
from rai.tools.time import WaitForSecondsTool
from rai.utils.model_initialization import get_llm_model, get_tracing_callbacks


# @st.cache_resource
def initialize_agent():
    rclpy.init()
    connector = ROS2ARIConnector()
    transform_tool = GetROS2TransformTool(connector=connector)
    image_tool = GetROS2ImageTool(connector=connector)
    start_action_tool = StartROS2ActionTool(connector=connector)
    get_action_feedback_tool = GetROS2ActionFeedbackTool(connector=connector)
    get_action_result_tool = GetROS2ActionResultTool()

    @tool
    def where_am_i():
        """
        Get your current position
        """
        return transform_tool._run(
            source_frame="base_link", target_frame="map", timeout_sec=5.0
        )

    @tool
    def what_do_i_see() -> str:
        """
        Get what you see
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
        return cast(str, response.content)

    @tool
    def go_to_place(x: float, y: float, z: float):
        """
        Go to a specific place
        """
        action_id = start_action_tool._run(
            action_name="navigate_to_pose",
            action_type="nav2_msgs/action/NavigateToPose",
            action_args={
                "pose": {
                    "header": {"frame_id": "map", "stamp": {"sec": 0, "nanosec": 0}},
                    "pose": {
                        "position": {"x": x, "y": y, "z": z},
                        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    },
                },
            },
        )
        result = ""
        while True:
            try:
                result = get_action_result_tool._run(action_id)
                print("Result: ", result)
                continue
            except KeyError as e:
                print(e)
                break
        return str(result)

    tools = [
        where_am_i,
        what_do_i_see,
        #  go_to_place,
        *ROS2ActionToolkit(connector=connector).get_tools(),
        GetROS2MessageInterfaceTool(connector=connector),
        WaitForSecondsTool(),
        # *ROS2Toolkit(
        #     connector=connector, forbidden=["/tf", "/cmd_vel"]
        # ).get_tools(),
        # WaitForSecondsTool(),
        # GetDetectionTool(connector=connector, node=connector.node),
        # GetDistanceToObjectsTool(connector=connector, node=connector.node),
        # GetObjectPositionsTool(
        #     connector=connector,
        #     target_frame="map",
        #     source_frame="sensor_frame",
        #     camera_topic="/camera/camera/color/image_raw",
        #     depth_topic="/camera/camera/depth/image_rect_raw",
        #     camera_info_topic="/camera/camera/color/camera_info",
        #     get_grabbing_point_tool=GetGrabbingPointTool(
        #         connector=connector,
        #     ),
        # ),
    ]
    SYSTEM_PROMPT = (
        """You are an autonomous robot connected to ros2 environment. Your main goal is to fulfill the user's requests.
    Do not make assumptions about the environment you are currently in.

    Here are the locations of places in your environment:

    Kitchen:
    (0.3, -0.85, 0.0),

    # Living room:
    (-0.82, 3.5, 0.0)

    Here are the tools you can use:
    """
        + f"{render_text_description_and_args(tools)}"
    ) + """
    Here are some examples of how to use the tools:
    start_ros2_action, args: {'action_name': '/navigate_to_pose', 'action_type': 'nav2_msgs/action/NavigateToPose', 'action_args': {'pose': {'header': {'stamp': {'sec': 0, 'nanosec': 0}, 'frame_id': 'map'}, 'pose': {'position': {'x': -0.82, 'y': 3.5, 'z': 0.0}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}}}}}
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
    st.set_page_config(
        page_title="RAI ROSBotXL Demo",
        page_icon=":robot:",
    )
    st.title("RAI ROSBotXL Demo")
    st.markdown("---")

    st.sidebar.header("Tool Calls History")

    if "graph" not in st.session_state:
        graph, callbacks = initialize_agent()
        st.session_state["graph"] = graph
        st.session_state["callbacks"] = callbacks

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            AIMessage(content="Hi! I am ROSBotXL. What can I do for you?")
        ]

    prompt = st.chat_input()
    for msg in st.session_state.messages:
        if isinstance(msg, AIMessage):
            if msg.content:
                st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMultimodalMessage):
            continue
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, ToolMessage):
            with st.sidebar.expander(f"Tool: {msg.name}", expanded=False):
                st.code(msg.content, language="json")

    if prompt:
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st_callback = get_streamlit_cb(st.container())
            streamlit_invoke(
                st.session_state["graph"],
                st.session_state.messages,
                [st_callback, *st.session_state["callbacks"]],
            )


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
