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
# See the License for the specific language goveself.rning permissions and
# limitations under the License.

from langchain_core.messages import HumanMessage

from rai.agents.langchain.runnables import create_state_based_runnable2
from rai.agents.state_based_agent import Aggregator
from rai.tools.ros2.generic.toolkit import ROS2Toolkit
from rai.tools.ros2.simple import GetROS2ImageConfiguredTool
from rai.utils import ROS2Context


@ROS2Context()
def main():
    aggregator = Aggregator()
    tools = [
        *ROS2Toolkit(connector=aggregator.connector).get_tools(),
        GetROS2ImageConfiguredTool(
            connector=aggregator.connector,
            topic="/camera/camera/color/image_raw",
            response_format="content_and_artifact",
        ),
    ]
    SYSTEM_PROMPT = "You are a very cool agent"
    agent = create_state_based_runnable2(
        tools=tools, state_retriever=aggregator.aggregate, system_prompt=SYSTEM_PROMPT
    )
    while True:
        inp = input("Enter your message: ")
        state = dict()
        state["task"] = HumanMessage(content=inp)
        state["available_tools"] = tools
        state["tool_messages"] = []
        agent.invoke(state, config={"recursion_limit": 2000})
        print(f"{state['result']=}")


if __name__ == "__main__":
    main()
