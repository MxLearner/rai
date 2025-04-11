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

import time
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
    cast,
)

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, render_text_description_and_args
from langgraph.graph import START, StateGraph
from langgraph.prebuilt.tool_node import tools_condition
from pydantic import BaseModel, Field

from rai.agents.tool_runner import ToolRunner
from rai.utils.model_initialization import get_llm_model


class ReActAgentState(TypedDict):
    """State type for the react agent.

    Parameters
    ----------
    messages : List[BaseMessage]
        List of messages in the conversation
    """

    messages: List[BaseMessage]


def llm_node(llm: BaseChatModel, system_prompt: Optional[str], state: ReActAgentState):
    """Process messages using the LLM.

    Parameters
    ----------
    llm : BaseChatModel
        The language model to use for processing
    state : ReActAgentState
        Current state containing messages

    Returns
    -------
    ReActAgentState
        Updated state with new AI message

    Raises
    ------
    ValueError
        If state is invalid or LLM processing fails
    """
    if system_prompt:
        # at this point, state['messages'] length should at least be 1
        if not isinstance(state["messages"][0], SystemMessage):
            state["messages"].insert(0, SystemMessage(content=system_prompt))
    ai_msg = llm.invoke(state["messages"])
    state["messages"].append(ai_msg)
    print(f"Responded with AI Message: {ai_msg}")


def create_react_runnable(
    llm: Optional[BaseChatModel] = None,
    tools: Optional[List[BaseTool]] = None,
    system_prompt: Optional[str] = None,
) -> Runnable[ReActAgentState, ReActAgentState]:
    """Create a react agent that can process messages and optionally use tools.

    Parameters
    ----------
    llm : Optional[BaseChatModel], default=None
        Language model to use. If None, will use complex_model from config
    tools : Optional[List[BaseTool]], default=None
        List of tools the agent can use

    Returns
    -------
    Runnable[ReActAgentState, ReActAgentState]
        A runnable that processes messages and optionally uses tools

    Raises
    ------
    ValueError
        If tools are provided but invalid
    """
    if llm is None:
        llm = get_llm_model("complex_model", streaming=True)

    graph = StateGraph(ReActAgentState)
    graph.add_edge(START, "llm")

    if tools:
        tool_runner = ToolRunner(tools)
        graph.add_node("tools", tool_runner)
        graph.add_conditional_edges(
            "llm",
            tools_condition,
        )
        graph.add_edge("tools", "llm")
        # Bind tools to LLM
        bound_llm = cast(BaseChatModel, llm.bind_tools(tools))
        graph.add_node("llm", partial(llm_node, bound_llm, system_prompt))
    else:
        graph.add_node("llm", partial(llm_node, llm, system_prompt))

    # Compile the graph
    return graph.compile()


class StateMessage(HumanMessage):
    pass


def set_current_state(state: ReActAgentState, state_message: StateMessage):
    new_messages = []
    for m in state["messages"]:
        if not isinstance(m, StateMessage):
            new_messages.append(m)
    new_messages.append(state_message)
    state["messages"] = new_messages
    print(state)


def retriever_wrapper(
    state_retriever: Callable[[], Dict[str, Any]], state: ReActAgentState
):
    """This wrapper is used to retrieve multimodal information from the output of state_retriever."""
    ts = time.perf_counter()
    retrieved_info: Dict[str, Any] = (
        state_retriever()
    )  # TODO)boczekbartek): define interface
    te = time.perf_counter() - ts
    print(f"Retrieved state in {te} seconds")
    state_message = StateMessage(content=str(retrieved_info))
    state["state"] = state_message
    return state
    # set_current_state(state, state_message)


def create_state_based_runnable(
    llm: Optional[BaseChatModel] = None,
    tools: Optional[List[BaseTool]] = None,
    system_prompt: Optional[str] = None,
    state_retriever: Optional[Callable[[], Dict[str, Any]]] = None,
) -> Runnable[ReActAgentState, ReActAgentState]:
    if llm is None:
        llm = get_llm_model("complex_model", streaming=True)
    graph = StateGraph(ReActAgentState)
    graph.add_edge(START, "state_retriever")
    graph.add_edge("state_retriever", "llm")
    graph.add_conditional_edges(
        "llm",
        tools_condition,
    )
    graph.add_edge("tools", "state_retriever")

    if state_retriever is None:
        state_retriever = lambda: {}

    graph.add_node("state_retriever", partial(retriever_wrapper, state_retriever))

    if tools is None:
        tools = []
    bound_llm = cast(BaseChatModel, llm.bind_tools(tools))
    graph.add_node("llm", partial(llm_node, bound_llm, system_prompt))

    tool_runner = ToolRunner(tools)
    graph.add_node("tools", tool_runner)

    return graph.compile()


def custom_tool_condition(
    state: Union[list[AnyMessage], dict[str, Any], BaseModel],
    messages_key: str = "messages",
):
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "state_retriever"


class State(TypedDict):
    plan: List[Tuple[str, bool]]
    plan_feasible: bool
    available_tools: List[BaseTool]
    task: str
    messages: List[AnyMessage]
    state: str
    tool_messages: List[AnyMessage]
    result: str
    # danger_detected: bool


def planner(state):
    class Plan(BaseModel):
        plan: List[str] = Field(..., description="Plan")
        feasable: bool = Field(..., description="Feasable")

    print(state)
    llm = get_llm_model("complex_model", streaming=True)
    llm = llm.with_structured_output(Plan)
    tool_render = render_text_description_and_args(state["available_tools"])

    system_prompt = SystemMessage(
        f"""You are a profesional task planner.
    Given a task create a plan on how the task should be done.
    The plan has to be feasable with given tools: {tool_render}
    If the plan is not feasable please inform me."""
    )
    msg = HumanMessage(content=f"The task is: {state['task']}")
    plan: Plan = llm.invoke([system_prompt, msg])

    # none of the points in the plan are done
    state["plan"] = [(plan_point, False) for plan_point in plan.plan]
    state["plan_feasible"] = plan.feasable
    return state


def tool_repeater(state):
    class Decision(BaseModel):
        tool_call_successful: bool = Field(
            ..., description="Decision if the tool call has been successfull"
        )

    # TODO it can be done with custom ToolNode
    tool_outputs = state["messages"]
    print(f'{state["messages"]=}')
    state["tool_messages"].append(tool_outputs[-1])
    state["messages"].clear()
    llm = get_llm_model("complex_model", streaming=True).with_structured_output(
        Decision
    )
    prompt = SystemMessage(
        f"""
        You are an expert in assessing wether the tool call was successful.
        Your answer should be one of the following:
        - tool_call_successful
        - call_tool
        available tools are: {render_text_description_and_args(state["available_tools"])}
        """
    )
    msgs = [
        prompt,
        HumanMessage(content="Previous tool calls were"),
        *state["tool_messages"],
        HumanMessage(
            content="Please tell me if the tool should be run once again, because it failed with an error."
        ),
    ]

    decision: Decision = llm.invoke(msgs)
    if decision.tool_call_successful:
        return "state_retriever"
    else:
        return "call_tool"


def call_tool(state):
    llm = get_llm_model("complex_model", streaming=True)
    llm = llm.bind_tools(state["available_tools"])
    system_prompt = SystemMessage(
        """ You are an expert in tool calling to achieve robotic tasks. Your responsibility is to call the tool so that the next step of the plan can be achieved. It might be that your previous tool call has not been successful. Then please try to correct yourself and call the tool correctly.
        You have information about:
        - current robot state
        - current task - the plan is to achieve the task
        - current plan - if task it True, than it has been already done.
        - previus tool calls
        """
    )
    msg = HumanMessage(
        content=f"""
    The task is: {state["task"]}
    Robot and environment state is: {state["state"]}
    The plan is: {state["plan"]}
    Previous tool calls were: {state["tool_messages"]}
    """
    )
    ai_msg = llm.invoke([system_prompt, msg])
    state["tool_messages"].append(ai_msg)
    return state


def decider(state):
    class Decision(BaseModel):
        decision: Literal["call_tool", "get_state", "end", "change_plan"] = Field(
            ..., description="Decision about the next step"
        )
        response: str = Field(..., description="Response")

    llm = get_llm_model("complex_model", streaming=True)
    system_prompt = SystemMessage(
        """ You are an expert in making decisions about the robot's actions.
        You have 3 information:
        - current robot state
        - current task - the plan is to achieve the task
        - current plan - if task it True, than it has been already done.
        - information if the plan is feasable
        You have to decide what the next step should be.
        Your answer should be one of the following:
        - call_tool
        - get_state
        - change_plan
        - end
        If tool run failed please decide to call the tool again.
        """
    )
    llm = llm.with_structured_output(Decision)
    msg = HumanMessage(
        content=f"""
    The task is: {state["task"]}
    Robot and environment state is: {state["state"]}
    The plan is: {state["plan"]}
    The plan is feasable: {state["plan_feasible"]}
    """
    )
    decision: Decision = llm.invoke([system_prompt, msg])
    state["result"] = decision.response
    return {
        "call_tool": "call_tool",
        "get_state": "state_retriever",
        "change_plan": "replanner",
        "end": "__end__",
    }[decision.decision]


def custom_tools_condition(
    state: Union[list[AnyMessage], dict[str, Any], BaseModel],
    messages_key: str = "tool_messages",
) -> Literal["tools", "decider"]:
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "state_retriever"


def replanner(state):
    return state


def create_state_based_runnable2(
    llm: Optional[BaseChatModel] = None,
    tools: Optional[List[BaseTool]] = None,
    system_prompt: Optional[str] = None,
    state_retriever: Optional[Callable[[], Dict[str, Any]]] = None,
) -> Runnable[ReActAgentState, ReActAgentState]:
    if llm is None:
        llm = get_llm_model("complex_model", streaming=True)
    if tools is None:
        tools = []

    graph = StateGraph(State)
    graph.add_edge(START, "planner")
    graph.add_node("planner", planner)
    graph.add_edge("planner", "state_retriever")
    graph.add_conditional_edges("state_retriever", decider)
    graph.add_node("call_tool", call_tool)
    graph.add_conditional_edges("call_tool", custom_tool_condition)
    tool_runner = ToolRunner(tools)
    graph.add_node("tools", tool_runner)
    graph.add_edge("tools", "state_retriever")
    graph.add_edge("replanner", "state_retriever")
    graph.add_node("state_retriever", partial(retriever_wrapper, state_retriever))
    graph.add_node("replanner", replanner)

    return graph.compile()
