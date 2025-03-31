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

import logging
from functools import partial
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

loggers_type = Union[logging.Logger]


class DualAgentState(TypedDict):
    vision_messages: List[BaseMessage]
    tool_messages: List[BaseMessage]
    vision_response: Optional[str]
    original_question: Optional[str]
    final_response: Optional[Dict[str, Any]]
    tool_results: Dict[str, Any]


def extract_text_from_message(message):
    """Extract plain text from a message that might be multimodal."""
    if hasattr(message, "content") and isinstance(message.content, list):
        # Handle multimodal messages
        for content_part in message.content:
            if content_part.get("type") == "text":
                return content_part.get("text", "")
        return ""
    elif hasattr(message, "content"):
        # Handle simple text messages
        return message.content
    return ""


def vision_agent(
    vision_llm: BaseChatModel,
    logger: loggers_type,
    system_prompt: str,
    state: DualAgentState,
):
    """The vision agent processes images and provides descriptions."""
    logger.info("Running vision agent")

    # Skip if we already have a vision processing result
    if state["vision_response"] is not None:
        return state

    messages = state["vision_messages"]

    # If there are no messages, do nothing
    if len(messages) == 0:
        return state

    # Save the original question from the first human message
    if state["original_question"] is None and len(messages) > 0:
        # Extract the text from the first message (which might be multimodal)
        original_question = extract_text_from_message(messages[0])
        state["original_question"] = original_question
        logger.info(f"Original question: {original_question}")

    # Insert system message if not already present
    if not isinstance(messages[0], SystemMessage):
        messages.insert(0, SystemMessage(content=system_prompt))

    # Invoke the vision model
    ai_msg = vision_llm.invoke(messages)
    state["vision_messages"].append(ai_msg)

    # Store the response for the tool agent to use
    state["vision_response"] = ai_msg.content

    return state


def tool_agent(
    tool_llm: BaseChatModel,
    logger: loggers_type,
    system_prompt: str,
    state: DualAgentState,
):
    """The tool agent takes the vision analysis and makes tool calls."""
    logger.info("Running tool agent")

    # If there's no vision response yet, we can't proceed
    if state["vision_response"] is None:
        return state

    messages = state["tool_messages"]

    # Insert system message if not already present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages.insert(0, SystemMessage(content=system_prompt))

    # Add the original question and vision processing result as context in a human message if not already present
    if len(messages) == 1:  # Only system message present
        original_question = state["original_question"] or "Please analyze the image."
        vision_context = (
            f"Original question: {original_question}\n\n"
            f"Image analysis: {state['vision_response']}\n\n"
            f"Please use the appropriate tool to respond to the original question based on the image analysis."
        )
        messages.append(HumanMessage(content=vision_context))

    # Invoke the tool model
    ai_msg = tool_llm.invoke(messages)
    state["tool_messages"].append(ai_msg)

    return state


def run_tools(state: DualAgentState, tools: List[BaseTool], logger: loggers_type):
    """Execute tools based on the tool calls in the messages."""
    logger.info("Running tools")

    if not state["tool_messages"]:
        return state

    # Find the last AI message that contains tool calls
    ai_messages = [msg for msg in state["tool_messages"] if isinstance(msg, AIMessage)]

    if not ai_messages:
        return state

    last_ai_msg = ai_messages[-1]

    # Check if the message has tool calls
    if not hasattr(last_ai_msg, "tool_calls") or not last_ai_msg.tool_calls:
        return state

    # Dictionary to store tool results
    tool_results = {}

    # Execute each tool call and add tool messages to respond to each call
    for tool_call in last_ai_msg.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        # Find the tool by name
        matching_tools = [tool for tool in tools if tool.name == tool_name]
        if matching_tools:
            try:
                # Execute the tool
                result = matching_tools[0]._run(**tool_args)
                tool_results[tool_call_id] = result
                logger.info(f"Tool {tool_name} executed with result: {result}")

                # Add a tool message with the result - IMPORTANT for OpenAI API
                state["tool_messages"].append(
                    ToolMessage(
                        content=str(result), tool_call_id=tool_call_id, name=tool_name
                    )
                )

            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                error_message = f"Error: {str(e)}"
                tool_results[tool_call_id] = error_message

                # Add a tool message with the error
                state["tool_messages"].append(
                    ToolMessage(
                        content=error_message, tool_call_id=tool_call_id, name=tool_name
                    )
                )
        else:
            logger.error(f"Tool {tool_name} not found")
            error_message = f"Error: Tool {tool_name} not found"
            tool_results[tool_call_id] = error_message

            # Add a tool message with the error
            state["tool_messages"].append(
                ToolMessage(
                    content=error_message, tool_call_id=tool_call_id, name=tool_name
                )
            )

    # Store the results in the state
    state["tool_results"] = tool_results
    state["final_response"] = tool_results

    return state


def summarize_result(
    tool_llm: BaseChatModel, logger: loggers_type, state: DualAgentState
):
    """Summarize the tool execution results."""
    logger.info("Summarizing results")

    if not state["tool_results"]:
        return state

    # Add a final prompt asking the model to summarize the results
    original_question = state["original_question"] or "the question"
    state["tool_messages"].append(
        HumanMessage(
            content=f"Based on the tool execution results, please provide a final answer to: {original_question}"
        )
    )

    # Get a final summary from the model
    summary_msg = tool_llm.invoke(state["tool_messages"])
    state["tool_messages"].append(summary_msg)

    return state


def router(state: DualAgentState) -> Literal["tool_agent", "tools", "summarize", "end"]:
    """Determine the next node in the graph."""
    # If there's no vision response yet, go to vision agent
    if state["vision_response"] is None:
        return "tool_agent"

    # Get the tool messages
    tool_messages = state["tool_messages"]
    if not tool_messages:
        return "tool_agent"

    # Find the last AI message
    ai_messages = [msg for msg in tool_messages if isinstance(msg, AIMessage)]
    if not ai_messages:
        return "tool_agent"

    last_ai_msg = ai_messages[-1]

    # Check if the last AI message has tool calls
    if hasattr(last_ai_msg, "tool_calls") and last_ai_msg.tool_calls:
        # Check if we've already executed these tools
        tool_call_ids = [tool_call["id"] for tool_call in last_ai_msg.tool_calls]
        tool_message_ids = [
            msg.tool_call_id
            for msg in tool_messages
            if isinstance(msg, ToolMessage) and hasattr(msg, "tool_call_id")
        ]

        if all(tool_id in tool_message_ids for tool_id in tool_call_ids):
            # We've already executed all tools in this message, get a summary
            if len(ai_messages) > len(
                [msg for msg in tool_messages if isinstance(msg, ToolMessage)]
            ):
                # If we have more AI messages than tool messages, we've already summarized
                return "end"
            return "summarize"
        else:
            # We haven't executed all tools yet
            return "tools"

    # If we've already processed tools and added a summary, end
    if state.get("final_response") is not None:
        return "end"

    return "end"


def create_dual_agent(
    vision_llm: BaseChatModel,
    tool_llm: BaseChatModel,
    tools: List[BaseTool],
    vision_system_prompt: str,
    tool_system_prompt: str,
    logger: Optional[loggers_type] = None,
    debug=False,
) -> CompiledStateGraph:
    """Create a dual agent with separate vision and tool calling components."""
    _logger = logger if logger else logging.getLogger(__name__)
    _logger.info("Creating dual agent")

    # Bind tools to the tool LLM
    tool_llm_with_tools = tool_llm.bind_tools(tools)

    # Create the workflow graph
    workflow = StateGraph(DualAgentState)

    # Add the nodes
    workflow.add_node(
        "vision_agent", partial(vision_agent, vision_llm, _logger, vision_system_prompt)
    )
    workflow.add_node(
        "tool_agent",
        partial(tool_agent, tool_llm_with_tools, _logger, tool_system_prompt),
    )
    workflow.add_node("tools", partial(run_tools, tools=tools, logger=_logger))
    workflow.add_node("summarize", partial(summarize_result, tool_llm, _logger))

    # Add edges
    workflow.add_edge(START, "vision_agent")
    workflow.add_edge("vision_agent", "tool_agent")

    # Add conditional edges from tool_agent
    workflow.add_conditional_edges(
        "tool_agent",
        router,
        {
            "tool_agent": "tool_agent",
            "tools": "tools",
            "summarize": "summarize",
            "end": END,
        },
    )

    # Add edges from tools and summarize
    workflow.add_edge("tools", "tool_agent")
    workflow.add_edge("summarize", END)

    # Compile the workflow
    app = workflow.compile(debug=debug)
    _logger.info("Dual agent created")
    return app
