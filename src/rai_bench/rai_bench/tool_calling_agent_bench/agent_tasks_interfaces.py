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
from abc import ABC, abstractmethod
from typing import Any, List, Literal

from langchain_core.messages import AIMessage, ToolCall
from langchain_core.runnables.config import DEFAULT_RECURSION_LIMIT
from langchain_core.tools import BaseTool
from pydantic import BaseModel

loggers_type = logging.Logger
ANY_VALUE = object()


class Result(BaseModel):
    success: bool = False
    errors: list[str] = []


class ToolCallingAgentTask(ABC):
    """Abstract class for tool calling agent tasks. Contains methods for requested tool calls verification.

    Parameters
    ----------
    logger : loggers_type | None, optional
        Logger, by default None
    """

    complexity: Literal["easy", "medium", "hard"]
    recursion_limit: int = DEFAULT_RECURSION_LIMIT

    def __init__(
        self,
        logger: loggers_type | None = None,
    ) -> None:
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
        self.expected_tools: List[BaseTool] = []
        self.result = Result()

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt that will be passed to agent

        Returns
        -------
        str
            System prompt
        """
        pass

    @abstractmethod
    def get_prompt(self) -> str:
        """Get the task instruction - the prompt that will be passed to agent.

        Returns
        -------
        str
            Prompt
        """
        pass

    @abstractmethod
    def verify_tool_calls(self, response: dict[str, Any]):
        """Verify correctness of the tool calls from the agent's response.

        Note
        ----
        This method should set self.result.success to True if the verification is successful and append occuring errors related to verification to self.result.errors.

        Parameters
        ----------
        response : dict[str, Any]
            Agent's response
        """
        pass

    def _check_tool_call(
        self,
        tool_call: ToolCall,
        expected_name: str,
        expected_args: dict[str, Any],
        expected_optional_args: dict[str, Any] = {},
    ) -> bool:
        """Check if a tool call matches expected parameters including nested dictionaries.

        Parameters
        ----------
        tool_call : ToolCall
            The tool call to check
        expected_name : str
            Expected name of the tool
        expected_args : dict[str, Any]
            Required arguments with their expected values (use ANY_VALUE to skip value check)
        expected_optional_args : dict[str, Any]
            Optional arguments with their expected values (use ANY_VALUE to skip value check)

        Returns
        -------
        bool
            True if the tool call matches expected parameters, False otherwise
        """
        # Check tool name
        if tool_call["name"] != expected_name:
            # self.log_error(
            #     msg=f"Expected tool call name should be '{expected_name}', but got {tool_call['name']}"
            # )
            return False

        # Recursively check arguments
        return self._check_args_structure(
            args=tool_call["args"],
            expected_args=expected_args,
            expected_optional_args=expected_optional_args,
            path="args",
        )

    def _check_multiple_tool_calls(
        self, message: AIMessage, expected_tool_calls: list[dict[str, Any]]
    ) -> bool:
        """Helper method to check multiple tool calls in a single AIMessage.

        Parameters
        ----------
        message : AIMessage
            The AIMessage to check
        expected_tool_calls : list[dict[str, Any]]
            A list of dictionaries, each containing expected 'name', 'args', and optional 'optional_args' for a tool call

        Returns
        -------
        bool
            True if all tool calls match expected patterns, False otherwise
        """
        if not self._check_tool_calls_num_in_ai_message(
            message, len(expected_tool_calls)
        ):
            return False

        matched_calls = [False] * len(expected_tool_calls)
        error_occurs = False

        for tool_call in message.tool_calls:
            found_match = False

            for i, expected in enumerate(expected_tool_calls):
                if matched_calls[i]:
                    continue

                expected_name = expected["name"]
                expected_args = expected["args"]
                expected_optional_args = expected.get("optional_args", {})

                if self._check_tool_call(
                    tool_call=tool_call,
                    expected_name=expected_name,
                    expected_args=expected_args,
                    expected_optional_args=expected_optional_args,
                ):
                    matched_calls[i] = True
                    found_match = True
                    break

            if not found_match:
                self.log_error(
                    msg=f"Tool call {tool_call['name']} with args {tool_call['args']} does not match any expected call"
                )
                error_occurs = True

        return not error_occurs

    # TODO (mkotynia): refactor the code to leave only one method for checking multiple tool calls
    def _check_multiple_tool_calls_from_list(
        self, tool_calls: list[ToolCall], expected_tool_calls: list[dict[str, Any]]
    ) -> bool:
        """Helper method to check multiple tool calls from a list of tool calls.

        Parameters
        ----------
        tool_calls : list[ToolCall]
            The list of tool calls to check
        expected_tool_calls : list[dict[str, Any]]
            A list of dictionaries, each containing expected 'name', 'args', and optional 'optional_args' for a tool call

        Returns
        -------
        bool
            True if all expected tool calls are found in the tool calls, False otherwise
        """
        matched_calls = [False] * len(expected_tool_calls)
        error_occurs = False

        for tool_call in tool_calls:
            for i, expected in enumerate(expected_tool_calls):
                if matched_calls[i]:
                    continue

                expected_name = expected["name"]
                expected_args = expected["args"]
                expected_optional_args = expected.get("optional_args", {})

                if self._check_tool_call(
                    tool_call=tool_call,
                    expected_name=expected_name,
                    expected_args=expected_args,
                    expected_optional_args=expected_optional_args,
                ):
                    matched_calls[i] = True
                    break

        for i, matched in enumerate(matched_calls):
            if not matched:
                self.log_error(
                    msg=f"Expected tool call '{expected_tool_calls[i]['name']}' with args {expected_tool_calls[i]['args']} was not found in the provided tool calls."
                )
                error_occurs = True

        return not error_occurs

    def _check_args_structure(
        self,
        args: dict[str, Any],
        expected_args: dict[str, Any],
        expected_optional_args: dict[str, Any] = {},
        path: str = "",
    ) -> bool:
        """Recursively check nested argument structure.

        Parameters
        ----------
        args : dict[str, Any]
            Arguments to check
        expected_args : dict[str, Any]
            Required arguments with their expected values
        expected_optional_args : dict[str, Any]
            Optional arguments with their expected values
        path : str
            Path to current args for error reporting

        Returns
        -------
        bool
            True if args match expected structure, False otherwise
        """
        # Check all required arguments are present with expected values
        for arg_name, expected_value in expected_args.items():
            if arg_name not in args:
                # self.log_error(
                #     msg=f"Required argument '{path}.{arg_name}' is missing"
                # )
                return False

            arg_value = args[arg_name]

            # If expected value is a dictionary, recursively check nested structure
            if (
                isinstance(expected_value, dict)
                and isinstance(arg_value, dict)
                and expected_value != ANY_VALUE
            ):
                # Get optional nested args if they exist
                optional_nested = {}
                if arg_name in expected_optional_args and isinstance(
                    expected_optional_args[arg_name], dict
                ):
                    optional_nested = expected_optional_args[arg_name]

                if not self._check_args_structure(
                    args=arg_value,
                    expected_args=expected_value,
                    expected_optional_args=optional_nested,
                    path=f"{path}.{arg_name}",
                ):
                    return False
            # Otherwise check if value matches (skip if ANY_VALUE)
            elif expected_value is not ANY_VALUE and arg_value != expected_value:
                # self.log_error(
                #     msg=f"Argument '{path}.{arg_name}' should have value '{expected_value}', but got '{arg_value}'"
                # )
                return False

        # Check no unexpected arguments are present
        for arg_name, arg_value in args.items():
            # Skip if this is a required argument (already checked above)
            if arg_name in expected_args:
                continue

            # Check if it's an allowed optional argument
            if arg_name not in expected_optional_args:
                # self.log_error(msg=f"Unexpected argument '{path}.{arg_name}' found")
                return False

            # Get the expected value for this optional argument
            expected_value = expected_optional_args[arg_name]

            # Skip value check if ANY_VALUE
            if expected_value is ANY_VALUE:
                continue

            # Handle nested dictionaries in optional args
            if isinstance(expected_value, dict) and isinstance(arg_value, dict):
                # For optional nested structures, all keys within are considered optional
                # unless explicitly specified as required in the expected structure
                nested_required = {}
                nested_optional = expected_value

                if not self._check_args_structure(
                    args=arg_value,
                    expected_args=nested_required,
                    expected_optional_args=nested_optional,
                    path=f"{path}.{arg_name}",
                ):
                    return False
            # Check if value matches for non-nested optional args
            elif arg_value is not ANY_VALUE and arg_value != expected_value:
                # self.log_error(
                #     msg=f"Optional argument '{path}.{arg_name}' should have value '{expected_value}', but got '{arg_value}'"
                # )
                return False

        return True

    def _check_tool_calls_num_in_ai_message(
        self, message: AIMessage, expected_num: int
    ) -> bool:
        """Helper method to check number of tool calls in a single AIMessage.

        Parameters
        ----------
        message : AIMessage
            The AIMessage to check
        expected_num : int
            The expected number of tool calls

        Returns
        -------
        bool
            True if the number of tool calls in the message matches the expected number, False otherwise
        """
        if len(message.tool_calls) != expected_num:
            self.log_error(
                msg=f"Expected number of tool calls should be {expected_num}, but got {len(message.tool_calls)}"
            )
            return False
        return True

    def log_error(self, msg: str):
        self.logger.error(msg)
        self.result.errors.append(msg)


class ROS2ToolCallingAgentTask(ToolCallingAgentTask, ABC):
    """Abstract class for ROS2 related tasks for tool calling agent.

    Parameters
    ----------
    logger : loggers_type | None
        Logger for the task.
    """

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger)

    def _is_ai_message_requesting_get_ros2_topics_and_types(
        self, ai_message: AIMessage
    ) -> bool:
        """Helper method to check if the given AIMessage is calling the exactly one tool that gets ROS2 topics names and types correctly.

        Parameters
        ----------
        ai_message : AIMessage
            The AIMessage to check

        Returns
        -------
        bool
            True if the ai_message is requesting get_ros2_topics_names_and_types correctly, False otherwise
        """
        if not self._check_tool_calls_num_in_ai_message(ai_message, expected_num=1):
            return False

        tool_call: ToolCall = ai_message.tool_calls[0]
        if not self._check_tool_call(
            tool_call=tool_call,
            expected_name="get_ros2_topics_names_and_types",
            expected_args={},
        ):
            return False
        return True
