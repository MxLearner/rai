from typing import Optional

from pydantic import BaseModel

from rai_bench.tool_calling_agent_bench.actions.action_base_model import (
    ActionBaseModel,
    PoseStamped,
    Time,
)


class Goal(BaseModel):
    pose: PoseStamped
    behavior_tree: Optional[str] = ""


class Result(BaseModel):
    result: dict


class Feedback(BaseModel):
    current_pose: PoseStamped
    navigation_time: Time
    estimated_time_remaining: Time
    number_of_recoveries: int
    distance_remaining: float


class NavigateToPoseAction(ActionBaseModel):
    action_name: str = "/navigate_to_pose"
    action_type: str = "nav2_msgs/action/NavigateToPose"
    goal: Goal
    result: Result
    feedback: Feedback


# TODO (mkotynia): Everything should be Optional?
