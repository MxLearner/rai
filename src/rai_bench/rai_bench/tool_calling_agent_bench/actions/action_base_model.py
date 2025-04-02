from typing import Any, Optional

from pydantic import BaseModel


class Time(BaseModel):
    sec: Optional[int] = 0
    nanosec: Optional[int] = 0


class Header(BaseModel):
    stamp: Optional[Time] = Time()
    frame_id: str


class Position(BaseModel):
    x: float
    y: float
    z: float


class Orientation(BaseModel):
    x: Optional[float] = 0.0
    y: Optional[float] = 0.0
    z: Optional[float] = 0.0
    w: Optional[float] = 1.0


class Pose(BaseModel):
    position: Position
    orientation: Optional[Orientation] = Orientation()


class PoseStamped(BaseModel):
    header: Header
    pose: Pose


class ActionBaseModel(BaseModel):
    action_name: str
    action_type: str
    goal: Any
    result: Any
    feedback: Any
