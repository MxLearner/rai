from typing import Generator, List

from cv_bridge import CvBridge
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

from rai.communication.ros2.api import TopicConfig
from rai.communication.ros2.connectors import ROS2ARIConnector
from rai.messages.multimodal import HumanMultimodalMessage
from rai.messages.utils import preprocess_image
from rai.utils.model_initialization import get_llm_model


def ros2_string_msgs_aggregator(msgs_stream: Generator[String, None, None]):
    """Returns only unique messages from stream keeping their"""
    buffer = []
    for msg in msgs_stream:
        buffer.append(msg.data)
    return list(dict.fromkeys(buffer))


class Ros2ImgDiffOutput(BaseModel):
    are_different: bool = Field(..., description="Whether the images are different")
    differences: List[str] = Field(..., description="Description of the difference")


class Ros2ImgDescription(BaseModel):
    key_elements: List[str] = Field(..., description="Key elements of the image")


class ros2_img_vlm_diff:
    @staticmethod
    def parse_ros2_img(message: Image | CompressedImage) -> str:
        msg_type = type(message)
        if msg_type == Image:
            image = CvBridge().imgmsg_to_cv2(  # type: ignore
                message, desired_encoding="rgb8"
            )
        elif msg_type == CompressedImage:
            image = CvBridge().compressed_imgmsg_to_cv2(  # type: ignore
                message, desired_encoding="rgb8"
            )
        else:
            raise ValueError(
                f"Unsupported message type: {message.metadata['msg_type']}"
            )
        return preprocess_image(image)

    @staticmethod
    def get_key_elements(lst):
        if len(lst) <= 3:
            return lst
        middle_index = len(lst) // 2
        return [lst[0], lst[middle_index], lst[-1]]

    def __call__(self, msgs_stream: Generator[Image, None, None]):
        llm = get_llm_model(model_type="simple_model", streaming=True)
        images = [self.parse_ros2_img(msg) for msg in msgs_stream]
        images = self.get_key_elements(images)
        if len(images) == 0:
            return "No images"
        print(f"Processing {len(images)} images")

        system_prompt = "You are an expert in image analysis and your speciality is the comparison of 2 images"

        task = [
            SystemMessage(content=system_prompt),
            HumanMultimodalMessage(
                content="Describe key elements that are currently in robot's view",
                images=[images[-1]],
            ),
        ]
        output1: Ros2ImgDescription = llm.with_structured_output(
            Ros2ImgDescription
        ).invoke(task)

        # task = [
        #     SystemMessage(content=system_prompt),
        #     HumanMultimodalMessage(
        #         content=f"Here are max 3 subsequent images from the robot camera. Robot might be moving. Outline key differences in robot's view.",
        #         images=images,
        #     ),
        # ]
        # output2: Ros2ImgDiffOutput = llm.with_structured_output(Ros2ImgDiffOutput).invoke(task)

        return output1


class Aggregator:
    def __init__(self) -> None:
        self.watched_sources = [
            (
                "/string_topic",
                TopicConfig(
                    msg_type="std_msgs/msg/String",
                    auto_qos_matching=True,
                    is_subscriber=True,
                ),
            ),
            (
                "/camera/camera/color/image_raw",
                TopicConfig(
                    msg_type="sensor_msgs/msg/Image",
                    auto_qos_matching=True,
                    is_subscriber=True,
                ),
            ),
        ]
        self.aggregator = [
            ros2_string_msgs_aggregator,
            ros2_img_vlm_diff(),
        ]

        self.connector = ROS2ARIConnector(
            use_configurable_topics_api=True, sources=self.watched_sources
        )
        self.agg_results = dict()
        # self.connector.node.create_timer(5.0, self.aggregate, callback_group=ReentrantCallbackGroup())
        # for topic_nam, _ in self.watched_sources:
        #     self.agg_results[topic_nam] = list()
        # TODO(boczekbartek): add max_size

    def aggregate(self):
        print("aggregating data")

        for (topic, _), agg in zip(self.watched_sources, self.aggregator):
            print(f"Getting topic: {topic}")
            msgs = list(self.connector.receive_all(topic))
            self.agg_results[topic] = agg(msgs)
        return self.agg_results
        # TODO(bnoczekbartek): define interface here
