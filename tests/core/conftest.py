import asyncio
import os
from typing import Text

import matplotlib
import pytest

import rasa.utils.io
from rasa.core.agent import Agent
from rasa.core.channels.channel import CollectingOutputChannel
from rasa.core.domain import Domain
from rasa.core.interpreter import RegexInterpreter
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.core.policies.ensemble import PolicyEnsemble, SimplePolicyEnsemble
from rasa.core.policies.memoization import (
    AugmentedMemoizationPolicy,
    Policy,
    MemoizationPolicy,
)
from rasa.core.processor import MessageProcessor
from rasa.core.slots import Slot
from rasa.core.tracker_store import InMemoryTrackerStore
from rasa.core.trackers import DialogueStateTracker
from rasa.train import train_async

matplotlib.use("Agg")

DEFAULT_DOMAIN_PATH_WITH_SLOTS = "data/test_domains/default_with_slots.yml"

DEFAULT_DOMAIN_PATH_WITH_MAPPING = "data/test_domains/default_with_mapping.yml"

DEFAULT_STORIES_FILE = "data/test_stories/stories_defaultdomain.md"

DEFAULT_STACK_CONFIG = "data/test_config/stack_config.yml"

DEFAULT_NLU_DATA = "examples/moodbot/data/nlu.md"

END_TO_END_STORY_FILE = "data/test_evaluations/end_to_end_story.md"

E2E_STORY_FILE_UNKNOWN_ENTITY = "data/test_evaluations/story_unknown_entity.md"

MOODBOT_MODEL_PATH = "examples/moodbot/models/"

RESTAURANTBOT_PATH = "examples/restaurantbot/"

DEFAULT_ENDPOINTS_FILE = "data/test_endpoints/example_endpoints.yml"

TEST_DIALOGUES = [
    "data/test_dialogues/default.json",
    "data/test_dialogues/formbot.json",
    "data/test_dialogues/moodbot.json",
    "data/test_dialogues/restaurantbot.json",
]

EXAMPLE_DOMAINS = [
    DEFAULT_DOMAIN_PATH_WITH_SLOTS,
    DEFAULT_DOMAIN_PATH_WITH_MAPPING,
    "examples/formbot/domain.yml",
    "examples/moodbot/domain.yml",
    "examples/restaurantbot/domain.yml",
]


class CustomSlot(Slot):
    def as_feature(self):
        return [0.5]


# noinspection PyAbstractClass,PyUnusedLocal,PyMissingConstructor
class ExamplePolicy(Policy):
    def __init__(self, example_arg):
        pass


@pytest.fixture
def default_channel():
    return CollectingOutputChannel()


@pytest.fixture
async def default_processor(default_domain, default_nlg):
    agent = Agent(
        default_domain,
        SimplePolicyEnsemble([AugmentedMemoizationPolicy()]),
        interpreter=RegexInterpreter(),
    )

    training_data = await agent.load_data(DEFAULT_STORIES_FILE)
    agent.train(training_data)
    tracker_store = InMemoryTrackerStore(default_domain)
    return MessageProcessor(
        agent.interpreter,
        agent.policy_ensemble,
        default_domain,
        tracker_store,
        default_nlg,
    )


@pytest.fixture
def moodbot_domain(trained_moodbot_path):
    domain_path = os.path.join("examples", "moodbot", "domain.yml")
    return Domain.load(domain_path)


@pytest.fixture
def moodbot_metadata(unpacked_trained_moodbot_path):
    return PolicyEnsemble.load_metadata(
        os.path.join(unpacked_trained_moodbot_path, "core")
    )


@pytest.fixture
def default_nlg(default_domain):
    return TemplatedNaturalLanguageGenerator(default_domain.templates)


@pytest.fixture
def default_tracker(default_domain):
    return DialogueStateTracker("my-sender", default_domain.slots)


@pytest.fixture
async def restaurantbot(tmpdir_factory) -> Text:
    model_path = tmpdir_factory.mktemp("model").strpath
    restaurant_domain = os.path.join(RESTAURANTBOT_PATH, "domain.yml")
    restaurant_config = os.path.join(RESTAURANTBOT_PATH, "config.yml")
    restaurant_data = os.path.join(RESTAURANTBOT_PATH, "data/")

    agent = await train_async(
        restaurant_domain, restaurant_config, restaurant_data, model_path
    )
    return agent
