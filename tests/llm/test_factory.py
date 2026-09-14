from regulars.llm import (
    CENTAUR_MODEL,
    DEFAULT_CHEAP_MODEL,
    DEFAULT_SCORING_MODEL,
    AnthropicLLM,
    OpenAICompatibleLLM,
    llm_from_env,
)

BASE_ENV = {"ANTHROPIC_API_KEY": "test-key"}


def test_defaults_without_enactment_endpoint() -> None:
    scoring = llm_from_env("scoring", BASE_ENV)
    cheap = llm_from_env("cheap", BASE_ENV)
    enactment = llm_from_env("enactment", BASE_ENV)
    assert isinstance(scoring, AnthropicLLM) and scoring.model == DEFAULT_SCORING_MODEL
    assert isinstance(cheap, AnthropicLLM) and cheap.model == DEFAULT_CHEAP_MODEL
    assert isinstance(enactment, AnthropicLLM) and enactment.model == DEFAULT_SCORING_MODEL
    assert scoring.cache_system is True
    assert DEFAULT_SCORING_MODEL == "claude-opus-5"
    assert DEFAULT_CHEAP_MODEL == "claude-haiku-4-5"


def test_env_overrides_models() -> None:
    env = {
        **BASE_ENV,
        "REGULARS_SCORING_MODEL": "claude-opus-4-6",
        "REGULARS_CHEAP_MODEL": "claude-sonnet-5",
    }
    assert llm_from_env("scoring", env).model == "claude-opus-4-6"
    assert llm_from_env("cheap", env).model == "claude-sonnet-5"
    assert llm_from_env("enactment", env).model == "claude-opus-4-6"


def test_enactment_uses_openai_compatible_when_base_url_set() -> None:
    env = {**BASE_ENV, "REGULARS_ENACTMENT_BASE_URL": "http://vllm.local:8000/v1"}
    llm = llm_from_env("enactment", env)
    assert isinstance(llm, OpenAICompatibleLLM)
    assert llm.model == CENTAUR_MODEL
    assert llm.base_url == "http://vllm.local:8000/v1"
    # scoring stays on Anthropic regardless of the enactment endpoint (§2: Centaur enacts,
    # frontier scores)
    assert isinstance(llm_from_env("scoring", env), AnthropicLLM)


def test_enactment_model_override_with_base_url() -> None:
    env = {
        "REGULARS_ENACTMENT_BASE_URL": "http://tgi:8080/v1",
        "REGULARS_ENACTMENT_MODEL": "some/other-model",
    }
    llm = llm_from_env("enactment", env)
    assert isinstance(llm, OpenAICompatibleLLM)
    assert llm.model == "some/other-model"


def test_enactment_model_override_without_base_url_is_anthropic() -> None:
    env = {**BASE_ENV, "REGULARS_ENACTMENT_MODEL": "claude-sonnet-5"}
    llm = llm_from_env("enactment", env)
    assert isinstance(llm, AnthropicLLM)
    assert llm.model == "claude-sonnet-5"
