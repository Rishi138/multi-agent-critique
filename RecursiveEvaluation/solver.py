from .rec_agent import new_response
from dataclasses import asdict, dataclass, field
from typing import Any

# mini-extra swebench-single --model RecursiveEvaluation/solver.py

@dataclass
class SageConfig:
    model_name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)


class SageAgentModel:
    def __init__(self, **kwargs):
        self.config = SageConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0

    def _query(self, messages: list[dict[str, str]]):
        print("_query running")
        try:
            return new_response(messages)
        except Exception as e:
            raise e

    def query(self, messages: list[dict[str, str]]) -> dict:
        print("query running")
        response = self._query(messages)
        print("Response successful")
        self.n_calls += 1
        self.cost += 0.0
        return {"content": response}

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}
