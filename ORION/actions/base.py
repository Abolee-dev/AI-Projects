from abc import ABC, abstractmethod
from typing import Any

from models.schemas import ActionPayload


class BaseAction(ABC):
    @abstractmethod
    async def execute(self, payload: ActionPayload) -> dict[str, Any]: ...

    @abstractmethod
    def can_handle(self, action_type: str) -> bool: ...
