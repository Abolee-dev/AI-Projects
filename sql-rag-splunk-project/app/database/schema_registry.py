import yaml
from app.config import DATASET_REGISTRY_PATH

class SchemaRegistry:
    def __init__(self, registry_path: str = DATASET_REGISTRY_PATH):
        self.registry_path = registry_path
        self._registry = self._load()

    def _load(self) -> dict:
        with open(self.registry_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def allowed_tables(self) -> list[str]:
        return [
            name for name, meta in self._registry.get("datasets", {}).items()
            if meta.get("allowed_for_sqlrag") is True
        ]

    def table_schema_text(self, table_name: str) -> str:
        dataset = self._registry["datasets"][table_name]
        lines = [f"Table: {table_name}", f"Description: {dataset.get('description', '')}"]
        for col in dataset.get("columns", []):
            lines.append(f"- {col['name']} ({col['type']}): {col.get('description', '')}")
        return "\n".join(lines)

    def all_schema_text(self) -> str:
        return "\n\n".join(self.table_schema_text(t) for t in self.allowed_tables())
