from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field, create_engine


class ExperimentRun(SQLModel, table=True):
    id: str = Field(primary_key=True)
    dataset: str
    mode: str
    sigma: float
    clients: int
    final_accuracy: float
    final_epsilon: float
    created_at: datetime = Field(default_factory=datetime.utcnow)
    config_json: Optional[str] = Field(default=None)
    result_json: Optional[str] = Field(default=None)


engine = create_engine("sqlite:///experiments.db", echo=False)


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


