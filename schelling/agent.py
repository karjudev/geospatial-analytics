from abc import abstractmethod
import random
from typing import Dict, List, Optional, Tuple, Union
from mesa.model import Model
from mesa.agent import Agent
import numpy as np
from pandas import DataFrame
from catboost import CatBoostClassifier


class SchellingAgent(Agent):
    """Agent in a Schelling model."""

    def __init__(
        self, unique_id: int, model: Model, pos: Tuple[int, int], is_minority: bool
    ) -> None:
        """Creates a new agent of the Schelling model.

        :param unique_id: Unique agent ID.
        :type unique_id: int
        :param model: Schelling model.
        :type model: SchellingModel
        :param pos: Coordinates of the agent in the Schelling grid.
        :type pos: Tuple[int, int]
        :param is_minority: If the model is in the minority class.
        :type is_minority: bool
        """
        # Checks the correctness of the parameters
        assert isinstance(unique_id, int), "Unique agent ID must be an integer."
        assert isinstance(
            model, Model
        ), "Model has to be (a sub-class of) a MESA model."
        assert isinstance(pos, Tuple) and list(map(type, pos)) == [
            int,
            int,
        ], "Position has to be an integer tuple of (x, y) coordinates."
        assert isinstance(is_minority, bool) or isinstance(
            is_minority, np.bool8
        ), "The minority flag has to be a boolean."

        # Creates the model
        super().__init__(unique_id, model)
        self.pos = pos
        self.is_minority = is_minority
        # Number of steps in the current location
        self.steps: int = 1
        # Number of relocations of the agent
        self.relocations: int = 0
        # Segregation of the agent in the current location
        self.segregation: float = None
        # Densi
        # Minimum and maximum segregation in the current location
        self.segregation_min: float = float("inf")
        self.segregation_max: float = 0.0
        # Sum of the segregation in the current location (needed to compute the average)
        self.segregation_sum: float = 0.0
        # List of relocation records
        self.relocation_records: List[Dict] = []

    def step(self) -> None:
        """Performs a step of the agent in the simulation."""
        # Computes the current segregation
        self.segregation = self.model.compute_segregation(self.pos, self.is_minority)
        self.segregation_sum += self.segregation
        self.segregation_min = min(self.segregation_min, self.segregation)
        self.segregation_max = max(self.segregation_max, self.segregation)
        # If the agent is not happy relocates
        if self.model.is_agent_happy(self.segregation):
            self.steps += 1
            self.model.total_happy += 1
        else:
            self.relocate()

    @abstractmethod
    def choose_cell(self, cells: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Chooses a cell to relocate the agent.

        :param cells: List of cells in which the agent will be happy.
        :type cells: List[Tuple[int, int]]
        :return: Coordinates of the cell.
        :rtype: Tuple[int, int]
        """
        pass

    def get_relocation_record(
        self, pos: Tuple[int, int]
    ) -> Dict[str, Union[int, float]]:
        """Creates a relocation record.

        :param pos: Destination position.
        :type pos: Tuple[int, int]
        :return: Relocation features, used to train a MLSchelling model or to predict happyness likelihood.
        :rtype: Dict[str, Union[int, float]]
        """
        # Checks that the coordinates are given in a tuple
        assert isinstance(pos, Tuple) and list(map(type, pos)) == [
            int,
            int,
        ], "Position has to be an integer tuple of (x, y) coordinates."
        # Agent-level features
        features: Dict[str, Union[int, float]] = self.get_agent_features()
        # Start (current) location features
        for key, value in self.model.get_location_features(self.pos).items():
            features["start_" + key] = value
        # End (next) location features
        for key, value in self.model.get_location_features(pos).items():
            features["end_" + key] = value
        return features

    def append_relocation_record(self, pos: Tuple[int, int]) -> None:
        """Appends a new relocation record and sets the `happyness_duration` label to the previous record.

        :param pos: Position of the destination record.
        :type pos: Tuple[int, int]
        """
        # Checks pos validity
        assert isinstance(pos, Tuple) and list(map(type, pos)) == [
            int,
            int,
        ], "Position has to be an integer tuple of (x, y) coordinates."
        record: Dict[str, Union[int, float]] = self.get_relocation_record(pos)
        if len(self.relocation_records) > 0:
            self.relocation_records[-1]["happyness_duration"] = self.steps
        self.relocation_records.append(record)

    def relocate(self) -> None:
        """Moves the agent to a random cell when it will be happy."""
        # List of free cells where the relocation will make the agent happy
        better_cells: List[Tuple[int, int]] = []
        for pos in self.model.iterate_empty_cells():
            segregation = self.model.compute_segregation(pos, self.is_minority)
            if self.model.is_agent_happy(segregation):
                better_cells.append(pos)
        if len(better_cells) == 0:
            return
        # Random choice beneath the cells
        pos: Tuple[int, int] = self.choose_cell(better_cells)
        # Adds a relocation record to the history
        self.append_relocation_record(pos)
        # Relocates to the cell
        self.model.grid.move_agent(self, pos)
        # Increments the total number of relocations
        self.relocations += 1
        # Resets the number of steps in the current location
        self.steps = 1
        # Resets the variables related to the segregation
        self.segregation = None
        self.segregation_min = float("inf")
        self.segregation_max = 0.0
        self.segregation_sum = 0.0

    def get_agent_features(self) -> Dict[str, Union[float, int]]:
        """Gets the agent features needed by the machine learning model.

        :return: Features of the agent in the current location.
        :rtype: Dict[str, float]
        """
        return {
            "steps": self.steps,
            "relocations": self.relocations,
            "segregation": self.segregation,
            "segregation_min": self.segregation_min,
            "segregation_max": self.segregation_max,
            "segregation_avg": self.segregation_sum / self.steps,
        }

    def collect_relocation_dataframe(self) -> DataFrame:
        """Creates the relocation dataframe for the agent in the current run.
        Also, appends the last `happyness_duration` label.

        :return: Dataframe with all the relocation records.
        :rtype: DataFrame
        """
        if len(self.relocation_records) > 0:
            self.relocation_records[-1]["happyness_duration"] = self.steps
        df: DataFrame = DataFrame.from_records(self.relocation_records)
        # Dataframe might be empty, hence the transformation into percentage hasn't to be done
        if "happyness_duration" in df:
            df["happyness_duration"] = (
                df["happyness_duration"] / self.model.schedule.steps
            )
        return df


class MildRandomSchellingAgent(SchellingAgent):
    """Schelling agent that moves itself based on a mild random policy,
    i.e. chooses a random chell among the ones in which it will be happy."""

    def __init__(
        self,
        unique_id: int,
        model: Model,
        pos: Tuple[int, int],
        is_minority: bool,
    ) -> None:
        super().__init__(unique_id, model, pos, is_minority)

    def choose_cell(self, cells: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        assert len(cells) > 0, "There are no valid cells"
        return random.choice(cells)


class MLSchellingAgent(SchellingAgent):
    """Schelling agent that deploys a prediction method
    to select the best cell to relocate in."""

    def __init__(
        self,
        unique_id: int,
        model: Model,
        pos: Tuple[int, int],
        is_minority: bool,
        relocation_model: CatBoostClassifier,
    ) -> None:
        super().__init__(unique_id, model, pos, is_minority)
        self.relocation_model = relocation_model

    def choose_cell(self, cells: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        assert len(cells) > 0, "There are no valid cells"
        # Computes features for each cell
        features: List[Dict] = [self.get_relocation_record(pos) for pos in cells]
        # Transforms the list of records into an array
        X: DataFrame = DataFrame.from_records(features)
        # Predicts the probability of being happy (second column of the prediction matrix)
        y_pred: np.ndarray = self.relocation_model.predict_proba(X)[:, 1]
        i_max: int = y_pred.argmax().item()
        return cells[i_max]
