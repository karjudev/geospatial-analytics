import random
from typing import Any, Dict, Iterator, Optional, Tuple, Union
from mesa import Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
import pandas as pd
from pandas import DataFrame
from schelling.agent import SchellingAgent, MildRandomSchellingAgent, MLSchellingAgent


class SchellingModel(Model):
    """Schelling model with mild random policy.
    Every time that an agent needs to relocate, computes the cells where it would be happy and picks one uniformly at random.
    """

    def __init__(
        self,
        side: int,
        density: float,
        minority_pc: float,
        homophily: float,
        neighbor_mode: str = "moore",
        relocation_model: Optional[Any] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Creates a new Schelling model.

        :param side: Side of the square.
        :type side: int
        :param density: Density of the population.
        :type density: float
        :param minority_pc: Percentage of the minority type.
        :type minority_pc: float
        :param homophily: Percentage of neighbors of the same type required for an agent to be happy.
        :type homophily: float
        :param neighbor_mode: Way of iterating the neighborhood of a cell. Defaults to "moore".
        :type neighbor_mode: str
        :param relocation_model: If provided, the ML model used by the MLSchellingAgent. Defaults to None (use MildRandomSchellingAgent).
        :type relocation_model: Optional[Any]
        """
        # Checks variable correctness
        assert isinstance(side, int), "Side must be an integer."
        assert (
            isinstance(density, float) and density > 0 and density <= 1
        ), "Density must be a float in the (0, 1] range."
        assert (
            isinstance(minority_pc, float) and minority_pc >= 0 and minority_pc <= 0.5
        ), "Minority percentage must be a float in the [0, 1/2] range."
        assert (
            isinstance(homophily, float) and homophily >= 0 and homophily <= 1
        ), "Homophily percentage must be a float in the [0, 1] range."
        assert isinstance(
            neighbor_mode, str
        ), "The neighbor iteration mode has to be a string."
        # Initializes the object
        super().__init__(*args, **kwargs)

        self.side = side
        self.density = density
        self.minority_pc = minority_pc
        self.homophily = homophily
        self.neighbor_mode = neighbor_mode
        # If the neighbor mode is "moore", the agent has 8 neighbors
        if neighbor_mode == "moore":
            self.total_neighbors: int = 8
        else:
            raise NotImplementedError(
                "Iteration modes other than `moore` are not yet implemented."
            )

        # The agents are activated in random order, once per step, with the order shuffled at every step
        self.schedule = RandomActivation(self)

        # Square grid where each cell contains at most one agent
        self.grid = SingleGrid(width=side, height=side, torus=False)

        # Total number of happy agents
        self.total_happy: int = 0

        # Flag that indicates that the model is running
        self.running: bool = True

        # Collector that computes the relocation dataset for each simulation of the model
        self.datacollector = DataCollector(
            model_reporters={
                "relocation_dataset": lambda m: m.collect_relocation_dataframe(),
                "steps": lambda m: m.schedule.steps,
                "happyness_avg": "happyness_avg",
            },
        )

        # Running ID of the agent
        agent_id: int = 0
        # Sets up the agents iterating through every cell
        for _, x, y in self.grid.coord_iter():
            # If true, selected cell is populated, else is empty
            if random.random() < self.density:
                # If true, cell is populated with minority, else with majority type
                is_minority: bool = random.random() < self.minority_pc
                # Instance of agent class
                agent: SchellingAgent
                if relocation_model is not None:
                    agent = MLSchellingAgent(
                        agent_id, self, (x, y), is_minority, relocation_model
                    )
                else:
                    agent = MildRandomSchellingAgent(
                        agent_id, self, (x, y), is_minority
                    )
                # Increments the running agent ID
                agent_id += 1
                # Places the agent in the grid and adds it to the schedule
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)

    def step(self) -> None:
        """Performs a step of the simulation."""
        # Resets the counter of the total happy people
        self.total_happy = 0
        # Executes a step
        self.schedule.step()
        # Collects the data for the iteration
        self.datacollector.collect(self)
        # If all the agents are happy terminates
        if self.total_happy == self.schedule.get_agent_count():
            self.running = False

    def iterate_empty_cells(self) -> Iterator[Tuple[int, int]]:
        """Gets the empty cells currently in the model.

        :return: Iterator of coordinates of empty cells.
        :rtype: Iterator[Tuple[int, int]]
        """
        for _, x, y in self.grid.coord_iter():
            if self.grid.is_cell_empty((x, y)):
                yield x, y

    def is_agent_happy(self, segregation: float) -> bool:
        """Checks if the agent is happy with the current segregation level.

        :param segregation: Segregation level.
        :type segregation: float
        :return: True if the agent is happy, False if it needs to relocate.
        :rtype: bool
        """
        # Checks parameters
        assert (
            isinstance(segregation, float) and segregation >= 0 and segregation <= 1
        ), "Segregation has to be a float in the [0, 1] range."
        return segregation >= self.homophily

    def compute_segregation(self, pos: Tuple[int, int], is_minority: bool) -> float:
        """Computes the segregation of the current agent in the given position.

        :param pos: Coordinates of the position where the segregation of the agent has to be computed.
        :type pos: Tuple[int, int]
        :param is_minority: If the segregation has to be computed for the majority or minority class.
        :type is_minority: bool
        :return: Segregation in `pos` coordinates.
        :rtype: float
        """
        equal_neighbors: int = sum(
            1
            for neighbor in self.grid.iter_neighbors(pos, self.neighbor_mode)
            if neighbor.is_minority == is_minority
        )
        return equal_neighbors / self.total_neighbors

    def compute_density(self, pos: Tuple[int, int]) -> float:
        """Computes the density in a given position.

        :param pos: Coordinates of the cell.
        :type pos: Tuple[int, int]
        :return: Density in the cell.
        :rtype: float
        """
        count_neighbors: int = sum(
            1 for _ in self.grid.iter_neighbors(pos, self.neighbor_mode)
        )
        return count_neighbors / self.total_neighbors

    def get_location_features(
        self, pos: Tuple[int, int]
    ) -> Dict[str, Union[float, int]]:
        """Computes the features for a certain location.

        :param pos: Coordinates of the cell.
        :type pos: Tuple[int, int]
        :return: Features for the location.
        :rtype: Dict[str, Union[float, int]]
        """
        return {
            "x": pos[0],
            "y": pos[1],
            "density": self.compute_density(pos),
            "segregation_minority": self.compute_segregation(pos, is_minority=True),
            "segregation_majority": self.compute_segregation(pos, is_minority=False),
        }

    def collect_relocation_dataframe(self) -> Optional[DataFrame]:
        """Collects the relocation DataFrame for each agent.

        :return: Concatenated relocation DataFrame for each agent. If there are no relocations, returns None.
        :rtype: Optional[DataFrame]
        """
        try:
            df: DataFrame = pd.concat(
                (
                    agent.collect_relocation_dataframe()
                    for agent in self.schedule.agents
                ),
                ignore_index=True,
            )
            if len(df) == 0:
                return None
            return df
        except ValueError:
            return None
