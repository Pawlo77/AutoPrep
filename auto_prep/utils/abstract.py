import importlib
import inspect
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils.config import config
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class Numerical(ABC):
    """Abstract interface to indicate numerical step"""

    pass


class Categorical(ABC):
    """Abstract interface to indicate categorical step"""

    pass


class Step(ABC, BaseEstimator, TransformerMixin):
    """
    Abstract class to be overwritten for implementing custom
    preprocessing steps. If step is parametrizable, it should have
    defined "param_grid" of all possible values for each parameter.
    """

    @abstractmethod
    def to_tex(self) -> dict:
        """
        Returns a short description in form of dictionary.
        Keys are: name - transformer name, desc - short description, params - class parameters (if None then {}).
        """
        pass


class RequiredStep(Step):
    """
    Required step that will be always considered in preprocessing.
    """

    pass


class NonRequiredStep(Step):
    """
    Non required step that will be only considered for preprocessing.
    """

    pass


class ModulesHandler(ABC):
    supported_interfaces: List[object] = [
        Numerical,
        Categorical,
        RequiredStep,
        NonRequiredStep,
    ]
    supported_combinations: List[List[object]] = [
        ("NumericalRequired", (Numerical, RequiredStep)),
        ("NumericalNonRequired", (Numerical, NonRequiredStep)),
        ("CategoricalRequired", (Categorical, RequiredStep)),
        ("CategoricalNonRequired", (Categorical, NonRequiredStep)),
    ]

    def __init__(self):
        """
        Performs checks.

        Raises:
            AssertionError - if any member of :obj:`ModulesHandler.supported_combinations`
                is not in :obj:`ModulesHandler.supported_interfaces`
        """
        for name, members in ModulesHandler.supported_combinations:
            for member in members:
                assert (
                    member in ModulesHandler.supported_interfaces
                ), f"Unsupported member in group {name} - {member}"

    @staticmethod
    def get_subpackage(__file__):
        """
        Returns the name of the package (directory) containing the given file
        as relative auto_prep subpackage.

        Args:
            __file__ (str): The absolute or relative path to the current file.
        Returns:
            str: The name of the directory containing the file, which is treated as the package name.
        Raises:
            ValueError - if it cannot find the module
        """
        current_file = os.path.abspath(__file__)
        abs_dir = os.path.dirname(current_file)

        if config.root_project_dir not in abs_dir:
            raise ValueError("Unknown relative module")

        rel_dir = abs_dir[len(config.root_project_dir) :].lstrip(os.path.sep)

        # Convert the path to a module-style dot-separated format
        return rel_dir.replace(os.path.sep, ".")

    @staticmethod
    def construct_pipelines_steps_helper(
        step_name: str,
        package_name: str,
        called_from: str,
        pipelines: List[List[Step]],
        required_only_: bool = False,
    ) -> List[List[Step]]:
        """
        A helper method to construct and extend pipelines steps by incorporating modules
        dynamically from a specified package.

        This method uses the `ModulesHandler.construct_pipelines` function to add
        modules to existing pipelines based on the package's name and the current
        file context. It logs the operation's start and end using the provided
        logger.

        Args:
            step_name (str): The name of the step, used for logging purposes.
            package_name (str): The name of the package containing the modules
                to be dynamically added to the pipelines.
            called_from (str) - python file from which this method is called. Required
                for relative imports.
            pipelines (List[List[Step]]): A list of existing pipelines to which
                new modules will be added.
            required_only_ (bool, optional): If `True`, only the required modules
                (determined by the package) will be added. If `False`, both required
                and non-required modules will be included. Defaults to `False`.

        Returns:
            List[List[Step]]: The updated list of pipelines steps after incorporating
            the modules from the specified package.
        """
        logger.start_operation(step_name)
        pipelines = ModulesHandler.construct_pipelines_steps(
            step_name,
            package_name,
            called_from,
            pipelines=pipelines,
            required_only_=required_only_,
        )
        logger.end_operation()
        return pipelines

    @staticmethod
    def construct_pipelines_steps(
        step_name: str,
        module_name: str,
        called_from: str,
        pipelines: List[List[Step]] = [],
        required_only_: bool = False,
    ) -> List[List[Step]]:
        """
        Constructs new pipelines (list of steps) by adding steps from the provided module. The
        method dynamically loads and groups classes from the module, and then
        extends existing pipelines by adding required and/or non-required steps.

        The method starts by loading and grouping classes from the module. It then
        explodes the existing pipelines by adding required steps. If the `required_only_`
        flag is `False`, non-required steps are also added to the pipelines.

        Args:
            step_name (str): The name of the step, used for logging purposes.
            module_name (str): The name of the module from which to load and group classes.
            called_from (str) - python file from which this method is called. Required
                for relative imports.
            pipelines (List[List[Step]]): A list of existing pipelines to be extended.
            required_only_ (bool, optional): If `True`, only required steps are added to the
                pipelines. If `False`, both required and non-required steps are added.
                Defaults to `False`.

        Returns:
            List[List[Step]]: A list of new pipelines steps created by adding the corresponding
            required and non-required steps to the original pipelines.
        """

        logger.start_operation("Constructing new pipelines.")

        package = ModulesHandler.get_subpackage(called_from)
        new_steps, _ = ModulesHandler._load_and_group_classes(
            module_name, package=package
        )
        logger.debug(f"New steps: {new_steps}")
        new_pipelines = []

        logger.debug(f"Starting with {len(pipelines)} pipelines")

        new_pipelines, num_required = ModulesHandler._explode_pipelines_steps(
            steps=ModulesHandler._get_required_steps(),
            new_steps=new_steps,
            pipelines=pipelines,
        )
        logger.debug(f"After required steps - {len(new_pipelines)}")

        num_non_required = 0
        if not required_only_:
            non_required, num_non_required = ModulesHandler._explode_pipelines_steps(
                steps=ModulesHandler._get_non_required_steps(),
                new_steps=new_steps,
                pipelines=new_pipelines,
            )

            # keep those from required only and those extended with non-required steps
            if num_non_required > 0:  # there were some non-required steps
                new_pipelines.extend(non_required)
            logger.debug(f"After non-required steps - {len(new_pipelines)}")

        logger.info(
            f"Extracted {num_required + num_non_required} steps for {step_name} ({num_required} required, {num_non_required} non required)"
        )

        logger.end_operation()
        return new_pipelines

    @staticmethod
    def _explode_pipelines_steps(
        steps: List[str],
        new_steps: Dict[str, List[object]],
        pipelines: List[List[Step]],
    ) -> Union[List[List[Step]], int]:
        """
        Explodes the given pipelines by adding new steps to them based on the
        provided `steps` and `new_steps`. This method creates new pipelines where
        each pipeline is extended by the corresponding steps from `new_steps`.

        Args:
            steps (List[str]): List of step names to match against `new_steps` keys.
            new_steps (Dict[str, List[object]]): A dictionary where each key is a
                step name and its value is a list of classes (steps) to be added
                to the corresponding pipeline.
            pipelines (List[List[PipelineStep]: A list of existing pipelines to be extended.

        Returns:
            List[List[Step]: A list of new pipelines created by adding the new steps
                to the existing pipelines. If pipelines are empty, it will just return
                new found steps.
            int: number of unique objects extracted.
        """
        new_pipelines = []
        num = 0

        if len(pipelines) == 0:
            for step in steps:
                if step in new_steps.keys():
                    for cls in new_steps[step]:
                        num += 1
                        new_pipelines.append([cls])
            return new_pipelines, num

        for step in steps:
            if step in new_steps.keys():
                for cls in new_steps[step]:
                    num += 1
                    for pipeline in pipelines:
                        new_pipelines.append([*pipeline, cls])

        if num > 0:
            return new_pipelines, num
        return pipelines, num

    @staticmethod
    def _get_required_steps() -> List[str]:
        """Returns list of names of required steps combinations"""
        return [
            e[0]
            for e in ModulesHandler.supported_combinations
            if "NonRequired" not in e[0]
        ]

    @staticmethod
    def _get_non_required_steps() -> List[str]:
        """Returns list of names of required steps combinations"""
        return [
            e[0] for e in ModulesHandler.supported_combinations if "NonRequired" in e[0]
        ]

    @staticmethod
    def _load_and_group_classes(
        module_name: str, package: str
    ) -> Union[Dict[str, List[object]], int]:
        """
        Import all objects from module_name that extends any of interfaces from
        :obj:`ModulesHandler.supported_interfaces` and groups them into steps
        defined in :obj:`ModulesHandler.supported_combinations`

        Args:
            module_name (str) - module to import.
            package (str) - python package from which this method is called. Required
                for relative imports.
        Returns:
            Dict[List[object]] - objects groupped into pre-defined groups.
            int - number of unique objects extracted.
        Raises:
            ValueError - if any of imported classes fits into more than one group.
        """
        logger.debug(f"Importing classes from {module_name}")

        module = importlib.import_module(module_name, package=package)

        classes = [
            cls
            for _, cls in inspect.getmembers(module, inspect.isclass)
            if cls.__module__.endswith(module_name)
        ]

        logger.debug(f"Found follwing classes: {classes}")

        combinations = {}
        groupped = set()
        for cls in classes:
            for name, members in ModulesHandler.supported_combinations:
                in_group_ = True
                for member in members:
                    if not issubclass(cls, member):
                        in_group_ = False

                if in_group_:
                    if cls not in groupped:
                        groupped.add(cls)
                        if name not in combinations:
                            combinations[name] = [cls]
                        else:
                            combinations[name].append(cls)
                    else:
                        raise ValueError(f"{cls} fits more than one group")

        logger.debug(f"Retrieved follwing combinations - {combinations}")

        return combinations, len(groupped)


class DimentionReducer(NonRequiredStep, Numerical, ABC):
    """
    Abstract class for dimensionality reduction techniques.
    """

    def __init__(self):
        super().__init__()
        self.reducer = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "DimentionReducer":
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def to_tex(self) -> dict:
        pass


class FeatureImportanceSelector(NonRequiredStep, Numerical):
    """
    Transformer to select k% (rounded to whole number) of features
    that are most important according to Random Forest model.

    Attributes:
        k (float): The percentage of top features to keep based on their importance.
        selected_columns (list): List of selected columns based on feature importance.
    """

    def __init__(self, k: float = 10.0):
        """
        Initializes the transformer with a specified model and percentage of top important features to keep.

        Args:
            k (float): The percentage of features to retain based on their importance.
        """
        if not (0 <= k <= 100):
            raise ValueError("k must be between 0 and 100.")
        self.k = k
        self.selected_columns = []

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "FeatureImportanceSelector":
        """
        Identifies the top k% (rounded to whole value) of features most important according to the model.

        Args:
            X (pd.DataFrame): The input feature data.
            y (pd.Series): The target variable.

        Returns:
            FeatureImportanceSelector: The fitted transformer instance.
        """
        pass

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Selects the top k% of features most important according to the model.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series, optional): The target variable (to append to the result).

        Returns:
            pd.DataFrame: The transformed data with only the selected top k% important features.
        """
        pass

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fits and transforms the data by selecting the top k% most important features. Performs fit and transform in one step.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series): The target variable.

        Returns:
            pd.DataFrame: The transformed data with selected features.
        """
        self.fit(X, y)
        return self.transform(X)
