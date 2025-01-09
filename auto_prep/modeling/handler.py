import importlib
import inspect
import pandas as pd
from typing import Dict, List, Union
from sklearn.metrics import roc_auc_score
import joblib
import numpy as np
import os

from sklearn.model_selection import RandomizedSearchCV
from ..utils.abstract import Classifier, Regressor
from ..utils.config import config
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ModelHandler:
    """
    Class responsible for loading and handling machine learning models and pipelines.
    """
    
    NUMBER_OF_MODELS = 3

    def __init__(self):
        """
        Initializes an instance of ModelHandler.
        """
        self._pipelines: Dict = {"file_name": [], "pipeline": []}
        self.regression_models: List = []
        self.classification_models: List = []
        self.modules: List = []
        self.results = pd.DataFrame()
        self.tuned_results = pd.DataFrame()
        self.task: str = None

    def load_pipelines(self):
        """
        Loads pipelines from the specified directory and stores them in the `_pipelines` attribute.
        """
        logger.start_operation("Loading pipelines.")
        try:
            for file_name in os.listdir(config.pipelines_dir):
                if file_name.endswith(".joblib"):
                    self._pipelines["file_name"].append(file_name)
                    pipeline = joblib.load(
                        os.path.join(config.pipelines_dir, file_name)
                    )
                    self._pipelines["pipeline"].append(pipeline)
                    logger.debug(f"Pipeline {pipeline} loaded.")
            logger.error(f"Loaded {len(self._pipelines)} pipelines.")
        except Exception as e:
            logger.error(f"Error in loading pipelines: {e}")
            raise e
        finally:
            logger.end_operation()

    def load_and_group_classes(self, module_name: str, package: str):
        """
        Loads and groups classes from the specified module and package.

        Args:
            module_name (str): The name of the module to import classes from.
            package (str): The package where the module is located.
        """
        print(f"importing classes from {module_name}")
        logger.debug(f"Importing classes from {module_name}")
        print("module_name: ", module_name)
        module = importlib.import_module(module_name, package=package)

        print(f"module: {module}")
        classes = [
            cls
            for _, cls in inspect.getmembers(module, inspect.isclass)
            if cls.__module__.endswith(module_name)
        ]

        logger.debug(f"Found following classes: {classes}")

        for name, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, (Regressor, Classifier)) and cls.__module__ == module.__name__:
                if issubclass(cls, Regressor):
                    self.regression_models.append(cls())
                else:
                    self.classification_models.append(cls())
                print(f"Added model: {name}")

    def load_modules(self, package: str):
        """
        Loads modules from the specified package and stores their names in the `modules` attribute.

        Args:
            package (str): The package to load modules from.
        """
        package = os.path.abspath(package)
        print(f"package: {package}")
        print(f"package: {os.listdir(package)}")

        for file_name in os.listdir(package):
            if file_name.startswith("Model"):
                module_name = f".{file_name[:-3]}"
                print(f"module_: {module_name}")
                self.modules.append(module_name)
        print(self.modules)

    def load_and_group_all_classes(self, package: str):
        """
        Loads and groups all classes from the modules stored in the `modules` attribute.

        Args:
            package (str): The package where the modules are located.
        """
        for module_name in self.modules:
            self.load_and_group_classes(module_name, package)

    def fit_all_models(self, 
                       X_train: pd.DataFrame, 
                       X_valid: pd.DataFrame, 
                       y_train: pd.Series,
                       y_valid: pd.Series, 
                       task: str):
        """
        Fits all the loaded models with the given data.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target variable.
            task (str): The type of task (classification or regression).
        """
        print("starting")
        for pipeline in self._pipelines["pipeline"]:
            
            print("Fitting pipeline: ", pipeline)
            X_transformed = pipeline.transform(X_train)
            print(X_transformed.head())
            if task == "classification":
                for model in self.classification_models:
                    print("Fitting model: ", model)
                    model.fit(X_transformed, y_train)
                    predictions = model.predict(pipeline.transform(X_valid))
                    auc = config.classification_pipeline_scoring_func(y_valid, predictions)
                    pipeline_file_name = self._pipelines["file_name"][self._pipelines["pipeline"].index(pipeline)]
                    self.results = self.results.append({"model": model, "roc_auc": auc, "pipeline": pipeline_file_name, "X_transformed": X_transformed}, ignore_index=True)
        self.results = self.results.sort_values(by="roc_auc", ascending=False).head(self.NUMBER_OF_MODELS)
               
    
    def hyperparameter_tuning(self, y_train: pd.Series):
        self.tuned_results = pd.DataFrame({"model": [], "params": [], "pipeline": [], "roc_auc": []})
        
        for i in range(self.NUMBER_OF_MODELS):
            model = self.results["model"].iloc[i]
            logger.info(f"Hyperparameter tuning for model: {model}")
            params = model.PARAM_GRID
            logger.info(f"Parameters: {params}")
            random_search = RandomizedSearchCV(estimator=model, param_distributions=params, cv=3, verbose=2, random_state=42, n_jobs=-1, scoring="roc_auc")
            random_search.fit(self.results["X_transformed"].iloc[i], y_train)
            logger.info("random search fitted")
            logger.info(f"Best params: {random_search.best_params_}")
            pipeline = self.results["pipeline"].iloc[i]
            self.tuned_results = self.tuned_results.append({"model": model, "params": random_search.best_params_, "pipeline": pipeline,  "roc_auc": random_search.best_score_}, ignore_index=True)
        self.tuned_results = self.tuned_results.sort_values(by="roc_auc", ascending=False)
    
    
    def write_to_raport(self, raport):
        
        raport.add_section(f"Modeling")
        raport.add_subsection("Overview")
        if self.task == "classification":
            report_models = self.classification_models
            score = "ROC AUC"
        else:
            report_models = self.regression_models 
            score = "R2"
        
        section_desc = f"This part of the report presents the results of the modeling process. There were {len(report_models)} {self.task} models trained and {self.NUMBER_OF_MODELS} of them selected based on the {score} score."
        raport.add_text(section_desc)
        raport.add_list([model.to_tex()["name"] for model in report_models], caption="Models used in the modeling process")
        table_desc = f"The table below presents the results of the modeling process on default parameters for each of the best piplelines. The models are sorted by the {score} score in descending order."
        raport.add_text(table_desc)
        modified_results = self.results.copy()
        modified_results.drop(columns=["X_transformed"], inplace=True)
        modified_results["pipeline"] = modified_results["pipeline"].apply(lambda x: x[:-7])
        modified_results["model"] = modified_results["model"].apply(lambda x: x.to_tex()["name"])
        modified_results["roc_auc"] = modified_results["roc_auc"].apply(lambda x: round(x, 5))
        modified_results = modified_results.to_dict(orient="list")
        modified_results = list(zip(modified_results['model'], modified_results['roc_auc'], modified_results['pipeline']))
        logger.info(f"modified_results: {modified_results}")
        raport.add_table(data= modified_results , caption="Results of the modeling process on default parameters", header=["Model", "AUC Score", "Pipeline"], widths=[30, 20, 50])  
        
        raport.add_subsection("Hyperparameter tuning")
        section_desc = f"This section presents the results of the hyperparameter tuning process for the best {self.NUMBER_OF_MODELS} models using RandomizedSearchCV."
        params_desc = f"The following parameters grids were used for hyperparameter tuning:"
        raport.add_text(section_desc)
        raport.add_text(params_desc)
        
        model_set = set()
        param_grids = []
        for i in range(self.NUMBER_OF_MODELS):
            model = self.results["model"].iloc[i]
            param_grid = model.PARAM_GRID
            param_grids.append(param_grid)
            if model not in model_set:
                model_set.add(model)
                model_name = model.to_tex()["name"]
                raport.add_table(data=param_grid, caption=f"Parameter grid for {model_name}", header=["Parameter", "Values"], widths=[30, 70])
        
        tuned_results_desc = f"The table below presents the results of the hyperparameter tuning process for the best {self.NUMBER_OF_MODELS} models. The models are sorted by the {score} score in descending order."
        raport.add_text(tuned_results_desc)
        modified_results = self.tuned_results.copy()
        modified_results["pipeline"] = modified_results["pipeline"].apply(lambda x: x[:-7])
        modified_results["model"] = modified_results["model"].apply(lambda x: x.to_tex()["name"])
        modified_results["roc_auc"] = modified_results["roc_auc"].apply(lambda x: round(x, 5))
        modified_results = modified_results.to_dict(orient="list")
        modified_results = list(zip(modified_results['model'], modified_results['params'], modified_results['pipeline'], modified_results['roc_auc'], ))
        logger.info(f"modified_results: {modified_results}")
        raport.add_table(data= modified_results , caption="Results of the hyperparameter tuning process on default parameters", header=["Model", "Params", "Pipeline", "ROC AUC"], widths=[15, 50, 60, 10])  
        
        
        return raport
        
        
        
        
    def run(self,
                X_train: pd.DataFrame, 
                X_valid: pd.DataFrame, 
                y_train: pd.Series,
                y_valid: pd.Series, 
                task: str):
        self.task = task
        self.load_pipelines()
        self.load_modules(package="../auto_prep/modeling")
        self.load_and_group_all_classes(package="auto_prep.modeling")
        
        self.fit_all_models(X_train, X_valid, y_train, y_valid, task)
        self.hyperparameter_tuning(y_train)
        return self.tuned_results
                    
            

        
        
        

