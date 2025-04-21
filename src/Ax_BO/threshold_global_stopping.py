# adapted from https://ax.dev/docs/tutorials/gss/
import numpy as np
from ax.global_stopping.strategies.base import BaseGlobalStoppingStrategy
from typing import Tuple
from ax.core.experiment import Experiment
from ax.core.base_trial import TrialStatus
from ax.global_stopping.strategies.improvement import constraint_satisfaction

class ThresholdGlobalStoppingStrategy(BaseGlobalStoppingStrategy):
    """
    A GSS that stops when we observe a point better than `threshold`.
    

    Args:
        BaseGlobalStoppingStrategy (_type_): _description_
    """
    def __init__(
        self,
        min_trials: int,
        inactive_when_pending_trials: bool = True,
        threshold: float = 0.8
    ):
        """_summary_

        Args:
            min_trials (int): _description_
            inactive_when_pending_trials (bool, optional): _description_. Defaults to True.
            threshold (float, optional): _description_. Defaults to 0.8.
        """
        self.threshold = threshold
        super().__init__(
            min_trials=min_trials,
            inactive_when_pending_trials=inactive_when_pending_trials
        )
    
    def _should_stop_optimization(
        self, experiment: Experiment
    ) -> Tuple[bool, str]:
        """Check if the best seen is better than `self.threshold`.

        Args:
            experiment (Experiment): _description_

        Returns:
            Tuple[bool, str]: _description_
        """
        feasible_objectives = [
            trial.objective_mean
            for trial in experiment.trials_by_status[TrialStatus.COMPLETED]
            if constraint_satisfaction(trial)
        ]

        # Computing the interquartile for scaling the difference
        if len(feasible_objectives) <= 1:
            message = "There are not enough feasible arms tried yet."
            return False, message
        
        minimize = experiment.optimization_config.objective.minimize
        if minimize:
            best = np.min(feasible_objectives)
            stop = best < self.threshold
        else:
            best = np.max(feasible_objectives)
            stop = best > self.threshold

        comparison = "less" if minimize else "greater"
        if stop:
            message = (
                f"The best objective seen is {best:.3f}, which is {comparison} "
                f"than the threshold of {self.threshold:.3f}."
            )
        else:
            message = ""

        return stop, message