from hydro.bellman import BellmanValuesProxy
import numpy as np
from tqdm import tqdm

class OptimalTrajectories:
    def __init__(self,
                 bellman_values : BellmanValuesProxy,
                 pbar : tqdm):
        """
        Initialize OptimalTrajectories with a BellmanValuesProxy instance.
        Prepares data and computes optimal trajectories.
        """
        self.bellman_values=bellman_values
        self.nb_weeks = bellman_values.nb_weeks
        self.scenarios = bellman_values.scenarios
        self.pbar = pbar
        
        self.mean_bv=bellman_values.mean_bv
        self.compute_trajectories()

    def compute_trajectories(self) -> None:
        """
        Compute optimal reservoir trajectories and controls schedules
        for all scenarios and weeks using Bellman values, penalties and inflows.
        Adjusts hourly inflows to avoid overflow or negative stock with external method.
        """
        self.pbar.set_postfix_str("Optimal trajectories computing") 
        self.trajectories = np.zeros((len(self.scenarios), self.nb_weeks))
        self.optimal_controls = np.zeros_like(self.trajectories)
        self.optimal_turb = np.zeros_like(self.trajectories)
        self.optimal_pump = np.zeros_like(self.trajectories)
        self.inflow_adjust_overflow = np.zeros((self.nb_weeks, len(self.scenarios), 168))

        penalty_by_week = [self.bellman_values.penalty_rule_curves(w) for w in range(self.nb_weeks)]
        bellman_by_week = [self.bellman_values.bellman_function(w) for w in range(self.nb_weeks)]   
        
        for s in self.scenarios:
            current_stock = self.bellman_values.proxy.reservoir.initial_level
            
            for w in range(self.nb_weeks):
                self.pbar.update(1)

                weekly_inflow = self.bellman_values.proxy.reservoir.weekly_inflow[w, s]
                hourly_inflow = self.bellman_values.proxy.reservoir.hourly_inflow[w * 168:(w + 1) * 168, s]


                cost_function = self.bellman_values.stage_cost_functions[w, s]
                penalty_function = penalty_by_week[w]

                controls = cost_function.x

                future_bellman_function = bellman_by_week[w]

                max_week_turb = self.bellman_values.proxy.reservoir.max_weekly_turb[w]
                max_week_pump = self.bellman_values.proxy.reservoir.max_weekly_pump[w]

                max_control = self.adjust_hourly_inflow_overflow(scenario=s, 
                                                                 week=w, 
                                                                 stock_init=current_stock, 
                                                                 inflow=hourly_inflow)                
                if max_control < controls[-1]  :
                    controls = controls[controls < max_control]
                    controls = np.concatenate([controls, [max_control]])

                weekly_inflow -= np.sum(self.inflow_adjust_overflow[w, s])

                best_value, best_stock, best_control = self.bellman_values.iterate_over_controls_vec(
                        controls=controls,
                        current_stock=current_stock,
                        weekly_inflow=weekly_inflow,
                        stage_cost_function=cost_function,
                        future_bellman_function=future_bellman_function,
                        penalty_function=penalty_function
                    )

                __, final_best_stock, final_best_control = self.bellman_values.iterate_over_stock_levels_vec(
                    best_value=best_value,
                    best_stock=best_stock,
                    best_control=best_control,
                    current_stock=current_stock,
                    weekly_inflow=weekly_inflow,
                    max_week_pump=max_week_pump,
                    max_week_turb=max_week_turb,
                    stage_cost_function=cost_function,
                    future_bellman_function=future_bellman_function,
                    penalty_function=penalty_function,
                    max_control=controls[-1])

                self.trajectories[s, w] = final_best_stock
                self.optimal_controls[s, w] = final_best_control
                current_stock = final_best_stock

    def adjust_hourly_inflow_overflow(
        self,
        scenario: int,
        week: int,
        stock_init: float,
        inflow: np.ndarray
    ) -> float:
        """
        Adjust hourly inflow to mitigate overflow or negative stock by detecting violations and recording adjustments.
        Returns the maximum feasible control (turbining) for the week after adjustments and taking in account potential overflow or negative stock during the week.
        """
        cap = float(self.bellman_values.proxy.reservoir.capacity)

        turb = self.bellman_values.proxy.reservoir.max_hourly_turb[week * 168:(week + 1) * 168]
        max_control = turb * self.bellman_values.proxy.turb_efficiency

        net_hourly_turb = inflow - max_control

        stock = float(stock_init)

        for h in range(168):
            stock += float(net_hourly_turb[h])

            if stock > cap:
                overflow = stock - cap

                self.inflow_adjust_overflow[week, scenario, h] = overflow

                net_hourly_turb[h] -= overflow
                stock = cap


            if stock < 0.0:
                neg = stock  

                max_control[h] += neg  


                net_hourly_turb[h] -= neg 
                stock = 0.0

        return float(np.sum(max_control))
