import time
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from viperleed.calc import LOGGER as logger


class Optimizer(ABC):
    """Class for all the optimizers.

    Args:
        fun: Objective function
    """
    def __init__(self, fun):
        self.fun = fun

    @abstractmethod
    def __call__(self):
        """Start the optimization."""

class GradOptimizer(Optimizer):
    """Class for optimizers that use a gradient.

    Args:
        fun: Objective function.
        grad: Gradient of the objective function.
        fun_and_grad: Function value and gradient together. This approach
            is faster, but it only makes sense, if they are always or almost 
            always computed together (e.g., in L-BFGS-B). Avoid using it when 
            having a lot function calls without the gradient (e.g., in SLSQP).
    """
    def __init__(self, fun, grad, fun_and_grad):
        self.fun = fun
        self.grad = grad
        self.fun_and_grad = fun_and_grad
        super().__init__(fun=fun)

class NonGradOptimizer(Optimizer):
    """Class for optimizers that don't use a gradient."""
    def __init__(self, fun):
        self.fun = fun
        super().__init__(fun=fun)

class LBFGSBOptimizer(GradOptimizer):
    """Class for setting up the L-BFGS-B algorithm for local minimization.
    The BFGS algorithm uses the BFGS approximation of the Hessian, which is
    always positive definite. Gradient and Hessian (approximation) are used to
    determine search direction. A line search is performed along this direction,
    which must satisfy the Wolfe conditions. These conditions provide an upper
    and lower limit for the step size, and one condition also ensures that the
    function value decreases for each iteration.

    Args:
        fun_and_grad: Function value and gradient together. This approach
            is faster, but it only makes sense, if they are always or almost 
            always computed together (e.g., in L-BFGS-B). Avoid using it when 
            having a lot function calls without the gradient (e.g., in SLSQP).
        bounds: Since the parameters are normalized, all bounds are by default
            set to [0,1].
        ftol: Convergence condition that sets a lower limit on the difference
            in function value between two iterations. A higher value has been
            shown to lead to a faster termination without significantly 
            changeing the R factor. It can even be set to 1e-6 for a faster 
            converence, but this slightly worsens the R factor.
        maxiter: Maximal number of iterations for the algorithm. Usually, the
            algorithm stops earlier due to convergence.
    """
    def __init__(self, fun_and_grad, bounds=None, ftol=1e-7, maxiter=1000):
        self.fun_and_grad = fun_and_grad
        self.bounds = bounds
        self.ftol = ftol
        self.maxiter = maxiter
        super().__init__(fun_and_grad=fun_and_grad, grad=None, fun=None)

    def __call__(self, start_point):
        """With the call, the algorithm starts with the given parameters.
        This function prints a termination message and returns all the values
        that are also returned by the SciPy function, plus a list of the
        function values for each iteration (fun_hystory) and the 
        runtime (duration).
        
        Args:
            start_point: Starting point of the algorithm. Usually it start at
                0.5 for each dimension, since it is in the middle of the bounds.
        """
        # Setting up Callback function to save function history in fun_history
        fun_history = []
        current_fun = [None]
        current_grad = [None]
        def fun_with_storage(x):
            current_fun[0], current_grad[0] = self.fun_and_grad(x)
            return current_fun[0], current_grad[0]
        def callback_function(x):
            fun_history.append(current_fun[0])

        # Setting up the bounds
        if self.bounds is None:
            bounds = [(0, 1) for _ in range(len(start_point))]
        else:
            bounds = self.bounds

        # Performing the optimization
        start_time = time.time()
        result = minimize(
            fun_with_storage,
            x0=start_point,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            callback=callback_function,
            options={'maxiter': self.maxiter,'ftol': self.ftol}
        )
        end_time = time.time()
        duration = end_time - start_time
        result.fun_history = fun_history
        result.duration = duration
        logger.info('Optimization Result:\n')
        logger.info(f'{str(result)} \n\n')
        return result

class SLSQPOptimizer(GradOptimizer):
    """Class for setting up the SLSQP algorithm for local minimization.
    The SLSQP algorithm uses a quadratic approximation of the Lagrangian to 
    include equality constraints. Inequality constraints are incorporated by 
    defining active and passive sets, as well as an upper limit of the step 
    size. For the Hessian, the BFGS approximation is used, and the line search 
    cannot satisfy the Wolfe conditions. Due to a poor Hessian approximation in 
    the initial iterations, the algorithm tends to overshoot during these 
    iterations. To address this issue, a damping factor (damp_fact) is used to
    reduce the gradient at the beginning. A damping factor of 0.1 has shown good 
    results.

    Args:
        fun: Objective function.
        grad: Gradient of the objective function.
        bounds: Since the parameters are normalized, all bounds are by default
            set to [0,1].
        damp_factor: Damping factor.
        ftol: Convergence condition that sets a lower limit on the difference
            in function value between two iterations.
        maxiter: Maximal number of iterations for the algorithm. Usually, the
            algorithm stops earlier due to convergence.
    """

    def __init__(self, fun, grad, bounds=None,
                 damp_fact=1, ftol=1e-6, maxiter=1000):
        self.fun = fun
        self.grad = grad
        self.bounds = bounds
        self.damp_fact = damp_fact
        self.ftol = ftol*damp_fact
        self.maxiter = maxiter
        super().__init__(fun_and_grad=None, grad=grad, fun=fun)

    def __call__(self, start_point):
        """With the call, the algorithm starts with the given parameters.
        This function prints a termination message and returns all the values
        that are also returned by the SciPy function, plus a list of the
        function values for each iteration (fun_hystory) and the 
        runtime (duration).
        
        Args:
            start_point: Starting point of the algorithm. Usually it start at
                0.5 for each dimension, since it is in the middle of the bounds.
        """
        # Setting up Callback function to save function history in fun_history
        fun_history = []
        current_fun = [None]
        def dampened_grad(x):
            return self.damp_fact*self.grad(x)
        def dampened_fun_storage(x):
            current_fun[0] = self.fun(x)
            return current_fun[0]*self.damp_fact
        def callback_function(x):
            fun_history.append(current_fun[0])

        # Setting up the bounds
        if self.bounds is None:
            bounds = [(0, 1) for _ in range(len(start_point))]
        else:
            bounds = self.bounds

        # Performing the optimization
        start_time = time.time()
        result = minimize(
            fun=dampened_fun_storage,
            x0=start_point,
            method='L-BFGS-B',
            jac=dampened_grad,
            bounds=bounds,
            callback=callback_function,
            options={'maxiter': self.maxiter,'ftol': self.ftol}
        )
        end_time = time.time()
        duration = end_time - start_time
        result.fun_history = fun_history
        result.duration = duration
        logger.info('Optimization Result:\n')
        logger.info(f'{str(result)} \n\n')
        return result
