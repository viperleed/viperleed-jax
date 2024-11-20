class Optimizer:
    """Class for all the optimizers.
    Args:
        fun: Objective function
    """
    def __init__(self, fun):
        self.fun = fun

class GradOptimizer(Optimizer):
    """Class for the optimizers, that use a gradient.
    Args:
        fun: Objective function
        grad: Gradient of the objective function
        fun_and_grad: Function value and gradient together. By computing them thogether it is faster, but it
            only makes sense, if they are always/almost always computed together (e.g., in L-BFGS-B). Espacially 
            don't use it when having a lot function calls without the gradient (e.g., in SLSQP).
    """
    def __init__(self, fun, grad, fun_and_grad):
        self.fun = fun
        self.grad = grad
        self.fun_and_grad = fun_and_grad
        super().__init__(fun=fun)

class NonGradOptimizer(Optimizer):
    def __init__(self, fun):
        self.fun = fun
        super().__init__(fun=fun)

class LBFGSBOptimizer(GradOptimizer):
    """Class setting up the L-BFGS-B algorithm for the local minimization. The BFGS algorithm uses the BFGS
    approximation of the Hessian, which is always positive devinite. Gradient and Hessian (approximation) are
    used for the search direction. A line search is performed along this direction, which has to satisfy the
    Wolfe conditions, that give an upper and lower limit for the step size. One condition also ensures, that
    the function decreases for each iteration.
    Args:
        fun_and_grad: Function value and gradient together. By computing them thogether it is faster, but it
            only makes sense, if they are always/almost always computed together (e.g., in L-BFGS-B). Espacially 
            don't use it when having a lot function calls without the gradient (e.g., in SLSQP).
        bounds: Since the parameters are normalized, all bounds are by default set to [0,1].
        ftol: Converence condition, that gives a lower limit to the difference in function value between
            two evaluations. It has shown that a higher value leads to a faster termination and does not
            segnificantly change the R factor. It can even be set to 1e-6 for a faster converence, but 
            worsens the R factor a little.
        maxiter: Maximal number of iterations for the algorithm. Usually it stops earlier, due to convergence.
    """
    def __init__(self, fun_and_grad, bounds=None, ftol=1e-7, maxiter=1000):
        self.fun_and_grad = fun_and_grad
        self.bounds = bounds
        self.ftol = ftol
        self.maxiter = maxiter
        super().__init__(fun_and_grad=fun_and_grad, grad=None, fun=None)

    def __call__(self, start_point):
        """With the call the algorithm starts with the given parameters. This function prints a 
        termination massage and returns all the values, that are also returned by the scipy function
        plus a list of the function values for each iteration (fun_hystory) and the runtime (duration).
        Args:
            start_point: Starting point of the algorithm. Usually it start at 0.5 for each dimension, since
            it is in the middle of the bounds.
        """
        from scipy.optimize import minimize
        import time
        # Setting up the Callback function to save the function history in fun_history
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
        print("Optimization Result:\n")
        print(str(result) + "\n\n")
        return result
