import cvxpy as cp
import numpy as np

# Gear ratios
A = np.asarray([3, 4, 5, 9])
logA = np.log(A)

fos = 1.5
in_torque = .4
out_torque = 100

min_ratio = None

# min_ratio = 240

# minimum required ratio
if(min_ratio):
    b = min_ratio
elif(in_torque):
    b = (out_torque/in_torque) * fos
else:
    exit("please specify a needed torque or min ratio")
log_b = np.log(b)

# Integer decision variables
x = cp.Variable(len(A), integer=True)

# Log of product = sum(x_i * log(A_i))
log_expr = logA @ x
total_gears = cp.sum(x)

# MILP: minimize log(product) subject to meeting the requirement
problem = cp.Problem(
    cp.Minimize(total_gears + .1 * log_expr),
    [
        x >= 0,
        log_expr >= log_b,
        x[3] <= 1
    ]
)

# Solve with SCIP (must be installed)
result = problem.solve(solver="SCIP")

print("Optimal x:", x.value)
print("Product:", np.exp(log_expr.value))
print("output torque:", in_torque * np.exp(log_expr.value))