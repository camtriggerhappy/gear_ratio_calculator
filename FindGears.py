import cvxpy as cp
import numpy as np

# Gear ratios
A = np.asarray([3, 4, 5, 9])
logA = np.log(A)

print("enter 1 or 2 ")
choice = int(input("Do you wish to calculate the gear ratio from a torque value(1) or do you wish to calculate the closest gear ratio to a gear ratio(2)"))

in_torque = None
fos = None
out_torque = None

if(choice == 1):
    in_torque = float(input("what is the input torque(nm) to the gearbox"))
    fos = float(input("What is the desired Factor of Safety"))
    out_torque = float(input("What is the needed output torque(nm) (no FOS)"))

min_ratio = None
if(choice == 2):
    min_ratio = float(input("What is your desired minimum gear ratio"))
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


gear_names = ["3:1 Cartridge", "4:1 Cartridge", "5:1 Cartridge", "9:1 Cartridge"]

x_vals = np.round(x.value).astype(int)

print("\n================ GEAR SELECTION SUMMARY ================\n")

for name, count in zip(gear_names, x_vals):
    print(f"{name:20} : {count}")

print("\n---------------------------------------------------------")
print(f"Total number of cartridges : {np.sum(x_vals)}")
print(f"Total ratio                : {np.exp(log_expr.value):.3f}")
if(in_torque):
    print(f"Input torque(nm)           : {in_torque}")
    print(f"Output torque(nm)          : {in_torque * np.exp(log_expr.value):.3f}")
    print(f"Requested Torque(nm)       : {out_torque}")
print(f"Factor of Safety           : {fos} ")
print("=========================================================\n")
# -------------------- Stage Torque Calculation (with before/after) --------------------

# Sort gears by descending ratio so largest stages go first
sorted_indices = np.argsort(-A)
sorted_ratios = A[sorted_indices]
sorted_counts = x_vals[sorted_indices]
sorted_names = [gear_names[i] for i in sorted_indices]

stage_torques = []
current_torque = in_torque
if(not in_torque):
    exit(0)
print("\n================ STAGE TORQUES =================\n")


stage_num = 1
for name, ratio, count in zip(sorted_names, sorted_ratios, sorted_counts):
    for _ in range(count):
        print(f"Stage {stage_num:2d} ({name:8s} x1, ratio {ratio:3d})")
        print(f"  Torque before stage : {current_torque:.3f} Nm")
        current_torque *= ratio
        stage_torques.append(current_torque)
        print(f"  Torque after stage  : {current_torque:.3f} Nm\n")
        stage_num += 1

print("---------------------------------------------------------")
print(f"Final output torque : {current_torque:.3f} Nm")
print("=========================================================\n")
