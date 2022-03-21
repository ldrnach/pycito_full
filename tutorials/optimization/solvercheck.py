from pydrake.all import SnoptSolver, GurobiSolver, IpoptSolver, OsqpSolver

print(f"SNOPT available: {SnoptSolver().available()}")
print(f"Gurobi available: {GurobiSolver().available()}")
print(f"IPOPT available: {IpoptSolver().available()}")
print(f"OSQP available: {OsqpSolver().available()}")