from pydrake.all import SnoptSolver, GurobiSolver, IpoptSolver, OsqpSolver

print(f"SNOPT available: {SnoptSolver().available()}")
print(f"SNOPT enabled: {SnoptSolver().enabled()}")

print(f"\nGurobi available: {GurobiSolver().available()}")
print(f"\Gurobi enabled: {GurobiSolver().enabled()}")

print(f"\nIPOPT available: {IpoptSolver().available()}")
print(f"IPOPT enabled: {IpoptSolver().enabled()}")

print(f"\nOSQP available: {OsqpSolver().available()}")
print(f"OSQP enabled: {OsqpSolver().enabled()}")