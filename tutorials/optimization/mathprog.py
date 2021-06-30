"""
Solve an optimization problem
    min x(0)^2 + x(1)^2 
    s.t. x(0) + x(1) = 1
        x(0) <= x(1)

this example also demonstrates how to add a callback to the program, and use it to visualize the progression of the solver.

Adapted from the MathematicalProgram tutorial on the Drake website: https://drake.mit.edu 
"""
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve 
import numpy as np 
import matplotlib.pyplot as plt  

# Create empty MathematicalProgram
prog = MathematicalProgram()

# Add 2 continuous decision variables
# x is a numpy array - we can use an optional second argument to name the variables
x = prog.NewContinuousVariables(2, name='worlds')
#print(x)
prog.AddConstraint(x[0] + x[1] == 1).evaluator().set_description("equality constraint")
prog.AddConstraint(x[0] <= x[1]).evaluator().set_description("inequality constraint")
prog.AddCost(x[0]**2 + x[1]**2).evaluator().set_description("cost")

# Make and add a visualization callback
fig = plt.figure()
curve_x = np.linspace(1,10,100)
ax = plt.gca()
ax.plot(curve_x, 9./curve_x)
ax.plot(-curve_x, -9./curve_x)
ax.plot(0,0,'o')
x_init = [4., 5.]
point_x, = ax.plot(x_init[0], x_init[1], 'x')
ax.axis('equal')

def update(x):
    global iter_count
    point_x.set_xdata(x[0])
    point_x.set_ydata(x[1])
    ax.set_title(f"iteration {iter_count}")
    fig.canvas.draw()
    fig.canvas.flush_events()
    # Update iter_counter
    iter_count += 1
    plt.pause(1.0)
# Add a visualization callback - it does more than just visualization
iter_count = 0
prog.AddVisualizationCallback(update, x)

# Solve the optimization problem - optional second argument is initial guess. Third argument is solver parameters
result = Solve(prog, x_init, None)
# Print the result
print("Success? ", result.is_success())
# Print the solution
print("x* = ", result.GetSolution(x))
# Print the optimal cost
print('optimal cost = ', result.get_optimal_cost())
# Print the name of the solver
print('solver is: ', result.get_solver_id().name())

# We can also evaluate the costs and constraints after the fact
costs = prog.GetAllCosts()
cstrs = prog.GetAllConstraints()

for cost in costs:
    dvars = cost.variables()
    dvals = result.GetSolution(dvars)
    cost_val = cost.evaluator().Eval(dvals)
    print(f"The value of cost {cost.evaluator().get_description()} is {cost_val}")

for cstr in cstrs:
    dvars = cstr.variables()
    dvals = result.GetSolution(dvars)
    cstr_val = cstr.evaluator().Eval(dvals)
    print(f"The value of constraint {cstr.evaluator().get_description()} is {cstr_val}")