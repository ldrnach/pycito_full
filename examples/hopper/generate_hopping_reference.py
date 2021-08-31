import examples.hopper.hoppingopt as opt 
import os

def main():
    hopper = opt.create_hopper()
    x0, xf = opt.boundary_conditions(hopper)
    weights = [1, 10, 100, 1000, 10000]
    result = None
    basedir = os.path.join('examples','hopper','reference_linear')
    trajopt = opt.create_hopper_optimization_contact_cost(hopper, x0, xf, N=101)
    for weight in weights:
        savedir = os.path.join(basedir,f"weight_{weight}")
        trajopt.complementarity_cost_weight = weight
        trajopt = opt.set_hopper_initial_conditions(trajopt, result, (x0, xf))
        result = opt.solve_hopper_opt_and_save(trajopt, savedir)

if __name__ == '__main__':
    main()