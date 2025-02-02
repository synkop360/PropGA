import matplotlib.pyplot as plt
import numpy as np
import random

from deap import base, creator, tools, algorithms

# Genetic Algorithm constants:
POPULATION_SIZE = 200 # number of individuals in population
P_CROSSOVER = 0.7 # probability for crossover
P_MUTATION = 0.3 # probability for mutating an individual
MAX_GENERATIONS = 100 # max number of generations for stopping condition

# Constants
rho = 1025 # Masse volumique eau de mer (kg/m3)
Va = 14.42 # Vitesse du navire (m/sec)
n = 720 # Vitesse de rotation de l'hélice (tr/min)
Pd = 440000 # Puissance à l'hélice (W)
Z = 5 # Nombre de pales
P0 = 101325 + 30000 # Static pressure at the propeller shaft (2m under water) (Pa)
Pv = 1267 # Vapor pressure of water (Pa)
k = 0.2 # Wake number, (0 for ships with smoot wake, 0.2 for single screw propeller)

ktCoef = np.array([
                    [+0.00880496,0,0,0,0],
                    [-0.204554,1,0,0,0],
                    [+0.166351,0,1,0,0],
                    [+0.158114,0,2,0,0],
                    [-0.147581,2,0,1,0],
                    [-0.481497,1,1,1,0],
                    [+0.415437,0,2,1,0],
                    [+0.0144043,0,0,0,1],
                    [-0.0530054,2,0,0,1],
                    [+0.0143481,0,1,0,1],
                    [+0.0606826,1,1,0,1],
                    [-0.0125894,0,0,1,1],
                    [+0.0109689,1,0,1,1],
                    [-0.133698,0,3,0,0],
                    [+0.00638407,0,6,0,0],
                    [-0.00132718,2,6,0,0],
                    [+0.168496,3,0,1,0],
                    [-0.0507214,0,0,2,0],
                    [+0.0854559,2,0,2,0],
                    [-0.0504475,3,0,2,0],
                    [+0.010465,1,6,2,0],
                    [-0.00648272,2,6,2,0],
                    [-0.00841728,0,3,0,1],
                    [+0.0168424,1,3,0,1],
                    [-0.00102296,3,3,0,1],
                    [-0.0317791,0,3,1,1],
                    [+0.018604,1,0,2,1],
                    [-0.00410798,0,2,2,1],
                    [-0.000606848,0,0,0,2],
                    [-0.0049819,1,0,0,2],
                    [+0.0025983,2,0,0,2],
                    [-0.000560528,3,0,0,2],
                    [-0.00163652,1,2,0,2],
                    [-0.000328787,1,6,0,2],
                    [+0.000116502,2,6,0,2],
                    [+0.000690904,0,0,1,2],
                    [+0.00421749,0,3,1,2],
                    [+0.0000565229,3,6,1,2],
                    [-0.00146564,0,3,2,2]
                    ])

kqCoef =  np.array([  
                    [+0.00379368,0,0,0,0],
                    [+0.00886523,2,0,0,0],
                    [-0.032241,1,1,0,0],
                    [+0.00344778,0,2,0,0],
                    [-0.0408811,0,1,1,0],
                    [-0.108009,1,1,1,0],
                    [-0.0885381,2,1,1,0],
                    [+0.188561,0,2,1,0],
                    [-0.00370871,1,0,0,1],
                    [+0.00513696,0,1,0,1],
                    [+0.0209449,1,1,0,1],
                    [+0.00474319,2,1,0,1],
                    [-0.00723408,2,0,1,1],
                    [+0.00438388,1,1,1,1],
                    [-0.0269403,0,2,1,1],
                    [+0.0558082,3,0,1,0],
                    [+0.0161886,0,3,1,0],
                    [+0.00318086,1,3,1,0],
                    [+0.015896,0,0,2,0],
                    [+0.0471729,1,0,2,0],
                    [+0.0196283,3,0,2,0],
                    [-0.0502782,0,1,2,0],
                    [-0.030055,3,1,2,0],
                    [+0.0417122,2,2,2,0],
                    [-0.0397722,0,3,2,0],
                    [-0.00350024,0,6,2,0],
                    [-0.0106854,3,0,0,1],
                    [+0.00110903,3,3,0,1],
                    [-0.000313912,0,6,0,1],
                    [+0.0035985,3,0,1,1],
                    [-0.00142121,0,6,1,1],
                    [-0.00383637,1,0,2,1],
                    [+0.0126803,0,2,2,1],
                    [-0.00318278,2,3,2,1],
                    [+0.00334268,0,6,2,1],
                    [-0.00183491,1,1,0,2],
                    [+0.000112451,3,2,0,2],
                    [-0.0000297228,3,6,0,2],
                    [+0.000269551,1,0,1,2],
                    [+0.00083265,2,0,1,2],
                    [+0.00155334,0,2,1,2],
                    [+0.000302683,0,6,1,2],
                    [-0.0001843,0,0,2,2],
                    [-0.000425399,0,3,2,2],
                    [+0.0000869243,3,3,2,2],
                    [-0.0004659,0,6,2,2],
                    [+0.0000554194,1,6,2,2]
                    ])

# Helping functions
def Kt (individual):
    result = 0
    for n in ktCoef:
        result += n[0]*(J(individual)**n[1])*(individual[1]**n[2])*(individual[0]**n[3])*(individual[4]**n[4])

    return result

def Kq (individual):
    result = 0
    for n in kqCoef:
        result += n[0]*(J(individual)**n[1])*(individual[1]**n[2])*(individual[0]**n[3])*(individual[4]**n[4])

    return result

def J (individual):
    return Va/(individual[3]*individual[2]/60)

def T (individual):
    return Kt(individual)*rho*((individual[3]/60)**2)*(individual[2]**4)

def EARCalc (individual):
    return ((1.3+0.3*individual[4])*T(individual))/((P0-Pv)*(individual[2]**2)) + k

def efficiency (individual):
    return (Kt(individual)*J(individual)) / (Kq(individual) * 2 * np.pi)

def evaluate(individual):
    objective1 = efficiency(individual)
    objective2 = T(individual)

    return objective1, objective2

def feasible(individual):
    c1 = True if 0 < efficiency(individual) <= 1 else False
    c2 = True if (Kt(individual) <= 1.6) and (0 <= Kt(individual)) else False
    c3 = True if (10*Kq(individual) <= 1.6) and (0 <= 10*Kq(individual)) else False
    c4 = True if (individual[0] - EARCalc(individual)) >= 0 else False
    c5 = True if 0.5 < individual[1] <= 1.4 else False
    c6 = True if 2 <= individual[4] <= 7 else False

    return c1 and c2 and c3 and c4 and c5 and c6

# Types creation to maximise 2 objectives
creator.create("FitnessMax", base.Fitness, weights=(1.0, 2.0))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Toolbox configuration
param_names = ["EAR", "PDR", "D", "n", "Z"]
param_ranges = {
    "EAR": (0.45, 1.05),
    "PDR": (0.5, 1.4),
    "D": (0.5, 10),
    "n": (100, 1000),
    "Z": (2, 7)
}

def generate_param(EAR=float, PDR=float, D=float, n=float, Z=int):
    args = locals()
    return[random.uniform(*param_ranges[name]) if args[name] == float else 
           random.randint(*param_ranges[name]) if args[name]==int else 
           args[name] for name in param_names]

toolbox = base.Toolbox()
toolbox.register("attr_param", generate_param, float, float, float, 720, int)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_param)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Custom mutation function to mutate differente types of genes if allowed
def mutate(individual, indpb, mutType, mutParam):
    """
    :param individual: Individual to be mutated.
    :param indpb: Independent probability for each attribute to be mutated.
    :param mutType: List of mutation types to apply to each gene.
    :param mutParam: List of parameters of each mutation
    :returns: A tuple of one individual.
    """
    for i in range(len(individual)):
        if random.random()<indpb:
            match mutType[i]:
                case None:
                    next
                case "mutGauss":
                    individual[i] = random.gauss(individual[i], mutParam[i]*individual[i])
                case "mutUniformInt":
                    individual[i] = random.randint(individual[i] - mutParam[i][0], individual[i] + mutParam[i][0])
                case _:
                    print("Not a valid mutation")
    return individual,

# Custom mating function to mate differente types of genes if allowed
def mate(ind1, ind2, matType, matParam):
    """
    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param matType: List of mating types to apply to each gene.
    :param matParam: List of parameters of each mating
    :returns: A tuple of two individuals.
    """
    for i in range(len(ind1)):
        match matType[i]:
            case None:
                next
            case "cxBlend":
                gamma = (1. + 2. * matParam[i]) * random.random() - matParam[i]
                ind1[i] = (1. - gamma) * ind1[i] + gamma * ind2[i]
                ind2[i] = gamma * ind1[i] + (1. - gamma) * ind2[i]
            case "cxUniform":
                if random.random() < matParam[i]:
                    ind1[i], ind2[i] = ind2[i], ind1[i]
            case _:
                print("Not a valid mating")
    return ind1, ind2

toolbox.register("mate", mate, matType=["cxBlend", "cxBlend", "cxBlend", None, "cxUniform"], matParam=[0.1, 0.1, 0.1, None, 0.5])
toolbox.register("mutate", mutate, indpb=0.2, mutType=["mutGauss", "mutGauss", "mutGauss", None, "mutUniformInt"], mutParam=[0.05, 0.05, 0.05, None, [1,1]])
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", evaluate)
toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 0.1))


# Paramètres de l'algorithme
def main():
    pop_size = 200
    generations = 250
    cx_prob = 0.6  # Probabilité de croisement
    mut_prob = 0.4  # Probabilité de mutation

    pop = toolbox.population(n=pop_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    logbook = tools.Logbook()

    hof = tools.HallOfFame(10)
    algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size, cxpb=cx_prob, mutpb=mut_prob, ngen=generations, stats=stats, halloffame=hof, verbose=True)
    
    return pop, hof, logbook

if __name__ == "__main__":
    pop, hof, log = main()
    print("Meilleure solution trouvée:")
    print(hof, hof[0].fitness.values)
    print(Kt(hof[0]))
    print(Kq(hof[0]))
    print(EARCalc(hof[0]))
    print(T(hof[0]))