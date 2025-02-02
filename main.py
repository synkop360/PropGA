import matplotlib.pyplot as plt
import numpy as np
import random

from deap import creator
from deap import base
from deap import tools
from deap import algorithms

# Genetic Algorithm constants:
POPULATION_SIZE = 200 # number of individuals in population
P_CROSSOVER = 0.7 # probability for crossover
P_MUTATION = 0.3 # probability for mutating an individual
MAX_GENERATIONS = 100 # max number of generations for stopping condition

rho = 1025 # Masse volumique eau de mer (kg/m3)
Va = 14.42 # Vitesse du navire (m/sec)
n = 720 # Vitesse de rotation de l'hélice (tr/min)
Pd = 440000 # Puissance à l'hélice (W)
Z = 5 # Nombre de pales
P0 = 101325 + 20000 # Static pressure at the propeller shaft (2m under water) (Pa)
Pv = 1267 # Vapor pressure of water (Pa)
k = 0.2 # Wake number, (0 for ships with smoot wake, 0.2 for single screw propeller)

EAR_LOW = 0.45
EAR_UP = 1.05
PDR_LOW = 0.5
PDR_UP = 1.4
D_LOW = 0.5
D_UP = 10.0
RPS_LOW = 100
RPS_UP = 1000


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

def Kt (J, PDR, EAR, Z):
    result = 0
    for n in ktCoef:
        result += n[0]*(J**n[1])*(PDR**n[2])*(EAR**n[3])*(Z**n[4])

    return result

def Kq (J, PDR, EAR, Z):
    result = 0
    for n in kqCoef:
        result += n[0]*(J**n[1])*(PDR**n[2])*(EAR**n[3])*(Z**n[4])

    return result

def J (D):
    return Va/(n*D/60)

def T (rho, n, D, Kt):
    return Kt*rho*((n/60)**2)*(D**4)

def EARCalc (T, Z, P0, Pv, D, k):
    return ((1.3+0.3*Z)*T)/((P0-Pv)*(D**2)) + k

def efficiency (PDR, EAR, D):
    return (Kt(J(D), PDR, EAR, Z)*J(D)) / (Kq(J(D), PDR, EAR, Z) * 2 * np.pi)


print(J(1.01))
print(Kt(J(1.01), 1.4, 0.750, 5))
print(Kq(J(1.01), 1.4, 0.750, 5))
print(efficiency(1.4, 0.750, 1.01))
print(T(rho, n, 1.01, Kt(J(1.01), 1.4, 0.750, 5)))

toolbox = base.Toolbox()

creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
creator.create("Individual", list, fitness=creator.FitnessMax)

def initIndiv():
    EAR = random.uniform(EAR_LOW, EAR_UP)
    PDR = random.uniform(PDR_LOW, PDR_UP)
    D = random.uniform(D_LOW, D_UP)
    return (EAR, PDR, D)

def evalFitness(individual):
    return efficiency(individual[0], individual[1], individual[2]),

def ktConstraint(individual):
    kt = Kt(J(individual[2]), individual[0], individual[1], Z)
    return  (kt <= 1.6) and (0 <= kt)

def kqConstraint(individual):
    kq = Kq(J(individual[2]), individual[0], individual[1], Z)
    return  (10*kq <= 1.6) and (0 <= 10*kq)

def EARConstraint(individual):
    kt = Kt(J(individual[2]), individual[0], individual[1], Z)
    t = T(rho, n, individual[2], kt)
    EAR = EARCalc (t, Z, P0, Pv, individual[2], k)
    return (individual[1] - EAR) >= 0    


def feasible(individual):
    return ktConstraint(individual) and kqConstraint(individual) and EARConstraint(individual) and 0 < efficiency(individual[0], individual[1], individual[2]) < 1 and PDR_LOW <= individual[0] and individual[0] <= PDR_UP

def distance(individual):
    """A distance function to the feasibility region."""
    return (efficiency(individual[0], individual[1], individual[2]) - 5.0)**2

toolbox.register("select", tools.selTournament, tournsize = 3)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma =0.05, indpb = 0.2)
toolbox.register("mate", tools.cxBlend, alpha=0.1)
toolbox.register("evaluate", evalFitness)
toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 0.01))

toolbox.register("attr_pdr", random.uniform, PDR_LOW, PDR_UP)
toolbox.register("attr_ear", random.uniform, EAR_LOW, EAR_UP)
toolbox.register("attr_d", random.uniform, D_LOW, D_UP)

toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_pdr, toolbox.attr_ear, toolbox.attr_d), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def main():
    pop = toolbox.population(n=POPULATION_SIZE)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    logbook = tools.Logbook()

    hof = tools.HallOfFame(10)
    """
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    print(pop)

    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    for gen in range(1, MAX_GENERATIONS):
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= P_MUTATION:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print(len(pop))
        print(offspring)

        pop = toolbox.select(pop + offspring, POPULATION_SIZE)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)
    """
    algorithms.eaMuPlusLambda(pop, toolbox, mu=POPULATION_SIZE, lambda_=POPULATION_SIZE, cxpb = P_CROSSOVER, mutpb = P_MUTATION, ngen = MAX_GENERATIONS, stats=stats, 
                        halloffame=hof, verbose = True)
    
    minFitnessValues, meanFitnessValues, maxFitnessValues = logbook.select("min", "avg", "max")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.plot(maxFitnessValues, color='purple') 
    plt.xlabel('Generation')
    plt.ylabel('Min / Max / Average Fitness')
    plt.title('Min, Max and Average fitness over Generations')
    plt.show()

    return pop, hof, logbook


if __name__ == "__main__":
    pop, hof, logbook = main()
    print(hof)
    i = 0
    for elem in hof:
        
        print("-----" + str(i) + "-----")

        """
        print(efficiency(elem[0], elem[1], elem[2]))
        print(EARConstraint(elem))
        print(ktConstraint(elem))
        print(kqConstraint(elem))

        kt = Kt(J(elem[2]), elem[0], elem[1], Z)
        t = T(rho, n, elem[2], kt)
        print(kt)
        print(t)
        print(EARCalc(t, Z, P0, Pv, elem[2], k))
        """

        print("PDR : " + str(elem[0]))
        print("D : " + str(elem[2]))
        print("EAR : " + str(elem[1]))
        print("J : " + str(J(elem[2])))
        print("Kt : " + str(Kt(J(elem[2]), elem[0], elem[2], Z)))
        print("Kq : " + str(Kq(J(elem[2]), elem[0], elem[2], Z)))
        print("Efficiency : " + str(efficiency(elem[0], elem[1], elem[2])))
        print("Thrust : " + str(T(rho, n, elem[2], Kt(J(elem[2]), elem[0], elem[2], Z))))

        i=i+1
