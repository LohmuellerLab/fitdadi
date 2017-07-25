#! /usr/bin/env python

"""
This script generates an approximate demography file that's compatible with prefersim.

Usage:
./make_prefersim_demography.py [dataset_name] [mu] [LNS/LS]

dataset_name can be:
1kg : 1000 Genomes EUR
esp : Exome Sequencing Project 6400 downsampled to 2600
lucamp : LuCamp Danish exomes

mu should be in units of 10^(-8)

LNS/LS is the ratio of the nonsynonymous mutation rate to the synonymous mutation rate
"""

import numpy, sys

def getpopn():
    while True:
        dataset = raw_input("""
Which of these results would you like to simulate? Please enter a number.
1. 1000 Genomes EUR
2. Exome Sequencing Project (ESP6400)
3. LuCamp
        \n:""")
        if dataset == "1":
            return "1kg"
        elif dataset == "2":
            return "esp"
        elif dataset == "3":
            return "lucamp"
        else:
            print "That's not a valid option."

def getmut():
    while True:
        mutrate = raw_input("""
Which mutation rate assumption? Please enter a number.
1. mu=1.5e-8, nonsynonymous to synonymous ratio 2.31
2. mu=1.8e-8, nonsynonymous to synonymous ratio 2.5 
        \n:""")
        if mutrate == "1":
            return ("1.5","2.31")
        elif mutrate == "2":
            return ("1.8","2.5")
        else:
            print "That's not a valid option."

def getprog():
    while True:
        simprog = raw_input("""
Which simulation program? Please enter a number.
1. PReFerSIM (Ortega-Del Vecchyo et al. 2016)
2. SLiM (Messer 2013)
3. SFS_CODE (Hernandez 2008)
        \n:""")
        if simprog == "1":
            return "prefersim"
        elif simprog == "2":
            return "slim"
        elif simprog == "3":
            return "sfscode"
        else:
            print "That's not a valid option."

def getsamplesize():
    while True:
        samplesize = raw_input("""
What sample size would you like to simulate? Please enter an integer in terms of diploids.\n""")
        return int(samplesize)

def main():
    dataset = getpopn()
    #mu, mutratio = getmut()
    mu = "1.5"; mutratio="2.31"
    simprog = getprog()
    samplesize = getsamplesize()
    N1, T1, N2, T2, Nc, Tc, theta, targetsize = params_d[dataset]["demography"]
    Nanc = theta/(4*float(mu)*1e-8*(targetsize/(float(mutratio)+1)))
    nanc = 2*Nanc
    if simprog == "slim":
        dfe, dfe_params = params_d[dataset]["selection"][mu]
        outfilename = '{0}_{1}_n{2}_SLiM.txt'.format(dataset, mu, samplesize)
        outfile = open(outfilename, 'w')
        Ns = [1, N1, N2, Nc]
        Ns = [x*Nanc for x in Ns]
        Ts = [1, (T1)*nanc + 1, (T1+T2)*nanc + 1, (T1+T2+Tc)*nanc + 1]
        nt = lambda t: Ns[2]*numpy.exp(numpy.log(Ns[3]/Ns[2])*(t)/(Ts[3]-Ts[2]))
        Ns = Ns[0:2] + [nt(x) for x in range(0,1+int(round(Ts[3]-Ts[2])))]
        Ts = Ts[0:3] + [Ts[2]+x for x in range(1,2+int(round(Ts[3]-Ts[2])))]
        Ts = [int(round(x)) for x in Ts]
        Ns = [int(round(x)) for x in Ns]
        demog_file = '\n'.join(['{0} {{ p1.setSubpopulationSize({1}); }}'.format(y,x) for x,y in zip(Ns[1:],Ts)])
        if dfe == "gamma":
            shape, scale = dfe_params
            scale = scale/(2.*nanc)
            mean = -2*shape*scale
            outline = slim_init_gamma.format(mu, mean, shape, Ns[0], demog_file, Ts[-1], samplesize)
        elif dfe == "neugamma":
            pneu, shape, scale = dfe_params
            scale = scale/(2.*nanc)
            mean = -2*shape*scale
            outline = slim_init_neugamma.format(mu, mean, shape, (1-pneu), pneu, Ns[0], demog_file, Ts[-1], samplesize)
        outfile.write(outline)
        outfile.close()
        print "Please use {0}".format('{0}_{1}_n{2}_SLiM.txt'.format(dataset, mu, samplesize))
    elif simprog == "prefersim":
        dfe, dfe_params = params_d[dataset]["selection"][mu]
        demofilename = '{0}_{1}_n{2}_prefersim_demo.txt'.format(dataset, mu, samplesize)
        Ns = [1, N1, N2, Nc]
        Ns = [x*2*Nanc for x in Ns]
        Ts = [8*Nanc, (T1)*nanc, T2*nanc, Tc*nanc]
        Ns = [int(round(x)) for x in Ns]
        Ts = [int(round(x)) for x in Ts]
        outfile = open(demofilename, 'w')
        for size, time in zip(Ns[:-1],Ts[:-1]):
            outline = '{0} {1}\n'.format(size, time)
            outfile.write(outline)
        nt = lambda t: Ns[2]*numpy.exp(numpy.log(Ns[3]/Ns[2])*(t)/(Ts[3]))
        for i in range(0,498):
            outline = '{0} {1}\n'.format(int(round(nt(i))),1)
            outfile.write(outline)
        outfile.close()
        if dfe == "gamma":
            outfilename = '{0}_{1}_n{2}_prefersim.txt'.format(dataset, mu, samplesize)
            outfile = open(outfilename, 'w')
            outline = prefersim_init_gamma.format(demofilename, dfe_params[0], dfe_params[1], Nanc, samplesize)
            outfile.write(outline)
            outfile.close()
            print "Please use {0}".format('{0}_{1}_n{2}_prefersim.txt'.format(dataset, mu, samplesize))
        elif dfe == "neugamma":
            outfilename = '{0}_{1}_n{2}_prefersim_gamma.txt'.format(dataset, mu, samplesize)
            outfile = open(outfilename, 'w')
            outline = prefersim_init_gamma.format(demofilename, dfe_params[1], dfe_params[2], Nanc, samplesize)
            outfile.write(outline)
            outfile.close()
            outfilename = '{0}_{1}_n{2}_prefersim_neutral.txt'.format(dataset, mu, samplesize)
            outfile = open(outfilename, 'w')
            outline = prefersim_init_gamma.format(demofilename, samplesize)
            outfile.write(outline)
            outfile.close()
            print 'You must multiply (i.e. rescale) the SFS created by {0} by {1} and the SFS created by {2} by {3}, then sum those together to get the correct SFS.'.format('{0}_{1}_n{2}_prefersim_gamma.txt'.format(dataset, mu, samplesize), dfe_params[0], '{0}_{1}_n{2}_prefersim_neutral.txt'.format(dataset, mu, samplesize), (1-dfe_params[0]))
    elif simprog == "sfscode":
        dfe, dfe_params = params_d[dataset]["selection"][mu]
        outfilename = '{0}_{1}_n{2}_SLiM.txt'.format(dataset, mu, samplesize)
        outfile = open(outfilename, 'w')
        if dfe == "gamma":
            outfilename = '{0}_{1}_n{2}_sfscode_gamma.sh'.format(dataset, mu, samplesize)
            outfile = open(outfilename, 'w')
            alpha = numpy.log(Nc/N2)/Tc
            outline = sfscode_init_gamma.format(int(round(Nanc)), samplesize, N1, N2/N1, T1+T2, alpha, T1+T2+Tc, dfe_params[0], 1/dfe_params[1])
            outfile.write(outline)
            outfile.close()
        elif dfe == "neugamma":
            print "neutral+gamma mixture distribution unavailable with SFS_CODE at the moment, sorry..."

prefersim_init_gamma = """MutationRate: 2000
DemographicHistory: {0} 
DFEType: gamma
DFEParameterOne: {1}
DFEParameterTwo: {2}
DFEParameterThree: {3}
PrintSFS: 1
FilePrefix: gamma
n: {4}
"""

prefersim_init_neutral = """MutationRate: 2000
DemographicHistory: {0}
DFEType: point
DFEPopintSelectionCoefficient: 0
PrintSFS: 1
FilePrefix: neutral
n: {1}
"""

slim_init_gamma = """// set up the simulation
initialize(){{
    initializeMutationRate({0}e-8);
    
    // m1 mutation type: gamma
    // note: some rescaling is done since we use 1,1+2sh,1+s and SLiM uses 1,1+hs,1+s
    initializeMutationType("m1", 0.5, "g", {1}, {2});
    
    // g1 genomic element type: uses 100% m1
    initializeGenomicElementType("g1", c(m1), c(1.0));

    // uniform chromosome of length 1 Mb with uniform recombination
    initalizeGenomicElement(g1, 0, 999999);
    initalizeRecombinationRate(1e-8);
}}

// this is the population demography
1 {{ sim.addSubpop("p1", {3}); }}
{4}

//output samples at the end
{5} late() {{ p1.outputSample({6}); }}
"""

slim_init_neugamma = """// set up the simulation
initialize(){{
    initializeMutationRate({0}e-8);
    
    // m1 mutation type: gamma
    // note: some rescaling is done since we use 1,1+2sh,1+s and SLiM uses 1,1+hs,1+s
    initializeMutationType("m1", 0.5, "g", {1}, {2});
    
    // m2 mutation type: neutral
    initializeMutationType("m2", 0.5, "f", 0.0);
    
    // g1 genomic element type: uses mixture of gamma and neutral
    initializeGenomicElementType("g1", c(m1,n2), c({3},{4}));

    // uniform chromosome of length 1 Mb with uniform recombination
    initalizeGenomicElement(g1, 0, 999999);
    initalizeRecombinationRate(1e-8);
}}

// this is the population demography
1 {{ sim.addSubpop("p1", {5}); }}
{6}

//output samples at the end
{7} late() {{ p1.outputSample({8}); }}
"""

sfscode_init_gamma = ("./sfs_code 1 200 -t 0.001 -L 10 5000 R -a C R "
"-N 10085 -n 2600 -Td 0.0 0.098984772 -Td 0.01 1.051243 -Tg 0.087303612 292.97280352296781 "
"-TE 0.098883812 -W 2 0 0 0 0.184 0.0004019467 --additive --outfile test.txt")

sfscode_init_neugamma = ("./sfs_code 1 200 -t 0.001 -L 10 5000 R -a C R "
"-N 10085 -n 2600 -Td 0.0 0.098984772 -Td 0.01 1.051243 -Tg 0.087303612 292.97280352296781 "
"-TE 0.098883812 -W 2 0 0 0 0.184 0.0004019467 --additive --outfile test.txt")

params_d = {
"lucamp":{"demography":[0.08984,0.01,1.0512,0.07304,31.270,0.01158,4261.2,20043582],"selection":{"1.5":["neugamma",[0.164,0.338,367.7]],"1.8":["discrete",[0.278,0.027,0.211,0.352]]}},
"esp":{"demography":[0.11949,0.01,1.3111,0.05254,98.65,0.01502,6415.1,31427992],"selection":{"1.5":["neugamma",[0.092,0.207,1082.3]],"1.8":["gamma",[0.164,2527.8]]}},
"1kg":{"demography":[0.08469,0.01,1.1007,0.07043,53.283,0.02009,5984.9,26673114],"selection":{"1.5":["gamma",[0.186,875.]],"1.8":["gamma",[0.181,1585.1]]}}
}
    
if __name__=="__main__":
    main()
