"""
Distribution functions to wrap demographics with scalar selection. Most
of this code is modified dadi code, and the selection stuff is a 
modified version of the script found at: 
https://groups.google.com/forum/#!topic/dadi-user/4xspqlITcvc

There are a few changes to the integration, so that anything below the
lower bound is assumed to be effectively neutral; and anything above
the lower bound is assumed to not be segregating, and weighted 0.

I added in multiprocessing capabilities because generating all the 
spectra takes really long for large values of gamma. One workaround
is to not integrate where the gammas are large, since the SFSes there
should be close to 0... 
"""

import os
import sys
import operator
import numpy
from numpy import logical_and, logical_not
from scipy.special import gammaln
import scipy.stats.distributions
import scipy.integrate
import scipy.optimize
from dadi import Numerics, Inference, Misc
from dadi.Spectrum_mod import Spectrum


class spectra:
    def __init__(self, params, ns, demo_sel_func, pts=500, pts_l=None, 
                 Npts=500, n=20170., int_breaks=None, 
                 int_bounds=(1e-4, 1000.), mp=False, echo=False, cpus=None):
        """
        params: optimized demographic parameters, don't include gamma 
        here
        demo_sel_func: dadi demographic function with selection. gamma
        must be the last argument.
        ns: sample sizes
        Npts: number of grid points over which to integrate
        steps: use this to set break points for spacing out the 
        intervals
        mp: True if you want to use multiple cores (utilizes 
        multiprocessing) if using the mp option you must also specify
        # of cpus, otherwise this will just use nthreads-1 on your
        machine
        """
        
        self.ns = ns
        self.spectra = []
        
        #create a vector of gammas that are log-spaced over sequential 
        #intervals or log-spaced over a single interval.
        if not (int_breaks is None):
            numbreaks = len(int_breaks)
            stepint = Npts/(numbreaks-1)
            self.gammas = []
            for i in reversed(range(0,numbreaks-1)):
                self.gammas = numpy.append(
                    self.gammas, -numpy.logspace(numpy.log10(int_breaks[i+1]), 
                                                 numpy.log10(int_breaks[i]), 
                                                 stepint))
        else:
            self.gammas = -numpy.logspace(numpy.log10(int_bounds[1]), 
                                          numpy.log10(int_bounds[0]), Npts)
        if pts_l == None:
            self.pts = pts
            self.pts_l = [self.pts,self.pts+(self.pts/5), \
                          self.pts+(self.pts/5)*2]
        else:
            self.pts_l = pts_l
        func_ex = Numerics.make_extrap_func(demo_sel_func)
        self.params = tuple(params)
        
        if not mp: #for running with a single thread
            for ii,gamma in enumerate(self.gammas):
                self.spectra.append(func_ex(tuple(params)+(gamma,), self.ns,
                                            self.pts_l))
                if echo:
                    print '{0}: {1}'.format(ii, gamma)
        else: #for running with with multiple cores
            import multiprocessing
            if cpus is None:
                cpus = multiprocessing.cpu_count() - 1
            def worker_sfs(in_queue, outlist, popn_func_ex, params, ns, 
                           pts_l):
                """
                Worker function -- used to generate SFSes for 
                single values of gamma. 
                """
                while True:
                    item = in_queue.get()
                    if item == None:
                        return
                    ii, gamma = item
                    sfs = popn_func_ex(tuple(params)+(gamma,), ns, pts_l)
                    print '{0}: {1}'.format(ii, gamma)
                    result = (gamma, sfs)
                    outlist.append(result)
            manager = multiprocessing.Manager()
            results = manager.list()
            work = manager.Queue(cpus)
            pool = []
            for i in xrange(cpus):
                p = multiprocessing.Process(target=worker_sfs, 
                                            args=(work, results, func_ex,
                                            params, self.ns, self.pts_l))
                p.start()
                pool.append(p)
            for ii,gamma in enumerate(self.gammas):
                work.put((ii, gamma))
            for jj in xrange(cpus):
                work.put(None)
            for p in pool:
                p.join()
            reslist = []
            for line in results:
                reslist.append(line)
            reslist.sort(key = operator.itemgetter(0))
            for gamma, sfs in reslist:
                self.spectra.append(sfs)
        
        #self.neu_spec = demo_sel_func(params+(0,), self.ns, self.pts)
        self.neu_spec = func_ex(tuple(params)+(0,), self.ns, self.pts_l)
        self.extrap_x = self.spectra[0].extrap_x
        self.spectra = numpy.array(self.spectra)
        
    def integrate(self, params, sel_dist, theta):
        """
        integration without re-normalizing the DFE. This assumes the
        portion of the DFE that is not integrated is not seen in your
        sample. 
        """
        #need to include tuple() here to make this function play nice
        #with numpy arrays
        sel_args = (self.gammas,) + tuple(params)
        #compute weights for each fs
        weights = sel_dist(*sel_args)
        
        #compute weight for the effectively neutral portion. not using
        #CDF function because I want this to be able to compute weight
        #for arbitrary mass functions
        weight_neu, err_neu = scipy.integrate.quad(sel_dist, self.gammas[-1], 
                                                   0, args=tuple(params))
        
        #function's adaptable for demographic models from 1-3 populations
        pops = len(self.neu_spec.shape)
        if pops == 1:
            integrated = self.neu_spec*weight_neu + Numerics.trapz(
                weights[:,numpy.newaxis]*self.spectra, self.gammas, axis=0)
        elif pops == 2:
            integrated = self.neu_spec*weight_neu + Numerics.trapz(
                weights[:,numpy.newaxis,numpy.newaxis]*self.spectra, 
                self.gammas, axis=0)
        elif pops == 3:
            integrated = self.neu_spec*weight_neu + Numerics.trapz(
                weights[:,numpy.newaxis,numpy.newaxis,numpy.newaxis]*self.spectra,
                self.gammas, axis=0)
        else:
            raise IndexError("Must have one to three populations")
        
        integrated_fs = Spectrum(integrated, extrap_x=self.extrap_x)
        
        #no normalization, allow lethal mutations to fall out
        return integrated_fs * theta
        
    def integrate_norm(self, params, sel_dist, theta):
        """
        """
        #need to include tuple() here to make this function play nice
        #with numpy arrays
        #compute weights for each fs
        sel_args = (self.gammas,) + tuple(params)
        weights = sel_dist(*sel_args)
        
        #compute weight for the effectively neutral portion. not using
        #CDF function because I want this to be able to compute weight
        #for arbitrary mass functions
        weight_neu, err_neu = scipy.integrate.quad(sel_dist, self.gammas[-1], 
                                                   0, args=tuple(params))
        
        #function's adaptable for demographic models from 1-3
        #populations but this assumes the selection coefficient is the
        #same in both populations
        pops = len(self.neu_spec.shape)
        if pops == 1:
            integrated = self.neu_spec*weight_neu + Numerics.trapz(
            	weights[:,numpy.newaxis]*self.spectra, self.gammas, axis=0)
        elif pops == 2:
            integrated = self.neu_spec*weight_neu + Numerics.trapz(
            	weights[:,numpy.newaxis,numpy.newaxis]*self.spectra, 
            	self.gammas, axis=0)
        elif pops == 3:
            integrated = self.neu_spec*weight_neu + Numerics.trapz(
            	weights[:,numpy.newaxis,numpy.newaxis,numpy.newaxis]*self.spectra,
            	self.gammas, axis=0)
        else:
            raise IndexError("Must have one to three populations")
        
        integrated_fs = Spectrum(integrated, extrap_x=self.extrap_x)
        
        #normalization
        dist_int = Numerics.trapz(weights, self.gammas) + weight_neu
        return integrated_fs/dist_int * theta


#define a bunch of default distributions just to make everything easier
def gamma_dist(mgamma, alpha, beta):
    """
    x, shape, scale
    """
    return scipy.stats.distributions.gamma.pdf(-mgamma, alpha, scale=beta)


def beta_dist(mgamma, alpha, beta):
    """
    x, alpha, beta
    """
    return scipy.stats.distributions.beta.pdf(-mgamma, alpha, beta)


def exponential_dist(mgamma, scale):
    return scipy.stats.distributions.expon.pdf(-mgamma, scale=scale)


def lognormal_dist(mgamma, mu, sigma, scal_fac=1):
    return scipy.stats.distributions.lognorm.pdf(
        -mgamma, sigma, scale=numpy.exp(mu + numpy.log(scal_fac)))


def normal_dist(mgamma, mu, sigma):
    return scipy.stats.distributions.norm.pdf(-mgamma, loc=mu, scale=sigma)

def neugamma(mgamma, p, alpha, beta):
        mgamma=-mgamma
        if (0 <= mgamma) and (mgamma < -smallgamma):
                return p/(-smallgamma) + (1-p)*dadi.Selection.gamma_dist(
                    -mgamma,alpha, beta)
        else:
                return dadi.Selection.gamma_dist(-mgamma, alpha, beta) * (1-p)

#The following code has been taken from dadi.Inference and slightly
#modified for compatibilty with the pre-generated spectra. There's also
#the additional option to use constrained optimization for complex
#mixture distributions.

#: Stores thetas
_theta_store = {}
#: Counts calls to object_func
_counter = 0
#: Returned when object_func is passed out-of-bounds params or gets a NaN ll.
_out_of_bounds_val = -1e8


def _object_func(params, data, model_func, sel_dist, theta,
                 lower_bound=None, upper_bound=None, 
                 verbose=0, multinom=False, flush_delay=0,
                 func_args=[], func_kwargs={}, fixed_params=None, ll_scale=1,
                 output_stream=sys.stdout, store_thetas=False):
    """
    Objective function for optimization.
    """
    global _counter
    _counter += 1
    
    # Deal with fixed parameters
    params_up = Inference._project_params_up(params, fixed_params)
    
    # Check our parameter bounds
    if lower_bound is not None:
        for pval,bound in zip(params_up, lower_bound):
            if bound is not None and pval < bound:
                return -_out_of_bounds_val/ll_scale
    if upper_bound is not None:
        for pval,bound in zip(params_up, upper_bound):
            if bound is not None and pval > bound:
                return -_out_of_bounds_val/ll_scale
     
    all_args = [params_up, sel_dist, theta] + list(func_args)
    # Pass the pts argument via keyword, but don't alter the passed-in 
    # func_kwargs
    #func_kwargs = func_kwargs.copy()
    #func_kwargs['pts'] = pts
    sfs = model_func(*all_args, **func_kwargs)
    if multinom:
        result = Inference.ll_multinom(sfs, data)
    else:
        result = Inference.ll(sfs, data)
    
    if store_thetas:
        global _theta_store
        _theta_store[tuple(params)] = optimal_sfs_scaling(sfs, data)
    
    # Bad result
    if numpy.isnan(result):
        result = _out_of_bounds_val
    
    if (verbose > 0) and (_counter % verbose == 0):
        param_str = 'array([%s])' % (', '.join(['%- 12g'%v for v in params_up]))
        output_stream.write('%-8i, %-12g, %s%s' % (_counter, result, param_str,
                                                   os.linesep))
        Misc.delayed_flush(delay=flush_delay)
    
    return -result/ll_scale


def optimize_log(p0, data, model_func, sel_dist, theta, lower_bound=None, 
                 upper_bound=None, verbose=0, flush_delay=0.5, epsilon=1e-3, 
                 gtol=1e-5, multinom=False, maxiter=None, full_output=False,
                 func_args=[], func_kwargs={}, fixed_params=None, ll_scale=1,
                 output_file=None):
    
    if output_file:
        output_stream = file(output_file, 'w')
    else:
        output_stream = sys.stdout
    
    args = (data, model_func, sel_dist, theta, lower_bound, upper_bound,
            verbose, multinom, flush_delay, func_args, func_kwargs,
            fixed_params, ll_scale, output_stream)
    
    p0 = Inference._project_params_down(p0, fixed_params)
    outputs = scipy.optimize.fmin_bfgs(_object_func_log, 
                                       numpy.log(p0), epsilon=epsilon,
                                       args = args, gtol=gtol, 
                                       full_output=True,
                                       disp=False,
                                       maxiter=maxiter)
    xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = outputs
    xopt = Inference._project_params_up(numpy.exp(xopt), fixed_params)
    
    if output_file:
        output_stream.close()
    
    if not full_output:
        return [-fopt, xopt]
    else:
        return xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag


def optimize_cons(p0, data, model_func, sel_dist, theta, lower_bound=None,
                  upper_bound=None, verbose=0, flush_delay=0.5, epsilon=1e-4,
                  constraint=None, gtol=1e-6, multinom=False, maxiter=None,
                  full_output=False, func_args=[], func_kwargs={},
                  fixed_params=None, ll_scale=1, output_file=None):
    """
    Constrained optimization needs a constraint function and bounds. 
    """
    
    if output_file:
        output_stream = file(output_file, 'w')
    else:
        output_stream = sys.stdout

    if not (lower_bound is None):
        lower_bound_a = lower_bound + [0]
    if not (upper_bound is None):
        upper_bound_a = upper_bound + [numpy.inf]
    
    args = (data, model_func, sel_dist, theta, lower_bound, upper_bound,
            verbose, multinom, flush_delay, func_args, func_kwargs,
            fixed_params, ll_scale, output_stream)

    p0 = Inference._project_params_down(p0, fixed_params)
    
    ####make sure to define consfunc and bnds ####
    if (not lower_bound is None) and (not upper_bound is None):
        bnds = tuple((x,y) for x,y in zip(lower_bound,upper_bound))
    outputs = scipy.optimize.fmin_slsqp(_object_func, 
                                       p0, bounds=bnds, args=args, 
                                       f_eqcons=constraint, epsilon=epsilon, 
                                       iter=maxiter,full_output=True,
                                       disp=False)
    xopt, fopt, func_calls, grad_calls, warnflag = outputs
    xopt = Inference._project_params_up(xopt, fixed_params)
    
    if output_file:
        output_stream.close()
    
    if not full_output:
        return [-fopt, xopt]
    else:
        return xopt, fopt, func_calls, grad_calls, warnflag


def _object_func_log(log_params, *args, **kwargs):
    """
    Objective function for optimization in log(params).
    """
    return _object_func(numpy.exp(log_params), *args, **kwargs)


def optimize(p0, data, model_func, sel_dist, theta, lower_bound=None,
             upper_bound=None, verbose=0, flush_delay=0.5, epsilon=1e-3, 
             gtol=1e-5, multinom=False, maxiter=None, full_output=False,
             func_args=[], func_kwargs={}, fixed_params=None, ll_scale=1,
             output_file=None):
    """
    optimizer for use with distributions where log transformations do not work,
    e.g. when gamma is positive and negative
    """
    if output_file:
        output_stream = file(output_file, 'w')
    else:
        output_stream = sys.stdout

    args = (data, model_func, sel_dist, theta, lower_bound, upper_bound, 
            verbose, multinom, flush_delay, func_args, func_kwargs,
            fixed_params, ll_scale, output_stream)

    p0 = _project_params_down(p0, fixed_params)
    outputs = scipy.optimize.fmin_bfgs(_object_func, p0, 
                                       epsilon=epsilon,
                                       args = args, gtol=gtol, 
                                       full_output=True,
                                       disp=False,
                                       maxiter=maxiter)
    xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = outputs
    xopt = Inference._project_params_up(xopt, fixed_params)

    if output_file:
        output_stream.close()

    if not full_output:
        return xopt
    else:
        return xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag

##end of dadi.Inference code

