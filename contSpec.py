#
# Help to find continuous retardation spectrum
#

# This is largely based on pyReSpect-time for extracting the continuous and discrete relaxation spectrum from G(t)
# using a Bayesian framework for deciding the strength of the regularization constraint.
#

from common import *

# HELPER FUNCTIONS

def InitializeSpectrum(texp, Jexp, s, kernMat, isLiquid):
    """
    Function: InitializeH(input)
    
    Input:  texp       = n*1 vector [t],
            Jexp       = n*1 vector [Jt],
               s       = relaxation modes,
               kernMat = matrix for faster kernel evaluation
               isLiquid = optional; if liquid
               
     Output:   Lspec = guessed L spectrum
    """
    #
    # To guess spectrum, pick a spectrum that is consistent with a large value of lambda to get a
    # solution that is most determined by the regularization
    # March 2019; a single guess is good enough now, because going from large lambda to small
    #             lambda in lcurve.

    Lspec = -5.0 * np.ones(len(s)) + np.sin(np.pi * s)
    lam   = 1e1
    
    if isLiquid:
        invEta0 = (Jexp[-1] - Jexp[-2])/(texp[-1] - texp[-2]) # guess dJ/dt t -> infty
        Je      = max(Jexp) - invEta0*texp[-1]
        Lplus= np.append(Lspec, [Je, invEta0])

    else:
        invEta0  = 0.
        Je       = max(Jexp)        
        Lplus    = np.append(Lspec, Je)


    LplusImp = getH(lam, texp, Jexp, Lplus, kernMat)

    # plt.plot(s, LplusImp[:-2])
    # plt.plot(s, Lplus[:-2])    
    # plt.xscale('log')
    # plt.show()


    # plt.clf()
    # K = kernel_prestore(LplusImp, kernMat, texp, Jexp)
    # plt.loglog(texp, Jexp,'o', texp, K, 'k-')
    # plt.xlabel(r'$t$')
    # plt.ylabel(r'$J(t)$')
    
    # plt.tight_layout()
    # plt.show()

    # quit()

    return LplusImp


def getAmatrix(ns):
    """Generate symmetric matrix A = L' * L required for error analysis:
       helper function for lcurve in error determination"""
    # L is a ns*ns tridiagonal matrix with 1 -2 and 1 on its diagonal;
    nl = ns - 2
    L  = np.diag(np.ones(ns-1), 1) + np.diag(np.ones(ns-1),-1) + np.diag(-2. * np.ones(ns))
    L  = L[1:nl+1,:]
    return np.dot(L.T, L)
    
def getBmatrix(Lplus, kernMat, texp, Jexp):
    """get the Bmatrix required for error analysis; helper for lcurve()
       not explicitly accounting for G0 in Jr because otherwise I get underflow problems"""
 
    n   = kernMat.shape[0];
    ns  = kernMat.shape[1];
    nl  = ns - 2;
    r   = np.zeros(n);         # vector of size (n);

    # furnish relevant portion of Jacobian and residual
    Kmatrix = np.dot((1./Jexp).reshape(n,1), np.ones((1,ns)));
    Jr      = -kernelD(Lplus[:ns], kernMat) * Kmatrix;    
    r       = 1. - kernel_prestore(Lplus, kernMat, texp, Jexp)/Jexp
    B       = np.dot(Jr.T, Jr) + np.diag(np.dot(r.T, Jr))

    return B


# changing interface
def lcurve(texp, Jexp, Lgs, kernMat, par):

    """ 
     Function: lcurve(input)
    
     Input: t/Jexp  = n*1 vector [t/Jt],
            Lgs     = guessed LplusGuess,
            kernMat = matrix for faster kernel evaluation
            par     = parameter dictionary

     Output: lamC and 3 vectors of size npoints*1 contains a range of lambda, rho
             and eta. "Elbow"  = lamC is estimated using a *NEW* heuristic AND by Hansen method
             
             
    March 2019: starting from large lambda to small cuts calculation time by a lot
                also gives an error estimate 
             
    """

    n     = len(Jexp)
    ns    = kernMat.shape[1];
    nl    = ns - 2
    Lplus = Lgs.copy()
        

    # lambda specific data
    npoints = int(par['lamDensity'] * (np.log10(par['lam_max']) - np.log10(par['lam_min'])))
    hlam    = (par['lam_max']/par['lam_min'])**(1./(npoints-1.))    
    lam     = par['lam_min'] * hlam**np.arange(npoints)
    eta     = np.zeros(npoints)
    rho     = np.zeros(npoints)
    logP    = np.zeros(npoints)
    logPmax = -np.inf                    # so nothing surprises me!

    Lplambda = np.zeros((len(Lplus), npoints))    # all the different Lspecs


    # Error Analysis: Furnish A_matrix (depends only on Lspec)
    Amat       = getAmatrix(ns)
    _, LogDetN = np.linalg.slogdet(Amat)
        
    #
    # This is the costliest step
    #
    for i in reversed(range(len(lam))):
        
        
        lamb   = lam[i]

        Lplus  = getH(lamb, texp, Jexp, Lplus, kernMat)
        rho[i] = np.linalg.norm((1. - kernel_prestore(Lplus, kernMat, texp, Jexp)/Jexp))
        Bmat   = getBmatrix(Lplus, kernMat, texp, Jexp)            

        eta[i]       = np.linalg.norm(np.diff(Lplus[:ns], n=2))
        Lplambda[:,i] = Lplus



        _, LogDetC = np.linalg.slogdet(lamb * Amat + Bmat)
        V          =  rho[i]**2 + lamb * eta[i]**2        
                    
        # this assumes a prior exp(-lam*1e6)
        # ~ logP[i]    = -V + 0.5 * (LogDetN + ns*np.log(lamb) - LogDetC) - lamb*1e6
        logP[i]    = -V + 0.5 * (LogDetN + ns*np.log(lamb) - LogDetC) - lamb
        
        # # print progress
        # if i == len(lam)-1:
        #     print("\n")
        # print('{:2d} {:.2e} {:.2e} {:.2e} {:.2e}'.format(i, lamb, rho[i], eta[i], logPmax-logP[i]))

        
        if(logP[i] > logPmax):
            logPmax = logP[i]
        elif(logP[i] < logPmax - 12):
            break


    # truncate all to significant lambda
    lam  = lam[i:]
    logP = logP[i:]
    eta  = eta[i:]
    rho  = rho[i:]
    logP = logP - max(logP)

    Lplambda = Lplambda[:,i:]
    
    #
    # new lamC discard old!
    #
    plam = np.exp(logP); plam = plam/np.sum(plam)
    lamC = np.exp(np.sum(plam*np.log(lam)))

    #
    # Dialling in the Smoothness Factor
    #
    if par['SmFacLam'] > 0:
        lamC = np.exp(np.log(lamC) + par['SmFacLam']*(max(np.log(lam)) - np.log(lamC)));
    elif par['SmFacLam'] < 0:
        lamC = np.exp(np.log(lamC) + par['SmFacLam']*(np.log(lamC) - min(np.log(lam))));

    #
    # printing this here for now because storing lamC for sometime only
    #
    if par['plotting']:
        plt.clf()
        plt.axvline(x=lamC, c='gray', label=r'$\lambda_c$')
        plt.ylim(-15,1)
        plt.plot(lam, logP, 'o-')
        plt.xscale('log')
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$\log\,p(\lambda)$')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig('output/logP.pdf')
        
        
    return lamC, lam, rho, eta, logP, Lplambda

def getH(lam, texp, Jexp, Lplus, kernMat):

    """Purpose: Given a lambda, this function finds the H_lambda(s) that minimizes V(lambda)
    
              V(lambda) := ||Jexp - kernel(L)||^2 +  lambda * ||F L||^2
    
     Input  : lambda     = regularization parameter ,
              texp, Jexp = experimental data,
              Lspec      = guessed Lspec,
              kernMat    = matrix for faster kernel evaluation
              Je         = initial Je guess
              invEta0    = optional
    
     Output : L_lam, Je, [invEta0]
              Default uses Trust-Region Method with Jacobian supplied by jacobianLM
    """

    ns  = kernMat.shape[1];
    nex = len(Lplus) - ns
    res_lsq = least_squares(residualLM, Lplus, jac=jacobianLM, args=(lam, texp, Jexp, kernMat))

    return res_lsq.x

def residualLM(Lplus, lam, texp, Jexp, kernMat):
    """
    %
    % HELPER FUNCTION: Gets Residuals r
     Input  : Lplus   = guessed [Lspec, Je] or [Lspec, Je, eta0inv]
              lambda  = regularization parameter ,
              t/Jexp  = experimental data,
             kernMat = matrix for faster kernel evaluation
    
     Output : a set of n+nl residuals,
              the first n correspond to the kernel
              the last  nl correspond to the smoothness criterion
    %"""


    n   = kernMat.shape[0];
    ns  = kernMat.shape[1];
    nl  = ns - 2;

    r   = np.zeros(n + nl);
    
    # normal residuals
    r[0:n] = 1. - kernel_prestore(Lplus, kernMat, texp, Jexp)/Jexp  # the Jt and
    
    # the curvature constraint is not affected by G0
    r[n:n+nl] = np.sqrt(lam) * np.diff(Lplus[:ns], n=2)  # second derivative

        
    return r
        
def jacobianLM(Lplus, lam, texp, Jexp, kernMat):
    """
    HELPER FUNCTION for optimization: Get Jacobian J
    
    returns a (n+nl * ns) matrix Jr; (ns + 1) if G0 is also supplied.
    
    Jr_(i, j) = dr_i/dH_j
    
    It uses kernelD, which approximates dK_i/dH_j, where K is the kernel
    
    """
    n   = kernMat.shape[0];
    ns  = kernMat.shape[1];
    nl  = ns - 2;
    nex = len(Lplus) - ns
    

    # L is a nl*ns tridiagonal matrix with 1 -2 and 1 on its diagonal;
    L  = np.diag(np.ones(ns-1), 1) + np.diag(np.ones(ns-1),-1) + np.diag(-2. * np.ones(ns))
    L  = L[1:nl+1,:]    
    
    
    # Furnish the Jacobian Jr (n+ns)*ns matrix
    Kmatrix         = np.dot((1./Jexp).reshape(n,1), np.ones((1,ns)));

    # ~ if len(H) > ns:

    Lspec = Lplus[:ns]
    Jr    = np.zeros((n + nl, ns+nex))

    Jr[0:n, 0:ns]   = -kernelD(Lspec, kernMat) * Kmatrix;
    Jr[0:n, ns]     = -1./Jexp                             # column for dr_i/dJe
    Jr[n:n+nl,0:ns] = np.sqrt(lam) * L;

    if nex == 2:
        Jr[0:n, ns+1]  = -texp/Jexp                        # column for dr_i/dinvEta0
         
    return Jr

def kernelD(Lspec, kernMat):
    """
     Function: kernelD(input)
    
     outputs the (n*ns) dimensional matrix DK(L)(t)
     It approximates dK_i/dL_j = K * e(L_j):
    
     Input: L       = substituted CRS,
            kernMat = matrix for faster kernel evaluation
    
     Output: DK = Jacobian of L
    """
    
    n   = kernMat.shape[0];
    ns  = kernMat.shape[1];

    # A n*ns matrix with all the rows = L'
    Lsuper  = np.dot(np.ones((n,1)), np.exp(Lspec).reshape(1, ns))  
    DK      = -kernMat  * Lsuper                                        # -ve sign for retardation spectrum
        
    return DK
    
def getContSpec(par):
    """
    This is the main driver routine for computing the continuous spectrum
    
    (*)   input  : "par" dictionary from "inp.dat" which specifies JexpFile (often 'Jt.dat')
    (*)   return : H and lambdaC; the latter can be used to microscpecify lambdaC as desired
                    without having to do the entire lcurve calculation again
    """
    # read input
    if par['verbose']:
        print('\n(*) Start\n(*) Loading Data File: {}...'.format(par['JexpFile']))

    texp, Jexp = GetExpData(par['JexpFile'])

    if par['verbose']:
        print('(*) Initial Set up...', end="")
  
    # Set up some internal variables
    n    = len(texp)
    ns   = par['ns']    # discretization of 'tau'

    tmin = texp[0];
    tmax = texp[n-1];
    
    # determine frequency window
    if par['FreqEnd'] == 1:
        smin = np.exp(-np.pi/2) * tmin; smax = np.exp(np.pi/2) * tmax        
    elif par['FreqEnd'] == 2:
        smin = tmin; smax = tmax                
    elif par['FreqEnd'] == 3:
        smin = np.exp(+np.pi/2) * tmin; smax = np.exp(-np.pi/2) * tmax        

    hs   = (smax/smin)**(1./(ns-1))
    s    = smin * hs**np.arange(ns)
    
    kernMat = getKernMat(s, texp)
    tic     = time.time()
        
    # get an initial guess for Lgs, Je, invEta0
    LpGS = InitializeSpectrum(texp, Jexp, s, kernMat, par['liquid'])

    if par['verbose']:
        te   = time.time() - tic
        print('\t({0:.1f} seconds)\n(*) Building the L-curve ...'.format(te), end="")    
        tic  = time.time()

    # Find Optimum Lambda with 'lcurve'
    if par['lamC'] == 0:
        lamC, lam, rho, eta, logP, Llam = lcurve(texp, Jexp, LpGS, kernMat, par)
    else:
        lamC = par['lamC']

    if par['verbose']:
        te = time.time() - tic
        print('({1:.1f} seconds)\n(*) Extracting CRS, ...\n\t... lamC = {0:0.3e}; '.
              format(lamC, te), end="")
        
        tic  = time.time()

    # Get the best spectrum

    if par['lamC'] == 0:
        Lplus = getH(lamC, texp, Jexp, Llam[:,-1], kernMat)
    else:
        Lplus = getH(lamC, texp, Jexp, LpGS, kernMat)
    
    print('Je = {0:0.3e};'.format(Lplus[ns]), end="")

    if par['liquid']:
        print('invEta0 = {0:0.3e} ...'.format(Lplus[ns+1]), end="")

    #----------------------
    # Print some datafiles
    #----------------------

    if par['verbose']:
        te = time.time() - tic
        print('done ({0:.1f} seconds)\n(*) Writing and Printing, ...'.format(te), end="")

        # Save inferred J(t)
        K = kernel_prestore(Lplus, kernMat, texp, Jexp)

        if par['liquid']:
            np.savetxt('output/L.dat', np.c_[s, Lplus[:ns]], fmt='%e', 
                        header='Je, invEta0 = {0:0.3e}\t{1:0.3e}'.format(Lplus[ns], Lplus[ns+1]))
        else:
            np.savetxt('output/L.dat', np.c_[s, Lplus[:ns]], fmt='%e', 
                        header='Je = {0:0.3e}'.format(Lplus[ns]))
            
        np.savetxt('output/Jfit.dat', np.c_[texp, K], fmt='%e')

        # print Llam, rho-eta, and logP if lcurve has been visited
        if par['lamC'] == 0:
            if os.path.exists("output/Llam.dat"):
                os.remove("output/Llam.dat")
                
            fLlam = open('output/Llam.dat','ab')
            for i, lamb in enumerate(lam):
                np.savetxt(fLlam, Llam[:,i])    # this include Je [and invEta0]
            fLlam.close()    

            # print logP
            np.savetxt('output/logPlam.dat', np.c_[lam, logP])
        
            # print rho-eta
            np.savetxt('output/rho-eta.dat', np.c_[lam, rho, eta], fmt='%e')

    #------------
    # Graphing
    #------------

    if par['plotting']:

        # plot spectrum "L.pdf" with errorbars
        plt.clf()

        plt.semilogx(s, Lplus[:ns],'o-')
        plt.xlabel(r'$s$')
        plt.ylabel(r'$L(s)$')

        # error bounds are only available if lcurve has been implemented
        if par['lamC'] == 0:
            plam = np.exp(logP); plam = plam/np.sum(plam)            
            Lm   = np.zeros(len(s))
            Lm2  = np.zeros(len(s))
            cnt  = 0
            for i in range(len(lam)):    
                # count all spectra within a threshold
                if plam[i] > 0.1:
                    Lm   += Llam[:ns,i]
                    Lm2  += Llam[:ns,i]**2
                    cnt  += 1

            Lm = Lm/cnt
            dL = np.sqrt(Lm2/cnt - Lm**2)

            plt.semilogx(s, Lm+2.5*dL, c='gray', alpha=0.5)
            plt.semilogx(s, Lm-2.5*dL, c='gray', alpha=0.5)

        plt.tight_layout()
        plt.savefig('output/L.pdf')


        #
        # plot comparison with input spectrum
        #

        plt.clf()


        K = kernel_prestore(Lplus, kernMat, texp, Jexp)

        plt.plot(texp, Jexp, 'o', alpha=0.7)
        plt.plot(texp, K, 'C1')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$t$')
        plt.ylabel(r'$J(t)$')
        
        plt.tight_layout()
        plt.savefig('output/Jfit.pdf')
        #
        # if lam not explicitly specified then print rho-eta.pdf
        #
        try:
            lam
        except NameError:
          print("lamC prespecified, so not printing rho-eta.pdf/dat")
        else:
            plt.clf()
            plt.scatter(rho, eta, marker='x')
            plt.plot(rho, eta)


            rhost = np.exp(np.interp(np.log(lamC), np.log(lam), np.log(rho)))
            etast = np.exp(np.interp(np.log(lamC), np.log(lam), np.log(eta)))

            plt.plot(rhost, etast, 'o', color='C1')
            plt.xscale('log')
            plt.yscale('log')
                        
            plt.xlabel(r'$\rho$')
            plt.ylabel(r'$\eta$')
            plt.tight_layout()
            plt.savefig('output/rho-eta.pdf')

    if par['verbose']:
        print('done\n(*) End\n')
            
    return Lplus, lamC

    
#     
# Main Driver: This part is not run when contSpec.py is imported as a module
#              For example as part of GUI
#
if __name__ == '__main__':
    #
    # Read input parameters from file "inp.dat"
    #
    par = readInput('inp.dat')
    Lspec, lamC = getContSpec(par)

