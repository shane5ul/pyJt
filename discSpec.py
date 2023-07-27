#
# 7/2023: allowing an optional weight column in the input data file + encapsulation of private functions
#         cleaning up nomenclature to be more compliance friendly
# 3/2019: Based on pyReSpect-time with G- adjustment
#
#
# 12/15/2018: 
# (*) Introducing NNLS optimization of previous optimal solution
#     - can make deltaBaseWeightDist : 0.2 or 0.25 [coarser from 0.05]
#     - wrote new routines: FineTuneSolution, res_tG (for vector of residuals)
# 

from common import *
np.set_printoptions(precision=2)

def initializeDiscSpec(par):
    """Returns:
        (*) the experimental data: texp, Jexp, wexp
        (*) the continuous spectrum: s, L (from output/L.dat)
        (*) Number of modes range: Nv
        (*) Error Weight estimate from Continuous Curve (AIC criterion)
    """
    
    # read input; initialize parameters
    if par['verbose']:
        print('\n(*) Start\n(*) Loading Data Files: ... {}...'.format(par['JexpFile']))

    # Read experimental data
    texp, Jexp, wexp = GetExpData(par['JexpFile'])

    # Read the continuous spectrum
    fNameH  = 'output/L.dat'
    s, Lspec = np.loadtxt(fNameH, unpack=True)

    n    = len(texp);
    
    # range of N scanned
    Nmin  = max(np.floor(0.5 * np.log10(max(texp)/min(texp))),2);   # minimum Nopt
    Nmax  = min(np.floor(3.0 * np.log10(max(texp)/min(texp))),n/4); # maximum Nopt

    if(par['MaxNumModes'] > 0):
        Nmax  = min(Nmax, par['MaxNumModes'])
    
    Nv    = np.arange(Nmin, Nmax + 1).astype(int)

    # Estimate Error Weight from Continuous Curve Fit
    kernMat = getKernMat(s, texp)
    
    # Read Je and invEta0
    try:
        with open(fNameH) as f:
            first_line = f.readline().split()
    except OSError:
        print('Problem reading G0 from H.dat; Plateau = True')
        quit()

    if par['liquid']:
        invEta0 = float(first_line[-1]); 
        Je      = float(first_line[-2]);
        Lplus= np.append(Lspec, [Je, invEta0])        
    else:
        Je      = float(first_line[-1]);
        Lplus= np.append(Lspec, Je)
        
    Jc = kernel_prestore(Lplus, kernMat, texp)
    
    Cerror  = 1./(np.std(wexp*(Jc/Jexp - 1.)))  #    Cerror = 1.?
    
    return texp, Jexp, wexp, s, Lplus, Nv, Jc, Cerror

def MaxwellModes(z, t, Jexp, wexp, isLiquid):
    """    
     Function: MaxwellModes(input)
    
     Solves the linear least squares problem to obtain the DRS

     Input: z  = points distributed according to the density, [z = log(lam)]
            t  = n*1 vector contains times,
            Jexp = n*1 vector contains J(t),
		    wexp = weight vector of experimental data
            isLiquid = True/False
    
     Output: j, lam = spectrum  (array)
             error = relative error between the input data and the G(t) inferred from the DRS
             condKp = condition number
    """
    N      = len(z)
    lam    = np.exp(z)

    #
    # Prune small and -ve weights j(i)
    #
    j, error, condKp = nnLLS(t, lam, Jexp, wexp, isLiquid)
        
    # search for small 
    izero = np.where(j[:N]/max(j[:N]) < 1e-7)
    
    lam   = np.delete(lam, izero)
    j     = np.delete(j, izero)

    return j, lam, error, condKp

def nnLLS(t, lam, Jexp, wexp, isLiquid):
    """
    # 4/14/2023: changing this to ensure Je > sum(j)
    #
    # Helper subfunction which does the actual LLS problem
    # helps MaxwellModes; relies on nnls
    #
    """

    # The K matrix has 1-exp(-t/s) to ensure Je > sum(j)
    n       = len(Jexp)
    nmodes  = len(lam)
    S, T    = np.meshgrid(lam, t)
    K       = 1 - np.exp(-T/S)        # n * nmodes (-ve sign for compliance) 4/23/2023

    # K is n*np [where np = ns+1 or ns+2]
    if isLiquid:
        K = np.hstack(( K, np.ones((n, 1)), t.reshape((n,1)) ))    # 1 for Je, and t for invEta0
    else:
        K = np.hstack(( K, np.ones((len(Jexp), 1)) ))
        
    #
    # gets wE*(Jt/JtE - 1)^2, instead of  (Jt -  JtE)^2
	# 7/27/2023: note with wexp; RHS becomes wexp rather than ones [wLLS is X'WX = X'Wy]
    #
    Kp      = np.dot(np.diag((wexp/Jexp)), K)
    condKp  = np.linalg.cond(Kp)
    j       = nnls(Kp, wexp)[0]

    JtM       = np.dot(K, j)
    error     = np.sum((wexp*(JtM/Jexp - 1.))**2)
        
    # Jep only lives inside NNLS
    j[nmodes] = j[nmodes] + np.sum(j[:nmodes])

    return j, error, condKp

def GetWeights(H, t, s, wb):
    """

    #
    # to test from below here!
    #
    #

    %
    % Function: GetWeights(input)
    %
    % Finds the weight of "each" mode by taking a weighted average of its contribution
    % to J(t)
    %
    % Input: H = CRS (ns * 1)
    %        t = n*1 vector contains times
    %        s = relaxation modes (ns * 1)
    %       wb = weightBaseDist
    %
    % Output: wt = weight of each mode
    %
    """
  
    ns         = len(s)
    n          = len(t)
    hs         = np.zeros(ns)
    wt         = hs
    
    hs[0]      = 0.5 * np.log(s[1]/s[0])
    hs[ns-1]   = 0.5 * np.log(s[ns-1]/s[ns-2])
    hs[1:ns-1] = 0.5 * (np.log(s[2:ns]) - np.log(s[0:ns-2]))

    S, T    = np.meshgrid(s, t)
    kern    = np.exp(-T/S)        # n * ns
    wij     = np.dot(kern, np.diag(hs * np.exp(H)))  # n * ns
    K       = np.dot(kern, hs * np.exp(H))         # n * 1, comparable with Jexp

    for i in np.arange(n):
        wij[i,:] = wij[i,:]/K[i]

    for k in np.arange(ns):
        wt[k] = np.sum(wij[:,k])

    wt  = wt/np.trapz(wt, np.log(s))
    wt  = (1. - wb) * wt + (wb * np.mean(wt)) * np.ones(len(wt))

    return wt

def GridDensity(x, px, N):

    """#
    #  PROGRAM: GridDensity(input)
    #
    #    Takes in a PDF or density function, and spits out a bunch of points in
    #       accordance with the PDF
    #
    #  Input:
    #       x  = vector of points. It need *not* be equispaced,
    #       px = vector of same size as x: probability distribution or
    #            density function. It need not be normalized but has to be positive.
    #          N  = Number of points >= 3. The end points of "x" are included
    #           necessarily,
    # 
    #  Output:
    #       z  = Points distributed according to the density
    #       hz = width of the "intervals" - useful to apportion domain to points
    #            if you are doing quadrature with the results, for example.
    #
    #  (c) Sachin Shanbhag, November 11, 2015
    #"""

    npts = 100;                              # can potentially change
    xi   = np.linspace(min(x),max(x),npts)   # reinterpolate on equi-spaced axis
    fint = interp1d(x,px,'cubic')             # smoothen using cubic splines
    pint = fint(xi)                             # interpolation
    ci   = cumtrapz(pint, xi, initial=0)                
    pint = pint/ci[npts-1]
    ci   = ci/ci[npts-1]                     # normalize ci

    alfa = 1./(N-1)                          # alfa/2 + (N-1)*alfa + alfa/2
    zij  = np.zeros(N+1)                     # quadrature interval end marker
    z    = np.zeros(N)                       # quadrature point

    z[0]    = min(x);  
    z[N-1]  = max(x); 

    # ci(Z_j,j+1) = (j - 0.5) * alfa
    beta       = np.arange(0.5, N-0.5) * alfa
    zij[0]     = z[0]
    zij[N]     = z[N-1]
    fint       = interp1d(ci, xi, 'cubic')
    zij[1:N]   = fint(beta)
    h          = np.diff(zij)

    # Quadrature points are not the centroids, but rather the center of masses
    # of the quadrature intervals
    beta     = np.arange(1, N-1) * alfa
    z[1:N-1] = fint(beta)

    return z, h

def mergeModes_magic(j, lam, imode):
    """merge modes imode and imode+1 into a single mode
       return gp and lamp corresponding to this new mode;
       12/2018 - also tries finetuning before returning
       
       uses helper functions:
       - normKern_magic()
       - costFcn_magic()   
    """
    ### begin encapsulation
    def costFcn_magic(par, j, lam, imode):
        """"helper function for mergeModes; establishes cost function to minimize"""

        jn   = par[0]
        lamn = par[1]

        j1   = j[imode]
        j2   = j[imode+1]
        lam1 = lam[imode]
        lam2 = lam[imode+1]

        tmin = min(lam1, lam2)/10.
        tmax = max(lam1, lam2)*10.

        def normKern_magic(t, jn, lamn, j1, lam1, j2, lam2):
            """helper function: for costFcn and mergeModes"""
            Jn = jn * np.exp(-t/lamn)
            Jo = j1 * np.exp(-t/lam1) + j2 * np.exp(-t/lam2)
            return (Jn/Jo - 1.)**2

        return quad(normKern_magic, tmin, tmax, args=(jn, lamn, j1, lam1, j2, lam2))[0]
    ###end encapsulation

    iniGuess = [j[imode] + j[imode+1], 0.5*(lam[imode] + lam[imode+1])]
    res      = minimize(costFcn_magic, iniGuess, args=(j, lam, imode))

    newlam   = np.delete(lam, imode+1)
    newlam[imode] = res.x[1]
        
    return newlam

def FineTuneSolution(lam, t, Jexp, wexp, isLiquid, estimateError=False):
    """Given a spacing of modes lam, tries to do NLLS to fine tune it further
       If it fails, then it returns the old lam back
       
       Uses helper function: res_tG which computes residuals
       """
    ### begin encapsulation
    def res_tG(lam, texp, Jexp, wexp, isLiquid):
        """
            Helper function for final optimization problem
        """
        j, _, _ = nnLLS(texp, lam, Jexp, wexp, isLiquid)
        Jmodel  = np.zeros(len(texp))

        for k in range(len(lam)):
            Jmodel -= j[k] * np.exp(-texp/lam[k])
        
        # add Je + t * invEta0
        if isLiquid:
            Jmodel += j[-2] + texp * j[-1]
        else:
            Jmodel += j[-1]
            
        residual = wexp * (Jmodel/Jexp - 1.)
            
        return residual
    ###end encapsulation

    success = False
           
    try:
        res  = least_squares(res_tG, lam, bounds=(0., np.inf), args=(t, Jexp, wexp, isLiquid))
        lam  = res.x
        lam0 = lam.copy()

        # Error Estimate    
        if estimateError:
            J = res.jac
            cov = np.linalg.pinv(J.T.dot(J)) * (res.fun**2).mean()
            dlam = np.sqrt(np.diag(cov))

        success = True            
    except:    
        pass
    
    j, lam, _, _ = MaxwellModes(np.log(lam), t, Jexp, wexp, isLiquid)   # Get g_i, lami

    #
    # if mode has dropped out, then need to delete corresponding dlam mode
    #
    if estimateError and success:
        if len(lam) < len(lam0):        
            nkill = 0
            for i in range(len(lam0)):
                if np.min(np.abs(lam0[i] - lam)) > 1e-12 * lam0[i]:
                    dlam = np.delete(dlam, i-nkill)
                    nkill += 1
        return j, lam, dlam
    elif estimateError:
        return j, lam, -1*np.ones(len(lam))
    else:
        return j, lam



def getDiscSpecMagic(par):
    """
    # Function: getDiscSpecMagic(par)
    #
    # Uses the continuous relaxation spectrum extracted using getContSpec()
    # to determine an approximate discrete approximation.
    #
    # Input: Communicated by the datastructure "par"
    #
    # Output: Nopt    = optimum number of discrete modes
    #         [j lam] = spectrum
    #         error   = error norm of the discrete fit
    #        
    #         dmodes.dat : Prints the [g lam] for the particular Nopt
    #         aic.dat    : [N error aic]
    #         Jfitd.dat  : The discrete J(t) for Nopt [t Jt]"""

    texp, Jexp, wexp, s, Lplus, Nv, Gc, Cerror = initializeDiscSpec(par)
                
    n    = len(texp);
    ns   = len(s);
    npts = len(Nv)

    # range of wtBaseDist scanned
    wtBase = par['deltaBaseWeightDist'] * np.arange(1, 1./par['deltaBaseWeightDist'])
    AICbst = np.zeros(len(wtBase))
    Nbst   = np.zeros(len(wtBase))
    nzNbst = np.zeros(len(wtBase))  # number of nonzeros
    
    # main loop over wtBaseDist
    for ib, wb in enumerate(wtBase):
                
        # Find the distribution of nodes you need
        wt  = GetWeights(Lplus[:ns], texp, s, wb)

        # Scan the range of number of Maxwell modes N = (Nmin, Nmax) 
        ev    = np.zeros(npts)
        nzNv  = np.zeros(npts)  # number of nonzero modes 

        for i, N in enumerate(Nv):

            z, hz  = GridDensity(np.log(s), wt, N)     # Select "lam" Points
            
            j, lam, ev[i], _ = MaxwellModes(z, texp, Jexp, wexp, par['liquid'])
            nzNv[i]          = len(j)

            
        # store the best solution for this particular wb

        AIC        = 2. * Nv + 2. * Cerror * ev

        #
        # Fine-Tune the best in class-fit further by trying an NLLS optimization on it.
        #        
        N      = Nv[np.argmin(AIC)]

        AICbst[ib] = min(AIC)
        Nbst[ib]   = Nv[np.argmin(AIC)]
        nzNbst[ib] = nzNv[np.argmin(AIC)]

    
    # global best settings of wb and Nopt; note this is nominal Nopt (!= len(g) due to NNLS)
    Nopt  = int(Nbst[np.argmin(AICbst)])
    wbopt = wtBase[np.argmin(AICbst)]

    #
    # Recompute the best data-set stats, and fine tune it
    #
    wt     = GetWeights(Lplus[:ns], texp, s, wbopt)    
    z, hz  = GridDensity(np.log(s), wt, Nopt)           # Select "lam" Points
    
    j, lam, error, cKp = MaxwellModes(z, texp, Jexp, wexp, par['liquid'])   # Get j_i, lami
    j, lam, dlam = FineTuneSolution(lam, texp, Jexp, wexp, par['liquid'], estimateError=True)

    #
    # Check if modes are close enough to merge
    #
    if len(lam) > 1:
        indx       = np.argsort(lam)
        lam        = lam[indx]
        lamSpacing = lam[1:]/lam[:-1]
        itry       = 0

        j[:len(lam)] = j[indx]

        while min(lamSpacing) < par['minLamSpacing'] and itry < 3:
            print("\tlam Spacing < minLamSpacing")

            imode   = np.argmin(lamSpacing)      # merge modes imode and imode + 1    
            lam     = mergeModes_magic(j, lam, imode)

            j, lam, dlam  = FineTuneSolution(lam, texp, Jexp, wexp, par['liquid'], estimateError=True)

            lamSpacing = lam[1:]/lam[:-1]
            itry      += 1

    if par['liquid']:
       Je = j[-2]; invEta0 = j[-1]
    else:
        Je = j[-1];
    j  = j[:len(lam)]


    if par['verbose']:
        print('(*) Number of optimum nodes = {0:d}'.format(len(j)))

    #
    # Some Plotting
    #
    if par['plotting']:
        plt.clf()
        plt.plot(wtBase, AICbst, label='AIC')
        plt.plot(wtBase, nzNbst, label='Nbst')
        #~ plt.scatter(wbopt, len(g), color='k')
        plt.axvline(x=wbopt, color='gray')
        plt.yscale('log')
        plt.xlabel('baseDistWt')
        plt.legend()
        plt.tight_layout()
        plt.savefig('output/AIC.pdf')        


        plt.clf()
        plt.loglog(lam, j,'o-', label='disc')
        plt.loglog(s, np.exp(Lplus[:ns]), label='cont')
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$j$')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig('output/dmodes.pdf')        


        plt.clf()
        S, T    = np.meshgrid(lam, texp)
        K       = -np.exp(-T/S)        # n * nmodes            
        JtM     = Je + np.dot(K, j) 
        
        if par['liquid']:
            JtM += texp * invEta0


        plt.loglog(texp, Jexp,'o')
        plt.loglog(texp, JtM, label='disc')    

        plt.loglog(texp, Gc, '--', label='cont')
        plt.xlabel(r'$t$')
        plt.ylabel(r'$J(t)$')
        plt.legend()
        plt.tight_layout()
        plt.savefig('output/Jfitd.pdf')
  
    #
    # Some Printing
    #

    if par['verbose']:

        print('(*) log10(Condition number) of matrix equation: {0:.2f}'.format(np.log10(cKp)))

        if par['liquid']:
            print('(*) Je     : {0:.8e}'.format(Je))
            print('(*) invEta0: {0:.8e}'.format(invEta0))
            np.savetxt('output/dmodes.dat', np.c_[j, lam, dlam], fmt='%e', 
                        header='Je, invEta0 = {0:0.8e}\t{1:0.8e}'.format(Je, invEta0))
        else:
            np.savetxt('output/dmodes.dat', np.c_[j, lam, dlam], fmt='%e',
                        header='Je = {0:0.8e}'.format(Je))

        print('\n\t\tModes\n\t\t-----\n\n')
        print('  i \t    j(i) \t    lam(i)\t    dlam(i)\n')
        print('-----------------------------------------------------\n')
        
        for i in range(len(j)):
            print('{0:3d} \t {1:.5e} \t {2:.5e} \t {3:.5e}'.format(i+1, j[i], lam[i], dlam[i]))
        print("\n")

        np.savetxt('output/aic.dat', np.c_[wtBase, nzNbst, AICbst], fmt='%f\t%i\t%e')

        S, T    = np.meshgrid(lam, texp)
        K       = -np.exp(-T/S)        # n * nmodes            
        JtM     = Je + np.dot(K, j) 
        
        if par['liquid']:
            JtM += texp * invEta0
        
        np.savetxt('output/Jfitd.dat', np.c_[texp, JtM], fmt='%e')


    return Nopt, j, lam, error

#############################
#
# M A I N  P R O G R A M
#
#############################

if __name__ == '__main__':
    #
    # Read input parameters from file "inp.dat"
    #
    par = readInput('inp.dat')
    _ = getDiscSpecMagic(par)    
