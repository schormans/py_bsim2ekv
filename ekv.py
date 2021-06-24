#Module implementation of EKV_v2.6, in accordance with doc from EPFL site from 1999.

#######################################################################
#
#
#  27-Jan-2021: Model is only partially implemented:
#  Focus is on DC currents only at the moment
#  AC features, noise, mismatch etc aren't implemented.
#  Some parameters relating to these features may be here, but they
#  are likely unused.
#
#  18-Feb-2021: Adding option for modified LETA and WETA
#  This modification replaces LETA and WETA with functions of Leff
#  and Weff respectively if self.LETA_WETA_MOD == True.
#  This allows a better fit if converting from BSIM.
#
#  08-Apr-2021: Implemented AC, noise etc. Can now generate a full
#  parameter set using the BSIM→EKV conversion process from
#  Stefanovic et al.
#
#
#######################################################################

#depends on mpmath for precision (and some dirty tricks).

import mpmath as mp

    #first set model parameters; following ekv spec

    #setting defaults to some XH018 min values so the model does /something/ out of the box

class MOS:
    import mpmath as mp

    def __init__(self, params):

        #######################################################################
        #
        # all parameters are defined here for a MOS object. first all the
        # defaults are set, then 'params' is read and implemented.
        # Any param can be overwritten by inclusion in the params dictionary.
        # Params dictionary entries should be of format 'name' : float, e.g.
        #     params = { 'W': 0.5e-6, 'L' : 1e-6, 'M' : 1 }.
        # These values will be converted to mpmath floats when initialized.
        #
        # Anything not given in params will use the defaults from below
        #
        #######################################################################

        #######################################################################
        #device input variables
        #######################################################################

        self.L = mp.mpf('0.18e-6') #device length (m)
        self.W = mp.mpf('0.22e-6') #device width (m)
        self.M = mp.mpf('1') #parallel multiplier
        self.N = mp.mpf('1') #series multiplier

        #######################################################################
        #basic process parameters
        #######################################################################

        #self.COX = mp.mpf('8.6328e-3') #we give a value here, but ideally should calc from TOX (F/m²)
        self.XJ = mp.mpf('150e-9') #junction depth, must be >= 1nm (m)
        self.DW = mp.mpf('0') #channel width correction to get WEFF (normally neg value) (m)
        self.DL = mp.mpf('0') #channel length correction to get LEFF (normally neg value) (m)

        #######################################################################
        #basic model parameters
        #######################################################################

        self.VTO = mp.mpf('595e-3') #long-channel bulk-referenced threshold voltage (V)
        #self.GAMMA = mp.mpf('0.16348') #body effect parameter (sqrt(V))
        #self.PHI = mp.mpf('0.8894') #2x bulk fermi potential (V)
        #self.KP = mp.mpf('141.48e-6') #transconductance parameter (A/V²)
        #self.E0 = mp.mpf('1e12') #mobility reduction coefficient: MUST BE CALC'D! (V/m)
        #self.UCRIT = mp.mpf('4.4535e6') #longitudinal critial field (V/m)

        #######################################################################
        #optional parameters
        #######################################################################

        self.TOX = mp.mpf('4e-9') #oxide thickness (physical rather than effective?) (m)
        self.NSUB = mp.mpf('60e15') #channel doping (1/cm³) NOT SI!
        #self.VFB = mp.mpf('-1.0') #flat-band voltage (V)
        #U0 BSIM is SI! XH018 nmos: 12.74e-3 m²/V·s
        self.U0 = mp.mpf('127.4') #low-field mobility (cm²/V·s) NOT SI!
        self.VMAX = mp.mpf('93969.47') #saturation velocity (m/s)
        #self.THETA = mp.mpf('0.28') #vertical field mobility reduction coefficient: MUST BE CALC'D (1/V)

        #######################################################################
        #channel length modulation and charge sharing parameters
        #######################################################################

        self.LAMBDA = mp.mpf('0.5') #channel length modulation depletion length coefficient: MUST BE CALC'D
        self.WETA = mp.mpf('0.25') #narrow-channel effect coefficient: MUST BE CALC'D
        self.LETA = mp.mpf('0.1') #narrow-channel effect coefficient: MUST BE CALC'D

        #######################################################################
        #reverse short channel effects
        #######################################################################

        self.Q0 = mp.mpf('0') #reverse short-channel effect peak charge density: MUST BE CALC'D (A·s/m²)
        self.LK = mp.mpf('0.29e-6') #reverse short-channel effect characteristic length: MUST BE CALC'D (m)

        #######################################################################
        #impact ionization related parameters
        #######################################################################

        self.IBA = mp.mpf('0') #first impact ionization coefficient: MUST BE CALC'D (1/m)
        self.IBB = mp.mpf('3.0e8') #second impact ionization coefficient: MUST BE CALC'D (V/m)
        self.IBN = mp.mpf('1.0') #saturation voltage factor for impact ionization: MUST BE CALC'D


        #######################################################################
        #intrinsic model temperature params
        #######################################################################

        self.TCV = mp.mpf('1e-3') #threshold voltage temperature coefficient: MUST BE CALC'D (V/K)
        self.BEX = mp.mpf('-1.5') #mobility temperature exponent: MUST BE CALC'D? UTE in BSIM?
        self.UCEX = mp.mpf('0.8') #longitudinal critical field temperature exponent: MUST BE CALC'D
        self.IBBT = mp.mpf('9.0e-4') #temperature coefficient for IBB: MUST BE CALC'D (1/K)

        #######################################################################
        #matching parameters
        #######################################################################

        self.AVTO = mp.mpf('0') #area related threshold mismatch parameter (V·m)
        self.AKP = mp.mpf('0') #area related gain mismatch parameter (m)
        self.AGAMMA = mp.mpf('0') #area related body effect mismatch parameter (sqrt(V)·m)


        #######################################################################
        #skipped param sections, params listed to check later
        #######################################################################

        #flicker noise parameters

        self.KF = 0
        self.AF = 1

        #setup parameters

        #skip for now: NQS, SATLIM, XQC

        #This should either be 0 or 1, flag to enable/disable Non-Quasi-Static AC
        
        self.NQS = 1 

        #######################################################################


        #######################################################################
        #static intrinsic model equations
        #######################################################################

        self.epsi = mp.mpf('104.5e-12') #permittivity of silicon (rel perm. approx 11.7) (F/m)
        self.epsiox = mp.mpf('34.5e-12') #permittivity of silicon dioxide (rel perm. approx 3.9) (F/m)
        self.q = mp.mpf('1.602176634e-19') #electron charge (C)
        self.k = mp.mpf('1.380649e-23') #boltzmann constant (J/K)
        self.Tref = mp.mpf('300.15') #reference temperature (K)
        self.Tnom = mp.mpf('300') #nominal temperature of model parameters (K)
        self.T = mp.mpf('300') #model operation temperature (K)

        #######################################################################
        #NMOS or PMOS
        #######################################################################

        self.polarity = 'n' #'n' for nmos 'p' for pmos


        self.q_OX = 0 #trapped oxide charge? optional, used in self.nodeCharges



        self.LETA_WETA_MOD = False #by default use normal LETA and WETA defs

        self.useFapprox = False #by default use the exact interpolation function


        self.nudge = mp.mpf('1e-48') #small number for calculating derivs

        self.thermfudge = mp.mpf('1.0') #fudge factor to control thermal noise
        #ideally shouldn't need this, but currently only way to control thermal noise
        #result is by re-adjusting beta? possibly just model limitation.
        
        #######################################################################
        #process params from input dictionary
        #######################################################################

        for k, v in params.items():
            if isinstance(v, bool):
                setattr(self, k, v)
            elif k=='polarity':
                setattr(self, k, v)
            else:
                setattr(self, k, mp.mpf(v))
                

        #######################################################################
        #derived parameters
        # these can be overridden by supplying them directly in params
        #######################################################################

        #if any of the following parameters haven't been given explicitly, derive them

        if not(hasattr(self, 'COX')):
            self.COX = self.epsiox/self.TOX

        if not(hasattr(self, 'GAMMA')):
            self.GAMMA = (mp.sqrt(2*self.q*self.epsi*(self.NSUB*mp.mpf('1e6'))))/self.COX

        if not(hasattr(self, 'PHI')):
            self.PHI = 2*self.Vt(self.Tnom)*mp.ln((self.NSUB*mp.mpf('1e6'))/(self.n_i(self.Tnom)))

        if not(hasattr(self, 'VTO')):
            if hasattr(self, 'VFB'):
                self.VTO = self.VFB + self.PHI + self.GAMMA*mp.sqrt(self.PHI)
            else:
                self.VTO = mp.mpf('0.5')

        if not(hasattr(self, 'KP')):
            if self.U0 > 0:
                self.KP = self.U0*mp.mpf('1e-4')*self.COX
            else:
                self.KP = mp.mpf('50e-6')

        if not(hasattr(self, 'UCRIT')):
            if (self.VMAX > 0 and self.U0 > 0):
                self.UCRIT = self.VMAX/(self.U0*mp.mpf('1e-4'))
            else:
                self.UCRIT = mp.mpf('2e6')

        if not(hasattr(self, 'E0')):
            if hasattr(self, 'THETA'):
                self.E0 = mp.mpf('0')
            else:
                #self.E0 = 0.1/(0.4*self.TOX)
                #line above is actually lower bound
                self.E0 = mp.mpf('1e12')
                
        #if mos is pmos, reverse the following param values
        # if self.polarity == 'p':
        #     if hasattr(self, 'VFB'):
        #         self.VFB = -self.VFB
        #     if hasattr(self, 'VTO'):
        #         self.VTO = -self.VTO
        #     if hasattr(self, 'TCV'):
        #         self.TCV = -self.TCV
        # THIS IS NOW HANDLED IN CURRENT EQUATIONS

        #########################################################
        # apply temperature dependences
        #########################################################

        #THESE ARE REDEF'D AS FUNCTIONS OF TEMPERATURE OUTSIDE OF INIT NOW
        #THIS IS DONE TO MAKE INIT SANE FOR THE CASE OF UPDATING VARS
        #BY CALLING RE-INIT. THIS MUST BE DONE FOR ANY ATTR THAT
        #UPDATES ITSELF IN THE INIT PROCEDURE
     
        # self.VTO = self.VTO - self.TCV*(self.T-self.Tnom)

        # self.KP = self.KP*(self.T/self.Tnom)**self.BEX

        # self.UCRIT = self.UCRIT*(self.T/self.Tnom)**self.UCEX

        # self.PHI = self.PHI*(self.T/self.Tnom) - 3*self.Vt(self.T)*mp.ln(self.T/self.Tnom) - self.Eg(self.Tnom)*(self.T/self.Tnom) + self.Eg(self.T)

        # self.IBB = self.IBB*(1.0 + self.IBBT*(self.T-self.Tnom))


        #########################################################
        # set Weff and Leff
        #########################################################

        # self.Weff = self.W + self.DW
        # self.Leff = self.L + self.DL

        #DEF'D AS FUNCTIONS

        

        #########################################################
        # At this point in the model, matching outputs are defined
        # We're doing things a little differently, definitions below
        # to return sigmaVTO, sigmaKP, sigmaGAMMA, these can be
        # passed out and used to generate stats
        #########################################################

        # self.sigmaVTO = self.AVTO/(mp.sqrt(self.M*self.Weff()*self.N*self.Leff())) # (Volts)
        # self.sigmaKP = self.AKP/(mp.sqrt(self.M*self.Weff()*self.N*self.Leff())) # (unitless)
        # self.sigmaGAMMA = self.AGAMMA/(mp.sqrt(self.M*self.Weff()*self.N*self.Leff())) # (sqrt(V))

        

        """
        Reverse short channel effect (RSCE)
        """

        # self.Cep = 4*mp.mpf('22e-3')**2
        # self.CA = 0.028
        # self.xi = self.CA*(10*(self.Leff())/(self.LK)-1)
        # self.deltaV_RSCE = ((2*self.Q0)/self.COX)*(1/(1+0.5*(self.xi + mp.sqrt(self.xi**2 + self.Cep)))**2)

        #this leads to function definitions further on

        #self.VC = self.UCRITt()*self.N*self.Leff() #velocity saturation voltage

        #self.LC = mp.sqrt((self.epsi/self.COX)*self.XJ)
        #self.Lmin = self.N*self.Leff()/10

    def Lmin(self):
        return self.N*self.Leff()/10

    def VC(self):
        return self.UCRITt()*self.N*self.Leff() #velocity saturation voltage

    def LC(self):
        return mp.sqrt((self.epsi/self.COX)*self.XJ)

    def deltaV_RSCE(self):
        self.Cep = 4*mp.mpf('22e-3')**2
        self.CA = 0.028
        self.xi = self.CA*(10*(self.Leff())/(self.LK)-1)
        return ((2*self.Q0)/self.COX)*(1/(1+0.5*(self.xi + mp.sqrt(self.xi**2 + self.Cep)))**2)

    def sigmaVTO(self):
        return self.AVTO/(mp.sqrt(self.M*self.Weff()*self.N*self.Leff())) # (Volts)

    def sigmaKP(self):
        return self.AKP/(mp.sqrt(self.M*self.Weff()*self.N*self.Leff())) # (unitless)

    def sigmaGAMMA(self):
        return self.AGAMMA/(mp.sqrt(self.M*self.Weff()*self.N*self.Leff())) # (sqrt(V))

    def Weff(self):
        return self.W + self.DW

    def Leff(self):
        return self.L + self.DL

    def VTOt(self): #VTO as a function of Temperature
        return self.VTO - self.TCV*(self.T-self.Tnom)

    def KPt(self):
        return self.KP*(self.T/self.Tnom)**self.BEX

    def UCRITt(self):
        return self.UCRIT*(self.T/self.Tnom)**self.UCEX

    def PHIt(self):
        return self.PHI*(self.T/self.Tnom) - 3*self.Vt(self.T)*mp.ln(self.T/self.Tnom) - self.Eg(self.Tnom)*(self.T/self.Tnom) + self.Eg(self.T)

    def IBBt(self):
        return self.IBB*(1.0 + self.IBBT*(self.T-self.Tnom))

    def Vt(self,T): #thermal voltage (V)
        return (self.k*T)/self.q

    def Eg(self,T): #energy gap (eV)
        return (1.16 - 0.000702*((T**2)/(T+1108)))

    def n_i(self,T): #intrinsic carrier conc (1/m³)
        return mp.mpf('1.45e16')*(T/self.Tref)*mp.exp((self.Eg(self.Tref))/(2*self.Vt(self.Tref)) - (self.Eg(T))/(2*self.Vt(T)))

    def Vgeff(self,Vg): #effective gate voltage including RSCE (V'_G Vgprim)
        return Vg - self.VTOt() - self.deltaV_RSCE() + self.PHIt() + self.GAMMA*mp.sqrt(self.PHIt())

    def VPO(self,Vg): #pinch-off voltage for narrow-channel effect for given Vg
        if self.Vgeff(Vg) > 0:
            return self.Vgeff(Vg) - self.PHIt() - self.GAMMA*(mp.sqrt(self.Vgeff(Vg) + (self.GAMMA/2)**2) - self.GAMMA/2)
        else:
            return -self.PHIt()

    def V_S_OR_D_EFF(self,Vsord): #see eq 35 in the spec (SETUP FOR GAMMAEFF)
        #seems to give a tweaked 'effective' V_S or V_D, accounting for charge-sharing
        return 0.5*(Vsord + self.PHIt() + mp.sqrt((Vsord + self.PHIt())**2 + (4*self.Vt(self.T))**2))

    def GAMMAO(self,Vs,Vd,Vg): #more setup for GAMMAEFF
        if self.LETA_WETA_MOD:
            leta = self.LETA*self.Leff()
            weta = self.WETA*self.Weff()
        else:
            leta = self.LETA
            weta = self.WETA
        return self.GAMMA - (self.epsi/self.COX)*((leta/self.Leff())*(mp.sqrt(self.V_S_OR_D_EFF(Vs)) + mp.sqrt(self.V_S_OR_D_EFF(Vd))) - (3*weta/self.Weff())*mp.sqrt(self.VPO(Vg) + self.PHIt()))

    def GAMMAEFF(self,Vs,Vd,Vg): #effective substrate-factor accounting for charge-sharing
        return 0.5*(self.GAMMAO(Vs,Vd,Vg) + mp.sqrt(self.GAMMAO(Vs,Vd,Vg)**2 + 0.1*self.Vt(self.T)))

    def VP(self,Vs,Vd,Vg): #pinch off voltage including short and narrow channel effects
        if self.Vgeff(Vg) > 0:
            return self.Vgeff(Vg) - self.PHIt() - self.GAMMAEFF(Vs,Vd,Vg)*(mp.sqrt(self.Vgeff(Vg) + (self.GAMMAEFF(Vs,Vd,Vg)/2)**2) - self.GAMMAEFF(Vs,Vd,Vg)/2)
        else:
            return -self.PHIt()

    def n(self,Vs,Vd,Vg):
        return 1 + self.GAMMA/(2*mp.sqrt(self.VP(Vs,Vd,Vg) + self.PHIt() + 4*self.Vt(self.T)))

    def sigmoid(self,x): #sigmoid function (basis for the shaping functions)
        return (1/(1+mp.exp(-x)))

    #approximate large signal interpolation function based on sigmoid shaping
    #CAN USE THIS FOR CALCULUS ETC? IS CONTINUOUS AND DIFFERENTIABLE ETC
    #GETS WOBBLY IN MODERATE INVERSION!
    def Fapprox(self,v):
        x = [
            2.6056285814204454,
            1.1161843683229469,
            2.301551756132304,
            1.6801113670707852,
            3.068042874834098,
            0.08871864817161682,
            5.159441803685059,
            0.11274480638341763,
            0.9913895848125953,
            0.13483044481449474,
            2.5392530031622393e-11,
            0.09984279260396758
        ]
        def shaping1(v): #first shaping function for shaping Fapprox
            #return (1 - 1/(1+mp.exp(3.0067-v*1.5)))
            return 1 - self.sigmoid(v*x[1] - x[0])

        def shaping2(v): #second shaping function for shaping Fapprox
            #return (1/(1+np.exp(2.2201-v*2)))
            return self.sigmoid(v*x[3] - x[2])

        def shaping3(v):
            return (self.sigmoid((v-x[4])**2)**x[5])*(self.sigmoid((v-x[6])**2)**x[7])*(1/(self.sigmoid((v+x[8])**2)**x[9]))*(1/(self.sigmoid(x[10]*(v-50)**6)**x[11]))
        return (mp.log(1+mp.exp(v))*shaping1(v) + ((v/2)**2)*shaping2(v))/shaping3(v)

    def F(self,v): #exact large signal interpolation function; uses root-finding
        def vfunc(i): #v as a function of i
            mp.mp.prec = 500 #this precision is overkill, but we need good precision here
            #TODO: tune this precision to be less overkill but still go!
            y = mp.sqrt(0.25 + i) - 0.5
            return 2*y + mp.ln(y)
        if isinstance(v, mp.mpc):
            print(f"Warning: v value in mos.F is complex: {v}, taking the real part.")
            print("If the complex part is significant here, do something about it.")
            v = v.real
        if v > 1:
            x0 = [(v/4)**2,v**2]
        else:
            x0 = [(1-1e-6)*mp.exp(v),(1+1e-6)*mp.exp(v)]
        return mp.findroot(lambda i: vfunc(i) - v,x0, method='brent',tol=1e-12)

    def i_f(self,VP,VS): #forward normalized current
        if self.useFapprox:
            return self.Fapprox((VP - VS)/self.Vt(self.T))
        else:
            return self.F((VP - VS)/self.Vt(self.T))

    def VDSS(self,i_f): #Drain source vel-sat voltage? 0.5x actual saturation voltage
        return self.VC()*(mp.sqrt(0.25 + (self.Vt(self.T)/self.VC())*mp.sqrt(i_f)) - 0.5)

    def VDSS_prim(self,i_f): #drain to source saturation voltage for reverse normalized current
        A = 0.25 + (self.Vt(self.T)/self.VC())*(mp.sqrt(i_f) - 0.75*mp.log(i_f))
        B = mp.log(self.VC()/(2*self.Vt(self.T))) - 0.6
        return self.VC()*(mp.sqrt(A) - 0.5) + self.Vt(self.T)*B

    #Channel length modulation functions
    def deltaV(self,i_f): #deltaV from an i_f
        return 4*self.Vt(self.T)*mp.sqrt(self.LAMBDA*(mp.sqrt(i_f)-self.VDSS(i_f)/self.Vt(self.T)) + 1/64)

    def Vds(self,VD,VS): #vds for clm, for some reason divided by 2??
        return (VD - VS)/2

    def Vip(self,i_f,VD,VS):
        return mp.sqrt(self.VDSS(i_f)**2 + self.deltaV(i_f)**2) - mp.sqrt((self.Vds(VD,VS) - self.VDSS(i_f))**2 + self.deltaV(i_f)**2)

    def deltaL(self,i_f,VD,VS):
        return self.LAMBDA*self.LC()*mp.log(1 + (self.Vds(VD,VS) - self.Vip(i_f,VD,VS))/(self.LC()*self.UCRITt()))
        
    def Lprim(self,i_f,VD,VS):
        return self.N*self.Leff() - self.deltaL(i_f,VD,VS) + (self.Vds(VD,VS) - self.Vip(i_f,VD,VS))/self.UCRITt()

    
    def Leq(self,i_f,VD,VS):
        return 0.5*(self.Lprim(i_f,VD,VS) + mp.sqrt(self.Lprim(i_f,VD,VS)**2 + self.Lmin()**2))

    def i_rprim(self,VS,VD,VG): #reverse normalized current
        VP = self.VP(VS,VD,VG)
        i_f = self.i_f(VP,VS)
        A = VP - self.Vds(VD,VS) - VS - mp.sqrt(self.VDSS_prim(i_f)**2 + self.deltaV(i_f)**2) + mp.sqrt((self.Vds(VD,VS) - self.VDSS_prim(i_f))**2 + self.deltaV(i_f)**2)
        frac = A/self.Vt(self.T)
        if self.useFapprox:
            return self.Fapprox(frac)
        else:
            return self.F(frac)

    def i_r(self,VS,VD,VG): #reverse normalized current for mobility model, charge/cap, thermal etc
        VP = self.VP(VS,VD,VG)
        Vt = self.Vt(self.T)
        if self.useFapprox:
            return self.Fapprox((VP-VD)/Vt)
        else:
            return self.F((VP-VD)/Vt)

    def beta0(self,VS,VD,VG): #transconductance factor and VFMR
        i_f = self.i_f(self.VP(VS,VD,VG),VS)
        return self.KPt()*((self.M*self.Weff())/(self.Leq(i_f,VD,VS)))

    def beta(self,VS,VD,VG):
        i_f = self.i_f(self.VP(VS,VD,VG),VS)
        eta = 0.5 #default eta for nmos
        if (self.polarity == 'p'):
            eta = 1/3 #if mos is pmos, shift eta
        qB0 = self.GAMMA*mp.sqrt(self.PHIt())
        beta0prim = self.beta0(VS,VD,VG)*(1 + ((self.COX)/(self.E0*self.epsi))*qB0)
        qB = self.nodeCharges(VS,VD,VG)[1]['q_B']
        qI = self.nodeCharges(VS,VD,VG)[1]['q_I']
        return beta0prim/(1 + (self.COX/(self.E0*self.epsi))*self.Vt(self.T)*abs(qB + eta*qI))

    def nodeCharges(self,VS,VD,VG): #returns dict of normalized intrinsic node charges
        #q_OX is assumed to be zero, but you can set a value at initialization if desired
        #q_OX can be used to model trapped oxide charges??
        VP = self.VP(VS,VD,VG)
        i_f = self.i_f(VP,VS)
        i_r = self.i_r(VS,VD,VG)
        n_q = 1 + (self.GAMMA/(2*mp.sqrt(VP + self.PHIt() + 1e-6)))
        x_f = mp.sqrt(0.25 + i_f)
        x_r = mp.sqrt(0.25 + i_r)
        q_D = -n_q*((4/15)*((3*x_r**3 + 6*(x_r**2)*x_f + 4*x_r*x_f**2 + 2*x_f**3)/((x_f + x_r)**2)) - 0.5)
        q_S = -n_q*((4/15)*((3*x_f**3 + 6*(x_f**2)*x_r + 4*x_f*x_r**2 + 2*x_r**3)/((x_f + x_r)**2)) - 0.5)
        q_I = q_S + q_D
        if self.Vgeff(VG) > 0:
            q_B = (-self.GAMMA*mp.sqrt(VP + self.PHIt() + 1e-6))/self.Vt(self.T) - ((n_q - 1)/n_q)*q_I
        else:
            q_B = -self.Vgeff(VG)/self.Vt(self.T)

        q_G = -q_I - self.q_OX - q_B

        #return two dictionaries, normcharges and abscharges
        #normcharges are q_(I,B,D,S,G), abscharges are Q_(I,B,D,S,G)

        C_ox = self.COX*self.M*self.Weff()*self.N*self.Leff()
        normcharges = {
            'q_I' : q_I,
            'q_B' : q_B,
            'q_D' : q_D,
            'q_S' : q_S,
            'q_G' : q_G
        }
        abscharges = {}
        for item in normcharges.items(): #convert to dict of absolute charges
            #names will be uppercased: e.g. 'q_I' → 'Q_I'
            abscharges[item[0].upper()] = item[1]*C_ox*self.Vt(self.T)

        return abscharges, normcharges

    def I_S(self,VS,VD,VG):
        return 2*self.n(VS,VD,VG)*self.beta(VS,VD,VG)*self.Vt(self.T)**2

    def I_DS(self,VS,VD,VG):
        # according to ekv spec pmos is 'pseudo-nmos'. So voltages have to be flipped here
        # DON'T FLIP HERE, DO IT BEFORE. FLIPPING INSIDE OBJECT IS TOO MESSY
        # WOULD NEED TO BE FLIPPED FOR ALMOST EVERY FUNCTION!!
        # ONLY FLIP OUTPUT CURRENT
        # if (self.polarity == 'p'):
        #     oldvals = (self.VTO, self.TCV) # have to reflip these before returning
        #     self.VTO, self.TCV = (-self.VTO, -self.TCV)
        #     VS,VD,VG = (-VS,-VD,-VG)
        VP = self.VP(VS,VD,VG)
        outcurrent = self.I_S(VS,VD,VG)*(self.i_f(VP,VS) - self.i_rprim(VS,VD,VG))
        if (self.polarity == 'p'):
            outcurrent = -outcurrent
            #self.VTO, self.TCV = oldvals
        return outcurrent

    def V_ib(self,VS,VD,VG):
        return VD - VS - self.IBN*2*self.VDSS(self.i_f(self.VP(VS,VD,VG),VS))

    def I_DB(self,VS,VD,VG):
        Vib = self.V_ib(VS,VD,VG)
        if Vib > 0:
            return self.I_DS(VS,VD,VG)*(self.IBA/self.IBBt())*Vib*mp.exp((-self.IBBt()*self.LC())/Vib)
        else:
            return 0

    def num_diff_dirty(self, func, var, nudge):
        #returns simple partial diff of func w.r.t. var
        #func should be such that its only variable is var
        #nudge is the amount that var is nudged by:
        #gradient is calculated between input bounds of
        #func(var-nudge) and func(var+nudge)
        #ASSUMES MPMATH IMPORTED AS 'mp'
        #by setting precision super high, we avoid rounding
        #err as long as nudge is v small.
        mp.mp.prec = 500
        if nudge < 0:
            raise Exception('error! : nudge must be > 0')

        val = (func(var+nudge) - func(var-nudge))/(2*nudge)
        mp.mp.prec = 53
        return val
        
    def gm(self,VS,VD,VG):
        return self.num_diff_dirty(lambda VG: self.I_DS(VS,VD,VG),VG,self.nudge)

    def gmg(self,VS,VD,VG):
        return self.gm(VS,VD,VG)

    def gms(self,VS,VD,VG):
        return self.num_diff_dirty(lambda VS: self.I_DS(VS,VD,VG),VS,self.nudge)

    def gmd(self,VS,VD,VG):
        return self.num_diff_dirty(lambda VDS: self.I_DS((VD-VDS),(VDS+VS),VG),(VD-VS),self.nudge)

    def gmbs(self,VS,VD,VG):
        return self.gms(VS,VD,VG) - self.gmg(VS,VD,VG) - self.gmd(VS,VD,VG)

    def gds(self,VS,VD,VG):
        return -self.gmd(VS,VD,VG)

    def x_f(self,VS,VD,VG):
        return mp.sqrt(0.25 + self.i_f(self.VP(VS,VD,VG),VS))

    def x_r(self,VS,VD,VG):
        return mp.sqrt(0.25 + self.i_r(VS,VD,VG))

    def transCaps(self,VS,VD,VG):
        #abscharges, normcharges = self.nodeCharges(VS,VD,VG)
        #DEFINE SUBFUNC HERE SO I HAVE A FUNC THAT CAN BE USED FOR Qx(Vy)
        def absQ(mos,VS,VD,VG,Qnode):
            abscharges, normcharges = mos.nodeCharges(VS,VD,VG)
            return abscharges['Q_' + Qnode]
        #return transcapacitances as a dict of dicts
        #Cxy, requires Qx and Vy.
        #declare VB manually here for sake of lambdas later... only way round it?
        VB = mp.mpf('0.0')
        capdict = {}
        nodes = ['G','D','S','B']
        lambdas = {
            'G' : lambda VG: absQ(self,VS,VD,VG,nodeq),
            'D' : lambda VD: absQ(self,VS,VD,VG,nodeq),
            'S' : lambda VS: absQ(self,VS,VD,VG,nodeq),
            'B' : lambda VB: absQ(self,VS-VB,VD-VB,VG-VB,nodeq)
        }
        #qfuncs = {}
        for nodeq in nodes:
            capdict[nodeq] = {}
            #qfuncs[nodeq] = {}
            for nodev in nodes:
                #qfuncs[nodeq][nodev] = lambdas[nodev]
                #capdict[nodeq][nodev] = self.num_diff_dirty(qfuncs[nodeq][nodev],nodev,self.nudge)
                capdict[nodeq][nodev] = eval(f'self.num_diff_dirty(lambdas[nodev],V{nodev},self.nudge)')
        return capdict

    def tau0(self,VS,VD,VG):
        return self.COX/(2*self.Vt(self.T)*self.beta(VS,VD,VG))

    def tau(self,VS,VD,VG):
        x_f = self.x_f(VS,VD,VG)
        x_r = self.x_r(VS,VD,VG)
        return self.tau0(VS,VD,VG)*(4/15)*((x_f**2 + 3*x_f*x_r + x_r**2)/((x_f + x_r)**3))

    def I_DS_AC(self,VS,VD,VG,s):
        #s is laplace domain frequency
        return self.I_DS(VS,VD,VG)/(1 + self.NQS*s*self.tau(VS,VD,VG))

    #admittances

    def Ymg(self,VS,VD,VG,s):
        return self.gmg(VS,VD,VG)/(1 + self.NQS*s*self.tau(VS,VD,VG))

    def Yms(self,VS,VD,VG,s):
        return self.gms(VS,VD,VG)/(1 + self.NQS*s*self.tau(VS,VD,VG))

    def Ymd(self,VS,VD,VG,s):
        return self.gmd(VS,VD,VG)/(1 + self.NQS*s*self.tau(VS,VD,VG))

    def Ymbs(self,VS,VD,VG,s):
        return self.Yms(VS,VD,VG,s) - self.Ymg(VS,VD,VG,s) - self.Ymd(VS,VD,VG,s)

    #intrinsic noise model equations

    def Sthermal(self,VS,VD,VG): #returns PSD
        abscharges, normcharges = self.nodeCharges(VS,VD,VG)
        q_I = normcharges['q_I']
        return self.thermfudge*4*self.k*self.T*self.beta(VS,VD,VG)*abs(q_I)

    def Sflicker(self,VS,VD,VG,f): #returns PSD (f is freq in Hz)
        return (self.KF*self.gmg(VS,VD,VG)**2)/(self.M*self.Weff()*self.N*self.Leff()*self.COX*f**self.AF)
