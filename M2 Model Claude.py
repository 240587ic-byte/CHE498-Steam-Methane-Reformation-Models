"""
================================================================================
 Steam Methane Reforming (SMR) Reactor - Model M2
 1D pseudo-homogeneous packed-bed reactor WITH AXIAL MASS & HEAT DISPERSION
--------------------------------------------------------------------------------
 Source: Cui, C.; Vo, D.-N.; Zhao, Y.; Qi, M.; Xia, M.; Ramkrishna, D.;
         Masuku, C.M. "Rigorous development and comparison of multi-dimensional
         reactor models encompassing the catalyst domains for steam methane
         reforming." Chemical Engineering Journal 496 (2024) 153581.

 GOVERNING EQUATIONS IMPLEMENTED (M2 = M1 + axial mass/heat dispersion):
   Eq.(1)  catalyst effectiveness factor definition (applied as constant eta_j)
   Eq.(3)  bed voidage correlation
   Eq.(4)  ideal-gas partial pressure  P_i = C_i R T   -> feeds kinetics (App. A)
   Eq.(5)  total pressure = sum of partial pressures
   Eq.(6)-(8) molar flow / cross-section / inlet superficial velocity
   Eq.(9)-(10) continuity -> algebraic velocity relation rho_g*Uz = rho_g0*Uz0
   Eq.(11) gas mixture density (ideal gas)
   Eq.(13)-(14) Ergun equation (mechanical pressure-drop, reported diagnostic)
   Eq.(18) SPECIES balance with AXIAL DISPERSION   (M2 material balance)
   Eq.(19) ENERGY  balance with AXIAL CONDUCTION + dispersive heat term (M2)
 Closure relations (used inside Eqs. 18-19 and for kinetics):
   Appendix A  Eq.(A.1)-(A.8), Table A.1   : Xu & Froment (1989) LHHW kinetics
   Appendix B  Eq.(B.11)-(B.14)            : effective axial mass dispersion
               Eq.(B.18)-(B.22), Table B.1 : molecular diffusivities (Blanc's law)
               Eq.(B.24),(B.26),(B.28)-(B.30): effective axial thermal conductivity
               Eq.(B.31)                    : tube-wall overall heat-transfer coef.
   Appendix C  Eq.(C.1)-(C.3), Table C.2   : gas mixture viscosity
               Eq.(C.4)-(C.8), Table C.3/C.4: gas mixture thermal conductivity
               Eq.(C.9)-(C.10), Table C.5  : gas mixture heat capacity
   Table 2     base operating/geometric data

 NUMERICAL METHOD (matches paper's approach, Sec. 3):
   Method of lines: axial derivatives are discretized with finite differences
   (backward 1st-derivative / central 2nd-derivative, as in the paper), and the
   resulting ODE system in TIME is integrated with solve_ivp (BDF, stiff) to
   its steady state ("pseudo-transient continuation") - this reproduces the
   paper's own workaround for the otherwise unsolvable steady PDE (see text
   below Eq. (9)).
================================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags, kron, csr_matrix
import matplotlib
import matplotlib.pyplot as plt

# ============================================================================
# 1. OPERATING / GEOMETRIC DATA  (Table 2)
# ============================================================================
Rg      = 8.314              # J/mol/K
Rg_bar  = 0.08314             # bar.m3/(kmol.K)   (=8314 J/(kmol.K) in bar.m3 units)
Rg_kJ   = 8.314e-3            # kJ/mol/K  (Arrhenius / van't Hoff exponents, App. A)

L       = 12.0                # m,   reactor length                 Table 2
dt_tube = 0.10                 # m,   tube inner diameter            Table 2
dp      = 0.01                 # m,   catalyst particle diameter     Table 2
rho_p   = 2355.2               # kg/m3, catalyst density             Table 2
Cp_p    = 950.0                # J/(kg.K), catalyst heat capacity    Table 2
lam_p   = 0.3489                # W/(m.K), catalyst solid conductivity Table 2
T0      = 793.15                # K,  inlet temperature               Table 2
P0      = 25.69                  # bar, inlet pressure                Table 2
Tw      = 1000.0                 # K,  constant tube-wall T (Assumption 7)
eps_p   = 0.519                   # catalyst pellet porosity  (App. B.3)
tau_p   = 2.74                     # catalyst tortuosity       (App. B.3)

species = ['CH4', 'CO', 'CO2', 'H2', 'H2O', 'N2']
MW = {'CH4':16.0428,'CO':28.0104,'CO2':44.0098,'H2':2.01588,'H2O':18.0153,'N2':28.0135}   # Table C.1

F0_h = {'CH4':5.17,'CO':0.0,'CO2':0.29,'H2':0.63,'H2O':17.35,'N2':0.85}                    # kmol/h, Table 2
F0   = {k: v/3600.0 for k, v in F0_h.items()}                                               # -> kmol/s

# Bed voidage,   Eq.(3)
eps_b = 0.38 + 0.073*(1.0 - (dt_tube/dp - 2.0)**2 / (dt_tube/dp)**2)

# Cross-sectional area,  Eq.(7)
Omega = np.pi*dt_tube**2/4.0

# Inlet superficial velocity,  Eq.(8)
F0_tot = sum(F0.values())
Uz0 = F0_tot*Rg_bar*T0/(Omega*P0)                 # m/s
rho_g0 = sum(F0[s]*MW[s] for s in species)/(Uz0*Omega)   # kg/m3  (mass flux / velocity)
G0 = rho_g0*Uz0        # kg/(m2.s), CONSTANT mass flux (steady 1D flow, no mass source, exact)

# Inlet concentrations,  Eq.(6) rearranged: C_i0 = F_i0/(Uz0*Omega)
C0 = np.array([F0[s]/(Uz0*Omega) for s in species])       # kmol/m3

# Effectiveness factor (constant, pseudo-homogeneous simplification, Eq.1 / Sec.4.1.2)
ETA = 1.0        # try 1, 0.1, 0.01, 0.001 as in the paper (Fig. 3-4, Table 3)

# ============================================================================
# 2. KINETIC PARAMETERS - Xu & Froment (1989), Appendix A, Table A.1
# ============================================================================
Ak  = {1: 4.225e15/3600.0, 2: 1.955e6/3600.0, 3: 1.020e15/3600.0}   # kmol/(kgcat.s) [/3600: h->s]
E_a = {1: 240.1, 2: 67.13, 3: 243.9}                                 # kJ/mol
AK  = {1: 4.707e12, 2: 1.142e-2, 3: 5.375e10}
dH298 = {1: 206.1, 2: -41.15, 3: 164.9}                              # kJ/mol   Eq.(A.6) exponent
AKi = {'CO':8.23e-5, 'H2':6.12e-9, 'CH4':6.65e-4, 'H2O':1.77e5}      # bar^-1 (H2O dimensionless)
dHi = {'CO':-70.65, 'H2':-82.90, 'CH4':-38.28, 'H2O':88.68}          # kJ/mol   Eq.(A.8) exponent

# Heat-capacity polynomial coefficients (J/mol/K), Table C.5,  Eq.(C.10)
CP = {
    'CH4': (34.942, -3.9957e-2, 1.9184e-4, -1.5303e-7, 3.9321e-11),
    'CO' : (29.556, -6.5807e-3, 2.0130e-5, -1.2227e-8, 2.2617e-12),
    'CO2': (27.437,  4.2315e-2,-1.9555e-5,  3.9968e-9,-2.9872e-13),
    'H2' : (25.399,  2.0178e-2,-3.8549e-5,  3.1880e-8,-8.7585e-12),
    'H2O': (33.933, -8.4186e-3, 2.9906e-5, -1.7825e-8, 3.6934e-12),
    'N2' : (29.342, -3.5395e-3, 1.0076e-5, -4.3116e-9, 2.5935e-13),
}
# Reaction stoichiometry (nu_ij) for the 3 reactions, used in Eq.(18) & Eq.(A.7)
NU = {  # {species: [nu_I, nu_II, nu_III]}
    'CH4': [-1, 0, -1], 'CO': [1, -1, 0], 'CO2': [0, 1, 1],
    'H2': [3, 1, 4], 'H2O': [-1, -1, -2], 'N2': [0, 0, 0]}
NU_MAT = np.array([NU[s] for s in species])          # (6,3)

# Viscosity correlation coefficients (micropoise, T in K), Table C.2, Eq.(C.3)
VISC = {
    'CH4': (3.844, 4.0112e-1, -1.4303e-4), 'CO': (23.811, 5.3944e-1, -1.5411e-4),
    'CO2': (11.811, 4.9838e-1, -1.0851e-4), 'H2': (27.758, 2.1200e-1, -3.2800e-5),
    'H2O': (-36.826, 4.2900e-1, -1.6200e-5), 'N2': (42.606, 4.7500e-1, -9.8800e-5)}

# Thermal conductivity correlation coefficients (W/m/K, T in K), Table C.4, Eq.(C.8)
LAMG = {
    'CH4': (-0.00935, 1.4028e-4, 3.3180e-8), 'CO': (0.00158, 8.2511e-5, -1.9081e-8),
    'CO2': (-0.01200, 1.0208e-4, -2.2403e-8), 'H2': (0.03951, 4.5918e-4, -6.4933e-8),
    'H2O': (0.00053, 4.7093e-5, 4.9551e-8), 'N2': (0.00309, 7.5930e-5, -1.1014e-8)}

# Critical properties, Table C.3, used in Eq.(C.5)-(C.7)
TC = {'CH4':190.58,'CO':132.92,'CO2':304.19,'H2':33.18,'H2O':647.13,'N2':126.10}   # K
PC = {'CH4':46.04, 'CO':34.99, 'CO2':73.82, 'H2':13.13, 'H2O':220.55,'N2':33.94}    # bar

# Lennard-Jones parameters, Table B.1, used in Eq.(B.19)-(B.22)
LJ_SIGMA = {'CH4':3.758,'CO':3.690,'CO2':3.941,'H2':2.827,'H2O':2.641,'N2':3.798}   # Angstrom
LJ_EPSK  = {'CH4':148.6,'CO':91.7 ,'CO2':195.2,'H2':59.7 ,'H2O':809.1,'N2':71.4 }    # K

MWv = np.array([MW[s] for s in species])

# ============================================================================
# 3. PHYSICAL-PROPERTY / TRANSPORT FUNCTIONS  (Appendices B & C)
#    All arrays are vectorized over the axial grid (length Nz).
# ============================================================================
def cp_i_molar(T, sp):
    """Eq.(C.10): pure-component molar Cp [J/mol/K]"""
    A, B, C, D, E = CP[sp]
    return A + B*T + C*T**2 + D*T**3 + E*T**4

def mu_i(T, sp):
    """Eq.(C.3): pure-component viscosity [micropoise]"""
    A, B, C = VISC[sp]
    return A + B*T + C*T**2

def lam_i(T, sp):
    """Eq.(C.8): pure-component thermal conductivity [W/m/K]"""
    A, B, C = LAMG[sp]
    return A + B*T + C*T**2

def mu_mix(T, ymol):
    """Gas mixture viscosity, Eq.(C.1)-(C.2)  -> Pa.s"""
    mus = np.array([mu_i(T, s) for s in species])          # (6,Nz) micropoise
    num = ymol*mus
    out = np.zeros_like(T)
    for i, si in enumerate(species):
        phij = np.sqrt(MW[si]/MWv)[:, None] if False else np.sqrt(MWv/MW[si])  # phi_ij=(Mj/Mi)^0.5, Eq.(C.2)
        denom = np.sum(ymol*phij[:, None], axis=0)
        out += num[i]/denom
    return out*1e-7   # micropoise -> Pa.s

def lambda_mix(T, ymol):
    """Gas mixture thermal conductivity, Eq.(C.4)-(C.7)  -> W/m/K"""
    lams = np.array([lam_i(T, s) for s in species])
    Tr = {s: T/TC[s] for s in species}
    gam = {s: 210.0*(TC[s]*MW[s]**3/PC[s]**4)**(1.0/6.0) for s in species}   # Eq.(C.7)
    ltr = {s: gam[s]*(np.exp(0.0464*Tr[s]) - np.exp(-0.2412*Tr[s])) for s in species}  # ~ 1/lam_tr scale, Eq.(C.6) num/denom split below
    out = np.zeros_like(T)
    for i, si in enumerate(species):
        Aij = np.zeros((len(species), len(T)))
        for j, sj in enumerate(species):
            ratio = ltr[sj]/ltr[si]           # (lam_tr,i/lam_tr,j) via Eq.(C.6) form (proportional)
            Aij[j] = (1.0 + np.sqrt(ratio)*(MW[si]/MW[sj])**0.25)**2 / np.sqrt(8.0*(1.0 + MW[si]/MW[sj]))
        denom = np.sum(ymol*Aij, axis=0)
        out += ymol[i]*lams[i]/denom
    return out

def cp_mix_mass(T, ymol):
    """Gas mixture heat capacity, Eq.(C.9) (molar) -> mass basis [J/kg/K]"""
    cps = np.array([cp_i_molar(T, s) for s in species])
    cp_molar = np.sum(ymol*cps, axis=0)              # J/mol/K
    Mavg = np.sum(ymol*MWv[:, None], axis=0)          # kg/kmol = g/mol
    return cp_molar*1000.0/Mavg                        # J/kg/K

def binary_diff(T, P_bar, si, sj):
    """Eq.(B.19)-(B.22): binary diffusivity D_ij [cm2/s]"""
    Mij = 2.0/(1.0/MW[si] + 1.0/MW[sj])                        # Eq.(B.20)
    sij = 0.5*(LJ_SIGMA[si] + LJ_SIGMA[sj])                     # Eq.(B.21)
    x = T/np.sqrt(LJ_EPSK[si]*LJ_EPSK[sj])                       # kT/sqrt(eps_i eps_j), Eq.(B.22) argument
    OmegaD = (1.06036/x**0.15610 + 0.19300/np.exp(0.47635*x) +
              1.03587/np.exp(1.52996*x) + 1.76474/np.exp(3.89411*x))
    return 0.00266*T**1.5/(P_bar*np.sqrt(Mij)*sij**2*OmegaD)     # cm2/s

def Dim_mix(T, P_bar, ymol):
    """Molecular diffusivity of each species in the mixture (Blanc's law), Eq.(B.18) -> m2/s"""
    Nz = len(T)
    Dim = np.zeros((len(species), Nz))
    for i, si in enumerate(species):
        inv_sum = np.zeros(Nz)
        for j, sj in enumerate(species):
            if i == j:
                continue
            Dij = binary_diff(T, P_bar, si, sj)     # cm2/s
            inv_sum += ymol[j]/np.maximum(Dij, 1e-30)
        Dim[i] = 1.0/np.maximum(inv_sum, 1e-30)
    return Dim*1e-4      # cm2/s -> m2/s

def eff_axial_mass_dispersion(T, P_bar, ymol, Uz, mu_g, rho_g):
    """Effective axial mass-dispersion coefficient, Eq.(B.11)-(B.14) -> m2/s"""
    Dim = Dim_mix(T, P_bar, ymol)                       # (6,Nz), m2/s
    uz_int = Uz/eps_b                                     # interstitial velocity (nomenclature u_z)
    De = 0.78*Dim + (0.54*uz_int*dp/eps_b) / (1.0 + 9.2*Dim/np.maximum(uz_int*dp/eps_b, 1e-30))
    return De          # Eq.(B.14)

def eff_axial_thermal_conductivity(T, Uz, mu_g, lam_g, rho_g, Cp_g):
    """Effective axial thermal conductivity, Eq.(B.24),(B.26),(B.28)-(B.30) -> W/m/K"""
    Re = dp*Uz*rho_g/mu_g                       # Eq.(B.4), superficial
    Pr = Cp_g*mu_g/lam_g                          # Eq.(B.9)
    B = 1.25*((1.0 - eps_b)/eps_b)**(10.0/9.0)     # Eq.(B.29)
    ratio = lam_g/lam_p
    term = ((1.0 - ratio)*B/(1.0 - ratio*B)**2)*np.log(1.0/np.maximum(ratio*B, 1e-30)) \
           - (B + 1.0)/2.0 - (B - 1.0)/(1.0 - ratio*B)
    lam_zp = (1.0 - np.sqrt(1.0 - eps_b))*lam_g + 2.0*lam_g*np.sqrt(1.0 - eps_b)/(1.0 - ratio*B)*term  # Eq.(B.28)
    Peh_z_inv = 0.5/(1.0 + 9.7*eps_b/np.maximum(Re*Pr, 1e-30)) + (0.73*eps_b + lam_zp/lam_g)/np.maximum(Re*Pr, 1e-30)  # Eq.(B.26)
    lam_z_eff = rho_g*Uz*Cp_g*dp*Peh_z_inv         # Eq.(B.24): lam_z^e = rho*Uz*Cp*dp/Pe -> lam = rho*Uz*Cp*dp * (1/Pe)
    return lam_z_eff

def wall_htc(T, Uz, mu_g, lam_g, rho_g, Cp_g):
    """Overall (tube-wall to bulk gas) heat-transfer coefficient U(z), Eq.(B.31) -> W/m2/K"""
    Re = dp*Uz*rho_g/mu_g
    Pr = Cp_g*mu_g/lam_g
    return 0.4*(lam_g/dp)*(2.58*Re**(1.0/3.0)*Pr**(1.0/3.0) + 0.094*Re**0.8*Pr**0.4)

def dHrxn_T(T, j):
    """Heat of reaction at T via Cp integration, Eq.(A.7) -> kJ/mol"""
    Tl = 298.15
    integral = np.zeros_like(T)
    for s in species:
        nu = NU[s][j-1]
        if nu == 0:
            continue
        A, B, C, D, E = CP[s]
        val = (A*(T - Tl) + B/2*(T**2 - Tl**2) + C/3*(T**3 - Tl**3) +
               D/4*(T**4 - Tl**4) + E/5*(T**5 - Tl**5))
        integral += nu*val
    return dH298[j] + integral/1000.0     # J/mol -> kJ/mol

def kinetics(T, Pp):
    """Xu & Froment LHHW kinetics, Eq.(A.1)-(A.8), Table A.1.
       Pp: dict of partial pressures [bar]. Returns R1,R2,R3 [kmol/(kgcat.s)]."""
    k = {j: Ak[j]*np.exp(-E_a[j]/(Rg_kJ*T)) for j in (1, 2, 3)}
    K = {j: AK[j]*np.exp(-dH298[j]/(Rg_kJ*T)) for j in (1, 2, 3)}
    Ki = {s: AKi[s]*np.exp(-dHi[s]/(Rg_kJ*T)) for s in ('CO', 'H2', 'CH4', 'H2O')}
    H2 = np.maximum(Pp['H2'], 1e-8)
    DEN = 1.0 + Ki['CO']*Pp['CO'] + Ki['H2']*H2 + Ki['CH4']*Pp['CH4'] + Ki['H2O']*Pp['H2O']/H2
    R1 = (k[1]/H2**2.5)*(Pp['CH4']*Pp['H2O'] - H2**3*Pp['CO']/K[1])/DEN**2         # Eq.(A.1)
    R2 = (k[2]/H2)     *(Pp['CO'] *Pp['H2O'] - H2*Pp['CO2']/K[2])/DEN**2            # Eq.(A.2)
    R3 = (k[3]/H2**3.5)*(Pp['CH4']*Pp['H2O']**2 - H2**4*Pp['CO2']/K[3])/DEN**2      # Eq.(A.3)
    return R1, R2, R3

# ============================================================================
# 4. AXIAL GRID & FINITE-DIFFERENCE OPERATORS
# ============================================================================
Nz = 100                       # number of axial nodes, as used in the paper (Sec. 4.1.2)
z = np.linspace(0.0, L, Nz)
dz = z[1] - z[0]

def d_dz_backward(f):
    """Backward finite difference (matches ACM discretization, Sec. 3)"""
    d = np.empty_like(f)
    d[1:] = (f[1:] - f[:-1])/dz
    d[0] = (f[1] - f[0])/dz
    return d

def d2_dz2_central(f):
    """Central 2nd derivative with zero-gradient (Neumann) outlet BC, Eq.(17)"""
    d2 = np.zeros_like(f)
    d2[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2])/dz**2
    d2[-1] = (f[-2] - f[-1])/dz**2       # ghost node f[N]=f[N-1] -> zero gradient
    return d2

n = Nz - 1     # number of evolved (non-inlet) nodes
tau_P = 0.01   # s, artificial pseudo-transient time constant for the pressure
               # field (see note below) -- fast relative to the ~few-second
               # convective/reactive timescale, so it does not distort the
               # steady-state answer, only how quickly P relaxes to it.

# ============================================================================
# 4b. COUPLING NOTE  (Eq. 11 + Eq. 13-14)
# ----------------------------------------------------------------------------
# Reactions I-III are REVERSIBLE (Eqs. A.1-A.3 all contain a -P^n/K_j back-
# reaction term), so the LOCAL partial pressures set how far each reaction
# sits from equilibrium. The total mass flux G = rho_g*Uz is exactly constant
# along z (mass conservation, steady 1D flow, no mass source: G0 defined
# above) -- but rho_g must come from the IDEAL-GAS LAW at the REAL local
# pressure (Eq. 11), and that real pressure must itself solve the Ergun
# momentum balance (Eq. 13-14), not be read off the raw evolving
# concentrations. To keep this coupling CHEAP, pressure P(z,t) is added as an
# 8th state field (alongside the 6 species and T) and advanced with the SAME
# method-of-lines machinery already used for Eqs.(18)-(19): a backward finite
# difference for dP/dz, driven toward the physical Ergun gradient by an
# artificial time derivative -- exactly the "transient term...to dynamically
# approach the steady-state solution" trick the paper itself uses for the
# whole PDE system (Sec. 3, discussion below Eq. 9). This avoids re-solving a
# nonlinear inner loop on every RHS call (which is what made the earlier
# version hang) while still fully coupling P into the density -> velocity ->
# convection terms and into the kinetics' partial pressures every step.
# ============================================================================

# ============================================================================
# 5. ODE RIGHT-HAND SIDE  (Eq. 18: species, Eq. 19: energy, Eq. 13-14: pressure)
# ============================================================================
def unpack(y):
    C = np.empty((6, Nz)); T = np.empty(Nz); P = np.empty(Nz)
    C[:, 0] = C0
    C[:, 1:] = y[:6*n].reshape(6, n)
    T[0] = T0
    T[1:] = y[6*n:7*n]
    P[0] = P0
    P[1:] = y[7*n:8*n]
    return C, T, P

def rhs(t, y):
    C, T, P = unpack(y)
    Ctot = np.sum(C, axis=0)
    ymol = C/Ctot                                            # mole fractions (composition only)
    Mavg = np.sum(ymol*MWv[:, None], axis=0)

    R_kmol = Rg*1000.0
    rho_g = P*1e5*Mavg/(R_kmol*T)                              # Eq.(11), using the REAL coupled P
    Uz = G0/rho_g                                               # mass-flux conservation (exact)
    P_i = ymol*P[None, :]                                        # partial pressures (bar) for kinetics
    Ptot = P

    Pp = {s: P_i[i] for i, s in enumerate(species)}
    R1, R2, R3 = kinetics(T, Pp)
    Rj = np.vstack([R1, R2, R3])                                   # (3,Nz)  kmol/(kgcat.s)

    mu_g = mu_mix(T, ymol)
    lam_g = lambda_mix(T, ymol)
    Cp_g = cp_mix_mass(T, ymol)                                      # J/kg/K

    Dax = eff_axial_mass_dispersion(T, Ptot, ymol, Uz, mu_g, rho_g)     # (6,Nz) m2/s
    lam_z_eff = eff_axial_thermal_conductivity(T, Uz, mu_g, lam_g, rho_g, Cp_g)
    Uwall = wall_htc(T, Uz, mu_g, lam_g, rho_g, Cp_g)

    dH = np.vstack([dHrxn_T(T, 1), dHrxn_T(T, 2), dHrxn_T(T, 3)])*1000.0/1000.0  # kJ/mol (kept)
    dH_Jkmol = dH*1e6         # kJ/mol -> J/kmol  (x1000 mol->kmol, x1000 kJ->J)

    # ---- species balance, Eq.(18) ----
    CU = C*Uz                                                       # convective flux
    dCU_dz = np.array([d_dz_backward(CU[i]) for i in range(6)])
    d2C_dz2 = np.array([d2_dz2_central(C[i]) for i in range(6)])

    rxn_source = (1.0 - eps_b)*rho_p*ETA*(NU_MAT @ Rj)                # (6,Nz), kmol/(m3.s)
    dCdt = (-dCU_dz + eps_b*Dax*d2C_dz2 + rxn_source)/eps_b

    # ---- energy balance, Eq.(19) ----
    dT_dz = d_dz_backward(T)
    d2T_dz2 = d2_dz2_central(T)
    dC_dz = np.array([d_dz_backward(C[i]) for i in range(6)])
    Cp_i_molar = np.array([cp_i_molar(T, s) for s in species])*1000.0  # J/mol/K -> J/kmol/K

    heat_disp = np.sum(eps_b*Dax*dC_dz*Cp_i_molar, axis=0)*dT_dz        # dispersive-heat term
    rxn_heat = (1.0 - eps_b)*rho_p*ETA*np.sum(Rj*(-dH_Jkmol[:, None] if False else -dH_Jkmol), axis=0)
    # careful broadcasting: dH_Jkmol shape (3,Nz), Rj shape (3,Nz)
    rxn_heat = (1.0 - eps_b)*rho_p*ETA*np.sum(Rj*(-dH_Jkmol), axis=0)

    wall_heat = (4.0*Uwall/dt_tube)*(Tw - T)
    conv_heat = rho_g*Cp_g*Uz*dT_dz

    cap = (1.0 - eps_b)*rho_p*Cp_p + eps_b*rho_g*Cp_g
    dTdt = (-conv_heat + wall_heat + heat_disp + lam_z_eff*d2T_dz2 + rxn_heat)/cap

    # ---- pressure balance, Eq.(13)-(14), relaxed via artificial pseudo-time
    #      term (see Sec. 4b note) so it is solved by the same FD machinery ----
    dPdz_FD = d_dz_backward(P)                                        # bar/m, current gradient
    dPdz_Ergun = -(G0/(rho_g*dp))*((1.0 - eps_b)/eps_b**3) * \
                  (150.0*(1.0 - eps_b)*mu_g/dp + 1.75*G0)/1e5           # bar/m, Eq.(13)-(14) target
    dPdt = (dPdz_Ergun - dPdz_FD)/tau_P

    return np.concatenate([dCdt[:, 1:].ravel(), dTdt[1:], dPdt[1:]])

# ============================================================================
# 6. INITIAL CONDITION & TIME INTEGRATION  (pseudo-transient continuation)
# ============================================================================
y0 = np.concatenate([np.tile(C0, (n, 1)).T.ravel(), np.full(n, T0), np.full(n, P0)])

# ---- Jacobian SPARSITY PATTERN --------------------------------------------
# The state is laid out as 8 stacked blocks (CH4,CO,CO2,H2,H2O,N2,T,P), each
# of length n, ordered by axial node. Because all derivatives use only
# backward 1st- and central 2nd-order finite differences (3-point stencils),
# node j's equations depend only on node j-1, j, j+1 (across ALL 8 fields,
# since kinetics/properties/density couple everything locally). This gives a
# block-tridiagonal (banded) Jacobian instead of a dense one.
# Without this hint, solve_ivp's BDF perturbs each of the ~800 states one at a
# time to build a DENSE numerical Jacobian (the dominant cost, confirmed by
# profiling). With the sparsity hint it uses graph-coloring to estimate the
# same Jacobian in a handful of vectorized rhs() evaluations -> much faster.
tri = diags([1.0, 1.0, 1.0], offsets=[-1, 0, 1], shape=(n, n), format='csr')
jac_sparsity = kron(csr_matrix(np.ones((8, 8))), tri, format='csr')

t_final = 300.0    # s, several residence times -> steady state (Sec. 3 of paper)
sol = solve_ivp(rhs, [0.0, t_final], y0, method='BDF',
                 rtol=1e-6, atol=1e-9, jac_sparsity=jac_sparsity)

C_ss, T_ss, P_profile_bar = unpack(sol.y[:, -1])

# ============================================================================
# 7. POST-PROCESSING (from the converged, fully-coupled steady-state fields)
# ============================================================================
ymol_ss = C_ss/np.sum(C_ss, axis=0)
Mavg_ss = np.sum(ymol_ss*MWv[:, None], axis=0)
rho_g_ss = P_profile_bar*1e5*Mavg_ss/(Rg*1000.0*T_ss)     # Eq.(11), real coupled P
Uz_ss = G0/rho_g_ss

# ============================================================================
# 8. PLOTS
# ============================================================================
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].plot(z, P_profile_bar, 'b-')
axs[0].set_xlabel('Reactor length (m)'); axs[0].set_ylabel('Pressure (bar)')
axs[0].set_title('Pressure profile (Ergun, Eq. 13-14)')
axs[1].plot(z, T_ss, 'r-')
axs[1].set_xlabel('Reactor length (m)'); axs[1].set_ylabel('Temperature (K)')
axs[1].set_title('Temperature profile (M2, Eq. 19)')
plt.tight_layout()

fig2, ax2 = plt.subplots(figsize=(6, 4.5))
for i, s in enumerate(species):
    ax2.plot(z, C_ss[i], label=s)
ax2.set_xlabel('Reactor length (m)'); ax2.set_ylabel('Concentration (kmol/m3)')
ax2.set_title('Component concentration profiles (M2)')
ax2.legend()
plt.tight_layout()

X_CH4 = 1.0 - (C_ss[0]*Uz_ss)/(C0[0]*Uz0)     # Eq.(88)-style CH4 conversion
print(f"CH4 conversion at outlet: {X_CH4[-1]:.4f}  (eta = {ETA})")

# fig.savefig('/mnt/user-data/outputs/M2_pressure_temperature.png', dpi=150)
# fig2.savefig('/mnt/user-data/outputs/M2_concentrations.png', dpi=150)
# NOTE: plt.show() is intentionally NOT called here. It opens a blocking GUI
# window and waits for you to close it before the script can exit - on a
# headless machine (no display) or some IDEs this hangs indefinitely, which
# looks like the script "never finishes". Uncomment the line below only if
# you are running this interactively on a machine with a display:
plt.show()