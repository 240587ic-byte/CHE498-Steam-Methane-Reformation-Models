"""
Flat-profile simplification of the M5 steam-methane-reforming tube model
=========================================================================

M5 (reference code "M5 Model Claude.py")
    non-flat velocity u_z(z,r), non-flat concentration C_i(z,r),
    non-flat temperature T(z,r), axial + radial dispersion.
This model
    FLAT velocity u_z(z), FLAT concentration C_i(z), FLAT temperature T(z),
    axial + radial dispersion kept.

Reactions (Xu & Froment kinetics, exactly as in M5)
    (I)   CH4 +   H2O  <->  CO  + 3 H2
    (II)  CO  +   H2O  <->  CO2 +   H2
    (III) CH4 + 2 H2O  <->  CO2 + 4 H2

All data, correlations, kinetics, axial grid, boundary conditions and solver
tolerances are taken unchanged from M5. No new data were introduced.

How the model follows from M5
-----------------------------
Every term of M5's Eq.(41)-(44) is averaged over the tube cross-section,

        <f>(z) = (2/r_t^2) * Integral_0^r_t  f(z,r) r dr ,

with u_z, C_i, T and P independent of r. A term without an r-derivative is
unchanged (a flat field is its own average). The radial-dispersion terms are
integrated exactly, and each one becomes its wall flux:

    <(1/r) d/dr (r df/dr)>  =  (2/r_t) * df/dr |_(r = r_t)

    species, Eq.(41): <eps_b D_r,i lap_r(C_i)> = (2/r_t) eps_b D_r,i dC_i/dr|_rt = 0
                      (impermeable wall, Eq. 49)
    energy,  Eq.(42): <lam_r lap_r(T)>          = (2/r_t) lam_r dT/dr|_rt
                                                = (2/r_t) U (T_w - T)
                      (wall BC, Eq. 53, with T(r_t) = T for a flat profile)
    energy,  Eq.(42): <sum_i eps_b D_r,i Cp_i dC_i/dr dT/dr> = 0   (dC_i/dr = 0)

So radial dispersion stays in the model; with flat profiles its only net
effect is the heat delivered through the tube wall,
(2/r_t) U (T_w - T) = (4/d_t) U (T_w - T), and the radial coefficients D_r and
lam_r cancel out of the balances. The equations solved are

  species   eps_b dC_i/dt = - d(u_z C_i)/dz + eps_b D_z,i d2C_i/dz2
                            + (1 - eps_b) rho_p eta sum_j nu_ij R_j
  energy    [(1 - eps_b) rho_p Cp_p + eps_b rho_g Cp_g] dT/dt
                          = - rho_g Cp_g u_z dT/dz + lam_z d2T/dz2
                            + (2/r_t) U (T_w - T)
                            + sum_i eps_b D_z,i Cp_i dC_i/dz dT/dz
                            + (1 - eps_b) rho_p eta sum_j R_j (-dH_j)
  pressure  dP/dz = Ergun, Eq.(43), with u_z = G0/rho_g, Eq.(44)

  inlet  z = 0 : C_i = C_i0, T = T0, P = P0          (Eq. 46, 50, 58)
  outlet z = L : zero axial gradient                   (Eq. 47, 51, 59)

Solved like M5: method of lines on the same 50-node axial grid, marched in
pseudo-time with solve_ivp (BDF) until the profiles stop changing.

Packages: numpy, scipy.integrate.solve_ivp, matplotlib (plus the standard-
library time module for the run-time print).
"""
import time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

start_time = time.time()

# ============================================================================
# 1. OPERATING / GEOMETRIC DATA  (Table 2) - identical to M5
# ============================================================================
Rg      = 8.314                 # J/mol/K
Rg_bar  = 0.08314               # bar.m3/(kmol.K)
Rg_kJ   = 8.314e-3              # kJ/mol/K  (Arrhenius / van't Hoff exponents, App. A)

L       = 12.0                  # m,     reactor length                  Table 2
dt_tube = 0.10                  # m,     tube inner diameter             Table 2
rt      = dt_tube/2.0           # m,     tube inner radius
dp      = 0.01                  # m,     catalyst particle diameter      Table 2
rho_p   = 2355.2                # kg/m3, catalyst density                Table 2
Cp_p    = 950.0                 # J/(kg.K), catalyst heat capacity       Table 2
lam_p   = 0.3489                # W/(m.K), catalyst solid conductivity   Table 2
T0      = 793.15                # K,   inlet temperature                 Table 2
P0      = 25.69                 # bar, inlet pressure                    Table 2
Tw      = 1000.0                # K,   constant tube-wall temperature    (Assumption 7)

species = ['CH4', 'CO', 'CO2', 'H2', 'H2O', 'N2']
MW = {'CH4': 16.0428, 'CO': 28.0104, 'CO2': 44.0098, 'H2': 2.01588, 'H2O': 18.0153,
      'N2': 28.0135}                                                        # Table C.1

F0_h = {'CH4': 5.17, 'CO': 0.0, 'CO2': 0.29, 'H2': 0.63, 'H2O': 17.35, 'N2': 0.85}  # kmol/h
F0   = {k: v/3600.0 for k, v in F0_h.items()}                                     # kmol/s

eps_b  = 0.38 + 0.073*(1.0 - (dt_tube/dp - 2.0)**2/(dt_tube/dp)**2)   # bed voidage, Eq.(3)
Omega  = np.pi*dt_tube**2/4.0                                          # cross-section, Eq.(7)
F0_tot = sum(F0.values())
Uz0    = F0_tot*Rg_bar*T0/(Omega*P0)                                   # inlet velocity, Eq.(8)
rho_g0 = sum(F0[s]*MW[s] for s in species)/(Uz0*Omega)                 # kg/m3
G0     = rho_g0*Uz0                                                    # kg/(m2.s), constant, Eq.(44)
C0     = np.array([F0[s]/(Uz0*Omega) for s in species])                # kmol/m3, Eq.(6)

ETA = 1.0     # effectiveness factor, as in M5 (1, 0.1, 0.01 or 0.001 for the Table 3 sweep)

MWv = np.array([MW[s] for s in species])

# ============================================================================
# 2. KINETIC PARAMETERS - Xu & Froment (1989), Table A.1 - identical to M5
# ============================================================================
Ak    = {1: 4.225e15/3600.0, 2: 1.955e6/3600.0, 3: 1.020e15/3600.0}   # kmol/(kgcat.s)
E_a   = {1: 240.1, 2: 67.13, 3: 243.9}                                 # kJ/mol
AK    = {1: 4.707e12, 2: 1.142e-2, 3: 5.375e10}
dH298 = {1: 206.1, 2: -41.15, 3: 164.9}      # kJ/mol, base of the Cp integration, Eq.(A.7)
dH948 = {1: 224.0, 2: -37.3, 3: 187.5}       # kJ/mol, used inside K_j, Eq.(A.6) (as in M5)
AKi   = {'CO': 8.23e-5, 'H2': 6.12e-9, 'CH4': 6.65e-4, 'H2O': 1.77e5}
dHi   = {'CO': -70.65, 'H2': -82.90, 'CH4': -38.28, 'H2O': 88.68}      # kJ/mol, Eq.(A.8)

# Heat-capacity polynomials (J/mol/K), Table C.5, Eq.(C.10)
CP = {
    'CH4': (34.942, -3.9957e-2,  1.9184e-4, -1.5303e-7,  3.9321e-11),
    'CO' : (29.556, -6.5807e-3,  2.0130e-5, -1.2227e-8,  2.2617e-12),
    'CO2': (27.437,  4.2315e-2, -1.9555e-5,  3.9968e-9, -2.9872e-13),
    'H2' : (25.399,  2.0178e-2, -3.8549e-5,  3.1880e-8, -8.7585e-12),
    'H2O': (33.933, -8.4186e-3,  2.9906e-5, -1.7825e-8,  3.6934e-12),
    'N2' : (29.342, -3.5395e-3,  1.0076e-5, -4.3116e-9,  2.5935e-13),
}
# Stoichiometry nu_ij for reactions I, II, III
NU = {'CH4': [-1, 0, -1], 'CO': [1, -1, 0], 'CO2': [0, 1, 1],
      'H2': [3, 1, 4], 'H2O': [-1, -1, -2], 'N2': [0, 0, 0]}
NU_MAT = np.array([NU[s] for s in species])          # (6, 3)

# Viscosity coefficients (micropoise), Table C.2, Eq.(C.3)
VISC = {
    'CH4': (3.844, 4.0112e-1, -1.4303e-4), 'CO': (23.811, 5.3944e-1, -1.5411e-4),
    'CO2': (11.811, 4.9838e-1, -1.0851e-4), 'H2': (27.758, 2.1200e-1, -3.2800e-5),
    'H2O': (-36.826, 4.2900e-1, -1.6200e-5), 'N2': (42.606, 4.7500e-1, -9.8800e-5)}

# Thermal conductivity coefficients (W/m/K), Table C.4, Eq.(C.8)
LAMG = {
    'CH4': (-0.00935, 1.4028e-4, 3.3180e-8), 'CO': (0.00158, 8.2511e-5, -1.9081e-8),
    'CO2': (-0.01200, 1.0208e-4, -2.2403e-8), 'H2': (0.03951, 4.5918e-4, -6.4933e-8),
    'H2O': (0.00053, 4.7093e-5, 4.9551e-8), 'N2': (0.00309, 7.5930e-5, -1.1014e-8)}

# Critical properties, Table C.3
TC = {'CH4': 190.58, 'CO': 132.92, 'CO2': 304.19, 'H2': 33.18, 'H2O': 647.13, 'N2': 126.10}
PC = {'CH4': 46.04, 'CO': 34.99, 'CO2': 73.82, 'H2': 13.13, 'H2O': 220.55, 'N2': 33.94}

# Lennard-Jones parameters, Table B.1
LJ_SIGMA = {'CH4': 3.758, 'CO': 3.690, 'CO2': 3.941, 'H2': 2.827, 'H2O': 2.641, 'N2': 3.798}
LJ_EPSK  = {'CH4': 148.6, 'CO': 91.7, 'CO2': 195.2, 'H2': 59.7, 'H2O': 809.1, 'N2': 71.4}

# ============================================================================
# 3. PHYSICAL-PROPERTY / TRANSPORT FUNCTIONS (Appendices B & C) - as in M5.
#    With flat profiles every field is a 1-D array over z, shape (Nz,).
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

def _bshape(T):
    """(n_species,) + (1,)*T.ndim, for broadcasting per-species vectors."""
    return (len(species),) + (1,)*np.ndim(T)

def mu_mix(T, ymol):
    """Mixture viscosity, Eq.(C.1)-(C.2) -> Pa.s"""
    mus = np.array([mu_i(T, s) for s in species])
    out = np.zeros_like(T)
    for i, si in enumerate(species):
        phij = np.sqrt(MWv/MW[si])
        denom = np.sum(ymol*phij.reshape(_bshape(T)), axis=0)
        out += ymol[i]*mus[i]/denom
    return out*1e-7      # micropoise -> Pa.s

def lambda_mix(T, ymol):
    """Mixture thermal conductivity, Eq.(C.4)-(C.7) -> W/m/K"""
    lams = np.array([lam_i(T, s) for s in species])
    Tr = {s: T/TC[s] for s in species}
    gam = {s: 210.0*(TC[s]*MW[s]**3/PC[s]**4)**(1.0/6.0) for s in species}          # Eq.(C.7)
    ltr = {s: gam[s]*(np.exp(0.0464*Tr[s]) - np.exp(-0.2412*Tr[s])) for s in species}  # Eq.(C.6)
    out = np.zeros_like(T)
    for i, si in enumerate(species):
        Aij = np.stack([
            (1.0 + np.sqrt(ltr[sj]/ltr[si])*(MW[si]/MW[sj])**0.25)**2 /
            np.sqrt(8.0*(1.0 + MW[si]/MW[sj]))
            for sj in species], axis=0)
        denom = np.sum(ymol*Aij, axis=0)
        out += ymol[i]*lams[i]/denom
    return out

def cp_mix_mass(T, ymol):
    """Mixture heat capacity, Eq.(C.9) -> J/kg/K"""
    cps = np.array([cp_i_molar(T, s) for s in species])
    cp_molar = np.sum(ymol*cps, axis=0)                           # J/mol/K
    Mavg = np.sum(ymol*MWv.reshape(_bshape(T)), axis=0)           # kg/kmol
    return cp_molar*1000.0/Mavg

def binary_diff(T, P_bar, si, sj):
    """Eq.(B.19)-(B.22): binary diffusivity D_ij [cm2/s]"""
    Mij = 2.0/(1.0/MW[si] + 1.0/MW[sj])
    sij = 0.5*(LJ_SIGMA[si] + LJ_SIGMA[sj])
    x = T/np.sqrt(LJ_EPSK[si]*LJ_EPSK[sj])
    OmegaD = (1.06036/x**0.15610 + 0.19300/np.exp(0.47635*x) +
              1.03587/np.exp(1.52996*x) + 1.76474/np.exp(3.89411*x))
    return 0.00266*T**1.5/(P_bar*np.sqrt(Mij)*sij**2*OmegaD)

def Dim_mix(T, P_bar, ymol):
    """Species diffusivity in the mixture (Blanc's law), Eq.(B.18) -> m2/s"""
    Dim = np.zeros(_bshape(T)[:1] + np.shape(T))
    for i, si in enumerate(species):
        inv_sum = np.zeros_like(T)
        for j, sj in enumerate(species):
            if i == j:
                continue
            Dij = binary_diff(T, P_bar, si, sj)
            inv_sum += ymol[j]/np.maximum(Dij, 1e-30)
        Dim[i] = 1.0/np.maximum(inv_sum, 1e-30)
    return Dim*1e-4      # cm2/s -> m2/s

def eff_axial_mass_dispersion(T, P_bar, ymol, Uz, mu_g, rho_g):
    """Effective axial mass-dispersion coefficient, Eq.(B.11)-(B.14) -> m2/s"""
    Dim = Dim_mix(T, P_bar, ymol)
    uz_int = Uz/eps_b
    De = 0.78*Dim + (0.54*uz_int*dp/eps_b)/(1.0 + 9.2*Dim/np.maximum(uz_int*dp/eps_b, 1e-30))
    return De

def eff_axial_thermal_conductivity(T, Uz, mu_g, lam_g, rho_g, Cp_g):
    """Effective axial thermal conductivity, Eq.(B.24),(B.26),(B.28)-(B.30) -> W/m/K"""
    Re = dp*Uz*rho_g/mu_g
    Pr = Cp_g*mu_g/lam_g
    Bfac = 1.25*((1.0 - eps_b)/eps_b)**(10.0/9.0)
    ratio = lam_g/lam_p
    term = ((1.0 - ratio)*Bfac/(1.0 - ratio*Bfac)**2)*np.log(1.0/np.maximum(ratio*Bfac, 1e-30)) \
           - (Bfac + 1.0)/2.0 - (Bfac - 1.0)/(1.0 - ratio*Bfac)
    lam_zp = (1.0 - np.sqrt(1.0 - eps_b))*lam_g + \
             2.0*lam_g*np.sqrt(1.0 - eps_b)/(1.0 - ratio*Bfac)*term                  # Eq.(B.28)
    Peh_z_inv = 0.5/(1.0 + 9.7*eps_b/np.maximum(Re*Pr, 1e-30)) + \
                (0.73*eps_b + lam_zp/lam_g)/np.maximum(Re*Pr, 1e-30)                  # Eq.(B.26)
    return rho_g*Uz*Cp_g*dp*Peh_z_inv                                                 # Eq.(B.24)

def eff_radial_thermal_conductivity(T, Uz, mu_g, lam_g, rho_g, Cp_g):
    """Effective radial thermal conductivity, Eq.(B.25),(B.27)-(B.30) -> W/m/K.
    It cancels out of the flat-profile balances (see header); it is used only
    in the post-processing check of how flat the real temperature profile is."""
    Re = dp*Uz*rho_g/mu_g
    Pr = Cp_g*mu_g/lam_g
    Bfac = 1.25*((1.0 - eps_b)/eps_b)**(10.0/9.0)
    ratio = lam_g/lam_p
    term = ((1.0 - ratio)*Bfac/(1.0 - ratio*Bfac)**2)*np.log(1.0/np.maximum(ratio*Bfac, 1e-30)) \
           - (Bfac + 1.0)/2.0 - (Bfac - 1.0)/(1.0 - ratio*Bfac)
    lam_rp = (1.0 - np.sqrt(1.0 - eps_b))*lam_g + \
             2.0*lam_g*np.sqrt(1.0 - eps_b)/(1.0 - ratio*Bfac)*term                  # Eq.(B.28)
    Peh_r_inv = 0.1 + (0.66*eps_b + lam_rp/lam_g)/np.maximum(Re*Pr, 1e-30)            # Eq.(B.27)
    return rho_g*Uz*Cp_g*dp*Peh_r_inv                                                 # Eq.(B.25)

def wall_htc(T, Uz, mu_g, lam_g, rho_g, Cp_g):
    """Tube-wall to bulk-gas heat-transfer coefficient U(z), Eq.(B.31) -> W/m2/K"""
    Re = dp*Uz*rho_g/mu_g
    Pr = Cp_g*mu_g/lam_g
    return 0.4*(lam_g/dp)*(2.58*Re**(1.0/3.0)*Pr**(1.0/3.0) + 0.094*Re**0.8*Pr**0.4)

def dHrxn_T(T, j):
    """Heat of reaction j at T via Cp integration from 298.15 K, Eq.(A.7) -> kJ/mol"""
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
    return dH298[j] + integral/1000.0

def kinetics(T, Pp):
    """Xu & Froment LHHW rates, Eq.(A.1)-(A.8) -> R1, R2, R3 [kmol/(kgcat.s)]"""
    k = {j: Ak[j]*np.exp(-E_a[j]/(Rg_kJ*T)) for j in (1, 2, 3)}
    K = {j: AK[j]*np.exp(-dH948[j]/(Rg_kJ*T)) for j in (1, 2, 3)}
    Ki = {s: AKi[s]*np.exp(-dHi[s]/(Rg_kJ*T)) for s in ('CO', 'H2', 'CH4', 'H2O')}
    H2 = np.maximum(Pp['H2'], 1e-8)
    DEN = 1.0 + Ki['CO']*Pp['CO'] + Ki['H2']*H2 + Ki['CH4']*Pp['CH4'] + Ki['H2O']*Pp['H2O']/H2
    R1 = (k[1]/H2**2.5)*(Pp['CH4']*Pp['H2O'] - H2**3*Pp['CO']/K[1])/DEN**2      # Eq.(A.1)
    R2 = (k[2]/H2)     *(Pp['CO'] *Pp['H2O'] - H2*Pp['CO2']/K[2])/DEN**2         # Eq.(A.2)
    R3 = (k[3]/H2**3.5)*(Pp['CH4']*Pp['H2O']**2 - H2**4*Pp['CO2']/K[3])/DEN**2   # Eq.(A.3)
    return R1, R2, R3

# ============================================================================
# 4. AXIAL GRID & FINITE-DIFFERENCE OPERATORS - same grid and stencils as M5
#    (no radial grid: the radial direction has been averaged out)
# ============================================================================
Nz = 50                          # axial nodes, as in M5
# Grid note: with the fixed-value inlet conditions of M5 (Eq. 46, 50) plus axial
# dispersion, a finer grid resolves the sharp temperature dip just inside the
# inlet, and heat then conducts in across the inlet face. The outlet CH4
# conversion therefore creeps up with Nz (0.449 at 50, 0.455 at 400, 0.458 at
# 800 nodes); M5 shares this. At 50 nodes the result is within 0.001 of the
# grid-converged value obtained with a flux (Danckwerts-type) inlet (0.450).
z = np.linspace(0.0, L, Nz)
dz = z[1] - z[0]
n = Nz - 1                       # evolved nodes; z = 0 is fixed by the inlet BC

def d_dz_backward(f):
    """Backward (upwind) first derivative in z, as in M5."""
    d = np.empty_like(f)
    d[..., 1:] = (f[..., 1:] - f[..., :-1])/dz
    d[..., 0] = (f[..., 1] - f[..., 0])/dz
    return d

def d2_dz2_central(f):
    """Central second derivative in z, zero-gradient outlet (Eq. 47, 51, 59)."""
    d2 = np.zeros_like(f)
    d2[..., 1:-1] = (f[..., 2:] - 2*f[..., 1:-1] + f[..., :-2])/dz**2
    d2[..., -1] = (f[..., -2] - f[..., -1])/dz**2
    return d2

tau_P = 0.01   # s, pseudo-transient relaxation constant for pressure (same as M5)

# ============================================================================
# 5. STATE PACKING: 6 species C(z), T(z), P(z) on the n evolved nodes
# ============================================================================
nC, nT, nP = 6*n, n, n

def unpack(y):
    C = np.empty((6, Nz))
    T = np.empty(Nz)
    P = np.empty(Nz)
    C[:, 0] = C0                  # Eq.(46)
    T[0] = T0                     # Eq.(50)
    P[0] = P0                     # Eq.(58)
    C[:, 1:] = y[:nC].reshape(6, n)
    T[1:] = y[nC:nC+nT]
    P[1:] = y[nC+nT:]
    return C, T, P

# ============================================================================
# 6. RIGHT-HAND SIDE: cross-section-averaged Eq.(41)-(44)
# ============================================================================
def rhs(t, y):
    C, T, P = unpack(y)
    ymol = C/np.sum(C, axis=0)
    Mavg = np.sum(ymol*MWv[:, None], axis=0)

    rho_g = P*1e5*Mavg/(Rg*1000.0*T)                        # Eq.(11)
    Uz = G0/rho_g                                           # Eq.(44), flat velocity
    Pp = {s: ymol[i]*P for i, s in enumerate(species)}      # partial pressures, bar
    Rj = np.stack(kinetics(T, Pp), axis=0)                  # (3, Nz) kmol/(kgcat.s)

    mu_g = mu_mix(T, ymol)
    lam_g = lambda_mix(T, ymol)
    Cp_g = cp_mix_mass(T, ymol)
    Dax = eff_axial_mass_dispersion(T, P, ymol, Uz, mu_g, rho_g)       # (6, Nz)
    lam_z_eff = eff_axial_thermal_conductivity(T, Uz, mu_g, lam_g, rho_g, Cp_g)
    Uwall = wall_htc(T, Uz, mu_g, lam_g, rho_g, Cp_g)                  # Eq.(B.31)
    dH_Jkmol = np.stack([dHrxn_T(T, j) for j in (1, 2, 3)], axis=0)*1e6

    # ---- species, Eq.(41) averaged ----
    # radial dispersion: (2/rt)*eps_b*D_r*dC/dr|rt = 0 by the no-flux wall BC (Eq. 49)
    dCU_dz = d_dz_backward(C*Uz)
    d2C_dz2 = d2_dz2_central(C)
    rxn_source = (1.0 - eps_b)*rho_p*ETA*(NU_MAT @ Rj)
    dCdt = (-dCU_dz + eps_b*Dax*d2C_dz2 + rxn_source)/eps_b

    # ---- energy, Eq.(42) averaged ----
    dT_dz = d_dz_backward(T)
    d2T_dz2 = d2_dz2_central(T)
    dC_dz = d_dz_backward(C)
    Cp_i = np.array([cp_i_molar(T, s) for s in species])*1000.0       # J/kmol/K
    heat_disp_axial = np.sum(eps_b*Dax*dC_dz*Cp_i, axis=0)*dT_dz
    # radial dispersion of heat: (2/rt)*lam_r*dT/dr|rt with Eq.(53) = (2/rt)*U*(Tw - T);
    # the radial term with D_r*dC/dr*dT/dr vanishes because dC/dr = 0
    radial_heat = (2.0/rt)*Uwall*(Tw - T)
    rxn_heat = (1.0 - eps_b)*rho_p*ETA*np.sum(Rj*(-dH_Jkmol), axis=0)
    conv_heat = rho_g*Cp_g*Uz*dT_dz
    cap = (1.0 - eps_b)*rho_p*Cp_p + eps_b*rho_g*Cp_g
    dTdt = (-conv_heat + lam_z_eff*d2T_dz2 + radial_heat + heat_disp_axial + rxn_heat)/cap

    # ---- pressure, Eq.(43)-(44): Ergun, relaxed in pseudo-time as in M5 ----
    dPdz_FD = d_dz_backward(P)
    dPdz_Ergun = -(G0/(rho_g*dp))*((1.0 - eps_b)/eps_b**3) * \
                  (150.0*(1.0 - eps_b)*mu_g/dp + 1.75*G0)/1e5
    dPdt = (dPdz_Ergun - dPdz_FD)/tau_P

    return np.concatenate([dCdt[:, 1:].ravel(), dTdt[1:], dPdt[1:]])

# ============================================================================
# 7. INITIAL CONDITION & PSEUDO-TIME INTEGRATION TO STEADY STATE
# ============================================================================
y0 = np.concatenate([np.repeat(C0, n), np.full(n, T0), np.full(n, P0)])

# Jacobian sparsity: every node couples only to itself and its two axial
# neighbours, for all 8 fields -> 8x8 blocks of tridiagonal matrices.
band = np.eye(n) + np.eye(n, k=1) + np.eye(n, k=-1)
jac_sparsity = np.kron(np.ones((8, 8)), band)

t_final = 3000.0   # s of pseudo-time. The packed-bed heat capacity makes the
                   # thermal field the slowest to settle; see the residual print.
sol = solve_ivp(rhs, [0.0, t_final], y0, method='BDF',
                rtol=1e-6, atol=1e-9, jac_sparsity=jac_sparsity)
if not sol.success:
    raise RuntimeError(sol.message)

C_ss, T_ss, P_ss = unpack(sol.y[:, -1])

# ============================================================================
# 8. POST-PROCESSING
# ============================================================================
res = rhs(sol.t[-1], sol.y[:, -1])
print(f"Steady-state residuals at t = {sol.t[-1]:.0f} s: "
      f"max|dC/dt| = {np.abs(res[:nC]).max():.1e} kmol/m3/s, "
      f"max|dT/dt| = {np.abs(res[nC:nC+nT]).max():.1e} K/s, "
      f"max|dP/dt| = {np.abs(res[nC+nT:]).max():.1e} bar/s")

ymol_ss = C_ss/np.sum(C_ss, axis=0)                   # mole fractions (wet basis)
Mavg_ss = np.sum(ymol_ss*MWv[:, None], axis=0)
rho_g_ss = P_ss*1e5*Mavg_ss/(Rg*1000.0*T_ss)
Uz_ss = G0/rho_g_ss
F_ss = C_ss*Uz_ss*Omega*3600.0                        # molar flows, kmol/h
X_CH4 = 1.0 - (C_ss[0]*Uz_ss)/(C0[0]*Uz0)             # Eq.(90) with flat profiles

print(f"CH4 conversion at outlet: {X_CH4[-1]:.4f}  (eta = {ETA}, Nz = {Nz})")
print(f"Outlet temperature: {T_ss[-1]:.1f} K    Outlet pressure: {P_ss[-1]:.2f} bar")
print(f"{'species':8s}{'y_in':>9s}{'y_out':>9s}{'F_in kmol/h':>13s}{'F_out kmol/h':>14s}")
for i, s in enumerate(species):
    print(f"{s:8s}{ymol_ss[i, 0]:9.4f}{ymol_ss[i, -1]:9.4f}{F_ss[i, 0]:13.3f}{F_ss[i, -1]:14.3f}")

# How flat would the real temperature profile be?  Radial Biot number
# Bi = U*r_t/lam_r,eff : the flat-T assumption is exact only for Bi -> 0.
mu_ss = mu_mix(T_ss, ymol_ss)
lamg_ss = lambda_mix(T_ss, ymol_ss)
Cpg_ss = cp_mix_mass(T_ss, ymol_ss)
U_ss = wall_htc(T_ss, Uz_ss, mu_ss, lamg_ss, rho_g_ss, Cpg_ss)
lam_r_ss = eff_radial_thermal_conductivity(T_ss, Uz_ss, mu_ss, lamg_ss, rho_g_ss, Cpg_ss)
Bi_r = U_ss*rt/lam_r_ss
print(f"Radial Biot number U*r_t/lam_r,eff along the tube: {Bi_r.min():.2f} to {Bi_r.max():.2f}")

# ============================================================================
# 9. PLOTS: pressure, temperature and composition vs reactor length
# ============================================================================
SURFACE, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, LINE = '#e1e0d9', '#c3c2b7', '#2a78d6'
STYLE = {
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.labelcolor': INK2,
    'axes.titlecolor': INK, 'axes.titlesize': 11, 'axes.titlelocation': 'left',
    'xtick.color': AXIS, 'ytick.color': AXIS, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'grid.linestyle': '-',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True,
    'lines.linewidth': 1.8, 'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round',
    'font.size': 10,
}
LABELS = {'CH4': 'CH$_4$', 'CO': 'CO', 'CO2': 'CO$_2$', 'H2': 'H$_2$', 'H2O': 'H$_2$O', 'N2': 'N$_2$'}

def plot_profile(ax, x, y, end_text, side='auto'):
    """Profile line with an end marker and its outlet value on the free side."""
    ax.plot(x, y, color=LINE)
    ax.plot(x[-1], y[-1], 'o', ms=6, color=LINE, mec=SURFACE, mew=1.5, zorder=3, clip_on=False)
    above = (y[-1] >= y[-4]) if side == 'auto' else (side == 'above')
    ax.annotate(end_text, (x[-1], y[-1]), xytext=(-6, 7 if above else -7),
                textcoords='offset points', ha='right', va='bottom' if above else 'top',
                color=INK, fontsize=9)
    ax.set_xlim(0.0, L)
    ax.margins(y=0.08)

with plt.rc_context(STYLE):
    # ---- pressure and temperature ----
    fig1, (axP, axT) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    plot_profile(axP, z, P_ss, f"outlet {P_ss[-1]:.2f} bar")
    axP.set_ylabel('Pressure (bar)')
    axP.set_title('Pressure')
    plot_profile(axT, z, T_ss, f"outlet {T_ss[-1]:.1f} K", side='below')
    axT.axhline(Tw, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    axT.annotate(f'tube wall, T$_w$ = {Tw:.0f} K', (0.2, Tw), xytext=(0, -5),
                 textcoords='offset points', va='top', color=INK2, fontsize=9)
    axT.set_ylabel('Temperature (K)')
    axT.set_title('Temperature')
    axT.set_xlabel('Reactor length (m)')
    fig1.suptitle(f'Flat-profile model (flat u, C, T; axial + radial dispersion), $\\eta$ = {ETA}',
                  x=0.02, ha='left', color=INK, fontsize=12)
    fig1.tight_layout()

    # ---- composition of each species ----
    fig2, axs2 = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for i, s in enumerate(species):
        ax = axs2.flat[i]
        plot_profile(ax, z, ymol_ss[i], f"outlet {ymol_ss[i, -1]:.3f}")
        ax.set_title(LABELS[s])
        if i % 3 == 0:
            ax.set_ylabel('Mole fraction (-)')
        if i >= 3:
            ax.set_xlabel('Reactor length (m)')
    fig2.suptitle(f'Composition along the tube, flat-profile model, $\\eta$ = {ETA}',
                  x=0.02, ha='left', color=INK, fontsize=12)
    fig2.tight_layout()

    fig1.savefig('flat_model_pressure_temperature.png', dpi=150)
    fig2.savefig('flat_model_composition.png', dpi=150)

print("Execution time:", round(time.time() - start_time, 1), "seconds")
plt.show()