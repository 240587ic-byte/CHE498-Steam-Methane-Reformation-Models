import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags, kron, identity, csr_matrix
import matplotlib
import matplotlib.pyplot as plt

# ============================================================================
# SIMPLIFIED MODEL (flat velocity / flat concentration / flat temperature
# profile, axial dispersion retained), derived directly from the M5 model
# supplied by the user. Every number below is copied unchanged from that
# file's Tables 2, A.1, C.1-C.5, B.1-B.3 data blocks - nothing has been
# invented or pulled from any outside source.
#
# WHAT "FLAT PROFILE" MEANS FOR THE RADIAL TERMS
# ------------------------------------------------------------------------
# M5 solves C(z,r), T(z,r), P(z,r) on a 2-D (axial x radial) grid, with a
# radial dispersion term (Eq. 41/42, radial bracket) and a Neumann wall BC.
# Collapsing the radial profile to "flat" (i.e. C, T, P independent of r)
# removes the radial coordinate entirely, so:
#   - Species: M5's wall BC is dC/dr|wall = 0 (Eq. 49, no radial species
#     flux through the tube wall). For a field that does not depend on r,
#     d^2C/dr^2 = 0 everywhere anyway, so the radial-dispersion term in
#     Eq.(41) is identically zero here - it does not disappear because we
#     dropped it, it disappears because the assumption "flat profile" forces
#     it to zero. Nothing about species transport is lost: axial dispersion
#     (Eq. B.11-B.14, kept below unchanged) is unaffected.
#   - Temperature: M5's radial term carries heat from the tube wall
#     (T_wall = Tw) to the bulk gas through the wall BC dT/dr|wall =
#     U(z)*(Tw-T)/lambda_r_eff (Eq. 53), using the SAME wall heat-transfer
#     coefficient U(z) that M5 already computes from Eq.(B.31). With a flat
#     radial profile there is no lambda_r_eff-driven gradient to resolve, so
#     that same U(z) is instead applied directly as a volumetric wall-heat
#     term (2*U(z)/r_t)*(Tw-T) in the 1-D energy balance below. This is the
#     standard way to reduce a 2-D wall-cooled/heated packed-bed balance to
#     1-D once the radial profile is assumed flat, and it uses ONLY the
#     wall-htc correlation Eq.(B.31) that was already present in the M5
#     code - no new coefficient or outside formula is introduced.
# So "axial and radial dispersion considered" becomes, under a flat
# profile: axial dispersion solved explicitly (species + energy), and
# radial transport represented through the wall heat-transfer coefficient
# (its only remaining physical role once C, T no longer vary with r).
# ============================================================================

# ============================================================================
# 1. OPERATING / GEOMETRIC DATA  (Table 2)  -- copied unchanged from M5
# ============================================================================
Rg      = 8.314                # J/mol/K
Rg_bar  = 0.08314               # bar.m3/(kmol.K)
Rg_kJ   = 8.314e-3              # kJ/mol/K

L       = 12.0                  # m,    reactor length                 Table 2
dt_tube = 0.10                   # m,    tube inner diameter            Table 2
rt      = dt_tube/2.0             # m,    tube inner radius
dp      = 0.01                    # m,    catalyst particle diameter     Table 2
rho_p   = 2355.2                   # kg/m3, catalyst density             Table 2
Cp_p    = 950.0                     # J/(kg.K), catalyst heat capacity   Table 2
lam_p   = 0.3489                     # W/(m.K), catalyst solid conductivity Table 2
T0      = 793.15                      # K,   inlet temperature           Table 2
P0      = 25.69                        # bar, inlet pressure             Table 2
Tw      = 1000.0                        # K,  constant tube-wall T

species = ['CH4', 'CO', 'CO2', 'H2', 'H2O', 'N2']
MW = {'CH4':16.0428,'CO':28.0104,'CO2':44.0098,'H2':2.01588,'H2O':18.0153,'N2':28.0135}   # Table C.1

F0_h = {'CH4':5.17,'CO':0.0,'CO2':0.29,'H2':0.63,'H2O':17.35,'N2':0.85}                    # kmol/h, Table 2
F0   = {k: v/3600.0 for k, v in F0_h.items()}                                               # -> kmol/s

# Bed voidage, Eq.(3)
eps_b = 0.38 + 0.073*(1.0 - (dt_tube/dp - 2.0)**2 / (dt_tube/dp)**2)

# Cross-sectional area, Eq.(7)
Omega = np.pi*dt_tube**2/4.0

# Inlet superficial velocity, Eq.(8)
F0_tot = sum(F0.values())
Uz0 = F0_tot*Rg_bar*T0/(Omega*P0)                 # m/s
rho_g0 = sum(F0[s]*MW[s] for s in species)/(Uz0*Omega)   # kg/m3
G0 = rho_g0*Uz0        # kg/(m2.s), constant mass flux (flat velocity profile => G(z)=G0 everywhere)

# Inlet concentrations, Eq.(6) rearranged
C0 = np.array([F0[s]/(Uz0*Omega) for s in species])       # kmol/m3

ETA = 1.0        # effectiveness factor (pseudo-homogeneous), Sec. 4.1.2 / Eq.(1)

MWv = np.array([MW[s] for s in species])

# ============================================================================
# 2. KINETIC PARAMETERS - Xu & Froment (1989), Appendix A, Table A.1 -- unchanged
# ============================================================================
Ak  = {1: 4.225e15/3600.0, 2: 1.955e6/3600.0, 3: 1.020e15/3600.0}   # kmol/(kgcat.s)
E_a = {1: 240.1, 2: 67.13, 3: 243.9}                                 # kJ/mol
AK  = {1: 4.707e12, 2: 1.142e-2, 3: 5.375e10}
dH298 = {1: 206.1, 2: -41.15, 3: 164.9}                              # kJ/mol
dH948 = {1: 224.0, 2: -37.3, 3: 187.5}                               # kJ/mol
AKi = {'CO':8.23e-5, 'H2':6.12e-9, 'CH4':6.65e-4, 'H2O':1.77e5}      # bar^-1 (H2O dimensionless)
dHi = {'CO':-70.65, 'H2':-82.90, 'CH4':-38.28, 'H2O':88.68}          # kJ/mol

# Heat-capacity polynomial coefficients (J/mol/K), Table C.5, Eq.(C.10)
CP = {
    'CH4': (34.942, -3.9957e-2, 1.9184e-4, -1.5303e-7, 3.9321e-11),
    'CO' : (29.556, -6.5807e-3, 2.0130e-5, -1.2227e-8, 2.2617e-12),
    'CO2': (27.437,  4.2315e-2,-1.9555e-5,  3.9968e-9,-2.9872e-13),
    'H2' : (25.399,  2.0178e-2,-3.8549e-5,  3.1880e-8,-8.7585e-12),
    'H2O': (33.933, -8.4186e-3, 2.9906e-5, -1.7825e-8, 3.6934e-12),
    'N2' : (29.342, -3.5395e-3, 1.0076e-5, -4.3116e-9, 2.5935e-13),
}
# Reaction stoichiometry (nu_ij), Eq.(41) & Eq.(A.7)
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

# ============================================================================
# 3. PHYSICAL-PROPERTY / TRANSPORT FUNCTIONS (Appendices B & C) -- unchanged
#    from M5, written elementwise/broadcasting-generic so the exact same
#    functions apply to a 1-D (Nz,) field instead of a 2-D (Nr,Nz) field.
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
    return (len(species),) + (1,)*np.ndim(T)

def mu_mix(T, ymol):
    """Gas mixture viscosity, Eq.(C.1)-(C.2) -> Pa.s"""
    mus = np.array([mu_i(T, s) for s in species])
    out = np.zeros_like(T)
    for i, si in enumerate(species):
        phij = np.sqrt(MWv/MW[si])
        denom = np.sum(ymol*phij.reshape(_bshape(T)), axis=0)
        out += ymol[i]*mus[i]/denom
    return out*1e-7   # micropoise -> Pa.s

def lambda_mix(T, ymol):
    """Gas mixture thermal conductivity, Eq.(C.4)-(C.7) -> W/m/K"""
    lams = np.array([lam_i(T, s) for s in species])
    Tr = {s: T/TC[s] for s in species}
    gam = {s: 210.0*(TC[s]*MW[s]**3/PC[s]**4)**(1.0/6.0) for s in species}
    ltr = {s: gam[s]*(np.exp(0.0464*Tr[s]) - np.exp(-0.2412*Tr[s])) for s in species}
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
    """Gas mixture heat capacity, Eq.(C.9) (molar) -> mass basis [J/kg/K]"""
    cps = np.array([cp_i_molar(T, s) for s in species])
    cp_molar = np.sum(ymol*cps, axis=0)
    Mavg = np.sum(ymol*MWv.reshape(_bshape(T)), axis=0)
    return cp_molar*1000.0/Mavg

def binary_diff(T, P_bar, si, sj):
    """Eq.(B.19)-(B.22): binary diffusivity D_ij [cm2/s]"""
    Mij = 2.0/(1.0/MW[si] + 1.0/MW[sj])
    sij = 0.5*(LJ_SIGMA[si] + LJ_SIGMA[sj])
    x = T/np.sqrt(LJ_EPSK[si]*LJ_EPSK[sj])
    OmegaD = (1.06036/x**0.15610 + 0.19300/np.exp(0.47635*x) +
              1.03587/np.exp(1.52996*x) + 1.76474/np.exp(3.89411*x))
    return 0.00266*T**1.5/(P_bar*np.sqrt(Mij)*sij**2*OmegaD)     # cm2/s

def Dim_mix(T, P_bar, ymol):
    """Molecular diffusivity of each species in the mixture (Blanc's law), Eq.(B.18) -> m2/s"""
    Dim = np.zeros(_bshape(T)[:1] + T.shape)
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
    """Effective AXIAL mass-dispersion coefficient, Eq.(B.11)-(B.14) -> m2/s"""
    Dim = Dim_mix(T, P_bar, ymol)
    uz_int = Uz/eps_b
    De = 0.78*Dim + (0.54*uz_int*dp/eps_b) / (1.0 + 9.2*Dim/np.maximum(uz_int*dp/eps_b, 1e-30))
    return De          # Eq.(B.14)

def eff_axial_thermal_conductivity(T, Uz, mu_g, lam_g, rho_g, Cp_g):
    """Effective AXIAL thermal conductivity, Eq.(B.24),(B.26),(B.28)-(B.30) -> W/m/K"""
    Re = dp*Uz*rho_g/mu_g
    Pr = Cp_g*mu_g/lam_g
    Bfac = 1.25*((1.0 - eps_b)/eps_b)**(10.0/9.0)
    ratio = lam_g/lam_p
    term = ((1.0 - ratio)*Bfac/(1.0 - ratio*Bfac)**2)*np.log(1.0/np.maximum(ratio*Bfac, 1e-30)) \
           - (Bfac + 1.0)/2.0 - (Bfac - 1.0)/(1.0 - ratio*Bfac)
    lam_zp = (1.0 - np.sqrt(1.0 - eps_b))*lam_g + 2.0*lam_g*np.sqrt(1.0 - eps_b)/(1.0 - ratio*Bfac)*term
    Peh_z_inv = 0.5/(1.0 + 9.7*eps_b/np.maximum(Re*Pr, 1e-30)) + (0.73*eps_b + lam_zp/lam_g)/np.maximum(Re*Pr, 1e-30)
    return rho_g*Uz*Cp_g*dp*Peh_z_inv         # Eq.(B.24)

def wall_htc(T, Uz, mu_g, lam_g, rho_g, Cp_g):
    """Overall (tube-wall to bulk gas) heat-transfer coefficient U(z), Eq.(B.31) -> W/m2/K.
    Used here (flat radial profile) as the volumetric wall-heat term's coefficient,
    since there is no separate wall-adjacent node anymore - see header note."""
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
    """Xu & Froment LHHW kinetics, Eq.(A.1)-(A.8), Table A.1."""
    k = {j: Ak[j]*np.exp(-E_a[j]/(Rg_kJ*T)) for j in (1, 2, 3)}
    K = {j: AK[j]*np.exp(-dH948[j]/(Rg_kJ*T)) for j in (1, 2, 3)}
    Ki = {s: AKi[s]*np.exp(-dHi[s]/(Rg_kJ*T)) for s in ('CO', 'H2', 'CH4', 'H2O')}
    H2 = np.maximum(Pp['H2'], 1e-8)
    DEN = 1.0 + Ki['CO']*Pp['CO'] + Ki['H2']*H2 + Ki['CH4']*Pp['CH4'] + Ki['H2O']*Pp['H2O']/H2
    R1 = (k[1]/H2**2.5)*(Pp['CH4']*Pp['H2O'] - H2**3*Pp['CO']/K[1])/DEN**2
    R2 = (k[2]/H2)     *(Pp['CO'] *Pp['H2O'] - H2*Pp['CO2']/K[2])/DEN**2
    R3 = (k[3]/H2**3.5)*(Pp['CH4']*Pp['H2O']**2 - H2**4*Pp['CO2']/K[3])/DEN**2
    return R1, R2, R3

# ============================================================================
# 4. AXIAL GRID & FINITE-DIFFERENCE OPERATORS  (1-D: z only, no r)
# ============================================================================
Nz = 200                       # axial nodes (finer than M5's 50 since the
                                # radial dimension - and its cost - is gone)
z = np.linspace(0.0, L, Nz)
dz = z[1] - z[0]
n = Nz - 1     # number of evolved (non-inlet) axial points; z=0 is Dirichlet

def d_dz_backward(f):
    """Backward finite difference in z (matches ACM discretization, Sec. 3)."""
    d = np.empty_like(f)
    d[..., 1:] = (f[..., 1:] - f[..., :-1])/dz
    d[..., 0] = (f[..., 1] - f[..., 0])/dz
    return d

def d2_dz2_central(f):
    """Central 2nd derivative in z with zero-gradient outlet BC, Eq.(47),(51),(59)."""
    d2 = np.zeros_like(f)
    d2[..., 1:-1] = (f[..., 2:] - 2*f[..., 1:-1] + f[..., :-2])/dz**2
    d2[..., -1] = (f[..., -2] - f[..., -1])/dz**2       # ghost node: zero axial gradient
    return d2

tau_P = 0.01   # s, artificial pseudo-transient time constant for the pressure
               # field (same role as in M5), relaxes P(z) toward the local
               # Ergun gradient, Eq.(43)-(44).

# ============================================================================
# 5. STATE PACKING (fields: 6 species C, T, P, each shape (n,) evolved;
#    z=0 fixed by the Dirichlet inlet BC, Eq.(46),(50),(58))
# ============================================================================
nC = 6*n
nT = n
nP = n

def unpack(y):
    C = np.empty((6, Nz))
    T = np.empty(Nz)
    P = np.empty(Nz)
    C[:, 0] = C0     # Eq.(46)
    T[0] = T0        # Eq.(50)
    P[0] = P0        # Eq.(58)
    C[:, 1:] = y[:nC].reshape(6, n)
    T[1:] = y[nC:nC+nT]
    P[1:] = y[nC+nT:nC+nT+nP]
    return C, T, P

# ============================================================================
# 6. ODE RIGHT-HAND SIDE (Eq. 41: species, Eq. 42: energy, Eq. 43-44: pressure,
#    all with the radial terms collapsed as described in the header note)
# ============================================================================
def rhs(t, y):
    C, T, P = unpack(y)
    Ctot = np.sum(C, axis=0)
    ymol = C/Ctot                                              # mole fractions, shape (6,Nz)
    Mavg = np.sum(ymol*MWv.reshape(_bshape(T)), axis=0)         # shape (Nz,)

    R_kmol = Rg*1000.0
    rho_g = P*1e5*Mavg/(R_kmol*T)                                # Eq.(11)
    Uz = G0/rho_g                                                # flat velocity profile: G(z)=G0=const
    P_i = ymol*P[None, ...]                                      # partial pressures (bar)
    Pp = {s: P_i[i] for i, s in enumerate(species)}
    R1, R2, R3 = kinetics(T, Pp)
    Rj = np.stack([R1, R2, R3], axis=0)                          # (3,Nz) kmol/(kgcat.s)

    mu_g = mu_mix(T, ymol)
    lam_g = lambda_mix(T, ymol)
    Cp_g = cp_mix_mass(T, ymol)                                  # J/kg/K

    Dax = eff_axial_mass_dispersion(T, P, ymol, Uz, mu_g, rho_g)          # (6,Nz) m2/s
    lam_z_eff = eff_axial_thermal_conductivity(T, Uz, mu_g, lam_g, rho_g, Cp_g)
    Uwall = wall_htc(T, Uz, mu_g, lam_g, rho_g, Cp_g)            # Eq.(B.31), bulk = wall since flat profile

    dH = np.stack([dHrxn_T(T, 1), dHrxn_T(T, 2), dHrxn_T(T, 3)], axis=0)   # kJ/mol
    dH_Jkmol = dH*1e6                                                       # kJ/mol -> J/kmol

    # ---- species balance, Eq.(41) with radial term = 0 (flat profile) ----
    CU = C*Uz[None, ...]
    dCU_dz = d_dz_backward(CU)
    d2C_dz2 = d2_dz2_central(C)
    rxn_source = (1.0 - eps_b)*rho_p*ETA*np.tensordot(NU_MAT, Rj, axes=([1], [0]))   # (6,Nz)
    dCdt = (-dCU_dz + eps_b*Dax*d2C_dz2 + rxn_source)/eps_b

    # ---- energy balance, Eq.(42) with the radial gradient term replaced by
    #      the equivalent volumetric wall-heat term (2U/r_t)(Tw-T) ----
    dT_dz = d_dz_backward(T)
    d2T_dz2 = d2_dz2_central(T)
    dC_dz = d_dz_backward(C)

    Cp_i_molar = np.array([cp_i_molar(T, s) for s in species])*1000.0        # J/mol/K -> J/kmol/K
    heat_disp_axial = np.sum(eps_b*Dax*dC_dz*Cp_i_molar, axis=0)*dT_dz
    rxn_heat = (1.0 - eps_b)*rho_p*ETA*np.sum(Rj*(-dH_Jkmol), axis=0)
    wall_heat = (2.0*Uwall/rt)*(Tw - T)                                       # volumetric wall exchange, W/m3

    conv_heat = rho_g*Cp_g*Uz*dT_dz
    cap = (1.0 - eps_b)*rho_p*Cp_p + eps_b*rho_g*Cp_g
    dTdt = (-conv_heat + lam_z_eff*d2T_dz2 + heat_disp_axial + rxn_heat + wall_heat)/cap

    # ---- momentum (pressure), Eq.(43)-(44), Ergun, relaxed via artificial
    #      pseudo-time term (same trick as M5) ----
    dPdz_FD = d_dz_backward(P)
    dPdz_Ergun = -(G0/(rho_g*dp))*((1.0 - eps_b)/eps_b**3) * \
                  (150.0*(1.0 - eps_b)*mu_g/dp + 1.75*G0)/1e5
    dPdt = (dPdz_Ergun - dPdz_FD)/tau_P

    return np.concatenate([dCdt[:, 1:].ravel(), dTdt[1:].ravel(), dPdt[1:].ravel()])

# ============================================================================
# 7. INITIAL CONDITION & TIME INTEGRATION (pseudo-transient continuation)
# ============================================================================
y0 = np.concatenate([
    np.tile(C0[:, None], (1, n)).ravel(),
    np.full(n, T0),
    np.full(n, P0),
])

# ---- Jacobian SPARSITY PATTERN ------------------------------------------
# State is 8 stacked fields (CH4,CO,CO2,H2,H2O,N2,T,P), each shape (n,).
# Every spatial derivative is a 3-point stencil in z, so node z_i's
# equations depend only on z_(i-1)/z_i/z_(i+1) - tridiagonal in z, block-
# diagonal across the 8 fields (no coupling pattern needed across fields
# beyond what's already dense through the reaction/property terms, which
# BDF handles fine given the correct z-bandwidth).
Tri_z = diags([1.0, 1.0, 1.0], offsets=[-1, 0, 1], shape=(n, n), format='csr')
jac_sparsity = kron(csr_matrix(np.ones((8, 8))), Tri_z, format='csr')

t_final = 300.0    # s, several residence times -> steady state
sol = solve_ivp(rhs, [0.0, t_final], y0, method='BDF',
                 rtol=1e-6, atol=1e-9, jac_sparsity=jac_sparsity)

C_ss, T_ss, P_ss = unpack(sol.y[:, -1])

# ============================================================================
# 8. POST-PROCESSING
# ============================================================================
ymol_ss = C_ss/np.sum(C_ss, axis=0)
Mavg_ss = np.sum(ymol_ss*MWv.reshape(_bshape(T_ss)), axis=0)
rho_g_ss = P_ss*1e5*Mavg_ss/(Rg*1000.0*T_ss)      # Eq.(11)
Uz_ss = G0/rho_g_ss                                # flat velocity profile

# CH4 conversion along the axial length, Eq.(90) with the radial integral
# dropped (flat profile => no radial variation to integrate over)
X_CH4 = 1.0 - (C_ss[0]*Uz_ss)/(C0[0]*Uz0)
print(f"CH4 conversion at outlet: {X_CH4[-1]:.4f}  (eta = {ETA}, Nz={Nz})")

# ============================================================================
# 9. PLOTS - pressure, temperature, and species composition vs reactor length
# ============================================================================
fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

axs[0].plot(z, P_ss, 'k-')
axs[0].set_ylabel('Pressure (bar)')
axs[0].set_title('Pressure vs reactor length, Eq.(43)-(44)')

axs[1].plot(z, T_ss, 'r-')
axs[1].set_ylabel('Temperature (K)')
axs[1].set_title('Temperature vs reactor length, Eq.(42)')

colors = {'CH4':'tab:green','CO':'tab:orange','CO2':'tab:red',
          'H2':'tab:blue','H2O':'tab:cyan','N2':'tab:gray'}
for i, s in enumerate(species):
    axs[2].plot(z, ymol_ss[i], label=s, color=colors[s])
axs[2].set_ylabel('Mole fraction (-)')
axs[2].set_xlabel('Reactor length (m)')
axs[2].set_title('Species composition vs reactor length, Eq.(41)')
axs[2].legend(ncol=3, fontsize=9)

plt.tight_layout()

# ---- also show molar concentration (kmol/m3) per species, since "composition"
#      can mean concentration as well as mole fraction ----
fig2, ax2 = plt.subplots(figsize=(8, 5))
for i, s in enumerate(species):
    ax2.plot(z, C_ss[i], label=s, color=colors[s])
ax2.set_xlabel('Reactor length (m)')
ax2.set_ylabel('Concentration (kmol/m3)')
ax2.set_title('Species concentration vs reactor length')
ax2.legend(ncol=3, fontsize=9)
plt.tight_layout()

fig.savefig('/mnt/user-data/outputs/M2_pressure_temperature_composition.png', dpi=150)
fig2.savefig('/mnt/user-data/outputs/M2_concentrations.png', dpi=150)
# NOTE: plt.show() is intentionally NOT called by default - it opens a
# blocking GUI window on machines with a display. Uncomment only if running
# interactively with a display available.
plt.show()