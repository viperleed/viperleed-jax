from functools import lru_cache
import numpy as np
from scipy.special import sph_harm, spherical_jn

@lru_cache(maxsize=None)
def fac(n):
    return n * fac(n-1) if n else 1


# dimension of array belm as function of lmax
nlmbs = (19, 126, 498, 1463, 3549, 7534, 14484, 25821, 43351, 69322,
         106470, 158067, 227969, 320664, 441320, 595833, 790876, 1033942)

# this function should not be JIT

#  BELMG computes Clebsh-Gordon coefficients to be used in tmatrix subroutine
def get_clebsh_gordon(lmax):
    nlmbs = (19, 126, 498, 1463, 3549, 7534, 14484, 25821, 43351, 69322,
             106470, 158067, 227969, 320664, 441320, 595833, 790876, 1033942)
    nf = 4*lmax+2
    belm = np.full(shape=(nlmbs[lmax-1]), fill_value=np.nan, dtype=np.float64)

    k = 0
    for l in range(lmax+1):
        for lp in range(lmax+1):
            ll2 = l+lp
            for m in range(-l, l+1):
                pre = 4*np.pi*(-1)**m
                for mp in range(-lp, lp+1):
                    mpp = mp-m
                    impp = abs(mpp)
                    ll1 = abs(l-lp)

                    ll1 = max(impp, ll1)

                    for lpp in range(ll2, ll1-1, -2):  # not sure about this...
                        belm[k] = pre*blm2(l, m, lpp, mpp, lp, -mp, ll2)
                        k += 1
    return belm

# this function should not be jited

def blm2(l1, m1, l2, m2, l3, m3, lmax):
    """provides the integral of the product of three spherical harmonics, each of which can be expressed as a prefactor
    times a legendre function. The three prefactors are lumped togetheras a factor C and the integral of the three
    legendre functions follow gaunts summation scheme set out by Slater.
    Author Pendry"""
    # easy checks first
    if (m1 + m2 + m3 != 0):
        return 0
    if (l1 - 2*lmax > 0) or (l2 - lmax > 0) or (l3 - lmax > 0):
        raise RuntimeError  # invalid l > lmax
    if (l1 - abs(m1) < 0) or (l2 - abs(m2) < 0) or (l3 - abs(m3) < 0):
        raise RuntimeError  # invalid quantum number m
    if ((l1+l2+l3) % 2 != 0):
        return 0

    nl = np.array((l1, l2, l3))
    nm = np.array((abs(m1), abs(m2), abs(m3)))
    ic = (np.sum(nm))/2

    #sort by size
    maxnm_id = np.argmax(nm)
    nl[0], nl[maxnm_id] = nl[maxnm_id], nl[0]
    nm[0], nm[maxnm_id] = nm[maxnm_id], nm[0]
    if (nl[2] > nl[1]):
        nl[1], nl[2] = nl[2], nl[1]
        nm[1], nm[2] = nm[2], nm[1]
    if nl[2] - abs(nl[1]-nl[0]) < 0:
        return 0

    nl1, nl2, nl3 = nl[0], nl[1], nl[2]
    nm1, nm2, nm3 = nm[0], nm[1], nm[2]
    # Faktor A

    iss = int(np.sum(nl)/2)
    ia1 = iss - nl2 - nm3
    ia2 = nl2 + nm2
    ia3 = nl2 - nm2
    ia4 = nl3 + nm3
    ia5 = nl1 + nl2 - nl3
    ia6 = iss - nl1
    ia7 = iss - nl2
    ia8 = iss - nl3
    ia9 = nl1 + nl2 + nl3 + 1

    A = ((-1.0)**ia1)/np.math.factorial(ia3)*np.math.factorial(ia2)/np.math.factorial(ia6)*np.math.factorial(ia4)
    A = A/np.math.factorial(ia7)*np.math.factorial(ia5)/np.math.factorial(ia8)*np.math.factorial(iss)/np.math.factorial(ia9)

    # Faktor B

    ib1 = nl1 + nm1
    ib2 = nl2 + nl3 - nm1
    ib3 = nl1 - nm1
    ib4 = nl2 - nl3 + nm1
    ib5 = nl3 - nm3
    it1 = max(0, - ib4) + 1
    it2 = min(ib2, ib3, ib5) + 1
    B = 0.
    sign = (- 1.0)**(it1)
    ib1 = ib1 + it1 - 2
    ib2 = ib2 - it1 + 2
    ib3 = ib3 - it1 + 2
    ib4 = ib4 + it1 - 2
    ib5 = ib5 - it1 + 2
    for it in range(it1, it2+1):
        sign = - sign
        ib1 = ib1 + 1
        ib2 = ib2 - 1
        ib3 = ib3 - 1
        ib4 = ib4 + 1
        ib5 = ib5 - 1
        bn = sign/np.math.factorial(it-1)*np.math.factorial(ib1)/np.math.factorial(ib3) * \
            np.math.factorial(ib2)/np.math.factorial(ib4)/np.math.factorial(ib5)
        B += bn

    # Faktor C
    ic1 = nl1 - nm1
    ic2 = nl1 + nm1
    ic3 = nl2 - nm2
    ic4 = nl2 + nm2
    ic5 = nl3 - nm3
    ic6 = nl3 + nm3
    cn = float((2 * nl1 + 1) * (2 * nl2 + 1) * (2 * nl3 + 1))/np.pi
    C = cn/np.math.factorial(ic2)*np.math.factorial(ic1)/np.math.factorial(ic4)*np.math.factorial(ic3)/np.math.factorial(ic6)*np.math.factorial(ic5)

    C = (np.sqrt(C))/2
    return float((-1)**ic)*A*B*C


def cppp(n1, n2, n3):
    """tapulates the function PPP(I1,I2,I3), each element containing the integral of the product of threee legendre
    functions P(I1),P(I2),P(I3). The integrals are calculated following Gaunt's summation scheme set out by Slater
    atomic structure.
    PPP is used by function PSTEMP in computing temperature-depending phase shifts.
    Author Pendry"""
    ppp = np.full(shape=(n1, n2, n3), fill_value=np.nan)
    for i1 in range(1, n1+1):
        for i2 in range(1, n2+1):
            for i3 in range(1, n3+1):
                im1, im2, im3 = sorted((i1, i2, i3), reverse=True)
                A = 0.0
                iss = i1 + i2 + i3 - 3
                if (iss % 2 == 1) or (abs(im2-im1)+1 > im3):
                    ppp[i1-1, i2-1, i3-1] = A
                else:
                    ssum = 0.0
                    iss = int(iss/2)
                    sign = 1.0
                    for it in range(1, im3+1):
                        sign = - sign
                        ia1 = im1 + it - 1
                        ia2 = im1 - it + 1
                        ia3 = im3 - it + 1
                        ia4 = im2 + im3 - it
                        ia5 = im2 - im3 + it
                        ssum -= sign*np.math.factorial(ia1-1)*np.math.factorial(ia4-1) / \
                            (np.math.factorial(ia2-1)*np.math.factorial(ia3-1)*np.math.factorial(ia5-1)*np.math.factorial(it-1))
                    ia1 = 2 + iss - im1
                    ia2 = 2 + iss - im2
                    ia3 = 2 + iss - im3
                    ia4 = 3 + 2 * (iss - im3)
                    A = - (-1)**(iss-im2)*fac(ia4-1)*fac(iss)*fac(im3-1) * \
                        ssum/(np.math.factorial(ia1-1)*np.math.factorial(ia2-1)*np.math.factorial(ia3-1)*np.math.factorial(2*iss+1))
                    ppp[i1-1, i2-1, i3-1] = A
    return ppp


# need to find a better way to do this; not available in JAX yet
def bessel(z, n1):
    """spherical besser functions. evaluated at z, up to degree n1"""
    bj = np.empty(shape=(n1,), dtype=np.complex128)

    for i in range(n1):
        bj[i] = spherical_jn(i, z)
    return bj

def HARMONY(C, LMAX, LMMAX):
    """generates the spherical harmonics for the vector C"""
    YLM = np.full((LMMAX,), dtype=np.complex128, fill_value=np.nan)
    r = np.sqrt(C[0] ** 2 + C[1] ** 2 + C[2] ** 2)
    theta = np.arccos(C[0] / r)
    phi = np.arctan2(C[2], C[1])
    i = 0
    for l in range(0, LMAX + 1):
        for m in range(-l, l + 1):
            YLM[i] = sph_harm(m, l, theta, phi)
            i += 1
    return YLM

"""

def HARMONY(C,LMAX,LMMAX):
    CD = C[1]*C[1] + C[2]*C[2]
    YA = np.sqrt(CD + C[0]*C[0])
    B = 0
    CF = 1 + 0j
    if CD > 1.0e-7:
        B = np.sqrt(CD)
        CF = C[1]/B + 1.0j*C[2]/B
    CT = C[0]/YA
    ST = B/YA
    YLM, FAC1, FAC2, FAC3 = SPHRM4(LMAX, LMMAX, CT,ST,CF)
    return YLM, FAC1, FAC2, FAC3


def SPHRM4(LMAX,LMMAX,CT, ST, CF):
    LM = 0
    CL = 0
    A = 1
    B = 1
    ASG = 1
    LL = LMAX + 1
    FAC1 = np.full((LMAX+1,),dtype=np.float64,fill_value=np.nan)
    FAC3 = np.full((LMAX+1,),dtype=np.float64,fill_value=np.nan)
    FAC2 = np.full((LMMAX,),dtype=np.float64,fill_value=np.nan)
    for L in range(1,LL+1):
        FAC1[L-1] = ASG*np.sqrt((2*CL+1)*A/(4*np.pi*B*B))
        FAC3[L-1] = np.sqrt(2*CL)
        CM = -CL
        LN = 2*L - 1
        for M in range(1,LN+1):
            LO = LM + M
            FAC2[LO-1] = np.sqrt((CL+1+CM)*(CL+1-CM)/((2*CL+3)*(2*CL+1)))
            CM += 1
        CL += 1
        A *= 2*CL*(2*CL-1)/4
        B *= CL
        ASG = -ASG
        LM += LN
    LM = 1
    CL = 1
    ASG = -1
    SF = CF
    SA = 1+0j
    YLM = np.full((LMMAX,),dtype=np.complex128,fill_value=np.nan)
    YLM[0] = FAC1[0] + 0.0j
    for L in range(1,LMAX+1):
        LN = LM + 2*L + 1
        YLM[LN-1] = FAC1[L]*SA*SF*ST
        YLM[LM] = ASG*FAC1[L]*SA*ST/SF
        YLM[LN-2] = -FAC3[L]*FAC1[L]*SA*SF*CT/CF
        YLM[LM+1] = ASG*FAC3[L]*FAC1[L]*SA*CT*CF/SF
        SA = ST * SA
        SF = SF * CF
        CL += 1
        ASG = -ASG
        LM = LN
    LM = 1
    LL = LMAX-1
    for L in range(1,LL+1):
        LN = 2*L-1
        LM2 = LM + LN + 4
        LM3 = LM - LN
        for M in range(1,LN+1):
            LO = LM2 + M
            LP = LM3 + M
            LQ = LM + M + 1
            YLM[LO-1]=-(FAC2[LP-1]*YLM[LP-1] - CT*YLM[LQ-1])/FAC2[LQ-1]
        LM += 2*L + 1
    return YLM, FAC1, FAC2, FAC3


def bessel2(z,n1):
    bj = np.empty(shape=(n1,),dtype=np.complex128)
    ZSQ = z*z/2
    PRE = 1
    for L in range(n1):
        TERM = 1
        SUM = 1
        K = 1
        while(True):
            TERM *= -(ZSQ/(K*(2*L+2*K+1)))
            SUM += TERM
            K += 1
            if abs(TERM) < 1e-16:
                break
        bj[L] = PRE*SUM
        PRE *= z/(2*(L+1) + 1)
    return bj
"""
