"""

This module calculates the parameters related to semiconductor and devices


"""

import numpy as np
import scipy.constants as sc


def builtin_volt(Nd, Na, ni, T):
    """
    calculate built-in voltages of a pn junction

    :param Nd: doping density of n-layer
    :param Na: doping density of p-layer
    :param ni: intrinsic doping density
    :param T: temperature (K)
    :return: built-in voltage
    """

    return (sc.k * T / sc.e) * np.log(Nd * Na / ni ** 2)


def n_intrinsic(Eg, mEff_e, mEff_h, T):
    """
    Calculate the intrinsic carrier concentration

    References:
    S. M. Sze, Physics of Semiconductor Devices, 3rd, p.18-19
    J. Nelson, The Physics of Solar Cells, p.58

    :param Eg: band gap
    :param mEff_e: effective mass of electron
    :param mEff_h: effective mass of holes
    :param T: temperature
    :return: intrinsic doping density
    """
    Nv = 2 * (mEff_h * sc.k * T / (2 * sc.pi * sc.hbar ** 2)) ** 1.5
    Nc = 2 * (mEff_e * sc.k * T / (2 * sc.pi * sc.hbar ** 2)) ** 1.5
    niSquared = Nc * Nv * np.exp(-Eg / (sc.k * T))

    return np.sqrt(niSquared)



def calc_jnp(V, Vbi, alphaBottom, alphaI, alphaTop, bsInitial, bs_incident_on_top, d_bottom, d_top, energies, T,
             l_bottom, l_top, ni, pn_or_np, s_bottom, s_top, w_bottom, w_top, x_bottom, x_top, xi):

    """
    Calulate electron and hole current densities using analytical equations in Jenny's book

    :param V: voltage
    :param Vbi: built-in voltage
    :param alphaBottom: absorption coefficient of the bottom layer
    :param alphaI: absorption coefficient of the intrinsic layer
    :param alphaTop: absorption coefficient of the top layer
    :param bsInitial: set to None if it is identical to bs_incident_on_top
    :param bs_incident_on_top: incident photon flux density. This can be unity because it will be normalized when calculating QE
    :param d_bottom: diffusion coefficient of bottom layer
    :param d_top: diffusion coefficient of top layer
    :param energies: the range of energy to be calculated
    :param T: temperature in Kelvins
    :param l_bottom: diffusion length of the bottom layer
    :param l_top: diffusion length of the top layer
    :param ni: intrinsic doping density
    :param pn_or_np:
    :param s_bottom: surface recombination velocity of the bottom layer
    :param s_top: surface recombination velocity of the top layer
    :param w_bottom: depletion width of the bottom layer
    :param w_top: depletion width of the top layer
    :param x_bottom: the thickness of the bottom layer
    :param x_top: the thickness of the top layer
    :param xi: the thickness of the intrinsic layer
    :return:
    """

    if bsInitial is None:
        bsInitial=bs_incident_on_top

    kbT=sc.k*T

    bs_incident_on_depleted=bs_incident_on_top*np.exp(-alphaTop*(x_top-w_top))
    bs_incident_on_bottom=bs_incident_on_top*np.exp(-alphaTop*x_top-alphaI*xi-alphaBottom*w_bottom)

    harg_bottom = (x_bottom - w_bottom) / l_bottom
    cosh_harg_bottom = np.cosh(harg_bottom)
    sinh_harg_bottom = np.sinh(harg_bottom)
    lsod_top = (l_top * s_top) / d_top
    # n0 = niSquared/Na
    harg_top = (x_top - w_top) / l_top
    sinh_harg_top = np.sinh(harg_top)
    cosh_harg_top = np.cosh(harg_top)
    lsod_bottom = (l_bottom * s_bottom) / d_bottom
    # p0 = niSquared/Nd
    # print ("lsod_bottom",lsod_bottom)
    # print ("lspd_top",lspd_top)
    # j_top= (q*bs_incident_on_top*alpha*l_bottom)/(alpha**2*l_bottom**2-1)* \
    #     ((lsod_bottom+ alpha*l_bottom - exp(-alpha*(x_top-w_top)) * (lsod_bottom*cosh_harg_top+sinh_harg_top))      /(lsod_bottom*sinh_harg_top+cosh_harg_top)
    #     - alpha*l_bottom* exp(-alpha*(x_top-w_top))
    # )
    # j_bottom= q*(bs_incident_on_bottom*alpha*l_top/(alpha**2*l_top**2-1) \
    #     *(l_top*alpha -
    #     ((lspd_top*(-exp(-alpha*(x_bottom-w_bottom))+cosh_harg_bottom))+sinh_harg_bottom + (l_top*alpha)*exp(-alpha*(x_bottom-w_bottom)))     /(cosh_harg_bottom +lspd_top*sinh_harg_bottom))
    # )
    j_top = (sc.e * bs_incident_on_top * alphaTop * l_top) / (alphaTop ** 2 * l_top ** 2 - 1) * \
            ((lsod_top + alphaTop * l_top - np.exp(-alphaTop * (x_top - w_top)) * (
                lsod_top * cosh_harg_top + sinh_harg_top)) / (lsod_top * sinh_harg_top + cosh_harg_top)
             - alphaTop * l_top * np.exp(-alphaTop * (x_top - w_top))
             )
    j_bottom = sc.e * (bs_incident_on_bottom * alphaBottom * l_bottom / (alphaBottom ** 2 * l_bottom ** 2 - 1) \
                    * (l_bottom * alphaBottom -
                       ((lsod_bottom * (
                           -np.exp(-alphaBottom * (x_bottom - w_bottom)) + cosh_harg_bottom)) + sinh_harg_bottom + (
                            l_bottom * alphaBottom) * np.exp(-alphaBottom * (x_bottom - w_bottom))) / (
                           cosh_harg_bottom + lsod_bottom * sinh_harg_bottom))
                    )
    # jgen=q*bs_incident_on_depleted*(1 - exp(-alpha*(depleted_width)))    # jgen. Jenny, p. 159

    jgen = sc.e * bs_incident_on_depleted * (1 - np.exp(-alphaI * xi - alphaTop * w_top - alphaBottom * w_bottom))  # jgen. Jenny, p. 159
    # hereby I define the subscripts to refer to the layer in which the current is generated:
    if pn_or_np == "pn":
        jn, jp = j_bottom, j_top
    else:
        jp, jn = j_bottom, j_top

    # Jrec. Jenny, p.159. Note capital J. This does not need integrating over energies
    lifetime_top = l_top ** 2 / d_top  # these might not be the right lifetimes. Check ###
    lifetime_bottom = l_bottom ** 2 / d_bottom  # Jenny p163
    Jrec = sc.e * ni * (w_top + w_bottom + xi) / np.sqrt(lifetime_top * lifetime_bottom) * np.sinh(sc.e * V / (2 * kbT)) / (
        sc.e * (Vbi - V) / kbT) * sc.pi
    # jgen = q* bs*(1 - exp(-depleted_width*alpha))*exp(-(xn-wn)*alpha); print ("Ned hack")
    #nDepletionCharge = wn * Nd * q
    #pDepletionCharge = wp * Na * q
    #Vbi2 = (0.5 * (wn + wp) + xi) * pDepletionCharge / (es)



    # Clean the calcualted results
    good_indeces = np.isfinite(jn) * np.isfinite(jp) * np.isfinite(jgen)
    jn[jn < 0] = 0
    jp[jp < 0] = 0
    jgen[jgen < 0] = 0
    energies = energies[good_indeces]
    jn = jn[good_indeces]
    jp = jp[good_indeces]
    jgen = jgen[good_indeces]
    bsInitial = bsInitial[good_indeces]
    Jn = np.trapz(y=jn, x=energies)
    Jp = np.trapz(y=jp, x=energies)
    Jgen = np.trapz(y=jgen, x=energies)


    return Jgen, Jn, Jp, Jrec, bsInitial, energies, jgen, jn, jp

