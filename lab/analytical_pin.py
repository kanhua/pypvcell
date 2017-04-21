import os,numpy
from numpy import exp, array, flipud as reverse,abs,sqrt,log,sinh,cosh,isfinite,trapz
from scipy.constants import k, pi, hbar,e
from pypvcell.photocurrent import lambert_abs

kb=k
q=e



def calculate_junction_sr(junc, energies, bs, bsInitial, V, printParameters=False):
    sn = 0 if not hasattr(junc ,"sn") else junc.sn
    sp = 0 if not hasattr(junc ,"sp") else junc.sp
    # print ("sn:",sn)
    pn_or_np = junc[0].material. role +junc[-1].material.role
    # print (junc, len(junc))

    pn_not_np = pn_or_np == "pn"

    if pn_or_np== "pn":
        # print ("p on n"#, pn_or_np)
        if len(junc) == 3:  # assume PIN
            assert junc[0].material.role == "p" and \
                   junc[1].material.role == "i" and \
                   junc[2].material.role == "n", "can only deal with pin and pn junctions right now."

            pRegion = junc[0]
            iRegion = junc[1]
            nRegion = junc[2]
        elif len(junc) == 2:
            assert junc[0].material.role == "p" and \
                   junc[1].material.role == "n", "can only deal with pin and pn junctions right now."
            pRegion = junc[0]
            iRegion = None
            nRegion = junc[1]
    elif pn_or_np == "np":
        # print ("n on p", pn_or_np)

        if len(junc) == 3:  # assume PIN
            assert junc[0].material.role == "n" and \
                   junc[1].material.role == "i" and \
                   junc[2].material.role == "p", "not a nip junction"

            pRegion = junc[2]
            iRegion = junc[1]
            nRegion = junc[0]
        elif len(junc) == 2:
            pRegion = junc[1]
            iRegion = None
            nRegion = junc[0]
    else:
        raise ValueError("Can only calculate analytic QEs of PN,PIN,NIP,NP junctions")

    shift = 0
    T = nRegion.material.T
    kbT = kb * T
    Egap = nRegion.material.band_gap
    xp = pRegion.width
    xn = nRegion.width

    xi = 0 if iRegion is None else iRegion.width

    if hasattr(junc, "dielectric_constant"):
        es = junc.dielectric_constant
    else:
        es = nRegion.material.dielectric_constant  # equal for n and p.  I hope.
    if hasattr(junc, "ln"):
        ln = junc.ln
    else:
        ln = pRegion.material.electron_minority_carrier_diffusion_length  # n refers to electrons

    if hasattr(junc, "lp"):
        lp = junc.lp
    else:
        lp = nRegion.material.hole_minority_carrier_diffusion_length  # p refers to holes

    Na = pRegion.material.Na
    Nd = nRegion.material.Nd

    if hasattr(junc, "mup"):
        dp = junc.mup * kb * T / q
    else:
        dp = pRegion.material.electron_mobility * kb * T / q

    if hasattr(junc, "mun"):
        dn = junc.mun * kb * T / q
    else:
        dn = nRegion.material.hole_mobility * kb * T / q

    # print (dn, dp)


    # dp,dn = dn,dp #### hack
    # print ("hack")
    # print ("*", pRegion.material.electron_mobility)
    print("assuming heavy hole eff_mass_hh_z for QE calculation")
    mEff_h = nRegion.material.eff_mass_hh_z * electron_mass
    mEff_e = pRegion.material.eff_mass_electron * electron_mass
    print(mEff_h, mEff_e)
    # mEff_h,mEff_e = mEff_e,mEff_h
    # print ("hack")



    Nv = 2 * (mEff_h * kb * T / (2 * pi * hbar ** 2)) ** 1.5  # Jenny p58
    Nc = 2 * (mEff_e * kb * T / (2 * pi * hbar ** 2)) ** 1.5
    niSquared = Nc * Nv * exp(-Egap / (kb * T))
    ni = sqrt(niSquared)
    # pn methods:
    # wp = 1/Na*sqrt((2*es*Vbi)/(q*(1/Na+1/Nd)))       # Jenny p151
    # wn = 1/Nd*sqrt((2*es*Vbi)/(q*(1/Na+1/Nd)))
    Vbi = (kb * T / q) * log(Nd * Na / niSquared)  # Jenny p146

    Vbi = (kb * T / q) * log(Nd * Na / niSquared) if not hasattr(junc, "Vbi") else junc.Vbi

    # Vbi = 1.3 ; print ("Vbi hack")
    # pin methods: (see documentation, "PIN.pdf")


    if not hasattr(junc, "wp") or not hasattr(junc, "wn"):

        if hasattr(junc, "depletion_approximation") and junc.depletion_approximation == "one-sided abrupt":
            print("using one-sided abrupt junction approximation for depletion width")
            # science_reference("Sze abrupt junction approximation", "Sze: The Physics of Semiconductor Devices (second edition)")
            wp = sqrt(2 * es * Vbi / (q * Na));
            wn = sqrt(2 * es * Vbi / (q * Nd));
        else:
            print(niSquared, xi, es, "...", Vbi, Na, Nd, (xi ** 2 + 2. * es * Vbi / q * (1 / Na + 1 / Nd)))
            wn = (-xi + sqrt(xi ** 2 + 2. * es * Vbi / q * (1 / Na + 1 / Nd))) / (1 + Nd / Na)
            wp = (-xi + sqrt(xi ** 2 + 2. * es * Vbi / q * (1 / Na + 1 / Nd))) / (1 + Na / Nd)

    wn = wn if not hasattr(junc, "wn") else junc.wn
    wp = wp if not hasattr(junc, "wp") else junc.wp
    print("wn, wp:", wn, wp)
    # bs = array([0.75+0.25*sin(i) for i in range(len(bs))])
    # we have an array of alpha values that needs to be interpolated to the right energies
    alphaN = nRegion.material.alphaE(energies - shift)  # create numpy array at right energies.
    alphaP = pRegion.material.alphaE(energies - shift)  # create numpy array at right energies.
    alphaI = iRegion.material.alphaE(energies - shift) if iRegion else 0
    # print (alphaN, alphaP, alphaI, "alhpa")
    depleted_width = wn + wp + xi
    bs_incident_on_top = bs

    if pn_not_np:

        # bs_incident_on_bottom = bs*exp(-alpha*(xp+wn+xi))
        # bs_incident_on_depleted = bs*exp(-alpha*(xp-wp))
        bs_incident_on_bottom = bs * exp(-alphaP * xp - alphaN * wn - xi * alphaI)
        print(xp - wp)
        bs_incident_on_depleted = bs * exp(-alphaP * (xp - wp))
        alphaTop = alphaP
        alphaBottom = alphaN
    else:
        # bs_incident_on_bottom = bs*exp(-alpha*(xn+wp+xi))
        # bs_incident_on_depleted = bs*exp(-alpha*(xn-wn))
        bs_incident_on_bottom = bs * exp(-alphaN * xn - alphaP * wp - xi * alphaI)
        bs_incident_on_depleted = bs * exp(-alphaN * (xn - wn))
        alphaTop = alphaN
        alphaBottom = alphaP

    if pn_or_np == "pn":
        l_top, l_bottom = ln, lp
        x_top, x_bottom = xp, xn
        w_top, w_bottom = wp, wn
        s_top, s_bottom = sp, sn
        d_top, d_bottom = dp, dn
    else:
        l_bottom, l_top = ln, lp
        x_bottom, x_top = xp, xn
        w_bottom, w_top = wp, wn
        s_bottom, s_top = sp, sn
        d_bottom, d_top = dp, dn

    Jgen, Jn, Jp, Jrec, bsInitial, energies, jgen, jn, jp = calc_jnp(V, Vbi, alphaBottom, alphaI, alphaTop, bsInitial,
                                                                     bs_incident_on_top, d_bottom, d_top, energies, T,
                                                                     l_bottom, l_top, ni, pn_or_np, s_bottom, s_top,
                                                                     w_bottom, w_top, x_bottom, x_top, xi)

    return {
        "qe_n": jn / q / bsInitial,
        "qe_p": jp / q / bsInitial,
        "qe_gen": jgen / q / bsInitial,
        "qe_tot": (jn + jp + jgen) / q / bsInitial,  # ,
        "Jn": Jn,
        "Jp": Jp,
        "Jgen": Jgen,
        "Jrec": Jrec,
        "J": (Jn + Jp + Jgen - Jrec),
        "e": energies,
        "Temporary locals dictionary for radiative efficiency": locals()
    }


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
    :param bs_incident_on_top: incident photon plux density. This can be unity because it will be normalized when calculating QE
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
    kbT=kb*T

    bs_incident_on_depleted=bs_incident_on_top*exp(-alphaTop*(x_top-w_top))
    bs_incident_on_bottom=bs_incident_on_top*exp(-alphaTop*x_top-alphaI*xi-alphaBottom*w_bottom)

    harg_bottom = (x_bottom - w_bottom) / l_bottom
    cosh_harg_bottom = cosh(harg_bottom)
    sinh_harg_bottom = sinh(harg_bottom)
    lsod_top = (l_top * s_top) / d_top
    # n0 = niSquared/Na
    harg_top = (x_top - w_top) / l_top
    sinh_harg_top = sinh(harg_top)
    cosh_harg_top = cosh(harg_top)
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
    j_top = (q * bs_incident_on_top * alphaTop * l_top) / (alphaTop ** 2 * l_top ** 2 - 1) * \
            ((lsod_top + alphaTop * l_top - exp(-alphaTop * (x_top - w_top)) * (
                lsod_top * cosh_harg_top + sinh_harg_top)) / (lsod_top * sinh_harg_top + cosh_harg_top)
             - alphaTop * l_top * exp(-alphaTop * (x_top - w_top))
             )
    j_bottom = q * (bs_incident_on_bottom * alphaBottom * l_bottom / (alphaBottom ** 2 * l_bottom ** 2 - 1) \
                    * (l_bottom * alphaBottom -
                       ((lsod_bottom * (
                           -exp(-alphaBottom * (x_bottom - w_bottom)) + cosh_harg_bottom)) + sinh_harg_bottom + (
                            l_bottom * alphaBottom) * exp(-alphaBottom * (x_bottom - w_bottom))) / (
                           cosh_harg_bottom + lsod_bottom * sinh_harg_bottom))
                    )
    # jgen=q*bs_incident_on_depleted*(1 - exp(-alpha*(depleted_width)))    # jgen. Jenny, p. 159

    jgen = q * bs_incident_on_depleted * (1 - exp(-alphaI * xi - alphaTop * w_top - alphaBottom * w_bottom))  # jgen. Jenny, p. 159
    # hereby I define the subscripts to refer to the layer in which the current is generated:
    if pn_or_np == "pn":
        jn, jp = j_bottom, j_top
    else:
        jp, jn = j_bottom, j_top

    # Jrec. Jenny, p.159. Note capital J. This does not need integrating over energies
    lifetime_top = l_top ** 2 / d_top  # these might not be the right lifetimes. Check ###
    lifetime_bottom = l_bottom ** 2 / d_bottom  # Jenny p163
    Jrec = q * ni * (w_top + w_bottom + xi) / sqrt(lifetime_top * lifetime_bottom) * sinh(q * V / (2 * kbT)) / (
        q * (Vbi - V) / kbT) * pi
    # jgen = q* bs*(1 - exp(-depleted_width*alpha))*exp(-(xn-wn)*alpha); print ("Ned hack")
    #nDepletionCharge = wn * Nd * q
    #pDepletionCharge = wp * Na * q
    #Vbi2 = (0.5 * (wn + wp) + xi) * pDepletionCharge / (es)



    # Clean the calcualted results
    good_indeces = isfinite(jn) * isfinite(jp) * isfinite(jgen)
    jn[jn < 0] = 0
    jp[jp < 0] = 0
    jgen[jgen < 0] = 0
    energies = energies[good_indeces]
    jn = jn[good_indeces]
    jp = jp[good_indeces]
    jgen = jgen[good_indeces]
    bsInitial = bsInitial[good_indeces]
    Jn = trapz(y=jn, x=energies)
    Jp = trapz(y=jp, x=energies)
    Jgen = trapz(y=jgen, x=energies)


    return Jgen, Jn, Jp, Jrec, bsInitial, energies, jgen, jn, jp

