import numpy as np

def add_Gabs_fit(color, Nbins_fit=128, path='./'):
    """
    Gets the absolute G-magnitude form G - RP colours.
    NOTE: uses an external file with mean MS locus and dispersion.
    """
    g_rp = color
    hist,edges = np.histogram(g_rp, bins=Nbins_fit, range=(0.35,1.1))
    centres = (edges[1::]+edges[:-1])/2
    g_rp_mode = centres[hist.argmax()]
    g_rp[g_rp<0.35] = g_rp_mode
    g_rp[g_rp>=1.1] = 1.099

    ### NB: limits should be the same as used to obtain mean_GABS
    spacing = edges[1]-edges[0]
    g_rp_binned = ((g_rp-0.35)//spacing).astype(np.int)
    g_rp_binned[g_rp_binned < 0] = 0

    #file with Gabs and error for stars in each of the 128 bins
    MSfit = np.load(f'{path}/MS-Gabs-mean-std.npy')
    MSmean, MSstd = MSfit[[0,1]], MSfit[2]

    # Returning the fitted Gabs, and with of the MS.
    return np.array(MSmean[1])[g_rp_binned], np.array(MSstd)[g_rp_binned]

def prop_Hg_uncertainty(**kw):
    """
    Propagate uncertainties for Hg, ignores uncertainty in G-magnitude.
    Procedurally generated, apologies for the long expression.
    """
    from numpy import sqrt, log

    pmra = kw['pmra']
    pmdec = kw['pmdec']
    pmra_error = kw['pmra_error']
    pmdec_error = kw['pmdec_error']
    pmra_pmdec_corr = kw['pmra_pmdec_corr']

    return sqrt(((((((5 * ((1 / (sqrt(((pmra ** 2) + (pmdec ** 2))) * log(10))) * ((0.5 * (((pmra ** 2) + (pmdec ** 2)) ** -0.5)) * (2 * pmdec)))) * (pmdec_error ** 2)) * (5 * ((1 / (sqrt(((pmra ** 2) + (pmdec ** 2))) * log(10))) * ((0.5 * (((pmra ** 2) + (pmdec ** 2)) ** -0.5)) * (2 * pmdec))))) + (((((5 * ((1 / (sqrt(((pmra ** 2) + (pmdec ** 2))) * log(10))) * ((0.5 * (((pmra ** 2) + (pmdec ** 2)) ** -0.5)) * (2 * pmdec)))) * pmra_pmdec_corr) * pmra_error) * pmdec_error) * (5 * ((1 / (sqrt(((pmra ** 2) + (pmdec ** 2))) * log(10))) * ((0.5 * (((pmra ** 2) + (pmdec ** 2)) ** -0.5)) * (2 * pmra)))))) + (((((5 * ((1 / (sqrt(((pmra ** 2) + (pmdec ** 2))) * log(10))) * ((0.5 * (((pmra ** 2) + (pmdec ** 2)) ** -0.5)) * (2 * pmra)))) * pmra_pmdec_corr) * pmra_error) * pmdec_error) * (5 * ((1 / (sqrt(((pmra ** 2) + (pmdec ** 2))) * log(10))) * ((0.5 * (((pmra ** 2) + (pmdec ** 2)) ** -0.5)) * (2 * pmdec)))))) + (((5 * ((1 / (sqrt(((pmra ** 2) + (pmdec ** 2))) * log(10))) * ((0.5 * (((pmra ** 2) + (pmdec ** 2)) ** -0.5)) * (2 * pmra)))) * (pmra_error ** 2)) * (5 * ((1 / (sqrt(((pmra ** 2) + (pmdec ** 2))) * log(10))) * ((0.5 * (((pmra ** 2) + (pmdec ** 2)) ** -0.5)) * (2 * pmra)))))))

#to select rpm sample from gaia dr3
def rpm_sel(**kw):
    '''
    Compute distance using RPM method see Viswanathan et al., 2023, MNRAS.
    This code uses simple Schlegel 2D extinction and not the 3D integration used in the paper.
    Because this is a halo catalogue, the results remain unchanged and saves some computer time.
    '''

    AG, ABP, ARP = gaia_extinction.get(kw['phot_g_mean_mag'],
                                       kw['phot_bp_mean_mag'],
                                       kw['phot_rp_mean_mag'],
                                       kw['ebv'],
                                       maxnit=1)


    G_rp = (kw['phot_g_mean_mag'] - AG) - (kw['phot_rp_mean_mag'] - ARP)
    G = kw['phot_g_mean_mag'] - AG
    Gabs = G + 5*np.log10(kw['parallax']) - 10
    PM = np.sqrt(kw['pmra']**2 + kw['pmdec']**2)
    Hg = G + 5*np.log10(PM) - 10
    Hg_uncertainty = prop_Hg_uncertainty(**kw)

    # Some masks
    POE5 = (kw['parallax']/kw['parallax_error'] > 5)
    quality = (kw['ruwe'] < 1.4)&(AG < 2)&(np.log10(Hg/Hg_uncertainty) > 1.75)

    #three linear fits to make cuts on the RPM diagram for the halo selection
    p1=[11.61608077, -0.24101326]
    p2=[8.13255287, 1.6000506 ]
    p3=[11.49163297, -0.87464927]

    #the lines
    L1 = p1[1] + p1[0] * G_rp
    L2 = p2[1] + p2[0] * G_rp
    L3 = p3[1] + p3[0] * G_rp

    #g_rp bounds
    g_rp_left = 0.35
    g_rp_right = 1.1
    g_rp_split12=0.5285055589970487
    g_rp_split23=0.7367195185319125

    #tangential velocity bounds
    VT_upper = 200
    VT_lower = 800

    #removing white dwarfs - see Viswanathan et al., 2023, MNRAS
    WDS1 = (Gabs < L1 + 2)&(G_rp < g_rp_split12)
    WDS2 = (Gabs < L2 + 2)&(G_rp > g_rp_split12)&(G_rp < g_rp_split23)
    WDS3 = (Gabs < L3 + 2)&(G_rp > g_rp_split23)
    WD = ((WDS1|WDS2|WDS3)*POE5)

    #selection of halo stars in the RPM diagram
    U1 = (Hg > L1 + 5*np.log10(VT_upper/4.74047))
    L1 = (Hg < L1 + 5*np.log10(VT_lower/4.74047))
    MSS1 = (G_rp > g_rp_left)&(G_rp < g_rp_split12)&(U1)&(L1)

    U2 = (Hg > L2 + 5*np.log10(VT_upper/4.74047))
    L2 = (Hg < L2 + 5*np.log10(VT_lower/4.74047))
    MSS2 = (G_rp > g_rp_split12)&(G_rp < g_rp_split23)&(U2)&(L2)

    U3 = (Hg > L3 + 5*np.log10(VT_upper/4.74047))
    L3 = (Hg < L3 + 5*np.log10(VT_lower/4.74047))
    MSS3 = (G_rp > g_rp_split23)&(G_rp < g_rp_right)&(U3)&(L3)

    MS = ((MSS1|MSS2|MSS3)*(quality)*(~WD))

    Grpm, Grpm_std = add_Gabs_fit(G_rp)

    phot_dist = 10**((G - Grpm -10)/5)
    phot_dist_uncertainty = phot_dist * np.log(10) * Grpm_std / 5

    mask = MS  ## Return all main-sequence, good astrometry and not WD.
    return(mask)