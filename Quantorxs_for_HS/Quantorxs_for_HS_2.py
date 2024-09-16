import hyperspy.api as hs

import hyperspy.api as hs

def background_and_normalize (s, mask=None, signal_range=(270.,282.), Pix_init=None):
    """ 
    This function does three things:
    - fit a model with a power law and creates a background subtracted signal
    - calculates a carbon abundance proxy
    - creates a carbon normalized signal
    
    ** signal_range : [e1, e2] --> energy range for the background fitting (default is [260.,282.])
    ** Pix_init : [x,y] --> pixel position to initialize the fit. Fitted parameters are applied to all pixels as initial values.
    
    example : s_od_back, s_od_norm, C_map, m = background and normalize (s_od)
    
    """

    import copy
    import numpy as np
    
    sb = s.remove_background(signal_range=signal_range,
        background_type='Power law',
        fast=False,
        return_model=True,)
    
    if mask is None: 
         s_od_back=sb[0]
    else: s_od_back=sb[0].deepcopy()*~mask

    C_map=(s_od_back.isig[282.:291.6].integrate1D(-1))
    s_od_norm=s_od_back.deepcopy()
    s_od_norm.data = s_od_back/C_map

    return s_od_back, s_od_norm, C_map, sb[1]
    
def Quantorxs (s, mask=None):
    """
    This function uses calibration obtained on reference material to quantify carbon functional group abundance.
    It creates a model with gaussians placed at fixed position. Their heights is fitted and used as abundance proxies.
    The input signal must be first background subtracted and normalized to the carbon abundance. 
    The function returns the fitted model as well as a list of four signals corresponding to each of the quantified functional groups:
    - Aromatics+Olefinics
    - Ketones+phenols+Nitriles
    - Aliphatics
    - carboxylics+esters
    
    example : m_quantorxs, Quant = Quantorxs(s_od_norm)
    """
    
    
    import copy
    Func_Group_C = (284.1, 284.4, 284.7, 285, 285.4, 285.8, 286.2, 286.5, 286.8, 287.2, 287.5, 287.8, 288.2, 288.5, 288.9, 289.4, 289.9, 290.3, 290.8, 291.2, 291.5, 292.1, 292.7, 293.3, 294, 295, 297.1, 297.5, 299.7, 300, 302.5, 305, 307.5, 310, 312.5)
    Emin = 270.              # Lower energy of the range used for the raw spectrum _ 270
    Emax_C = 700.            # Higher end of the energy range used for the raw spectrum
    Estop_C = 282.           # Energy up to which the pre-edge background fitting is performed _ 282
    Enorm_C = 291.5         # Energy up to which the area normalization is calculated _ 291.5
    Efit_C = 305.0            # Energy up to which the gaussian fitting is performed _ 305
    EpeMin_C = 355          # Energy from which the fitting of the carbon step function is considered for fitting the post-edge _ 355
    w_C = 0.2
    dE = 0.1
    s.metadata.binned = False        
    
    mf = s.create_model()
    mf.set_signal_range(270., 305.)
        
    g = []
    for i in range (len(Func_Group_C)-3):
        g.append(hs.model.components1D.Expression(expression="(height*exp(-(x - x0)**2/(2*fwhm**2)))",
                                            name="Gaussian_"+str(i),
                                            position="x0",
                                            height=0.,
                                            fwhm=w_C,
                                            x0=Func_Group_C[i],
                                            module="numpy"))
        g[i].height.bmin=0
        mf.extend([g[i]])
    
    for i in range (4): mf.set_parameters_value('fwhm', 1.3, component_list =  [g[21+i]])

    mf.set_parameters_value('fwhm', 1.5, component_list =  [g[25]])
    mf.set_parameters_value('fwhm', 0.4, component_list =  [g[26]])
    mf.set_parameters_value('fwhm', 2., component_list =  [g[27]])
    mf.set_parameters_value('fwhm', 0.4, component_list =  [g[28]])
    mf.set_parameters_value('fwhm', 2., component_list =  [g[29]])

    for i in range (2): mf.set_parameters_value('fwhm', 2., component_list =  [g[30+i]])
    
    for i in range (len(Func_Group_C)-3):
        mf.set_parameters_not_free(component_list =  [g[i]])
        mf.set_parameters_free (parameter_name_list=['height'], component_list =  [g[i]])
        
    mf.multifit(optimizer = 'lm', bounded = True, mask=mask)
    
    Aro_ordo = copy.deepcopy(mf.components.Gaussian_0.height.as_signal())
    Aro_ordo.data= -2.944
    Aro_pente = copy.deepcopy(mf.components.Gaussian_0.height.as_signal())
    Aro_pente.data= 655.9
    Ket_ordo = copy.deepcopy(mf.components.Gaussian_0.height.as_signal())
    Ket_ordo.data= -2.6
    Ket_pente = copy.deepcopy(mf.components.Gaussian_0.height.as_signal())
    Ket_pente.data= 274.7
    Ali_ordo = copy.deepcopy(mf.components.Gaussian_0.height.as_signal())
    Ali_ordo.data= -19.531
    Ali_pente = copy.deepcopy(mf.components.Gaussian_0.height.as_signal())
    Ali_pente.data= 681.86
    Carb_ordo = copy.deepcopy(mf.components.Gaussian_0.height.as_signal())
    Carb_ordo.data= -7.425078356238871
    Carb_pente = copy.deepcopy(mf.components.Gaussian_0.height.as_signal())
    Carb_pente.data= 86.15169672487131

    Aro = Aro_ordo+(mf.components.Gaussian_0.height.as_signal()+mf.components.Gaussian_1.height.as_signal()+mf.components.Gaussian_2.height.as_signal()+mf.components.Gaussian_3.height.as_signal()+mf.components.Gaussian_4.height.as_signal())/5*Aro_pente
    Ket = Ket_ordo+(mf.components.Gaussian_6.height.as_signal()+mf.components.Gaussian_7.height.as_signal()+mf.components.Gaussian_8.height.as_signal())/3*Ket_pente
    Ali = Ali_ordo+(mf.components.Gaussian_10.height.as_signal()+mf.components.Gaussian_11.height.as_signal())/2*Ali_pente
    Carb = (mf.components.Gaussian_13.height.as_signal()*Carb_pente)+Carb_ordo
    Quantorxs = [Aro, Ket, Ali, Carb]

    return mf, Quantorxs
    
def background_and_normalize_N (s, mask=None, signal_range = (360., 396.), Pix_init=None):
    """ 
    This function does three things:
    - fit a model with a power law and creates a background subtracted signal
    - calculates a carbon abundance proxy
    - creates a carbon normalized signal
    
    ** signal_range : [e1, e2] --> energy range for the background fitting (default is [260.,282.])
    ** Pix_init : [x,y] --> pixel position to initialize the fit. Fitted parameters are applied to all pixels as initial values.
    
    example : s_od_back, s_od_norm, N_map, m = background and normalize (s_od)
    
    """

    import numpy as np
    
    sb = s.remove_background(signal_range=signal_range,
        background_type='Power law',
        fast=False,
        return_model=True,)
        
    if mask is None: 
         s_od_back=sb[0]
    else: s_od_back=sb[0].deepcopy()*~mask
    
    C_map=(s_od_back.isig[395.:406.5].integrate1D(-1))
    s_od_norm=s_od_back.deepcopy()
    s_od_norm.data = s_od_back/N_map

    return s_od_back, s_od_norm, N_map, sb[1]
