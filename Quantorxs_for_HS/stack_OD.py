def stack_OD(filenumber):
    #mfile = directory+'532_'+date+filenumber+'/532_'+date+filenumber+'.hdr'
    m = stxm.STXM(filenumber)
    shifts = find_image_shifts(m)
    d = apply_image_shifts(m,shifts,edit_in_place=True)
    d = crop_shifted_stack(m,shifts,edit_in_place=True)
    autoIOCutoff = 0.90
    io = autoIO(m.data)
    plt.show()
    ODdata = to_OD(m.data,io)
    return m, np.flip(ODdata,axis=1)

# Register images using Fourier Shift Theorem
def find_image_shifts(s,ref_image=25):
    images = s.data.shape[0]
    shape = s.data[0].shape
    shifts = np.zeros((images,2))
    #calculate Fourier transforms
    ref_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(s.data[ref_image])))
    for i in range(images):
        img_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(s.data[i])))
        fr = (ref_fft * img_fft.conjugate()) / (np.abs(ref_fft) * np.abs(img_fft))
        fr = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fr)))
        fr = np.abs(fr)    
        xc, yc = np.unravel_index(np.argmax(fr), shape)
        # Limit the search to 1 pixel border
        if xc == 0:
            xc = 1
        if xc == shape[0] - 1:
            xc = shape[0] - 2
        if yc == 0:
            yc = 1
        if yc == shape[1] - 1:
            yc = shape[1] - 2
        # Use peak fit to find the shifts
        xpts = [xc - 1, xc, xc + 1]
        ypts = fr[xpts, yc]
        xf = peak_fit(xpts, ypts)
        xpts = [yc - 1, yc, yc + 1]
        ypts = fr[xc, xpts]
        yf = peak_fit(xpts, ypts)
        shifts[i,0] = xf - float(shape[0]) / 2.0
        shifts[i,1] = yf - float(shape[1]) / 2.0
    plt.plot(s.energies,shifts[:,0],label='Y shift')
    plt.plot(s.energies,shifts[:,1],label='X shift')
    plt.ylabel('Image Shift (pixels)')
    plt.xlabel('eV')
    plt.legend()
    plt.show()
    return shifts

def peak_fit(x, y):
    y1y0 = y[1] - y[0]
    y2y0 = y[2] - y[0]
    x1x0 = float(x[1] - x[0])
    x2x0 = float(x[2] - x[0])
    x1x0sq = float(x[1] * x[1] - x[0] * x[0])
    x2x0sq = float(x[2] * x[2] - x[0] * x[0])

    c_num = y2y0 * x1x0 - y1y0 * x2x0
    c_denom = x2x0sq * x1x0 - x1x0sq * x2x0
    if c_denom == 0:
        print('Divide by zero error')
        return
    c = c_num / float(c_denom)
    if x1x0 == 0:
        print('Divide by zero error')
        return
    b = (y1y0 - c * x1x0sq) / float(x1x0)
    a = y[0] - b * x[0] - c * x[0] * x[0]
    fit = [a, b, c]
    if c == 0:
        xpeak = 0.
        print('Cannot find xpeak')
        return
    else:
        # Constrain the fit to be within these three points.
        xpeak = -b / (2.0 * c)
        if xpeak < x[0]:
            xpeak = float(x[0])
        if xpeak > x[2]:
            xpeak = float(x[2])
    return xpeak

def apply_image_shifts(s,shifts,edit_in_place=True):
    images = s.data.shape[0]
    shifted_data = np.zeros(s.data.shape)
    shape = s.data[0].shape
    for i in range(images):
        out_of_boundaries_value = np.mean(s.data[i])
        img = ndi.interpolation.shift(s.data[i],[shifts[i,0],shifts[i,1]],mode='constant',cval=out_of_boundaries_value)
        if edit_in_place==True: s.data[i] = img
        shifted_data[i] = img
    return shifted_data

def crop_shifted_stack(s, shifts, edit_in_place=True):
    shape = s.data[0].shape
    xleft = int(np.round(shifts[:,1]).max())
    if xleft < 0: xleft = 0
    xright = shape[1]+int(np.round(shifts[:,1]).min())
    if xright > shape[1]: xright = shape[1]
    ytop = shape[0]+int(np.round(shifts[:,0]).min())
    if ytop > shape[0]: ytop = shape[0]
    ybottom = int(np.round(shifts[:,0]).max())
    if ybottom < 0: ybottom = 0
    if edit_in_place==True: s.data = s.data[:,ybottom:ytop,xleft:xright]
    cropped_data = s.data[:,ybottom:ytop,xleft:xright]
    return cropped_data

def autoIO(data, autoIOCutoff=0.9, showFig = True):
    #n_ev = data.shape[2] #for .stk files
    #avg_image = data.sum(axis=2)/n_ev #for .stk files
    n_ev = data.shape[0]
    avg_image = data.sum(axis=0)/n_ev
    cutoff = autoIOCutoff*(avg_image.max()-avg_image.min())+avg_image.min()  
    IOROI = np.where(avg_image>cutoff)
    if showFig==True:
        figIO = plt.figure()
        plt.subplot(121)
        plt.title('Average Intensity')
        plt.imshow(avg_image, origin='lower')
        plt.subplot(122)
        plt.title('I0 ROI ({0}% Intensity)'.format(100*autoIOCutoff))    
        plt.imshow(np.where(avg_image>cutoff,avg_image,0.0), origin='lower')
    #IO = data[IOROI[0],IOROI[1]].sum(axis=0)/IOROI[0].size #for .stk files
    IO = data[:,IOROI[0],IOROI[1]].sum(axis=1)/IOROI[0].size
    return IO

def to_OD(data, IO):
    #ODdata = -1.0*np.log(data/IO) #for .stk files
    ODdata = np.zeros(data.shape)
    for i in range(IO.size): ODdata[i] = -1.0*np.log(data[i]/IO[i])
    return ODdata