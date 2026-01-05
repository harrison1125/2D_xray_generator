#File looks at broadening functions as Fourier transforms of electron density. 

#Defined as window functions (multi-dimensional tophats)

#Should I consider bringing Bessel function analytical solutions to speed this up? 

#Point: result of fourier transforming an infinitely large crystal

#Gaussian: result of fourier transforming a normal distribution - good for quick, early stage visualization

#Sinc function: result of fourier transforming a rectangular electron density

def gaussian_spread (instrumental_broadening : float):
    return instrumental_broadening
    
