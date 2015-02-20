import numpy as np
from numpy import fft
from numpy import linalg as LA
from scipy import ndimage
# from scipy import linalg 
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import gmpy
import operator

import os

def rgb2gray(rgb):
    # Convert from rgb to gray level
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def oversampling(image, factor = 7):
    # oversample image by a certain factor
    old_shape = image.shape
    new_shape = (factor*old_shape[0], factor*old_shape[1])
    new_image = np.zeros(new_shape, dtype = image.dtype)
    for i in range(old_shape[0]):
        for j in range(old_shape[1]):
            new_image[factor*i:factor*i+factor,factor*j:factor*j+factor] = image[i,j]*np.ones((factor,factor))
    return new_image

def int2intvec(a):
	# Auxiliary function to recover a vector with the
	# digits of a given integer (in inverse order)
    digit = a%10
    vec = np.array([digit],dtype=int)
    a = (a-digit)/10
    while a!=0:
        digit = a%10
        vec = np.append(vec,int(digit))
        a = (a-digit)/10
    return vec

# Base 7 conversion

# ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
ALPHABET7 = "0123456"
ALPHABET10 = "0123456789"

def base_encode(num, alphabet):
    """Encode a number in Base X

    `num`: The number to encode
    """
    if (str(num) == alphabet[0]):
        return int(0)
    arr = []
    base = len(alphabet)
    while num:
        rem = num % base
        num = num // base
        arr.append(alphabet[rem])
    arr.reverse()
    return int(''.join(arr))

def base7to10(num):
    arr = int2intvec(num)
    num = 0
    for i in range(len(arr)):
        num += arr[i]*(7**(i))
    return num
    
def base10to7(num):
    return base_encode(num, ALPHABET7)

# The centered hyperpel
hyperpel = np.array([\
                [-1,4],[0,4],[1,4],[2,4],[3,4],\
                [-2,3],[-1,3], [0,3], [1,3], [2,3], [3,3], [4,3],\
                [-2,2],[-1,2], [0,2], [1,2], [2,2], [3,2], [4,2],\
                [-3,1],[-2,1],[-1,1], [0,1], [1,1], [2,1], [3,1], [4,1],[5,1],\
                [-3,0],[-2,0],[-1,0], [0,0], [1,0], [2,0], [3,0], [4,0],[5,0],\
                [-2,-1],[-1,-1], [0,-1], [1,-1], [2,-1], [3,-1], [4,-1],\
                [-2,-2],[-1,-2], [0,-2], [1,-2], [2,-2], [3,-2], [4,-2],\
                [-1,-3], [0,-3], [1,-3], [2,-3], [3,-3]])

hyperpel_sa = hyperpel - np.array([1,1])

# Here, given a spiral address, we return the corresponding hexagon 
# in the form row/column

def sa2hex(spiral_address):
    # Split the number in basic unit and call the auxiliary function
    # Here we reverse the order, so that the index corresponds to the 
    # decimal position
    digits = str(spiral_address)[::-1] 
    
    hex_address = np.array([0,0])
    
    for i in range(len(digits)):
        if int(digits[i])<0 or int(digits[i])>6:
            print("Invalid spiral address!")
            return 
        elif digits[i]!= '0':
            hex_address += sa2hex_aux(int(digits[i]),i)
    return hex_address
        
# This computes the row/column positions of the base cases,
# that is, in the form a*10^(zeros).
def sa2hex_aux(a, zeros):
    # Base cases
    if zeros == 0:
        if a == 0:
            return np.array([0,0])
        elif a == 1:
            return np.array([0,8])
        elif a == 2:
            return np.array([-7,4])
        elif a == 3:
            return np.array([-7,-4])
        elif a == 4:
            return np.array([0,-8])
        elif a == 5:
            return np.array([7,-4])
        elif a == 6:
            return np.array([7,4])
    
    return sa2hex_aux(a,zeros-1)+ 2*sa2hex_aux(a%6 +1,zeros-1)

# Computes the value of the hyperpel corresponding to the given
# spiral coordinate.

def sa_value(oversampled_image,spiral_address):
    # The center of the hyperpel for them is the point [1,1] for us
    hp = hyperpel_sa + sa2hex(spiral_address)
    val = 0.
    for i in range(56):
        val += oversampled_image[hp[i,0],hp[i,1]]
    
    return val/56

# Assigns the given value to the hyperpel corresponding to the given
# spiral coordinate

def sa_put_value(spiral_address, value, oversampled_image):
    hp = hyperpel_sa + sa2hex(spiral_address)
    for i in range(56):
        oversampled_image[hp[i,0],hp[i,1]] = value
    
    return oversampled_image

addition_table = [
    [0,1,2,3,4,5,6],
    [1,63,15,2,0,6,64],
    [2,15,14,26,3,0,1],
    [3,2,26,25,31,4,0],
    [4,0,3,31,36,42,5],
    [5,6,0,4,42,41,53],
    [6,64,1,0,5,53,52]
]

def spiral_add(a,b,mod=0):
    dig_a = int2intvec(a)
    dig_b = int2intvec(b) 
    
    if (dig_a<0).any() or (dig_a>7).any() \
      or (dig_b<0).any() or (dig_b>7).any():
        print("Invalid spiral address!")
        return
    
    if len(dig_a) == 1 and len(dig_b)==1:
        return addition_table[a][b]
    
    if len(dig_a) < len(dig_b):
        dig_a.resize(len(dig_b))
    elif len(dig_b) < len(dig_a):
        dig_b.resize(len(dig_a))
        
    res = 0
    
    for i in range(len(dig_a)):
        
        if i == len(dig_a)-1:
            res += spiral_add(dig_a[i],dig_b[i])*(10**i)
        else:
            temp = spiral_add(dig_a[i],dig_b[i])
            res += (temp%10)*(10**i)
        
            carry_on = spiral_add(dig_a[i+1],(temp - temp%10)/10)
            dig_a[i+1] = str(carry_on)
    
    if mod!=0:
        return res%mod
    
    return res

multiplication_table = [
    [0,0,0,0,0,0,0],
    [0,1,2,3,4,5,6],
    [0,2,3,4,5,6,1],
    [0,3,4,5,6,1,2],
    [0,4,5,6,1,2,3],
    [0,5,6,1,2,3,4],
    [0,6,1,2,3,4,5],
]

def spiral_mult(a,b, mod=0):
    dig_a = int2intvec(a)
    dig_b = int2intvec(b) 
    
    if (dig_a<0).any() or (dig_a>7).any() \
      or (dig_b<0).any() or (dig_b>7).any():
        print("Invalid spiral address!")
        return
    
    sa_mult = int(0)
    
    for i in range(len(dig_b)):
        for j in range(len(dig_a)):
            temp = multiplication_table[dig_a[j]][dig_b[i]]*(10**(i+j))
            sa_mult=spiral_add(sa_mult,temp)
    
    if mod!=0:
        return sa_mult%mod
    
    return sa_mult

def omegaf(fft_oversampled, sa):
    # Evaluates the vector omegaf corresponding to the 
    # given spiral address
    
    omegaf = np.zeros(6, dtype=fft_oversampled.dtype)
    
    for i in range(1,7):
        omegaf[i-1] = sa_value(fft_oversampled,spiral_mult(sa,i))
    
    return omegaf

def invariant(fft_oversampled, sa1,sa2,sa3):
    # Evaluates the generalized invariant of f on sa1, sa2 and sa3
    
    omega1 = omegaf(fft_oversampled,sa1)
    omega2 = omegaf(fft_oversampled,sa2)
    omega3 = omegaf(fft_oversampled,sa3)
    
    # Attention: np.vdot uses the scalar product with the complex 
    # conjugation at the first place!
    return np.vdot(omega1*omega2,omega3)

def bispectral_inv(fft_oversampled_example, sa_size = 3):
    # Computes the bispectral invariants for any sa1 in the camembert slice
    # and any sa2
    
    bispectrum = np.zeros((7**(sa_size-1)+2,7**sa_size),dtype = fft_oversampled_example.dtype)
    
    # In -1 we put the bispectrum evaluated at 0
    # In -2 we put the bispectrum evaluated at 1
    for k in range(7**sa_size):
        sa2 = base10to7(k)
        sa3 = base10to7(k+1)
        bispectrum[-1,k]=invariant(fft_oversampled_example,0,sa2,sa2)
        bispectrum[-2,k]=invariant(fft_oversampled_example,1,sa2,sa3)
    
    # In the other places we put the bispectrum evaluated at '1'+str(base7_encode(i))
    for i in range(7**(sa_size-1)):
        sa1 = int('1'+str(base10to7(i)))
        sa1_base10 = base7to10(sa1)
        for k in range(7**sa_size):
            sa2 = base10to7(k)
            sa3 = base10to7(sa1_base10+k)
            bispectrum[i,k]=invariant(fft_oversampled_example,sa1,sa2,sa3)
    
    return bispectrum

def evaluate_invariants(image, sa_size = 2):
    # Evaluates the invariants of the given image.
    
    # compute the normalized FFT
    fft = np.fft.fftshift(np.fft.fft2(image))
    fft /= fft / LA.norm(fft)
    
    # oversample it
    fft_oversampled = oversampling(fft)
    
    return bispectral_inv(fft_oversampled, sa_size)

def bispectral_folder(folder_name, sa_size = 2): 
    # Evaluates all the invariants of the images in the 
    # selected folder, storing them in a dictionary
    
    # we store the results in a dictionary
    results = {}
    
    for filename in os.listdir(folder_name):
        infilename = os.path.join(folder_name, filename)
        if not os.path.isfile(infilename): 
            continue

        base, extension = os.path.splitext(infilename)
        if extension == '.png':
        	test_img = plt.imread(infilename)
        	
        	if len(test_img.shape) == 3:
        		test_img = 1 - rgb2gray(test_img)
        	else:
        		test_img = 1 - test_img
        	bispectrum = evaluate_invariants(test_img, sa_size)
        	results[os.path.splitext(filename)[0]] = bispectrum
            
    return results

def bispectral_plot(bispectrums, comparison = 'triangle', log_scale = True):
    """
    Plots the difference of the norms of the given invariants w.r.t. the 
    comparison element (by default in logarithmic scale)
    
    `bispectrums`: a dictionary with as keys the names of the images and 
                    as values their invariants
    """
    
    if comparison not in bispectrums:
        print("The requested comparison is not in the folder")    
        return
    
    
    bispectrum_diff = {}
    for elem in bispectrums:
        diff = LA.norm(bispectrums[elem]-bispectrums[comparison])
        # we remove nan results
        if not np.isnan(diff):
            bispectrum_diff[elem] = diff
    
    plt.plot(bispectrum_diff.values(),'ro')
    if log_scale == True:
        plt.yscale('log')
    for i in range(len(bispectrum_diff.values())):
        # if we plot in log scale, we do not put labels on items that are
        # too small, otherwise they exit the plot area.
        if log_scale and bispectrum_diff.values()[i] < 10**(-3):
            continue
        plt.text(i,bispectrum_diff.values()[i],bispectrum_diff.keys()[i][:3])
        plt.title("Comparison with as reference '"+ comparison +"'")
        
    return

def plot_list(bispectrums=bispectrums, comparison = 'elaine.512'):
    bispectrum_diff = {}
    for elem in bispectrums:
        diff = LA.norm(bispectrums[elem]-bispectrums[comparison])
        # we remove nan results
        if not np.isnan(diff):
            bispectrum_diff[elem] = diff
    
    sorted_diff = sorted(bispectrum_diff.items(), key=operator.itemgetter(1))
    
    plt.rcParams['figure.figsize'] = (20.0, 100.0)
    for i in range(len(sorted_diff)):
        plt.subplot(len(sorted_diff),1,i+1)
        plt.imshow(plt.imread(folder + sorted_diff[i][0] + '.png'),cmap=cm.binary_r)
        plt.annotate(str(gmpy.mpf(sorted_diff[i][1]).digits(10, 0, -1, 1)),xy = (120.,0.),xycoords='axes points')
    plt.rcParams = plt.rcParamsDefault