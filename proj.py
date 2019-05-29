import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.fftpack import dctn, idctn

# 50% resol.
# QUANTIZATION_TABLE = np.array([
#   [16,11,10,16,24,40,51,61],
#   [12,12,14,19,26,58,60,55],
#   [14,13,16,24,40,57,69,56],
#   [14,17,22,29,51,87,80,62],
#   [18,22,37,56,68,109,103,77],
#   [24,35,55,64,81,104,113,92],
#   [49,64,78,87,103,121,120,101],
#   [72,92,95,98,112,100,103,99]
# ])

QUANTIZATION_TABLE = np.array([
  [16,12,14,14,18,24,49,72],
  [11,12,13,17,22,35,64,92],
  [10,14,16,22,37,55,78,95],
  [16,19,24,29,56,64,87,98],
  [24,26,40,51,68,81,103,112],
  [40,58,57,87,109,104,121,100],
  [51,60,69,80,103,113,120,103],
  [61,55,56,62,77,92,101,99]
])

np.set_printoptions(precision=2, suppress=True)

# DCT Type 2
def img_dct(img, q_table):
  assert q_table.shape == (8,8), "q_table parameter should be a 8x8 matrix"

  # Shift the whole image, so we can have values centered in 0
  print(img)
  img = np.subtract(img, 128)

  print(img)

  # We will work on 8x8 blocks of the image
  m,n = img.shape

  for i in range(0, m, 8):
    for j in range(0, n, 8):
      # apply DCT for every 8x8 block
      b = img[i:i+8, j:j+8]

      b_dct = dctn(b, norm='ortho')

      print(b_dct)

      # Quantize using the quantization table provided, rounding values to integers
      b_qntz = np.round(np.divide(b_dct, q_table))
      print(b_qntz)

      print(np.multiply(b_qntz, q_table))





def rmse_compare(A, B):
  assert A.shape == B.shape, "Shapes not compatible"
  return np.sqrt(np.mean( (A.astype(int) - B.astype(int))**2 ))

# filename = input().strip()
# img = imageio.imread(filename).astype(np.int8)


# TEST 1
test = np.array([
  [62,55,55,54,49,48,47,55],
  [62,57,54,52,48,47,48,53],
  [61,60,52,49,48,47,49,54],
  [63,61,60,60,63,65,68,65],
  [67,67,70,74,79,85,91,92],
  [82,95,101,106,114,115,112,117],
  [96,111,115,119,128,128,130,127],
  [109,121,127,133,139,141,140,133],
])

# TEST 2 (from wikipedia)
# test = np.array([
#   [52,55,61,66,70,61,64,73],
#   [63,59,55,90,109,85,69,72],
#   [62,59,68,113,144,104,66,73],
#   [63,58,71,122,154,106,70,69],
#   [67,61,68,104,126,88,68,70],
#   [79,65,60,70,77,68,58,75],
#   [85,71,64,59,55,61,65,83],
#   [87,79,69,68,65,76,78,94],
# ])

img_dct(test, QUANTIZATION_TABLE)

# Divide the image into 8x8 blocks
# Transform each block using DCT mathematical operations
# Quantitize each DCT block (lossy compression)
# Embed the message bits from the quantitized coefficients (avoid 0, 1, -1, and the AC)
