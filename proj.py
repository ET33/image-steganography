import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.fftpack import dctn, idctn
from bitarray import bitarray

# 50% quality factor
QUANTIZATION_TABLE = np.array([
  [16,11,10,16,24,40,51,61],
  [12,12,14,19,26,58,60,55],
  [14,13,16,24,40,57,69,56],
  [14,17,22,29,51,87,80,62],
  [18,22,37,56,68,109,103,77],
  [24,35,55,64,81,104,113,92],
  [49,64,78,87,103,121,120,101],
  [72,92,95,98,112,100,103,99]
])

# QUANTIZATION_TABLE = np.array([
#   [16,12,14,14,18,24,49,72],
#   [11,12,13,17,22,35,64,92],
#   [10,14,16,22,37,55,78,95],
#   [16,19,24,29,56,64,87,98],
#   [24,26,40,51,68,81,103,112],
#   [40,58,57,87,109,104,121,100],
#   [51,60,69,80,103,113,120,103],
#   [61,55,56,62,77,92,101,99]
# ])

np.set_printoptions(precision=2, suppress=True)

# DCT Type 2
def img_dct(img, q_table, msg):
  assert q_table.shape == (8,8), "q_table parameter should be a 8x8 matrix"

  # Shift the whole image, so we can have values centered in 0
  img = np.subtract(img, 128)

  # We will work on 8x8 blocks of the image
  # Thus, we will pad the image so it is divisible by 8 in both dimensions
  m,n = img.shape

  h_pad = (m % 8)
  v_pad = (n % 8)
  pad_img = np.pad(img, ((0, v_pad), (0, h_pad)), "constant", constant_values=0)

  # preparing data to be embed
  data_to_be_embed = bitarray()
  data_to_be_embed.fromstring(msg)
  data = data_to_be_embed.tolist()

  m2,n2 = pad_img.shape
  G = np.zeros(pad_img.shape)

  for i in range(0, m2, 8):
    for j in range(0, n2, 8):
      # apply DCT for every 8x8 block
      b = pad_img[i:i+8, j:j+8]

      b_dct = dctn(b, norm='ortho')

      # Quantize using the quantization table provided, rounding values to integers
      b_qntz = np.round(np.divide(b_dct, q_table)).astype(int)

      # Embeding data
      if len(data) > 0:
        for s, row in enumerate(b_qntz):
          if len(data) <= 0: break
          for t, c in enumerate(row):
            if len(data) <= 0: break
            else:
              if c != 0 and c != 1 and c != -1:
                c_bin = list(bin(c))
                c_bin[-1] = '1' if data.pop(0) == True else '0'
                b_qntz[s,t] = int(''.join(c_bin),2)
                
      G[i:i+8, j:j+8] = b_qntz
  
  return G.astype(int)

def img_recov(img, q_table, original_shape):
  m,n = img.shape

  r_img = np.zeros(img.shape)

  for i in range(0, m, 8):
    for j in range(0, n, 8):
      b = img[i:i+8, j:j+8]

      # get coeficients back
      b_dct = np.multiply(b, q_table)

      # apply idct type 2 (DCT III) to get the image in the spatial domain
      # r_b: recovered block
      r_b = idctn(b_dct, norm='ortho')

      r_img[i:i+8, j:j+8] = r_b

  o_w, o_h = original_shape
  # shift back the image
  shifted_r_img = np.add(r_img[:o_w, :o_h], 128).astype(np.uint8)

  return shifted_r_img

def recover_msg(img, q_table, msg_len):
  img = np.subtract(img, 128)

  m,n = img.shape

  h_pad = (m % 8)
  v_pad = (n % 8)
  pad_img = np.pad(img, ((0, v_pad), (0, h_pad)), "constant", constant_values=0)

  m2,n2 = pad_img.shape

  count = 0
  bits = []
  msg = ''

  for i in range(0, m2, 8):
    for j in range(0, n2, 8):
      # divide in 8x8 blocks to recover
      b = pad_img[i:i+8, j:j+8]

      b_dct = dctn(b, norm='ortho')

      b_qntz = np.round(np.divide(b_dct, q_table)).astype(int)

      # retrieving data
      for row in b_qntz:
        if msg_len == len(msg): break
        for c in row:
          if msg_len == len(msg): break

          if c != 0 and c != 1 and c != -1:
            count = (count+1) % 8
            c_bin = list(bin(c))
            bits.append( True if c_bin[-1] == '1' else False )
            if count == 0:
              msg += bitarray(bits).tostring()
              bits.clear()

  return msg

def rmse_compare(A, B):
  assert A.shape == B.shape, "Shapes not compatible"
  return np.sqrt(np.mean( (A.astype(int) - B.astype(int))**2 ))

# filename = input().strip()
# img = imageio.imread(filename).astype(np.int8)

# TEST 1
# test = np.array([
#   [62,55,55,54,49,48,47,55],
#   [62,57,54,52,48,47,48,53],
#   [61,60,52,49,48,47,49,54],
#   [63,61,60,60,63,65,68,65],
#   [67,67,70,74,79,85,91,92],
#   [82,95,101,106,114,115,112,117],
#   [96,111,115,119,128,128,130,127],
#   [109,121,127,133,139,141,140,133],
# ])

# TEST 2 (DIP (Gonzalez, Woods) 3rd ed., page. 383)
# test = np.array([
#   [52,55,61,66,70,61,64,73],
#   [63,59,66,90,109,85,69,72],
#   [62,59,68,113,144,104,66,73],
#   [63,58,71,122,154,106,70,69],
#   [67,61,68,104,126,88,68,70],
#   [79,65,60,70,77,63,58,75],
#   [85,71,64,59,55,61,65,83],
#   [87,79,69,68,65,76,78,94],
# ])


def jsteg():
  pass


# dcted_image = img_dct(test, QUANTIZATION_TABLE)

img = imageio.imread("images/arara.jpg")

test_message = "Hello dasd re qwer ffff jsfdjkdaf oiwfjqwei jewoi fewj rweq weqjro e"

# DCT + embed
dcted_image = img_dct(img, QUANTIZATION_TABLE, test_message)

# Transform back to image (with message embeded)
r_img = img_recov(dcted_image, QUANTIZATION_TABLE, img.shape)

# Recover message
message = recover_msg(r_img, QUANTIZATION_TABLE, len(test_message))

print('Hidden message is: ', message)

plt.subplot(211)
plt.imshow(img)

plt.subplot(212)
plt.imshow(r_img)

plt.show()

# Divide the image into 8x8 blocks
# Transform each block using DCT mathematical operations
# Quantitize each DCT block (lossy compression)
# Embed the message bits from the quantitized coefficients (avoid 0, 1, -1, and the AC)

# luminance/chrominance