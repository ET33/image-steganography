import sys
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



def histogram(img, n):
  hist = np.zeros(n).astype(int)

  # computes for all levels in the range
  for i in range(n):
    hist[i] = len(img[img == i])

  return hist

def normalize_image(A, min, max):
  return min + ( (A-np.min(A))*(max-min) / (np.max(A)-np.min(A)) )

# return a bitarray
def bytes_to_bit(data):
  d = bitarray()
  d.frombytes(data)
  return d

# DCT Type 2
def img_dct(img, q_table, data):
  assert q_table.shape == (8,8), "q_table parameter should be a 8x8 matrix"

  # We will work on 8x8 blocks of the image
  # Thus, we will pad the image so it is divisible by 8 in both dimensions
  m,n = img.shape[:2]
  n_channels = img.shape[2] if len(img.shape) > 2 else 1

  # Padding image so we can work with a divisible 8x8 image
  h_pad = (m % 8)
  v_pad = (n % 8)
  padding = ((0, v_pad), (0, h_pad)) if n_channels == 1 else ((0, v_pad), (0, h_pad), (0,0))
  pad_img = np.pad(img, padding, "constant", constant_values=0)

  m,n = pad_img.shape[:2]

  # preparing data to be embed
  data = bytes_to_bit(data)

  G = np.zeros(pad_img.shape)

  for ch in range(n_channels):
    cur_channel = pad_img if n_channels == 1 else pad_img[:,:,ch]

    for i in range(0, n, 8):
      for j in range(0, m, 8):
        # apply DCT for every 8x8 block
        b = cur_channel[i:i+8, j:j+8]


        b_dct = dctn(b, norm='ortho')

          
        # Quantize using the quantization table provided, rounding values to integers
        b_qntz = np.round(np.divide(b_dct, q_table)).astype(int)

        if ch == 0 and i == 0 and j == 0:
          print(b_qntz)

        # Embeding data
        # if len(data) > 0:
        #   for s, row in enumerate(b_qntz):
        #     if len(data) <= 0: break
        #     for t, c in enumerate(row):
        #       if len(data) <= 0: break
        #       else:
        #         if c != 0 and c != 1 and c != -1:
        #           Need to fix this step. bitarray might be wrong
        #           c_bin = list(bin(c))
        #           c_bin[-1] = '1' if data.pop(0) == True else '0'
        #           b_qntz[s,t] = int(''.join(c_bin),2)
        
        if n_channels == 1:
          G[i:i+8, j:j+8] = b_qntz
        else:
          G[i:i+8, j:j+8, ch] = b_qntz
  
  return G.astype(int)

# original_shape is needed to undo the zero-padding
def img_recov(img, q_table, original_shape):
  m,n = img.shape[:2]
  n_channels = img.shape[2] if len(img.shape) > 2 else 1

  r_img = np.zeros(img.shape)

  for ch in range(n_channels):
    cur_channel = img if n_channels == 1 else img[:,:,ch]

    for i in range(0, n, 8):
      for j in range(0, m, 8):
        b = cur_channel[i:i+8, j:j+8]

        # get coeficients back
        b_dct = np.multiply(b, q_table)

        # apply idct type 2 (DCT III) to get the image in the spatial domain
        # r_b: recovered block
        r_b = idctn(b_dct, norm='ortho')

        if n_channels == 1:
          r_img[i:i+8, j:j+8] = r_b
        else:
          r_img[i:i+8, j:j+8, ch] = r_b

  o_w, o_h = original_shape[:2]
  # restore the original shape and shift back the image
  shifted_r_img = normalize_image(r_img[:o_w, :o_h], 0, 255).astype(np.uint8)

  return shifted_r_img

def recover_msg(img, q_table, msg_len):
  img = np.subtract(img, 128)

  m,n = img.shape[:2]
  n_channels = img.shape[2] if len(img.shape) > 2 else 1

  h_pad = (m % 8)
  v_pad = (n % 8)
  padding = ((0, v_pad), (0, h_pad)) if n_channels == 1 else ((0, v_pad), (0, h_pad), (0,0))
  pad_img = np.pad(img, padding, "constant", constant_values=0)
  
  m2,n2 = pad_img.shape[:2]

  count = 0
  bits = []
  msg = bytearray()

  for ch in range(n_channels):
    cur_channel = img if n_channels == 1 else img[:,:,ch]

    for i in range(0, n2, 8):
      for j in range(0, m2, 8):
        # divide in 8x8 blocks to recover
        b = cur_channel[i:i+8, j:j+8]

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
                print(bits)
                print(msg,'\n------\n\n')
                msg.extend(bitarray(bits).tobytes())
                bits.clear()

  return msg

def create_stego_img(cover_img, data, q_table):
  # DCT + embed
  dcted_image = img_dct(cover_img, q_table, data)

  # Transform back to image (with message embeded)
  return img_recov(dcted_image, QUANTIZATION_TABLE, cover_img.shape)

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


# dcted_image = img_dct(test, QUANTIZATION_TABLE)
if len(sys.argv) > 1:
  fname = sys.argv[1]
else:
  fname = "images/arara.jpg"

cover_img = imageio.imread(fname)

print(cover_img.shape)

test_message = "Haio"
data = test_message.encode('utf-8')

stego_img = create_stego_img(cover_img, data, QUANTIZATION_TABLE)

print(cover_img[67, 137])
print(stego_img[67, 137])

plt.imshow(normalize_image(np.subtract(stego_img, cover_img), 0, 255), cmap='gray')
plt.show()

# Recover message
# message = recover_msg(stego_img, QUANTIZATION_TABLE, len(test_message))

# print(message)
# print('Hidden message is: ', message.decode('utf-8'))

plt.subplot(221)
plt.imshow(cover_img, cmap='gray')
plt.subplot(222)
plt.bar(range(256), histogram(cover_img, 256))

plt.subplot(223)
plt.imshow(stego_img, cmap='gray')
plt.subplot(224)
plt.bar(range(256), histogram(stego_img, 256))

plt.show()

print(rmse_compare(stego_img, cover_img))

# Steps
# Divide the image into 8x8 blocks
# Transform each block using DCT mathematical operations
# Quantitize each DCT block (lossy compression)
# Embed the message bits from the quantitized coefficients (avoid 0, 1, -1, and the AC)

# luminance/chrominance


# Warning during quantization on RGB images! (color overflowing)