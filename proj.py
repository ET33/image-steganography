import sys
import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.fftpack import dctn, idctn
from bitstring import Bits, BitArray

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

def main():
  if len(sys.argv) < 3:
    print("Usage: >> python progname.py <cover_image> <target_file> [<lsb_qty>]")
    print("<cover_image>: The image that will be used as cover.")
    print("<target_file>: File containing information to be embed.")
    print("<lsb_qty>: Quantity of LSB that will be used. More bits means more data can be stored, but stego image quality goes down!")
  else:
    fname = sys.argv[1]
    target_file = sys.argv[2]
    lsb_qty = 1 if len(sys.argv) < 4 else int(sys.argv[3])

  cover_img = imageio.imread(fname)
  
  data = open(target_file, "rb")
  data_size = os.path.getsize(target_file)

  # In order to not lose the hidden data, we should not normalize `stego_img` before the recover process
  stego_img = create_stego_img(cover_img, data, QUANTIZATION_TABLE, lsb_qty)
  data.close()

  # Recover message
  hidden_info = recover_msg(stego_img, QUANTIZATION_TABLE, data_size, lsb_qty)

  is_colored = (len(cover_img.shape) > 2) and (cover_img.shape[2] > 1)

  # "Shift back" step is applied using normalization instead of addition
  norm_stego_image = normalize_image(stego_img, 0, 255).astype(np.uint8)

  show_images(cover_img, norm_stego_image, is_color_img=is_colored)
  print(rmse_compare(norm_stego_image, cover_img))

  file_extension = target_file.split('.')[-1]

  if len(hidden_info) == data_size:
    try:
      result_file = open(f"hidden_info.{file_extension}", "wb")
      result_file.write(hidden_info)
      result_file.close()
      print('Huge success :D')
      print('Stored and recovered', data_size, 'bytes!')
    except:
      print('Something went wrong. Could not save info')
  else:
    print('Too many data to be embed. Try using a bigger image')
    print('Could store up to: ', len(hidden_info), 'bytes')
    print('Leftover: ', data_size - len(hidden_info), 'bytes')

  # Steps
  # Divide the image into 8x8 blocks
  # Transform each block using DCT mathematical operations
  # Quantitize each DCT block (lossy compression)
  # Embed the message bits from the quantitized coefficients (avoid 0, 1, -1, and the AC) "AC stands for Alternate Current, a.k.a. the most up-left coeficient"

  # luminance/chrominance

  # Warning during quantization on RGB images! (color overflowing)

def histogram(img, n):
  hist = np.zeros(n, dtype=int)

  # computes for all levels in the range
  for i in range(n):
    hist[i] = len(img[img == i])

  return hist

def normalize_image(A, min, max):
  return min + ( (A-np.min(A))*(max-min) / (np.max(A)-np.min(A)) )

def create_stego_img(cover_img, data, q_table, lsb_qty):
  # DCT + embed
  dcted_image = img_dct(cover_img, q_table, data, lsb_qty)

  # Transform back to image (with message embeded)
  return img_recov(dcted_image, QUANTIZATION_TABLE, cover_img.shape)

def rmse_compare(A, B):
  assert A.shape == B.shape, "Shapes not compatible"
  return np.sqrt(np.mean( (A.astype(int) - B.astype(int))**2 ))

def show_images(*imgs, is_color_img=False):
  n = len(imgs)

  if is_color_img:
    n_cols = 4
    c_map = None
  else:
    n_cols = 2
    c_map = 'gray'
  
  i = 0
  for img in imgs:
    plt.subplot(n, n_cols, i+1)
    plt.imshow(img, cmap=c_map)
    plt.axis('off')
    if i == 0:
      plt.title('Cover Image')
    if i == 2 or i == 4:
      plt.title('Stego Image')

    if is_color_img:
      for c, ch in enumerate(('Red', 'Green', 'Blue')):
        plt.subplot(n, n_cols, i+c+2)
        plt.bar(range(256), histogram(img[:,:,c], 256), color=ch)
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')

    else:
      plt.subplot(n, n_cols, i+2)
      plt.bar(range(256), histogram(img, 256))
      plt.xlabel('Intensity')
      plt.ylabel('Frequency')
    
    i += n_cols
  
  plt.subplots_adjust(wspace=0.32)
  plt.show()

# DCT Type 2
def img_dct(img, q_table, data, lsb_qty=1):
  assert q_table.shape == (8,8), "q_table parameter should be a 8x8 matrix"

  # Center values around 0
  img = np.subtract(img.astype(int), 128)

  # We will work on 8x8 blocks of the image
  # Thus, we will pad the image so it is divisible by 8 in both dimensions
  m,n = img.shape[:2]
  n_channels = img.shape[2] if len(img.shape) > 2 else 1

  # Padding image so we can work with a divisible 8x8 image
  h_pad = 8 - (m % 8)
  v_pad = 8 - (n % 8)
  padding = ((0, v_pad), (0, h_pad)) if n_channels == 1 else ((0, v_pad), (0, h_pad), (0,0))
  pad_img = np.pad(img, padding, "constant", constant_values=0)

  m,n = pad_img.shape[:2]

  # preparing data to be embed
  data = list(Bits(data).bin)
  data.reverse()

  G = np.zeros(pad_img.shape)

  for ch in range(n_channels):
    cur_channel = pad_img if n_channels == 1 else pad_img[:,:,ch]

    for i in range(0, m, 8):
      for j in range(0, n, 8):
        # apply DCT for every 8x8 block
        b = cur_channel[i:i+8, j:j+8]

        b_dct = dctn(b, norm='ortho')

        # Quantize using the quantization table provided, rounding values to integers
        b_qntz = np.round(np.divide(b_dct, q_table)).astype(int)

        # Embeding data
        # `data` is a bitarray
        if len(data) > 0:
          for s, row in enumerate(b_qntz):
            if len(data) <= 0: break
            for t, c in enumerate(row):
              if len(data) <= 0: break
              else:
                if c != 0 and c != 1:
                  c_bin = BitArray(int=c, length=8)

                  lsb = []
                  for _ in range(lsb_qty):
                    if len(data) == 0: break
                    lsb.append(data.pop())
                  lsb = ''.join(lsb)

                  c_bin.overwrite(f'0b{lsb}', 8 - len(lsb))

                  b_qntz[s,t] = c_bin.int
        
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

    for i in range(0, m, 8):
      for j in range(0, n, 8):
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

  # Important to NOT normalize here!! otherwise we will lose the embed data!!
  o_w, o_h = original_shape[:2]
  return r_img[:o_w, :o_h]

def recover_msg(img, q_table, data_len, lsb_qty=1):
  m,n = img.shape[:2]
  n_channels = img.shape[2] if len(img.shape) > 2 else 1

  h_pad = 8 - (m % 8)
  v_pad = 8 - (n % 8)
  padding = ((0, v_pad), (0, h_pad)) if n_channels == 1 else ((0, v_pad), (0, h_pad), (0,0))
  pad_img = np.pad(img, padding, "constant", constant_values=0)
  
  m2,n2 = pad_img.shape[:2]

  data = BitArray()

  for ch in range(n_channels):
    cur_channel = pad_img if n_channels == 1 else pad_img[:,:,ch]

    for i in range(0, m2, 8):
      for j in range(0, n2, 8):
        # divide in 8x8 blocks to recover
        b = cur_channel[i:i+8, j:j+8]

        b_dct = dctn(b, norm='ortho')

        b_qntz = np.round(np.divide(b_dct, q_table)).astype(int)

        # retrieving data
        for s, row in enumerate(b_qntz):
          # Multiply by 8 because data_len is the number of bytes!
          if data_len*8 == data.len: break
          for t, c in enumerate(row):
            if data_len*8 == data.len: break

            if c != 0 and c != 1:
              c_bit = Bits(int=c, length=8).bin[-lsb_qty:]
              c_bit = ''.join(c_bit)
              data.append(f'0b{c_bit}')

  return data.tobytes()

main()