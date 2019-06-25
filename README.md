# Project specifications
**Title:**  
Image steganography  

**Students:**  
Edylson Torikai - 5248962  

**Abstract**:  
Apply steganography techniques to embed information inside a given image. This project intends to apply DCT (Discrete Cosine Transform) techniques as a method to hide information in the frequency domain of the image. The idea will be to hide binary data into a image (Will be trying to use text data at first). There might have a method to extract the data back as well.  
  
This method will consist of using DCT II (type 2) to on each 8x8 pixel-block of the image in order to calculate the weight coeficient for each of the 64 cosine pattern, described in the image below:  
![DCT cosine patterns](https://upload.wikimedia.org/wikipedia/commons/2/23/Dctjpeg.png)
  
In short, we will try to represent the same 8x8 pixels using only cosines functions.  
After finding out the coeficients, we will quantize these using some standard quantization table for JPEG images. This step help us see which patterns have more weight on the overall representation so that we can "discard" some information from the image without changing too much its content. Although the image was changed, it would be barely noticible to the naked eyes.  

From this idea, we will use the fact that high-frequencies cosine functions have lower impact to hide information in these.  
!Notice: We can't use 0 and 1 coeficients because 

Initial tests will be applied on gray-level images, then we will try using RGB images, where DCT will be applied on each color component (this will let us hide bigger messages).  
Update: Correctly working on RGB images and gray-level as well.

### Steps
- Embeding the information:
    - Divide image in 8x8 blocks (padding might be needed to obtain dimensions divisible by 8)
    - Shift the pixel values to be centered around 0 (subtract 128, assuming we are working with 0-255 intensities)
    - Apply DCT II on every 8x8 block
    - Quantize the results using a pre-defined quantization table
    - Apply JSTEG steganography technique:
        - Embed data into the LSB of DCT Coeficients
    - Decoding the image doing inverse process
        - Multiply the quantized coeficients with quantization table (to obtain DCT coeficients again)
        - Apply IDCT II (DCT III) on every 8x8 block
        - shift back the image to 0-255 values
        - Now we have a image with hidden information in it
- Recover message:
    - Divide image by 8x8 block and apply DCT again
    - Quantize stego image, using the **same** quantization table in the forward process
    - Recover bits from each DCT coeficient
    - Reassemble bits and save the data.

### Image source
Image source from [Image Database](http://www.imageprocessingplace.com/DIP-3E/dip3e_book_images_downloads.htm) from the book Digital Image Processing 3rd edition, chapter 8