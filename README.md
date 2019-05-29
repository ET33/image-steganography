# Project specifications
**Title:**  
Image steganography  

**Students:**  
Edylson Torikai - 5248962  

**Abstract**:  
Apply steganography techniques to embed information inside a given image. This project intends to apply DCT (Discrete Cosine Transform) techniques as a method to hide information in the frequency domain of the image. The idea will be to hide binary data into a image. There might have a method to extract the data back as well.  
  
This method will consist of using DCT II (type 2) to on each 8x8 pixel-block of the image in order to calculate the weight coeficient for each of the 64 cosine pattern, described in the image below:  
![DCT cosine patterns](https://upload.wikimedia.org/wikipedia/commons/2/23/Dctjpeg.png)
  
In short, we will try to represent the same 8x8 pixels using only cosines functions.  
After finding out the coeficients, we will quantize these using some standard quantization table for JPEG images. This step help us see which patterns have more weight on the overall representation so that we can "discard" some information from the image without changing too much its content. Although the image was changed, it would be barely noticible to the naked eyes.  

From this idea, we will use the fact that high-frequencies cosine functions have lower impact to hide information in these.
