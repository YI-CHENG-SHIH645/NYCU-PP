#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__ void mandelKernel(float lowerX, float lowerY, float stepX,
                             float stepY, int *img_d, int maxIterations,
                             int resX, int resY) {
  // To avoid error caused by the floating number, use the following pseudocode
  //
  int thisX = blockIdx.x * blockDim.x + threadIdx.x;
  int thisY = blockIdx.y * blockDim.y + threadIdx.y;
  if (thisX >= resX || thisY >= resY)
    return;
  int index = thisY * resX + thisX;
  float x = lowerX + thisX * stepX;
  float y = lowerY + thisY * stepY;

  float z_re = x, z_im = y;
  int i;
  for (i = 0; i < maxIterations; ++i) {
    if (z_re * z_re + z_im * z_im > 4.f)
      break;
    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = x + new_re;
    z_im = y + new_im;
  }
  img_d[index] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int *img,
            int resX, int resY, int maxIterations) {
  float stepX = (upperX - lowerX) / resX;
  float stepY = (upperY - lowerY) / resY;

  int *img_h;
  // 宣告 Host 記憶體 (線性)
  img_h = (int *)malloc(resX * resY * sizeof(int));

  // 我想應該不用初始化 Host 數值

  int *img_d;
  // 宣告 Device (GPU) 記憶體
  cudaMalloc((void **)&img_d, resX * resY * sizeof(int));

  // 我想應該不用將資料傳給 Device

  dim3 threadsPerBlock(16, 16);
  dim3 numBlock(resX / threadsPerBlock.x, resY / threadsPerBlock.y);

  // 執行 mandelKernel
  mandelKernel<<<numBlock, threadsPerBlock>>>(lowerX, lowerY, stepX, stepY,
                                              img_d, maxIterations, resX, resY);

  // 等待 GPU 所有 thread 完成
  cudaDeviceSynchronize();

  // 將 Device 的資料傳回給 Host
  cudaMemcpy(img_h, img_d, resX * resY * sizeof(int), cudaMemcpyDeviceToHost);

  memcpy(img, img_h, resX * resY * sizeof(int));

  free(img_h);
  cudaFree(img_d);
}
