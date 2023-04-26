__kernel void convolution(int halffilterSize, int filterWidth,
                          __constant float *filter, int imageHeight, int imageWidth,
                          __global float *inputImage, __global float *outputImage)
{
    int i, j, id, k, l;
    float sum = 0.0f;
    id = get_global_id(0);
    i = id / imageWidth;
    j = id % imageWidth;
    int e, x, z = halffilterSize;

    for (k = -halffilterSize; k <= halffilterSize; k++, z+=filterWidth)
    {
       x = i + k;
       if (x >= 0 && x < imageHeight)
       {
           e = x * imageWidth + j;

           for (l = -halffilterSize; l <= halffilterSize; l++)
           {
               if (j + l >= 0 && j + l < imageWidth)
               {
                   sum += inputImage[e + l] * filter[z + l];
               }
           }
       }
    }
    outputImage[id] = sum;
}
