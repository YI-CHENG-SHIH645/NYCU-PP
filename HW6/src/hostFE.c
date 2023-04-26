#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;
    int imageSize = imageWidth * imageHeight;
    int halffilterSize = filterWidth / 2;

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(*context, *device, 0, NULL);

    // Create memory buffers on the device for each vector
    cl_mem filter_mem_obj = clCreateBuffer(*context, CL_MEM_READ_ONLY,
                                           filterSize * sizeof(float), NULL, NULL);
    cl_mem inImg_mem_obj = clCreateBuffer(*context, CL_MEM_READ_ONLY,
                                          imageSize * sizeof(float), NULL, NULL);
    cl_mem outImg_mem_obj = clCreateBuffer(*context, CL_MEM_WRITE_ONLY,
                                           imageSize * sizeof(float), NULL, NULL);

    clEnqueueWriteBuffer(command_queue, filter_mem_obj, CL_TRUE, 0,
                         filterSize * sizeof(float), filter, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, inImg_mem_obj, CL_TRUE, 0,
                         imageSize * sizeof(float), inputImage, 0, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", NULL);

    // Set the arguments of the kernel
    clSetKernelArg(kernel, 0, sizeof(int), (void *)&halffilterSize);
    clSetKernelArg(kernel, 1, sizeof(int), (void *)&filterWidth);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&filter_mem_obj);
    clSetKernelArg(kernel, 3, sizeof(int), (void *)&imageHeight);
    clSetKernelArg(kernel, 4, sizeof(int), (void *)&imageWidth);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&inImg_mem_obj);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&outImg_mem_obj);

    // Execute the OpenCL kernel on the list
    size_t global_item_size = imageSize; // Process the entire lists
    size_t local_item_size = 128; // Divide work items into groups of 64
    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                           &global_item_size, NULL, 0, NULL, NULL);

    // Read the memory buffer C on the device to the local variable C
    clEnqueueReadBuffer(command_queue, outImg_mem_obj, CL_TRUE, 0,
                        imageSize * sizeof(float), outputImage, 0, NULL, NULL);
}
