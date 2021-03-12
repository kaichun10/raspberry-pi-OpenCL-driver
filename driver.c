// OS (H) Assessed Exercise: OpenCL Host Programming
// 2173845H
// Kaichun Hu

#include "driver.h"
#include <stdio.h>
#include <stdlib.h>

#ifndef OSX
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>
#else
#include <OpenCL/opencl.h>
#endif

////////////////////////////////////////////////////////////////////////////////
CLObject *init_driver()
{
    CLObject *ocl = (CLObject *)malloc(sizeof(CLObject));
    int err; // error code returned from api calls

    unsigned int status[1] = {0}; // number of correct results returned

    size_t global; // global domain size for our calculation
    size_t local;  // local domain size for our calculation

    cl_device_id device_id;         // compute device id
    cl_context context;             // compute context
    cl_command_queue command_queue; // compute command queue
    cl_program program;             // compute program
    cl_kernel kernel;               // compute kernel

    cl_mem input1, input2;     // device memory used for the input array
    cl_mem output, status_buf; // device memory used for the output array

    FILE *programHandle;
    size_t programSize;
    char *programBuffer;

    cl_uint nplatforms;
    err = clGetPlatformIDs(0, NULL, &nplatforms);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to get number of platform: %d!\n", err);
        exit(EXIT_FAILURE);
    }

    // Now ask OpenCL for the platform IDs:
    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * nplatforms);
    err = clGetPlatformIDs(nplatforms, platforms, NULL);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to get platform IDs: %d!\n", err);
        exit(EXIT_FAILURE);
    }
#ifdef GPU
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
#else
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
#endif
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to create a device group: %d!\n", err);
        exit(EXIT_FAILURE);
    }

    // Create a compute context
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        fprintf(stderr, "Error: Failed to create a compute context: %d!\n", err);
        exit(EXIT_FAILURE);
    }

    // Create a command command_queue
    //
    command_queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (!command_queue)
    {
        fprintf(stderr, "Error: Failed to create a command command_queue: %d!\n", err);
        exit(EXIT_FAILURE);
    }
    // get size of kernel source
    programHandle = fopen("./firmware.cl", "r");
    fseek(programHandle, 0, SEEK_END);
    programSize = ftell(programHandle);
    rewind(programHandle);

    // read kernel source into buffer
    programBuffer = (char *)malloc(programSize + 1);
    programBuffer[programSize] = '\0';
    fread(programBuffer, sizeof(char), programSize, programHandle);
    fclose(programHandle);

    // create program from buffer
    program = clCreateProgramWithSource(context, 1, (const char **)&programBuffer, &programSize, &err);
    free(programBuffer);
    if (!program)
    {
        fprintf(stderr, "Error: Failed to create compute program: %d!\n", err);
        exit(EXIT_FAILURE);
    }

    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        fprintf(stderr, "Error: Failed to build program executable: %d!\n", err);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        fprintf(stderr, "%s\n", buffer);
        exit(EXIT_FAILURE);
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, "firmware", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to create compute kernel: %d!\n", err);
        exit(EXIT_FAILURE);
    }
    ocl->context = context;
    ocl->command_queue = command_queue;
    ocl->kernel = kernel;
    ocl->program = program;
    ocl->device_id = device_id;

    //===============================================================================================================================================================
    // START of assignment code section

    // Create a mutex lock for device
    pthread_mutex_t lock;

    err = pthread_mutex_init(&lock, NULL);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to start device lock. %d!\n", err);
        exit(EXIT_FAILURE);
    }

    ocl->device_lock = lock;
    // END of assignment code section
    //===============================================================================================================================================================

    return ocl;
}

int shutdown_driver(CLObject *ocl)
{
    int err = clReleaseProgram(ocl->program);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to release Program: %d!\n", err);
        exit(EXIT_FAILURE);
    }
    err = clReleaseKernel(ocl->kernel);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to release Kernel: %d!\n", err);
        exit(EXIT_FAILURE);
    }
    err = clReleaseCommandQueue(ocl->command_queue);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to release Command Queue: %d!\n", err);
        exit(EXIT_FAILURE);
    }
    err = clReleaseContext(ocl->context);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to release Context: %d!\n", err);
        exit(EXIT_FAILURE);
    }
    //===============================================================================================================================================================
    // START of assignment code section

    // Terminate the mutex lock for device
    pthread_mutex_destroy(&ocl->device_lock);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to release device lock. %d!\n", err);
        exit(EXIT_FAILURE);
    }

    // END of assignment code section
    //===============================================================================================================================================================

    free(ocl);
    return 0;
}

////////////////////////////////////////////////////////////////////////////////

int run_driver(CLObject *ocl, unsigned int buffer_size, int *input_buffer_1, int *input_buffer_2, int w1, int w2, int *output_buffer)
{
    long long unsigned int tid = ocl->thread_num;
#if VERBOSE
    printf("run_driver thread: %llu\n", tid);
#endif
    int err;              // error code returned from api calls
    int status[1] = {-1}; // number of correct results returned
    unsigned int max_iters;
    max_iters = MAX_ITERS;

    size_t global; // global domain size for our calculation
    size_t local;  // local domain size for our calculation

    cl_mem input1, input2;     // device memory used for the input array
    cl_mem output, status_buf; // device memory used for the output array

    // Get the maximum work group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(ocl->kernel, ocl->device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(EXIT_FAILURE);
    }

    global = buffer_size; // create as meany threads on the device as there are elements in the array

    //===============================================================================================================================================================
    // START of assignment code section

    // You must make sure the driver is thread-safe by using the appropriate POSIX mutex operations
    // You must also check the return value of every API call and handle any errors

    // Create the buffer objects to link the input and output arrays in device memory to the buffers in host memory

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

#if VERBOSE
    printf("Creating the buffer objects to link the input and output arrays...\n");
#endif
    input1 = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, buffer_size * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to create input_buffer_1. %d\n", err);
        exit(EXIT_FAILURE);
    }

    input2 = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, buffer_size * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to create input_buffer_2. %d\n", err);
        exit(EXIT_FAILURE);
    }

    output = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY, buffer_size * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to create output_buffer. %d\n", err);
        exit(EXIT_FAILURE);
    }

    status_buf = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY, buffer_size * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to create status. %d\n", err);
        exit(EXIT_FAILURE);
    }

    err = pthread_mutex_lock(&ocl->device_lock);
    if (err != 0)
    {
        fprintf(stderr, "Error: Failed to lock the mutex. %d!\n", err);
        exit(EXIT_FAILURE);
    }

#if VERBOSE
    printf("Buffer objects have been created.\n");
#endif

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    // Write the data in input arrays into the device memory

#if VERBOSE
    printf("Start writing data into the device memory...\n");
#endif

    err = clEnqueueWriteBuffer(ocl->command_queue, input1, CL_TRUE, 0, buffer_size * sizeof(int), input_buffer_1, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to write data into input_buffer_1. %d\n", err);
        exit(EXIT_FAILURE);
    }
    err = clEnqueueWriteBuffer(ocl->command_queue, input2, CL_TRUE, 0, buffer_size * sizeof(int), input_buffer_2, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to write data into input_buffer_2. %d\n", err);
        exit(EXIT_FAILURE);
    }

#if VERBOSE
    printf("Writing data into the device memory is completed.\n");
#endif

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    //Set the arguments to our compute kernel

#if VERBOSE
    printf("Setting the arguments to our compute kernel...\n");
#endif

    err = clSetKernelArg(ocl->kernel, 0, sizeof(cl_mem), &input1);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Setting input1 to the kernel failed. %d\n", err);
        exit(EXIT_FAILURE);
    }
    err = clSetKernelArg(ocl->kernel, 1, sizeof(cl_mem), &input2);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Setting input2 to the kernel failed. %d\n", err);
        exit(EXIT_FAILURE);
    }
    err = clSetKernelArg(ocl->kernel, 2, sizeof(cl_mem), &output);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Setting output to the kernel failed. %d\n", err);
        exit(EXIT_FAILURE);
    }
    err = clSetKernelArg(ocl->kernel, 3, sizeof(cl_mem), &status_buf);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Setting status to the kernel failed. %d\n", err);
        exit(EXIT_FAILURE);
    }
    err = clSetKernelArg(ocl->kernel, 4, sizeof(int), &w1);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Setting w1 to the kernel failed. %d\n", err);
        exit(EXIT_FAILURE);
    }
    err = clSetKernelArg(ocl->kernel, 5, sizeof(int), &w2);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Setting w2 to the kernel failed. %d\n", err);
        exit(EXIT_FAILURE);
    }
    err = clSetKernelArg(ocl->kernel, 6, sizeof(int), &buffer_size);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Setting buffer_size to the kernel failed. %d\n", err);
        exit(EXIT_FAILURE);
    }
#if VERBOSE
    printf("Setting the arguments to our compute kernel is completed.\n");
#endif

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    // Execute the kernel, i.e. tell the device to process the data using the given global and local ranges
#if VERBOSE
    printf("Executing the kernel...\n");
#endif
    err = clEnqueueNDRangeKernel(ocl->command_queue, ocl->kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to execute the kernal on device. %d\n", err);
        exit(EXIT_FAILURE);
    }
#if VERBOSE
    printf("Executing the kernel is completed.\n");
#endif

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    // Wait for the command commands to get serviced before reading back results. This is the device sending an interrupt to the host
#if VERBOSE
    printf("Waiting for the commands to get serviced...\n");
#endif
    err = clFinish(ocl->command_queue);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to block processes running on the device. %d\n", err);
        exit(EXIT_FAILURE);
    }
#if VERBOSE
    printf("All action on the device has finished.\n");
#endif

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    // Check the status
#if VERBOSE
    printf("Checking the status...\n");
#endif
    while (max_iters >= 0 && (status[1]) != 0)
    {

        err = clEnqueueReadBuffer(ocl->command_queue, status_buf, CL_TRUE, 0, sizeof(int), &(status[1]), 0, NULL, NULL);
        max_iters -= 1;
        if (err != CL_SUCCESS)
        {
            fprintf(stderr, "Error: Failed to check the status. %d!/n", err);
            exit(EXIT_FAILURE);
        }
    }
#if VERBOSE
    printf("Status has been checked.\n");
#endif

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    // When the status is 0, read back the results from the device to verify the output
#if VERBOSE
    printf("Read back the results from the device...\n");
#endif

    if ((status[1]) == 0)
    {
        err = clEnqueueReadBuffer(ocl->command_queue, output, CL_TRUE, 0, buffer_size * sizeof(int), output_buffer, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            fprintf(stderr, "Error: Failed to read back the results from the device. %d!/n", err);
            exit(EXIT_FAILURE);
        }
    }

#if VERBOSE
    printf("Finished reading back the results from the device.\n");
#endif

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

// Shutdown and cleanup
#if VERBOSE
    printf("Shuting down and cleaning up... \n");
#endif

    pthread_mutex_unlock(&(ocl->device_lock));
    if (err != 0)
    {
        fprintf(stderr, "Error: Failed to unlock the mutex. %d!\n", err);
        exit(EXIT_FAILURE);
    }

    err = clReleaseMemObject(input1);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to release input1. %d!/n", err);
        exit(EXIT_FAILURE);
    }

    err = clReleaseMemObject(input2);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to release input2. %d!/n", err);
        exit(EXIT_FAILURE);
    }

    err = clReleaseMemObject(output);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to release output. %d!/n", err);
        exit(EXIT_FAILURE);
    }

    err = clReleaseMemObject(status_buf);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to release status_buf. %d!/n", err);
        exit(EXIT_FAILURE);
    }
#if VERBOSE
    printf("Shutdown and cleanup are both completed. \n");
#endif

    // END of assignment code section
    //===============================================================================================================================================================
    return *status;
}