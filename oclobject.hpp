// This file contains a couple of handy structures and functions for
// selection, creation and automatic deletion of basic OpenCL objects,
// such as platform, device, context, queue, program, kernel and buffer.


#ifndef OCLOBJECT_HPP_
#define OCLOBJECT_HPP_

#include <CL/cl.h>
#include <string>
#include <vector>

#include "basic.hpp"

using std::string;


// Pick one available platform id.
// Platform is selected by name or by index.
// To select by index, platform_name_or_index should contain textual
// representation of decimal number, for example "0", "1" etc.
// To select by name, this argument should be a string that is not
// a number, for example "Intel". This string will be used as a sub-string,
// to select particular platform. Comparison of strings is case-sensitive.
cl_platform_id selectPlatform (const string& platform_name_or_index);

// Pick one or multiple devices of specified type.
std::vector<cl_device_id> selectDevices (
    cl_platform_id platform,
    const string& device_type
);

// Pick a single device of specified name/index and type.
// Device_name_or_index is treated similarly to platform_name_or_index
// in selectPlatform function.
cl_device_id selectDevice (
    cl_platform_id platform,
    const string& device_name_or_index,
    const string& device_type_name
);


void readProgramFile (const string& program_file_name, std::vector<char>& program_text_prepared);

cl_program createAndBuildProgram (
    const std::vector<char>& program_text_prepared,
    cl_context context,
    size_t num_of_devices,
    const cl_device_id* devices,
    const string& build_options
);

// Helper structure to initialize and hold basic OpenCL objects.
// Contains platform, device, context and queue.
// Platfrom and device are selected by given attributes (see the constructor);
// context is simply created for the chosen device without any special properties;
// and queue is created in this context and for selected device with additional
// optional properties provided through the constructor arguments.
struct OpenCLBasic
{
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    // Initializes all objects by given attributes:
    //   - for platform: platfrom name substring (for example, "Intel") or index (for example, "1")
    //   - for device: device name substring or index
    //   - for device type: name of the device type (for example, "cpu"); see all supported
    //        device types in parseDeviceType description
    //   - for queue: by queue properties
    // In case of empty string for platfrom or device the first available item is selected.
    // In case when device_type is not empty, it limits the set of devices with devices
    // with a given type only; so device_name_or_index is searched among the devices of
    // a given type only.
    OpenCLBasic (
        const string& platform_name_or_index,
        const string& device_type,
        const string& device_name_or_index="0", //default is the first device in the filtered list
        cl_command_queue_properties queue_properties = 0
    );

    ~OpenCLBasic ();

private:

    void selectPlatform (const string& platform_name_or_index)
    {
        platform = ::selectPlatform(platform_name_or_index);
    }

    void selectDevice (const string& device_name_or_index, const string& device_type_name);
    void createContext ();
    void createQueue (cl_command_queue_properties queue_properties = 0);

    // Disable copying and assignment to avoid incorrect resource deallocation.
    OpenCLBasic (const OpenCLBasic&);
    OpenCLBasic& operator= (const OpenCLBasic&);
};


// Helper structure to hold program with one selected kernel.
// The program can be loaded from file or created from string.
// In case of file, file name should be provided.
// All basic objects that are represented by OpenCLBasic should be
// pre-initialized and passed to this structure.
struct OpenCLProgramOneKernel
{
    cl_program program;
    cl_kernel kernel;

    // Create and build program and extract kernel.
    // Only one of program_file_name or program_text should be non-empty.
    OpenCLProgramOneKernel (
        OpenCLBasic& oclobjects,
        const string& program_file_name,
        const string& program_text,
        const string& kernel_name,
        const string& build_options = ""
    );

    ~OpenCLProgramOneKernel ();

private:

    // Disable copying and assignment to avoid incorrect resource deallocation.
    OpenCLProgramOneKernel (const OpenCLProgramOneKernel&);
    OpenCLProgramOneKernel& operator= (const OpenCLProgramOneKernel&);
};


// Helper structure to hold OpenCL buffer
// It does not allocate/initialize the OpenCL buffer.
// It is just a container. The only activity it does is
// automatic resource deallocation. When deallocating, the destructor
// use aligned_free to deallocate memory by host pointer. 
template <typename T>
struct OpenCLDeviceMemory
{
    cl_mem device;

    OpenCLDeviceMemory () :
        device(0)
    {
    }

    ~OpenCLDeviceMemory ();

private:

    // Disable copying and assignment to avoid incorrect resource deallocation.
    OpenCLDeviceMemory (const OpenCLDeviceMemory&);
    OpenCLDeviceMemory& operator= (const OpenCLDeviceMemory&);
};


template <typename T>
OpenCLDeviceMemory<T>::~OpenCLDeviceMemory ()
{
    try
    {
        if(device)
        {
            cl_int err = clReleaseMemObject(device);
            SAMPLE_CHECK_ERRORS(err);
        }
    }
    catch(...)
    {
        destructorException();
    }
}


// Parse textual representation of device type as cl_device_type enum.
// Supported formats for textual representation:
//   - CL_DEVICE_TYPE_ALL: "all", "ALL", "CL_DEVICE_TYPE_ALL" or empty string ""
//   - CL_DEVICE_TYPE_CPU: "cpu", "CPU", "CL_DEVICE_TYPE_CPU"
//   - CL_DEVICE_TYPE_GPU: "gpu", "GPU", "CL_DEVICE_TYPE_GPU"
//   - CL_DEVICE_TYPE_ACCELERATOR: "acc", "ACC", "accelerator", "ACCELERATOR", "CL_DEVICE_TYPE_ACCELERATOR"
//   - CL_DEVICE_TYPE_DEFAULT: "default", "DEFAULT", "CL_DEVICE_TYPE_DEFAULT"
cl_device_type parseDeviceType (const string& device_type_name);


#endif  // end of the include guard
