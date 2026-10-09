//#include "cmdlineparser.h"
#include <iostream>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

#define IMG_HEIGHT 100
#define IMG_WIDTH 100
#define IMG_SIZE IMG_HEIGHT * IMG_WIDTH
#define KERNEL_LEN 9

#ifndef TARGET
#define TARGET "hw"
#endif

int main(int argc, char** argv) {

    int device_index = 0;
    std::string binaryFile = "testing.xclbin";
    std::cout << "Open the device" << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);
    
    size_t img_size_bytes = sizeof(float) * IMG_HEIGHT * IMG_WIDTH;
    size_t int_size_img = sizeof(int) * IMG_HEIGHT * IMG_WIDTH;
    size_t control_size_bytes = sizeof(int);

    auto krnl = xrt::kernel(device, uuid, "top");

    std::cout << "allocate buffer in global mem\n";
    auto ext_data        = xrt::bo(device, img_size_bytes, krnl.group_id(0));
    auto x_trans = xrt::bo(device, img_size_bytes, krnl.group_id(1));
    auto y_trans = xrt::bo(device, img_size_bytes, krnl.group_id(1));
    auto output = xrt::bo(device, img_size_bytes, krnl.group_id(1));

    auto ext_data_map = ext_data.map<int*>();
    auto x_trans_map = x_trans.map<int*>();
    auto y_trans_map = y_trans.map<int*>();
    auto output_map = output.map<int*>();

    std::cout << "filling with zero" << std::endl;

    std::string data1 = "input_test.txt";
    std::ifstream inFile;

    inFile.open(data1);
    if (inFile.is_open()) {
        for (int i = 0; i < IMG_HEIGHT * IMG_WIDTH; i++) {
            inFile >> ext_data_map[i];
        }
        inFile.close(); // CLose input file
    }
    else { //Error message
        std::cout << "can't find test data " << std::endl;
    }

    data1 = "x_trans_test.txt";
    inFile.open(data1);
    if (inFile.is_open()) {
        for (int i = 0; i < IMG_HEIGHT * IMG_WIDTH; i++) {
            inFile >> x_trans_map[i];
        }
        inFile.close(); 
    }
    else {
        std::cout << "can't find x trans " << std::endl;
    }

    data1 = "y_trans_test.txt";
    inFile.open(data1);
    if (inFile.is_open()) {
        for (int i = 0; i < IMG_HEIGHT * IMG_WIDTH; i++) {
            inFile >> y_trans_map[i];
        }
        inFile.close();
    }
    else {
        std::cout << "can't find y trans " << std::endl;
    }

    ext_data.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    x_trans.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    y_trans.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "kernel execution" << std::endl; 

    double kernel_time_in_sec = 0;
    std::chrono::duration<double> kernel_time(0);
    auto kernel_start = std::chrono::high_resolution_clock::now();
    
    auto run = krnl(ext_data, x_trans, y_trans, output);
    run.wait();

    auto kernel_end = std::chrono::high_resolution_clock::now();
    std::cout << "Done.\n";
    kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);
    kernel_time_in_sec = kernel_time.count();
    std::cout << "Execution time = " << kernel_time_in_sec << std::endl;
    std::cout << "Time: " << kernel_time_in_sec << std::endl;

    std::cout << "Get the output data from the device" << std::endl << std::endl;
    output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    for (int i = 0; i < 10; i++) {
        std::cout << "output: " << output_map[i] << std::endl;
    }

    std::ofstream outFile("output.txt");
    if (!outFile) {
        std::cout << "error writing output.txt" << std::endl;
    }
    for (int i = 0; i < IMG_HEIGHT; i++) {
        for (int j = 0; j < IMG_WIDTH; j++) {
            outFile << std::setw(12) << output_map[i * IMG_WIDTH + j] << " ";
        }
        outFile << "\n";
    }
    outFile.close();

    return 0;
}