
#include "kernel.cuh"
#include "csv.h"

#include <fstream>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <sstream>
#include "io.h" // dup
#include <sstream> // stringstream

/**
* Utility method for CSV line parsing
*/
void splitString(char* input, std::vector<std::string>& output) {
    while (*input) {
        // Find the next ',' character or the end of the string
        char* commaPosition = std::strchr(input, ',');
        if (commaPosition) {
            // Replace the ',' with a null-terminator to create a substring
            *commaPosition = '\0';
            output.emplace_back(std::string(input));
            input = commaPosition + 1; // Move to the next substring
        }
        else {
            // No more ',' found, add the remaining substring and break
            output.emplace_back(input);
            break;
        }
    }
}


/**
* Utility method for csv reading
*/
std::vector<std::vector<std::string>> readCSV(const std::string& filename) {
    std::vector<std::vector<std::string>> data;
    try {
        int index = 0;
        io::LineReader in(filename);
        while (char* line = in.next_line()) {
            data.push_back(std::vector<std::string>());
            splitString(line, data.at(index));
            index++;
        }

    }
    catch (const std::exception& e) {
        std::cerr << "Error reading CSV file: " << e.what() << std::endl;
    }
    return data;
}

/**
* Different calls:
*
*/
int main(int argc, char* argv[])
{
    //All required variables
    /*
        geno1,
        pheno,
        outfile = "episcan", - belongs here
        suffix = ".txt", - belongs here
        zpthres = 1e-6, - fed into the kernel
        chunksize = 1000, - fed into kernel
    */

    printf("Reading data file...\n");
    std::vector<std::vector<std::string>> content = readCSV(TEST_PATH);

    printf("Read finished successfully, creating matrices...\n");

    //Processing phenotype data first cuz it is easier -> switch to using rows variable but for now just hardcoded
    Matrix phenotype_data{
        1,
        content.size() - 1,
        new double[content.size() - 1]
    };

    //M(row, col) = *(M.elements + row * M.width + col)
    for (int i = 1; i < content.size(); i++) {
        //printf("%s", (content.at(i).at(1)).c_str());
        phenotype_data.elements[i - 1] = std::stod(content.at(i).at(1).c_str());
    }

    //Time to replicate with the genotype data :(
    Matrix genotype_data{
        content.at(0).size() - 2,
        content.size() - 1,
        new double[(content.at(0).size() - 2) * (content.size() - 1)]
    };

    //M(row, col) = *(M.elements + row * M.width + col)
    for (int i = 1; i < content.size(); i++) {
        for (int j = 2; j < content.at(0).size(); j++) {
            genotype_data.elements[(i - 1) * genotype_data.width + j - 2] 
                = std::stod(content.at(i).at(j).c_str());
        }        
    }

    printf("Matrices created, writing feature_labels char**...\n");

    size_t length = genotype_data.width * MAX_HEADER_LENGTH;
    char** feature_labels = new char* [length];
    for (int i = 0; i < length; i++) {
        // Check if the index is valid for content
        if (i + 2 >= 0 && i + 2 < content.at(0).size()) {
            const std::string& str = content.at(0).at(i + 2); // Access the string from content
            int str_length = str.length(); // Get the length of the string

            // Allocate memory for the string in feature_labels[i]
            feature_labels[i] = new char[str_length + 1]; // +1 for null-termination

            // Copy the string into feature_labels[i]
            std::strcpy(feature_labels[i], str.c_str() + '\0');
        }
        else {
            // Handle invalid index
            feature_labels[i] = new char[1];
            feature_labels[i][0] = '\0';
        }
    }

    //Check to make sure that original data is recoverable - WORKS
    /*
    for (int i = 0; i < length; i++) {
        printf("%s, ", std::string(feature_labels[i]).c_str());
    }
    */

    printf("Write executed properly, executing Kernel. . .\n");

    //Make sure I am not closing too early
    //data.close();

    cudaError_t cudaStatus = EpiScan(genotype_data, 
                                     phenotype_data, 
                                     ZPTHRES, 
                                     CHUNK_SIZE);

    //Ok so the reason I never pass the feature_labels array into CUDA memory -> data transfer isn't necessary
    // and it would be stored in unified memory (char related memory management) which I do not want. Rather, 
    // I just deal with it when I am writing to my file on the CPU side and hope things work out.

    //Deallocate the feature_labels array
    for (size_t i = 0; i < length; i++) {
        delete[] feature_labels[i];
    }
    delete[] feature_labels;

    //cudaDeviceReset must be called before exiting in order for profiling and
    //tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
