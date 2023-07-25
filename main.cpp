
#include "kernel.cuh"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

#define TEST_PATH "C:\\Users\\lance\\Desktop\\data\\data\\ALVM_imp_maf20perc_w_Target.csv"

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
        outfile = "episcan",
        suffix = ".txt",
        zpthres = 1e-6,
        chunksize = 1000,
        scale = TRUE,
        ncores = detectCores()
    */

    std::ifstream data;
    data.open(TEST_PATH);
    if (!data.is_open()) {
        printf("Could not open data file");
        return 0;
    }

    printf("File opened\n");

    std::vector<std::vector<std::string>> content;
    std::vector<std::string> row;
    std::string line, word;

    printf("Reading file. . .\n");

    //Getting row number for data
    while (getline(data, line)) {
        row.clear();
        std::stringstream str(line);
        while (getline(str, word, ','))
            row.push_back(word);
        content.push_back(row);
    }

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

    printf("Matrices defined from data");

    //Make sure I am not closing too early
    data.close();

    cudaError_t cudaStatus = EpiScan(phenotype_data, genotype_data);

    //cudaDeviceReset must be called before exiting in order for profiling and
    //tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
