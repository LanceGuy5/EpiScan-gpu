
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
    //Parsing through cmd line arguments - add command line compatability later
    for (int i = 0; i < argc; i++) {
        printf("%s\n", argv[i]);
    }
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

    std::vector<std::vector<std::string>> content;
    std::vector<std::string> row;
    std::string line, word;

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
        content.size(),
        new double[content.size()]
    };

    //M(row, col) = *(M.elements + row * M.width + col)
    for (int i = 0; i < content.size(); i++) {
        phenotype_data.elements[i] = std::stod(content.at(i).at(1));
    }

    Matrix A{
        2,
        3,
        new double[6] {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
    };

    Matrix B{
        2,
        3,
        new double[6] {7.0, 8.0, 9.0, 10.0, 11.0, 12.0}
    };

    cudaError_t cudaStatus = EpiScan(phenotype_data, B);

    //cudaDeviceReset must be called before exiting in order for profiling and
    //tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    data.close();
    return 0;
}
