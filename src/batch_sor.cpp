#include <iostream>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <filesystem>

#include <omp.h>
using PointT = pcl::PointXYZI;

void printUsage()
{
    std::cerr << "batch_sor -k [meanK] -d [std_dist] [cloud1] [cloud2] ...\n";
}

int main(int argc, char **argv)
{

    if (argc < 2)
    {
        printUsage();
        exit(0);
    }

    int option;
    int meanK = 6;
    float std_dist = 1;
    while ((option = getopt(argc, argv, "k:d:")) != -1)
    {
        switch (option)
        {
        case 'k':
            meanK = atoi(optarg);
            break;

        case 'd':
            std_dist = atof(optarg);
            break;

        case '?':
            exit(-1);
            break;
        }
    }

    std::cout << "Using meanK: " << meanK << std::endl;
    std::cout << "Using std_dist: " << std_dist << std::endl;
    
    std::filesystem::create_directories("sor");

    #pragma omp parallel for
    for (int i = optind; i < argc; i++)
    {
        pcl::PointCloud<PointT>::Ptr input_cloud(new pcl::PointCloud<PointT>);
        if (pcl::io::loadPCDFile(argv[i], *input_cloud) == -1)
        {
            continue;
        }
        std::cout << "Processing: " << argv[i] << std::endl;
        pcl::PointCloud<PointT>::Ptr output_cloud(new pcl::PointCloud<PointT>);

        pcl::StatisticalOutlierRemoval<PointT> sor;
        sor.setInputCloud(input_cloud);
        sor.setMeanK(meanK);
        sor.setStddevMulThresh(std_dist);
        sor.filter(*output_cloud);

        std::string output_filename(argv[i]);
        output_filename = "sor/" + output_filename.substr(0, output_filename.find_first_of('.')) + "_sor.pcd";

        std::cout << "Saving filtered pointcloud: " << output_filename << std::endl;
        std::cout << "Downsampled " << input_cloud->size() << " --> " << output_cloud->size() << std::endl;
     pcl::io::savePCDFile(output_filename, *output_cloud, true);
    }

    return 0;
}