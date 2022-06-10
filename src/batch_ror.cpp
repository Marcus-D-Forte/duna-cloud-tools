#include <iostream>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <filesystem>

#include <omp.h>
using PointT = pcl::PointXYZI;

void printUsage()
{
    std::cerr << "batch_ror -r [radius] -n [min_pts] [cloud1] [cloud2] ...\n";
}

int main(int argc, char **argv)
{

    if (argc < 2)
    {
        printUsage();
        exit(0);
    }

    int option;
    float radius = 0.2;
    int min_pts = 1;
    while ((option = getopt(argc, argv, "r:n:")) != -1)
    {
        switch (option)
        {
        case 'r':
            radius = atof(optarg);
            break;

        case 'n':
            min_pts = atoi(optarg);
            break;

        case '?':
            exit(-1);
            break;
        }
    }

    std::cout << "Using radius: " << radius << std::endl;
    std::cout << "Using min_pts: " << min_pts << std::endl;
    
    std::filesystem::create_directories("ror");

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

        pcl::RadiusOutlierRemoval<PointT> ror;
        ror.setMinNeighborsInRadius(min_pts);
        ror.setRadiusSearch(radius);
        ror.setInputCloud(input_cloud);
        ror.filter(*output_cloud);

        std::string output_filename(argv[i]);
        output_filename = "ror/" + output_filename.substr(0, output_filename.find_first_of('.')) + "_ror.pcd";

        std::cout << "Saving filtered pointcloud: " << output_filename << std::endl;
        std::cout << "Downsampled " << input_cloud->size() << " --> " << output_cloud->size() << std::endl;
        pcl::io::savePCDFile(output_filename, *output_cloud, true);
    }

    return 0;
}