#include <iostream>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <filesystem>

#include <omp.h>
using PointT = pcl::PointXYZI;

void printUsage()
{
    std::cerr << "batch_voxel -r [voxel res] [cloud1] [cloud2] ...\n";
}

int main(int argc, char **argv)
{

    if (argc < 2)
    {
        printUsage();
        exit(0);
    }

    int option;
    float res = 0.01;
    while ((option = getopt(argc, argv, "r:")) != -1)
    {
        switch (option)
        {
        case 'r':
            res = atof(optarg);
            break;

        case '?':
            exit(-1);
            break;
        }
    }

    std::cout << "Using voxel size: " << res << std::endl;
    std::filesystem::create_directories("voxel");

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

        pcl::VoxelGrid<PointT> voxel;
        voxel.setInputCloud(input_cloud);
        voxel.setLeafSize(res, res, res);
        voxel.filter(*output_cloud);

        std::string output_filename(argv[i]);
        output_filename = "voxel/" + output_filename.substr(0, output_filename.find_first_of('.')) + "_voxel.pcd";

        std::cout << "Saving filtered pointcloud: " << output_filename << std::endl;
        std::cout << "Downsampled " << input_cloud->size() << " --> " << output_cloud->size() << std::endl;
        pcl::io::savePCDFile(output_filename, *output_cloud, true);
    }

    return 0;
}