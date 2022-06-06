#include <iostream>
#include <asl_parser.h>
#include <global_registration.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/fpfh.h>

using PointT = pcl::PointNormal;
using PointIntensity = pcl::PointXYZI;

void printUsage()
{
    std::cerr << "keypoints [poincloud1.pcd] [pointcloud2.pcd]\n";
}

int main(int argc, char **argv)
{

    if (argc < 3)
    {
        printUsage();
        exit(-1);
    }

    std::string cloud_filename1(argv[1]);
    std::string cloud_filename2(argv[2]);

    AslParser reader;
    if (reader.readFromFile(cloud_filename1) != 0)
    {
        std::cerr << "Error reading dataset. Make sure you run this executable at the dataset folder.\n";
        exit(-1);
    }
    pcl::PointCloud<PointT>::Ptr input_cloud1(new pcl::PointCloud<PointT>);
    reader.toPCLPointCloud<PointT>(*input_cloud1);
    std::cout << "Read: " << input_cloud1->size() << " Points.\n";

    if (reader.readFromFile(cloud_filename2) != 0)
    {
        std::cerr << "Error reading dataset. Make sure you run this executable at the dataset folder.\n";
        exit(-1);
    }

    pcl::PointCloud<PointT>::Ptr input_cloud2(new pcl::PointCloud<PointT>);
    reader.toPCLPointCloud<PointT>(*input_cloud2);
    std::cout << "Read: " << input_cloud2->size() << " Points.\n";

    pcl::NormalEstimation<PointT, PointT> ne;
    ne.setInputCloud(input_cloud1);
    ne.setKSearch(25);
    ne.compute(*input_cloud1);

    pcl::PointCloud<pcl::PointXYZI>::Ptr output(new pcl::PointCloud<pcl::PointXYZI>);
    // pcl::SIFTKeypoint<PointT, pcl::PointWithScale> keypoints;
    pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI,PointT> keypoints;
    keypoints.setInputCloud(input_cloud1);
    keypoints.setNormals(input_cloud1);
    keypoints.setRadius(0.05);
    // keypoints.setKSearch(25);
    
    // keypoints.setScales(0.01, 3, 4);
    // keypoints.setMinimumContrast(0.005f);

    keypoints.setNonMaxSupression(true);
    keypoints.setRefine(false);
    keypoints.setThreshold(1e-6);
    keypoints.compute(*output);

    std::cout << "Keypoints: " << output->size() << std::endl;

    pcl::visualization::PCLVisualizer viewer;
    viewer.addCoordinateSystem();
    viewer.addPointCloud<PointT>(input_cloud1, "input");
    viewer.addPointCloud<pcl::PointXYZI>(output, "output");

    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "output");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "output");

    while (!viewer.wasStopped())
        viewer.spin();

    // // KDTree
    // pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);

    // pcl::NormalEstimation<PointT, PointT> ne;
    // ne.setSearchMethod(kdtree);
    // ne.setKSearch(25);

    // kdtree->setInputCloud(input_cloud1);
    // ne.setInputCloud(input_cloud1);
    // ne.compute(*input_cloud1);

    // kdtree->setInputCloud(input_cloud2);
    // ne.setInputCloud(input_cloud2);
    // ne.compute(*input_cloud2);

    // std::cout << "Normals successfully computed.\n";

    // GlobalRegistration g_res;
    // g_res.addPointCloud(input_cloud1);
    // g_res.addPointCloud(input_cloud2);

    // std::cout << "Computing Keypoints...\n";
    // g_res.computeKeypointsAt(0);
    // g_res.computeKeypointsAt(1);

    // std::cout << "Computing Features...\n";
    // g_res.computeFeaturesAt(0);
    // g_res.computeFeaturesAt(1);

    // auto indices1 = g_res.getKeyPointsIndicesAt(0);
    // auto indices2 = g_res.getKeyPointsIndicesAt(1);

    // if (!indices1 || !indices2)
    //     exit(0);
    // pcl::io::savePCDFile(cloud_filename1 + "_keypoints.pcd", *input_cloud1, *indices1, false);
    // pcl::io::savePCDFile(cloud_filename1 + "_full.pcd", *input_cloud1, false);

    // pcl::io::savePCDFile(cloud_filename2 + "_keypoints.pcd", *input_cloud2, *indices2, false);
    // pcl::io::savePCDFile(cloud_filename2 + "_full.pcd", *input_cloud2, false);
    //    pcl::Correspondences corrs;

    // if (g_res.estimateFeatureCorrespondence(0,5,corrs) != 0)
    //     std::cerr << "Pair 0,5 dont exist\n";

    // g_res.estimateFeatureCorrespondence(0,1,corrs);

    // std::cout << "Found: " << corrs.size() << " Correspondences\n";

    // auto keypts0 = g_res.getKeyPointsAt(0);
    // auto keypts1 = g_res.getKeyPointsAt(1);

    // // VISUALIZE

    // pcl::visualization::PCLVisualizer viewer;

    // viewer.addPointCloud<PointIntensity>(keypts0,"cloud0");
    // viewer.addPointCloud<PointIntensity>(keypts1,"cloud1");

    // for(const auto& it : corrs)
    // {

    // }

    // while(!viewer.wasStopped())
    // {
    //     viewer.spin();
    // }

    return 0;
}