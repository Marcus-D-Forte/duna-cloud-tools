#include <iostream>
#include <asl_parser.h>
#include <global_registration.h>
#include <omp.h>

#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/common/time.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/visualization/pcl_plotter.h>

#include <pcl/filters/voxel_grid.h>

#include <pcl/registration/icp_nl.h>

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

    pcl::console::setVerbosityLevel(pcl::console::L_VERBOSE);

    std::string cloud_filename1(argv[1]);
    std::string cloud_filename2(argv[2]);

    AslParser reader;
    if (reader.readFromFile(cloud_filename1) != 0)
    {
        std::cerr << "Error reading dataset. Make sure you run this executable at the dataset folder.\n";
        exit(-1);
    }
    pcl::PointCloud<PointT>::Ptr input_source(new pcl::PointCloud<PointT>);
    reader.toPCLPointCloud<PointT>(*input_source);
    std::cout << "Read: " << input_source->size() << " Points.\n";

    if (reader.readFromFile(cloud_filename2) != 0)
    {
        std::cerr << "Error reading dataset. Make sure you run this executable at the dataset folder.\n";
        exit(-1);
    }

    pcl::PointCloud<PointT>::Ptr input_target(new pcl::PointCloud<PointT>);
    reader.toPCLPointCloud<PointT>(*input_target);
    std::cout << "Read: " << input_target->size() << " Points.\n";

    // // KDTree
    pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);

    pcl::NormalEstimation<PointT, PointT> ne;
    ne.setSearchMethod(kdtree);
    ne.setKSearch(25);

    kdtree->setInputCloud(input_source);
    ne.setInputCloud(input_source);
    ne.compute(*input_source);

    kdtree->setInputCloud(input_target);
    ne.setInputCloud(input_target);
    ne.compute(*input_target);

    std::cout << "Normals successfully computed.\n";

    GlobalRegistration g_res;
    g_res.addPointCloud(input_source);
    g_res.addPointCloud(input_target);

#pragma omp parallel sections num_threads(2)
    {
#pragma omp section
        {
            std::cout << "Computing Keypoints source\n";
            g_res.computeKeypointsAt(0);
            std::cout << "Computing features source\n";
            g_res.computeFeaturesAt(0);
        }
#pragma omp section
        {
            std::cout << "Computing Keypoints target\n";
            g_res.computeKeypointsAt(1);
            std::cout << "Computing features target\n";
            g_res.computeFeaturesAt(1);
        }
    }

    std::cout << "Keypts[0]: " << g_res.getKeyPointsIndicesAt(0)->size() << std::endl;
    std::cout << "Keypts[1]: " << g_res.getKeyPointsIndicesAt(1)->size() << std::endl;

    auto keypts0 = g_res.getKeyPointsIndicesAt(0);
    auto keypts1 = g_res.getKeyPointsIndicesAt(1);

    if (!keypts0 || !keypts0)
    {
        std::cerr << "No keypoints detected\n";
        exit(0);
    }

    pcl::Correspondences corrs;
    pcl::Correspondences corrs_relative_to_feature;
    if (g_res.estimateFeatureCorrespondence(0, 5, corrs) != 0)
        std::cerr << "Pair 0,5 dont exist\n";

    std::cout << "Computing Correspondences...\n";

    pcl::StopWatch timer;
    timer.reset();
    g_res.estimateFeatureCorrespondence(0, 1, corrs);
    g_res.estimateFeatureCorrespondence(0, 1, corrs_relative_to_feature, false);

    std::cout << "Found: " << corrs.size() << " Correspondences\n";
    std::cout << "Corr Estimate: " << timer.getTimeSeconds() << std::endl;

    const auto features0 = g_res.getFeatureCloudAt(0);
    const auto features1 = g_res.getFeatureCloudAt(1);

    Eigen::Matrix4f transform;
    transform.setIdentity();
    timer.reset();
    g_res.estimateFeatureTransform(0, 1, transform);
    std::cout << "Transform Estimate: " << timer.getTimeSeconds() << std::endl;

    std::cout << "Estimated transform:\n"
              << transform << std::endl;

    pcl::PointCloud<PointT>::Ptr aligned_source(new pcl::PointCloud<PointT>);
    pcl::transformPointCloud(*input_source, *aligned_source, transform);

    /* Local Registration */
    std::cout << "Computing ICP \n";
    pcl::IterativeClosestPointWithNormals<PointT, PointT> icp;
    pcl::VoxelGrid<PointT> voxel;
    voxel.setLeafSize(0.05, 0.05, 0.05);

    pcl::PointCloud<PointT>::Ptr input_source_ss(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr input_target_ss(new pcl::PointCloud<PointT>);
    voxel.setInputCloud(input_source);
    voxel.filter(*input_source_ss);

    voxel.setInputCloud(input_target);
    voxel.filter(*input_target_ss);

    icp.setInputSource(input_source_ss);
    icp.setInputTarget(input_target_ss);
    icp.setMaximumIterations(50);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setMaxCorrespondenceDistance(0.5);
    pcl::PointCloud<PointT> aligned_icp;
    icp.align(aligned_icp);
    pcl::transformPointCloud(*input_source, aligned_icp, icp.getFinalTransformation());
    /*   */

    pcl::io::savePCDFile(cloud_filename1 + "_aligned.pcd", *aligned_source, true);
    pcl::io::savePCDFile(cloud_filename1 + "_icp_aligned.pcd", aligned_icp, true);
    pcl::io::savePCDFile(cloud_filename1 + "_full.pcd", *input_source, true);
    pcl::io::savePCDFile(cloud_filename2 + "_full.pcd", *input_target, true);

    // // VISUALIZE
    pcl::PointCloud<PointT>::Ptr keypts0_cloud(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr keypts1_cloud(new pcl::PointCloud<PointT>);

    pcl::copyPointCloud(*input_source, *keypts0, *keypts0_cloud);
    pcl::copyPointCloud(*input_target, *keypts1, *keypts1_cloud);

    pcl::visualization::PCLVisualizer viewer;
    viewer.addCoordinateSystem();
    int vp0, vp1;
    // viewer.createViewPort(0, 0, 0.5, 1, vp0);
    // viewer.createViewPort(0.5, 0, 1, 1, vp1);

    viewer.addPointCloud<PointT>(input_source, "input0", 0);
    viewer.addPointCloud<PointT>(keypts0_cloud, "keypts0", 0);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "keypts0");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1.0, 0, "keypts0");

    viewer.addPointCloud<PointT>(input_target, "input1", 0);
    viewer.addPointCloud<PointT>(keypts1_cloud, "keypts1", 0);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "keypts1");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1.0, 0, "keypts1");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 1.0, 0, "input1");

    // viewer.addPointCloud<PointIntensity>(keypts1,"cloud1");

    // for (int i = 0; i < corrs_relative_to_feature.size(); i+=50)
    // {
    //     const int &src_idx = corrs_relative_to_feature[i].index_query;
    //     const int &tgt_idx = corrs_relative_to_feature[i].index_match;

    //     // Print some correspondences
    //     const auto &src_feature = features0->points[src_idx];
    //     const auto &tgt_feature = features1->points[tgt_idx];

    //     for (int i = 0; i < 33; ++i)
    //     {
    //         std::cout << src_feature.histogram[i] << " ";
    //     }
    //     std::cout << "\n";
    //     for (int i = 0; i < 33; ++i)
    //     {
    //         std::cout << tgt_feature.histogram[i] << " ";
    //     }

    //     std::cout << "\nDistance: " << corrs[i].distance;

    //     std::cout << "\n\n";
    // }

    for (int i = 0; i < corrs.size(); i += 2)
    {
        // Indices from FeatureCloud
        const int &src_idx = corrs[i].index_query;
        const int &tgt_idx = corrs[i].index_match;

        const PointT &src_pt = input_source->points[src_idx];
        const PointT &tgt_pt = input_target->points[tgt_idx];

        viewer.addLine(src_pt, tgt_pt, 0.4, 0, 0, std::string("arrow") + std::to_string(i));
    }

    while (!viewer.wasStopped())
    {
        viewer.spin();
        // histogram_viewer.spin();
    }

    return 0;
}

/* REFERENCE RESULT

 0.991436362267 0.130590602756 0.000000000000 -0.637333095074
-0.130590602756 0.991436362267 0.000000000000 0.082644641399
0.000000000000 0.000000000000 1.000000000000 -0.019415784627
0.000000000000 0.000000000000 0.000000000000 1.000000000000

*/