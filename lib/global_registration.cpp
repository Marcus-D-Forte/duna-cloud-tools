#include <global_registration.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/registration/correspondence_rejection_features.h>
#include <pcl/registration/correspondence_rejection_trimmed.h>
#include <pcl/registration/icp.h>

int GlobalRegistration::computeKeypointsAt(const int index)
{

    if (m_clouds.size() <= index)
        return -1;

    if (m_keypoints.find(index) == m_keypoints.end())
    {
        // Not found. Creating one
        omp_set_lock(&map_lock);
        std::cout << "Creating keypoint containter with index " << index << std::endl;
        m_keypoints[index].reset(new PointCloudKeyPoint);
        omp_unset_lock(&map_lock);
    }
    // pcl::HarrisKeypoint3D<PointT, KeyPointT, PointT>::Ptr keypoint_method(new pcl::HarrisKeypoint3D<PointT, KeyPointT, PointT>);
    // keypoint_method->setInputCloud(m_clouds[index]);
    // keypoint_method->setSearchMethod(m_kdtrees[index]);
    // keypoint_method->setNormals(m_clouds[index]);

    // keypoint_method->setRadius(0.01f);
    // // keypoint_method->setKSearch(50);
    // // keypoint_method->setMinimumContrast(0.005f);
    // // keypoint_method->setScales(0.01, 3, 4);
    // keypoint_method->setNonMaxSupression(false);
    // keypoint_method->setThreshold(1e-6);
    // keypoint_method->compute(*m_keypoints[index]);

    pcl::VoxelGrid<PointT> voxel;
    voxel.setInputCloud(m_clouds[index]);
    voxel.setLeafSize(0.05, 0.05, 0.05);
    voxel.filter(*m_keypoints[index]);

    std::cout << m_keypoints[index]->points[10];

    std::cout << "Keypoints " << index << ": " << m_keypoints[index]->size();

    return 0;
}

int GlobalRegistration::computeFeaturesAt(const int index)
{
    if (m_clouds.size() <= index)
        return -1;

    if (m_keypoints.find(index) == m_keypoints.end())
    {
        // Keypoints not found.
        std::cout << "Keypoint at index: " << index << " not found.\n";
        return -1;
    }

    if (m_feature_clouds.find(index) == m_feature_clouds.end())
    {
        omp_set_lock(&map_lock);
        std::cout << "Creating feature containter with index " << index << std::endl;
        m_feature_clouds[index].reset(new PointCloudFeature);
        omp_unset_lock(&map_lock);
    }
    pcl::FPFHEstimation<PointT, PointT, pcl::FPFHSignature33>::Ptr fpfh_estimation(new pcl::FPFHEstimation<PointT, PointT, pcl::FPFHSignature33>);
    fpfh_estimation->setInputCloud(m_keypoints[index]);
    fpfh_estimation->setSearchMethod(m_kdtrees[index]);
    fpfh_estimation->setSearchSurface(m_clouds[index]);
    fpfh_estimation->setInputNormals(m_clouds[index]);

    // fpfh_estimation->setKSearch(100);
    fpfh_estimation->setRadiusSearch(0.2f);
    fpfh_estimation->compute(*m_feature_clouds[index]);

    // for (int i = 0; i < 1000; ++i)
    // {
    //     for (int j = 0; j < 33; ++j)
    //         std::cout << m_feature_clouds[index]->points[i].histogram[j] << " ";
    //     std::cout << "\n";
    // }

    return 0;
}

// Source -> I0, Target -> I1
int GlobalRegistration::estimateFeatureCorrespondence(const int src_index, const int tgt_index, pcl::Correspondences &corrs)
{
    if (m_feature_clouds.find(src_index) == m_feature_clouds.end() ||
        m_feature_clouds.find(tgt_index) == m_feature_clouds.end())
    {
        std::cerr << "Feature Clouds indices " << src_index << ", " << tgt_index << " not found.\n";
        return -1;
    }

    std::cout << "Features points " << src_index << " : " << m_feature_clouds[src_index]->size() << std::endl;
    std::cout << "Features points " << tgt_index << " : " << m_feature_clouds[tgt_index]->size() << std::endl;

    pcl::search::KdTree<pcl::FPFHSignature33>::Ptr kdtree_src(new pcl::search::KdTree<pcl::FPFHSignature33>);
    pcl::search::KdTree<pcl::FPFHSignature33>::Ptr kdtree_tgt(new pcl::search::KdTree<pcl::FPFHSignature33>);

    std::cout << "Computing KDTree: " << src_index << std::endl;
    kdtree_src->setInputCloud(m_feature_clouds[src_index]);
    std::cout << "Computing KDTree: " << tgt_index << std::endl;
    kdtree_tgt->setInputCloud(m_feature_clouds[tgt_index]);

    pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> corr_estimator;
    corr_estimator.setInputSource(m_feature_clouds[src_index]);
    corr_estimator.setSearchMethodSource(kdtree_src, true);

    corr_estimator.setInputTarget(m_feature_clouds[tgt_index]);
    corr_estimator.setSearchMethodTarget(kdtree_tgt, true);

    std::cout << "Computing Reciprocal Correspondences...\n";
    corr_estimator.determineReciprocalCorrespondences(corrs, 1);
    // corr_estimator.determineCorrespondences(corrs, 10);

    /* Rejectors */
    pcl::CorrespondencesPtr corr_ptr(new pcl::Correspondences);
    std::cout << "Applying one-to-one rejector\n";
    pcl::registration::CorrespondenceRejectorOneToOne rejector0;
    rejector0.getRemainingCorrespondences(corrs, *corr_ptr);
    std::cout << "Before: " << corrs.size() << std::endl;
    std::cout << "Remaining: " << corr_ptr->size() << std::endl;
    corrs = *corr_ptr;

    pcl::registration::CorrespondenceRejectorTrimmed rejector1;
    rejector1.setOverlapRatio(0.5);
    rejector1.setMinCorrespondences(50);
    rejector1.getRemainingCorrespondences(corrs, *corr_ptr);
    std::cout << "Before: " << corrs.size() << std::endl;
    std::cout << "Remaining: " << corr_ptr->size() << std::endl;
    corrs = *corr_ptr;

    // pcl::registration::CorrespondenceRejectorFeatures rejector1;
    // rejector1.setSourceFeature<pcl::FPFHSignature33>(m_feature_clouds[i0], "src");
    // rejector1.getRemainingCorrespondences(corrs, *corr_ptr);
    // std::cout << "Remaining corrs Feature: " << corr_ptr->size() << std::endl;


    return 0;
}

int GlobalRegistration::estimateFeatureTransform(const int i0, const int i1, Eigen::Matrix4f &transform)
{
    if (m_clouds.size() <= i0 || m_clouds.size() <= i1)
        return -1;

    pcl::registration::TransformationEstimationSVD<PointT, PointT> estimator;

    pcl::Correspondences corrs;
    estimateFeatureCorrespondence(i0, i1, corrs);
    pcl::PointCloud<PointT>::Ptr transformed_source(new pcl::PointCloud<PointT>);

    estimator.estimateRigidTransformation(*m_clouds[i0], *m_clouds[i1], corrs, transform);

    return 0;
}
