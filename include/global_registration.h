#pragma once

#include <iostream>

#include <vector>
#include <pcl/point_types.h>
#include <pcl/keypoints/keypoint.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/registration.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/transformation_estimation.h>
#include <pcl/features/fpfh.h>

#include <unordered_map>
#include <omp.h>

class GlobalRegistration
{
public:
    using PointT = pcl::PointNormal;
    using KeyPointT = pcl::PointNormal;

    using PointCloudT = pcl::PointCloud<PointT>;
    using PointCloudTPtr = typename PointCloudT::Ptr;
    using PointCloudTConstPtr = typename PointCloudT::ConstPtr;

    using PointCloudKeyPoint = pcl::PointCloud<KeyPointT>;
    using PointCloudKeyPointPtr = typename PointCloudKeyPoint::Ptr;
    using PointCloudKeyPointConstPtr = typename PointCloudKeyPoint::ConstPtr;

    using PointCloudFeature = pcl::PointCloud<pcl::FPFHSignature33>;
    using PointCloudFeaturePtr = typename PointCloudFeature::Ptr;
    using PointCloudFeatureConstPtr = typename PointCloudFeature::ConstPtr;

    using KDTree = pcl::search::KdTree<PointT>;
    using KDTreePtr = typename KDTree::Ptr;

    GlobalRegistration()
    {
        omp_init_lock(&map_lock);
    }
    virtual ~GlobalRegistration() = default;

    // Assume normals are present.
    // TODO compute normals internally (optinal)
    void addPointCloud(const PointCloudTConstPtr &cloud)
    {
        KDTreePtr kdtree(new KDTree);
        kdtree->setInputCloud(cloud);

        m_kdtrees.push_back(kdtree);
        m_clouds.push_back(cloud);
    }

    // Computes keypoint for all pointclouds
    // int computeKeypoints();
    // int computeFeatures();
    int computeKeypointsAt(const int index);
    int computeFeaturesAt(const int index);

    int estimateFeatureCorrespondence(const int i0, const int i1, pcl::Correspondences &corrs);
    int estimateFeatureTransform(const int i0, const int i1, Eigen::Matrix4f &transform);

    PointCloudKeyPointConstPtr getKeyPointsAt(const int index)
    {
        return m_keypoints[index];
    }

    // // Return keypoint indices
    // pcl::IndicesConstPtr getKeyPointsIndicesAt(const int index)
    // {
    //     return m_keypoints_indices[index];
    // }

    PointCloudFeatureConstPtr getFeatureCloudAt(const int index)
    {
        return m_feature_clouds[index];
    }

private:
    std::vector<PointCloudTConstPtr> m_clouds;
    std::vector<KDTreePtr> m_kdtrees;

    // Mappings
    std::unordered_map<int, PointCloudKeyPointPtr> m_keypoints;
    std::unordered_map<int, PointCloudFeaturePtr> m_feature_clouds;

    // std::unordered_map<int, pcl::IndicesPtr> m_keypoints_indices;

    omp_lock_t map_lock;
};