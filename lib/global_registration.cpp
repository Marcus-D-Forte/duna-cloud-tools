#include <global_registration.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/registration/correspondence_rejection_features.h>
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
    pcl::HarrisKeypoint3D<PointT, KeyPointT, PointT>::Ptr keypoint_method(new pcl::HarrisKeypoint3D<PointT, KeyPointT, PointT>);
    keypoint_method->setInputCloud(m_clouds[index]);
    keypoint_method->setSearchMethod(m_kdtrees[index]);
    keypoint_method->setNormals(m_clouds[index]);

    keypoint_method->setRadius(0.025f);
    // keypoint_method->setKSearch(50);
    // keypoint_method->setMinimumContrast(0.005f);
    // keypoint_method->setScales(0.01, 3, 4);
    keypoint_method->setNonMaxSupression(true);
    keypoint_method->setThreshold(1e-6);
    keypoint_method->compute(*m_keypoints[index]);

    // Copy indices to vector (WONT WORK FOR SIFT)
    omp_set_lock(&map_lock);
    m_keypoints_indices[index].reset(new pcl::Indices);
    *(m_keypoints_indices[index]) = keypoint_method->getKeypointsIndices()->indices;
    omp_unset_lock(&map_lock);

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
    fpfh_estimation->setInputCloud(m_clouds[index]);
    fpfh_estimation->setIndices(m_keypoints_indices[index]); // Comment to Ignore keypoints
    fpfh_estimation->setInputNormals(m_clouds[index]);
    fpfh_estimation->setSearchMethod(m_kdtrees[index]);

    
    // fpfh_estimation->setKSearch(50);
    fpfh_estimation->setRadiusSearch(0.05f);
    fpfh_estimation->compute(*m_feature_clouds[index]);

    return 0;
}

// Source -> I0, Target -> I1
int GlobalRegistration::estimateFeatureCorrespondence(const int i0, const int i1, pcl::Correspondences &corrs, bool relative_to_input)
{
    if (m_feature_clouds.find(i0) == m_feature_clouds.end() ||
        m_feature_clouds.find(i1) == m_feature_clouds.end())
    {
        std::cerr << "Feature Clouds indices " << i0 << ", " << i1 << " not found.\n";
        return -1;
    }

    std::cout << "Features points " << i0 << " : " << m_feature_clouds[i0]->size() << std::endl;
    std::cout << "Features points " << i1 << " : " << m_feature_clouds[i1]->size() << std::endl;

    pcl::search::KdTree<pcl::FPFHSignature33>::Ptr kdtree_i0(new pcl::search::KdTree<pcl::FPFHSignature33>);
    pcl::search::KdTree<pcl::FPFHSignature33>::Ptr kdtree_i1(new pcl::search::KdTree<pcl::FPFHSignature33>);

    // std::cout << "Computing KDTree: " << i0 << std::endl;
    // kdtree_i0->setInputCloud(m_feature_clouds[i0]);
    std::cout << "Computing KDTree: " << i1 << std::endl;
    kdtree_i1->setInputCloud(m_feature_clouds[i1]);

    pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> corr_estimator;
    corr_estimator.setInputSource(m_feature_clouds[i0]);
    // corr_estimator.setSearchMethodSource(kdtree_i0);

    corr_estimator.setInputTarget(m_feature_clouds[i1]);
    corr_estimator.setSearchMethodTarget(kdtree_i1);

    corrs.clear();
    std::cout << "Computing Correspondences...\n";
    corr_estimator.determineCorrespondences(corrs, 4);

    /* Rejectors */
    // pcl::CorrespondencesPtr corr_ptr(new pcl::Correspondences);
    // std::cout << "Applying one-to-one rejector\n";
    // pcl::registration::CorrespondenceRejectorOneToOne rejector0;
    // rejector0.getRemainingCorrespondences(corrs, *corr_ptr);
    // std::cout << "Original corrs: " << corrs.size() << std::endl;
    // std::cout << "Remaining corrs 1-1: " << corr_ptr->size() << std::endl;

    // corrs = *corr_ptr;

    // pcl::registration::CorrespondenceRejectorFeatures rejector1;
    // rejector1.setSourceFeature<pcl::FPFHSignature33>(m_feature_clouds[i0], "src");
    // rejector1.getRemainingCorrespondences(corrs, *corr_ptr);
    // std::cout << "Remaining corrs Feature: " << corr_ptr->size() << std::endl;

    // corrs = *corr_ptr;

    if (relative_to_input)
    {
        // Remap from feature index to input cloud index
        for (int i = 0; i < corrs.size(); ++i)
        {
            // from Feature
            const int &src_feature_idx = corrs[i].index_query;
            const int &tgt_feature_idx = corrs[i].index_match;

            // Original <- Keypoint indices <- Features
            const int &src_idx = (*m_keypoints_indices[i0])[src_feature_idx];
            const int &tgt_idx = (*m_keypoints_indices[i1])[tgt_feature_idx];

            corrs[i].index_query = src_idx;
            corrs[i].index_match = tgt_idx;
        }
    }

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

    // Simple ICP. PCL ICP will not work due to different point types of ICP and Correspondences (features)

    estimator.estimateRigidTransformation(*m_clouds[i0], *m_clouds[i1], corrs, transform);

    // // Estimate the transform
    // transformation_estimation_->estimateRigidTransformation(
    //     *input_transformed, *target_, *correspondences_, transformation_);

    // // Transform the data
    // transformCloud(*input_transformed, *input_transformed, transformation_);

    // // Obtain the final transformation
    // final_transformation_ = transformation_ * final_transformation_;

    return 0;
}
