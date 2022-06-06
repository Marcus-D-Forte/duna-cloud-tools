#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <fstream>
#include <vector>

struct HokuyoData
{
    double timestamp;
    float x;
    float y;
    float z;
    float intensity;
    int scanID;
    int pointID;
};

class AslParser
{
public:
    AslParser() = default;
    virtual ~AslParser() = default;

    int readFromFile(const std::string &file);

    template <typename PointT>
    int toPCLPointCloud(pcl::PointCloud<PointT> &cloud) const
    {
        cloud.clear();

        cloud.resize(data.size());
        for (int i = 0; i < cloud.size(); ++i)
        {
            PointT &pt = cloud.points[i];
            pt.x = data[i].x;
            pt.y = data[i].y;
            pt.z = data[i].z;
        }
        return 0;
    }

private:
    std::vector<HokuyoData> data;
};