#include <asl_parser.h>
#include <iostream>
int AslParser::readFromFile(const std::string &file)
{
    std::ifstream filestream(file);
    data.clear();

    if (!filestream.is_open())
    {
        return -1;
    }
    std::string line;
    // Time_in_sec,x,y,z,Intensities,2DscanId,PointId
    while (getline(filestream, line))
    {
        HokuyoData line_data;
        int ret = std::sscanf(line.c_str(), "%lf,%f,%f,%f,%f,%d,%d",
                    &line_data.timestamp,
                    &line_data.x,
                    &line_data.y,
                    &line_data.z,
                    &line_data.intensity,
                    &line_data.scanID,
                    &line_data.pointID);
        if(ret != 7)
            continue;
        data.push_back(line_data);        
    }

    return 0;
}



