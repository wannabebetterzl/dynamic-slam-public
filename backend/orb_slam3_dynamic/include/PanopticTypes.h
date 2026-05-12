/**
* Lightweight panoptic-data structures for the clean-room STSLAM rebuild.
* The goal of this header is to establish a stable input contract from
* preprocessing into Tracking without changing the original ORB-SLAM3 flow.
*/

#ifndef PANOPTIC_TYPES_H
#define PANOPTIC_TYPES_H

#include <algorithm>
#include <climits>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core/core.hpp>

namespace ORB_SLAM3
{

static const int kDefaultPanopticDivisor = 1000;

struct InstanceObservation
{
    int panopticId = 0;
    int semanticId = 0;
    int instanceId = 0;
    int trackId = 0;
    int area = 0;
    cv::Rect bbox;
    cv::Mat mask;
};

struct PanopticFrameObservation
{
    double timestamp = -1.0;
    std::string frameName;
    cv::Mat rawPanopticMask;
    std::vector<InstanceObservation> instances;

    bool empty() const
    {
        return rawPanopticMask.empty();
    }

    int CountThingInstances() const
    {
        return static_cast<int>(instances.size());
    }
};

inline int DecodePanopticSemanticId(const int panopticId, const int nMax = kDefaultPanopticDivisor)
{
    return panopticId / nMax;
}

inline int DecodePanopticInstanceId(const int panopticId, const int nMax = kDefaultPanopticDivisor)
{
    return panopticId % nMax;
}

inline std::vector<InstanceObservation> ExtractInstanceObservations(const cv::Mat& panopticMask,
                                                                   const int nMax = kDefaultPanopticDivisor)
{
    std::vector<InstanceObservation> observations;
    if(panopticMask.empty())
        return observations;

    CV_Assert(panopticMask.type() == CV_16UC1 || panopticMask.type() == CV_32SC1);

    struct InstanceAccumulator
    {
        int minX = INT_MAX;
        int minY = INT_MAX;
        int maxX = -1;
        int maxY = -1;
        int area = 0;
    };

    std::unordered_map<int, InstanceAccumulator> accumulators;

    for(int y = 0; y < panopticMask.rows; ++y)
    {
        for(int x = 0; x < panopticMask.cols; ++x)
        {
            int panopticId = 0;
            if(panopticMask.type() == CV_16UC1)
                panopticId = static_cast<int>(panopticMask.at<unsigned short>(y, x));
            else
                panopticId = panopticMask.at<int>(y, x);

            if(panopticId <= 0)
                continue;

            const int instanceId = DecodePanopticInstanceId(panopticId, nMax);
            if(instanceId <= 0)
                continue;

            InstanceAccumulator& acc = accumulators[panopticId];
            acc.minX = std::min(acc.minX, x);
            acc.minY = std::min(acc.minY, y);
            acc.maxX = std::max(acc.maxX, x);
            acc.maxY = std::max(acc.maxY, y);
            ++acc.area;
        }
    }

    observations.reserve(accumulators.size());
    for(const auto& entry : accumulators)
    {
        const int panopticId = entry.first;
        const InstanceAccumulator& acc = entry.second;

        InstanceObservation obs;
        obs.panopticId = panopticId;
        obs.semanticId = DecodePanopticSemanticId(panopticId, nMax);
        obs.instanceId = DecodePanopticInstanceId(panopticId, nMax);
        obs.trackId = obs.instanceId;
        obs.area = acc.area;
        obs.bbox = cv::Rect(acc.minX, acc.minY, acc.maxX - acc.minX + 1, acc.maxY - acc.minY + 1);
        cv::compare(panopticMask, cv::Scalar(panopticId), obs.mask, cv::CMP_EQ);
        observations.push_back(obs);
    }

    std::sort(observations.begin(), observations.end(),
              [](const InstanceObservation& lhs, const InstanceObservation& rhs)
              {
                  return lhs.panopticId < rhs.panopticId;
              });

    return observations;
}

inline PanopticFrameObservation BuildPanopticFrameObservation(const cv::Mat& panopticMask,
                                                             const double timestamp,
                                                             const std::string& frameName,
                                                             const int nMax = kDefaultPanopticDivisor)
{
    PanopticFrameObservation observation;
    observation.timestamp = timestamp;
    observation.frameName = frameName;

    if(panopticMask.empty())
        return observation;

    if(panopticMask.type() == CV_16UC1 || panopticMask.type() == CV_32SC1)
        observation.rawPanopticMask = panopticMask.clone();
    else
        panopticMask.convertTo(observation.rawPanopticMask, CV_16UC1);

    observation.instances = ExtractInstanceObservations(observation.rawPanopticMask, nMax);
    return observation;
}

} // namespace ORB_SLAM3

#endif // PANOPTIC_TYPES_H
