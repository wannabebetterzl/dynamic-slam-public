/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#include "Tracking.h"

#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "G2oTypes.h"
#include "Optimizer.h"
#include "Pinhole.h"
#include "KannalaBrandt8.h"
#include "MLPnPsolver.h"
#include "GeometricTools.h"
#include "Map.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

#include <iostream>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <chrono>
#include <unordered_map>
#include <sstream>
#include <cctype>
#include <set>
#include <limits>

#include <Eigen/SVD>


using namespace std;

namespace ORB_SLAM3
{

namespace
{
cv::Mat ExtractDescriptorRows(const cv::Mat& descriptors, const std::vector<int>& indices)
{
    if(descriptors.empty() || indices.empty())
        return cv::Mat();

    cv::Mat extracted(static_cast<int>(indices.size()), descriptors.cols, descriptors.type());
    for(size_t row = 0; row < indices.size(); ++row)
    {
        descriptors.row(indices[row]).copyTo(extracted.row(static_cast<int>(row)));
    }
    return extracted;
}

int ReadPanopticIdAt(const cv::Mat& panopticMask, const int x, const int y)
{
    if(panopticMask.type() == CV_16UC1)
        return static_cast<int>(panopticMask.at<unsigned short>(y, x));
    return panopticMask.at<int>(y, x);
}

const InstanceObservation* FindInstanceObservation(const Frame& frame, const int instanceId)
{
    const std::map<int, InstanceObservation>::const_iterator it =
        frame.mmInstanceObservations.find(instanceId);
    return (it == frame.mmInstanceObservations.end()) ? NULL : &it->second;
}

double SafeRectIoU(const cv::Rect& lhs, const cv::Rect& rhs)
{
    const cv::Rect intersection = lhs & rhs;
    const double intersectionArea = static_cast<double>(intersection.area());
    const double unionArea =
        static_cast<double>(lhs.area()) + static_cast<double>(rhs.area()) - intersectionArea;
    return unionArea > 0.0 ? intersectionArea / unionArea : 0.0;
}

double EstimateFrameImageArea()
{
    const double width = static_cast<double>(Frame::mnMaxX - Frame::mnMinX);
    const double height = static_cast<double>(Frame::mnMaxY - Frame::mnMinY);
    return std::max(1.0, width * height);
}

double MedianValue(std::vector<double> values)
{
    if(values.empty())
        return 0.0;

    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if(values.size() % 2 == 1)
        return values[mid];
    return 0.5 * (values[mid - 1] + values[mid]);
}

int GetEnvIntOrDefault(const char* name, const int defaultValue, const int minValue)
{
    const char* envValue = std::getenv(name);
    return envValue ? std::max(minValue, std::atoi(envValue)) : defaultValue;
}

bool GetEnvFlagOrDefault(const char* name, const bool defaultValue)
{
    const char* envValue = std::getenv(name);
    if(!envValue)
        return defaultValue;
    return std::string(envValue) != "0";
}

bool ForceFilterDetectedDynamicFeatures()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES", false);
    return value;
}

bool EnableSemanticGeometricVerification()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_SEMANTIC_GEOMETRIC_VERIFICATION", false);
    return value;
}

bool EnableSemanticCandidateGeometryGate()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_SEMANTIC_CANDIDATE_GEOMETRY_GATE", false);
    return value;
}

bool EnableSemanticGeometricVerificationForStage(const std::string& stage)
{
    if(!EnableSemanticGeometricVerification())
        return false;

    const char* envValue = std::getenv("STSLAM_SEMANTIC_GEOMETRIC_RESCUE_STAGES");
    const std::string configuredStages =
        (envValue && std::string(envValue).size() > 0)
            ? std::string(envValue)
            : std::string("track_local_map_pre_pose");

    std::stringstream ss(configuredStages);
    std::string token;
    while(std::getline(ss, token, ','))
    {
        token.erase(std::remove_if(token.begin(),
                                   token.end(),
                                   [](unsigned char ch) { return std::isspace(ch); }),
                    token.end());
        if(token == "*" || token == stage)
            return true;
    }
    return false;
}

bool StageIsEnabledByEnvList(const char* envName,
                             const std::string& defaultStages,
                             const std::string& stage)
{
    const char* envValue = std::getenv(envName);
    const std::string configuredStages =
        (envValue && std::string(envValue).size() > 0)
            ? std::string(envValue)
            : defaultStages;

    std::stringstream ss(configuredStages);
    std::string token;
    while(std::getline(ss, token, ','))
    {
        token.erase(std::remove_if(token.begin(),
                                   token.end(),
                                   [](unsigned char ch) { return std::isspace(ch); }),
                    token.end());
        if(token == "*" || token == stage)
            return true;
    }
    return false;
}

bool ForceFilterDetectedDynamicFeaturesForStage(const std::string& stage)
{
    if(!ForceFilterDetectedDynamicFeatures())
        return false;

    return StageIsEnabledByEnvList("STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURE_STAGES",
                                   "*",
                                   stage);
}

bool EnableGeometricDynamicRejection()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_GEOMETRIC_DYNAMIC_REJECTION", false);
    return value;
}

bool EnableGeometricDynamicRejectionForStage(const std::string& stage)
{
    if(!EnableGeometricDynamicRejection())
        return false;

    const char* envValue = std::getenv("STSLAM_GEOMETRIC_DYNAMIC_REJECTION_STAGES");
    const std::string configuredStages =
        (envValue && std::string(envValue).size() > 0)
            ? std::string(envValue)
            : std::string("track_local_map_pre_pose");

    std::stringstream ss(configuredStages);
    std::string token;
    while(std::getline(ss, token, ','))
    {
        token.erase(std::remove_if(token.begin(),
                                   token.end(),
                                   [](unsigned char ch) { return std::isspace(ch); }),
                    token.end());
        if(token == "*" || token == stage)
            return true;
    }
    return false;
}

int GetSemanticGeometricMinStaticMapObservations()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_SEMANTIC_GEOMETRIC_MIN_STATIC_MAP_OBSERVATIONS", 1, 1);
    return value;
}

double GetSemanticGeometricMaxReprojectionErrorPx()
{
    const char* envValue = std::getenv("STSLAM_SEMANTIC_GEOMETRIC_MAX_REPROJ_ERROR_PX");
    static const double value = envValue ? std::max(0.1, std::atof(envValue)) : 2.5;
    return value;
}

double GetSemanticGeometricMaxDepthErrorM()
{
    const char* envValue = std::getenv("STSLAM_SEMANTIC_GEOMETRIC_MAX_DEPTH_ERROR_M");
    static const double value = envValue ? std::max(0.0, std::atof(envValue)) : 0.15;
    return value;
}

int GetGeometricDynamicRejectionMinStaticMapObservations()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_GEOMETRIC_DYNAMIC_REJECTION_MIN_STATIC_MAP_OBSERVATIONS", 1, 1);
    return value;
}

double GetGeometricDynamicRejectionMaxReprojectionErrorPx()
{
    const char* envValue = std::getenv("STSLAM_GEOMETRIC_DYNAMIC_REJECTION_MAX_REPROJ_ERROR_PX");
    static const double value = envValue ? std::max(0.1, std::atof(envValue)) : 5.0;
    return value;
}

double GetGeometricDynamicRejectionMaxDepthErrorM()
{
    const char* envValue = std::getenv("STSLAM_GEOMETRIC_DYNAMIC_REJECTION_MAX_DEPTH_ERROR_M");
    static const double value = envValue ? std::max(0.0, std::atof(envValue)) : 0.10;
    return value;
}

bool EnableSemanticCandidateSparseFlowGate()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_SEMANTIC_CANDIDATE_SPARSE_FLOW_GATE", false);
    return value;
}

bool EnableSemanticCandidateSparseFlowGateForStage(const std::string& stage)
{
    if(!EnableSemanticCandidateSparseFlowGate())
        return false;

    return StageIsEnabledByEnvList("STSLAM_SEMANTIC_SPARSE_FLOW_GATE_STAGES",
                                   "track_local_map_pre_pose",
                                   stage);
}

bool EnableSemanticConservativeDynamicDelete()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_SEMANTIC_CONSERVATIVE_DYNAMIC_DELETE", false);
    return value;
}

bool EnableSemanticStrictStaticKeep()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_SEMANTIC_STRICT_STATIC_KEEP", false);
    return value;
}

double GetSemanticSparseFlowMaxDynamicRejectRatio()
{
    const char* envValue =
        std::getenv("STSLAM_SEMANTIC_FLOW_MAX_DYNAMIC_REJECT_RATIO");
    static const double value =
        envValue ? std::min(1.0, std::max(0.0, std::atof(envValue))) : 0.15;
    return value;
}

double GetSemanticSparseFlowMaxForwardBackwardErrorPx()
{
    const char* envValue =
        std::getenv("STSLAM_SEMANTIC_FLOW_MAX_FORWARD_BACKWARD_ERROR_PX");
    static const double value = envValue ? std::max(0.1, std::atof(envValue)) : 1.5;
    return value;
}

double GetSemanticSparseFlowMaxEpipolarErrorPx()
{
    const char* envValue = std::getenv("STSLAM_SEMANTIC_FLOW_MAX_EPIPOLAR_ERROR_PX");
    static const double value = envValue ? std::max(0.1, std::atof(envValue)) : 2.0;
    return value;
}

double GetSemanticSparseFlowRansacThresholdPx()
{
    const char* envValue = std::getenv("STSLAM_SEMANTIC_FLOW_RANSAC_THRESHOLD_PX");
    static const double value = envValue ? std::max(0.1, std::atof(envValue)) : 2.0;
    return value;
}

int GetSemanticSparseFlowMinRansacInliers()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_SEMANTIC_FLOW_MIN_RANSAC_INLIERS", 30, 8);
    return value;
}

int GetSparseFlowGeometryLabel(const Frame& frame, const int idx)
{
    if(idx < 0 || idx >= static_cast<int>(frame.mvSparseFlowGeometryLabels.size()))
        return 0;
    return frame.mvSparseFlowGeometryLabels[idx];
}

bool EnableRgbdDynamicFrontendSplit()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_RGBD_DYNAMIC_FRONTEND_SPLIT", false);
    return value;
}

bool EnablePanopticSideChannelOnly()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_PANOPTIC_SIDE_CHANNEL_ONLY", false);
    return value;
}

bool EnableRgbdDynamicSplitObservationAppend()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_RGBD_DYNAMIC_SPLIT_APPEND_OBSERVATIONS", true);
    return value;
}

bool RequireMotionEvidenceForRgbdDynamicSplit()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_RGBD_DYNAMIC_SPLIT_REQUIRE_MOTION_EVIDENCE", true);
    return value;
}

bool RequireBackendMotionEvidenceForRgbdDynamicSplit()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_RGBD_DYNAMIC_SPLIT_REQUIRE_BACKEND_EVIDENCE", true);
    return value;
}

int GetRgbdDynamicSplitMinBackendMotionEvidence()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_RGBD_DYNAMIC_SPLIT_MIN_BACKEND_EVIDENCE", 1, 0);
    return value;
}

double GetRgbdDynamicSplitMinTranslation()
{
    const char* envValue = std::getenv("STSLAM_RGBD_DYNAMIC_SPLIT_MIN_TRANSLATION");
    static const double value = envValue ? std::max(0.0, std::atof(envValue)) : 0.01;
    return value;
}

double GetRgbdDynamicSplitMinRotationDeg()
{
    const char* envValue = std::getenv("STSLAM_RGBD_DYNAMIC_SPLIT_MIN_ROTATION_DEG");
    static const double value = envValue ? std::max(0.0, std::atof(envValue)) : 1.0;
    return value;
}

bool EnableRgbdDepthBackedDynamicObservations()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_RGBD_DEPTH_BACKED_DYNAMIC_OBSERVATIONS", false);
    return value;
}

bool PromoteRgbdDepthBackedObservationsToStructure()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_RGBD_DEPTH_BACKED_STRUCTURE_POINTS", true);
    return value;
}

int GetRgbdDepthBackedMaxPointsPerInstance()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_RGBD_DEPTH_BACKED_MAX_POINTS_PER_INSTANCE", 40, 1);
    return value;
}

float GetRgbdDepthBackedMaxDepth()
{
    const char* envValue = std::getenv("STSLAM_RGBD_DEPTH_BACKED_MAX_DEPTH");
    if(!envValue)
        return 6.0f;
    return std::max(0.1f, static_cast<float>(std::atof(envValue)));
}

struct SemanticGeometricVerificationResult
{
    bool checked = false;
    bool rescued = false;
    bool hasDepthMeasurement = false;
    bool hasMapPoint = false;
    bool lowStaticSupport = false;
    int mapPointObservations = 0;
    double reprojectionErrorPx = 0.0;
    double depthErrorM = 0.0;
    const char* rejectReason = "not_checked";
};

SemanticGeometricVerificationResult VerifyStaticMapPointWithGeometry(const Frame& frame,
                                                                     const int idx,
                                                                     const int minStaticObservations,
                                                                     const double maxReprojectionErrorPx,
                                                                     const double maxDepthErrorM)
{
    SemanticGeometricVerificationResult result;

    result.checked = true;

    if(idx < 0 || idx >= frame.N ||
       idx >= static_cast<int>(frame.mvpMapPoints.size()) ||
       idx >= static_cast<int>(frame.mvKeysUn.size()))
    {
        result.rejectReason = "feature_range";
        return result;
    }

    if(!frame.HasPose() || !frame.mpCamera)
    {
        result.rejectReason = "missing_pose_or_camera";
        return result;
    }

    MapPoint* pMP = frame.mvpMapPoints[idx];
    if(!pMP)
    {
        result.rejectReason = "missing_map_point";
        return result;
    }
    result.hasMapPoint = true;
    if(pMP->isBad())
    {
        result.rejectReason = "bad_map_point";
        return result;
    }
    if(pMP->GetInstanceId() > 0)
    {
        result.rejectReason = "dynamic_bound_map_point";
        return result;
    }
    result.mapPointObservations = pMP->Observations();
    result.lowStaticSupport = result.mapPointObservations < minStaticObservations;

    const Eigen::Vector3f pointWorld = pMP->GetWorldPos();
    if(!pointWorld.allFinite())
    {
        result.rejectReason = "invalid_world_point";
        return result;
    }

    const Sophus::SE3f Tcw = frame.GetPose();
    const Eigen::Vector3f pointCamera = Tcw * pointWorld;
    if(!pointCamera.allFinite() || pointCamera[2] <= 0.0f)
    {
        result.rejectReason = "invalid_camera_point";
        return result;
    }

    const Eigen::Vector3d pointCameraD = pointCamera.cast<double>();
    const Eigen::Vector2d projection = frame.mpCamera->project(pointCameraD);
    if(!projection.allFinite())
    {
        result.rejectReason = "invalid_projection";
        return result;
    }

    const Eigen::Vector2d observation(frame.mvKeysUn[idx].pt.x,
                                      frame.mvKeysUn[idx].pt.y);
    result.reprojectionErrorPx = (observation - projection).norm();
    if(result.reprojectionErrorPx > maxReprojectionErrorPx)
    {
        result.rejectReason = "reprojection_gate";
        return result;
    }

    if(idx < static_cast<int>(frame.mvDepth.size()) && frame.mvDepth[idx] > 0.0f)
    {
        result.hasDepthMeasurement = true;
        result.depthErrorM =
            std::fabs(static_cast<double>(frame.mvDepth[idx]) -
                      static_cast<double>(pointCamera[2]));
        if(result.depthErrorM > maxDepthErrorM)
        {
            result.rejectReason = "depth_gate";
            return result;
        }
    }

    if(result.lowStaticSupport)
    {
        result.rejectReason = "low_static_support";
        return result;
    }

    result.rescued = true;
    result.rejectReason = "rescued_static";
    return result;
}

SemanticGeometricVerificationResult VerifySemanticCandidateWithGeometry(const Frame& frame,
                                                                        const int idx)
{
    if(!EnableSemanticGeometricVerification())
        return SemanticGeometricVerificationResult();

    return VerifyStaticMapPointWithGeometry(frame,
                                            idx,
                                            GetSemanticGeometricMinStaticMapObservations(),
                                            GetSemanticGeometricMaxReprojectionErrorPx(),
                                            GetSemanticGeometricMaxDepthErrorM());
}

int ForceFilterDetectedDynamicFeatureMatches(Frame& frame, const std::string& stage)
{
    if(!ForceFilterDetectedDynamicFeaturesForStage(stage))
        return 0;

    const bool geometryVerificationEnabledForStage =
        EnableSemanticGeometricVerificationForStage(stage);
    const bool geometricDynamicRejectionEnabledForStage =
        EnableGeometricDynamicRejectionForStage(stage);
    int detectedInstanceFeatures = 0;
    int removedMatches = 0;
    int taggedOutliers = 0;
    int geometryChecked = 0;
    int geometryRescued = 0;
    int geometryRejectedMissingMapPoint = 0;
    int geometryRejectedDynamicBoundMapPoint = 0;
    int geometryRejectedReprojection = 0;
    int geometryRejectedDepth = 0;
    int geometryRejectedOther = 0;
    int geometryCandidateKept = 0;
    int geometryCandidateUndecided = 0;
    int sparseFlowChecked = 0;
    int sparseFlowStaticKept = 0;
    int sparseFlowDynamicRejected = 0;
    int sparseFlowDynamicCapped = 0;
    int sparseFlowDynamicRiskOnly = 0;
    int sparseFlowUnknown = 0;
    std::vector<double> rescuedReprojectionErrors;
    std::vector<double> rescuedDepthErrors;
    int geometryDynamicChecked = 0;
    int geometryDynamicRejectedReprojection = 0;
    int geometryDynamicRejectedDepth = 0;
    int geometryDynamicRejectedOther = 0;
    int geometryDynamicRemovedMatches = 0;
    const int nFeatures = std::min(frame.N, static_cast<int>(frame.mvpMapPoints.size()));
    int instanceFeatureTotal = 0;
    for(int idx = 0; idx < nFeatures; ++idx)
    {
        if(frame.GetFeatureInstanceId(static_cast<size_t>(idx)) > 0)
            ++instanceFeatureTotal;
    }
    const bool semanticCandidateGeometryGate =
        EnableSemanticCandidateGeometryGate() && geometryVerificationEnabledForStage;
    const bool sparseFlowGate =
        semanticCandidateGeometryGate && EnableSemanticCandidateSparseFlowGateForStage(stage);
    const bool conservativeDynamicDelete =
        semanticCandidateGeometryGate && EnableSemanticConservativeDynamicDelete();
    const bool strictStaticKeep =
        semanticCandidateGeometryGate && EnableSemanticStrictStaticKeep();
    const int maxSparseFlowDynamicRejects =
        sparseFlowGate
            ? static_cast<int>(std::floor(instanceFeatureTotal *
                                          GetSemanticSparseFlowMaxDynamicRejectRatio()))
            : 0;
    for(int idx = 0; idx < nFeatures; ++idx)
    {
        const int instanceId = frame.GetFeatureInstanceId(static_cast<size_t>(idx));
        if(instanceId <= 0)
            continue;

        ++detectedInstanceFeatures;
        const SemanticGeometricVerificationResult geomResult =
            geometryVerificationEnabledForStage
                ? VerifySemanticCandidateWithGeometry(frame, idx)
                : SemanticGeometricVerificationResult();
        if(geomResult.checked)
        {
            ++geometryChecked;
            if(geomResult.rescued)
            {
                ++geometryRescued;
                rescuedReprojectionErrors.push_back(geomResult.reprojectionErrorPx);
                if(geomResult.hasDepthMeasurement)
                    rescuedDepthErrors.push_back(geomResult.depthErrorM);
                ++geometryCandidateKept;
                continue;
            }

            const std::string rejectReason = geomResult.rejectReason;
            if(rejectReason == "missing_map_point")
                ++geometryRejectedMissingMapPoint;
            else if(rejectReason == "dynamic_bound_map_point")
                ++geometryRejectedDynamicBoundMapPoint;
            else if(rejectReason == "reprojection_gate")
                ++geometryRejectedReprojection;
            else if(rejectReason == "depth_gate")
                ++geometryRejectedDepth;
            else
                ++geometryRejectedOther;

            if(semanticCandidateGeometryGate)
            {
                const bool geometryAnomaly =
                    rejectReason == "reprojection_gate" ||
                    rejectReason == "depth_gate";
                const bool dynamicBoundMapPoint =
                    rejectReason == "dynamic_bound_map_point";
                const bool insufficientStaticSupport =
                    geomResult.lowStaticSupport ||
                    rejectReason == "missing_map_point";

                if(strictStaticKeep)
                {
                    int flowLabel = 0;
                    if(sparseFlowGate)
                    {
                        flowLabel = GetSparseFlowGeometryLabel(frame, idx);
                        if(flowLabel > 0)
                        {
                            ++sparseFlowChecked;
                            ++sparseFlowStaticKept;
                        }
                        else if(flowLabel < 0)
                        {
                            ++sparseFlowChecked;
                            ++sparseFlowDynamicRiskOnly;
                        }
                        else
                            ++sparseFlowUnknown;
                    }

                    const bool allowSparseStaticKeep =
                        flowLabel > 0 &&
                        (rejectReason == "missing_map_point" ||
                         rejectReason == "low_static_support");
                    if(allowSparseStaticKeep)
                    {
                        ++geometryCandidateUndecided;
                        continue;
                    }
                    if(flowLabel < 0)
                        ++sparseFlowDynamicRejected;
                }
                else if(conservativeDynamicDelete)
                {
                    int flowLabel = 0;
                    if(sparseFlowGate)
                    {
                        flowLabel = GetSparseFlowGeometryLabel(frame, idx);
                        if(flowLabel > 0)
                        {
                            ++sparseFlowChecked;
                            ++sparseFlowStaticKept;
                        }
                        else if(flowLabel < 0)
                        {
                            ++sparseFlowChecked;
                            ++sparseFlowDynamicRiskOnly;
                        }
                        else
                            ++sparseFlowUnknown;
                    }

                    const bool rejectAsDynamic =
                        dynamicBoundMapPoint ||
                        (geometryAnomaly && insufficientStaticSupport && flowLabel <= 0);
                    if(!rejectAsDynamic)
                    {
                        ++geometryCandidateUndecided;
                        continue;
                    }
                    if(flowLabel < 0)
                        ++sparseFlowDynamicRejected;
                }
                else
                {
                    const bool rejectAsDynamic =
                    rejectReason == "reprojection_gate" ||
                    rejectReason == "depth_gate" ||
                    rejectReason == "dynamic_bound_map_point";
                    if(!rejectAsDynamic)
                    {
                        if(sparseFlowGate)
                        {
                            const int flowLabel = GetSparseFlowGeometryLabel(frame, idx);
                            if(flowLabel < 0)
                            {
                                ++sparseFlowChecked;
                                if(sparseFlowDynamicRejected >= maxSparseFlowDynamicRejects)
                                {
                                    ++geometryCandidateUndecided;
                                    ++sparseFlowDynamicCapped;
                                    continue;
                                }
                                ++sparseFlowDynamicRejected;
                            }
                            else
                            {
                                ++geometryCandidateUndecided;
                                if(flowLabel > 0)
                                {
                                    ++sparseFlowChecked;
                                    ++sparseFlowStaticKept;
                                }
                                else
                                    ++sparseFlowUnknown;
                                continue;
                            }
                        }
                        else
                        {
                            ++geometryCandidateUndecided;
                            continue;
                        }
                    }
                }
            }
        }
        else if(semanticCandidateGeometryGate)
        {
            if(strictStaticKeep)
            {
                int flowLabel = 0;
                if(sparseFlowGate)
                {
                    flowLabel = GetSparseFlowGeometryLabel(frame, idx);
                    if(flowLabel > 0)
                    {
                        ++sparseFlowChecked;
                        ++sparseFlowStaticKept;
                        ++geometryCandidateUndecided;
                        continue;
                    }
                    else if(flowLabel < 0)
                    {
                        ++sparseFlowChecked;
                        ++sparseFlowDynamicRiskOnly;
                        ++sparseFlowDynamicRejected;
                    }
                    else
                        ++sparseFlowUnknown;
                }
            }
            else if(conservativeDynamicDelete)
            {
                int flowLabel = 0;
                if(sparseFlowGate)
                {
                    flowLabel = GetSparseFlowGeometryLabel(frame, idx);
                    if(flowLabel > 0)
                    {
                        ++sparseFlowChecked;
                        ++sparseFlowStaticKept;
                    }
                    else if(flowLabel < 0)
                    {
                        ++sparseFlowChecked;
                        ++sparseFlowDynamicRiskOnly;
                    }
                    else
                        ++sparseFlowUnknown;
                }
                ++geometryCandidateUndecided;
                continue;
            }
            else if(sparseFlowGate)
            {
                const int flowLabel = GetSparseFlowGeometryLabel(frame, idx);
                if(flowLabel < 0)
                {
                    ++sparseFlowChecked;
                    if(sparseFlowDynamicRejected >= maxSparseFlowDynamicRejects)
                    {
                        ++geometryCandidateUndecided;
                        ++sparseFlowDynamicCapped;
                        continue;
                    }
                    ++sparseFlowDynamicRejected;
                }
                else
                {
                    ++geometryCandidateUndecided;
                    if(flowLabel > 0)
                    {
                        ++sparseFlowChecked;
                        ++sparseFlowStaticKept;
                    }
                    else
                        ++sparseFlowUnknown;
                    continue;
                }
            }
            else
            {
                ++geometryCandidateUndecided;
                continue;
            }
        }

        if(frame.mvpMapPoints[idx])
        {
            frame.mvpMapPoints[idx] = static_cast<MapPoint*>(NULL);
            ++removedMatches;
        }
        if(idx < static_cast<int>(frame.mvbOutlier.size()) && !frame.mvbOutlier[idx])
        {
            frame.mvbOutlier[idx] = true;
            ++taggedOutliers;
        }
    }

    if(geometricDynamicRejectionEnabledForStage)
    {
        for(int idx = 0; idx < nFeatures; ++idx)
        {
            if(frame.GetFeatureInstanceId(static_cast<size_t>(idx)) > 0)
                continue;
            if(!frame.mvpMapPoints[idx])
                continue;

            const SemanticGeometricVerificationResult geomResult =
                VerifyStaticMapPointWithGeometry(frame,
                                                 idx,
                                                 GetGeometricDynamicRejectionMinStaticMapObservations(),
                                                 GetGeometricDynamicRejectionMaxReprojectionErrorPx(),
                                                 GetGeometricDynamicRejectionMaxDepthErrorM());
            if(!geomResult.checked)
                continue;

            ++geometryDynamicChecked;
            bool rejectAsDynamic = false;
            const std::string rejectReason = geomResult.rejectReason;
            if(rejectReason == "reprojection_gate")
            {
                ++geometryDynamicRejectedReprojection;
                rejectAsDynamic = true;
            }
            else if(rejectReason == "depth_gate")
            {
                ++geometryDynamicRejectedDepth;
                rejectAsDynamic = true;
            }
            else if(rejectReason != "rescued_static" &&
                    rejectReason != "low_static_support" &&
                    rejectReason != "missing_map_point")
            {
                ++geometryDynamicRejectedOther;
            }

            if(!rejectAsDynamic)
                continue;

            frame.mvpMapPoints[idx] = static_cast<MapPoint*>(NULL);
            ++geometryDynamicRemovedMatches;
            if(idx < static_cast<int>(frame.mvbOutlier.size()) && !frame.mvbOutlier[idx])
                frame.mvbOutlier[idx] = true;
        }
    }

    if(detectedInstanceFeatures > 0 || geometryDynamicRemovedMatches > 0)
    {
        std::cout << "[STSLAM_FORCE_DYNAMIC_FILTER]"
                  << " frame=" << frame.mnId
                  << " stage=" << stage
                  << " detected_instance_features=" << detectedInstanceFeatures
                  << " geom_verify_enabled=" << (EnableSemanticGeometricVerification() ? 1 : 0)
                  << " geom_stage_enabled=" << (geometryVerificationEnabledForStage ? 1 : 0)
                  << " semantic_candidate_geometry_gate="
                  << (semanticCandidateGeometryGate ? 1 : 0)
                  << " conservative_dynamic_delete="
                  << (conservativeDynamicDelete ? 1 : 0)
                  << " strict_static_keep="
                  << (strictStaticKeep ? 1 : 0)
                  << " geom_checked=" << geometryChecked
                  << " geom_rescued=" << geometryRescued
                  << " geom_candidate_kept=" << geometryCandidateKept
                  << " geom_candidate_undecided=" << geometryCandidateUndecided
                  << " sparse_flow_gate=" << (sparseFlowGate ? 1 : 0)
                  << " sparse_flow_checked=" << sparseFlowChecked
                  << " sparse_flow_static_kept=" << sparseFlowStaticKept
                  << " sparse_flow_dynamic_rejected=" << sparseFlowDynamicRejected
                  << " sparse_flow_dynamic_capped=" << sparseFlowDynamicCapped
                  << " sparse_flow_max_dynamic_rejects=" << maxSparseFlowDynamicRejects
                  << " sparse_flow_dynamic_risk_only=" << sparseFlowDynamicRiskOnly
                  << " sparse_flow_unknown=" << sparseFlowUnknown
                  << " geom_rejected_missing_map_point=" << geometryRejectedMissingMapPoint
                  << " geom_rejected_dynamic_bound_map_point=" << geometryRejectedDynamicBoundMapPoint
                  << " geom_rejected_reprojection=" << geometryRejectedReprojection
                  << " geom_rejected_depth=" << geometryRejectedDepth
                  << " geom_rejected_other=" << geometryRejectedOther
                  << " rescued_reprojection_px_median="
                  << (rescuedReprojectionErrors.empty() ? 0.0 : MedianValue(rescuedReprojectionErrors))
                  << " rescued_depth_error_m_median="
                  << (rescuedDepthErrors.empty() ? 0.0 : MedianValue(rescuedDepthErrors))
                  << " removed_matches=" << removedMatches
                  << " tagged_outliers=" << taggedOutliers
                  << " geom_dyn_reject_enabled=" << (geometricDynamicRejectionEnabledForStage ? 1 : 0)
                  << " geom_dyn_checked=" << geometryDynamicChecked
                  << " geom_dyn_rejected_reprojection=" << geometryDynamicRejectedReprojection
                  << " geom_dyn_rejected_depth=" << geometryDynamicRejectedDepth
                  << " geom_dyn_rejected_other=" << geometryDynamicRejectedOther
                  << " geom_dyn_removed_matches=" << geometryDynamicRemovedMatches
                  << std::endl;
    }

    return removedMatches + geometryDynamicRemovedMatches;
}

bool UseStrictPaperArchitectureDefaults()
{
    const char* envValue = std::getenv("STSLAM_MODULE8_PROFILE");
    if(!envValue || std::string(envValue).empty())
        return true;

    const std::string profile(envValue);
    return profile == "paper_strict" || profile == "paper_eq16";
}

int GetInstancePredictionWarmupFrames()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_INSTANCE_PREDICTION_WARMUP_FRAMES", 2, 0);
    return value;
}

int GetInstancePredictionMinSupport()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_INSTANCE_PREDICTION_MIN_SUPPORT", 8, 1);
    return value;
}

int GetInstanceStaticMotionConfirmFrames()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_INSTANCE_STATIC_CONFIRM_FRAMES", 2, 1);
    return value;
}

int GetInstanceDynamicMotionConfirmFrames()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_INSTANCE_DYNAMIC_CONFIRM_FRAMES", 2, 1);
    return value;
}

double GetEnvDoubleOrDefault(const char* name, const double defaultValue, const double minValue)
{
    const char* envValue = std::getenv(name);
    if(!envValue)
        return defaultValue;

    char* endPtr = NULL;
    const double value = std::strtod(envValue, &endPtr);
    if(endPtr == envValue || !std::isfinite(value))
        return defaultValue;

    return std::max(minValue, value);
}

bool EnableInstanceResidualMotionGate()
{
    return GetEnvFlagOrDefault("STSLAM_INSTANCE_RESIDUAL_MOTION_GATE",
                               !UseStrictPaperArchitectureDefaults());
}

bool EnablePanopticRefinementDescriptorGate()
{
    const char* envValue = std::getenv("STSLAM_PANOPTIC_REFINEMENT_DESCRIPTOR_GATE");
    return envValue && std::string(envValue) != "0";
}

int GetPanopticRefinementDescriptorMaxDistance()
{
    static const int value =
        static_cast<int>(std::round(GetEnvDoubleOrDefault("STSLAM_PANOPTIC_REFINEMENT_DESCRIPTOR_MAX_DISTANCE",
                                                          ORBmatcher::TH_LOW,
                                                          0.0)));
    return value;
}

bool DebugInstanceResidualMotionGate()
{
    const char* envValue = std::getenv("STSLAM_DEBUG_INSTANCE_MOTION_GATE");
    return envValue && std::string(envValue) != "0";
}

double GetInstanceDynamicResidualRatio()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_INSTANCE_DYNAMIC_RESIDUAL_RATIO", 0.85, 0.0);
    return value;
}

double GetInstanceStaticVelocityDecay()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_INSTANCE_STATIC_VELOCITY_DECAY", 0.35, 0.0);
    return std::min(1.0, value);
}

double GetInstanceInitializationSVDInlierScale()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_INSTANCE_INIT_SVD_INLIER_SCALE", 2.5, 1.0);
    return value;
}

bool EnableInstanceTrackletMotionCoherenceGate()
{
    return GetEnvFlagOrDefault("STSLAM_INSTANCE_TRACKLET_MOTION_COHERENCE_GATE",
                               !UseStrictPaperArchitectureDefaults());
}

double GetInstanceTrackletMotionMadScale()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_INSTANCE_TRACKLET_MOTION_MAD_SCALE", 3.5, 0.0);
    return value;
}

double GetInstanceTrackletMotionMinGatePx()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_INSTANCE_TRACKLET_MOTION_MIN_GATE_PX", 8.0, 0.0);
    return value;
}

bool EnableInstanceTrackletStrictDescriptorGate()
{
    const char* envValue = std::getenv("STSLAM_INSTANCE_TRACKLET_STRICT_DESCRIPTOR_GATE");
    return envValue && std::string(envValue) != "0";
}

double GetInstanceTrackletDescriptorRatio()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_INSTANCE_TRACKLET_DESCRIPTOR_RATIO", 0.8, 0.0);
    return value;
}

int GetInstanceTrackletDescriptorMaxDistance()
{
    static const int value =
        static_cast<int>(std::round(GetEnvDoubleOrDefault("STSLAM_INSTANCE_TRACKLET_DESCRIPTOR_MAX_DISTANCE",
                                                          ORBmatcher::TH_LOW,
                                                          0.0)));
    return value;
}

bool EnableInstanceIdStabilityGate()
{
    return GetEnvFlagOrDefault("STSLAM_INSTANCE_ID_STABILITY_GATE",
                               UseStrictPaperArchitectureDefaults());
}

double GetInstanceIdMinTrackletRetention()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_INSTANCE_ID_MIN_TRACKLET_RETENTION", 0.05, 0.0);
    return std::min(1.0, value);
}

double GetInstanceIdMaxBboxCenterShift()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_INSTANCE_ID_MAX_BBOX_CENTER_SHIFT", 0.55, 0.0);
    return value;
}

double GetInstanceIdMaxAreaRatio()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_INSTANCE_ID_MAX_AREA_RATIO", 8.0, 1.0);
    return value;
}

double GetInstanceIdBboxAreaJumpHardRejectMaxIoU()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_INSTANCE_ID_BBOX_AREA_JUMP_HARD_REJECT_MAX_IOU", 0.2, 0.0);
    return std::min(1.0, value);
}

double GetInstanceIdAreaJumpHardRejectMinRetention()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_INSTANCE_ID_AREA_JUMP_HARD_REJECT_MIN_RETENTION", 0.35, 0.0);
    return std::min(1.0, value);
}

double GetInstanceIdBboxAreaJumpHardRejectMaxCenterShift()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_INSTANCE_ID_BBOX_AREA_JUMP_HARD_REJECT_MAX_CENTER_SHIFT", 0.29, 0.0);
    return value;
}

double GetInstanceIdMaxFeatureDensityRatio()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_INSTANCE_ID_MAX_FEATURE_DENSITY_RATIO", 12.0, 1.0);
    return value;
}

double GetInstanceIdMinTrackletCoverage()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_INSTANCE_ID_MIN_TRACKLET_COVERAGE", 0.003, 0.0);
    return std::min(1.0, value);
}

double GetInstanceIdMaxFlowAccelerationNormalized()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_INSTANCE_ID_MAX_FLOW_ACCEL_NORM", 0.75, 0.0);
    return value;
}

double GetInstanceIdMinFlowDirectionCosine()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_INSTANCE_ID_MIN_FLOW_DIRECTION_COS", -0.25, -1.0);
    return std::min(1.0, value);
}

double GetInstanceIdMinTriangulationSuccessRatio()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_INSTANCE_ID_MIN_TRIANGULATION_SUCCESS_RATIO", 0.12, 0.0);
    return std::min(1.0, value);
}

double GetInstanceIdMaxTriangulationInvalidRatio()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_INSTANCE_ID_MAX_TRIANGULATION_INVALID_RATIO", 0.80, 0.0);
    return std::min(1.0, value);
}

int GetInstanceIdGeometryTriangulationSampleLimit()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_INSTANCE_ID_GEOM_TRIANGULATION_SAMPLE_LIMIT", 80, 0);
    return value;
}

double GetInstanceInitializationMinInlierRatio()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_INSTANCE_INIT_MIN_INLIER_RATIO", 0.5, 0.0);
    return value;
}

double GetInstanceInitializationMaxDynamicRmsePx()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_INSTANCE_INIT_MAX_DYNAMIC_RMSE_PX", 8.0, 0.0);
    return value;
}

double GetInstanceInitializationMaxDynamicRotationDeg()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_INSTANCE_INIT_MAX_DYNAMIC_ROTATION_DEG", 45.0, 0.0);
    return value;
}

int GetInstanceInitializationDynamicConfirmFrames()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_INSTANCE_INIT_DYNAMIC_CONFIRM_FRAMES", 2, 1);
    return value;
}

bool EnableInstanceInitializationTranslationOnlyCandidate()
{
    return GetEnvFlagOrDefault("STSLAM_INSTANCE_INIT_TRANSLATION_ONLY_CANDIDATE",
                               !UseStrictPaperArchitectureDefaults());
}

bool EnableInstanceInitializationTranslationOnlyDynamicPromotion()
{
    const char* envValue = std::getenv("STSLAM_INSTANCE_INIT_TRANSLATION_ONLY_PROMOTE_DYNAMIC");
    return envValue && std::string(envValue) != "0";
}

bool EnableStrictStaticZeroVelocityInitialization()
{
    return GetEnvFlagOrDefault("STSLAM_STRICT_STATIC_ZERO_VELOCITY_INIT",
                               UseStrictPaperArchitectureDefaults());
}

bool EnableStrictUncertainCentroidMotionInitialization()
{
    return GetEnvFlagOrDefault("STSLAM_STRICT_UNCERTAIN_CENTROID_MOTION_INIT",
                               UseStrictPaperArchitectureDefaults());
}

double GetStrictStaticZeroVelocityMaxTranslation()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_STRICT_STATIC_ZERO_MAX_TRANSLATION", 0.03, 0.0);
    return value;
}

double GetStrictStaticZeroVelocityMaxRotationDeg()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_STRICT_STATIC_ZERO_MAX_ROTATION_DEG", 5.0, 0.0);
    return value;
}

double GetStrictStaticZeroVelocityMaxReprojectionRmsePx()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_STRICT_STATIC_ZERO_MAX_REPROJ_RMSE_PX", 8.0, 0.0);
    return value;
}

double GetStrictStaticZeroVelocityStrongMaxReprojectionRmsePx()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_STRICT_STATIC_ZERO_STRONG_MAX_REPROJ_RMSE_PX", 3.0, 0.0);
    return value;
}

bool EnableZeroVelocityDynamicReactivation()
{
    return GetEnvFlagOrDefault("STSLAM_ZERO_VELOCITY_DYNAMIC_REACTIVATION",
                               UseStrictPaperArchitectureDefaults());
}

int GetZeroVelocityDynamicReactivationMinTracklets()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_ZERO_VELOCITY_REACTIVATION_MIN_TRACKLETS", 8, 1);
    return value;
}

bool EnableStrictSvdMotionReliability()
{
    return GetEnvFlagOrDefault("STSLAM_STRICT_SVD_MOTION_RELIABILITY",
                               UseStrictPaperArchitectureDefaults());
}

double GetStrictSvdMotionMaxCondition()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_STRICT_SVD_MOTION_MAX_CONDITION", 100.0, 1.0);
    return value;
}

double GetStrictSvdMotionMinPlanarityRatio()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_STRICT_SVD_MOTION_MIN_PLANARITY_RATIO", 0.01, 0.0);
    return value;
}

double GetStrictSvdMotionMaxReprojectionRmsePx()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_STRICT_SVD_MOTION_MAX_REPROJECTION_RMSE_PX", 80.0, 0.0);
    return value;
}

bool DebugInstanceInitialization()
{
    const char* envValue = std::getenv("STSLAM_DEBUG_INSTANCE_INIT");
    return envValue && std::string(envValue) != "0";
}

bool DebugInstanceTargetMatches(const int instanceId)
{
    const char* envValue = std::getenv("STSLAM_DEBUG_INSTANCE_ID");
    if(!envValue || std::string(envValue).empty())
        return true;

    std::stringstream stream(envValue);
    std::string token;
    while(std::getline(stream, token, ','))
    {
        if(!token.empty() && std::atoi(token.c_str()) == instanceId)
            return true;
    }
    return false;
}

bool DebugInstanceInitializationFor(const int instanceId)
{
    return DebugInstanceInitialization() && DebugInstanceTargetMatches(instanceId);
}

bool DebugInstanceLifecycleFor(const int instanceId)
{
    const char* envValue = std::getenv("STSLAM_DEBUG_INSTANCE_LIFECYCLE");
    return envValue && std::string(envValue) != "0" && DebugInstanceTargetMatches(instanceId);
}

bool DebugTriangulationQuality()
{
    const char* envValue = std::getenv("STSLAM_DEBUG_TRIANGULATION_QUALITY");
    return envValue && std::string(envValue) != "0";
}

bool DebugInstanceTracklets()
{
    const char* envValue = std::getenv("STSLAM_DEBUG_INSTANCE_TRACKLETS");
    return envValue && std::string(envValue) != "0";
}

bool DebugDynamicObservationSupply()
{
    const char* envValue = std::getenv("STSLAM_DEBUG_DYNAMIC_SUPPLY");
    return envValue && std::string(envValue) != "0";
}

bool EnableDynamicSupplyDescriptorMatching()
{
    const char* envValue = std::getenv("STSLAM_DYNAMIC_SUPPLY_DESCRIPTOR_MATCHING");
    return envValue && std::string(envValue) != "0";
}

bool EnableDynamicSupplyProjectionMatching()
{
    return GetEnvFlagOrDefault("STSLAM_DYNAMIC_SUPPLY_PROJECTION_MATCHING", true);
}

bool RequireDynamicSupplyProjectionGateSupport()
{
    const char* envValue = std::getenv("STSLAM_DYNAMIC_SUPPLY_PROJECTION_REQUIRE_GATE_SUPPORT");
    return envValue && std::string(envValue) != "0";
}

bool EnableInstanceTrackletStructurePoints()
{
    const char* envValue = std::getenv("STSLAM_INSTANCE_TRACKLETS_AS_STRUCTURE_POINTS");
    return !envValue || std::string(envValue) != "0";
}

bool RegisterInstanceStructurePointsToMap()
{
    return GetEnvFlagOrDefault("STSLAM_REGISTER_INSTANCE_STRUCTURE_POINTS_TO_MAP",
                               UseStrictPaperArchitectureDefaults());
}

bool RegisterRgbdInstanceStructurePointsToMap()
{
    return GetEnvFlagOrDefault("STSLAM_RGBD_REGISTER_INSTANCE_STRUCTURE_POINTS_TO_MAP",
                               false);
}

bool KeepInstanceStructurePointsOutOfStaticKeyFrameSlots()
{
    return GetEnvFlagOrDefault("STSLAM_KEEP_INSTANCE_STRUCTURE_POINTS_OUT_OF_STATIC_KF",
                               UseStrictPaperArchitectureDefaults());
}

int GetDynamicSupplyDescriptorThreshold()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_SUPPLY_DESCRIPTOR_THRESHOLD", 70, 1);
    return value;
}

double GetDynamicSupplyProjectionMaxError()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_SUPPLY_PROJECTION_MAX_ERROR", 8.0, 0.0);
    return value;
}

double GetDynamicSupplyDescriptorRatio()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_SUPPLY_DESCRIPTOR_RATIO", 0.8, 0.0);
    return std::min(1.0, value);
}

double GetDynamicSupplyMaxReprojectionError()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_SUPPLY_MAX_REPROJ_ERROR", 12.0, 0.0);
    return value;
}

bool EnableDynamicSupplyInstanceQualityGate()
{
    return GetEnvFlagOrDefault("STSLAM_DYNAMIC_SUPPLY_INSTANCE_QUALITY_GATE",
                               UseStrictPaperArchitectureDefaults());
}

int GetDynamicSupplyGateMinSupport()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_SUPPLY_GATE_MIN_SUPPORT", 3, 1);
    return value;
}

int GetDynamicSupplyGateDescriptorMinSupport()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_SUPPLY_GATE_DESCRIPTOR_MIN_SUPPORT", 8, 1);
    return value;
}

double GetDynamicSupplyGateInlierThreshold()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_SUPPLY_GATE_INLIER_REPROJ_PX", 8.0, 0.0);
    return value;
}

double GetDynamicSupplyGateMaxMeanError()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_SUPPLY_GATE_MAX_MEAN_REPROJ_PX", 4.0, 0.0);
    return value;
}

double GetDynamicSupplyGateMinInlierRatio()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_SUPPLY_GATE_MIN_INLIER_RATIO", 0.6, 0.0);
    return std::min(1.0, value);
}

double GetDynamicSupplySparseDirectWeight()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_SUPPLY_SPARSE_DIRECT_WEIGHT", 0.5, 0.05);
    return std::min(1.0, value);
}

int GetDynamicSupplyPriorFailureBlockMinEvidence()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_SUPPLY_PRIOR_FAILURE_BLOCK_MIN_EVIDENCE", 3, 1);
    return value;
}

bool EnableTriangulationQualityGate()
{
    return GetEnvFlagOrDefault("STSLAM_TRIANGULATION_QUALITY_GATE",
                               UseStrictPaperArchitectureDefaults());
}

bool EnableTriangulationQualityWeights()
{
    return GetEnvFlagOrDefault("STSLAM_TRIANGULATION_QUALITY_WEIGHTS",
                               UseStrictPaperArchitectureDefaults());
}

double GetTriangulationMinQualityWeight()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_TRIANGULATION_MIN_QUALITY_WEIGHT", 0.05, 0.0);
    return std::min(1.0, value);
}

double GetTriangulationMinDepth()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_TRIANGULATION_MIN_DEPTH",
                              UseStrictPaperArchitectureDefaults() ? 0.0 : 0.01,
                              0.0);
    return value;
}

double GetTriangulationMaxReprojectionError()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_TRIANGULATION_MAX_REPROJ_ERROR", 5.0, 0.0);
    return value;
}

double GetTriangulationMinParallaxDeg()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_TRIANGULATION_MIN_PARALLAX_DEG", 0.0, 0.0);
    return value;
}

double GetTriangulationMaxParallaxDeg()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_TRIANGULATION_MAX_PARALLAX_DEG", 0.0, 0.0);
    return value;
}

double GetTriangulationMaxDisparity()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_TRIANGULATION_MAX_DISPARITY_PX", 0.0, 0.0);
    return value;
}

double GetTriangulationQualityParallaxDeg()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_TRIANGULATION_QUALITY_PARALLAX_DEG", 1.0, 0.0);
    return value;
}

bool EnableTriFrameConsistencyWeights()
{
    return GetEnvFlagOrDefault("STSLAM_TRIFRAME_CONSISTENCY_WEIGHTS",
                               UseStrictPaperArchitectureDefaults());
}

double GetTriFrameConsistencyMaxReprojectionPx()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_TRIFRAME_CONSISTENCY_MAX_REPROJ_PX", 8.0, 0.0);
    return value;
}

double GetTriFrameConsistencyMinWeight()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_TRIFRAME_CONSISTENCY_MIN_WEIGHT", 0.10, 0.0);
    return std::min(1.0, value);
}

bool EnableTriFrameConsistencyStructurePromotionGate()
{
    return GetEnvFlagOrDefault("STSLAM_TRIFRAME_CONSISTENCY_STRUCTURE_PROMOTION_GATE",
                               UseStrictPaperArchitectureDefaults());
}

bool EnableTriFrameTrackletCandidateQuality()
{
    return GetEnvFlagOrDefault("STSLAM_TRIFRAME_CANDIDATE_QUALITY",
                               UseStrictPaperArchitectureDefaults());
}

bool EnableTriFrameMultiPairTriangulation()
{
    return GetEnvFlagOrDefault("STSLAM_TRIFRAME_MULTI_PAIR_TRIANGULATION",
                               UseStrictPaperArchitectureDefaults());
}

double GetTriFrameMultiPairMaxStaticReprojectionPx()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_TRIFRAME_MULTI_PAIR_MAX_STATIC_REPROJ_PX",
                              GetTriFrameConsistencyMaxReprojectionPx(),
                              0.0);
    return value;
}

double GetTriFrameMultiPairFallbackWeight()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_TRIFRAME_MULTI_PAIR_FALLBACK_WEIGHT", 0.25, 0.0);
    return std::min(1.0, value);
}

double GetTriFrameCandidateMaxFlowAccelerationNormalized()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_TRIFRAME_CANDIDATE_MAX_FLOW_ACCEL_NORM",
                              GetInstanceIdMaxFlowAccelerationNormalized(),
                              0.0);
    return value;
}

double GetTriFrameCandidateMinStructureWeight()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_TRIFRAME_CANDIDATE_MIN_STRUCTURE_WEIGHT", 0.5, 0.0);
    return std::min(1.0, value);
}

bool RequireBackendEvidenceForPanopticPose()
{
    const char* envValue = std::getenv("STSLAM_PANOPTIC_POSE_REQUIRE_BACKEND_EVIDENCE");
    return !envValue || std::string(envValue) != "0";
}

int GetPanopticPoseMinBackendEvidence()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_PANOPTIC_POSE_MIN_BACKEND_EVIDENCE", 1, 0);
    return value;
}

bool EnableDynamicTrackletMaterialization()
{
    return GetEnvFlagOrDefault("STSLAM_ENABLE_DYNAMIC_TRACKLET_MATERIALIZATION",
                               UseStrictPaperArchitectureDefaults());
}

bool EnableRgbdDynamicTrackletMaterialization()
{
    return GetEnvFlagOrDefault("STSLAM_RGBD_ENABLE_DYNAMIC_TRACKLET_MATERIALIZATION",
                               false);
}

bool PropagateUncertainDynamicEntityState()
{
    return GetEnvFlagOrDefault("STSLAM_PROPAGATE_UNCERTAIN_DYNAMIC_ENTITY_STATE",
                               UseStrictPaperArchitectureDefaults());
}

double GetUncertainDynamicEntityPredictionMinConfidence()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_UNCERTAIN_DYNAMIC_ENTITY_PREDICTION_MIN_CONFIDENCE",
                              0.30,
                              0.0);
    return std::min(1.0, value);
}

int GetInstanceMapWarmupFrames()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_INSTANCE_MAP_WARMUP_FRAMES", 10, 0);
    return value;
}

bool HasMatureInstancePrediction(const Instance* pInstance,
                                 const int frameId,
                                 const int matchedMapPointCount)
{
    if(!pInstance || !pInstance->IsInitialized())
        return false;

    const int initializedFrame = pInstance->GetInitializedFrame();
    if(initializedFrame < 0 ||
       frameId < initializedFrame + GetInstancePredictionWarmupFrames())
    {
        return false;
    }

    const int minSupport = GetInstancePredictionMinSupport();
    if(matchedMapPointCount >= minSupport)
        return true;

    Instance::InstanceMotionStateRecord latestMotionState;
    const bool hasLatestMotionState =
        pInstance->GetLatestInstanceMotionState(latestMotionState) &&
        latestMotionState.state != Instance::kDynamicEntityUnknown &&
        latestMotionState.frameId >= static_cast<unsigned long>(initializedFrame) &&
        latestMotionState.frameId <= static_cast<unsigned long>(frameId);
    const bool hasEnoughInstanceStructure =
        static_cast<int>(pInstance->NumMapPoints()) >= minSupport;
    const bool hasBackendMotionEvidence =
        pInstance->GetBackendMotionEvidence() >= 1;
    const bool hasUncertainDynamicEntityCandidate =
        UseStrictPaperArchitectureDefaults() &&
        latestMotionState.state == Instance::kUncertainDynamicEntity &&
        latestMotionState.confidence >= GetUncertainDynamicEntityPredictionMinConfidence();

    return hasLatestMotionState &&
           hasEnoughInstanceStructure &&
           (latestMotionState.reliable ||
            hasBackendMotionEvidence ||
            hasUncertainDynamicEntityCandidate);
}

enum InstanceMotionGateState
{
    kInstanceMotionUncertain = 0,
    kInstanceMotionStatic = 1,
    kInstanceMotionDynamic = 2
};

struct InstanceMotionGateResult
{
    bool valid = false;
    bool useDynamicMotion = true;
    InstanceMotionGateState state = kInstanceMotionUncertain;
    int support = 0;
    double staticMeanError = 0.0;
    double dynamicMeanError = 0.0;
};

struct InstanceSvdMotionDiagnostics
{
    bool valid = false;
    int support = 0;
    double totalWeight = 0.0;
    double singular0 = 0.0;
    double singular1 = 0.0;
    double singular2 = 0.0;
    double condition = 0.0;
    double planarityRatio = 0.0;
    double linearityRatio = 0.0;
    double srcStdRadius = 0.0;
    double dstStdRadius = 0.0;
    double meanDisplacement = 0.0;
    double medianDisplacement = 0.0;
    double maxDisplacement = 0.0;
    double meanResidual = 0.0;
    double medianResidual = 0.0;
    double maxResidual = 0.0;
};

const char* InstanceMotionGateStateName(const InstanceMotionGateState state)
{
    if(state == kInstanceMotionDynamic)
        return "dynamic";
    if(state == kInstanceMotionStatic)
        return "static";
    return "uncertain";
}

const char* DynamicEntityMotionStateName(const Instance::DynamicEntityMotionState state)
{
    if(state == Instance::kMovingDynamicEntity)
        return "moving_dynamic_entity";
    if(state == Instance::kZeroVelocityDynamicEntity)
        return "zero_velocity_dynamic_entity";
    if(state == Instance::kUncertainDynamicEntity)
        return "uncertain_dynamic_entity";
    return "unknown_dynamic_entity";
}

std::string ClassifyStrictSvdMotionReliability(const InstanceSvdMotionDiagnostics& diag,
                                               const InstanceMotionGateResult& zeroMotionGate)
{
    if(!EnableStrictSvdMotionReliability())
        return "disabled";
    if(!diag.valid)
        return "invalid_svd_diagnostics";
    if(diag.condition > GetStrictSvdMotionMaxCondition())
        return "ill_conditioned_svd";
    if(diag.planarityRatio < GetStrictSvdMotionMinPlanarityRatio())
        return "degenerate_svd_geometry";
    if(zeroMotionGate.valid && zeroMotionGate.dynamicMeanError > 0.0)
    {
        const double dynamicRmse = std::sqrt(zeroMotionGate.dynamicMeanError);
        if(std::isfinite(dynamicRmse) &&
           dynamicRmse > GetStrictSvdMotionMaxReprojectionRmsePx())
        {
            return "high_svd_reprojection_error";
        }
    }
    return "reliable";
}

bool IsNearlyIdentityInstanceMotion(const Sophus::SE3f& motion)
{
    const double translationNorm = motion.translation().cast<double>().norm();
    const Eigen::AngleAxisd angleAxis(motion.rotationMatrix().cast<double>());
    return translationNorm < 1e-6 && std::abs(angleAxis.angle()) < 1e-6;
}

double InstanceRotationAngleDeg(const Eigen::Matrix3f& rotation)
{
    const Eigen::AngleAxisd angleAxis(rotation.cast<double>());
    return std::abs(angleAxis.angle()) * 180.0 / 3.14159265358979323846;
}

bool IsSmallInstanceMotion(const Sophus::SE3f& motion,
                           const double maxTranslation,
                           const double maxRotationDeg)
{
    const double translationNorm = motion.translation().cast<double>().norm();
    const double rotationDeg = InstanceRotationAngleDeg(motion.rotationMatrix());
    return translationNorm <= maxTranslation && rotationDeg <= maxRotationDeg;
}

void ClassifyInstanceMotionGate(InstanceMotionGateResult& result)
{
    const double ratio = GetInstanceDynamicResidualRatio();
    result.state = kInstanceMotionUncertain;
    result.useDynamicMotion = false;

    if(!result.valid ||
       !std::isfinite(result.staticMeanError) ||
       !std::isfinite(result.dynamicMeanError))
    {
        return;
    }

    if(result.dynamicMeanError < result.staticMeanError * ratio)
    {
        result.state = kInstanceMotionDynamic;
        result.useDynamicMotion = true;
        return;
    }

    if(result.staticMeanError < result.dynamicMeanError * ratio)
        result.state = kInstanceMotionStatic;
}

InstanceSvdMotionDiagnostics ComputeInstanceSvdMotionDiagnostics(
    const std::vector<Eigen::Vector3f>& src,
    const std::vector<Eigen::Vector3f>& dst,
    const std::vector<double>& weights,
    const Sophus::SE3f& motion)
{
    InstanceSvdMotionDiagnostics diag;
    if(src.size() != dst.size() || src.empty())
        return diag;

    Eigen::Vector3d srcCentroid = Eigen::Vector3d::Zero();
    Eigen::Vector3d dstCentroid = Eigen::Vector3d::Zero();
    std::vector<double> effectiveWeights(src.size(), 1.0);
    for(size_t i = 0; i < src.size(); ++i)
    {
        double weight = 1.0;
        if(i < weights.size() && std::isfinite(weights[i]) && weights[i] > 0.0)
            weight = weights[i];
        effectiveWeights[i] = weight;
        srcCentroid += weight * src[i].cast<double>();
        dstCentroid += weight * dst[i].cast<double>();
        diag.totalWeight += weight;
        ++diag.support;
    }
    if(diag.totalWeight <= 0.0 || diag.support == 0)
        return diag;

    srcCentroid /= diag.totalWeight;
    dstCentroid /= diag.totalWeight;
    if(!srcCentroid.allFinite() || !dstCentroid.allFinite())
        return diag;

    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    double srcRadiusSum = 0.0;
    double dstRadiusSum = 0.0;
    std::vector<double> displacements;
    std::vector<double> residuals;
    displacements.reserve(src.size());
    residuals.reserve(src.size());
    for(size_t i = 0; i < src.size(); ++i)
    {
        const double weight = effectiveWeights[i];
        const Eigen::Vector3d srcCentered = src[i].cast<double>() - srcCentroid;
        const Eigen::Vector3d dstCentered = dst[i].cast<double>() - dstCentroid;
        covariance += weight * srcCentered * dstCentered.transpose();
        srcRadiusSum += weight * srcCentered.squaredNorm();
        dstRadiusSum += weight * dstCentered.squaredNorm();

        const double displacement = (dst[i] - src[i]).cast<double>().norm();
        const double residual = (motion * src[i] - dst[i]).cast<double>().norm();
        if(std::isfinite(displacement))
        {
            displacements.push_back(displacement);
            diag.meanDisplacement += displacement;
            diag.maxDisplacement = std::max(diag.maxDisplacement, displacement);
        }
        if(std::isfinite(residual))
        {
            residuals.push_back(residual);
            diag.meanResidual += residual;
            diag.maxResidual = std::max(diag.maxResidual, residual);
        }
    }
    if(!covariance.allFinite())
        return diag;

    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance);
    const Eigen::Vector3d singularValues = svd.singularValues();
    if(!singularValues.allFinite())
        return diag;

    diag.singular0 = singularValues[0];
    diag.singular1 = singularValues[1];
    diag.singular2 = singularValues[2];
    diag.condition = diag.singular0 / std::max(1e-12, diag.singular2);
    diag.planarityRatio = diag.singular2 / std::max(1e-12, diag.singular0);
    diag.linearityRatio = diag.singular1 / std::max(1e-12, diag.singular0);
    diag.srcStdRadius = std::sqrt(std::max(0.0, srcRadiusSum / diag.totalWeight));
    diag.dstStdRadius = std::sqrt(std::max(0.0, dstRadiusSum / diag.totalWeight));
    if(!displacements.empty())
    {
        diag.meanDisplacement /= static_cast<double>(displacements.size());
        diag.medianDisplacement = MedianValue(displacements);
    }
    if(!residuals.empty())
    {
        diag.meanResidual /= static_cast<double>(residuals.size());
        diag.medianResidual = MedianValue(residuals);
    }
    diag.valid = true;
    return diag;
}

InstanceMotionGateResult ComputeInstanceMotionResidualErrors(const Frame& frame,
                                                             const std::vector<int>& featureIndices,
                                                             const Sophus::SE3f& velocity)
{
    InstanceMotionGateResult result;
    if(!frame.mpCamera || !frame.HasPose())
        return result;

    const Sophus::SE3f Tcw = frame.GetPose();
    double staticErrorSum = 0.0;
    double dynamicErrorSum = 0.0;
    int support = 0;

    for(size_t i = 0; i < featureIndices.size(); ++i)
    {
        const int idx = featureIndices[i];
        if(idx < 0 ||
           idx >= static_cast<int>(frame.mvpMapPoints.size()) ||
           idx >= static_cast<int>(frame.mvKeysUn.size()))
        {
            continue;
        }

        MapPoint* pMP = frame.mvpMapPoints[idx];
        if(!pMP || pMP->isBad())
            continue;

        const Eigen::Vector3f staticPoint = pMP->GetWorldPos();
        if(!staticPoint.allFinite())
            continue;

        const Eigen::Vector3f dynamicPoint = velocity * staticPoint;
        if(!dynamicPoint.allFinite())
            continue;

        const Eigen::Vector3f staticCam = Tcw * staticPoint;
        const Eigen::Vector3f dynamicCam = Tcw * dynamicPoint;
        if(staticCam[2] <= 0.0f || dynamicCam[2] <= 0.0f)
            continue;

        const Eigen::Vector3d staticCamD = staticCam.cast<double>();
        const Eigen::Vector3d dynamicCamD = dynamicCam.cast<double>();
        const Eigen::Vector2d staticProjection = frame.mpCamera->project(staticCamD);
        const Eigen::Vector2d dynamicProjection = frame.mpCamera->project(dynamicCamD);
        if(!staticProjection.allFinite() || !dynamicProjection.allFinite())
            continue;

        const Eigen::Vector2d observation(frame.mvKeysUn[idx].pt.x,
                                          frame.mvKeysUn[idx].pt.y);
        staticErrorSum += (observation - staticProjection).squaredNorm();
        dynamicErrorSum += (observation - dynamicProjection).squaredNorm();
        ++support;
    }

    if(support < GetInstancePredictionMinSupport())
        return result;

    result.valid = true;
    result.support = support;
    result.staticMeanError = staticErrorSum / static_cast<double>(support);
    result.dynamicMeanError = dynamicErrorSum / static_cast<double>(support);
    ClassifyInstanceMotionGate(result);
    if(result.valid && IsNearlyIdentityInstanceMotion(velocity))
    {
        result.state = kInstanceMotionStatic;
        result.useDynamicMotion = false;
    }
    return result;
}

InstanceMotionGateResult EvaluateInstanceResidualMotionGate(const Frame& frame,
                                                            const std::vector<int>& featureIndices,
                                                            const Sophus::SE3f& velocity)
{
    if(!EnableInstanceResidualMotionGate())
        return InstanceMotionGateResult();

    InstanceMotionGateResult result =
        ComputeInstanceMotionResidualErrors(frame, featureIndices, velocity);
    return result;
}

bool ComputeObservedInstanceMeanRadius(const Frame& frame,
                                       const std::vector<int>& featureIndices,
                                       double& meanRadius,
                                       int& validPointCount)
{
    std::vector<Eigen::Vector3f> vPoints;
    vPoints.reserve(featureIndices.size());
    for(size_t i = 0; i < featureIndices.size(); ++i)
    {
        const int idx = featureIndices[i];
        if(idx < 0 || idx >= static_cast<int>(frame.mvpMapPoints.size()))
            continue;

        MapPoint* pMP = frame.mvpMapPoints[idx];
        if(!pMP || pMP->isBad())
            continue;

        const Eigen::Vector3f pos = pMP->GetWorldPos();
        if(!pos.allFinite())
            continue;

        vPoints.push_back(pos);
    }

    validPointCount = static_cast<int>(vPoints.size());
    if(validPointCount < 3)
        return false;

    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for(size_t i = 0; i < vPoints.size(); ++i)
        centroid += vPoints[i];
    centroid /= static_cast<float>(vPoints.size());

    meanRadius = 0.0;
    for(size_t i = 0; i < vPoints.size(); ++i)
        meanRadius += (vPoints[i] - centroid).norm();
    meanRadius /= static_cast<double>(vPoints.size());

    return std::isfinite(meanRadius) && meanRadius > 0.0;
}

bool IsFiniteSE3(const Sophus::SE3f& pose)
{
    return pose.matrix3x4().allFinite();
}

bool ShouldDetachRgbdInstanceFromStaticPath(Instance* pInstance)
{
    if(!RequireMotionEvidenceForRgbdDynamicSplit())
        return true;

    if(!pInstance)
        return false;

    if(RequireBackendMotionEvidenceForRgbdDynamicSplit() &&
       pInstance->GetBackendMovingMotionEvidence() <
           GetRgbdDynamicSplitMinBackendMotionEvidence())
    {
        return false;
    }

    Instance::InstanceMotionStateRecord record;
    const bool hasMotionState =
        pInstance->GetLatestInstanceMotionState(record) &&
        record.state != Instance::kDynamicEntityUnknown &&
        IsFiniteSE3(record.velocity);
    if(!hasMotionState)
        return false;

    if(record.state == Instance::kMovingDynamicEntity)
        return true;
    if(record.state == Instance::kZeroVelocityDynamicEntity)
        return false;

    if(record.state == Instance::kUncertainDynamicEntity)
    {
        if(RequireBackendMotionEvidenceForRgbdDynamicSplit())
            return false;

        const double translationNorm = record.velocity.translation().cast<double>().norm();
        const double rotationDeg = InstanceRotationAngleDeg(record.velocity.rotationMatrix());
        return translationNorm >= GetRgbdDynamicSplitMinTranslation() ||
               rotationDeg >= GetRgbdDynamicSplitMinRotationDeg();
    }

    return false;
}

bool DebugPanopticFallback()
{
    const char* envValue = std::getenv("STSLAM_DEBUG_PANOPTIC_FALLBACK");
    return envValue && std::string(envValue) != "0";
}

bool DebugFocusFrame(const unsigned long frameId)
{
    const char* startValue = std::getenv("STSLAM_DEBUG_FRAME_START");
    const char* endValue = std::getenv("STSLAM_DEBUG_FRAME_END");
    if(!startValue && !endValue)
        return false;

    const long startFrame = startValue ? std::atol(startValue) : 0;
    const long endFrame = endValue ? std::atol(endValue) : startFrame;
    return static_cast<long>(frameId) >= startFrame && static_cast<long>(frameId) <= endFrame;
}

bool EnableObservabilityLogging()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_OBSERVABILITY_LOG", false);
    return value;
}

std::string GetMaskModeName()
{
    const char* envValue = std::getenv("ORB_SLAM3_MASK_MODE");
    if(!envValue || std::string(envValue).empty())
        return "off";
    return std::string(envValue);
}

double ComputeMaskAreaRatio(const Frame& frame)
{
    if(!frame.HasPanopticObservation())
        return 0.0;

    const cv::Mat& mask = frame.mPanopticObservation.rawPanopticMask;
    if(mask.empty())
        return 0.0;

    const double totalArea =
        std::max(1.0, static_cast<double>(mask.rows) * static_cast<double>(mask.cols));
    const double dynamicArea = static_cast<double>(cv::countNonZero(mask > 0));
    return dynamicArea / totalArea;
}

double ComputeFeatureGridCoverage(const Frame& frame,
                                  const bool onlyStatic,
                                  const bool onlyDynamic)
{
    bool occupied[FRAME_GRID_COLS][FRAME_GRID_ROWS] = {{false}};
    int occupiedCells = 0;
    const float minX = Frame::mnMinX;
    const float minY = Frame::mnMinY;
    const float maxX = Frame::mnMaxX;
    const float maxY = Frame::mnMaxY;
    const float gridWidthInv = Frame::mfGridElementWidthInv;
    const float gridHeightInv = Frame::mfGridElementHeightInv;

    for(int i = 0; i < frame.N; ++i)
    {
        const int instanceId = frame.GetFeatureInstanceId(i);
        const bool isDynamic = instanceId > 0;
        if(onlyStatic && isDynamic)
            continue;
        if(onlyDynamic && !isDynamic)
            continue;

        const cv::Point2f& pt = frame.mvKeysUn[i].pt;
        if(pt.x < minX || pt.x >= maxX || pt.y < minY || pt.y >= maxY)
            continue;
        const int gridPosX =
            std::min(FRAME_GRID_COLS - 1,
                     std::max(0, static_cast<int>((pt.x - minX) * gridWidthInv)));
        const int gridPosY =
            std::min(FRAME_GRID_ROWS - 1,
                     std::max(0, static_cast<int>((pt.y - minY) * gridHeightInv)));

        if(!occupied[gridPosX][gridPosY])
        {
            occupied[gridPosX][gridPosY] = true;
            ++occupiedCells;
        }
    }

    const double totalCells =
        static_cast<double>(FRAME_GRID_COLS) * static_cast<double>(FRAME_GRID_ROWS);
    return occupiedCells / std::max(1.0, totalCells);
}

int CountTrackedDynamicMapPoints(const Frame& frame)
{
    int count = 0;
    for(size_t i = 0; i < frame.mvpMapPoints.size(); ++i)
    {
        MapPoint* pMP = frame.mvpMapPoints[i];
        if(!pMP || pMP->isBad())
            continue;
        if(pMP->GetInstanceId() > 0)
            ++count;
    }
    return count;
}

int CountTrackedStaticMapPoints(const Frame& frame)
{
    int count = 0;
    for(size_t i = 0; i < frame.mvpMapPoints.size(); ++i)
    {
        MapPoint* pMP = frame.mvpMapPoints[i];
        if(!pMP || pMP->isBad())
            continue;
        if(pMP->GetInstanceId() <= 0)
            ++count;
    }
    return count;
}

int CountTrackedMapPoints(const Frame& frame)
{
    int count = 0;
    for(size_t i = 0; i < frame.mvpMapPoints.size(); ++i)
    {
        MapPoint* pMP = frame.mvpMapPoints[i];
        if(pMP && !pMP->isBad())
            ++count;
    }
    return count;
}

struct FrameFeatureDebugStats
{
    int total = 0;
    int instanceBound = 0;
    int nonInstance = 0;
    int unlabeled = 0;
    int person = 0;
    int rider = 0;
    int bicycle = 0;
    int otherSemantic = 0;
};

struct MapPointDebugStats
{
    int total = 0;
    int withInstance = 0;
    int withoutInstance = 0;
    int unlabeled = 0;
    int person = 0;
    int rider = 0;
    int bicycle = 0;
    int otherSemantic = 0;
};

FrameFeatureDebugStats CollectFrameFeatureDebugStats(const Frame& frame,
                                                    const bool requireMapPoint,
                                                    const bool requireInlier,
                                                    const bool requireOutlier)
{
    FrameFeatureDebugStats stats;
    for(int i = 0; i < frame.N; ++i)
    {
        MapPoint* pMP = (i < static_cast<int>(frame.mvpMapPoints.size())) ? frame.mvpMapPoints[i] : static_cast<MapPoint*>(NULL);
        if(requireMapPoint && (!pMP || pMP->isBad()))
            continue;

        const bool isOutlier =
            (i < static_cast<int>(frame.mvbOutlier.size())) ? frame.mvbOutlier[i] : false;
        if(requireInlier && isOutlier)
            continue;
        if(requireOutlier && !isOutlier)
            continue;

        const int semanticLabel = frame.GetFeatureSemanticLabel(i);
        const int instanceId = frame.GetFeatureInstanceId(i);
        ++stats.total;
        if(instanceId > 0)
            ++stats.instanceBound;
        else
            ++stats.nonInstance;

        if(semanticLabel <= 0)
        {
            ++stats.unlabeled;
            continue;
        }

        if(semanticLabel == 11)
            ++stats.person;
        else if(semanticLabel == 12)
            ++stats.rider;
        else if(semanticLabel == 18)
            ++stats.bicycle;
        else
            ++stats.otherSemantic;
    }

    return stats;
}

MapPointDebugStats CollectMapPointDebugStats(const std::vector<MapPoint*>& mapPoints,
                                            const bool requireProjected)
{
    MapPointDebugStats stats;
    for(size_t i = 0; i < mapPoints.size(); ++i)
    {
        MapPoint* pMP = mapPoints[i];
        if(!pMP || pMP->isBad())
            continue;
        if(requireProjected && !pMP->mbTrackInView && !pMP->mbTrackInViewR)
            continue;

        const int semanticLabel = pMP->GetSemanticLabel();
        const int instanceId = pMP->GetInstanceId();
        ++stats.total;
        if(instanceId > 0)
            ++stats.withInstance;
        else
            ++stats.withoutInstance;

        if(semanticLabel <= 0)
        {
            ++stats.unlabeled;
            continue;
        }

        if(semanticLabel == 11)
            ++stats.person;
        else if(semanticLabel == 12)
            ++stats.rider;
        else if(semanticLabel == 18)
            ++stats.bicycle;
        else
            ++stats.otherSemantic;
    }

    return stats;
}

std::string FormatFrameFeatureDebugStats(const FrameFeatureDebugStats& stats)
{
    std::ostringstream oss;
    oss << "total=" << stats.total
        << " instance_bound=" << stats.instanceBound
        << " non_instance=" << stats.nonInstance
        << " unlabeled=" << stats.unlabeled
        << " person=" << stats.person
        << " rider=" << stats.rider
        << " bicycle=" << stats.bicycle
        << " other_semantic=" << stats.otherSemantic;
    return oss.str();
}

std::string FormatMapPointDebugStats(const MapPointDebugStats& stats)
{
    std::ostringstream oss;
    oss << "total=" << stats.total
        << " with_instance=" << stats.withInstance
        << " without_instance=" << stats.withoutInstance
        << " unlabeled=" << stats.unlabeled
        << " person=" << stats.person
        << " rider=" << stats.rider
        << " bicycle=" << stats.bicycle
        << " other_semantic=" << stats.otherSemantic;
    return oss.str();
}

void WriteObservabilityFrameStats(std::ofstream& stream,
                                  const Frame& frame,
                                  const int trackingState,
                                  const int sensor,
                                  const bool keyFrameCreated)
{
    if(!stream.is_open())
        return;

    const FrameFeatureDebugStats featureStats =
        CollectFrameFeatureDebugStats(frame, false, false, false);
    const int trackedMapPoints = CountTrackedMapPoints(frame);
    const int trackedStaticMapPoints = CountTrackedStaticMapPoints(frame);
    const int trackedDynamicMapPoints = CountTrackedDynamicMapPoints(frame);

    stream << frame.mnId
           << "," << std::fixed << std::setprecision(6) << frame.mTimeStamp
           << "," << trackingState
           << "," << sensor
           << "," << GetMaskModeName()
           << "," << (frame.HasPanopticObservation() ? 1 : 0)
           << "," << std::setprecision(6) << ComputeMaskAreaRatio(frame)
           << "," << featureStats.total
           << "," << featureStats.instanceBound
           << "," << featureStats.nonInstance
           << "," << trackedMapPoints
           << "," << trackedStaticMapPoints
           << "," << trackedDynamicMapPoints
           << "," << frame.GetNumThingFeatures()
           << "," << frame.GetNumStuffFeatures()
           << "," << std::setprecision(6) << ComputeFeatureGridCoverage(frame, true, false)
           << "," << std::setprecision(6) << ComputeFeatureGridCoverage(frame, false, true)
           << "," << std::setprecision(6) << ComputeFeatureGridCoverage(frame, false, false)
           << "," << frame.mmPredictedInstanceMotions.size()
           << "," << (frame.mpReferenceKF ? static_cast<long>(frame.mpReferenceKF->mnId) : -1)
           << "," << (keyFrameCreated ? 1 : 0)
           << std::endl;
}

cv::Mat ToProjectionMatrix(const Eigen::Matrix<float, 3, 4>& projection)
{
    cv::Mat cvProjection(3, 4, CV_32F);
    for(int row = 0; row < 3; ++row)
    {
        for(int col = 0; col < 4; ++col)
            cvProjection.at<float>(row, col) = projection(row, col);
    }
    return cvProjection;
}

int ComputeTargetInstanceFeatureCount(const Frame& frame,
                                      const cv::Mat& imGray,
                                      const int maskArea,
                                      const int existingCount)
{
    const int imageArea = std::max(1, imGray.rows * imGray.cols);
    const float globalDensity =
        static_cast<float>(std::max(frame.N, 1)) / static_cast<float>(imageArea);
    const int densityTarget =
        static_cast<int>(std::round(maskArea * globalDensity * 1.6f));
    const int areaTarget =
        static_cast<int>(std::round(std::sqrt(static_cast<float>(std::max(maskArea, 1)))));
    const int target =
        std::max(existingCount, std::max(12, std::max(densityTarget, areaTarget)));
    return std::min(120, target);
}

cv::KeyPoint UndistortExtraKeyPoint(const cv::KeyPoint& keypoint, const Frame& frame)
{
    cv::KeyPoint undistorted = keypoint;
    if(frame.mDistCoef.empty() || frame.mDistCoef.rows == 0 || frame.mDistCoef.cols == 0)
        return undistorted;

    if(std::fabs(frame.mDistCoef.at<float>(0)) < 1e-6f)
        return undistorted;

    if(!frame.mpCamera || frame.mpCamera->GetType() != GeometricCamera::CAM_PINHOLE)
        return undistorted;

    cv::Mat mat(1, 2, CV_32F);
    mat.at<float>(0, 0) = keypoint.pt.x;
    mat.at<float>(0, 1) = keypoint.pt.y;
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, frame.mK, frame.mDistCoef, cv::Mat(), frame.mK);
    mat = mat.reshape(1);
    undistorted.pt.x = mat.at<float>(0, 0);
    undistorted.pt.y = mat.at<float>(0, 1);
    return undistorted;
}

bool HasNearbyInstanceFeature(const Frame& frame,
                              const cv::KeyPoint& keypointUn,
                              const int instanceId)
{
    const std::vector<size_t> nearby =
        frame.GetFeaturesInArea(keypointUn.pt.x, keypointUn.pt.y, 6.0f);
    for(size_t i = 0; i < nearby.size(); ++i)
    {
        if(frame.GetFeatureInstanceId(nearby[i]) == instanceId)
            return true;
    }

    return false;
}

void AppendMonoFeatureToFrame(Frame& frame,
                              const cv::KeyPoint& keypoint,
                              const cv::KeyPoint& keypointUn,
                              const cv::Mat& descriptorRow)
{
    const int newIdx = frame.N;

    frame.mvKeys.push_back(keypoint);
    frame.mvKeysUn.push_back(keypointUn);
    frame.mDescriptors.push_back(descriptorRow);
    frame.mvuRight.push_back(-1.0f);
    frame.mvDepth.push_back(-1.0f);
    frame.mvpMapPoints.push_back(static_cast<MapPoint*>(NULL));
    frame.mvbOutlier.push_back(false);
    frame.N = newIdx + 1;

    int gridPosX = -1;
    int gridPosY = -1;
    if(frame.PosInGrid(keypointUn, gridPosX, gridPosY))
        frame.mGrid[gridPosX][gridPosY].push_back(newIdx);
}

Instance* EnsureInstanceInMap(Map* pMap,
                              const int instanceId,
                              const int semanticLabel)
{
    if(!pMap || instanceId <= 0)
        return static_cast<Instance*>(NULL);

    Instance* pInstance = pMap->GetInstance(instanceId);
    if(!pInstance)
        pInstance = pMap->AddInstance(new Instance(instanceId, semanticLabel));

    if(pInstance && pInstance->GetSemanticLabel() == 0 && semanticLabel > 0)
        pInstance->SetSemanticLabel(semanticLabel);

    return pInstance;
}

bool ShouldPreserveExistingInstanceBinding(MapPoint* pMP)
{
    if(!pMP)
        return false;
    if(pMP->IsDynamicInstanceObservationPoint())
        return false;

    return pMP->mnObservationCount >= 3 || pMP->Observations() >= 2;
}

Instance* BindMapPointToInstance(MapPoint* pMP,
                                 const int instanceId,
                                 const int semanticLabel,
                                 const bool allowConfirmedRebind)
{
    if(!pMP || pMP->isBad() || instanceId <= 0)
        return static_cast<Instance*>(NULL);

    Map* pMap = pMP->GetMap();
    if(!pMap)
        return static_cast<Instance*>(NULL);

    const int oldInstanceId = pMP->GetInstanceId();
    const bool observationOnlyPoint = pMP->IsDynamicInstanceObservationPoint();
    if(oldInstanceId > 0 && oldInstanceId != instanceId)
    {
        if(!allowConfirmedRebind && ShouldPreserveExistingInstanceBinding(pMP))
        {
            Instance* pOldInstance =
                EnsureInstanceInMap(pMap, oldInstanceId, pMP->GetSemanticLabel());
            if(pOldInstance)
                pOldInstance->AddMapPoint(pMP);
            return static_cast<Instance*>(NULL);
        }

        Instance* pOldInstance = pMap->GetInstance(oldInstanceId);
        if(pOldInstance)
        {
            pOldInstance->RemoveMapPoint(pMP);
            if(!observationOnlyPoint && pOldInstance->NumMapPoints() == 0)
                pMap->EraseInstance(oldInstanceId);
        }
    }

    pMP->SetInstanceId(instanceId);
    if(semanticLabel > 0)
        pMP->SetSemanticLabel(semanticLabel);

    Instance* pInstance = EnsureInstanceInMap(pMap, instanceId, semanticLabel);
    if(!pInstance)
        return static_cast<Instance*>(NULL);

    if(!observationOnlyPoint)
        pInstance->AddMapPoint(pMP);
    return pInstance;
}

Instance* AnnotateMapPointWithInstance(MapPoint* pMP,
                                       const int instanceId,
                                       const int semanticLabel,
                                       const bool allowConfirmedRebind)
{
    if(!pMP || pMP->isBad() || instanceId <= 0)
        return static_cast<Instance*>(NULL);

    Map* pMap = pMP->GetMap();
    if(!pMap)
        return static_cast<Instance*>(NULL);

    const int oldInstanceId = pMP->GetInstanceId();
    if(oldInstanceId > 0 && oldInstanceId != instanceId &&
       !allowConfirmedRebind && ShouldPreserveExistingInstanceBinding(pMP))
    {
        return static_cast<Instance*>(NULL);
    }

    pMP->SetInstanceId(instanceId);
    if(semanticLabel > 0)
        pMP->SetSemanticLabel(semanticLabel);

    return EnsureInstanceInMap(pMap, instanceId, semanticLabel);
}
}


Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Atlas *pAtlas, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, Settings* settings, const string &_nameSeq):
    mState(NO_IMAGES_YET), mSensor(sensor), mTrackedFr(0), mbStep(false),
    mbOnlyTracking(false), mbMapUpdated(false), mbVO(false), mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB),
    mbReadyToInitializate(false), mpSystem(pSys), mpViewer(NULL), bStepByStep(false),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpAtlas(pAtlas), mnLastRelocFrameId(0), time_recently_lost(5.0),
    mnInitialFrameId(0), mbCreatedMap(false), mnFirstFrameId(0), mpCamera2(nullptr), mpLastKeyFrame(static_cast<KeyFrame*>(NULL))
{
    // Load camera parameters from settings file
    if(settings){
        newParameterLoader(settings);
    }
    else{
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        bool b_parse_cam = ParseCamParamFile(fSettings);
        if(!b_parse_cam)
        {
            std::cout << "*Error with the camera parameters in the config file*" << std::endl;
        }

        // Load ORB parameters
        bool b_parse_orb = ParseORBParamFile(fSettings);
        if(!b_parse_orb)
        {
            std::cout << "*Error with the ORB parameters in the config file*" << std::endl;
        }

        bool b_parse_imu = true;
        if(sensor==System::IMU_MONOCULAR || sensor==System::IMU_STEREO || sensor==System::IMU_RGBD)
        {
            b_parse_imu = ParseIMUParamFile(fSettings);
            if(!b_parse_imu)
            {
                std::cout << "*Error with the IMU parameters in the config file*" << std::endl;
            }

            mnFramesToResetIMU = mMaxFrames;
        }

        if(!b_parse_cam || !b_parse_orb || !b_parse_imu)
        {
            std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
            try
            {
                throw -1;
            }
            catch(exception &e)
            {

            }
        }
    }

    initID = 0; lastID = 0;
    mbInitWith3KFs = false;
    mnNumDataset = 0;

    vector<GeometricCamera*> vpCams = mpAtlas->GetAllCameras();
    std::cout << "There are " << vpCams.size() << " cameras in the atlas" << std::endl;
    for(GeometricCamera* pCam : vpCams)
    {
        std::cout << "Camera " << pCam->GetId();
        if(pCam->GetType() == GeometricCamera::CAM_PINHOLE)
        {
            std::cout << " is pinhole" << std::endl;
        }
        else if(pCam->GetType() == GeometricCamera::CAM_FISHEYE)
        {
            std::cout << " is fisheye" << std::endl;
        }
        else
        {
            std::cout << " is unknown" << std::endl;
        }
    }

#ifdef REGISTER_TIMES
    vdRectStereo_ms.clear();
    vdResizeImage_ms.clear();
    vdORBExtract_ms.clear();
    vdStereoMatch_ms.clear();
    vdIMUInteg_ms.clear();
    vdPosePred_ms.clear();
    vdLMTrack_ms.clear();
    vdNewKF_ms.clear();
    vdTrackTotal_ms.clear();
#endif

    if(EnableObservabilityLogging())
    {
        f_observability_stats.open("observability_frame_stats.csv");
        if(f_observability_stats.is_open())
        {
            f_observability_stats
                << "frame_id,timestamp,state,sensor,mask_mode,has_panoptic,mask_ratio,"
                << "total_features,instance_bound_features,non_instance_features,"
                << "tracked_map_points,tracked_static_map_points,tracked_dynamic_map_points,"
                << "thing_features,stuff_features,static_grid_coverage,dynamic_grid_coverage,total_grid_coverage,"
                << "predicted_instances,reference_kf_id,is_keyframe_created"
                << std::endl;
        }
    }
}

#ifdef REGISTER_TIMES
double calcAverage(vector<double> v_times)
{
    double accum = 0;
    for(double value : v_times)
    {
        accum += value;
    }

    return accum / v_times.size();
}

double calcDeviation(vector<double> v_times, double average)
{
    double accum = 0;
    for(double value : v_times)
    {
        accum += pow(value - average, 2);
    }
    return sqrt(accum / v_times.size());
}

double calcAverage(vector<int> v_values)
{
    double accum = 0;
    int total = 0;
    for(double value : v_values)
    {
        if(value == 0)
            continue;
        accum += value;
        total++;
    }

    return accum / total;
}

double calcDeviation(vector<int> v_values, double average)
{
    double accum = 0;
    int total = 0;
    for(double value : v_values)
    {
        if(value == 0)
            continue;
        accum += pow(value - average, 2);
        total++;
    }
    return sqrt(accum / total);
}

void Tracking::LocalMapStats2File()
{
    ofstream f;
    f.open("LocalMapTimeStats.txt");
    f << fixed << setprecision(6);
    f << "#Stereo rect[ms], MP culling[ms], MP creation[ms], LBA[ms], KF culling[ms], Total[ms]" << endl;
    for(int i=0; i<mpLocalMapper->vdLMTotal_ms.size(); ++i)
    {
        f << mpLocalMapper->vdKFInsert_ms[i] << "," << mpLocalMapper->vdMPCulling_ms[i] << ","
          << mpLocalMapper->vdMPCreation_ms[i] << "," << mpLocalMapper->vdLBASync_ms[i] << ","
          << mpLocalMapper->vdKFCullingSync_ms[i] <<  "," << mpLocalMapper->vdLMTotal_ms[i] << endl;
    }

    f.close();

    f.open("LBA_Stats.txt");
    f << fixed << setprecision(6);
    f << "#LBA time[ms], KF opt[#], KF fixed[#], MP[#], Edges[#]" << endl;
    for(int i=0; i<mpLocalMapper->vdLBASync_ms.size(); ++i)
    {
        f << mpLocalMapper->vdLBASync_ms[i] << "," << mpLocalMapper->vnLBA_KFopt[i] << ","
          << mpLocalMapper->vnLBA_KFfixed[i] << "," << mpLocalMapper->vnLBA_MPs[i] << ","
          << mpLocalMapper->vnLBA_edges[i] << endl;
    }


    f.close();
}

void Tracking::TrackStats2File()
{
    ofstream f;
    f.open("SessionInfo.txt");
    f << fixed;
    f << "Number of KFs: " << mpAtlas->GetAllKeyFrames().size() << endl;
    f << "Number of MPs: " << mpAtlas->GetAllMapPoints().size() << endl;

    f << "OpenCV version: " << CV_VERSION << endl;

    f.close();

    f.open("TrackingTimeStats.txt");
    f << fixed << setprecision(6);

    f << "#Image Rect[ms], Image Resize[ms], ORB ext[ms], Stereo match[ms], IMU preint[ms], Pose pred[ms], LM track[ms], KF dec[ms], Total[ms]" << endl;

    for(int i=0; i<vdTrackTotal_ms.size(); ++i)
    {
        double stereo_rect = 0.0;
        if(!vdRectStereo_ms.empty())
        {
            stereo_rect = vdRectStereo_ms[i];
        }

        double resize_image = 0.0;
        if(!vdResizeImage_ms.empty())
        {
            resize_image = vdResizeImage_ms[i];
        }

        double stereo_match = 0.0;
        if(!vdStereoMatch_ms.empty())
        {
            stereo_match = vdStereoMatch_ms[i];
        }

        double imu_preint = 0.0;
        if(!vdIMUInteg_ms.empty())
        {
            imu_preint = vdIMUInteg_ms[i];
        }

        f << stereo_rect << "," << resize_image << "," << vdORBExtract_ms[i] << "," << stereo_match << "," << imu_preint << ","
          << vdPosePred_ms[i] <<  "," << vdLMTrack_ms[i] << "," << vdNewKF_ms[i] << "," << vdTrackTotal_ms[i] << endl;
    }

    f.close();
}

void Tracking::PrintTimeStats()
{
    // Save data in files
    TrackStats2File();
    LocalMapStats2File();


    ofstream f;
    f.open("ExecMean.txt");
    f << fixed;
    //Report the mean and std of each one
    std::cout << std::endl << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
    f << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
    cout << "OpenCV version: " << CV_VERSION << endl;
    f << "OpenCV version: " << CV_VERSION << endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << "Tracking" << std::setprecision(5) << std::endl << std::endl;
    f << "---------------------------" << std::endl;
    f << "Tracking" << std::setprecision(5) << std::endl << std::endl;
    double average, deviation;
    if(!vdRectStereo_ms.empty())
    {
        average = calcAverage(vdRectStereo_ms);
        deviation = calcDeviation(vdRectStereo_ms, average);
        std::cout << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
        f << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
    }

    if(!vdResizeImage_ms.empty())
    {
        average = calcAverage(vdResizeImage_ms);
        deviation = calcDeviation(vdResizeImage_ms, average);
        std::cout << "Image Resize: " << average << "$\\pm$" << deviation << std::endl;
        f << "Image Resize: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdORBExtract_ms);
    deviation = calcDeviation(vdORBExtract_ms, average);
    std::cout << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;
    f << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;

    if(!vdStereoMatch_ms.empty())
    {
        average = calcAverage(vdStereoMatch_ms);
        deviation = calcDeviation(vdStereoMatch_ms, average);
        std::cout << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
        f << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
    }

    if(!vdIMUInteg_ms.empty())
    {
        average = calcAverage(vdIMUInteg_ms);
        deviation = calcDeviation(vdIMUInteg_ms, average);
        std::cout << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
        f << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdPosePred_ms);
    deviation = calcDeviation(vdPosePred_ms, average);
    std::cout << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;
    f << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdLMTrack_ms);
    deviation = calcDeviation(vdLMTrack_ms, average);
    std::cout << "LM Track: " << average << "$\\pm$" << deviation << std::endl;
    f << "LM Track: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdNewKF_ms);
    deviation = calcDeviation(vdNewKF_ms, average);
    std::cout << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;
    f << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdTrackTotal_ms);
    deviation = calcDeviation(vdTrackTotal_ms, average);
    std::cout << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;
    f << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping time stats
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "Local Mapping" << std::endl << std::endl;
    f << std::endl << "Local Mapping" << std::endl << std::endl;

    average = calcAverage(mpLocalMapper->vdKFInsert_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFInsert_ms, average);
    std::cout << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;
    f << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCulling_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCulling_ms, average);
    std::cout << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;
    f << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCreation_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCreation_ms, average);
    std::cout << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;
    f << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLBA_ms);
    deviation = calcDeviation(mpLocalMapper->vdLBA_ms, average);
    std::cout << "LBA: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdKFCulling_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFCulling_ms, average);
    std::cout << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;
    f << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLMTotal_ms);
    deviation = calcDeviation(mpLocalMapper->vdLMTotal_ms, average);
    std::cout << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;
    f << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping LBA complexity
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "LBA complexity (mean$\\pm$std)" << std::endl;
    f << "---------------------------" << std::endl;
    f << std::endl << "LBA complexity (mean$\\pm$std)" << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_edges);
    deviation = calcDeviation(mpLocalMapper->vnLBA_edges, average);
    std::cout << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFopt);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFopt, average);
    std::cout << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFfixed);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFfixed, average);
    std::cout << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_MPs);
    deviation = calcDeviation(mpLocalMapper->vnLBA_MPs, average);
    std::cout << "LBA MP: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    f << "LBA MP: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    std::cout << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
    std::cout << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;
    f << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
    f << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;

    // Map complexity
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "Map complexity" << std::endl;
    std::cout << "KFs in map: " << mpAtlas->GetAllKeyFrames().size() << std::endl;
    std::cout << "MPs in map: " << mpAtlas->GetAllMapPoints().size() << std::endl;
    f << "---------------------------" << std::endl;
    f << std::endl << "Map complexity" << std::endl;
    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    Map* pBestMap = vpMaps[0];
    for(int i=1; i<vpMaps.size(); ++i)
    {
        if(pBestMap->GetAllKeyFrames().size() < vpMaps[i]->GetAllKeyFrames().size())
        {
            pBestMap = vpMaps[i];
        }
    }

    f << "KFs in map: " << pBestMap->GetAllKeyFrames().size() << std::endl;
    f << "MPs in map: " << pBestMap->GetAllMapPoints().size() << std::endl;

    f << "---------------------------" << std::endl;
    f << std::endl << "Place Recognition (mean$\\pm$std)" << std::endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "Place Recognition (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdDataQuery_ms);
    deviation = calcDeviation(mpLoopClosing->vdDataQuery_ms, average);
    f << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdEstSim3_ms);
    deviation = calcDeviation(mpLoopClosing->vdEstSim3_ms, average);
    f << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdPRTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdPRTotal_ms, average);
    f << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << std::endl << "Loop Closing (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Loop Closing (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopFusion_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopFusion_ms, average);
    f << "Loop Fusion: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Loop Fusion: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopOptEss_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopOptEss_ms, average);
    f << "Essential Graph: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Essential Graph: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopTotal_ms, average);
    f << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nLoop << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nLoop << std::endl;
    average = calcAverage(mpLoopClosing->vnLoopKFs);
    deviation = calcDeviation(mpLoopClosing->vnLoopKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;

    f << std::endl << "Map Merging (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Map Merging (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeMaps_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeMaps_ms, average);
    f << "Merge Maps: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Merge Maps: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdWeldingBA_ms);
    deviation = calcDeviation(mpLoopClosing->vdWeldingBA_ms, average);
    f << "Welding BA: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Welding BA: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeOptEss_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeOptEss_ms, average);
    f << "Optimization Ess.: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Optimization Ess.: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeTotal_ms, average);
    f << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nMerges << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nMerges << std::endl;
    average = calcAverage(mpLoopClosing->vnMergeKFs);
    deviation = calcDeviation(mpLoopClosing->vnMergeKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vnMergeMPs);
    deviation = calcDeviation(mpLoopClosing->vnMergeMPs, average);
    f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;

    f << std::endl << "Full GBA (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Full GBA (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdGBA_ms);
    deviation = calcDeviation(mpLoopClosing->vdGBA_ms, average);
    f << "GBA: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "GBA: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdUpdateMap_ms);
    deviation = calcDeviation(mpLoopClosing->vdUpdateMap_ms, average);
    f << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdFGBATotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdFGBATotal_ms, average);
    f << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nFGBA_exec << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nFGBA_exec << std::endl;
    f << "Numb abort: " << mpLoopClosing->nFGBA_abort << std::endl;
    std::cout << "Num abort: " << mpLoopClosing->nFGBA_abort << std::endl;
    average = calcAverage(mpLoopClosing->vnGBAKFs);
    deviation = calcDeviation(mpLoopClosing->vnGBAKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vnGBAMPs);
    deviation = calcDeviation(mpLoopClosing->vnGBAMPs, average);
    f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;

    f.close();

}

#endif

Tracking::~Tracking()
{
    //f_track_stats.close();
    if(f_observability_stats.is_open())
        f_observability_stats.close();

}

void Tracking::newParameterLoader(Settings *settings) {
    mpCamera = settings->camera1();
    mpCamera = mpAtlas->AddCamera(mpCamera);

    if(settings->needToUndistort()){
        mDistCoef = settings->camera1DistortionCoef();
    }
    else{
        mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    }

    //TODO: missing image scaling and rectification
    mImageScale = 1.0f;

    mK = cv::Mat::eye(3,3,CV_32F);
    mK.at<float>(0,0) = mpCamera->getParameter(0);
    mK.at<float>(1,1) = mpCamera->getParameter(1);
    mK.at<float>(0,2) = mpCamera->getParameter(2);
    mK.at<float>(1,2) = mpCamera->getParameter(3);

    mK_.setIdentity();
    mK_(0,0) = mpCamera->getParameter(0);
    mK_(1,1) = mpCamera->getParameter(1);
    mK_(0,2) = mpCamera->getParameter(2);
    mK_(1,2) = mpCamera->getParameter(3);

    if((mSensor==System::STEREO || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD) &&
        settings->cameraType() == Settings::KannalaBrandt){
        mpCamera2 = settings->camera2();
        mpCamera2 = mpAtlas->AddCamera(mpCamera2);

        mTlr = settings->Tlr();

        mpFrameDrawer->both = true;
    }

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD ){
        mbf = settings->bf();
        mThDepth = settings->b() * settings->thDepth();
    }

    if(mSensor==System::RGBD || mSensor==System::IMU_RGBD){
        mDepthMapFactor = settings->depthMapFactor();
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

    mMinFrames = 0;
    mMaxFrames = settings->fps();
    mbRGB = settings->rgb();

    //ORB parameters
    int nFeatures = settings->nFeatures();
    int nLevels = settings->nLevels();
    int fIniThFAST = settings->initThFAST();
    int fMinThFAST = settings->minThFAST();
    float fScaleFactor = settings->scaleFactor();

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::STEREO || mSensor==System::IMU_STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR)
        mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    //IMU parameters
    Sophus::SE3f Tbc = settings->Tbc();
    mInsertKFsLost = settings->insertKFsWhenLost();
    mImuFreq = settings->imuFrequency();
    mImuPer = 0.001; //1.0 / (double) mImuFreq;     //TODO: ESTO ESTA BIEN?
    float Ng = settings->noiseGyro();
    float Na = settings->noiseAcc();
    float Ngw = settings->gyroWalk();
    float Naw = settings->accWalk();

    const float sf = sqrt(mImuFreq);
    mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
}

bool Tracking::ParseCamParamFile(cv::FileStorage &fSettings)
{
    mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    cout << endl << "Camera Parameters: " << endl;
    bool b_miss_params = false;

    string sCameraName = fSettings["Camera.type"];
    if(sCameraName == "PinHole")
    {
        float fx, fy, cx, cy;
        mImageScale = 1.f;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if(!node.empty() && node.isReal())
        {
            fx = node.real();
        }
        else
        {
            std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.fy"];
        if(!node.empty() && node.isReal())
        {
            fy = node.real();
        }
        else
        {
            std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if(!node.empty() && node.isReal())
        {
            cx = node.real();
        }
        else
        {
            std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if(!node.empty() && node.isReal())
        {
            cy = node.real();
        }
        else
        {
            std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        // Distortion parameters
        node = fSettings["Camera.k1"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(0) = node.real();
        }
        else
        {
            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k2"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(1) = node.real();
        }
        else
        {
            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p1"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(2) = node.real();
        }
        else
        {
            std::cerr << "*Camera.p1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p2"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(3) = node.real();
        }
        else
        {
            std::cerr << "*Camera.p2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.resize(5);
            mDistCoef.at<float>(4) = node.real();
        }

        node = fSettings["Camera.imageScale"];
        if(!node.empty() && node.isReal())
        {
            mImageScale = node.real();
        }

        if(b_miss_params)
        {
            return false;
        }

        if(mImageScale != 1.f)
        {
            // K matrix parameters must be scaled.
            fx = fx * mImageScale;
            fy = fy * mImageScale;
            cx = cx * mImageScale;
            cy = cy * mImageScale;
        }

        vector<float> vCamCalib{fx,fy,cx,cy};

        mpCamera = new Pinhole(vCamCalib);

        mpCamera = mpAtlas->AddCamera(mpCamera);

        std::cout << "- Camera: Pinhole" << std::endl;
        std::cout << "- Image scale: " << mImageScale << std::endl;
        std::cout << "- fx: " << fx << std::endl;
        std::cout << "- fy: " << fy << std::endl;
        std::cout << "- cx: " << cx << std::endl;
        std::cout << "- cy: " << cy << std::endl;
        std::cout << "- k1: " << mDistCoef.at<float>(0) << std::endl;
        std::cout << "- k2: " << mDistCoef.at<float>(1) << std::endl;


        std::cout << "- p1: " << mDistCoef.at<float>(2) << std::endl;
        std::cout << "- p2: " << mDistCoef.at<float>(3) << std::endl;

        if(mDistCoef.rows==5)
            std::cout << "- k3: " << mDistCoef.at<float>(4) << std::endl;

        mK = cv::Mat::eye(3,3,CV_32F);
        mK.at<float>(0,0) = fx;
        mK.at<float>(1,1) = fy;
        mK.at<float>(0,2) = cx;
        mK.at<float>(1,2) = cy;

        mK_.setIdentity();
        mK_(0,0) = fx;
        mK_(1,1) = fy;
        mK_(0,2) = cx;
        mK_(1,2) = cy;
    }
    else if(sCameraName == "KannalaBrandt8")
    {
        float fx, fy, cx, cy;
        float k1, k2, k3, k4;
        mImageScale = 1.f;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if(!node.empty() && node.isReal())
        {
            fx = node.real();
        }
        else
        {
            std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
        node = fSettings["Camera.fy"];
        if(!node.empty() && node.isReal())
        {
            fy = node.real();
        }
        else
        {
            std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if(!node.empty() && node.isReal())
        {
            cx = node.real();
        }
        else
        {
            std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if(!node.empty() && node.isReal())
        {
            cy = node.real();
        }
        else
        {
            std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        // Distortion parameters
        node = fSettings["Camera.k1"];
        if(!node.empty() && node.isReal())
        {
            k1 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
        node = fSettings["Camera.k2"];
        if(!node.empty() && node.isReal())
        {
            k2 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if(!node.empty() && node.isReal())
        {
            k3 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k3 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k4"];
        if(!node.empty() && node.isReal())
        {
            k4 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k4 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.imageScale"];
        if(!node.empty() && node.isReal())
        {
            mImageScale = node.real();
        }

        if(!b_miss_params)
        {
            if(mImageScale != 1.f)
            {
                // K matrix parameters must be scaled.
                fx = fx * mImageScale;
                fy = fy * mImageScale;
                cx = cx * mImageScale;
                cy = cy * mImageScale;
            }

            vector<float> vCamCalib{fx,fy,cx,cy,k1,k2,k3,k4};
            mpCamera = new KannalaBrandt8(vCamCalib);
            mpCamera = mpAtlas->AddCamera(mpCamera);
            std::cout << "- Camera: Fisheye" << std::endl;
            std::cout << "- Image scale: " << mImageScale << std::endl;
            std::cout << "- fx: " << fx << std::endl;
            std::cout << "- fy: " << fy << std::endl;
            std::cout << "- cx: " << cx << std::endl;
            std::cout << "- cy: " << cy << std::endl;
            std::cout << "- k1: " << k1 << std::endl;
            std::cout << "- k2: " << k2 << std::endl;
            std::cout << "- k3: " << k3 << std::endl;
            std::cout << "- k4: " << k4 << std::endl;

            mK = cv::Mat::eye(3,3,CV_32F);
            mK.at<float>(0,0) = fx;
            mK.at<float>(1,1) = fy;
            mK.at<float>(0,2) = cx;
            mK.at<float>(1,2) = cy;

            mK_.setIdentity();
            mK_(0,0) = fx;
            mK_(1,1) = fy;
            mK_(0,2) = cx;
            mK_(1,2) = cy;
        }

        if(mSensor==System::STEREO || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD){
            // Right camera
            // Camera calibration parameters
            cv::FileNode node = fSettings["Camera2.fx"];
            if(!node.empty() && node.isReal())
            {
                fx = node.real();
            }
            else
            {
                std::cerr << "*Camera2.fx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera2.fy"];
            if(!node.empty() && node.isReal())
            {
                fy = node.real();
            }
            else
            {
                std::cerr << "*Camera2.fy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.cx"];
            if(!node.empty() && node.isReal())
            {
                cx = node.real();
            }
            else
            {
                std::cerr << "*Camera2.cx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.cy"];
            if(!node.empty() && node.isReal())
            {
                cy = node.real();
            }
            else
            {
                std::cerr << "*Camera2.cy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            // Distortion parameters
            node = fSettings["Camera2.k1"];
            if(!node.empty() && node.isReal())
            {
                k1 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k1 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera2.k2"];
            if(!node.empty() && node.isReal())
            {
                k2 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k2 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.k3"];
            if(!node.empty() && node.isReal())
            {
                k3 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k3 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.k4"];
            if(!node.empty() && node.isReal())
            {
                k4 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k4 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }


            int leftLappingBegin = -1;
            int leftLappingEnd = -1;

            int rightLappingBegin = -1;
            int rightLappingEnd = -1;

            node = fSettings["Camera.lappingBegin"];
            if(!node.empty() && node.isInt())
            {
                leftLappingBegin = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera.lappingBegin not correctly defined" << std::endl;
            }
            node = fSettings["Camera.lappingEnd"];
            if(!node.empty() && node.isInt())
            {
                leftLappingEnd = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera.lappingEnd not correctly defined" << std::endl;
            }
            node = fSettings["Camera2.lappingBegin"];
            if(!node.empty() && node.isInt())
            {
                rightLappingBegin = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera2.lappingBegin not correctly defined" << std::endl;
            }
            node = fSettings["Camera2.lappingEnd"];
            if(!node.empty() && node.isInt())
            {
                rightLappingEnd = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera2.lappingEnd not correctly defined" << std::endl;
            }

            node = fSettings["Tlr"];
            cv::Mat cvTlr;
            if(!node.empty())
            {
                cvTlr = node.mat();
                if(cvTlr.rows != 3 || cvTlr.cols != 4)
                {
                    std::cerr << "*Tlr matrix have to be a 3x4 transformation matrix*" << std::endl;
                    b_miss_params = true;
                }
            }
            else
            {
                std::cerr << "*Tlr matrix doesn't exist*" << std::endl;
                b_miss_params = true;
            }

            if(!b_miss_params)
            {
                if(mImageScale != 1.f)
                {
                    // K matrix parameters must be scaled.
                    fx = fx * mImageScale;
                    fy = fy * mImageScale;
                    cx = cx * mImageScale;
                    cy = cy * mImageScale;

                    leftLappingBegin = leftLappingBegin * mImageScale;
                    leftLappingEnd = leftLappingEnd * mImageScale;
                    rightLappingBegin = rightLappingBegin * mImageScale;
                    rightLappingEnd = rightLappingEnd * mImageScale;
                }

                static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0] = leftLappingBegin;
                static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1] = leftLappingEnd;

                mpFrameDrawer->both = true;

                vector<float> vCamCalib2{fx,fy,cx,cy,k1,k2,k3,k4};
                mpCamera2 = new KannalaBrandt8(vCamCalib2);
                mpCamera2 = mpAtlas->AddCamera(mpCamera2);

                mTlr = Converter::toSophus(cvTlr);

                static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0] = rightLappingBegin;
                static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1] = rightLappingEnd;

                std::cout << "- Camera1 Lapping: " << leftLappingBegin << ", " << leftLappingEnd << std::endl;

                std::cout << std::endl << "Camera2 Parameters:" << std::endl;
                std::cout << "- Camera: Fisheye" << std::endl;
                std::cout << "- Image scale: " << mImageScale << std::endl;
                std::cout << "- fx: " << fx << std::endl;
                std::cout << "- fy: " << fy << std::endl;
                std::cout << "- cx: " << cx << std::endl;
                std::cout << "- cy: " << cy << std::endl;
                std::cout << "- k1: " << k1 << std::endl;
                std::cout << "- k2: " << k2 << std::endl;
                std::cout << "- k3: " << k3 << std::endl;
                std::cout << "- k4: " << k4 << std::endl;

                std::cout << "- mTlr: \n" << cvTlr << std::endl;

                std::cout << "- Camera2 Lapping: " << rightLappingBegin << ", " << rightLappingEnd << std::endl;
            }
        }

        if(b_miss_params)
        {
            return false;
        }

    }
    else
    {
        std::cerr << "*Not Supported Camera Sensor*" << std::endl;
        std::cerr << "Check an example configuration file with the desired sensor" << std::endl;
    }

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD )
    {
        cv::FileNode node = fSettings["Camera.bf"];
        if(!node.empty() && node.isReal())
        {
            mbf = node.real();
            if(mImageScale != 1.f)
            {
                mbf *= mImageScale;
            }
        }
        else
        {
            std::cerr << "*Camera.bf parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

    }

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD)
    {
        float fx = mpCamera->getParameter(0);
        cv::FileNode node = fSettings["ThDepth"];
        if(!node.empty()  && node.isReal())
        {
            mThDepth = node.real();
            mThDepth = mbf*mThDepth/fx;
            cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
        }
        else
        {
            std::cerr << "*ThDepth parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }


    }

    if(mSensor==System::RGBD || mSensor==System::IMU_RGBD)
    {
        cv::FileNode node = fSettings["DepthMapFactor"];
        if(!node.empty() && node.isReal())
        {
            mDepthMapFactor = node.real();
            if(fabs(mDepthMapFactor)<1e-5)
                mDepthMapFactor=1;
            else
                mDepthMapFactor = 1.0f/mDepthMapFactor;
        }
        else
        {
            std::cerr << "*DepthMapFactor parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

    }

    if(b_miss_params)
    {
        return false;
    }

    return true;
}

bool Tracking::ParseORBParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;
    int nFeatures, nLevels, fIniThFAST, fMinThFAST;
    float fScaleFactor;

    cv::FileNode node = fSettings["ORBextractor.nFeatures"];
    if(!node.empty() && node.isInt())
    {
        nFeatures = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.nFeatures parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.scaleFactor"];
    if(!node.empty() && node.isReal())
    {
        fScaleFactor = node.real();
    }
    else
    {
        std::cerr << "*ORBextractor.scaleFactor parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.nLevels"];
    if(!node.empty() && node.isInt())
    {
        nLevels = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.nLevels parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.iniThFAST"];
    if(!node.empty() && node.isInt())
    {
        fIniThFAST = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.iniThFAST parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.minThFAST"];
    if(!node.empty() && node.isInt())
    {
        fMinThFAST = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.minThFAST parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    if(b_miss_params)
    {
        return false;
    }

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::STEREO || mSensor==System::IMU_STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR)
        mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    return true;
}

bool Tracking::ParseIMUParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;

    cv::Mat cvTbc;
    cv::FileNode node = fSettings["Tbc"];
    if(!node.empty())
    {
        cvTbc = node.mat();
        if(cvTbc.rows != 4 || cvTbc.cols != 4)
        {
            std::cerr << "*Tbc matrix have to be a 4x4 transformation matrix*" << std::endl;
            b_miss_params = true;
        }
    }
    else
    {
        std::cerr << "*Tbc matrix doesn't exist*" << std::endl;
        b_miss_params = true;
    }
    cout << endl;
    cout << "Left camera to Imu Transform (Tbc): " << endl << cvTbc << endl;
    Eigen::Matrix<float,4,4,Eigen::RowMajor> eigTbc(cvTbc.ptr<float>(0));
    Sophus::SE3f Tbc(eigTbc);

    node = fSettings["InsertKFsWhenLost"];
    mInsertKFsLost = true;
    if(!node.empty() && node.isInt())
    {
        mInsertKFsLost = (bool) node.operator int();
    }

    if(!mInsertKFsLost)
        cout << "Do not insert keyframes when lost visual tracking " << endl;



    float Ng, Na, Ngw, Naw;

    node = fSettings["IMU.Frequency"];
    if(!node.empty() && node.isInt())
    {
        mImuFreq = node.operator int();
        mImuPer = 0.001; //1.0 / (double) mImuFreq;
    }
    else
    {
        std::cerr << "*IMU.Frequency parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseGyro"];
    if(!node.empty() && node.isReal())
    {
        Ng = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseGyro parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseAcc"];
    if(!node.empty() && node.isReal())
    {
        Na = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseAcc parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.GyroWalk"];
    if(!node.empty() && node.isReal())
    {
        Ngw = node.real();
    }
    else
    {
        std::cerr << "*IMU.GyroWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.AccWalk"];
    if(!node.empty() && node.isReal())
    {
        Naw = node.real();
    }
    else
    {
        std::cerr << "*IMU.AccWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.fastInit"];
    mFastInit = false;
    if(!node.empty())
    {
        mFastInit = static_cast<int>(fSettings["IMU.fastInit"]) != 0;
    }

    if(mFastInit)
        cout << "Fast IMU initialization. Acceleration is not checked \n";

    if(b_miss_params)
    {
        return false;
    }

    const float sf = sqrt(mImuFreq);
    cout << endl;
    cout << "IMU frequency: " << mImuFreq << " Hz" << endl;
    cout << "IMU gyro noise: " << Ng << " rad/s/sqrt(Hz)" << endl;
    cout << "IMU gyro walk: " << Ngw << " rad/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer noise: " << Na << " m/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer walk: " << Naw << " m/s^3/sqrt(Hz)" << endl;

    mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);


    return true;
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

void Tracking::SetStepByStep(bool bSet)
{
    bStepByStep = bSet;
}

bool Tracking::GetStepByStep()
{
    return bStepByStep;
}



Sophus::SE3f Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, string filename)
{
    //cout << "GrabImageStereo" << endl;

    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;
    mImRight = imRectRight;

    if(mImGray.channels()==3)
    {
        //cout << "Image with 3 channels" << endl;
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        //cout << "Image with 4 channels" << endl;
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_BGRA2GRAY);
        }
    }

    //cout << "Incoming frame creation" << endl;

    if (mSensor == System::STEREO && !mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera);
    else if(mSensor == System::STEREO && mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr);
    else if(mSensor == System::IMU_STEREO && !mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,&mLastFrame,*mpImuCalib);
    else if(mSensor == System::IMU_STEREO && mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr,&mLastFrame,*mpImuCalib);

    //cout << "Incoming frame ended" << endl;

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
    vdStereoMatch_ms.push_back(mCurrentFrame.mTimeStereoMatch);
#endif

    //cout << "Tracking start" << endl;
    Track();
    //cout << "Tracking end" << endl;

    return mCurrentFrame.GetPose();
}


Sophus::SE3f Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, string filename)
{
    return GrabImageRGBD(imRGB,
                         imD,
                         timestamp,
                         PanopticFrameObservation(),
                         cv::Mat(),
                         filename);
}

Sophus::SE3f Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, const cv::Mat &panopticMask, string filename)
{
    return GrabImageRGBD(imRGB,
                         imD,
                         timestamp,
                         panopticMask,
                         cv::Mat(),
                         filename);
}

Sophus::SE3f Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, const cv::Mat &panopticMask, const cv::Mat &dynamicDepth, string filename)
{
    return GrabImageRGBD(imRGB,
                         imD,
                         timestamp,
                         BuildPanopticFrameObservation(panopticMask, timestamp, filename),
                         dynamicDepth,
                         filename);
}

Sophus::SE3f Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, const PanopticFrameObservation& panopticObservation, string filename)
{
    return GrabImageRGBD(imRGB,
                         imD,
                         timestamp,
                         panopticObservation,
                         cv::Mat(),
                         filename);
}

Sophus::SE3f Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, const PanopticFrameObservation& panopticObservation, const cv::Mat &dynamicDepth, string filename)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;
    cv::Mat backendDynamicDepth = dynamicDepth;
    mbCurrentFrameCreatedKeyFrame = false;
    mRgbdBackendDynamicDepth.release();

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);
    if(!backendDynamicDepth.empty())
    {
        if((fabs(mDepthMapFactor-1.0f)>1e-5) || backendDynamicDepth.type()!=CV_32F)
            backendDynamicDepth.convertTo(mRgbdBackendDynamicDepth,CV_32F,mDepthMapFactor);
        else
            mRgbdBackendDynamicDepth = backendDynamicDepth.clone();
    }

    if (mSensor == System::RGBD)
        mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera);
    else if(mSensor == System::IMU_RGBD)
        mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,&mLastFrame,*mpImuCalib);






    const string effectiveFilename = filename.empty() ? panopticObservation.frameName : filename;
    if(!panopticObservation.empty() &&
       (panopticObservation.rawPanopticMask.cols != imRGB.cols ||
        panopticObservation.rawPanopticMask.rows != imRGB.rows))
    {
        Verbose::PrintMess("Panoptic mask size mismatch for RGB-D frame " + effectiveFilename,
                           Verbose::VERBOSITY_NORMAL);
    }

    if(!panopticObservation.empty())
    {
        mCurrentFrame.AssignPanopticObservation(panopticObservation);
        if(!EnablePanopticSideChannelOnly())
        {
            ExtractInstanceRegionORB(mImGray, panopticObservation.rawPanopticMask, mCurrentFrame);
            mCurrentFrame.AssignPanopticObservation(panopticObservation);
        }

        if(!EnablePanopticSideChannelOnly() && !mRecentPanopticFrames.empty())
        {
            const cv::Mat refinedPanopticMask = RefinePanopticWithORBMatches(mRecentPanopticFrames.back(),
                                                                             mCurrentFrame,
                                                                             panopticObservation.rawPanopticMask);
            if(!refinedPanopticMask.empty())
            {
                const PanopticFrameObservation refinedObservation =
                    BuildPanopticFrameObservation(refinedPanopticMask, timestamp, effectiveFilename);
                mCurrentFrame.AssignPanopticObservation(refinedObservation);
            }
        }
    }

    if(DebugFocusFrame(mCurrentFrame.mnId) && mCurrentFrame.HasPanopticObservation())
    {
        const FrameFeatureDebugStats frameStats =
            CollectFrameFeatureDebugStats(mCurrentFrame, false, false, false);
        std::cout << "[STSLAM_FOCUS] frame=" << mCurrentFrame.mnId
                  << " stage=panoptic_rgbd_frame_summary"
                  << " raw_total_features=" << mCurrentFrame.N
                  << " raw_thing_features=" << mCurrentFrame.GetNumThingFeatures()
                  << " raw_stuff_features=" << mCurrentFrame.GetNumStuffFeatures()
                  << " " << FormatFrameFeatureDebugStats(frameStats)
                  << std::endl;
    }

    mCurrentFrame.mNameFile = effectiveFilename;
    mCurrentFrame.mnDataset = mnNumDataset;
    UpdateSparseFlowGeometryEvidence(mCurrentFrame, mImGray);

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
#endif

    Track();

    if(mCurrentFrame.HasPanopticObservation() && mCurrentFrame.isSet())
    {
        PushPanopticHistory(mCurrentFrame);
        if(!mbCurrentFrameCreatedKeyFrame)
            AppendWindowFrameSnapshot(mCurrentFrame);
    }
    else
    {
        mRecentPanopticFrames.clear();
        mvFramesSinceLastKeyFrame.clear();
    }

    mPreviousImGrayForSparseFlow = mImGray.clone();
    return mCurrentFrame.GetPose();
}


Sophus::SE3f Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename)
{
    mbCurrentFrameCreatedKeyFrame = false;
    mImGray = im;
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
    }

    if (mSensor == System::MONOCULAR)
    {
        if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET ||(lastID - initID) < mMaxFrames)
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
        else
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
    }
    else if(mSensor == System::IMU_MONOCULAR)
    {
        if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        {
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib);
        }
        else
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib);
    }

    if (mState==NO_IMAGES_YET)
        t0=timestamp;

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;
    UpdateSparseFlowGeometryEvidence(mCurrentFrame, mImGray);

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
#endif

    lastID = mCurrentFrame.mnId;
    Track();
    mRecentPanopticFrames.clear();
    mvFramesSinceLastKeyFrame.clear();
    mbCurrentFrameCreatedKeyFrame = false;

    mPreviousImGrayForSparseFlow = mImGray.clone();
    return mCurrentFrame.GetPose();
}

Sophus::SE3f Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, const cv::Mat &panopticMask, string filename)
{
    return GrabImageMonocular(im,
                              timestamp,
                              BuildPanopticFrameObservation(panopticMask, timestamp, filename),
                              filename);
}

Sophus::SE3f Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, const PanopticFrameObservation& panopticObservation, string filename)
{
    mbCurrentFrameCreatedKeyFrame = false;
    mImGray = im;
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
    }

    if (mSensor == System::MONOCULAR)
    {
        if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET ||(lastID - initID) < mMaxFrames)
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
        else
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
    }
    else if(mSensor == System::IMU_MONOCULAR)
    {
        if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        {
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib);
        }
        else
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib);
    }

    if (mState==NO_IMAGES_YET)
        t0=timestamp;

    const string effectiveFilename = filename.empty() ? panopticObservation.frameName : filename;
    if(!panopticObservation.empty() &&
       (panopticObservation.rawPanopticMask.cols != im.cols ||
        panopticObservation.rawPanopticMask.rows != im.rows))
    {
        Verbose::PrintMess("Panoptic mask size mismatch for frame " + effectiveFilename,
                           Verbose::VERBOSITY_NORMAL);
    }

    if(!panopticObservation.empty())
    {
        mCurrentFrame.AssignPanopticObservation(panopticObservation);
        if(!EnablePanopticSideChannelOnly())
        {
            ExtractInstanceRegionORB(mImGray, panopticObservation.rawPanopticMask, mCurrentFrame);
            mCurrentFrame.AssignPanopticObservation(panopticObservation);
        }

        if(!EnablePanopticSideChannelOnly() && !mRecentPanopticFrames.empty())
        {
            const cv::Mat refinedPanopticMask = RefinePanopticWithORBMatches(mRecentPanopticFrames.back(),
                                                                             mCurrentFrame,
                                                                             panopticObservation.rawPanopticMask);
            if(!refinedPanopticMask.empty())
            {
                const PanopticFrameObservation refinedObservation =
                    BuildPanopticFrameObservation(refinedPanopticMask, timestamp, effectiveFilename);
                mCurrentFrame.AssignPanopticObservation(refinedObservation);
            }
        }
    }

    if(DebugFocusFrame(mCurrentFrame.mnId) && mCurrentFrame.HasPanopticObservation())
    {
        const FrameFeatureDebugStats frameStats =
            CollectFrameFeatureDebugStats(mCurrentFrame, false, false, false);
        std::cout << "[STSLAM_FOCUS] frame=" << mCurrentFrame.mnId
                  << " stage=panoptic_frame_summary"
                  << " raw_total_features=" << mCurrentFrame.N
                  << " raw_thing_features=" << mCurrentFrame.GetNumThingFeatures()
                  << " raw_stuff_features=" << mCurrentFrame.GetNumStuffFeatures()
                  << " " << FormatFrameFeatureDebugStats(frameStats)
                  << std::endl;
    }

    mCurrentFrame.mNameFile = effectiveFilename;
    mCurrentFrame.mnDataset = mnNumDataset;
    UpdateSparseFlowGeometryEvidence(mCurrentFrame, mImGray);

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
#endif

    lastID = mCurrentFrame.mnId;
    Track();

    if(mCurrentFrame.HasPanopticObservation() && mCurrentFrame.isSet())
    {
        PushPanopticHistory(mCurrentFrame);
        if(!mbCurrentFrameCreatedKeyFrame)
            AppendWindowFrameSnapshot(mCurrentFrame);
    }
    else
    {
        mRecentPanopticFrames.clear();
        mvFramesSinceLastKeyFrame.clear();
    }

    mPreviousImGrayForSparseFlow = mImGray.clone();
    return mCurrentFrame.GetPose();
}

void Tracking::UpdateSparseFlowGeometryEvidence(Frame& frame, const cv::Mat& currentGray)
{
    frame.mvSparseFlowGeometryLabels.assign(frame.N, 0);
    frame.mvSparseFlowForwardBackwardErrors.assign(frame.N, -1.0f);
    frame.mvSparseFlowEpipolarErrors.assign(frame.N, -1.0f);

    if(!EnableSemanticCandidateSparseFlowGate() ||
       currentGray.empty() ||
       mPreviousImGrayForSparseFlow.empty() ||
       frame.N <= 0 ||
       frame.mvKeysUn.empty())
        return;

    cv::Mat current8u = currentGray;
    cv::Mat previous8u = mPreviousImGrayForSparseFlow;
    if(current8u.type() != CV_8UC1)
        currentGray.convertTo(current8u, CV_8UC1);
    if(previous8u.type() != CV_8UC1)
        mPreviousImGrayForSparseFlow.convertTo(previous8u, CV_8UC1);
    if(current8u.size() != previous8u.size())
        return;

    std::vector<cv::Point2f> currentPts;
    currentPts.reserve(frame.mvKeysUn.size());
    for(const cv::KeyPoint& keypoint : frame.mvKeysUn)
        currentPts.push_back(keypoint.pt);

    std::vector<cv::Point2f> previousPts;
    std::vector<uchar> backwardStatus;
    std::vector<float> backwardError;
    cv::calcOpticalFlowPyrLK(current8u,
                             previous8u,
                             currentPts,
                             previousPts,
                             backwardStatus,
                             backwardError,
                             cv::Size(21, 21),
                             3,
                             cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                                              30,
                                              0.01));

    std::vector<cv::Point2f> roundTripPts;
    std::vector<uchar> forwardStatus;
    std::vector<float> forwardError;
    cv::calcOpticalFlowPyrLK(previous8u,
                             current8u,
                             previousPts,
                             roundTripPts,
                             forwardStatus,
                             forwardError,
                             cv::Size(21, 21),
                             3,
                             cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                                              30,
                                              0.01));

    const double maxForwardBackwardError =
        GetSemanticSparseFlowMaxForwardBackwardErrorPx();
    std::vector<int> goodFeatureIndices;
    std::vector<cv::Point2f> goodCurrentPts;
    std::vector<cv::Point2f> goodPreviousPts;
    goodFeatureIndices.reserve(currentPts.size());
    goodCurrentPts.reserve(currentPts.size());
    goodPreviousPts.reserve(currentPts.size());

    const int nFlow = std::min({static_cast<int>(currentPts.size()),
                                static_cast<int>(previousPts.size()),
                                static_cast<int>(roundTripPts.size()),
                                static_cast<int>(backwardStatus.size()),
                                static_cast<int>(forwardStatus.size())});
    for(int idx = 0; idx < nFlow; ++idx)
    {
        if(!backwardStatus[idx] || !forwardStatus[idx])
            continue;

        const double fbError = cv::norm(currentPts[idx] - roundTripPts[idx]);
        frame.mvSparseFlowForwardBackwardErrors[idx] = static_cast<float>(fbError);
        if(fbError > maxForwardBackwardError)
            continue;

        goodFeatureIndices.push_back(idx);
        goodCurrentPts.push_back(currentPts[idx]);
        goodPreviousPts.push_back(previousPts[idx]);
    }

    const int minInliers = GetSemanticSparseFlowMinRansacInliers();
    if(static_cast<int>(goodFeatureIndices.size()) < minInliers)
        return;

    cv::Mat inlierMask;
    const cv::Mat fundamental = cv::findFundamentalMat(goodCurrentPts,
                                                       goodPreviousPts,
                                                       cv::FM_RANSAC,
                                                       GetSemanticSparseFlowRansacThresholdPx(),
                                                       0.99,
                                                       inlierMask);
    if(fundamental.empty() ||
       fundamental.rows != 3 ||
       fundamental.cols != 3 ||
       inlierMask.empty())
        return;

    std::vector<cv::Vec3f> epilinesInPrevious;
    cv::computeCorrespondEpilines(goodCurrentPts, 1, fundamental, epilinesInPrevious);
    if(epilinesInPrevious.size() != goodFeatureIndices.size())
        return;

    const double maxEpipolarError = GetSemanticSparseFlowMaxEpipolarErrorPx();
    int staticEvidence = 0;
    int dynamicEvidence = 0;
    int ransacInliers = 0;
    for(size_t i = 0; i < goodFeatureIndices.size(); ++i)
    {
        const int idx = goodFeatureIndices[i];
        const cv::Vec3f& line = epilinesInPrevious[i];
        const double denom = std::sqrt(static_cast<double>(line[0]) * line[0] +
                                       static_cast<double>(line[1]) * line[1]);
        if(denom <= 1e-9)
            continue;

        const double epipolarError =
            std::fabs(line[0] * goodPreviousPts[i].x +
                      line[1] * goodPreviousPts[i].y +
                      line[2]) / denom;
        frame.mvSparseFlowEpipolarErrors[idx] = static_cast<float>(epipolarError);
        const bool ransacInlier = inlierMask.at<uchar>(static_cast<int>(i)) != 0;
        if(ransacInlier)
            ++ransacInliers;

        if(ransacInlier && epipolarError <= maxEpipolarError)
        {
            frame.mvSparseFlowGeometryLabels[idx] = 1;
            ++staticEvidence;
        }
        else if(epipolarError > maxEpipolarError)
        {
            frame.mvSparseFlowGeometryLabels[idx] = -1;
            ++dynamicEvidence;
        }
    }

    if(static_cast<int>(ransacInliers) < minInliers)
    {
        std::fill(frame.mvSparseFlowGeometryLabels.begin(),
                  frame.mvSparseFlowGeometryLabels.end(),
                  0);
        return;
    }

    int candidateStatic = 0;
    int candidateDynamic = 0;
    int candidateUnknown = 0;
    for(int idx = 0; idx < frame.N; ++idx)
    {
        if(frame.GetFeatureInstanceId(static_cast<size_t>(idx)) <= 0)
            continue;
        const int label = GetSparseFlowGeometryLabel(frame, idx);
        if(label > 0)
            ++candidateStatic;
        else if(label < 0)
            ++candidateDynamic;
        else
            ++candidateUnknown;
    }

    std::cout << "[STSLAM_SPARSE_FLOW_GEOMETRY]"
              << " frame=" << frame.mnId
              << " features=" << frame.N
              << " good_flow=" << goodFeatureIndices.size()
              << " ransac_inliers=" << ransacInliers
              << " static_evidence=" << staticEvidence
              << " dynamic_evidence=" << dynamicEvidence
              << " candidate_static=" << candidateStatic
              << " candidate_dynamic=" << candidateDynamic
              << " candidate_unknown=" << candidateUnknown
              << std::endl;
}

cv::Mat Tracking::RefinePanopticWithORBMatches(const Frame& previousFrame,
                                               const Frame& currentFrame,
                                               const cv::Mat& currentRawPanopticMask) const
{
    if(currentRawPanopticMask.empty() || !previousFrame.HasPanopticObservation() || !currentFrame.HasPanopticObservation())
        return currentRawPanopticMask;

    if(mnLastPanopticRefinementFrameId != static_cast<unsigned long>(-1) &&
       currentFrame.mnId != mnLastPanopticRefinementFrameId + 1)
    {
        mmPanopticIdCorrectionStreaks.clear();
    }
    mnLastPanopticRefinementFrameId = currentFrame.mnId;

    std::map<int, std::vector<int>> previousGroups;
    std::map<int, std::vector<int>> currentGroups;

    for(int idx = 0; idx < previousFrame.N; ++idx)
    {
        const int instanceId = previousFrame.GetFeatureInstanceId(idx);
        if(instanceId > 0)
            previousGroups[instanceId].push_back(idx);
    }

    for(int idx = 0; idx < currentFrame.N; ++idx)
    {
        const int instanceId = currentFrame.GetFeatureInstanceId(idx);
        if(instanceId > 0)
            currentGroups[instanceId].push_back(idx);
    }

    std::map<int, int> rawToCanonicalMap;
    std::map<int, PanopticCanonicalAssociation> canonicalAssociationMap;
    for(const std::pair<const int, std::vector<int>>& currentEntry : currentGroups)
    {
        rawToCanonicalMap[currentEntry.first] = currentEntry.first;
        PanopticCanonicalAssociation association;
        association.rawInstanceId = currentEntry.first;
        association.canonicalInstanceId = currentEntry.first;
        association.featureCount = static_cast<int>(currentEntry.second.size());
        const InstanceObservation* observation =
            FindInstanceObservation(currentFrame, currentEntry.first);
        if(observation)
        {
            association.semanticId = observation->semanticId;
            association.maskArea = observation->area;
            association.bbox = observation->bbox;
        }
        canonicalAssociationMap[currentEntry.first] = association;
    }
    auto storeCurrentCanonicalMap = [&]()
    {
        mmFramePanopticRawToCanonicalIds[currentFrame.mnId] = rawToCanonicalMap;
        mmFramePanopticCanonicalAssociations[currentFrame.mnId] = canonicalAssociationMap;
        while(mmFramePanopticRawToCanonicalIds.size() > 12)
            mmFramePanopticRawToCanonicalIds.erase(mmFramePanopticRawToCanonicalIds.begin());
        while(mmFramePanopticCanonicalAssociations.size() > 12)
            mmFramePanopticCanonicalAssociations.erase(mmFramePanopticCanonicalAssociations.begin());
    };
    auto formatCanonicalAssociations = [&]() -> std::string
    {
        std::ostringstream associations;
        size_t printed = 0;
        for(const std::pair<const int, PanopticCanonicalAssociation>& entry : canonicalAssociationMap)
        {
            if(printed > 0)
                associations << ",";
            const PanopticCanonicalAssociation& association = entry.second;
            associations << association.rawInstanceId << "->" << association.canonicalInstanceId
                         << ":sem=" << association.semanticId
                         << ":feat=" << association.featureCount
                         << ":best=" << association.bestMatches
                         << ":second=" << association.secondBestMatches
                         << ":streak=" << association.correctionStreak
                         << ":perm=" << (association.permanentCorrection ? 1 : 0)
                         << ":frame=" << (association.frameCorrection ? 1 : 0);
            ++printed;
            if(printed >= 8)
                break;
        }
        return printed > 0 ? associations.str() : "none";
    };

    if(previousGroups.empty() || currentGroups.empty())
    {
        storeCurrentCanonicalMap();
        return currentRawPanopticMask.clone();
    }

    struct MatchCandidate
    {
        int previousInstanceId = -1;
        int currentInstanceId = -1;
        int bestMatches = 0;
        int secondBestMatches = 0;
    };

    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<MatchCandidate> candidates;
    candidates.reserve(previousGroups.size());
    const bool descriptorGate = EnablePanopticRefinementDescriptorGate();
    const int descriptorMaxDistance = GetPanopticRefinementDescriptorMaxDistance();

    for(const auto& previousEntry : previousGroups)
    {
        const cv::Mat previousDescriptors = ExtractDescriptorRows(previousFrame.mDescriptors, previousEntry.second);
        if(previousDescriptors.empty())
            continue;

        std::vector<std::pair<int, int>> localScores;
        localScores.reserve(currentGroups.size());

        for(const auto& currentEntry : currentGroups)
        {
            const cv::Mat currentDescriptors = ExtractDescriptorRows(currentFrame.mDescriptors, currentEntry.second);
            if(currentDescriptors.empty())
                continue;

            std::vector<cv::DMatch> matches;
            matcher.match(previousDescriptors, currentDescriptors, matches);
            int validMatches = 0;
            for(size_t matchIdx = 0; matchIdx < matches.size(); ++matchIdx)
            {
                if(!descriptorGate || matches[matchIdx].distance <= descriptorMaxDistance)
                    ++validMatches;
            }
            localScores.push_back(std::make_pair(currentEntry.first, validMatches));
        }

        if(localScores.empty())
            continue;

        std::sort(localScores.begin(), localScores.end(),
                  [](const std::pair<int, int>& lhs, const std::pair<int, int>& rhs)
                  {
                      if(lhs.second != rhs.second)
                          return lhs.second > rhs.second;
                      return lhs.first < rhs.first;
                  });

        MatchCandidate candidate;
        candidate.previousInstanceId = previousEntry.first;
        candidate.currentInstanceId = localScores.front().first;
        candidate.bestMatches = localScores.front().second;
        candidate.secondBestMatches = (localScores.size() > 1) ? localScores[1].second : 0;
        candidates.push_back(candidate);
    }

    if(candidates.empty())
    {
        storeCurrentCanonicalMap();
        return currentRawPanopticMask.clone();
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const MatchCandidate& lhs, const MatchCandidate& rhs)
              {
                  if(lhs.bestMatches != rhs.bestMatches)
                      return lhs.bestMatches > rhs.bestMatches;
                  if(lhs.secondBestMatches != rhs.secondBestMatches)
                      return lhs.secondBestMatches < rhs.secondBestMatches;
                  if(lhs.previousInstanceId != rhs.previousInstanceId)
                      return lhs.previousInstanceId < rhs.previousInstanceId;
                  return lhs.currentInstanceId < rhs.currentInstanceId;
              });

    std::map<int, int> correctionMap;
    std::set<int> usedPreviousIds;
    std::set<int> usedCurrentIds;
    int permanentCorrections = 0;
    for(const auto& permanentEntry : mmPermanentPanopticIdCorrections)
    {
        const int currentInstanceId = permanentEntry.first;
        const int canonicalInstanceId = permanentEntry.second;
        if(currentInstanceId <= 0 ||
           canonicalInstanceId <= 0 ||
           currentGroups.find(currentInstanceId) == currentGroups.end() ||
           usedPreviousIds.count(canonicalInstanceId) ||
           usedCurrentIds.count(currentInstanceId))
        {
            continue;
        }

        correctionMap[currentInstanceId] = canonicalInstanceId;
        rawToCanonicalMap[currentInstanceId] = canonicalInstanceId;
        PanopticCanonicalAssociation& association = canonicalAssociationMap[currentInstanceId];
        association.canonicalInstanceId = canonicalInstanceId;
        association.matchedToPrevious = true;
        association.permanentCorrection = true;
        usedPreviousIds.insert(canonicalInstanceId);
        usedCurrentIds.insert(currentInstanceId);
        ++permanentCorrections;
    }

    std::vector<std::pair<int, int>> acceptedFrameCorrections;
    for(const MatchCandidate& candidate : candidates)
    {
        if(candidate.bestMatches < mnPanopticRefinementMinMatches)
            continue;
        if(candidate.secondBestMatches > 0 &&
           static_cast<float>(candidate.bestMatches) <
               static_cast<float>(candidate.secondBestMatches) * mfPanopticRefinementSecondBestRatio)
            continue;
        if(usedPreviousIds.count(candidate.previousInstanceId) || usedCurrentIds.count(candidate.currentInstanceId))
            continue;

        correctionMap[candidate.currentInstanceId] = candidate.previousInstanceId;
        rawToCanonicalMap[candidate.currentInstanceId] = candidate.previousInstanceId;
        PanopticCanonicalAssociation& association =
            canonicalAssociationMap[candidate.currentInstanceId];
        association.canonicalInstanceId = candidate.previousInstanceId;
        association.bestMatches = candidate.bestMatches;
        association.secondBestMatches = candidate.secondBestMatches;
        association.matchedToPrevious = true;
        association.frameCorrection = true;
        usedPreviousIds.insert(candidate.previousInstanceId);
        usedCurrentIds.insert(candidate.currentInstanceId);
        acceptedFrameCorrections.push_back(
            std::make_pair(candidate.currentInstanceId, candidate.previousInstanceId));
    }

    std::set<std::pair<int, int>> currentAcceptedPairs;
    for(const std::pair<int, int>& acceptedPair : acceptedFrameCorrections)
    {
        currentAcceptedPairs.insert(acceptedPair);
        const int streak = ++mmPanopticIdCorrectionStreaks[acceptedPair];
        canonicalAssociationMap[acceptedPair.first].correctionStreak = streak;
        if(acceptedPair.first != acceptedPair.second &&
           streak >= mnPanopticPermanentCorrectionMinStreak)
        {
            mmPermanentPanopticIdCorrections[acceptedPair.first] = acceptedPair.second;
            canonicalAssociationMap[acceptedPair.first].permanentCorrection = true;
        }
    }
    for(std::map<std::pair<int, int>, int>::iterator it = mmPanopticIdCorrectionStreaks.begin();
        it != mmPanopticIdCorrectionStreaks.end(); )
    {
        if(currentAcceptedPairs.count(it->first) == 0)
            it = mmPanopticIdCorrectionStreaks.erase(it);
        else
            ++it;
    }

    if(correctionMap.empty())
    {
        if(DebugInstanceTracklets() || DebugFocusFrame(currentFrame.mnId))
        {
            std::ostringstream canonicalAssociations;
            size_t printed = 0;
            for(const std::pair<const int, int>& entry : rawToCanonicalMap)
            {
                if(printed > 0)
                    canonicalAssociations << ",";
                canonicalAssociations << entry.first << "->" << entry.second;
                ++printed;
                if(printed >= 12)
                    break;
            }
            std::cout << "[STSLAM_PANOPTIC_REFINE] frame=" << currentFrame.mnId
                      << " previous_frame=" << previousFrame.mnId
                      << " descriptor_gate=" << (descriptorGate ? 1 : 0)
                      << " descriptor_max_distance=" << descriptorMaxDistance
                      << " previous_instances=" << previousGroups.size()
                      << " current_instances=" << currentGroups.size()
                      << " candidates=" << candidates.size()
                      << " corrections=0"
                      << " permanent_corrections=0"
                      << " permanent_table_size=" << mmPermanentPanopticIdCorrections.size()
                      << " correction_streaks=" << mmPanopticIdCorrectionStreaks.size()
                      << " permanent_min_streak=" << mnPanopticPermanentCorrectionMinStreak
                      << " correction_map=none"
                      << " canonical_map=" << canonicalAssociations.str()
                      << " canonical_assoc=" << formatCanonicalAssociations()
                      << std::endl;
        }
        storeCurrentCanonicalMap();
        return currentRawPanopticMask.clone();
    }

    if(DebugInstanceTracklets() || DebugFocusFrame(currentFrame.mnId))
    {
        std::ostringstream corrections;
        std::ostringstream canonicalAssociations;
        size_t printed = 0;
        for(const std::pair<const int, int>& entry : correctionMap)
        {
            if(printed > 0)
                corrections << ",";
            corrections << entry.first << "->" << entry.second;
            ++printed;
            if(printed >= 8)
                break;
        }
        printed = 0;
        for(const std::pair<const int, int>& entry : rawToCanonicalMap)
        {
            if(printed > 0)
                canonicalAssociations << ",";
            canonicalAssociations << entry.first << "->" << entry.second;
            ++printed;
            if(printed >= 12)
                break;
        }
        std::cout << "[STSLAM_PANOPTIC_REFINE] frame=" << currentFrame.mnId
                  << " previous_frame=" << previousFrame.mnId
                  << " descriptor_gate=" << (descriptorGate ? 1 : 0)
                  << " descriptor_max_distance=" << descriptorMaxDistance
                  << " previous_instances=" << previousGroups.size()
                  << " current_instances=" << currentGroups.size()
                  << " candidates=" << candidates.size()
                  << " corrections=" << correctionMap.size()
                  << " permanent_corrections=" << permanentCorrections
                  << " permanent_table_size=" << mmPermanentPanopticIdCorrections.size()
                  << " correction_streaks=" << mmPanopticIdCorrectionStreaks.size()
                  << " permanent_min_streak=" << mnPanopticPermanentCorrectionMinStreak
                  << " correction_map=" << corrections.str()
                  << " canonical_map=" << canonicalAssociations.str()
                  << " canonical_assoc=" << formatCanonicalAssociations()
                  << std::endl;
    }

    storeCurrentCanonicalMap();

    cv::Mat refinedPanoptic = currentRawPanopticMask.clone();
    for(int y = 0; y < refinedPanoptic.rows; ++y)
    {
        for(int x = 0; x < refinedPanoptic.cols; ++x)
        {
            const int panopticId = ReadPanopticIdAt(refinedPanoptic, x, y);
            if(panopticId <= 0)
                continue;

            const int semanticId = DecodePanopticSemanticId(panopticId);
            const int instanceId = DecodePanopticInstanceId(panopticId);
            const auto correctionIt = correctionMap.find(instanceId);
            if(correctionIt == correctionMap.end())
                continue;

            const int refinedId = semanticId * kDefaultPanopticDivisor + correctionIt->second;
            if(refinedPanoptic.type() == CV_16UC1)
                refinedPanoptic.at<unsigned short>(y, x) = static_cast<unsigned short>(refinedId);
            else
                refinedPanoptic.at<int>(y, x) = refinedId;
        }
    }

    return refinedPanoptic;
}

void Tracking::ExtractInstanceRegionORB(const cv::Mat& imGray,
                                        const cv::Mat& panopticMask,
                                        Frame& frame) const
{
    if(imGray.empty() || panopticMask.empty() || frame.Nleft != -1)
        return;

    frame.mBowVec.clear();
    frame.mFeatVec.clear();

    for(auto it = frame.mmInstanceObservations.begin(); it != frame.mmInstanceObservations.end(); ++it)
    {
        const int instanceId = it->first;
        const InstanceObservation& observation = it->second;
        if(instanceId <= 0 || observation.mask.empty())
            continue;

        const int existingCount =
            static_cast<int>(CollectFeatureIndicesForInstance(frame, instanceId).size());
        const int maskArea = cv::countNonZero(observation.mask);
        if(maskArea <= 0)
            continue;

        const int targetCount =
            ComputeTargetInstanceFeatureCount(frame, imGray, maskArea, existingCount);
        if(existingCount >= targetCount)
            continue;

        const int extraCount = targetCount - existingCount;
        cv::Ptr<cv::ORB> orbExtra = cv::ORB::create(extraCount,
                                                    frame.mfScaleFactor,
                                                    frame.mnScaleLevels,
                                                    19,
                                                    0,
                                                    2,
                                                    cv::ORB::HARRIS_SCORE,
                                                    31,
                                                    20);
        std::vector<cv::KeyPoint> extraKeypoints;
        cv::Mat extraDescriptors;
        orbExtra->detectAndCompute(imGray, observation.mask, extraKeypoints, extraDescriptors, false);
        if(extraDescriptors.empty())
            continue;

        int addedCount = 0;
        for(size_t idx = 0; idx < extraKeypoints.size() && addedCount < extraCount; ++idx)
        {
            const cv::KeyPoint keypoint = extraKeypoints[idx];
            const cv::KeyPoint keypointUn = UndistortExtraKeyPoint(keypoint, frame);
            if(HasNearbyInstanceFeature(frame, keypointUn, instanceId))
                continue;

            AppendMonoFeatureToFrame(frame,
                                     keypoint,
                                     keypointUn,
                                     extraDescriptors.row(static_cast<int>(idx)));
            ++addedCount;
        }

        if(DebugFocusFrame(frame.mnId))
        {
            std::cout << "[STSLAM_FOCUS] frame=" << frame.mnId
                      << " stage=instance_orb_extract"
                      << " instance_id=" << instanceId
                      << " semantic=" << observation.semanticId
                      << " mask_area=" << maskArea
                      << " existing_features=" << existingCount
                      << " target_features=" << targetCount
                      << " requested_extra=" << extraCount
                      << " detected_extra=" << static_cast<int>(extraKeypoints.size())
                      << " added_extra=" << addedCount
                      << std::endl;
        }
    }
}

void Tracking::ProcessInstances()
{
    if(!mCurrentFrame.HasPanopticObservation())
        return;

    long currentMapId = -1;
    long currentMapOriginFrame = -1;
    long currentMapOriginKF = -1;
    int currentMapAgeFrames = -1;
    if(mpAtlas)
    {
        Map* pCurrentMap = mpAtlas->GetCurrentMap();
        if(pCurrentMap)
        {
            currentMapId = static_cast<long>(pCurrentMap->GetId());
            KeyFrame* pOriginKF = pCurrentMap->GetOriginKF();
            if(pOriginKF)
            {
                currentMapOriginFrame = static_cast<long>(pOriginKF->mnFrameId);
                currentMapOriginKF = static_cast<long>(pOriginKF->mnId);
                currentMapAgeFrames =
                    std::max(0, static_cast<int>(static_cast<long>(mCurrentFrame.mnId) -
                                                 currentMapOriginFrame));
            }
        }
    }

    for(const auto& observationEntry : mCurrentFrame.mmInstanceObservations)
    {
        const int instanceId = observationEntry.first;
        const InstanceObservation& observation = observationEntry.second;
        if(instanceId <= 0)
            continue;

        Instance* pInstance = mpAtlas->GetInstance(instanceId);
        if(!pInstance)
        {
            pInstance = new Instance(instanceId, observation.semanticId);
            pInstance = mpAtlas->AddInstance(pInstance);
        }
        if(observation.semanticId > 0)
            pInstance->SetSemanticLabel(observation.semanticId);
        const int previousSeenFrame = pInstance->MarkSeenInFrame(static_cast<int>(mCurrentFrame.mnId));

        const std::vector<int> featureIndices = CollectFeatureIndicesForInstance(mCurrentFrame, instanceId);
        int matchedMapPointCount = 0;
        for(const int featureIdx : featureIndices)
        {
            if(featureIdx < 0 || featureIdx >= static_cast<int>(mCurrentFrame.mvpMapPoints.size()))
                continue;

            MapPoint* pMP = mCurrentFrame.mvpMapPoints[featureIdx];
            if(!pMP || pMP->isBad())
                continue;

            Instance* pAnnotatedInstance =
                (mSensor == System::RGBD || mSensor == System::IMU_RGBD) ?
                AnnotateMapPointWithInstance(pMP, instanceId, observation.semanticId, false) :
                BindMapPointToInstance(pMP, instanceId, observation.semanticId, false);
            if(!pAnnotatedInstance)
                continue;
            pMP->UpdateObservationStats(static_cast<int>(mCurrentFrame.mnId));
            ++matchedMapPointCount;
        }

        if(!pInstance->IsInitialized())
        {
            if(static_cast<int>(featureIndices.size()) < mnInstanceInitializationMinFeatures)
            {
                if(DebugInstanceInitializationFor(instanceId) ||
                   DebugInstanceLifecycleFor(instanceId) ||
                   DebugFocusFrame(mCurrentFrame.mnId))
                {
                    std::cout << "[STSLAM_INSTANCE_PROCESS] frame=" << mCurrentFrame.mnId
                              << " instance_id=" << instanceId
                              << " map_id=" << currentMapId
                              << " map_origin_frame=" << currentMapOriginFrame
                              << " map_origin_kf=" << currentMapOriginKF
                              << " map_age_frames=" << currentMapAgeFrames
                              << " initialized=0"
                              << " reason=low_current_features"
                              << " current_features=" << featureIndices.size()
                              << " min_features=" << mnInstanceInitializationMinFeatures
                              << " previous_seen_frame=" << previousSeenFrame
                              << " recent_history=" << mRecentPanopticFrames.size()
                              << std::endl;
                }
                pInstance->ResetInitializationCounter();
                continue;
            }

            const int initFrameCount =
                pInstance->AdvanceInitializationCounter(previousSeenFrame + 1 == static_cast<int>(mCurrentFrame.mnId));

            if(DebugInstanceLifecycleFor(instanceId) || DebugFocusFrame(mCurrentFrame.mnId))
            {
                std::cout << "[STSLAM_INSTANCE_PROCESS] frame=" << mCurrentFrame.mnId
                          << " instance_id=" << instanceId
                          << " map_id=" << currentMapId
                          << " map_origin_frame=" << currentMapOriginFrame
                          << " map_origin_kf=" << currentMapOriginKF
                          << " map_age_frames=" << currentMapAgeFrames
                          << " initialized=0"
                          << " reason=init_candidate"
                          << " current_features=" << featureIndices.size()
                          << " matched_mappoints=" << matchedMapPointCount
                          << " previous_seen_frame=" << previousSeenFrame
                          << " consecutive="
                          << ((previousSeenFrame + 1 == static_cast<int>(mCurrentFrame.mnId)) ? 1 : 0)
                          << " init_frame_count=" << initFrameCount
                          << " recent_history=" << mRecentPanopticFrames.size()
                          << std::endl;
            }

            if(initFrameCount >= 3 && mRecentPanopticFrames.size() >= 2)
                InitializeInstance(pInstance, featureIndices);
            else if(DebugInstanceInitializationFor(instanceId) ||
                    DebugInstanceLifecycleFor(instanceId) ||
                    DebugFocusFrame(mCurrentFrame.mnId))
            {
                std::cout << "[STSLAM_INSTANCE_PROCESS] frame=" << mCurrentFrame.mnId
                          << " instance_id=" << instanceId
                          << " map_id=" << currentMapId
                          << " map_origin_frame=" << currentMapOriginFrame
                          << " map_origin_kf=" << currentMapOriginKF
                          << " map_age_frames=" << currentMapAgeFrames
                          << " initialized=0"
                          << " reason=waiting_initialization_window"
                          << " init_frame_count=" << initFrameCount
                          << " required_init_frame_count=3"
                          << " recent_history=" << mRecentPanopticFrames.size()
                          << " required_recent_history=2"
                          << std::endl;
            }

            continue;
        }

        const bool maturedPrediction =
            HasMatureInstancePrediction(pInstance,
                                        static_cast<int>(mCurrentFrame.mnId),
                                        matchedMapPointCount);
        double observedMeanRadius = 0.0;
        int observedSizePointCount = 0;
        const bool recordedClassSizePrior =
            observation.semanticId > 0 &&
            ComputeObservedInstanceMeanRadius(mCurrentFrame,
                                              featureIndices,
                                              observedMeanRadius,
                                              observedSizePointCount);
        const bool shouldRecordClassSizePrior =
            recordedClassSizePrior &&
            observedSizePointCount >= mnInstanceInitializationMinFeatures;
        if(shouldRecordClassSizePrior)
        {
            Map* pCurrentMap = mpAtlas ? mpAtlas->GetCurrentMap() : static_cast<Map*>(NULL);
            if(pCurrentMap)
                pCurrentMap->RecordInstanceClassSizePrior(observation.semanticId,
                                                          instanceId,
                                                          observedMeanRadius);
        }

        if(DebugFocusFrame(mCurrentFrame.mnId))
        {
            std::cout << "[STSLAM_FOCUS] frame=" << mCurrentFrame.mnId
                      << " stage=process_instance"
                      << " instance_id=" << instanceId
                      << " semantic=" << observation.semanticId
                      << " frame_features=" << static_cast<int>(featureIndices.size())
                      << " matched_map_points=" << matchedMapPointCount
                      << " map_points=" << static_cast<int>(pInstance->NumMapPoints())
                      << " initialized=" << (pInstance->IsInitialized() ? 1 : 0)
                      << " initialized_frame=" << pInstance->GetInitializedFrame()
                      << " matured_prediction=" << (maturedPrediction ? 1 : 0)
                      << " class_size_prior_recorded=" << (shouldRecordClassSizePrior ? 1 : 0)
                      << " class_size_points=" << observedSizePointCount
                      << " class_size_mean_radius=" << observedMeanRadius
                      << std::endl;
        }

        Sophus::SE3f selectedVelocity;
        bool selectedZeroVelocity = false;
        bool selectedDynamicVelocity = false;
        bool selectedUncertainVelocity = false;
        Instance::InstanceMotionStateRecord latestMotionState;
        const bool hasLatestMotionState =
            pInstance->GetLatestInstanceMotionState(latestMotionState) &&
            latestMotionState.state != Instance::kDynamicEntityUnknown &&
            IsFiniteSE3(latestMotionState.pose) &&
            IsFiniteSE3(latestMotionState.velocity);
        InstanceMotionGateResult motionGate;
        if(maturedPrediction)
        {
            const Sophus::SE3f rawVelocity =
                hasLatestMotionState ? latestMotionState.velocity : pInstance->GetVelocity();
            selectedVelocity = rawVelocity;
            bool zeroVelocityReactivated = false;
            int zeroVelocityReactivationTracklets = 0;
            int zeroVelocityReactivationTriangulated = 0;
            int zeroVelocityReactivationCandidateRejected = 0;
            bool zeroVelocityReactivationSvdReliable = false;
            std::string zeroVelocityReactivationSvdReason = "not_checked";
            double zeroVelocityReactivationSvdCondition = 0.0;
            double zeroVelocityReactivationSvdPlanarity = 0.0;
            double zeroVelocityReactivationTranslationOnlyNorm = 0.0;
            bool zeroVelocityReactivationUsedTranslationOnly = false;
            std::string zeroVelocityReactivationReason = "none";
            const bool latestNearZeroUncertainDynamicEntity =
                hasLatestMotionState &&
                latestMotionState.state == Instance::kUncertainDynamicEntity &&
                IsSmallInstanceMotion(latestMotionState.velocity,
                                      GetStrictStaticZeroVelocityMaxTranslation(),
                                      GetStrictStaticZeroVelocityMaxRotationDeg());
            const bool dormantDynamicEntityCanReactivate =
                hasLatestMotionState &&
                (latestMotionState.state == Instance::kZeroVelocityDynamicEntity ||
                 latestNearZeroUncertainDynamicEntity);
            if(UseStrictPaperArchitectureDefaults() &&
               EnableZeroVelocityDynamicReactivation() &&
               dormantDynamicEntityCanReactivate &&
               mRecentPanopticFrames.size() >= 2)
            {
                const Frame& frameTau2 = mRecentPanopticFrames[mRecentPanopticFrames.size() - 2];
                const Frame& frameTau1 = mRecentPanopticFrames.back();
                const std::vector<std::array<int, 3>> reactivationTracklets =
                    CollectInstanceTracklets(frameTau2, frameTau1, mCurrentFrame, instanceId);
                zeroVelocityReactivationTracklets =
                    static_cast<int>(reactivationTracklets.size());
                if(zeroVelocityReactivationTracklets >=
                   GetZeroVelocityDynamicReactivationMinTracklets())
                {
                    std::vector<Eigen::Vector3f> pointsTau2;
                    std::vector<Eigen::Vector3f> pointsTau1;
                    std::vector<double> weights;
                    pointsTau2.reserve(reactivationTracklets.size());
                    pointsTau1.reserve(reactivationTracklets.size());
                    weights.reserve(reactivationTracklets.size());
                    for(const std::array<int, 3>& tracklet : reactivationTracklets)
                    {
                        Eigen::Vector3f pointTau2;
                        Eigen::Vector3f pointTau1;
                        TriangulationQuality qualityTau2Tau1;
                        TriangulationQuality qualityTau1Tau;
                        const bool okTau2Tau1 =
                            TriangulateMatchedFeatures(frameTau2,
                                                       tracklet[0],
                                                       frameTau1,
                                                       tracklet[1],
                                                       pointTau2,
                                                       &qualityTau2Tau1);
                        const bool okTau1Tau =
                            TriangulateMatchedFeatures(frameTau1,
                                                       tracklet[1],
                                                       mCurrentFrame,
                                                       tracklet[2],
                                                       pointTau1,
                                                       &qualityTau1Tau);
                        if(!okTau2Tau1 || !okTau1Tau ||
                           !pointTau2.allFinite() || !pointTau1.allFinite())
                        {
                            continue;
                        }

                        double trackletWeight =
                            std::max(0.05,
                                     std::min(1.0,
                                              std::min(qualityTau2Tau1.weight,
                                                       qualityTau1Tau.weight)));
                        if(EnableTriFrameTrackletCandidateQuality())
                        {
                            const TriFrameTrackletCandidateQuality candidateQuality =
                                EvaluateTriFrameTrackletCandidate(frameTau2,
                                                                  tracklet[0],
                                                                  frameTau1,
                                                                  tracklet[1],
                                                                  mCurrentFrame,
                                                                  tracklet[2],
                                                                  qualityTau2Tau1,
                                                                  qualityTau1Tau);
                            if(!candidateQuality.valid ||
                               !candidateQuality.structureEligible)
                            {
                                ++zeroVelocityReactivationCandidateRejected;
                            }
                        }

                        pointsTau2.push_back(pointTau2);
                        pointsTau1.push_back(pointTau1);
                        weights.push_back(trackletWeight);
                    }
                    zeroVelocityReactivationTriangulated =
                        static_cast<int>(pointsTau2.size());
                    if(zeroVelocityReactivationTriangulated >=
                       GetZeroVelocityDynamicReactivationMinTracklets())
                    {
                        const Sophus::SE3f tentativeSvdVelocity =
                            SolveWeightedRigidTransformSVD(pointsTau2, pointsTau1, weights);
                        Sophus::SE3f tentativeVelocity = tentativeSvdVelocity;
                        if(IsFiniteSE3(tentativeSvdVelocity))
                        {
                            const InstanceSvdMotionDiagnostics reactivationSvdDiagnostics =
                                ComputeInstanceSvdMotionDiagnostics(pointsTau2,
                                                                    pointsTau1,
                                                                    weights,
                                                                    tentativeSvdVelocity);
                            InstanceMotionGateResult noZeroMotionGate;
                            zeroVelocityReactivationSvdReason =
                                ClassifyStrictSvdMotionReliability(reactivationSvdDiagnostics,
                                                                   noZeroMotionGate);
                            zeroVelocityReactivationSvdReliable =
                                zeroVelocityReactivationSvdReason == "reliable" ||
                                zeroVelocityReactivationSvdReason == "disabled";
                            zeroVelocityReactivationSvdCondition =
                                reactivationSvdDiagnostics.condition;
                            zeroVelocityReactivationSvdPlanarity =
                                reactivationSvdDiagnostics.planarityRatio;

                            double totalWeight = 0.0;
                            Eigen::Vector3f centroidTau2 = Eigen::Vector3f::Zero();
                            Eigen::Vector3f centroidTau1 = Eigen::Vector3f::Zero();
                            for(size_t idx = 0; idx < pointsTau2.size() &&
                                                  idx < pointsTau1.size(); ++idx)
                            {
                                const double weight =
                                    (idx < weights.size() &&
                                     std::isfinite(weights[idx]) &&
                                     weights[idx] > 0.0) ? weights[idx] : 1.0;
                                centroidTau2 += static_cast<float>(weight) * pointsTau2[idx];
                                centroidTau1 += static_cast<float>(weight) * pointsTau1[idx];
                                totalWeight += weight;
                            }
                            Sophus::SE3f translationOnlyVelocity;
                            if(totalWeight > 0.0)
                            {
                                centroidTau2 /= static_cast<float>(totalWeight);
                                centroidTau1 /= static_cast<float>(totalWeight);
                                translationOnlyVelocity =
                                    Sophus::SE3f(Eigen::Matrix3f::Identity(),
                                                 centroidTau1 - centroidTau2);
                            }
                            if(IsFiniteSE3(translationOnlyVelocity))
                                zeroVelocityReactivationTranslationOnlyNorm =
                                    translationOnlyVelocity.translation().norm();

                            (void)translationOnlyVelocity;
                        }

                        if(IsFiniteSE3(tentativeVelocity) &&
                           !IsSmallInstanceMotion(tentativeVelocity,
                                                  GetStrictStaticZeroVelocityMaxTranslation(),
                                                  GetStrictStaticZeroVelocityMaxRotationDeg()))
                        {
                            selectedVelocity = tentativeVelocity;
                            selectedUncertainVelocity = true;
                            pInstance->RecordMotionGateState(
                                static_cast<int>(kInstanceMotionUncertain),
                                tentativeVelocity,
                                static_cast<int>(mCurrentFrame.mnId),
                                static_cast<float>(GetInstanceStaticVelocityDecay()));
                            zeroVelocityReactivated = true;
                            zeroVelocityReactivationReason = "nonzero_triframe_motion";
                        }
                        else
                        {
                            zeroVelocityReactivationReason = "small_or_invalid_motion";
                        }
                    }
                    else
                    {
                        zeroVelocityReactivationReason = "low_triangulated_tracklets";
                    }
                }
                else
                {
                    zeroVelocityReactivationReason = "low_tracklets";
                }
            }
            const bool enableMotionGate = EnableInstanceResidualMotionGate();
            if(zeroVelocityReactivated)
            {
                selectedZeroVelocity = false;
                selectedDynamicVelocity = false;
            }
            else if(enableMotionGate)
                motionGate = EvaluateInstanceResidualMotionGate(mCurrentFrame,
                                                                featureIndices,
                                                                rawVelocity);
            else
            {
                if(UseStrictPaperArchitectureDefaults() &&
                   hasLatestMotionState &&
                   latestMotionState.state == Instance::kZeroVelocityDynamicEntity)
                    selectedZeroVelocity = true;
                else
                    selectedDynamicVelocity = true;
            }

            bool stableStaticMotion = false;
            bool stableDynamicMotion = false;
            if(enableMotionGate && motionGate.valid)
            {
                pInstance->RecordMotionGateState(static_cast<int>(motionGate.state),
                                                 rawVelocity,
                                                 static_cast<int>(mCurrentFrame.mnId),
                                                 static_cast<float>(GetInstanceStaticVelocityDecay()));
                stableStaticMotion =
                    pInstance->HasStaticMotionEvidence(GetInstanceStaticMotionConfirmFrames());
                stableDynamicMotion =
                    pInstance->HasDynamicMotionEvidence(GetInstanceDynamicMotionConfirmFrames());
                selectedVelocity = pInstance->GetVelocity();
            }

            if(enableMotionGate && motionGate.valid &&
               motionGate.state == kInstanceMotionStatic && stableStaticMotion)
            {
                selectedZeroVelocity = true;
            }
            else if(enableMotionGate && motionGate.valid &&
                    motionGate.state == kInstanceMotionDynamic && stableDynamicMotion)
            {
                selectedDynamicVelocity = true;
            }
            else if(enableMotionGate && motionGate.valid && motionGate.state == kInstanceMotionUncertain)
            {
                selectedUncertainVelocity = true;
            }
            else if(enableMotionGate && motionGate.valid)
            {
                selectedUncertainVelocity = true;
            }
            else if(enableMotionGate && !motionGate.valid)
            {
                pInstance->RecordMotionGateState(static_cast<int>(kInstanceMotionUncertain),
                                                 rawVelocity,
                                                 static_cast<int>(mCurrentFrame.mnId),
                                                 static_cast<float>(GetInstanceStaticVelocityDecay()));
                selectedUncertainVelocity = true;
            }

            if(DebugInstanceResidualMotionGate() || DebugFocusFrame(mCurrentFrame.mnId))
            {
                std::cout << "[STSLAM_INSTANCE_GATE] frame=" << mCurrentFrame.mnId
                          << " instance_id=" << instanceId
                          << " semantic=" << observation.semanticId
                          << " valid=" << (motionGate.valid ? 1 : 0)
                          << " support=" << motionGate.support
                          << " static_mean_error=" << motionGate.staticMeanError
                          << " dynamic_mean_error=" << motionGate.dynamicMeanError
                          << " dynamic_ratio_threshold=" << GetInstanceDynamicResidualRatio()
                          << " state=" << InstanceMotionGateStateName(motionGate.state)
                          << " static_evidence=" << pInstance->GetStaticMotionEvidence()
                          << " dynamic_evidence=" << pInstance->GetDynamicMotionEvidence()
                          << " uncertain_evidence=" << pInstance->GetUncertainMotionEvidence()
                          << " static_confirm_frames=" << GetInstanceStaticMotionConfirmFrames()
                          << " dynamic_confirm_frames=" << GetInstanceDynamicMotionConfirmFrames()
                          << " static_velocity_decay=" << GetInstanceStaticVelocityDecay()
                          << " backend_motion_evidence=" << pInstance->GetBackendMotionEvidence()
                          << " latest_motion_state="
                          << (hasLatestMotionState ?
                              DynamicEntityMotionStateName(latestMotionState.state) : "none")
                          << " latest_motion_frame="
                          << (hasLatestMotionState ?
                              static_cast<long>(latestMotionState.frameId) : -1)
                          << " latest_motion_confidence="
                          << (hasLatestMotionState ? latestMotionState.confidence : 0.0)
                          << " use_dynamic=" << (selectedDynamicVelocity ? 1 : 0)
                          << " zero_velocity=" << (selectedZeroVelocity ? 1 : 0)
                          << " uncertain=" << (selectedUncertainVelocity ? 1 : 0)
                          << " zero_velocity_reactivated=" << (zeroVelocityReactivated ? 1 : 0)
                          << " zero_velocity_reactivation_reason=" << zeroVelocityReactivationReason
                          << " zero_velocity_reactivation_tracklets="
                          << zeroVelocityReactivationTracklets
                          << " zero_velocity_reactivation_triangulated="
                          << zeroVelocityReactivationTriangulated
                          << " zero_velocity_reactivation_candidate_rejected="
                          << zeroVelocityReactivationCandidateRejected
                          << " zero_velocity_reactivation_svd_reliable="
                          << (zeroVelocityReactivationSvdReliable ? 1 : 0)
                          << " zero_velocity_reactivation_svd_reason="
                          << zeroVelocityReactivationSvdReason
                          << " zero_velocity_reactivation_svd_condition="
                          << zeroVelocityReactivationSvdCondition
                          << " zero_velocity_reactivation_svd_planarity="
                          << zeroVelocityReactivationSvdPlanarity
                          << " zero_velocity_reactivation_translation_only_norm="
                          << zeroVelocityReactivationTranslationOnlyNorm
                          << " zero_velocity_reactivation_translation_only="
                          << (zeroVelocityReactivationUsedTranslationOnly ? 1 : 0)
                          << " selected_translation_norm=" << selectedVelocity.translation().norm()
                          << std::endl;
            }
        }

        bool propagatedUncertainDynamicEntity = false;
        bool stateChainPredictionAvailable = maturedPrediction && !selectedUncertainVelocity;
        if(mCurrentFrame.mpReferenceKF && maturedPrediction && !selectedUncertainVelocity)
        {
            if(selectedDynamicVelocity &&
               previousSeenFrame >= 0 &&
               previousSeenFrame + 1 == static_cast<int>(mCurrentFrame.mnId))
            {
                pInstance->PredictPose(selectedVelocity);
            }
            else if(hasLatestMotionState)
            {
                pInstance->SetInstanceMotionState(static_cast<unsigned long>(mCurrentFrame.mnId),
                                                  latestMotionState.pose,
                                                  selectedVelocity,
                                                  latestMotionState.state,
                                                  latestMotionState.confidence,
                                                  latestMotionState.reliable);
            }
            pInstance->UpdateMotionPrior(mCurrentFrame.mpReferenceKF, selectedVelocity);
            pInstance->UpdatePoseProxy(mCurrentFrame.mpReferenceKF, pInstance->GetLastPoseEstimate());
        }
        else if(mCurrentFrame.mpReferenceKF &&
                maturedPrediction &&
                selectedUncertainVelocity &&
                hasLatestMotionState &&
                PropagateUncertainDynamicEntityState() &&
                IsFiniteSE3(selectedVelocity))
        {
            const bool consecutiveObservation =
                previousSeenFrame >= 0 &&
                previousSeenFrame + 1 == static_cast<int>(mCurrentFrame.mnId);
            const Sophus::SE3f basePose =
                IsFiniteSE3(latestMotionState.pose) ?
                latestMotionState.pose : pInstance->GetLastPoseEstimate();
            const Sophus::SE3f propagatedPose =
                consecutiveObservation ? selectedVelocity * basePose : basePose;

            if(IsFiniteSE3(propagatedPose))
            {
                const double propagatedConfidence =
                    std::max(0.20, std::min(0.45, latestMotionState.confidence));
                pInstance->SetInstanceMotionState(static_cast<unsigned long>(mCurrentFrame.mnId),
                                                  propagatedPose,
                                                  selectedVelocity,
                                                  Instance::kUncertainDynamicEntity,
                                                  propagatedConfidence,
                                                  false);
                pInstance->UpdateMotionPrior(mCurrentFrame.mpReferenceKF, selectedVelocity);
                pInstance->UpdatePoseProxy(mCurrentFrame.mpReferenceKF, propagatedPose);
                propagatedUncertainDynamicEntity = true;
                stateChainPredictionAvailable = true;
            }
        }
        if(stateChainPredictionAvailable)
            mCurrentFrame.mmPredictedInstanceMotions[instanceId] = selectedVelocity;

        if((DebugInstanceResidualMotionGate() || DebugFocusFrame(mCurrentFrame.mnId)) &&
           maturedPrediction)
        {
            std::cout << "[STSLAM_INSTANCE_STATE_CHAIN] frame=" << mCurrentFrame.mnId
                      << " instance_id=" << instanceId
                      << " selected_uncertain=" << (selectedUncertainVelocity ? 1 : 0)
                      << " propagated_uncertain_dynamic_entity="
                      << (propagatedUncertainDynamicEntity ? 1 : 0)
                      << " state_chain_prediction_available="
                      << (stateChainPredictionAvailable ? 1 : 0)
                      << " propagate_uncertain_enabled="
                      << (PropagateUncertainDynamicEntityState() ? 1 : 0)
                      << " selected_translation_norm="
                      << selectedVelocity.translation().norm()
                      << std::endl;
        }

        const int suppliedDynamicObservations =
            SupplyDynamicObservationsForInstance(pInstance,
                                                 observation,
                                                 featureIndices,
                                                 stateChainPredictionAvailable,
                                                 selectedVelocity);
        (void)suppliedDynamicObservations;
    }
}

int Tracking::SplitRgbdDynamicFeatureMatches(const std::string& stage,
                                             bool appendDynamicObservations)
{
    if(!EnableRgbdDynamicFrontendSplit() ||
       (mSensor != System::RGBD && mSensor != System::IMU_RGBD) ||
       !mCurrentFrame.HasPanopticObservation())
    {
        return 0;
    }

    Map* pCurrentMap = mpAtlas ? mpAtlas->GetCurrentMap() : static_cast<Map*>(NULL);
    const bool canAppend =
        appendDynamicObservations && pCurrentMap && mCurrentFrame.HasPose();
    const bool appendSplitDynamicObservations =
        canAppend && EnableRgbdDynamicSplitObservationAppend();
    const unsigned long frameId = mCurrentFrame.mnId;

    int detectedInstanceFeatures = 0;
    int detachedStaticMatches = 0;
    int retainedStaticMatches = 0;
    int retainedNoMotionEvidence = 0;
    int appendedDynamicObservations = 0;
    int bindingFailures = 0;
    int latestMovingStateFeatures = 0;
    int latestZeroVelocityStateFeatures = 0;
    int latestUncertainStateFeatures = 0;
    int latestUnknownStateFeatures = 0;
    int backendMovingEvidenceFeatures = 0;
    int backendZeroEvidenceFeatures = 0;
    int backendUncertainEvidenceFeatures = 0;
    std::set<std::pair<int, MapPoint*> > sExistingDynamicObservations;
    for(size_t obsIdx = 0; obsIdx < mCurrentFrame.mvDynamicInstancePointObservations.size(); ++obsIdx)
    {
        const DynamicInstancePointObservation& observation =
            mCurrentFrame.mvDynamicInstancePointObservations[obsIdx];
        if(observation.instanceId > 0 && observation.pBackendPoint)
            sExistingDynamicObservations.insert(
                std::make_pair(observation.instanceId, observation.pBackendPoint));
    }

    const int nFeatures =
        std::min(mCurrentFrame.N, static_cast<int>(mCurrentFrame.mvpMapPoints.size()));
    for(int idx = 0; idx < nFeatures; ++idx)
    {
        const int instanceId = mCurrentFrame.GetFeatureInstanceId(static_cast<size_t>(idx));
        if(instanceId <= 0)
            continue;

        ++detectedInstanceFeatures;
        const int semanticLabel =
            std::max(0, mCurrentFrame.GetFeatureSemanticLabel(static_cast<size_t>(idx)));
        Instance* pKnownInstance =
            pCurrentMap ? pCurrentMap->GetInstance(instanceId) : static_cast<Instance*>(NULL);
        if(pKnownInstance)
        {
            if(pKnownInstance->GetBackendMovingMotionEvidence() >=
               GetRgbdDynamicSplitMinBackendMotionEvidence())
            {
                ++backendMovingEvidenceFeatures;
            }
            if(pKnownInstance->GetBackendZeroMotionEvidence() >=
               GetRgbdDynamicSplitMinBackendMotionEvidence())
            {
                ++backendZeroEvidenceFeatures;
            }
            if(pKnownInstance->GetBackendUncertainMotionEvidence() >=
               GetRgbdDynamicSplitMinBackendMotionEvidence())
            {
                ++backendUncertainEvidenceFeatures;
            }

            Instance::InstanceMotionStateRecord latestState;
            if(pKnownInstance->GetLatestInstanceMotionState(latestState))
            {
                if(latestState.state == Instance::kMovingDynamicEntity)
                    ++latestMovingStateFeatures;
                else if(latestState.state == Instance::kZeroVelocityDynamicEntity)
                    ++latestZeroVelocityStateFeatures;
                else if(latestState.state == Instance::kUncertainDynamicEntity)
                    ++latestUncertainStateFeatures;
                else
                    ++latestUnknownStateFeatures;
            }
            else
            {
                ++latestUnknownStateFeatures;
            }
        }
        else
        {
            ++latestUnknownStateFeatures;
        }
        const bool detachFromStatic =
            ShouldDetachRgbdInstanceFromStaticPath(pKnownInstance);
        if(!detachFromStatic)
        {
            ++retainedStaticMatches;
            if(!pKnownInstance || !pKnownInstance->IsInitialized())
                ++retainedNoMotionEvidence;
        }

        MapPoint* pMP = mCurrentFrame.mvpMapPoints[idx];
        if(pMP && appendSplitDynamicObservations && !pMP->isBad() &&
           pMP->GetMap() == pCurrentMap)
        {
            Instance* pInstance = BindMapPointToInstance(pMP, instanceId, semanticLabel, true);
            if(pInstance)
            {
                if(detachFromStatic)
                    pMP->SetLifecycleType(MapPoint::kDynamicInstanceObservationPoint);
                pMP->UpdateObservationStats(static_cast<int>(frameId));
                const std::pair<int, MapPoint*> observationKey(instanceId, pMP);
                if(sExistingDynamicObservations.count(observationKey) == 0)
                {
                    const Eigen::Vector3f pointWorld = pMP->GetWorldPos();
                    if(pointWorld.allFinite())
                    {
                        DynamicInstancePointObservation dynamicObservation;
                        dynamicObservation.instanceId = instanceId;
                        dynamicObservation.semanticLabel = semanticLabel;
                        dynamicObservation.featureIdx = idx;
                        dynamicObservation.pointWorld = pointWorld;
                        dynamicObservation.qualityWeight = 1.0;
                        dynamicObservation.pBackendPoint = pMP;
                        mCurrentFrame.mvDynamicInstancePointObservations.push_back(dynamicObservation);
                        pInstance->SetObservationQualityWeight(frameId, pMP, 1.0);
                        pInstance->AddDynamicObservation(frameId, idx, pMP, pointWorld, 1.0);
                        sExistingDynamicObservations.insert(observationKey);
                        ++appendedDynamicObservations;
                    }
                }
            }
            else
            {
                ++bindingFailures;
            }
        }

        if(detachFromStatic && pMP)
        {
            mCurrentFrame.mvpMapPoints[idx] = static_cast<MapPoint*>(NULL);
            ++detachedStaticMatches;
        }
        if(detachFromStatic && idx < static_cast<int>(mCurrentFrame.mvbOutlier.size()))
            mCurrentFrame.mvbOutlier[idx] = true;
    }

    if(detectedInstanceFeatures > 0)
    {
        std::cout << "[STSLAM_RGBD_DYNAMIC_SPLIT]"
                  << " frame=" << mCurrentFrame.mnId
                  << " stage=" << stage
                  << " detected_instance_features=" << detectedInstanceFeatures
                  << " detached_static_matches=" << detachedStaticMatches
                  << " retained_static_matches=" << retainedStaticMatches
                  << " retained_no_motion_evidence=" << retainedNoMotionEvidence
                  << " require_motion_evidence="
                  << (RequireMotionEvidenceForRgbdDynamicSplit() ? 1 : 0)
                  << " append_dynamic_observations="
                  << (EnableRgbdDynamicSplitObservationAppend() ? 1 : 0)
                  << " backend_moving_evidence_features="
                  << backendMovingEvidenceFeatures
                  << " backend_zero_evidence_features="
                  << backendZeroEvidenceFeatures
                  << " backend_uncertain_evidence_features="
                  << backendUncertainEvidenceFeatures
                  << " latest_moving_state_features="
                  << latestMovingStateFeatures
                  << " latest_zero_velocity_state_features="
                  << latestZeroVelocityStateFeatures
                  << " latest_uncertain_state_features="
                  << latestUncertainStateFeatures
                  << " latest_unknown_state_features="
                  << latestUnknownStateFeatures
                  << " appended_dynamic_observations=" << appendedDynamicObservations
                  << " binding_failures=" << bindingFailures
                  << std::endl;
    }

    return detachedStaticMatches;
}

int Tracking::SupplyRgbdDepthBackedDynamicObservations(const std::string& stage)
{
    if(!EnableRgbdDepthBackedDynamicObservations() ||
       (mSensor != System::RGBD && mSensor != System::IMU_RGBD) ||
       !mCurrentFrame.HasPanopticObservation() ||
       !mCurrentFrame.HasPose() ||
       !mpAtlas)
    {
        return 0;
    }

    Map* pCurrentMap = mpAtlas->GetCurrentMap();
    if(!pCurrentMap)
        return 0;

    const unsigned long frameId = mCurrentFrame.mnId;
    const int maxPerInstance = GetRgbdDepthBackedMaxPointsPerInstance();
    const float maxDepth = GetRgbdDepthBackedMaxDepth();
    const bool promoteAsStructure = PromoteRgbdDepthBackedObservationsToStructure();
    std::map<int, int> createdPerInstance;
    std::set<int> sUsedFeatures;
    for(size_t obsIdx = 0; obsIdx < mCurrentFrame.mvDynamicInstancePointObservations.size(); ++obsIdx)
    {
        const int featureIdx = mCurrentFrame.mvDynamicInstancePointObservations[obsIdx].featureIdx;
        if(featureIdx >= 0)
            sUsedFeatures.insert(featureIdx);
    }

    int considered = 0;
    int rejectedUsed = 0;
    int rejectedNoDepth = 0;
    int rejectedProjection = 0;
    int rejectedBinding = 0;
    int capped = 0;
    int created = 0;

    mCurrentFrame.UpdatePoseMatrices();
    const Sophus::SE3f Tcw = mCurrentFrame.GetPose();
    const Sophus::SE3f Twc = Tcw.inverse();
    const bool hasBackendDynamicDepth =
        !mRgbdBackendDynamicDepth.empty() &&
        mRgbdBackendDynamicDepth.rows == mCurrentFrame.imgLeft.rows &&
        mRgbdBackendDynamicDepth.cols == mCurrentFrame.imgLeft.cols;
    const int nFeatures =
        std::min(mCurrentFrame.N, static_cast<int>(mCurrentFrame.mvDepth.size()));
    for(int idx = 0; idx < nFeatures; ++idx)
    {
        const int instanceId = mCurrentFrame.GetFeatureInstanceId(static_cast<size_t>(idx));
        if(instanceId <= 0)
            continue;
        ++considered;

        if(sUsedFeatures.count(idx) > 0)
        {
            ++rejectedUsed;
            continue;
        }
        if(createdPerInstance[instanceId] >= maxPerInstance)
        {
            ++capped;
            continue;
        }

        float depth = mCurrentFrame.mvDepth[idx];
        if(hasBackendDynamicDepth && idx < static_cast<int>(mCurrentFrame.mvKeysUn.size()))
        {
            const int u = cvRound(mCurrentFrame.mvKeysUn[idx].pt.x);
            const int v = cvRound(mCurrentFrame.mvKeysUn[idx].pt.y);
            if(u >= 0 && u < mRgbdBackendDynamicDepth.cols &&
               v >= 0 && v < mRgbdBackendDynamicDepth.rows)
            {
                const float backendDepth = mRgbdBackendDynamicDepth.at<float>(v, u);
                if(backendDepth > 0.0f)
                    depth = backendDepth;
            }
        }
        if(!(depth > 0.0f) || depth > maxDepth)
        {
            ++rejectedNoDepth;
            continue;
        }

        Eigen::Vector3f pointWorld;
        if(hasBackendDynamicDepth && idx < static_cast<int>(mCurrentFrame.mvKeysUn.size()))
        {
            const Eigen::Vector3f ray =
                mCurrentFrame.mpCamera ?
                mCurrentFrame.mpCamera->unprojectEig(mCurrentFrame.mvKeysUn[idx].pt) :
                Eigen::Vector3f::Zero();
            const Eigen::Vector3f pointCamera = ray * depth;
            pointWorld = Twc * pointCamera;
        }
        else if(!mCurrentFrame.UnprojectStereo(idx, pointWorld))
        {
            pointWorld = Eigen::Vector3f::Zero();
        }
        if(!pointWorld.allFinite())
        {
            ++rejectedNoDepth;
            continue;
        }

        const Eigen::Vector3f pointCamera = Tcw * pointWorld;
        if(!pointCamera.allFinite() || pointCamera[2] <= 0.0f)
        {
            ++rejectedProjection;
            continue;
        }

        MapPoint* pBackendPoint = new MapPoint(pointWorld, pCurrentMap, &mCurrentFrame, idx);
        pBackendPoint->SetLifecycleType(promoteAsStructure ?
                                        MapPoint::kInstanceStructurePoint :
                                        MapPoint::kDynamicInstanceObservationPoint);
        const int semanticLabel =
            std::max(0, mCurrentFrame.GetFeatureSemanticLabel(static_cast<size_t>(idx)));
        pBackendPoint->SetInstanceId(instanceId);
        pBackendPoint->SetSemanticLabel(semanticLabel);
        pBackendPoint->UpdateObservationStats(static_cast<int>(frameId));

        Instance* pInstance =
            BindMapPointToInstance(pBackendPoint, instanceId, semanticLabel, true);
        if(!pInstance)
        {
            ++rejectedBinding;
            delete pBackendPoint;
            continue;
        }

        if(promoteAsStructure)
        {
            const Sophus::SE3f instancePose = pInstance->GetLastPoseEstimate();
            const Eigen::Vector3f localPoint =
                IsFiniteSE3(instancePose) ? instancePose.inverse() * pointWorld : pointWorld;
            pInstance->SetStructureLocalPoint(pBackendPoint, localPoint);
        }

        DynamicInstancePointObservation dynamicObservation;
        dynamicObservation.instanceId = instanceId;
        dynamicObservation.semanticLabel = semanticLabel;
        dynamicObservation.featureIdx = idx;
        dynamicObservation.pointWorld = pointWorld;
        dynamicObservation.qualityWeight = 1.0;
        dynamicObservation.pBackendPoint = pBackendPoint;
        mCurrentFrame.mvDynamicInstancePointObservations.push_back(dynamicObservation);
        pInstance->SetObservationQualityWeight(frameId, pBackendPoint, 1.0);
        pInstance->AddDynamicObservation(frameId, idx, pBackendPoint, pointWorld, 1.0);
        sUsedFeatures.insert(idx);
        ++createdPerInstance[instanceId];
        ++created;
    }

    if(considered > 0)
    {
        std::cout << "[STSLAM_RGBD_DEPTH_BACKED_OBS]"
                  << " frame=" << mCurrentFrame.mnId
                  << " stage=" << stage
                  << " considered_instance_features=" << considered
                  << " created=" << created
                  << " promote_as_structure=" << (promoteAsStructure ? 1 : 0)
                  << " rejected_used=" << rejectedUsed
                  << " rejected_no_depth=" << rejectedNoDepth
                  << " rejected_projection=" << rejectedProjection
                  << " rejected_binding=" << rejectedBinding
                  << " capped=" << capped
                  << " max_per_instance=" << maxPerInstance
                  << " max_depth=" << maxDepth
                  << std::endl;
    }

    return created;
}

int Tracking::SupplyDynamicObservationsForInstance(Instance* pInstance,
                                                   const InstanceObservation& observation,
                                                   const std::vector<int>& featureIndices,
                                                   bool usePredictedMotion,
                                                   const Sophus::SE3f& predictedMotion)
{
    if(!pInstance || !pInstance->IsInitialized() || featureIndices.empty() || !mpAtlas)
        return 0;

    Map* pCurrentMap = mpAtlas->GetCurrentMap();
    if(!pCurrentMap || !mCurrentFrame.HasPose())
        return 0;

    const int instanceId = pInstance->GetId();
    const int semanticLabel =
        observation.semanticId > 0 ? observation.semanticId : pInstance->GetSemanticLabel();
    const unsigned long frameId = mCurrentFrame.mnId;
    const bool enableDescriptorMatching = EnableDynamicSupplyDescriptorMatching();
    const bool enableProjectionMatching = EnableDynamicSupplyProjectionMatching();
    const double descriptorThreshold =
        static_cast<double>(GetDynamicSupplyDescriptorThreshold());
    const double descriptorRatio = GetDynamicSupplyDescriptorRatio();
    const double projectionMaxError = GetDynamicSupplyProjectionMaxError();
    const double maxReprojectionError = GetDynamicSupplyMaxReprojectionError();
    const bool requireProjectionGateSupport =
        RequireDynamicSupplyProjectionGateSupport();

    std::set<int> sUsedFeatures;
    std::set<MapPoint*> sUsedBackendPoints;
    for(size_t i = 0; i < mCurrentFrame.mvDynamicInstancePointObservations.size(); ++i)
    {
        const DynamicInstancePointObservation& existing =
            mCurrentFrame.mvDynamicInstancePointObservations[i];
        if(existing.instanceId != instanceId)
            continue;
        if(existing.featureIdx >= 0)
            sUsedFeatures.insert(existing.featureIdx);
        if(existing.pBackendPoint)
            sUsedBackendPoints.insert(existing.pBackendPoint);
    }

    int directStructureAppends = 0;
    int projectionStructureAppends = 0;
    int descriptorStructureAppends = 0;
    int rejectedFeatureRange = 0;
    int rejectedOutlier = 0;
    int rejectedNullPoint = 0;
    int rejectedBadPoint = 0;
    int rejectedBinding = 0;
    int rejectedNoDescriptorCandidate = 0;
    int rejectedDescriptor = 0;
    int rejectedProjection = 0;
    int rejectedProjectionNoFeature = 0;
    int rejectedDuplicate = 0;
    int rejectedInstanceGate = 0;
    int poseProxyStructurePredictions = 0;
    int velocityPredictions = 0;
    int identityStructurePredictions = 0;

    Eigen::Vector3f instanceStructureCentroid = Eigen::Vector3f::Zero();
    int instanceStructureCentroidSupport = 0;
    if(UseStrictPaperArchitectureDefaults())
    {
        const std::set<MapPoint*> sInstancePoints = pInstance->GetMapPoints();
        for(std::set<MapPoint*>::const_iterator it = sInstancePoints.begin();
            it != sInstancePoints.end(); ++it)
        {
            MapPoint* pMP = *it;
            if(!pMP || pMP->isBad() || pMP->GetMap() != pCurrentMap ||
               pMP->IsDynamicInstanceObservationPoint())
            {
                continue;
            }

            const Eigen::Vector3f pointWorld = pMP->GetWorldPos();
            if(!pointWorld.allFinite())
                continue;

            instanceStructureCentroid += pointWorld;
            ++instanceStructureCentroidSupport;
        }

        if(instanceStructureCentroidSupport > 0)
            instanceStructureCentroid /= static_cast<float>(instanceStructureCentroidSupport);
    }

    const Sophus::SE3f instancePoseProxy = pInstance->GetLastPoseEstimate();
    const bool useInstancePoseStructurePrediction =
        UseStrictPaperArchitectureDefaults() &&
        instanceStructureCentroidSupport >= 3 &&
        instanceStructureCentroid.allFinite() &&
        IsFiniteSE3(instancePoseProxy);

    auto predictedWorldPoint = [&](MapPoint* pBackendPoint,
                                   const Eigen::Vector3f& baseWorld) -> Eigen::Vector3f
    {
        if(useInstancePoseStructurePrediction &&
           pBackendPoint &&
           pBackendPoint->IsInstanceStructurePoint())
        {
            Eigen::Vector3f localPoint;
            if(!pInstance->GetStructureLocalPoint(pBackendPoint, localPoint))
                localPoint = baseWorld - instanceStructureCentroid;
            const Eigen::Vector3f movedWorld = instancePoseProxy * localPoint;
            if(movedWorld.allFinite())
            {
                ++poseProxyStructurePredictions;
                return movedWorld;
            }
        }

        if(usePredictedMotion && IsFiniteSE3(predictedMotion))
        {
            const Eigen::Vector3f movedWorld = predictedMotion * baseWorld;
            if(movedWorld.allFinite())
            {
                ++velocityPredictions;
                return movedWorld;
            }
        }

        ++identityStructurePredictions;
        return baseWorld;
    };

    const Sophus::SE3f Tcw = mCurrentFrame.GetPose();
    const bool enableInstanceGate = EnableDynamicSupplyInstanceQualityGate();
    const int gateMinSupport = GetDynamicSupplyGateMinSupport();
    const int gateDescriptorMinSupport = GetDynamicSupplyGateDescriptorMinSupport();
    const double gateInlierThreshold = GetDynamicSupplyGateInlierThreshold();
    const double gateMaxMeanError = GetDynamicSupplyGateMaxMeanError();
    const double gateMinInlierRatio = GetDynamicSupplyGateMinInlierRatio();
    int gateTrackedPoints = 0;
    int gateInlierPoints = 0;
    double gateErrorSum = 0.0;
    double gateMaxError = 0.0;
    for(size_t i = 0; i < featureIndices.size(); ++i)
    {
        const int featureIdx = featureIndices[i];
        if(featureIdx < 0 ||
           featureIdx >= static_cast<int>(mCurrentFrame.mvpMapPoints.size()) ||
           featureIdx >= static_cast<int>(mCurrentFrame.mvKeysUn.size()))
        {
            continue;
        }
        if(featureIdx < static_cast<int>(mCurrentFrame.mvbOutlier.size()) &&
           mCurrentFrame.mvbOutlier[featureIdx])
        {
            continue;
        }

        MapPoint* pMP = mCurrentFrame.mvpMapPoints[featureIdx];
        if(!pMP || pMP->isBad() || pMP->GetMap() != pCurrentMap || !mCurrentFrame.mpCamera)
            continue;

        const Eigen::Vector3f pointWorld = predictedWorldPoint(pMP, pMP->GetWorldPos());
        const Eigen::Vector3f pointCamera = Tcw * pointWorld;
        if(!pointCamera.allFinite() || pointCamera[2] <= 0.0f)
            continue;

        const Eigen::Vector3d pointCameraD = pointCamera.cast<double>();
        const Eigen::Vector2d projection =
            mCurrentFrame.mpCamera->project(pointCameraD);
        const Eigen::Vector2d observationPoint(mCurrentFrame.mvKeysUn[featureIdx].pt.x,
                                               mCurrentFrame.mvKeysUn[featureIdx].pt.y);
        const double reprojectionError = (observationPoint - projection).norm();
        if(!std::isfinite(reprojectionError))
            continue;

        ++gateTrackedPoints;
        gateErrorSum += reprojectionError;
        gateMaxError = std::max(gateMaxError, reprojectionError);
        if(reprojectionError <= gateInlierThreshold)
            ++gateInlierPoints;
    }

    const double gateMeanError =
        gateTrackedPoints > 0 ? gateErrorSum / static_cast<double>(gateTrackedPoints) : 0.0;
    const double gateInlierRatio =
        gateTrackedPoints > 0 ?
        static_cast<double>(gateInlierPoints) / static_cast<double>(gateTrackedPoints) : 0.0;
    const double gateTrackedRatio =
        featureIndices.empty() ? 0.0 :
        static_cast<double>(gateTrackedPoints) / static_cast<double>(featureIndices.size());
    const bool gateHasEnoughSupport = gateTrackedPoints >= gateMinSupport;
    const bool gateHasAnyProjectionEvidence = gateTrackedPoints > 0;
    const bool gateProjectionPassed =
        !enableInstanceGate ||
        !gateHasAnyProjectionEvidence ||
        (gateMeanError <= gateMaxMeanError && gateInlierRatio >= gateMinInlierRatio);
    const int priorDynamicSupplyGateFailures =
        enableInstanceGate ? pInstance->GetDynamicSupplyGateFailureEvidence() : 0;
    const bool blockProjectionByPriorFailure =
        enableInstanceGate &&
        !gateHasAnyProjectionEvidence &&
        priorDynamicSupplyGateFailures >= GetDynamicSupplyPriorFailureBlockMinEvidence();
    const bool gatePassed =
        !enableInstanceGate ||
        !gateHasEnoughSupport ||
        (gateMeanError <= gateMaxMeanError && gateInlierRatio >= gateMinInlierRatio);
    const bool allowDirectSupply = gatePassed;
    const bool projectionGateSupportOk =
        !enableInstanceGate ||
        !requireProjectionGateSupport ||
        gateHasEnoughSupport;
    const bool allowProjectionSupply =
        enableProjectionMatching &&
        projectionGateSupportOk &&
        gateProjectionPassed &&
        !blockProjectionByPriorFailure;
    const bool allowDescriptorSupply =
        enableDescriptorMatching &&
        (!enableInstanceGate ||
         (gateTrackedPoints >= gateDescriptorMinSupport &&
          gateMeanError <= gateMaxMeanError &&
          gateInlierRatio >= gateMinInlierRatio));
    const double gateQualityWeight =
        !enableInstanceGate ? 1.0 :
        (!gateHasEnoughSupport ? GetDynamicSupplySparseDirectWeight() :
         std::max(0.05,
                  std::min(1.0,
                           0.5 * gateInlierRatio +
                           0.5 * std::max(0.0, 1.0 - gateMeanError / std::max(1.0, gateInlierThreshold)))));
    if(enableInstanceGate && gateHasAnyProjectionEvidence)
        pInstance->RecordDynamicSupplyGateResult(gateProjectionPassed,
                                                 true,
                                                 static_cast<int>(frameId));

    auto appendDynamicObservation = [&](const int featureIdx,
                                        MapPoint* pBackendPoint,
                                        const Eigen::Vector3f& pointWorld,
                                        const double qualityWeight) -> bool
    {
        if(!pBackendPoint || featureIdx < 0 ||
           featureIdx >= static_cast<int>(mCurrentFrame.mvKeysUn.size()) ||
           !pointWorld.allFinite())
        {
            return false;
        }
        if(sUsedFeatures.count(featureIdx) > 0 || sUsedBackendPoints.count(pBackendPoint) > 0)
        {
            ++rejectedDuplicate;
            return false;
        }
        if(UseStrictPaperArchitectureDefaults())
        {
            if(!mCurrentFrame.mpCamera)
            {
                ++rejectedProjection;
                return false;
            }
            const Eigen::Vector3f pointCamera = Tcw * pointWorld;
            if(!pointCamera.allFinite() || pointCamera[2] <= 0.0f)
            {
                ++rejectedProjection;
                return false;
            }
            const Eigen::Vector3d pointCameraD = pointCamera.cast<double>();
            const Eigen::Vector2d projection =
                mCurrentFrame.mpCamera->project(pointCameraD);
            if(!projection.allFinite())
            {
                ++rejectedProjection;
                return false;
            }
            const Eigen::Vector2d observed(mCurrentFrame.mvKeysUn[featureIdx].pt.x,
                                           mCurrentFrame.mvKeysUn[featureIdx].pt.y);
            const double reprojectionError = (observed - projection).norm();
            if(!std::isfinite(reprojectionError) ||
               reprojectionError > maxReprojectionError)
            {
                ++rejectedProjection;
                return false;
            }
        }

        const double clampedQuality =
            std::max(0.05, std::min(1.0, std::isfinite(qualityWeight) ? qualityWeight : 1.0));

        DynamicInstancePointObservation dynamicObservation;
        dynamicObservation.instanceId = instanceId;
        dynamicObservation.semanticLabel = semanticLabel;
        dynamicObservation.featureIdx = featureIdx;
        dynamicObservation.pointWorld = pointWorld;
        dynamicObservation.qualityWeight = clampedQuality;
        dynamicObservation.pBackendPoint = pBackendPoint;
        mCurrentFrame.mvDynamicInstancePointObservations.push_back(dynamicObservation);

        pInstance->SetObservationQualityWeight(frameId, pBackendPoint, clampedQuality);
        pInstance->AddDynamicObservation(frameId,
                                         featureIdx,
                                         pBackendPoint,
                                         pointWorld,
                                         clampedQuality);
        sUsedFeatures.insert(featureIdx);
        sUsedBackendPoints.insert(pBackendPoint);
        return true;
    };

    if(allowDirectSupply)
    {
        for(size_t i = 0; i < featureIndices.size(); ++i)
        {
            const int featureIdx = featureIndices[i];
            if(featureIdx < 0 || featureIdx >= static_cast<int>(mCurrentFrame.mvpMapPoints.size()))
            {
                ++rejectedFeatureRange;
                continue;
            }
            if(featureIdx < static_cast<int>(mCurrentFrame.mvbOutlier.size()) &&
               mCurrentFrame.mvbOutlier[featureIdx])
            {
                ++rejectedOutlier;
                continue;
            }

            MapPoint* pMP = mCurrentFrame.mvpMapPoints[featureIdx];
            if(!pMP)
            {
                ++rejectedNullPoint;
                continue;
            }
            if(pMP->isBad() || pMP->GetMap() != pCurrentMap)
            {
                ++rejectedBadPoint;
                continue;
            }

            if(!pMP->IsDynamicInstanceObservationPoint())
            {
                Instance* pBoundOrAnnotated =
                    (mSensor == System::RGBD || mSensor == System::IMU_RGBD) ?
                    AnnotateMapPointWithInstance(pMP, instanceId, semanticLabel, false) :
                    BindMapPointToInstance(pMP, instanceId, semanticLabel, false);
                if(!pBoundOrAnnotated)
                {
                    ++rejectedBinding;
                    continue;
                }
                if(mSensor != System::RGBD && mSensor != System::IMU_RGBD)
                    pMP->SetLifecycleType(MapPoint::kInstanceStructurePoint);
                if(useInstancePoseStructurePrediction && pMP->IsInstanceStructurePoint())
                    pInstance->SetStructureLocalPoint(pMP,
                                                      pMP->GetWorldPos() -
                                                          instanceStructureCentroid);
            }

            if(useInstancePoseStructurePrediction && pMP->IsInstanceStructurePoint())
            {
                Eigen::Vector3f existingLocalPoint;
                if(!pInstance->GetStructureLocalPoint(pMP, existingLocalPoint))
                    pInstance->SetStructureLocalPoint(pMP,
                                                      pMP->GetWorldPos() -
                                                          instanceStructureCentroid);
            }

            pMP->UpdateObservationStats(static_cast<int>(frameId));
            const Eigen::Vector3f pointWorld = predictedWorldPoint(pMP, pMP->GetWorldPos());
            if(appendDynamicObservation(featureIdx, pMP, pointWorld, gateQualityWeight))
                ++directStructureAppends;
        }
    }
    else
    {
        rejectedInstanceGate = static_cast<int>(featureIndices.size());
    }

    std::vector<MapPoint*> vStructurePoints;
    const std::set<MapPoint*> sStructurePoints = pInstance->GetMapPoints();
    vStructurePoints.reserve(sStructurePoints.size());
    for(std::set<MapPoint*>::const_iterator it = sStructurePoints.begin();
        it != sStructurePoints.end(); ++it)
    {
        MapPoint* pMP = *it;
        if(!pMP || pMP->isBad() || pMP->GetMap() != pCurrentMap ||
           pMP->IsDynamicInstanceObservationPoint())
        {
            continue;
        }
        const cv::Mat descriptor = pMP->GetDescriptor();
        if(descriptor.empty())
            continue;
        vStructurePoints.push_back(pMP);
    }

    if(allowProjectionSupply && !vStructurePoints.empty() && mCurrentFrame.mpCamera)
    {
        for(size_t pointIdx = 0; pointIdx < vStructurePoints.size(); ++pointIdx)
        {
            MapPoint* pStructurePoint = vStructurePoints[pointIdx];
            if(!pStructurePoint || sUsedBackendPoints.count(pStructurePoint) > 0)
                continue;

            const Eigen::Vector3f pointWorld =
                predictedWorldPoint(pStructurePoint, pStructurePoint->GetWorldPos());
            const Eigen::Vector3f pointCamera = Tcw * pointWorld;
            if(!pointCamera.allFinite() || pointCamera[2] <= 0.0f)
            {
                ++rejectedProjection;
                continue;
            }

            const Eigen::Vector3d pointCameraD = pointCamera.cast<double>();
            const Eigen::Vector2d projection =
                mCurrentFrame.mpCamera->project(pointCameraD);
            if(!projection.allFinite())
            {
                ++rejectedProjection;
                continue;
            }

            int bestFeatureIdx = -1;
            double bestError = std::numeric_limits<double>::infinity();
            for(size_t featurePos = 0; featurePos < featureIndices.size(); ++featurePos)
            {
                const int featureIdx = featureIndices[featurePos];
                if(featureIdx < 0 ||
                   featureIdx >= static_cast<int>(mCurrentFrame.mvKeysUn.size()) ||
                   featureIdx >= static_cast<int>(mCurrentFrame.mvpMapPoints.size()) ||
                   sUsedFeatures.count(featureIdx) > 0)
                {
                    continue;
                }
                if(featureIdx < static_cast<int>(mCurrentFrame.mvbOutlier.size()) &&
                   mCurrentFrame.mvbOutlier[featureIdx])
                {
                    continue;
                }

                MapPoint* pExistingPoint = mCurrentFrame.mvpMapPoints[featureIdx];
                if(pExistingPoint && pExistingPoint != pStructurePoint)
                    continue;

                const Eigen::Vector2d observed(mCurrentFrame.mvKeysUn[featureIdx].pt.x,
                                               mCurrentFrame.mvKeysUn[featureIdx].pt.y);
                const double error = (observed - projection).norm();
                if(std::isfinite(error) && error < bestError)
                {
                    bestError = error;
                    bestFeatureIdx = featureIdx;
                }
            }

            if(bestFeatureIdx < 0)
            {
                ++rejectedProjectionNoFeature;
                continue;
            }
            if(bestError > projectionMaxError)
            {
                ++rejectedProjection;
                continue;
            }

            const double projectionQuality =
                std::max(0.05, 1.0 - bestError / std::max(1.0, projectionMaxError));
            const double qualityWeight = std::min(gateQualityWeight, projectionQuality);
            if(appendDynamicObservation(bestFeatureIdx, pStructurePoint, pointWorld, qualityWeight))
                ++projectionStructureAppends;
        }
    }

    if(allowDescriptorSupply)
    {
        for(size_t i = 0; i < featureIndices.size(); ++i)
        {
            const int featureIdx = featureIndices[i];
            if(featureIdx < 0 ||
               featureIdx >= static_cast<int>(mCurrentFrame.mvKeysUn.size()) ||
               featureIdx >= static_cast<int>(mCurrentFrame.mvpMapPoints.size()))
            {
                continue;
            }
            if(sUsedFeatures.count(featureIdx) > 0)
                continue;
            if(featureIdx < static_cast<int>(mCurrentFrame.mvbOutlier.size()) &&
               mCurrentFrame.mvbOutlier[featureIdx])
            {
                continue;
            }
            if(mCurrentFrame.mvpMapPoints[featureIdx])
                continue;
            if(vStructurePoints.empty() || mCurrentFrame.mDescriptors.empty() ||
               featureIdx >= mCurrentFrame.mDescriptors.rows)
            {
                ++rejectedNoDescriptorCandidate;
                continue;
            }

            const cv::Mat queryDescriptor = mCurrentFrame.mDescriptors.row(featureIdx);
            MapPoint* pBestPoint = static_cast<MapPoint*>(NULL);
            int bestDistance = std::numeric_limits<int>::max();
            int secondDistance = std::numeric_limits<int>::max();
            for(size_t candidateIdx = 0; candidateIdx < vStructurePoints.size(); ++candidateIdx)
            {
                MapPoint* pCandidate = vStructurePoints[candidateIdx];
                if(sUsedBackendPoints.count(pCandidate) > 0)
                    continue;

                const cv::Mat candidateDescriptor = pCandidate->GetDescriptor();
                if(candidateDescriptor.empty() ||
                   candidateDescriptor.cols != queryDescriptor.cols ||
                   candidateDescriptor.type() != queryDescriptor.type())
                {
                    continue;
                }

                const int distance =
                    ORBmatcher::DescriptorDistance(candidateDescriptor, queryDescriptor);
                if(distance < bestDistance)
                {
                    secondDistance = bestDistance;
                    bestDistance = distance;
                    pBestPoint = pCandidate;
                }
                else if(distance < secondDistance)
                {
                    secondDistance = distance;
                }
            }

            const bool descriptorAccepted =
                pBestPoint &&
                bestDistance <= static_cast<int>(descriptorThreshold) &&
                (secondDistance == std::numeric_limits<int>::max() ||
                 static_cast<double>(bestDistance) <= descriptorRatio * static_cast<double>(secondDistance));
            if(!descriptorAccepted)
            {
                ++rejectedDescriptor;
                continue;
            }

            const Eigen::Vector3f pointWorld =
                predictedWorldPoint(pBestPoint, pBestPoint->GetWorldPos());
            const Eigen::Vector3f pointCamera = Tcw * pointWorld;
            if(!pointCamera.allFinite() || pointCamera[2] <= 0.0f || !mCurrentFrame.mpCamera)
            {
                ++rejectedProjection;
                continue;
            }

            const Eigen::Vector3d pointCameraD = pointCamera.cast<double>();
            const Eigen::Vector2d projection =
                mCurrentFrame.mpCamera->project(pointCameraD);
            const Eigen::Vector2d observationPoint(mCurrentFrame.mvKeysUn[featureIdx].pt.x,
                                                   mCurrentFrame.mvKeysUn[featureIdx].pt.y);
            const double reprojectionError = (observationPoint - projection).norm();
            if(!std::isfinite(reprojectionError) || reprojectionError > maxReprojectionError)
            {
                ++rejectedProjection;
                continue;
            }

            const double descriptorQuality =
                std::max(0.05, 1.0 - static_cast<double>(bestDistance) / descriptorThreshold);
            const double projectionQuality =
                std::max(0.05, 1.0 - reprojectionError / std::max(1.0, maxReprojectionError));
            const double qualityWeight = std::min(1.0, descriptorQuality * projectionQuality);
            if(appendDynamicObservation(featureIdx, pBestPoint, pointWorld, qualityWeight))
                ++descriptorStructureAppends;
        }
    }

    const int appended = directStructureAppends + projectionStructureAppends + descriptorStructureAppends;
    if(appended > 0 && usePredictedMotion)
        mCurrentFrame.mmPredictedInstanceMotions[instanceId] = predictedMotion;

    if(DebugDynamicObservationSupply() || DebugFocusFrame(mCurrentFrame.mnId))
    {
        std::cout << "[STSLAM_DYNAMIC_SUPPLY] frame=" << frameId
                  << " instance_id=" << instanceId
                  << " semantic=" << semanticLabel
                  << " candidates=" << static_cast<int>(featureIndices.size())
                  << " structure_points=" << static_cast<int>(vStructurePoints.size())
                  << " appended=" << appended
                  << " direct_structure=" << directStructureAppends
                  << " projection_structure=" << projectionStructureAppends
                  << " descriptor_structure=" << descriptorStructureAppends
                  << " rejected_range=" << rejectedFeatureRange
                  << " rejected_outlier=" << rejectedOutlier
                  << " rejected_null_point=" << rejectedNullPoint
                  << " rejected_bad_point=" << rejectedBadPoint
                  << " rejected_binding=" << rejectedBinding
                  << " rejected_no_descriptor_candidate=" << rejectedNoDescriptorCandidate
                  << " rejected_descriptor=" << rejectedDescriptor
                  << " rejected_projection=" << rejectedProjection
                  << " rejected_projection_no_feature=" << rejectedProjectionNoFeature
                  << " rejected_duplicate=" << rejectedDuplicate
                  << " rejected_instance_gate=" << rejectedInstanceGate
                  << " gate_enabled=" << (enableInstanceGate ? 1 : 0)
                  << " gate_tracked_points=" << gateTrackedPoints
                  << " gate_tracked_ratio=" << gateTrackedRatio
                  << " gate_mean_error=" << gateMeanError
                  << " gate_max_error=" << gateMaxError
	                  << " gate_inlier_ratio=" << gateInlierRatio
	                  << " gate_passed=" << (gatePassed ? 1 : 0)
	                  << " gate_projection_passed=" << (gateProjectionPassed ? 1 : 0)
	                  << " gate_prior_failures=" << priorDynamicSupplyGateFailures
	                  << " gate_prior_block_projection="
	                  << (blockProjectionByPriorFailure ? 1 : 0)
	                  << " allow_direct=" << (allowDirectSupply ? 1 : 0)
                  << " allow_projection=" << (allowProjectionSupply ? 1 : 0)
                  << " allow_descriptor=" << (allowDescriptorSupply ? 1 : 0)
                  << " gate_quality_weight=" << gateQualityWeight
                  << " use_predicted_motion=" << (usePredictedMotion ? 1 : 0)
                  << " projection_matching=" << (enableProjectionMatching ? 1 : 0)
                  << " projection_require_gate_support=" << (requireProjectionGateSupport ? 1 : 0)
                  << " descriptor_matching=" << (enableDescriptorMatching ? 1 : 0)
                  << " instance_pose_structure_prediction="
                  << (useInstancePoseStructurePrediction ? 1 : 0)
                  << " instance_structure_centroid_support="
                  << instanceStructureCentroidSupport
                  << " pose_proxy_structure_predictions="
                  << poseProxyStructurePredictions
                  << " velocity_predictions=" << velocityPredictions
                  << " identity_structure_predictions="
                  << identityStructurePredictions
                  << std::endl;
    }

    return appended;
}

bool Tracking::InitializeInstance(Instance* pInstance, const std::vector<int>& vCurrIndices)
{
    if(!pInstance || mRecentPanopticFrames.size() < 2)
        return false;

    const int instanceId = pInstance->GetId();
    const bool debugInitForInstance = DebugInstanceInitializationFor(instanceId);
    const bool debugLifecycleForInstance = DebugInstanceLifecycleFor(instanceId);
    long currentMapId = -1;
    long currentMapOriginFrame = -1;
    long currentMapOriginKF = -1;
    int currentMapAgeFrames = -1;
    if(mpAtlas)
    {
        Map* pCurrentMap = mpAtlas->GetCurrentMap();
        if(pCurrentMap)
        {
            currentMapId = static_cast<long>(pCurrentMap->GetId());
            KeyFrame* pOriginKF = pCurrentMap->GetOriginKF();
            if(pOriginKF)
            {
                currentMapOriginFrame = static_cast<long>(pOriginKF->mnFrameId);
                currentMapOriginKF = static_cast<long>(pOriginKF->mnId);
                currentMapAgeFrames =
                    std::max(0, static_cast<int>(static_cast<long>(mCurrentFrame.mnId) -
                                                 currentMapOriginFrame));
            }
        }
    }

    const Frame& frameTau2 = mRecentPanopticFrames[mRecentPanopticFrames.size() - 2];
    const Frame& frameTau1 = mRecentPanopticFrames.back();
    const std::vector<int> featuresTau2 = CollectFeatureIndicesForInstance(frameTau2, instanceId);
    const std::vector<int> featuresTau1 = CollectFeatureIndicesForInstance(frameTau1, instanceId);

    if(static_cast<int>(featuresTau2.size()) < mnInstanceInitializationMinFeatures ||
       static_cast<int>(featuresTau1.size()) < mnInstanceInitializationMinFeatures ||
       static_cast<int>(vCurrIndices.size()) < mnInstanceInitializationMinFeatures)
    {
        if(debugInitForInstance || debugLifecycleForInstance || DebugFocusFrame(mCurrentFrame.mnId))
        {
            std::cout << "[STSLAM_INSTANCE_INIT] frame=" << mCurrentFrame.mnId
                      << " instance_id=" << instanceId
                      << " rejected=low_three_frame_features"
                      << " map_id=" << currentMapId
                      << " map_origin_frame=" << currentMapOriginFrame
                      << " map_origin_kf=" << currentMapOriginKF
                      << " map_age_frames=" << currentMapAgeFrames
                      << " tau2_frame=" << frameTau2.mnId
                      << " tau1_frame=" << frameTau1.mnId
                      << " tau_frame=" << mCurrentFrame.mnId
                      << " features_tau2=" << featuresTau2.size()
                      << " features_tau1=" << featuresTau1.size()
                      << " features_tau=" << vCurrIndices.size()
                      << " min_features=" << mnInstanceInitializationMinFeatures
                      << std::endl;
        }
        return false;
    }

    const std::vector<std::array<int, 3>> tracklets =
        CollectInstanceTracklets(frameTau2, frameTau1, mCurrentFrame, instanceId);
    const InstanceTrackletStabilityDiagnostics stability =
        EvaluateInstanceTrackletStability(frameTau2,
                                          frameTau1,
                                          mCurrentFrame,
                                          instanceId,
                                          featuresTau2,
                                          featuresTau1,
                                          vCurrIndices,
                                          tracklets);
    const CanonicalAssociationDiagnostics canonicalStability =
        EvaluateCanonicalAssociationStability(frameTau2,
                                              frameTau1,
                                              mCurrentFrame,
                                              instanceId);

    const bool rejectCanonicalStability =
        !canonicalStability.valid || !canonicalStability.stable;
    const bool rejectTriFrameGeometry =
        !stability.valid || stability.hardReject;
    if(EnableInstanceIdStabilityGate() &&
       (rejectCanonicalStability || rejectTriFrameGeometry))
    {
        if(debugInitForInstance || debugLifecycleForInstance ||
           DebugInstanceTracklets() || DebugFocusFrame(mCurrentFrame.mnId))
        {
            const std::string canonicalReason =
                canonicalStability.rejectReason.empty() ? "none" : canonicalStability.rejectReason;
            const std::string geometricReason =
                stability.rejectReason.empty() ? "none" : stability.rejectReason;
            std::cout << "[STSLAM_INSTANCE_INIT] frame=" << mCurrentFrame.mnId
                      << " instance_id=" << instanceId
                      << " rejected="
                      << (rejectCanonicalStability ?
                          "unstable_canonical_vps_refinement" :
                          "unstable_triframe_geometry")
                      << " map_id=" << currentMapId
                      << " map_origin_frame=" << currentMapOriginFrame
                      << " map_origin_kf=" << currentMapOriginKF
                      << " map_age_frames=" << currentMapAgeFrames
                      << " tau2_frame=" << frameTau2.mnId
                      << " tau1_frame=" << frameTau1.mnId
                      << " tau_frame=" << mCurrentFrame.mnId
                      << " reason=" << canonicalReason
                      << " geometry_reason=" << geometricReason
                      << " hard_reject_reason=" << (stability.hardRejectReason.empty() ? "unknown" : stability.hardRejectReason)
                      << " canonical_valid=" << (canonicalStability.valid ? 1 : 0)
                      << " canonical_stable=" << (canonicalStability.stable ? 1 : 0)
                      << " canonical_raw_tau2=" << canonicalStability.rawTau2
                      << " canonical_raw_tau1=" << canonicalStability.rawTau1
                      << " canonical_raw_tau=" << canonicalStability.rawTau
                      << " canonical_semantic_tau2=" << canonicalStability.semanticTau2
                      << " canonical_semantic_tau1=" << canonicalStability.semanticTau1
                      << " canonical_semantic_tau=" << canonicalStability.semanticTau
                      << " canonical_matched_frames=" << canonicalStability.matchedFrames
                      << " canonical_frame_corrections=" << canonicalStability.frameCorrections
                      << " canonical_permanent_corrections=" << canonicalStability.permanentCorrections
                      << " canonical_transient_corrections=" << canonicalStability.transientCorrections
                      << " canonical_ambiguous_frames=" << canonicalStability.ambiguousFrames
                      << " canonical_missing_frames=" << canonicalStability.missingFrames
                      << " canonical_min_streak=" << canonicalStability.minCorrectionStreak
                      << " hard_reject=" << (stability.hardReject ? 1 : 0)
                      << " geometry_stable=" << (stability.stable ? 1 : 0)
                      << " semantic_tau2=" << stability.semanticTau2
                      << " semantic_tau1=" << stability.semanticTau1
                      << " semantic_tau=" << stability.semanticTau
                      << " features_tau2=" << stability.featuresTau2
                      << " features_tau1=" << stability.featuresTau1
                      << " features_tau=" << stability.featuresTau
                      << " tracklets=" << stability.tracklets
                      << " tracklet_retention=" << stability.trackletRetention
                      << " bbox_iou_tau2_tau1=" << stability.bboxIoUTau2Tau1
                      << " bbox_iou_tau1_tau=" << stability.bboxIoUTau1Tau
                      << " bbox_center_shift_tau2_tau1=" << stability.bboxCenterShiftTau2Tau1
                      << " bbox_center_shift_tau1_tau=" << stability.bboxCenterShiftTau1Tau
                      << " bbox_area_ratio=" << stability.bboxAreaRatio
                      << " mask_area_ratio=" << stability.maskAreaRatio
                      << " feature_density_ratio=" << stability.featureDensityRatio
                      << " current_tracklet_coverage=" << stability.currentTrackletCoverage
                      << " current_tracklet_spread_x=" << stability.currentTrackletSpreadX
                      << " current_tracklet_spread_y=" << stability.currentTrackletSpreadY
                      << " median_flow_tau2_tau1_px=" << stability.medianFlowTau2Tau1Px
                      << " median_flow_tau1_tau_px=" << stability.medianFlowTau1TauPx
                      << " median_flow_accel_px=" << stability.medianFlowAccelerationPx
                      << " median_flow_accel_norm=" << stability.medianFlowAccelerationNormalized
                      << " median_flow_direction_cos=" << stability.medianFlowDirectionCosine
                      << " triangulation_samples=" << stability.triangulationSamples
                      << " triangulation_success_tau2_tau1=" << stability.triangulationSuccessRatioTau2Tau1
                      << " triangulation_success_tau1_tau=" << stability.triangulationSuccessRatioTau1Tau
                      << " triangulation_nonpositive_ratio=" << stability.triangulationNonpositiveRatio
                      << " triangulation_high_reproj_ratio=" << stability.triangulationHighReprojectionRatio
                      << " triangulation_invalid_geometry_ratio=" << stability.triangulationInvalidGeometryRatio
                      << std::endl;
        }
        return false;
    }

    if(static_cast<int>(tracklets.size()) < mnInstanceInitializationMinTracklets)
    {
        if(debugInitForInstance || debugLifecycleForInstance || DebugFocusFrame(mCurrentFrame.mnId))
        {
            std::cout << "[STSLAM_INSTANCE_INIT] frame=" << mCurrentFrame.mnId
                      << " instance_id=" << instanceId
                      << " rejected=low_tracklets"
                      << " map_id=" << currentMapId
                      << " map_origin_frame=" << currentMapOriginFrame
                      << " map_origin_kf=" << currentMapOriginKF
                      << " map_age_frames=" << currentMapAgeFrames
                      << " tau2_frame=" << frameTau2.mnId
                      << " tau1_frame=" << frameTau1.mnId
                      << " tau_frame=" << mCurrentFrame.mnId
                      << " features_tau2=" << featuresTau2.size()
                      << " features_tau1=" << featuresTau1.size()
                      << " features_tau=" << vCurrIndices.size()
                      << " tracklets=" << tracklets.size()
                      << " min_tracklets=" << mnInstanceInitializationMinTracklets
                      << std::endl;
        }
        return false;
    }

    std::vector<Eigen::Vector3f> pointsTau2;
    std::vector<Eigen::Vector3f> pointsTau1;
    std::vector<int> currentTrackletIndices;
    std::vector<std::array<int, 3>> acceptedTracklets;
    std::vector<double> triangulationWeights;
    std::vector<bool> triframeStructureEligibility;
    std::vector<bool> triframeCandidateSvdEligibility;
    std::vector<bool> triframeBackendObservationEligibility;
    std::vector<bool> triframeRecoveredByMultiPair;
    pointsTau2.reserve(tracklets.size());
    pointsTau1.reserve(tracklets.size());
    currentTrackletIndices.reserve(tracklets.size());
    acceptedTracklets.reserve(tracklets.size());
    triangulationWeights.reserve(tracklets.size());
    triframeStructureEligibility.reserve(tracklets.size());
    triframeCandidateSvdEligibility.reserve(tracklets.size());
    triframeBackendObservationEligibility.reserve(tracklets.size());
    triframeRecoveredByMultiPair.reserve(tracklets.size());
    int lowDepthTracklets = 0;
    int highReprojectionTracklets = 0;
    int lowParallaxTracklets = 0;
    int highParallaxTracklets = 0;
    int highDisparityTracklets = 0;
    int triframeCandidateChecked = 0;
    int triframeCandidateDownweighted = 0;
    int triframeCandidateStructureRejected = 0;
    int triframeCandidateHighReprojection = 0;
    int triframeCandidateHighFlowAcceleration = 0;
    int triframeCandidateLowDepth = 0;
    int triframeMultiPairChecked = 0;
    int triframeMultiPairRecovered = 0;
    int triframeMultiPairRecoveredTau2Tau1 = 0;
    int triframeMultiPairRecoveredTau1Tau = 0;
    int triframeMultiPairRecoveredTau2Tau = 0;
    int triframeMultiPairRejectedThirdView = 0;
    double triframeMultiPairThirdViewReprojectionSum = 0.0;
    double triframeMultiPairThirdViewReprojectionMax = 0.0;
    double triframeCandidateWeightSum = 0.0;
    double triframeCandidatePairReprojectionSum = 0.0;
    double triframeCandidateFlowAccelerationSum = 0.0;
    double triframeCandidateFlowAccelerationNormSum = 0.0;
    int failedTau2Tau1Triangulations = 0;
    int failedTau1TauTriangulations = 0;
    int invalidInputTriangulations = 0;
    int degenerateHomogeneousTriangulations = 0;
    int nonfiniteWorldTriangulations = 0;
    int nonpositiveDepthTriangulations = 0;
    int rejectedLowDepthTriangulations = 0;
    int rejectedHighReprojectionTriangulations = 0;
    int rejectedLowParallaxTriangulations = 0;
    int rejectedHighParallaxTriangulations = 0;
    int rejectedHighDisparityTriangulations = 0;
    int unknownTriangulationFailures = 0;
    double triangulationWeightSum = 0.0;
    double minTriangulationWeight = std::numeric_limits<double>::infinity();
    double maxTriangulationWeight = 0.0;
    double minTriangulationDepth = std::numeric_limits<double>::infinity();
    double maxTriangulationReprojection = 0.0;
    double reprojectionSum = 0.0;
    int reprojectionSamples = 0;
    double minTriangulationParallax = std::numeric_limits<double>::infinity();
    double maxTriangulationParallax = 0.0;
    double parallaxSum = 0.0;
    int parallaxSamples = 0;
    double disparityTau2Tau1Sum = 0.0;
    double disparityTau1TauSum = 0.0;
    double maxDisparityTau2Tau1 = 0.0;
    double maxDisparityTau1Tau = 0.0;
    auto accumulateTriangulationFailure = [&](const TriangulationQuality& quality)
    {
        const std::string& reason = quality.rejectReason;
        if(reason == "invalid_input")
            ++invalidInputTriangulations;
        else if(reason == "degenerate_homogeneous_w")
            ++degenerateHomogeneousTriangulations;
        else if(reason == "nonfinite_world_point")
            ++nonfiniteWorldTriangulations;
        else if(reason == "nonpositive_depth")
            ++nonpositiveDepthTriangulations;
        else if(reason == "low_depth")
            ++rejectedLowDepthTriangulations;
        else if(reason == "high_reprojection_error")
            ++rejectedHighReprojectionTriangulations;
        else if(reason == "low_parallax")
            ++rejectedLowParallaxTriangulations;
        else if(reason == "high_parallax")
            ++rejectedHighParallaxTriangulations;
        else if(reason == "high_disparity")
            ++rejectedHighDisparityTriangulations;
        else
            ++unknownTriangulationFailures;
    };

    auto reprojectionErrorToFrame = [](const Frame& frame,
                                       const int featureIdx,
                                       const Eigen::Vector3f& pointWorld,
                                       bool& negativeDepth) -> double
    {
        if(featureIdx < 0 ||
           featureIdx >= static_cast<int>(frame.mvKeysUn.size()) ||
           !frame.HasPose() ||
           !frame.mpCamera ||
           !pointWorld.allFinite())
        {
            return std::numeric_limits<double>::infinity();
        }

        const Eigen::Vector3f pointCamera = frame.GetPose() * pointWorld;
        if(!pointCamera.allFinite() || pointCamera[2] <= 0.0f)
        {
            negativeDepth = true;
            return std::numeric_limits<double>::infinity();
        }

        const Eigen::Vector3d pointCameraD = pointCamera.cast<double>();
        const Eigen::Vector2d projection =
            frame.mpCamera->project(pointCameraD);
        if(!projection.allFinite())
            return std::numeric_limits<double>::infinity();

        const cv::Point2f& observation = frame.mvKeysUn[featureIdx].pt;
        return (Eigen::Vector2d(observation.x, observation.y) - projection).norm();
    };

    for(const std::array<int, 3>& tracklet : tracklets)
    {
        Eigen::Vector3f pointTau2;
        Eigen::Vector3f pointTau1;
        TriangulationQuality qualityTau2Tau1;
        TriangulationQuality qualityTau1Tau;
        Eigen::Vector3f pointTau2Tau;
        TriangulationQuality qualityTau2Tau;
        const bool okTau2Tau1 =
            TriangulateMatchedFeatures(frameTau2, tracklet[0], frameTau1, tracklet[1], pointTau2, &qualityTau2Tau1);
        if(!okTau2Tau1)
        {
            ++failedTau2Tau1Triangulations;
            accumulateTriangulationFailure(qualityTau2Tau1);
        }
        const bool okTau1Tau =
            TriangulateMatchedFeatures(frameTau1, tracklet[1], mCurrentFrame, tracklet[2], pointTau1, &qualityTau1Tau);
        if(!okTau1Tau)
        {
            ++failedTau1TauTriangulations;
            accumulateTriangulationFailure(qualityTau1Tau);
        }

        bool usedMultiPairFallback = false;
        int multiPairFallbackSource = 0;
        double multiPairFallbackThirdViewReprojection = std::numeric_limits<double>::infinity();
        if(!(okTau2Tau1 && okTau1Tau))
        {
            if(!EnableTriFrameMultiPairTriangulation())
                continue;

            ++triframeMultiPairChecked;
            const bool okTau2Tau =
                TriangulateMatchedFeatures(frameTau2, tracklet[0], mCurrentFrame, tracklet[2], pointTau2Tau, &qualityTau2Tau);
            struct MultiPairCandidate
            {
                bool valid = false;
                int source = 0;
                Eigen::Vector3f point = Eigen::Vector3f::Zero();
                TriangulationQuality quality;
                double thirdViewReprojection = std::numeric_limits<double>::infinity();
            };

            std::vector<MultiPairCandidate> multiPairCandidates;
            multiPairCandidates.reserve(3);
            auto addMultiPairCandidate = [&](const bool ok,
                                             const int source,
                                             const Eigen::Vector3f& point,
                                             const TriangulationQuality& quality,
                                             const Frame& thirdFrame,
                                             const int thirdIdx)
            {
                if(!ok || !point.allFinite())
                    return;

                bool negativeDepth = false;
                const double thirdError =
                    reprojectionErrorToFrame(thirdFrame, thirdIdx, point, negativeDepth);
                if(negativeDepth ||
                   !std::isfinite(thirdError) ||
                   thirdError > GetTriFrameMultiPairMaxStaticReprojectionPx())
                {
                    ++triframeMultiPairRejectedThirdView;
                    return;
                }

                MultiPairCandidate candidate;
                candidate.valid = true;
                candidate.source = source;
                candidate.point = point;
                candidate.quality = quality;
                candidate.thirdViewReprojection = thirdError;
                multiPairCandidates.push_back(candidate);
            };

            addMultiPairCandidate(okTau2Tau1,
                                  1,
                                  pointTau2,
                                  qualityTau2Tau1,
                                  mCurrentFrame,
                                  tracklet[2]);
            addMultiPairCandidate(okTau1Tau,
                                  2,
                                  pointTau1,
                                  qualityTau1Tau,
                                  frameTau2,
                                  tracklet[0]);
            addMultiPairCandidate(okTau2Tau,
                                  3,
                                  pointTau2Tau,
                                  qualityTau2Tau,
                                  frameTau1,
                                  tracklet[1]);

            if(multiPairCandidates.empty())
                continue;

            const MultiPairCandidate* bestCandidate = &multiPairCandidates.front();
            auto candidateScore = [](const MultiPairCandidate& candidate) -> double
            {
                const double reproj =
                    std::isfinite(candidate.quality.maxReprojectionError) ?
                    candidate.quality.maxReprojectionError :
                    std::numeric_limits<double>::infinity();
                return candidate.thirdViewReprojection + reproj;
            };
            for(size_t i = 1; i < multiPairCandidates.size(); ++i)
            {
                if(candidateScore(multiPairCandidates[i]) < candidateScore(*bestCandidate))
                    bestCandidate = &multiPairCandidates[i];
            }

            pointTau2 = bestCandidate->point;
            pointTau1 = bestCandidate->point;
            qualityTau2Tau1 = bestCandidate->quality;
            qualityTau1Tau = bestCandidate->quality;
            usedMultiPairFallback = true;
            multiPairFallbackSource = bestCandidate->source;
            multiPairFallbackThirdViewReprojection = bestCandidate->thirdViewReprojection;
            ++triframeMultiPairRecovered;
            if(multiPairFallbackSource == 1)
                ++triframeMultiPairRecoveredTau2Tau1;
            else if(multiPairFallbackSource == 2)
                ++triframeMultiPairRecoveredTau1Tau;
            else if(multiPairFallbackSource == 3)
                ++triframeMultiPairRecoveredTau2Tau;
            triframeMultiPairThirdViewReprojectionSum += multiPairFallbackThirdViewReprojection;
            triframeMultiPairThirdViewReprojectionMax =
                std::max(triframeMultiPairThirdViewReprojectionMax,
                         multiPairFallbackThirdViewReprojection);
        }

        const double trackletWeight =
            std::min(qualityTau2Tau1.weight, qualityTau1Tau.weight) *
            (usedMultiPairFallback ? GetTriFrameMultiPairFallbackWeight() : 1.0);
        minTriangulationWeight = std::min(minTriangulationWeight, trackletWeight);
        maxTriangulationWeight = std::max(maxTriangulationWeight, trackletWeight);
        minTriangulationDepth =
            std::min(minTriangulationDepth,
                     std::min(qualityTau2Tau1.minDepth, qualityTau1Tau.minDepth));
        if(std::isfinite(qualityTau2Tau1.maxReprojectionError))
        {
            maxTriangulationReprojection =
                std::max(maxTriangulationReprojection, qualityTau2Tau1.maxReprojectionError);
            reprojectionSum += qualityTau2Tau1.maxReprojectionError;
            ++reprojectionSamples;
        }
        if(std::isfinite(qualityTau1Tau.maxReprojectionError))
        {
            maxTriangulationReprojection =
                std::max(maxTriangulationReprojection, qualityTau1Tau.maxReprojectionError);
            reprojectionSum += qualityTau1Tau.maxReprojectionError;
            ++reprojectionSamples;
        }
        if(std::isfinite(qualityTau2Tau1.parallaxDeg))
        {
            minTriangulationParallax = std::min(minTriangulationParallax, qualityTau2Tau1.parallaxDeg);
            maxTriangulationParallax = std::max(maxTriangulationParallax, qualityTau2Tau1.parallaxDeg);
            parallaxSum += qualityTau2Tau1.parallaxDeg;
            ++parallaxSamples;
        }
        if(std::isfinite(qualityTau1Tau.parallaxDeg))
        {
            minTriangulationParallax = std::min(minTriangulationParallax, qualityTau1Tau.parallaxDeg);
            maxTriangulationParallax = std::max(maxTriangulationParallax, qualityTau1Tau.parallaxDeg);
            parallaxSum += qualityTau1Tau.parallaxDeg;
            ++parallaxSamples;
        }
        if(tracklet[0] >= 0 && tracklet[0] < static_cast<int>(frameTau2.mvKeysUn.size()) &&
           tracklet[1] >= 0 && tracklet[1] < static_cast<int>(frameTau1.mvKeysUn.size()))
        {
            const double disparity = cv::norm(frameTau2.mvKeysUn[tracklet[0]].pt -
                                              frameTau1.mvKeysUn[tracklet[1]].pt);
            disparityTau2Tau1Sum += disparity;
            maxDisparityTau2Tau1 = std::max(maxDisparityTau2Tau1, disparity);
        }
        if(tracklet[1] >= 0 && tracklet[1] < static_cast<int>(frameTau1.mvKeysUn.size()) &&
           tracklet[2] >= 0 && tracklet[2] < static_cast<int>(mCurrentFrame.mvKeysUn.size()))
        {
            const double disparity = cv::norm(frameTau1.mvKeysUn[tracklet[1]].pt -
                                              mCurrentFrame.mvKeysUn[tracklet[2]].pt);
            disparityTau1TauSum += disparity;
            maxDisparityTau1Tau = std::max(maxDisparityTau1Tau, disparity);
        }
        if(qualityTau2Tau1.lowDepth || qualityTau1Tau.lowDepth)
            ++lowDepthTracklets;
        if(qualityTau2Tau1.highReprojectionError || qualityTau1Tau.highReprojectionError)
            ++highReprojectionTracklets;
        if(qualityTau2Tau1.lowParallax || qualityTau1Tau.lowParallax)
            ++lowParallaxTracklets;
        if(qualityTau2Tau1.highParallax || qualityTau1Tau.highParallax)
            ++highParallaxTracklets;
        if(qualityTau2Tau1.highDisparity || qualityTau1Tau.highDisparity)
            ++highDisparityTracklets;

        double candidateWeight = trackletWeight;
        bool candidateStructureEligible = true;
        bool candidateSvdEligible = true;
        if(EnableTriFrameTrackletCandidateQuality())
        {
            const TriFrameTrackletCandidateQuality candidateQuality =
                EvaluateTriFrameTrackletCandidate(frameTau2,
                                                  tracklet[0],
                                                  frameTau1,
                                                  tracklet[1],
                                                  mCurrentFrame,
                                                  tracklet[2],
                                                  qualityTau2Tau1,
                                                  qualityTau1Tau);
            if(candidateQuality.valid)
            {
                ++triframeCandidateChecked;
                triframeCandidateWeightSum += candidateQuality.weight;
                triframeCandidatePairReprojectionSum +=
                    std::isfinite(candidateQuality.pairMaxReprojectionError) ?
                    candidateQuality.pairMaxReprojectionError : 0.0;
                triframeCandidateFlowAccelerationSum +=
                    std::isfinite(candidateQuality.flowAccelerationPx) ?
                    candidateQuality.flowAccelerationPx : 0.0;
                triframeCandidateFlowAccelerationNormSum +=
                    std::isfinite(candidateQuality.flowAccelerationNormalized) ?
                    candidateQuality.flowAccelerationNormalized : 0.0;
                if(candidateQuality.lowDepth)
                    ++triframeCandidateLowDepth;
                if(candidateQuality.highReprojectionError)
                    ++triframeCandidateHighReprojection;
                if(candidateQuality.highFlowAcceleration)
                    ++triframeCandidateHighFlowAcceleration;

                candidateWeight =
                    std::max(GetTriFrameConsistencyMinWeight(),
                             std::min(1.0, trackletWeight * candidateQuality.weight));
                if(candidateWeight + 1e-9 < trackletWeight)
                    ++triframeCandidateDownweighted;
                candidateStructureEligible = candidateQuality.structureEligible;
                candidateSvdEligible = candidateQuality.structureEligible;
                if(!candidateStructureEligible)
                    ++triframeCandidateStructureRejected;
            }
            else
            {
                candidateWeight = GetTriFrameConsistencyMinWeight();
                candidateStructureEligible = false;
                candidateSvdEligible = false;
                ++triframeCandidateStructureRejected;
            }
        }

        pointsTau2.push_back(pointTau2);
        pointsTau1.push_back(pointTau1);
        currentTrackletIndices.push_back(tracklet[2]);
        acceptedTracklets.push_back(tracklet);
        triangulationWeights.push_back(candidateWeight);
        triframeStructureEligibility.push_back(candidateStructureEligible);
        triframeCandidateSvdEligibility.push_back(candidateSvdEligible);
        triframeBackendObservationEligibility.push_back(!usedMultiPairFallback ||
                                                        candidateStructureEligible);
        triframeRecoveredByMultiPair.push_back(usedMultiPairFallback);
        triangulationWeightSum += candidateWeight;
    }

    if(static_cast<int>(pointsTau2.size()) < mnInstanceInitializationMinTracklets)
    {
        if(debugInitForInstance || debugLifecycleForInstance || DebugFocusFrame(mCurrentFrame.mnId))
        {
            std::cout << "[STSLAM_INSTANCE_INIT] frame=" << mCurrentFrame.mnId
                      << " instance_id=" << instanceId
                      << " rejected=low_triangulated_tracklets"
                      << " map_id=" << currentMapId
                      << " map_origin_frame=" << currentMapOriginFrame
                      << " map_origin_kf=" << currentMapOriginKF
                      << " map_age_frames=" << currentMapAgeFrames
                      << " tau2_frame=" << frameTau2.mnId
                      << " tau1_frame=" << frameTau1.mnId
                      << " tau_frame=" << mCurrentFrame.mnId
                      << " tracklets=" << tracklets.size()
                      << " triangulated=" << pointsTau2.size()
                      << " min_tracklets=" << mnInstanceInitializationMinTracklets
                      << " low_depth_tracklets=" << lowDepthTracklets
                      << " high_reprojection_tracklets=" << highReprojectionTracklets
                      << " low_parallax_tracklets=" << lowParallaxTracklets
                      << " high_parallax_tracklets=" << highParallaxTracklets
                      << " high_disparity_tracklets=" << highDisparityTracklets
                      << " failed_tau2_tau1_triangulations=" << failedTau2Tau1Triangulations
                      << " failed_tau1_tau_triangulations=" << failedTau1TauTriangulations
                      << " invalid_input_triangulations=" << invalidInputTriangulations
                      << " degenerate_homogeneous_triangulations=" << degenerateHomogeneousTriangulations
                      << " nonfinite_world_triangulations=" << nonfiniteWorldTriangulations
                      << " nonpositive_depth_triangulations=" << nonpositiveDepthTriangulations
                      << " rejected_low_depth_triangulations=" << rejectedLowDepthTriangulations
                      << " rejected_high_reprojection_triangulations=" << rejectedHighReprojectionTriangulations
                      << " rejected_low_parallax_triangulations=" << rejectedLowParallaxTriangulations
                      << " rejected_high_parallax_triangulations=" << rejectedHighParallaxTriangulations
                      << " rejected_high_disparity_triangulations=" << rejectedHighDisparityTriangulations
                      << " unknown_triangulation_failures=" << unknownTriangulationFailures
                      << " triframe_candidate_quality="
                      << (EnableTriFrameTrackletCandidateQuality() ? 1 : 0)
                      << " triframe_candidate_checked=" << triframeCandidateChecked
                      << " triframe_candidate_downweighted=" << triframeCandidateDownweighted
                      << " triframe_candidate_structure_rejected=" << triframeCandidateStructureRejected
                      << " triframe_candidate_low_depth=" << triframeCandidateLowDepth
                      << " triframe_candidate_high_reprojection=" << triframeCandidateHighReprojection
                      << " triframe_candidate_high_flow_accel=" << triframeCandidateHighFlowAcceleration
                      << " triframe_candidate_mean_weight="
                      << (triframeCandidateChecked > 0 ?
                          triframeCandidateWeightSum / static_cast<double>(triframeCandidateChecked) : 0.0)
                      << " triframe_candidate_mean_pair_reproj="
                      << (triframeCandidateChecked > 0 ?
                          triframeCandidatePairReprojectionSum / static_cast<double>(triframeCandidateChecked) : 0.0)
                      << " triframe_candidate_mean_flow_accel_px="
                      << (triframeCandidateChecked > 0 ?
                          triframeCandidateFlowAccelerationSum / static_cast<double>(triframeCandidateChecked) : 0.0)
                      << " triframe_candidate_mean_flow_accel_norm="
                      << (triframeCandidateChecked > 0 ?
                          triframeCandidateFlowAccelerationNormSum / static_cast<double>(triframeCandidateChecked) : 0.0)
                      << " triframe_multi_pair="
                      << (EnableTriFrameMultiPairTriangulation() ? 1 : 0)
                      << " triframe_multi_pair_checked=" << triframeMultiPairChecked
                      << " triframe_multi_pair_recovered=" << triframeMultiPairRecovered
                      << " triframe_multi_pair_recovered_tau2_tau1=" << triframeMultiPairRecoveredTau2Tau1
                      << " triframe_multi_pair_recovered_tau1_tau=" << triframeMultiPairRecoveredTau1Tau
                      << " triframe_multi_pair_recovered_tau2_tau=" << triframeMultiPairRecoveredTau2Tau
                      << " triframe_multi_pair_rejected_third_view=" << triframeMultiPairRejectedThirdView
                      << " triframe_multi_pair_mean_third_reproj="
                      << (triframeMultiPairRecovered > 0 ?
                          triframeMultiPairThirdViewReprojectionSum / static_cast<double>(triframeMultiPairRecovered) : 0.0)
                      << " triframe_multi_pair_max_third_reproj=" << triframeMultiPairThirdViewReprojectionMax
                      << std::endl;
        }
        return false;
    }

    std::vector<Eigen::Vector3f> seedPointsTau2;
    std::vector<Eigen::Vector3f> seedPointsTau1;
    std::vector<double> seedTriangulationWeights;
    seedPointsTau2.reserve(pointsTau2.size());
    seedPointsTau1.reserve(pointsTau1.size());
    seedTriangulationWeights.reserve(triangulationWeights.size());
    for(size_t i = 0; i < pointsTau2.size() &&
                      i < pointsTau1.size() &&
                      i < triangulationWeights.size(); ++i)
    {
        const bool candidateSvdEligible =
            !EnableTriFrameTrackletCandidateQuality() ||
            i >= triframeCandidateSvdEligibility.size() ||
            triframeCandidateSvdEligibility[i];
        if(!candidateSvdEligible)
            continue;
        seedPointsTau2.push_back(pointsTau2[i]);
        seedPointsTau1.push_back(pointsTau1[i]);
        seedTriangulationWeights.push_back(triangulationWeights[i]);
    }
    const bool useCandidateSeedSvd =
        EnableTriFrameTrackletCandidateQuality() &&
        static_cast<int>(seedPointsTau2.size()) >= mnInstanceInitializationMinTracklets;
    Sophus::SE3f velocity = useCandidateSeedSvd ?
        SolveWeightedRigidTransformSVD(seedPointsTau2, seedPointsTau1, seedTriangulationWeights) :
        SolveWeightedRigidTransformSVD(pointsTau2, pointsTau1, triangulationWeights);
    if(!IsFiniteSE3(velocity))
    {
        if(debugInitForInstance || debugLifecycleForInstance || DebugFocusFrame(mCurrentFrame.mnId))
        {
            std::cout << "[STSLAM_INSTANCE_INIT] frame=" << mCurrentFrame.mnId
                      << " instance_id=" << instanceId
                      << " rejected=nonfinite_svd_velocity"
                      << " map_id=" << currentMapId
                      << " map_origin_frame=" << currentMapOriginFrame
                      << " map_origin_kf=" << currentMapOriginKF
                      << " map_age_frames=" << currentMapAgeFrames
                      << " tau2_frame=" << frameTau2.mnId
                      << " tau1_frame=" << frameTau1.mnId
                      << " tau_frame=" << mCurrentFrame.mnId
                      << " triangulated=" << pointsTau2.size()
                      << " triframe_candidate_quality="
                      << (EnableTriFrameTrackletCandidateQuality() ? 1 : 0)
                      << " triframe_candidate_svd_seed=" << seedPointsTau2.size()
                      << " triframe_candidate_seed_svd_used=" << (useCandidateSeedSvd ? 1 : 0)
                      << std::endl;
        }
        return false;
    }
    const Sophus::SE3f rawSvdVelocity = velocity;

    int triframeConsistencyChecked = 0;
    int triframeConsistencyDownweighted = 0;
    int triframeConsistencyBad = 0;
    int triframeConsistencyNegativeDepth = 0;
    int triframeStructurePromotionRejected = 0;
    double triframeConsistencyWeightSum = 0.0;
    double triframeConsistencyMaxReprojection = 0.0;
    double triframeConsistencyReprojectionSum = 0.0;
    double triframeConsistencyDynamicReprojectionSum = 0.0;
    double triframeConsistencyMotion3dResidualSum = 0.0;
    if(EnableTriFrameConsistencyWeights())
    {
        const double maxTriFrameReprojection = GetTriFrameConsistencyMaxReprojectionPx();
        for(size_t i = 0; i < pointsTau2.size() &&
                          i < pointsTau1.size() &&
                          i < acceptedTracklets.size() &&
                          i < triangulationWeights.size() &&
                          i < triframeStructureEligibility.size(); ++i)
        {
            const std::array<int, 3>& tracklet = acceptedTracklets[i];
            const TriFrameConsistencyQuality consistency =
                EvaluateTriFrameConsistency(frameTau2,
                                            tracklet[0],
                                            frameTau1,
                                            tracklet[1],
                                            mCurrentFrame,
                                            tracklet[2],
                                            velocity,
                                            pointsTau2[i],
                                            pointsTau1[i]);
            if(!consistency.valid)
                continue;

            ++triframeConsistencyChecked;
            triframeConsistencyWeightSum += consistency.weight;
            if(consistency.negativeDepth)
                ++triframeConsistencyNegativeDepth;
            bool badTriFrameGeometry = consistency.negativeDepth ||
                                       !std::isfinite(consistency.maxReprojectionError);
            if(std::isfinite(consistency.maxReprojectionError))
            {
                triframeConsistencyMaxReprojection =
                    std::max(triframeConsistencyMaxReprojection,
                             consistency.maxReprojectionError);
                triframeConsistencyReprojectionSum += consistency.maxReprojectionError;
                if(maxTriFrameReprojection > 0.0 &&
                   consistency.maxReprojectionError > maxTriFrameReprojection)
                {
                    ++triframeConsistencyBad;
                    badTriFrameGeometry = true;
                }
            }
            else
            {
                ++triframeConsistencyBad;
            }
            if(std::isfinite(consistency.dynamicReprojectionError))
                triframeConsistencyDynamicReprojectionSum += consistency.dynamicReprojectionError;
            if(std::isfinite(consistency.motion3dResidual))
                triframeConsistencyMotion3dResidualSum += consistency.motion3dResidual;

            const double oldWeight =
                std::isfinite(triangulationWeights[i]) ? triangulationWeights[i] : 1.0;
            const double newWeight =
                std::max(GetTriFrameConsistencyMinWeight(),
                         std::min(1.0, oldWeight * consistency.weight));
            if(newWeight + 1e-9 < oldWeight)
                ++triframeConsistencyDownweighted;
            triangulationWeights[i] = newWeight;

            if(EnableTriFrameConsistencyStructurePromotionGate() &&
               badTriFrameGeometry &&
               triframeStructureEligibility[i])
            {
                triframeStructureEligibility[i] = false;
                ++triframeStructurePromotionRejected;
            }
            if(badTriFrameGeometry &&
               i < triframeRecoveredByMultiPair.size() &&
               triframeRecoveredByMultiPair[i] &&
               i < triframeBackendObservationEligibility.size())
            {
                triframeBackendObservationEligibility[i] = false;
            }
        }
    }

    std::vector<double> svdResiduals;
    svdResiduals.reserve(pointsTau2.size());
    for(size_t i = 0; i < pointsTau2.size(); ++i)
    {
        const double rawResidual = (velocity * pointsTau2[i] - pointsTau1[i]).cast<double>().norm();
        const double weight =
            (i < triangulationWeights.size() && std::isfinite(triangulationWeights[i])) ?
            std::max(0.0, triangulationWeights[i]) : 1.0;
        svdResiduals.push_back(rawResidual * std::sqrt(weight));
    }

    std::vector<double> sortedResiduals = svdResiduals;
    std::sort(sortedResiduals.begin(), sortedResiduals.end());
    const double medianResidual =
        sortedResiduals.empty() ? 0.0 : sortedResiduals[sortedResiduals.size() / 2];
    if(!std::isfinite(medianResidual))
        return false;

    const double inlierThreshold =
        std::max(1e-6, medianResidual * GetInstanceInitializationSVDInlierScale());
    std::vector<Eigen::Vector3f> inlierPointsTau2;
    std::vector<Eigen::Vector3f> inlierPointsTau1;
    std::vector<int> inlierCurrentTrackletIndices;
    std::vector<std::array<int, 3>> inlierAcceptedTracklets;
    std::vector<double> inlierTriangulationWeights;
    std::vector<bool> inlierTriFrameStructureEligibility;
    std::vector<bool> inlierTriFrameCandidateSvdEligibility;
    std::vector<bool> inlierTriFrameBackendObservationEligibility;
    std::vector<bool> inlierTriFrameRecoveredByMultiPair;
    inlierPointsTau2.reserve(pointsTau2.size());
    inlierPointsTau1.reserve(pointsTau1.size());
    inlierCurrentTrackletIndices.reserve(currentTrackletIndices.size());
    inlierAcceptedTracklets.reserve(acceptedTracklets.size());
    inlierTriangulationWeights.reserve(triangulationWeights.size());
    inlierTriFrameStructureEligibility.reserve(triframeStructureEligibility.size());
    inlierTriFrameCandidateSvdEligibility.reserve(triframeCandidateSvdEligibility.size());
    inlierTriFrameBackendObservationEligibility.reserve(triframeBackendObservationEligibility.size());
    inlierTriFrameRecoveredByMultiPair.reserve(triframeRecoveredByMultiPair.size());
    double inlierResidualSum = 0.0;
    for(size_t i = 0; i < pointsTau2.size(); ++i)
    {
        if(svdResiduals[i] > inlierThreshold)
            continue;

        inlierPointsTau2.push_back(pointsTau2[i]);
        inlierPointsTau1.push_back(pointsTau1[i]);
        inlierCurrentTrackletIndices.push_back(currentTrackletIndices[i]);
        if(i < acceptedTracklets.size())
            inlierAcceptedTracklets.push_back(acceptedTracklets[i]);
        if(i < triangulationWeights.size())
            inlierTriangulationWeights.push_back(triangulationWeights[i]);
        if(i < triframeStructureEligibility.size())
            inlierTriFrameStructureEligibility.push_back(triframeStructureEligibility[i]);
        if(i < triframeCandidateSvdEligibility.size())
            inlierTriFrameCandidateSvdEligibility.push_back(triframeCandidateSvdEligibility[i]);
        if(i < triframeBackendObservationEligibility.size())
            inlierTriFrameBackendObservationEligibility.push_back(triframeBackendObservationEligibility[i]);
        if(i < triframeRecoveredByMultiPair.size())
            inlierTriFrameRecoveredByMultiPair.push_back(triframeRecoveredByMultiPair[i]);
        inlierResidualSum += svdResiduals[i];
    }

    const double inlierRatio =
        pointsTau2.empty() ? 0.0 :
        static_cast<double>(inlierPointsTau2.size()) / static_cast<double>(pointsTau2.size());
    const size_t acceptedInlierCount = inlierPointsTau2.size();
    const double meanInlierResidual =
        acceptedInlierCount == 0 ? 0.0 :
        inlierResidualSum / static_cast<double>(acceptedInlierCount);
    if(static_cast<int>(inlierPointsTau2.size()) < mnInstanceInitializationMinTracklets ||
       inlierRatio < GetInstanceInitializationMinInlierRatio())
    {
        if(debugInitForInstance || debugLifecycleForInstance || DebugFocusFrame(mCurrentFrame.mnId))
        {
            std::cout << "[STSLAM_INSTANCE_INIT] frame=" << mCurrentFrame.mnId
                      << " instance_id=" << instanceId
                      << " rejected=low_quality_tracklets"
                      << " map_id=" << currentMapId
                      << " map_origin_frame=" << currentMapOriginFrame
                      << " map_origin_kf=" << currentMapOriginKF
                      << " map_age_frames=" << currentMapAgeFrames
                      << " tau2_frame=" << frameTau2.mnId
                      << " tau1_frame=" << frameTau1.mnId
                      << " tau_frame=" << mCurrentFrame.mnId
                      << " tracklets=" << static_cast<int>(tracklets.size())
                      << " triangulated=" << static_cast<int>(pointsTau2.size())
                      << " inliers=" << static_cast<int>(inlierPointsTau2.size())
                      << " inlier_ratio=" << inlierRatio
                      << " median_svd_residual=" << medianResidual
                      << " inlier_threshold=" << inlierThreshold
                      << " mean_triangulation_weight="
                      << (pointsTau2.empty() ? 0.0 : triangulationWeightSum / static_cast<double>(pointsTau2.size()))
                      << " triframe_candidate_quality="
                      << (EnableTriFrameTrackletCandidateQuality() ? 1 : 0)
                      << " triframe_candidate_checked=" << triframeCandidateChecked
                      << " triframe_candidate_svd_seed=" << seedPointsTau2.size()
                      << " triframe_candidate_seed_svd_used=" << (useCandidateSeedSvd ? 1 : 0)
                      << " triframe_candidate_downweighted=" << triframeCandidateDownweighted
                      << " triframe_candidate_structure_rejected=" << triframeCandidateStructureRejected
                      << " triframe_candidate_low_depth=" << triframeCandidateLowDepth
                      << " triframe_candidate_high_reprojection=" << triframeCandidateHighReprojection
                      << " triframe_candidate_high_flow_accel=" << triframeCandidateHighFlowAcceleration
                      << " triframe_candidate_mean_weight="
                      << (triframeCandidateChecked > 0 ?
                          triframeCandidateWeightSum / static_cast<double>(triframeCandidateChecked) : 0.0)
                      << " triframe_candidate_mean_pair_reproj="
                      << (triframeCandidateChecked > 0 ?
                          triframeCandidatePairReprojectionSum / static_cast<double>(triframeCandidateChecked) : 0.0)
                      << " triframe_candidate_mean_flow_accel_px="
                      << (triframeCandidateChecked > 0 ?
                          triframeCandidateFlowAccelerationSum / static_cast<double>(triframeCandidateChecked) : 0.0)
                      << " triframe_candidate_mean_flow_accel_norm="
                      << (triframeCandidateChecked > 0 ?
                          triframeCandidateFlowAccelerationNormSum / static_cast<double>(triframeCandidateChecked) : 0.0)
                      << " triframe_multi_pair="
                      << (EnableTriFrameMultiPairTriangulation() ? 1 : 0)
                      << " triframe_multi_pair_checked=" << triframeMultiPairChecked
                      << " triframe_multi_pair_recovered=" << triframeMultiPairRecovered
                      << " triframe_multi_pair_recovered_tau2_tau1=" << triframeMultiPairRecoveredTau2Tau1
                      << " triframe_multi_pair_recovered_tau1_tau=" << triframeMultiPairRecoveredTau1Tau
                      << " triframe_multi_pair_recovered_tau2_tau=" << triframeMultiPairRecoveredTau2Tau
                      << " triframe_multi_pair_rejected_third_view=" << triframeMultiPairRejectedThirdView
                      << " triframe_multi_pair_mean_third_reproj="
                      << (triframeMultiPairRecovered > 0 ?
                          triframeMultiPairThirdViewReprojectionSum / static_cast<double>(triframeMultiPairRecovered) : 0.0)
                      << " triframe_multi_pair_max_third_reproj=" << triframeMultiPairThirdViewReprojectionMax
                      << " triframe_consistency_enabled="
                      << (EnableTriFrameConsistencyWeights() ? 1 : 0)
                      << " triframe_checked=" << triframeConsistencyChecked
                      << " triframe_bad=" << triframeConsistencyBad
                      << " triframe_downweighted=" << triframeConsistencyDownweighted
                      << " triframe_negative_depth=" << triframeConsistencyNegativeDepth
                      << " triframe_structure_promotion_gate="
                      << (EnableTriFrameConsistencyStructurePromotionGate() ? 1 : 0)
                      << " triframe_structure_rejected=" << triframeStructurePromotionRejected
                      << " triframe_mean_weight="
                      << (triframeConsistencyChecked > 0 ?
                          triframeConsistencyWeightSum / static_cast<double>(triframeConsistencyChecked) : 0.0)
                      << " triframe_mean_reprojection_error="
                      << (triframeConsistencyChecked > 0 ?
                          triframeConsistencyReprojectionSum / static_cast<double>(triframeConsistencyChecked) : 0.0)
                      << " triframe_max_reprojection_error=" << triframeConsistencyMaxReprojection
                      << " triframe_mean_dynamic_reprojection_error="
                      << (triframeConsistencyChecked > 0 ?
                          triframeConsistencyDynamicReprojectionSum / static_cast<double>(triframeConsistencyChecked) : 0.0)
                      << " triframe_mean_motion3d_residual="
                      << (triframeConsistencyChecked > 0 ?
                          triframeConsistencyMotion3dResidualSum / static_cast<double>(triframeConsistencyChecked) : 0.0)
                      << " low_depth_tracklets=" << lowDepthTracklets
                      << " high_reprojection_tracklets=" << highReprojectionTracklets
                      << " low_parallax_tracklets=" << lowParallaxTracklets
                      << " high_parallax_tracklets=" << highParallaxTracklets
                      << " high_disparity_tracklets=" << highDisparityTracklets
                      << " min_triangulation_weight="
                      << (std::isfinite(minTriangulationWeight) ? minTriangulationWeight : 0.0)
                      << " max_triangulation_weight=" << maxTriangulationWeight
                      << " min_triangulation_depth="
                      << (std::isfinite(minTriangulationDepth) ? minTriangulationDepth : 0.0)
                      << " mean_reprojection_error="
                      << (reprojectionSamples > 0 ? reprojectionSum / static_cast<double>(reprojectionSamples) : 0.0)
                      << " max_reprojection_error=" << maxTriangulationReprojection
                      << " mean_parallax_deg="
                      << (parallaxSamples > 0 ? parallaxSum / static_cast<double>(parallaxSamples) : 0.0)
                      << " min_parallax_deg="
                      << (std::isfinite(minTriangulationParallax) ? minTriangulationParallax : 0.0)
                      << " max_parallax_deg=" << maxTriangulationParallax
                      << " mean_disparity_tau2_tau1="
                      << (pointsTau2.empty() ? 0.0 : disparityTau2Tau1Sum / static_cast<double>(pointsTau2.size()))
                      << " mean_disparity_tau1_tau="
                      << (pointsTau2.empty() ? 0.0 : disparityTau1TauSum / static_cast<double>(pointsTau2.size()))
                      << " max_disparity_tau2_tau1=" << maxDisparityTau2Tau1
                      << " max_disparity_tau1_tau=" << maxDisparityTau1Tau
                      << std::endl;
        }
        return false;
    }

    if(inlierPointsTau2.size() < pointsTau2.size())
    {
        velocity = SolveWeightedRigidTransformSVD(inlierPointsTau2, inlierPointsTau1, inlierTriangulationWeights);
        if(!IsFiniteSE3(velocity))
            return false;
        pointsTau2.swap(inlierPointsTau2);
        pointsTau1.swap(inlierPointsTau1);
        currentTrackletIndices.swap(inlierCurrentTrackletIndices);
        acceptedTracklets.swap(inlierAcceptedTracklets);
        triangulationWeights.swap(inlierTriangulationWeights);
        triframeStructureEligibility.swap(inlierTriFrameStructureEligibility);
        triframeCandidateSvdEligibility.swap(inlierTriFrameCandidateSvdEligibility);
        triframeBackendObservationEligibility.swap(inlierTriFrameBackendObservationEligibility);
        triframeRecoveredByMultiPair.swap(inlierTriFrameRecoveredByMultiPair);
    }

    double activeTriangulationWeightSum = 0.0;
    for(double weight : triangulationWeights)
    {
        if(std::isfinite(weight) && weight > 0.0)
            activeTriangulationWeightSum += weight;
    }
    if(activeTriangulationWeightSum <= 0.0)
        return false;
    const InstanceSvdMotionDiagnostics svdMotionDiagnostics =
        ComputeInstanceSvdMotionDiagnostics(pointsTau2,
                                            pointsTau1,
                                            triangulationWeights,
                                            velocity);

    Eigen::Vector3f centroidTau2 = Eigen::Vector3f::Zero();
    Eigen::Vector3f centroidTau1ForVelocity = Eigen::Vector3f::Zero();
    for(size_t i = 0; i < pointsTau1.size(); ++i)
    {
        const double qualityWeight =
            (i < triangulationWeights.size() && std::isfinite(triangulationWeights[i])) ?
            std::max(0.0, triangulationWeights[i]) : 1.0;
        centroidTau2 += static_cast<float>(qualityWeight) * pointsTau2[i];
        centroidTau1ForVelocity += static_cast<float>(qualityWeight) * pointsTau1[i];
    }
    centroidTau2 /= static_cast<float>(activeTriangulationWeightSum);
    centroidTau1ForVelocity /= static_cast<float>(activeTriangulationWeightSum);
    Sophus::SE3f translationOnlyVelocity(
        Eigen::Matrix3f::Identity(),
        centroidTau1ForVelocity - centroidTau2);
    if(!centroidTau2.allFinite() ||
       !centroidTau1ForVelocity.allFinite() ||
       !IsFiniteSE3(translationOnlyVelocity))
    {
        translationOnlyVelocity = Sophus::SE3f();
    }

    InstanceMotionGateResult initGate;
    InstanceMotionGateResult zeroMotionInitGate;
    InstanceMotionGateResult translationOnlyInitGate;
    const bool strictPaperInitialization = UseStrictPaperArchitectureDefaults();
    bool strictStaticZeroVelocityInit = false;
    std::string strictStaticZeroVelocityReason = "none";
    bool initDynamicQualityRejected = false;
    bool initDynamicRotationRejected = false;
    bool translationOnlyDynamicQualityRejected = false;
    bool translationOnlyDynamicRotationRejected = false;
    int initGateChecked = 0;
    int initGateInvalidIndex = 0;
    int initGateNonFinitePoint = 0;
    int initGateNonPositiveDepth = 0;
    int initGateNonFiniteProjection = 0;
    int translationOnlyNonPositiveDepth = 0;
    int translationOnlySupport = 0;
    float initGateMinStaticDepth = 1e30f;
    float initGateMaxStaticDepth = -1e30f;
    float initGateMinDynamicDepth = 1e30f;
    float initGateMaxDynamicDepth = -1e30f;
    float translationOnlyMinDynamicDepth = 1e30f;
    float translationOnlyMaxDynamicDepth = -1e30f;
    if(mCurrentFrame.mpCamera && mCurrentFrame.HasPose())
    {
        const Sophus::SE3f Tcw = mCurrentFrame.GetPose();
        const auto applyInitializationDynamicQualityGate =
            [](InstanceMotionGateResult& gate,
               const Sophus::SE3f& candidateVelocity,
               bool& residualRejected,
               bool& rotationRejected) -> void
        {
            residualRejected = false;
            rotationRejected = false;
            if(!gate.valid || gate.state != kInstanceMotionDynamic)
                return;

            const double dynamicRmse =
                (gate.dynamicMeanError > 0.0 && std::isfinite(gate.dynamicMeanError)) ?
                std::sqrt(gate.dynamicMeanError) : std::numeric_limits<double>::infinity();
            const Eigen::AngleAxisd angleAxis(candidateVelocity.rotationMatrix().cast<double>());
            const double rotationDeg =
                std::abs(angleAxis.angle()) * 180.0 / 3.14159265358979323846;
            if(dynamicRmse > GetInstanceInitializationMaxDynamicRmsePx())
                residualRejected = true;
            if(rotationDeg > GetInstanceInitializationMaxDynamicRotationDeg())
                rotationRejected = true;
            if(residualRejected || rotationRejected)
            {
                gate.state = kInstanceMotionUncertain;
                gate.useDynamicMotion = false;
            }
        };
        double staticErrorSum = 0.0;
        double dynamicErrorSum = 0.0;
        double translationOnlyErrorSum = 0.0;
        double selfCheckWeightSum = 0.0;
        double translationOnlyWeightSum = 0.0;
        int support = 0;
        for(size_t i = 0; i < pointsTau2.size() && i < currentTrackletIndices.size(); ++i)
        {
            ++initGateChecked;
            const int currentIdx = currentTrackletIndices[i];
            if(currentIdx < 0 || currentIdx >= static_cast<int>(mCurrentFrame.mvKeysUn.size()))
            {
                ++initGateInvalidIndex;
                continue;
            }

            const Eigen::Vector3f staticPoint = pointsTau2[i];
            const Eigen::Vector3f dynamicPoint = velocity * pointsTau2[i];
            const Eigen::Vector3f translationOnlyPoint =
                translationOnlyVelocity * pointsTau2[i];
            if(!staticPoint.allFinite() ||
               !dynamicPoint.allFinite() ||
               !translationOnlyPoint.allFinite())
            {
                ++initGateNonFinitePoint;
                continue;
            }

            const Eigen::Vector3f staticCam = Tcw * staticPoint;
            const Eigen::Vector3f dynamicCam = Tcw * dynamicPoint;
            const Eigen::Vector3f translationOnlyCam = Tcw * translationOnlyPoint;
            if(staticCam.allFinite())
            {
                initGateMinStaticDepth = std::min(initGateMinStaticDepth, staticCam[2]);
                initGateMaxStaticDepth = std::max(initGateMaxStaticDepth, staticCam[2]);
            }
            if(dynamicCam.allFinite())
            {
                initGateMinDynamicDepth = std::min(initGateMinDynamicDepth, dynamicCam[2]);
                initGateMaxDynamicDepth = std::max(initGateMaxDynamicDepth, dynamicCam[2]);
            }
            if(translationOnlyCam.allFinite())
            {
                translationOnlyMinDynamicDepth =
                    std::min(translationOnlyMinDynamicDepth, translationOnlyCam[2]);
                translationOnlyMaxDynamicDepth =
                    std::max(translationOnlyMaxDynamicDepth, translationOnlyCam[2]);
            }
            if(staticCam[2] <= 0.0f || dynamicCam[2] <= 0.0f)
            {
                ++initGateNonPositiveDepth;
            }
            if(staticCam[2] <= 0.0f || translationOnlyCam[2] <= 0.0f)
                ++translationOnlyNonPositiveDepth;

            const Eigen::Vector3d staticCamD = staticCam.cast<double>();
            const Eigen::Vector2d staticProjection =
                mCurrentFrame.mpCamera->project(staticCamD);
            if(!staticProjection.allFinite())
            {
                ++initGateNonFiniteProjection;
                continue;
            }

            const Eigen::Vector2d observation(mCurrentFrame.mvKeysUn[currentIdx].pt.x,
                                              mCurrentFrame.mvKeysUn[currentIdx].pt.y);
            const double qualityWeight =
                (i < triangulationWeights.size() && std::isfinite(triangulationWeights[i])) ?
                std::max(0.0, triangulationWeights[i]) : 1.0;
            const double staticError =
                qualityWeight * (observation - staticProjection).squaredNorm();

            if(dynamicCam[2] > 0.0f)
            {
                const Eigen::Vector3d dynamicCamD = dynamicCam.cast<double>();
                const Eigen::Vector2d dynamicProjection =
                    mCurrentFrame.mpCamera->project(dynamicCamD);
                if(dynamicProjection.allFinite())
                {
                    staticErrorSum += staticError;
                    dynamicErrorSum +=
                        qualityWeight * (observation - dynamicProjection).squaredNorm();
                    selfCheckWeightSum += qualityWeight;
                    ++support;
                }
                else
                {
                    ++initGateNonFiniteProjection;
                }
            }

            if(translationOnlyCam[2] > 0.0f)
            {
                const Eigen::Vector3d translationOnlyCamD = translationOnlyCam.cast<double>();
                const Eigen::Vector2d translationOnlyProjection =
                    mCurrentFrame.mpCamera->project(translationOnlyCamD);
                if(translationOnlyProjection.allFinite())
                {
                    translationOnlyInitGate.staticMeanError += staticError;
                    translationOnlyErrorSum +=
                        qualityWeight * (observation - translationOnlyProjection).squaredNorm();
                    translationOnlyWeightSum += qualityWeight;
                    ++translationOnlySupport;
                }
            }
        }

        if(support >= mnInstanceInitializationMinTracklets && selfCheckWeightSum > 0.0)
        {
            initGate.valid = true;
            initGate.support = support;
            initGate.staticMeanError = staticErrorSum / selfCheckWeightSum;
            initGate.dynamicMeanError = dynamicErrorSum / selfCheckWeightSum;
            ClassifyInstanceMotionGate(initGate);
            applyInitializationDynamicQualityGate(initGate,
                                                  velocity,
                                                  initDynamicQualityRejected,
                                                  initDynamicRotationRejected);
        }
        if(translationOnlySupport >= mnInstanceInitializationMinTracklets &&
           translationOnlyWeightSum > 0.0)
        {
            translationOnlyInitGate.valid = true;
            translationOnlyInitGate.support = translationOnlySupport;
            translationOnlyInitGate.staticMeanError /=
                translationOnlyWeightSum;
            translationOnlyInitGate.dynamicMeanError =
                translationOnlyErrorSum / translationOnlyWeightSum;
            ClassifyInstanceMotionGate(translationOnlyInitGate);
            applyInitializationDynamicQualityGate(translationOnlyInitGate,
                                                  translationOnlyVelocity,
                                                  translationOnlyDynamicQualityRejected,
                                                  translationOnlyDynamicRotationRejected);
            if(translationOnlyInitGate.valid &&
               IsNearlyIdentityInstanceMotion(translationOnlyVelocity))
            {
                translationOnlyInitGate.state = kInstanceMotionStatic;
                translationOnlyInitGate.useDynamicMotion = false;
            }
        }
    }
    zeroMotionInitGate = initGate;
    const double zeroMotionStaticRmsePx =
        (zeroMotionInitGate.valid &&
         zeroMotionInitGate.staticMeanError > 0.0 &&
         std::isfinite(zeroMotionInitGate.staticMeanError)) ?
        std::sqrt(zeroMotionInitGate.staticMeanError) :
        std::numeric_limits<double>::infinity();
    const bool zeroMotionReprojectionQualityOk =
        zeroMotionInitGate.valid &&
        std::isfinite(zeroMotionStaticRmsePx) &&
        zeroMotionStaticRmsePx <= GetStrictStaticZeroVelocityMaxReprojectionRmsePx();
    const bool useTranslationOnlyInit =
        !strictPaperInitialization &&
        EnableInstanceInitializationTranslationOnlyCandidate() &&
        translationOnlyInitGate.valid &&
        translationOnlyInitGate.state == kInstanceMotionDynamic &&
        ((initGate.valid &&
          initGate.state == kInstanceMotionDynamic &&
          translationOnlyInitGate.dynamicMeanError < initGate.dynamicMeanError) ||
         EnableInstanceInitializationTranslationOnlyDynamicPromotion());
    if(useTranslationOnlyInit)
    {
        initGate = translationOnlyInitGate;
        velocity = translationOnlyVelocity;
    }
    std::string svdMotionReliabilityReason =
        ClassifyStrictSvdMotionReliability(svdMotionDiagnostics, zeroMotionInitGate);
    bool svdMotionReliable =
        svdMotionReliabilityReason == "reliable" ||
        svdMotionReliabilityReason == "disabled";
    bool unreliableSvdMotionSuppressed = false;
    bool nonzeroUnreliableMotionInitializedAsUncertain = false;
    if(!strictPaperInitialization &&
       initGate.valid &&
       initGate.state == kInstanceMotionDynamic)
    {
        const int dynamicInitializationConfirmFrames =
            GetInstanceInitializationDynamicConfirmFrames();
        pInstance->RecordMotionGateState(static_cast<int>(initGate.state),
                                         velocity,
                                         static_cast<int>(mCurrentFrame.mnId),
                                         static_cast<float>(GetInstanceStaticVelocityDecay()));
        const int dynamicInitializationEvidence = pInstance->GetDynamicMotionEvidence();
        if(dynamicInitializationEvidence < dynamicInitializationConfirmFrames)
        {
            if(debugInitForInstance || debugLifecycleForInstance || DebugFocusFrame(mCurrentFrame.mnId))
            {
                const Eigen::AngleAxisf angleAxis(velocity.rotationMatrix());
                std::cout << "[STSLAM_INSTANCE_INIT] frame=" << mCurrentFrame.mnId
                          << " instance_id=" << instanceId
                          << " rejected=dynamic_confirmation_wait"
                          << " map_id=" << currentMapId
                          << " map_origin_frame=" << currentMapOriginFrame
                          << " map_origin_kf=" << currentMapOriginKF
                          << " map_age_frames=" << currentMapAgeFrames
                          << " tau2_frame=" << frameTau2.mnId
                          << " tau1_frame=" << frameTau1.mnId
                          << " tau_frame=" << mCurrentFrame.mnId
                          << " state=" << InstanceMotionGateStateName(initGate.state)
                          << " dynamic_evidence=" << dynamicInitializationEvidence
                          << " required_dynamic_evidence=" << dynamicInitializationConfirmFrames
                          << " support=" << initGate.support
                          << " static_mean_error=" << initGate.staticMeanError
                          << " dynamic_mean_error=" << initGate.dynamicMeanError
                          << " dynamic_rmse_px="
                          << (initGate.dynamicMeanError > 0.0 && std::isfinite(initGate.dynamicMeanError) ?
                              std::sqrt(initGate.dynamicMeanError) : 0.0)
                          << " selected_translation_only=" << (useTranslationOnlyInit ? 1 : 0)
                          << " translation_only_state=" << InstanceMotionGateStateName(translationOnlyInitGate.state)
                          << " translation_only_support=" << translationOnlyInitGate.support
                          << " translation_only_mean_error=" << translationOnlyInitGate.dynamicMeanError
                          << " translation_norm=" << velocity.translation().norm()
                          << " rotation_deg=" << std::abs(angleAxis.angle()) * 180.0 / 3.14159265358979323846
                          << std::endl;
            }
            return false;
        }
    }
    if(strictPaperInitialization)
    {
        initGate = zeroMotionInitGate;
        if(EnableStrictStaticZeroVelocityInitialization())
        {
            const bool staticPreferredByReprojection =
                zeroMotionInitGate.valid &&
                zeroMotionInitGate.state == kInstanceMotionStatic &&
                zeroMotionReprojectionQualityOk;
            const bool smallSvdMotion =
                zeroMotionInitGate.valid &&
                zeroMotionReprojectionQualityOk &&
                IsSmallInstanceMotion(velocity,
                                      GetStrictStaticZeroVelocityMaxTranslation(),
                                      GetStrictStaticZeroVelocityMaxRotationDeg());
            const bool significantCentroidMotion =
                EnableStrictUncertainCentroidMotionInitialization() &&
                IsFiniteSE3(translationOnlyVelocity) &&
                !IsSmallInstanceMotion(translationOnlyVelocity,
                                       GetStrictStaticZeroVelocityMaxTranslation(),
                                       GetStrictStaticZeroVelocityMaxRotationDeg());
            const bool rawSvdMotionNonZero =
                !IsSmallInstanceMotion(rawSvdVelocity,
                                       GetStrictStaticZeroVelocityMaxTranslation(),
                                       GetStrictStaticZeroVelocityMaxRotationDeg());
            const bool zeroMotionVeryStrong =
                staticPreferredByReprojection &&
                zeroMotionStaticRmsePx <=
                    GetStrictStaticZeroVelocityStrongMaxReprojectionRmsePx();
            const bool nonzeroUnreliableMotion =
                rawSvdMotionNonZero &&
                EnableStrictSvdMotionReliability() &&
                !svdMotionReliable;
            if(nonzeroUnreliableMotion && !zeroMotionVeryStrong)
            {
                strictStaticZeroVelocityReason =
                    "nonzero_unreliable_svd_motion_kept_uncertain";
                velocity = significantCentroidMotion ? translationOnlyVelocity : rawSvdVelocity;
                initGate.valid = true;
                initGate.state = kInstanceMotionUncertain;
                initGate.useDynamicMotion = false;
                nonzeroUnreliableMotionInitializedAsUncertain = true;
                if(initGate.support <= 0)
                    initGate.support = zeroMotionInitGate.valid ?
                        zeroMotionInitGate.support : static_cast<int>(pointsTau2.size());
            }
            else if(staticPreferredByReprojection &&
               significantCentroidMotion &&
               !smallSvdMotion)
            {
                strictStaticZeroVelocityReason =
                    "zero_motion_reprojection_preferred_but_centroid_motion_uncertain";
                velocity = translationOnlyVelocity;
                initGate.valid = true;
                initGate.state = kInstanceMotionUncertain;
                initGate.useDynamicMotion = false;
                if(initGate.support <= 0)
                    initGate.support = zeroMotionInitGate.valid ?
                        zeroMotionInitGate.support : static_cast<int>(pointsTau2.size());
            }
            else if(staticPreferredByReprojection || smallSvdMotion)
            {
                strictStaticZeroVelocityInit = true;
                strictStaticZeroVelocityReason =
                    staticPreferredByReprojection ? "zero_motion_reprojection_preferred" :
                                                    "small_svd_motion";
                velocity = Sophus::SE3f();
                initGate.valid = true;
                initGate.state = kInstanceMotionStatic;
                initGate.useDynamicMotion = false;
                if(initGate.support <= 0)
                    initGate.support = zeroMotionInitGate.valid ?
                        zeroMotionInitGate.support : static_cast<int>(pointsTau2.size());
            }
            else if(initGate.valid && initGate.state == kInstanceMotionStatic &&
                    !zeroMotionReprojectionQualityOk)
            {
                strictStaticZeroVelocityReason = "zero_motion_reprojection_rejected";
                initGate.state = kInstanceMotionUncertain;
                initGate.useDynamicMotion = false;
            }
        }

        if(initGate.valid && initGate.state == kInstanceMotionDynamic)
            initGate.useDynamicMotion = !IsNearlyIdentityInstanceMotion(velocity);
        else if(initGate.valid && initGate.state == kInstanceMotionStatic)
        {
            initGate.useDynamicMotion = false;
            velocity = Sophus::SE3f();
        }
    }
    const InstanceMotionGateState initGateStateBeforeSvdSuppression = initGate.state;
    const bool initGateUseDynamicBeforeSvdSuppression = initGate.useDynamicMotion;
    if(strictPaperInitialization &&
       initGate.state == kInstanceMotionDynamic &&
       EnableStrictSvdMotionReliability() &&
       !svdMotionReliable)
    {
        unreliableSvdMotionSuppressed = true;
        initGate.useDynamicMotion = false;
        initGate.state = kInstanceMotionUncertain;
    }
    if(!initGate.valid || initGate.state == kInstanceMotionUncertain)
    {
        if(strictPaperInitialization && !pointsTau2.empty())
        {
            if(!initGate.valid)
            {
                initGate.valid = true;
                initGate.support = static_cast<int>(pointsTau2.size());
            }
            initGate.state = kInstanceMotionUncertain;
            initGate.useDynamicMotion = false;
        }
        else
        {
        if(debugInitForInstance || debugLifecycleForInstance || DebugFocusFrame(mCurrentFrame.mnId))
        {
            const Eigen::AngleAxisf angleAxis(velocity.rotationMatrix());
            std::cout << "[STSLAM_INSTANCE_INIT] frame=" << mCurrentFrame.mnId
                      << " instance_id=" << instanceId
                      << " rejected=motion_residual_self_check"
                      << " map_id=" << currentMapId
                      << " map_origin_frame=" << currentMapOriginFrame
                      << " map_origin_kf=" << currentMapOriginKF
                      << " map_age_frames=" << currentMapAgeFrames
                      << " tau2_frame=" << frameTau2.mnId
                      << " tau1_frame=" << frameTau1.mnId
                      << " tau_frame=" << mCurrentFrame.mnId
                      << " state=" << InstanceMotionGateStateName(initGate.state)
                      << " support=" << initGate.support
                      << " static_mean_error=" << initGate.staticMeanError
	                      << " dynamic_mean_error=" << initGate.dynamicMeanError
	                      << " dynamic_rmse_px="
	                      << (initGate.dynamicMeanError > 0.0 && std::isfinite(initGate.dynamicMeanError) ?
	                          std::sqrt(initGate.dynamicMeanError) : 0.0)
	                      << " zero_motion_state=" << InstanceMotionGateStateName(zeroMotionInitGate.state)
	                      << " zero_motion_support=" << zeroMotionInitGate.support
	                      << " zero_motion_mean_error=" << zeroMotionInitGate.staticMeanError
	                      << " svd_motion_mean_error=" << zeroMotionInitGate.dynamicMeanError
	                      << " zero_motion_rmse_px="
	                      << (zeroMotionInitGate.staticMeanError > 0.0 &&
	                          std::isfinite(zeroMotionInitGate.staticMeanError) ?
	                          std::sqrt(zeroMotionInitGate.staticMeanError) : 0.0)
	                      << " svd_motion_rmse_px="
	                      << (zeroMotionInitGate.dynamicMeanError > 0.0 &&
	                          std::isfinite(zeroMotionInitGate.dynamicMeanError) ?
	                          std::sqrt(zeroMotionInitGate.dynamicMeanError) : 0.0)
	                      << " dynamic_quality_rejected=" << (initDynamicQualityRejected ? 1 : 0)
	                      << " dynamic_rotation_rejected=" << (initDynamicRotationRejected ? 1 : 0)
	                      << " max_dynamic_rmse_px=" << GetInstanceInitializationMaxDynamicRmsePx()
	                      << " max_dynamic_rotation_deg=" << GetInstanceInitializationMaxDynamicRotationDeg()
	                      << " strict_static_zero_reason=" << strictStaticZeroVelocityReason
	                      << " strict_static_zero_max_reproj_rmse_px="
	                      << GetStrictStaticZeroVelocityMaxReprojectionRmsePx()
	                      << " checked=" << initGateChecked
                      << " invalid_index=" << initGateInvalidIndex
                      << " nonfinite_point=" << initGateNonFinitePoint
                      << " nonpositive_depth=" << initGateNonPositiveDepth
                      << " nonfinite_projection=" << initGateNonFiniteProjection
                      << " translation_only_enabled=" << (EnableInstanceInitializationTranslationOnlyCandidate() ? 1 : 0)
                      << " translation_only_state=" << InstanceMotionGateStateName(translationOnlyInitGate.state)
                      << " translation_only_support=" << translationOnlyInitGate.support
                      << " translation_only_mean_error=" << translationOnlyInitGate.dynamicMeanError
                      << " translation_only_rmse_px="
                      << (translationOnlyInitGate.dynamicMeanError > 0.0 &&
                          std::isfinite(translationOnlyInitGate.dynamicMeanError) ?
                          std::sqrt(translationOnlyInitGate.dynamicMeanError) : 0.0)
                      << " translation_only_quality_rejected="
                      << (translationOnlyDynamicQualityRejected ? 1 : 0)
                      << " translation_only_rotation_rejected="
                      << (translationOnlyDynamicRotationRejected ? 1 : 0)
                      << " translation_only_static_mean_error=" << translationOnlyInitGate.staticMeanError
                      << " translation_only_nonpositive_depth=" << translationOnlyNonPositiveDepth
                      << " translation_only_depth_min="
                      << (translationOnlyMinDynamicDepth < 1e29f ? translationOnlyMinDynamicDepth : 0.0f)
                      << " translation_only_depth_max="
                      << (translationOnlyMaxDynamicDepth > -1e29f ? translationOnlyMaxDynamicDepth : 0.0f)
                      << " translation_only_norm=" << translationOnlyVelocity.translation().norm()
                      << " static_depth_min=" << initGateMinStaticDepth
                      << " static_depth_max=" << initGateMaxStaticDepth
                      << " dynamic_depth_min=" << initGateMinDynamicDepth
                      << " dynamic_depth_max=" << initGateMaxDynamicDepth
                      << " mean_triangulation_weight="
                      << (triangulationWeights.empty() ? 0.0 : activeTriangulationWeightSum / static_cast<double>(triangulationWeights.size()))
                      << " triframe_candidate_quality="
                      << (EnableTriFrameTrackletCandidateQuality() ? 1 : 0)
                      << " triframe_candidate_checked=" << triframeCandidateChecked
                      << " triframe_candidate_svd_seed=" << seedPointsTau2.size()
                      << " triframe_candidate_seed_svd_used=" << (useCandidateSeedSvd ? 1 : 0)
                      << " triframe_candidate_downweighted=" << triframeCandidateDownweighted
                      << " triframe_candidate_structure_rejected=" << triframeCandidateStructureRejected
                      << " triframe_candidate_low_depth=" << triframeCandidateLowDepth
                      << " triframe_candidate_high_reprojection=" << triframeCandidateHighReprojection
                      << " triframe_candidate_high_flow_accel=" << triframeCandidateHighFlowAcceleration
                      << " triframe_candidate_mean_weight="
                      << (triframeCandidateChecked > 0 ?
                          triframeCandidateWeightSum / static_cast<double>(triframeCandidateChecked) : 0.0)
                      << " triframe_candidate_mean_pair_reproj="
                      << (triframeCandidateChecked > 0 ?
                          triframeCandidatePairReprojectionSum / static_cast<double>(triframeCandidateChecked) : 0.0)
                      << " triframe_candidate_mean_flow_accel_px="
                      << (triframeCandidateChecked > 0 ?
                          triframeCandidateFlowAccelerationSum / static_cast<double>(triframeCandidateChecked) : 0.0)
                      << " triframe_candidate_mean_flow_accel_norm="
                      << (triframeCandidateChecked > 0 ?
                          triframeCandidateFlowAccelerationNormSum / static_cast<double>(triframeCandidateChecked) : 0.0)
                      << " triframe_multi_pair="
                      << (EnableTriFrameMultiPairTriangulation() ? 1 : 0)
                      << " triframe_multi_pair_checked=" << triframeMultiPairChecked
                      << " triframe_multi_pair_recovered=" << triframeMultiPairRecovered
                      << " triframe_multi_pair_recovered_tau2_tau1=" << triframeMultiPairRecoveredTau2Tau1
                      << " triframe_multi_pair_recovered_tau1_tau=" << triframeMultiPairRecoveredTau1Tau
                      << " triframe_multi_pair_recovered_tau2_tau=" << triframeMultiPairRecoveredTau2Tau
                      << " triframe_multi_pair_rejected_third_view=" << triframeMultiPairRejectedThirdView
                      << " triframe_multi_pair_mean_third_reproj="
                      << (triframeMultiPairRecovered > 0 ?
                          triframeMultiPairThirdViewReprojectionSum / static_cast<double>(triframeMultiPairRecovered) : 0.0)
                      << " triframe_multi_pair_max_third_reproj=" << triframeMultiPairThirdViewReprojectionMax
                      << " triframe_consistency_enabled="
                      << (EnableTriFrameConsistencyWeights() ? 1 : 0)
                      << " triframe_checked=" << triframeConsistencyChecked
                      << " triframe_bad=" << triframeConsistencyBad
                      << " triframe_downweighted=" << triframeConsistencyDownweighted
                      << " triframe_negative_depth=" << triframeConsistencyNegativeDepth
                      << " triframe_structure_promotion_gate="
                      << (EnableTriFrameConsistencyStructurePromotionGate() ? 1 : 0)
                      << " triframe_structure_rejected=" << triframeStructurePromotionRejected
                      << " triframe_mean_weight="
                      << (triframeConsistencyChecked > 0 ?
                          triframeConsistencyWeightSum / static_cast<double>(triframeConsistencyChecked) : 0.0)
                      << " triframe_mean_reprojection_error="
                      << (triframeConsistencyChecked > 0 ?
                          triframeConsistencyReprojectionSum / static_cast<double>(triframeConsistencyChecked) : 0.0)
                      << " triframe_max_reprojection_error=" << triframeConsistencyMaxReprojection
                      << " triframe_mean_dynamic_reprojection_error="
                      << (triframeConsistencyChecked > 0 ?
                          triframeConsistencyDynamicReprojectionSum / static_cast<double>(triframeConsistencyChecked) : 0.0)
                      << " triframe_mean_motion3d_residual="
                      << (triframeConsistencyChecked > 0 ?
                          triframeConsistencyMotion3dResidualSum / static_cast<double>(triframeConsistencyChecked) : 0.0)
                      << " low_depth_tracklets=" << lowDepthTracklets
                      << " high_reprojection_tracklets=" << highReprojectionTracklets
                      << " low_parallax_tracklets=" << lowParallaxTracklets
                      << " high_parallax_tracklets=" << highParallaxTracklets
                      << " high_disparity_tracklets=" << highDisparityTracklets
                      << " min_triangulation_weight="
                      << (std::isfinite(minTriangulationWeight) ? minTriangulationWeight : 0.0)
                      << " max_triangulation_weight=" << maxTriangulationWeight
                      << " min_triangulation_depth="
                      << (std::isfinite(minTriangulationDepth) ? minTriangulationDepth : 0.0)
                      << " mean_reprojection_error="
                      << (reprojectionSamples > 0 ? reprojectionSum / static_cast<double>(reprojectionSamples) : 0.0)
                      << " max_reprojection_error=" << maxTriangulationReprojection
                      << " mean_parallax_deg="
                      << (parallaxSamples > 0 ? parallaxSum / static_cast<double>(parallaxSamples) : 0.0)
                      << " min_parallax_deg="
                      << (std::isfinite(minTriangulationParallax) ? minTriangulationParallax : 0.0)
                      << " max_parallax_deg=" << maxTriangulationParallax
                      << " mean_disparity_tau2_tau1="
                      << (pointsTau2.empty() ? 0.0 : disparityTau2Tau1Sum / static_cast<double>(pointsTau2.size()))
                      << " mean_disparity_tau1_tau="
                      << (pointsTau2.empty() ? 0.0 : disparityTau1TauSum / static_cast<double>(pointsTau2.size()))
                      << " max_disparity_tau2_tau1=" << maxDisparityTau2Tau1
                      << " max_disparity_tau1_tau=" << maxDisparityTau1Tau
                      << " translation_norm=" << velocity.translation().norm()
                      << " rotation_deg=" << std::abs(angleAxis.angle()) * 180.0 / 3.14159265358979323846
                      << std::endl;
        }
        return false;
        }
    }

    const bool initAsStatic = (initGate.state == kInstanceMotionStatic);
    if(initAsStatic)
        velocity = Sophus::SE3f();
    const Instance::DynamicEntityMotionState initialDynamicEntityState =
        initGate.state == kInstanceMotionDynamic ? Instance::kMovingDynamicEntity :
        (initGate.state == kInstanceMotionStatic ? Instance::kZeroVelocityDynamicEntity :
                                                   Instance::kUncertainDynamicEntity);
    const bool uncertainNonzeroInitialization =
        initialDynamicEntityState == Instance::kUncertainDynamicEntity &&
        !IsSmallInstanceMotion(velocity,
                               GetStrictStaticZeroVelocityMaxTranslation(),
                               GetStrictStaticZeroVelocityMaxRotationDeg());
    const double initialMotionConfidence =
        initGate.state == kInstanceMotionDynamic ? (svdMotionReliable ? 0.85 : 0.55) :
        (initGate.state == kInstanceMotionStatic ? 0.85 :
         (uncertainNonzeroInitialization ? 0.45 : 0.35));

    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for(size_t i = 0; i < pointsTau1.size(); ++i)
    {
        const double qualityWeight =
            (i < triangulationWeights.size() && std::isfinite(triangulationWeights[i])) ?
            std::max(0.0, triangulationWeights[i]) : 1.0;
        centroid += static_cast<float>(qualityWeight) * pointsTau1[i];
    }
    if(activeTriangulationWeightSum <= 0.0)
        return false;
    centroid /= static_cast<float>(activeTriangulationWeightSum);
    if(!centroid.allFinite())
        return false;

    const Sophus::SE3f initialPose(Eigen::Matrix3f::Identity(), centroid);
    if(!IsFiniteSE3(initialPose))
        return false;
    const bool initializationMotionReliable =
        initGate.state == kInstanceMotionStatic ||
        (!strictPaperInitialization && initGate.state == kInstanceMotionDynamic) ||
        (strictPaperInitialization && initGate.state == kInstanceMotionDynamic && svdMotionReliable);
    if(!pInstance->SetInitializedMotionState(velocity,
                                             initialPose,
                                             pointsTau1,
                                             static_cast<int>(mCurrentFrame.mnId),
                                             initializationMotionReliable,
                                             initialDynamicEntityState,
                                             initialMotionConfidence))
        return false;
    pInstance->RecordMotionGateState(static_cast<int>(initGate.state),
                                     velocity,
                                     static_cast<int>(mCurrentFrame.mnId),
                                     static_cast<float>(GetInstanceStaticVelocityDecay()));

    if(debugInitForInstance || debugLifecycleForInstance || DebugFocusFrame(mCurrentFrame.mnId))
    {
        const Eigen::AngleAxisf angleAxis(velocity.rotationMatrix());
        std::cout << "[STSLAM_INSTANCE_INIT] frame=" << mCurrentFrame.mnId
                  << " instance_id=" << instanceId
                  << " accepted=1"
                  << " map_id=" << currentMapId
                  << " map_origin_frame=" << currentMapOriginFrame
                  << " map_origin_kf=" << currentMapOriginKF
                  << " map_age_frames=" << currentMapAgeFrames
                  << " tau2_frame=" << frameTau2.mnId
                  << " tau1_frame=" << frameTau1.mnId
                  << " tau_frame=" << mCurrentFrame.mnId
                  << " init_state=" << InstanceMotionGateStateName(initGate.state)
                  << " dynamic_entity_state="
                  << DynamicEntityMotionStateName(initialDynamicEntityState)
                  << " initial_motion_confidence=" << initialMotionConfidence
                  << " init_state_before_svd_suppression="
                  << InstanceMotionGateStateName(initGateStateBeforeSvdSuppression)
                  << " use_dynamic_before_svd_suppression="
                  << (initGateUseDynamicBeforeSvdSuppression ? 1 : 0)
                  << " use_dynamic_after_svd_suppression="
                  << (initGate.useDynamicMotion ? 1 : 0)
                  << " strict_static_zero_velocity=" << (strictStaticZeroVelocityInit ? 1 : 0)
	                  << " strict_static_zero_reason=" << strictStaticZeroVelocityReason
	                  << " strict_static_zero_max_translation=" << GetStrictStaticZeroVelocityMaxTranslation()
	                  << " strict_static_zero_max_rotation_deg=" << GetStrictStaticZeroVelocityMaxRotationDeg()
	                  << " strict_static_zero_max_reproj_rmse_px="
	                  << GetStrictStaticZeroVelocityMaxReprojectionRmsePx()
	                  << " tracklets=" << static_cast<int>(tracklets.size())
                  << " triangulated=" << static_cast<int>(pointsTau2.size())
                  << " inlier_ratio=" << inlierRatio
                  << " mean_inlier_svd_residual=" << meanInlierResidual
                  << " svd_motion_reliable=" << (svdMotionReliable ? 1 : 0)
                  << " svd_motion_reliability_reason=" << svdMotionReliabilityReason
                  << " unreliable_svd_motion_suppressed=" << (unreliableSvdMotionSuppressed ? 1 : 0)
                  << " nonzero_unreliable_motion_initialized_as_uncertain="
                  << (nonzeroUnreliableMotionInitializedAsUncertain ? 1 : 0)
                  << " initialization_motion_reliable=" << (initializationMotionReliable ? 1 : 0)
                  << " selected_velocity_identity="
                  << (IsNearlyIdentityInstanceMotion(velocity) ? 1 : 0)
                  << " raw_svd_translation_norm=" << rawSvdVelocity.translation().norm()
                  << " raw_svd_rotation_deg=" << InstanceRotationAngleDeg(rawSvdVelocity.rotationMatrix())
                  << " svd_diag_valid=" << (svdMotionDiagnostics.valid ? 1 : 0)
                  << " svd_diag_support=" << svdMotionDiagnostics.support
                  << " svd_total_weight=" << svdMotionDiagnostics.totalWeight
                  << " svd_singular0=" << svdMotionDiagnostics.singular0
                  << " svd_singular1=" << svdMotionDiagnostics.singular1
                  << " svd_singular2=" << svdMotionDiagnostics.singular2
                  << " svd_condition=" << svdMotionDiagnostics.condition
                  << " svd_planarity_ratio=" << svdMotionDiagnostics.planarityRatio
                  << " svd_linearity_ratio=" << svdMotionDiagnostics.linearityRatio
                  << " svd_src_std_radius=" << svdMotionDiagnostics.srcStdRadius
                  << " svd_dst_std_radius=" << svdMotionDiagnostics.dstStdRadius
                  << " svd_mean_displacement=" << svdMotionDiagnostics.meanDisplacement
                  << " svd_median_displacement=" << svdMotionDiagnostics.medianDisplacement
                  << " svd_max_displacement=" << svdMotionDiagnostics.maxDisplacement
                  << " svd_mean_3d_residual=" << svdMotionDiagnostics.meanResidual
                  << " svd_median_3d_residual=" << svdMotionDiagnostics.medianResidual
                  << " svd_max_3d_residual=" << svdMotionDiagnostics.maxResidual
                  << " mean_triangulation_weight="
                  << (triangulationWeights.empty() ? 0.0 : activeTriangulationWeightSum / static_cast<double>(triangulationWeights.size()))
                  << " triframe_candidate_quality="
                  << (EnableTriFrameTrackletCandidateQuality() ? 1 : 0)
                  << " triframe_candidate_checked=" << triframeCandidateChecked
                  << " triframe_candidate_svd_seed=" << seedPointsTau2.size()
                  << " triframe_candidate_seed_svd_used=" << (useCandidateSeedSvd ? 1 : 0)
                  << " triframe_candidate_downweighted=" << triframeCandidateDownweighted
                  << " triframe_candidate_structure_rejected=" << triframeCandidateStructureRejected
                  << " triframe_candidate_low_depth=" << triframeCandidateLowDepth
                  << " triframe_candidate_high_reprojection=" << triframeCandidateHighReprojection
                  << " triframe_candidate_high_flow_accel=" << triframeCandidateHighFlowAcceleration
                  << " triframe_candidate_mean_weight="
                  << (triframeCandidateChecked > 0 ?
                      triframeCandidateWeightSum / static_cast<double>(triframeCandidateChecked) : 0.0)
                  << " triframe_candidate_mean_pair_reproj="
                  << (triframeCandidateChecked > 0 ?
                      triframeCandidatePairReprojectionSum / static_cast<double>(triframeCandidateChecked) : 0.0)
                  << " triframe_candidate_mean_flow_accel_px="
                  << (triframeCandidateChecked > 0 ?
                      triframeCandidateFlowAccelerationSum / static_cast<double>(triframeCandidateChecked) : 0.0)
                  << " triframe_candidate_mean_flow_accel_norm="
                  << (triframeCandidateChecked > 0 ?
                      triframeCandidateFlowAccelerationNormSum / static_cast<double>(triframeCandidateChecked) : 0.0)
                  << " triframe_multi_pair="
                  << (EnableTriFrameMultiPairTriangulation() ? 1 : 0)
                  << " triframe_multi_pair_checked=" << triframeMultiPairChecked
                  << " triframe_multi_pair_recovered=" << triframeMultiPairRecovered
                  << " triframe_multi_pair_recovered_tau2_tau1=" << triframeMultiPairRecoveredTau2Tau1
                  << " triframe_multi_pair_recovered_tau1_tau=" << triframeMultiPairRecoveredTau1Tau
                  << " triframe_multi_pair_recovered_tau2_tau=" << triframeMultiPairRecoveredTau2Tau
                  << " triframe_multi_pair_rejected_third_view=" << triframeMultiPairRejectedThirdView
                  << " triframe_multi_pair_mean_third_reproj="
                  << (triframeMultiPairRecovered > 0 ?
                      triframeMultiPairThirdViewReprojectionSum / static_cast<double>(triframeMultiPairRecovered) : 0.0)
                  << " triframe_multi_pair_max_third_reproj=" << triframeMultiPairThirdViewReprojectionMax
                  << " triframe_consistency_enabled="
                  << (EnableTriFrameConsistencyWeights() ? 1 : 0)
                  << " triframe_checked=" << triframeConsistencyChecked
                  << " triframe_bad=" << triframeConsistencyBad
                  << " triframe_downweighted=" << triframeConsistencyDownweighted
                  << " triframe_negative_depth=" << triframeConsistencyNegativeDepth
                  << " triframe_structure_promotion_gate="
                  << (EnableTriFrameConsistencyStructurePromotionGate() ? 1 : 0)
                  << " triframe_structure_rejected=" << triframeStructurePromotionRejected
                  << " triframe_mean_weight="
                  << (triframeConsistencyChecked > 0 ?
                      triframeConsistencyWeightSum / static_cast<double>(triframeConsistencyChecked) : 0.0)
                  << " triframe_mean_reprojection_error="
                  << (triframeConsistencyChecked > 0 ?
                      triframeConsistencyReprojectionSum / static_cast<double>(triframeConsistencyChecked) : 0.0)
                  << " triframe_max_reprojection_error=" << triframeConsistencyMaxReprojection
                  << " triframe_mean_dynamic_reprojection_error="
                  << (triframeConsistencyChecked > 0 ?
                      triframeConsistencyDynamicReprojectionSum / static_cast<double>(triframeConsistencyChecked) : 0.0)
                  << " triframe_mean_motion3d_residual="
                  << (triframeConsistencyChecked > 0 ?
                      triframeConsistencyMotion3dResidualSum / static_cast<double>(triframeConsistencyChecked) : 0.0)
                  << " low_depth_tracklets=" << lowDepthTracklets
                  << " high_reprojection_tracklets=" << highReprojectionTracklets
                  << " low_parallax_tracklets=" << lowParallaxTracklets
                  << " high_parallax_tracklets=" << highParallaxTracklets
                  << " high_disparity_tracklets=" << highDisparityTracklets
                  << " min_triangulation_weight="
                  << (std::isfinite(minTriangulationWeight) ? minTriangulationWeight : 0.0)
                  << " max_triangulation_weight=" << maxTriangulationWeight
                  << " min_triangulation_depth="
                  << (std::isfinite(minTriangulationDepth) ? minTriangulationDepth : 0.0)
                  << " mean_reprojection_error="
                  << (reprojectionSamples > 0 ? reprojectionSum / static_cast<double>(reprojectionSamples) : 0.0)
                  << " max_reprojection_error=" << maxTriangulationReprojection
                  << " mean_parallax_deg="
                  << (parallaxSamples > 0 ? parallaxSum / static_cast<double>(parallaxSamples) : 0.0)
                  << " min_parallax_deg="
                  << (std::isfinite(minTriangulationParallax) ? minTriangulationParallax : 0.0)
                  << " max_parallax_deg=" << maxTriangulationParallax
                  << " mean_disparity_tau2_tau1="
                  << (pointsTau2.empty() ? 0.0 : disparityTau2Tau1Sum / static_cast<double>(pointsTau2.size()))
                  << " mean_disparity_tau1_tau="
                  << (pointsTau2.empty() ? 0.0 : disparityTau1TauSum / static_cast<double>(pointsTau2.size()))
                  << " max_disparity_tau2_tau1=" << maxDisparityTau2Tau1
                  << " max_disparity_tau1_tau=" << maxDisparityTau1Tau
                  << " static_mean_error=" << initGate.staticMeanError
                  << " dynamic_mean_error=" << initGate.dynamicMeanError
                  << " dynamic_rmse_px="
                  << (initGate.dynamicMeanError > 0.0 && std::isfinite(initGate.dynamicMeanError) ?
                      std::sqrt(initGate.dynamicMeanError) : 0.0)
                  << " zero_motion_state=" << InstanceMotionGateStateName(zeroMotionInitGate.state)
                  << " zero_motion_support=" << zeroMotionInitGate.support
                  << " zero_motion_mean_error=" << zeroMotionInitGate.staticMeanError
                  << " svd_motion_mean_error=" << zeroMotionInitGate.dynamicMeanError
                  << " zero_motion_rmse_px="
                  << (zeroMotionInitGate.staticMeanError > 0.0 &&
                      std::isfinite(zeroMotionInitGate.staticMeanError) ?
                      std::sqrt(zeroMotionInitGate.staticMeanError) : 0.0)
                  << " svd_motion_rmse_px="
                  << (zeroMotionInitGate.dynamicMeanError > 0.0 &&
                      std::isfinite(zeroMotionInitGate.dynamicMeanError) ?
                      std::sqrt(zeroMotionInitGate.dynamicMeanError) : 0.0)
                  << " dynamic_quality_rejected=" << (initDynamicQualityRejected ? 1 : 0)
                  << " dynamic_rotation_rejected=" << (initDynamicRotationRejected ? 1 : 0)
                  << " max_dynamic_rmse_px=" << GetInstanceInitializationMaxDynamicRmsePx()
                  << " max_dynamic_rotation_deg=" << GetInstanceInitializationMaxDynamicRotationDeg()
                  << " selected_translation_only=" << (useTranslationOnlyInit ? 1 : 0)
                  << " translation_only_enabled=" << (EnableInstanceInitializationTranslationOnlyCandidate() ? 1 : 0)
                  << " translation_only_state=" << InstanceMotionGateStateName(translationOnlyInitGate.state)
                  << " translation_only_support=" << translationOnlyInitGate.support
                  << " translation_only_static_mean_error=" << translationOnlyInitGate.staticMeanError
                  << " translation_only_mean_error=" << translationOnlyInitGate.dynamicMeanError
                  << " translation_only_rmse_px="
                  << (translationOnlyInitGate.dynamicMeanError > 0.0 &&
                      std::isfinite(translationOnlyInitGate.dynamicMeanError) ?
                      std::sqrt(translationOnlyInitGate.dynamicMeanError) : 0.0)
                  << " translation_only_quality_rejected="
                  << (translationOnlyDynamicQualityRejected ? 1 : 0)
                  << " translation_only_rotation_rejected="
                  << (translationOnlyDynamicRotationRejected ? 1 : 0)
                  << " translation_only_nonpositive_depth=" << translationOnlyNonPositiveDepth
                  << " translation_only_depth_min="
                  << (translationOnlyMinDynamicDepth < 1e29f ? translationOnlyMinDynamicDepth : 0.0f)
                  << " translation_only_depth_max="
                  << (translationOnlyMaxDynamicDepth > -1e29f ? translationOnlyMaxDynamicDepth : 0.0f)
                  << " translation_only_norm=" << translationOnlyVelocity.translation().norm()
                  << " translation_norm=" << velocity.translation().norm()
                  << " rotation_deg=" << std::abs(angleAxis.angle()) * 180.0 / 3.14159265358979323846
                  << std::endl;
    }

    int bufferedDynamicTrackletPoints = 0;
    int frameTau2DynamicObservationAppends = 0;
    int frameTau1DynamicObservationAppends = 0;
    int frameTauDynamicObservationAppends = 0;
    int snapshotTau2DynamicObservationAppends = 0;
    int snapshotTau1DynamicObservationAppends = 0;
    int snapshotTauDynamicObservationAppends = 0;
    int rejectedDynamicTrackletsInvalidCurrentIndex = 0;
    int sharedBackendTrackletPoints = 0;
    int structureTrackletPointsBuffered = 0;
    int structureTrackletPointsRegisteredToMap = 0;
    int triframeRejectedStructureTrackletPoints = 0;
    int triframeRejectedBackendObservationTrackletPoints = 0;
    const bool trackletsAsStructurePoints =
        EnableInstanceTrackletStructurePoints();
    const bool registerStructurePointsToMap =
        trackletsAsStructurePoints &&
        RegisterInstanceStructurePointsToMap() &&
        ((mSensor != System::RGBD && mSensor != System::IMU_RGBD) ||
         RegisterRgbdInstanceStructurePointsToMap());
    std::vector<MapPoint*> vBufferedTrackletBackendPoints;
    if(mpAtlas)
    {
        Map* pCurrentMap = mpAtlas->GetCurrentMap();

        auto appendDynamicObservationToFrame = [&](Frame& frame,
                                                   const int featureIdx,
                                                   MapPoint* pMP,
                                                   const Eigen::Vector3f& pointWorld,
                                                   const double qualityWeight) -> bool
        {
            if(!pMP || featureIdx < 0 || featureIdx >= static_cast<int>(frame.mvKeysUn.size()))
                return false;

            DynamicInstancePointObservation observation;
            observation.instanceId = pInstance->GetId();
            observation.semanticLabel = pInstance->GetSemanticLabel();
            observation.featureIdx = featureIdx;
            observation.pointWorld = pointWorld;
            observation.qualityWeight = qualityWeight;
            observation.pBackendPoint = pMP;
            frame.mvDynamicInstancePointObservations.push_back(observation);
            frame.mmPredictedInstanceMotions[pInstance->GetId()] = velocity;
            return true;
        };

        auto appendDynamicObservationToSnapshot = [&](const unsigned long frameId,
                                                      const int featureIdx,
                                                      MapPoint* pMP,
                                                      const Eigen::Vector3f& pointWorld,
                                                      const double qualityWeight) -> int
        {
            int nAppended = 0;
            for(size_t i = 0; i < mvFramesSinceLastKeyFrame.size(); ++i)
            {
                WindowFrameSnapshot& snapshot = mvFramesSinceLastKeyFrame[i];
                if(snapshot.mnFrameId != frameId ||
                   featureIdx < 0 ||
                   featureIdx >= static_cast<int>(snapshot.mvKeysUn.size()))
                {
                    continue;
                }

                DynamicInstancePointObservation observation;
                observation.instanceId = pInstance->GetId();
                observation.semanticLabel = pInstance->GetSemanticLabel();
                observation.featureIdx = featureIdx;
                observation.pointWorld = pointWorld;
                observation.qualityWeight = qualityWeight;
                observation.pBackendPoint = pMP;
                snapshot.mvDynamicInstancePointObservations.push_back(observation);
                snapshot.mmPredictedInstanceMotions[pInstance->GetId()] = velocity;
                ++nAppended;
            }
            return nAppended;
        };

        if(pCurrentMap && mRecentPanopticFrames.size() >= 2)
        {
            Frame& mutableFrameTau2 = mRecentPanopticFrames[mRecentPanopticFrames.size() - 2];
            Frame& mutableFrameTau1 = mRecentPanopticFrames.back();
            const unsigned long frameTau2Id = mutableFrameTau2.mnId;
            const unsigned long frameTau1Id = mutableFrameTau1.mnId;
            const unsigned long frameTauId = mCurrentFrame.mnId;
            const size_t nBufferedTracklets =
                std::min(pointsTau1.size(), acceptedTracklets.size());
            vBufferedTrackletBackendPoints.assign(nBufferedTracklets, static_cast<MapPoint*>(NULL));

            for(size_t i = 0; i < nBufferedTracklets; ++i)
            {
                const std::array<int, 3>& tracklet = acceptedTracklets[i];
                const int currentIdx = tracklet[2];
                if(currentIdx < 0 ||
                   currentIdx >= static_cast<int>(mCurrentFrame.mvKeysUn.size()))
                {
                    ++rejectedDynamicTrackletsInvalidCurrentIndex;
                    continue;
                }

                // pointsTau1 is triangulated from (tau-1, tau), so it already
                // represents the current-frame instance point used in Eq. (16).
                // Applying the tau-1->tau velocity again would move it to tau+1
                // and corrupt the shape/rigidity factors initialized here.
                const Eigen::Vector3f pointTau = pointsTau1[i];
                if(!pointTau.allFinite())
                {
                    ++rejectedDynamicTrackletsInvalidCurrentIndex;
                    continue;
                }
                const bool backendObservationEligible =
                    (i >= triframeBackendObservationEligibility.size()) ||
                    triframeBackendObservationEligibility[i];
                if(!backendObservationEligible)
                {
                    ++triframeRejectedBackendObservationTrackletPoints;
                    continue;
                }

                MapPoint* pBackendPoint =
                    new MapPoint(pointsTau1[i], pCurrentMap, &mCurrentFrame, currentIdx);
                const double qualityWeight =
                    (i < triangulationWeights.size() && std::isfinite(triangulationWeights[i])) ?
                    std::max(0.05, std::min(1.0, triangulationWeights[i])) : 1.0;
                const bool triframeStructureEligible =
                    (i >= triframeStructureEligibility.size()) ||
                    triframeStructureEligibility[i];
                const bool promoteAsStructurePoint =
                    trackletsAsStructurePoints && triframeStructureEligible;
                if(trackletsAsStructurePoints && !triframeStructureEligible)
                    ++triframeRejectedStructureTrackletPoints;
                pBackendPoint->SetLifecycleType(promoteAsStructurePoint ?
                                                MapPoint::kInstanceStructurePoint :
                                                MapPoint::kDynamicInstanceObservationPoint);
                pBackendPoint->SetInstanceId(pInstance->GetId());
                pBackendPoint->SetSemanticLabel(pInstance->GetSemanticLabel());
                pBackendPoint->UpdateObservationStats(static_cast<int>(frameTau2Id));
                pBackendPoint->UpdateObservationStats(static_cast<int>(frameTau1Id));
                pBackendPoint->UpdateObservationStats(static_cast<int>(frameTauId));
                if(promoteAsStructurePoint && registerStructurePointsToMap)
                {
                    pCurrentMap->AddMapPoint(pBackendPoint);
                    ++structureTrackletPointsRegisteredToMap;
                }
                BindMapPointToInstance(pBackendPoint,
                                       pInstance->GetId(),
                                       pInstance->GetSemanticLabel(),
                                       false);
                if(promoteAsStructurePoint)
                {
                    pInstance->AddMapPoint(pBackendPoint);
                    pInstance->SetStructureLocalPoint(pBackendPoint, pointTau - centroid);
                    ++structureTrackletPointsBuffered;
                }
                vBufferedTrackletBackendPoints[i] = pBackendPoint;

                pInstance->SetObservationQualityWeight(frameTau2Id, pBackendPoint, qualityWeight);
                pInstance->SetObservationQualityWeight(frameTau1Id, pBackendPoint, qualityWeight);
                pInstance->SetObservationQualityWeight(frameTauId, pBackendPoint, qualityWeight);
                pInstance->AddDynamicObservation(frameTau2Id, tracklet[0], pBackendPoint, pointsTau2[i], qualityWeight);
                pInstance->AddDynamicObservation(frameTau1Id, tracklet[1], pBackendPoint, pointsTau1[i], qualityWeight);
                pInstance->AddDynamicObservation(frameTauId, tracklet[2], pBackendPoint, pointTau, qualityWeight);

                const bool appendedTau2 =
                    appendDynamicObservationToFrame(mutableFrameTau2, tracklet[0], pBackendPoint, pointsTau2[i], qualityWeight);
                const bool appendedTau1 =
                    appendDynamicObservationToFrame(mutableFrameTau1, tracklet[1], pBackendPoint, pointsTau1[i], qualityWeight);
                const bool appendedTau =
                    appendDynamicObservationToFrame(mCurrentFrame, tracklet[2], pBackendPoint, pointTau, qualityWeight);
                frameTau2DynamicObservationAppends += appendedTau2 ? 1 : 0;
                frameTau1DynamicObservationAppends += appendedTau1 ? 1 : 0;
                frameTauDynamicObservationAppends += appendedTau ? 1 : 0;

                snapshotTau2DynamicObservationAppends +=
                    appendDynamicObservationToSnapshot(frameTau2Id, tracklet[0], pBackendPoint, pointsTau2[i], qualityWeight);
                snapshotTau1DynamicObservationAppends +=
                    appendDynamicObservationToSnapshot(frameTau1Id, tracklet[1], pBackendPoint, pointsTau1[i], qualityWeight);
                snapshotTauDynamicObservationAppends +=
                    appendDynamicObservationToSnapshot(frameTauId, tracklet[2], pBackendPoint, pointTau, qualityWeight);
                if(appendedTau2 && appendedTau1 && appendedTau)
                    ++sharedBackendTrackletPoints;
                ++bufferedDynamicTrackletPoints;
            }
        }
    }

    if(debugInitForInstance || debugLifecycleForInstance || DebugFocusFrame(mCurrentFrame.mnId))
    {
        std::cout << "[STSLAM_INSTANCE_INIT] frame=" << mCurrentFrame.mnId
                  << " instance_id=" << instanceId
                  << " dynamic_tracklet_points_buffered=" << bufferedDynamicTrackletPoints
                  << " map_id=" << currentMapId
                  << " map_origin_frame=" << currentMapOriginFrame
                  << " map_origin_kf=" << currentMapOriginKF
                  << " map_age_frames=" << currentMapAgeFrames
                  << " tau2_frame=" << frameTau2.mnId
                  << " tau1_frame=" << frameTau1.mnId
                  << " tau_frame=" << mCurrentFrame.mnId
                  << " tracklets_as_structure_points=" << (trackletsAsStructurePoints ? 1 : 0)
                  << " register_structure_points_to_map=" << (registerStructurePointsToMap ? 1 : 0)
                  << " structure_tracklet_points_buffered=" << structureTrackletPointsBuffered
                  << " structure_tracklet_points_registered_to_map=" << structureTrackletPointsRegisteredToMap
                  << " triframe_structure_promotion_gate="
                  << (EnableTriFrameConsistencyStructurePromotionGate() ? 1 : 0)
                  << " triframe_structure_rejected=" << triframeRejectedStructureTrackletPoints
                  << " triframe_backend_observation_rejected=" << triframeRejectedBackendObservationTrackletPoints
                  << " accepted_tracklets=" << static_cast<int>(acceptedTracklets.size())
                  << " shared_backend_tracklet_points=" << sharedBackendTrackletPoints
                  << " frame_tau2_appends=" << frameTau2DynamicObservationAppends
                  << " frame_tau1_appends=" << frameTau1DynamicObservationAppends
                  << " frame_tau_appends=" << frameTauDynamicObservationAppends
                  << " snapshot_tau2_appends=" << snapshotTau2DynamicObservationAppends
                  << " snapshot_tau1_appends=" << snapshotTau1DynamicObservationAppends
                  << " snapshot_tau_appends=" << snapshotTauDynamicObservationAppends
                  << " rejected_invalid_current_index=" << rejectedDynamicTrackletsInvalidCurrentIndex
                  << std::endl;
    }

    int materializedDynamicTrackletPoints = 0;
    int materializedReusedStructureTrackletPoints = 0;
    const bool allowDynamicTrackletMaterialization =
        EnableDynamicTrackletMaterialization() &&
        ((mSensor != System::RGBD && mSensor != System::IMU_RGBD) ||
         EnableRgbdDynamicTrackletMaterialization());
    if(!initAsStatic && allowDynamicTrackletMaterialization && mpAtlas)
    {
        Map* pCurrentMap = mpAtlas->GetCurrentMap();
        auto markSnapshotDynamic = [&](const unsigned long frameId,
                                       const int featureIdx,
                                       MapPoint* pMP)
        {
            for(size_t i = 0; i < mvFramesSinceLastKeyFrame.size(); ++i)
            {
                WindowFrameSnapshot& snapshot = mvFramesSinceLastKeyFrame[i];
                if(snapshot.mnFrameId != frameId ||
                   featureIdx < 0 ||
                   featureIdx >= static_cast<int>(snapshot.mvpMapPoints.size()))
                {
                    continue;
                }

                snapshot.mvpMapPoints[featureIdx] = pMP;
                if(featureIdx < static_cast<int>(snapshot.mvbOutlier.size()))
                    snapshot.mvbOutlier[featureIdx] = false;
                snapshot.mmPredictedInstanceMotions[pInstance->GetId()] = velocity;
            }
        };

        auto markRecentDynamic = [&](Frame& frame,
                                     const int featureIdx,
                                     MapPoint* pMP)
        {
            if(featureIdx < 0 || featureIdx >= static_cast<int>(frame.mvpMapPoints.size()))
                return;

            frame.mvpMapPoints[featureIdx] = pMP;
            if(featureIdx < static_cast<int>(frame.mvbOutlier.size()))
                frame.mvbOutlier[featureIdx] = false;
            frame.mmPredictedInstanceMotions[pInstance->GetId()] = velocity;
        };

        if(pCurrentMap)
        {
            const unsigned long frameTau2Id = frameTau2.mnId;
            const unsigned long frameTau1Id = frameTau1.mnId;
            const unsigned long frameTauId = mCurrentFrame.mnId;
            const size_t nMaterialization =
                std::min(pointsTau1.size(), acceptedTracklets.size());

            for(size_t i = 0; i < nMaterialization; ++i)
            {
                const std::array<int, 3>& tracklet = acceptedTracklets[i];
                const int currentIdx = tracklet[2];
                if(currentIdx < 0 ||
                   currentIdx >= static_cast<int>(mCurrentFrame.mvpMapPoints.size()) ||
                   currentIdx >= static_cast<int>(mCurrentFrame.mvKeysUn.size()))
                {
                    continue;
                }
                const bool backendObservationEligible =
                    (i >= triframeBackendObservationEligibility.size()) ||
                    triframeBackendObservationEligibility[i];
                if(!backendObservationEligible)
                    continue;
                const bool triframeStructureEligible =
                    (i >= triframeStructureEligibility.size()) ||
                    triframeStructureEligibility[i];
                if(!trackletsAsStructurePoints || !triframeStructureEligible)
                    continue;

                MapPoint* pMP = NULL;
                bool reusedStructureTrackletPoint = false;
                if(trackletsAsStructurePoints &&
                   i < vBufferedTrackletBackendPoints.size() &&
                   vBufferedTrackletBackendPoints[i] &&
                   !vBufferedTrackletBackendPoints[i]->isBad() &&
                   vBufferedTrackletBackendPoints[i]->GetMap() == pCurrentMap &&
                   vBufferedTrackletBackendPoints[i]->IsInstanceStructurePoint())
                {
                    pMP = vBufferedTrackletBackendPoints[i];
                    reusedStructureTrackletPoint = true;
                    ++materializedReusedStructureTrackletPoints;
                }
                else
                {
                    pMP = mCurrentFrame.mvpMapPoints[currentIdx];
                }
                if(!pMP || pMP->isBad() || pMP->GetMap() != pCurrentMap)
                {
                    pMP = new MapPoint(pointsTau1[i], pCurrentMap, &mCurrentFrame, currentIdx);
                    pCurrentMap->AddMapPoint(pMP);
                    mCurrentFrame.mvpMapPoints[currentIdx] = pMP;
                    if(currentIdx < static_cast<int>(mCurrentFrame.mvbOutlier.size()))
                        mCurrentFrame.mvbOutlier[currentIdx] = false;
                    ++materializedDynamicTrackletPoints;
                }
                else if(reusedStructureTrackletPoint)
                {
                    mCurrentFrame.mvpMapPoints[currentIdx] = pMP;
                    if(currentIdx < static_cast<int>(mCurrentFrame.mvbOutlier.size()))
                        mCurrentFrame.mvbOutlier[currentIdx] = false;
                }

                if(!BindMapPointToInstance(pMP,
                                           pInstance->GetId(),
                                           pInstance->GetSemanticLabel(),
                                           false))
                {
                    continue;
                }

                pMP->UpdateObservationStats(static_cast<int>(frameTau2Id));
                pMP->UpdateObservationStats(static_cast<int>(frameTau1Id));
                pMP->UpdateObservationStats(static_cast<int>(frameTauId));

                if(mRecentPanopticFrames.size() >= 2)
                {
                    markRecentDynamic(mRecentPanopticFrames[mRecentPanopticFrames.size() - 2],
                                      tracklet[0],
                                      pMP);
                    markRecentDynamic(mRecentPanopticFrames.back(),
                                      tracklet[1],
                                      pMP);
                }
                markSnapshotDynamic(frameTau2Id, tracklet[0], pMP);
                markSnapshotDynamic(frameTau1Id, tracklet[1], pMP);
                markSnapshotDynamic(frameTauId, tracklet[2], pMP);
            }

            mCurrentFrame.mmPredictedInstanceMotions[pInstance->GetId()] = velocity;
        }
    }

    if((debugInitForInstance || debugLifecycleForInstance || DebugFocusFrame(mCurrentFrame.mnId)) && !initAsStatic)
    {
        std::cout << "[STSLAM_INSTANCE_INIT] frame=" << mCurrentFrame.mnId
                  << " instance_id=" << instanceId
                  << " dynamic_tracklet_points_materialized=" << materializedDynamicTrackletPoints
                  << " map_id=" << currentMapId
                  << " map_origin_frame=" << currentMapOriginFrame
                  << " map_origin_kf=" << currentMapOriginKF
                  << " map_age_frames=" << currentMapAgeFrames
                  << " accepted_tracklets=" << static_cast<int>(acceptedTracklets.size())
                  << " reused_structure_tracklet_points=" << materializedReusedStructureTrackletPoints
                  << std::endl;
    }

    if(mCurrentFrame.mpReferenceKF)
    {
        if(initializationMotionReliable)
            pInstance->UpdateMotionPrior(mCurrentFrame.mpReferenceKF, pInstance->GetVelocity());
        pInstance->UpdatePoseProxy(mCurrentFrame.mpReferenceKF, pInstance->GetLastPoseEstimate());
        mCurrentFrame.mmPredictedInstanceMotions[pInstance->GetId()] = pInstance->GetVelocity();
    }
    return true;
}

bool Tracking::OptimizePoseWithPanoptic()
{
    std::map<int, Sophus::SE3f> originalPredictedInstanceMotions;
    const bool filterByBackendEvidence =
        RequireBackendEvidenceForPanopticPose() &&
        !mCurrentFrame.mmPredictedInstanceMotions.empty();
    if(filterByBackendEvidence)
    {
        originalPredictedInstanceMotions = mCurrentFrame.mmPredictedInstanceMotions;
        for(std::map<int, Sophus::SE3f>::iterator itMotion =
                mCurrentFrame.mmPredictedInstanceMotions.begin();
            itMotion != mCurrentFrame.mmPredictedInstanceMotions.end(); )
        {
            if(IsNearlyIdentityInstanceMotion(itMotion->second))
            {
                ++itMotion;
                continue;
            }

            Instance* pInstance = mpAtlas ? mpAtlas->GetInstance(itMotion->first) : static_cast<Instance*>(NULL);
            const int backendEvidence = pInstance ? pInstance->GetBackendMotionEvidence() : 0;
            if(backendEvidence >= GetPanopticPoseMinBackendEvidence())
            {
                ++itMotion;
                continue;
            }

            if(DebugInstanceInitialization() || DebugInstanceResidualMotionGate() ||
               DebugFocusFrame(mCurrentFrame.mnId))
            {
                const Eigen::AngleAxisf angleAxis(itMotion->second.rotationMatrix());
                std::cout << "[STSLAM_PANOPTIC_POSE_GATE] frame=" << mCurrentFrame.mnId
                          << " instance_id=" << itMotion->first
                          << " rejected=insufficient_backend_motion_evidence"
                          << " backend_motion_evidence=" << backendEvidence
                          << " min_backend_motion_evidence=" << GetPanopticPoseMinBackendEvidence()
                          << " translation_norm=" << itMotion->second.translation().norm()
                          << " rotation_deg="
                          << std::abs(angleAxis.angle()) * 180.0 / 3.14159265358979323846
                          << std::endl;
            }

            itMotion = mCurrentFrame.mmPredictedInstanceMotions.erase(itMotion);
        }
    }

    const int nInliers = Optimizer::PoseOptimizationPanoptic(&mCurrentFrame);
    if(filterByBackendEvidence)
        mCurrentFrame.mmPredictedInstanceMotions = originalPredictedInstanceMotions;

    return nInliers >= 10;
}

void Tracking::PushPanopticHistory(const Frame& frame)
{
    if(!frame.HasPanopticObservation() || !frame.isSet())
    {
        mRecentPanopticFrames.clear();
        return;
    }

    mRecentPanopticFrames.push_back(Frame(frame));
    while(mRecentPanopticFrames.size() > mnPanopticHistoryLength)
        mRecentPanopticFrames.pop_front();
}

void Tracking::AppendWindowFrameSnapshot(const Frame& frame)
{
    if(!frame.HasPanopticObservation() || !frame.isSet())
        return;

    mvFramesSinceLastKeyFrame.push_back(WindowFrameSnapshot(frame));
}

std::vector<int> Tracking::CollectFeatureIndicesForInstance(const Frame& frame, int instanceId) const
{
    return frame.GetFeatureIndicesForInstance(instanceId);
}

Tracking::CanonicalAssociationDiagnostics Tracking::EvaluateCanonicalAssociationStability(
    const Frame& frameTau2,
    const Frame& frameTau1,
    const Frame& frameTau,
    int canonicalInstanceId) const
{
    CanonicalAssociationDiagnostics diag;
    diag.valid = true;
    diag.stable = true;
    diag.minCorrectionStreak = std::numeric_limits<int>::max();

    std::vector<std::string> reasons;
    auto appendReason = [&](const std::string& reason)
    {
        if(std::find(reasons.begin(), reasons.end(), reason) == reasons.end())
            reasons.push_back(reason);
    };

    auto readAssociation =
        [&](const Frame& frame,
            int& rawIdOut,
            int& semanticOut) -> bool
    {
        const std::map<unsigned long, std::map<int, PanopticCanonicalAssociation>>::const_iterator frameIt =
            mmFramePanopticCanonicalAssociations.find(frame.mnId);
        if(frameIt == mmFramePanopticCanonicalAssociations.end())
        {
            ++diag.missingFrames;
            appendReason("missing_canonical_frame");
            return false;
        }

        const PanopticCanonicalAssociation* selectedAssociation = NULL;
        int matches = 0;
        for(const std::pair<const int, PanopticCanonicalAssociation>& entry : frameIt->second)
        {
            const PanopticCanonicalAssociation& association = entry.second;
            if(association.canonicalInstanceId != canonicalInstanceId)
                continue;
            selectedAssociation = &association;
            ++matches;
        }

        if(matches == 0 || !selectedAssociation)
        {
            ++diag.missingFrames;
            appendReason("missing_canonical_association");
            return false;
        }
        if(matches > 1)
        {
            ++diag.ambiguousFrames;
            appendReason("ambiguous_canonical_association");
        }

        rawIdOut = selectedAssociation->rawInstanceId;
        semanticOut = selectedAssociation->semanticId;
        if(selectedAssociation->matchedToPrevious)
            ++diag.matchedFrames;
        if(selectedAssociation->frameCorrection)
            ++diag.frameCorrections;
        if(selectedAssociation->permanentCorrection)
            ++diag.permanentCorrections;
        if(selectedAssociation->correctionStreak > 0)
            diag.minCorrectionStreak =
                std::min(diag.minCorrectionStreak, selectedAssociation->correctionStreak);

        const bool isRemapped =
            selectedAssociation->rawInstanceId != selectedAssociation->canonicalInstanceId;
        const bool matureRemap =
            selectedAssociation->permanentCorrection ||
            selectedAssociation->correctionStreak >= mnPanopticPermanentCorrectionMinStreak;
        if(isRemapped && !matureRemap)
        {
            ++diag.transientCorrections;
            appendReason("transient_canonical_remap");
        }
        return true;
    };

    const bool hasTau2 = readAssociation(frameTau2, diag.rawTau2, diag.semanticTau2);
    const bool hasTau1 = readAssociation(frameTau1, diag.rawTau1, diag.semanticTau1);
    const bool hasTau = readAssociation(frameTau, diag.rawTau, diag.semanticTau);
    diag.valid = hasTau2 && hasTau1 && hasTau;

    if(diag.valid &&
       (diag.semanticTau2 != diag.semanticTau1 || diag.semanticTau1 != diag.semanticTau))
    {
        appendReason("canonical_semantic_mismatch");
    }

    if(diag.minCorrectionStreak == std::numeric_limits<int>::max())
        diag.minCorrectionStreak = 0;

    diag.stable = reasons.empty();
    if(!reasons.empty())
    {
        std::ostringstream oss;
        for(size_t i = 0; i < reasons.size(); ++i)
        {
            if(i > 0)
                oss << ",";
            oss << reasons[i];
        }
        diag.rejectReason = oss.str();
    }
    return diag;
}

Tracking::InstanceTrackletStabilityDiagnostics Tracking::EvaluateInstanceTrackletStability(
    const Frame& frameTau2,
    const Frame& frameTau1,
    const Frame& frameTau,
    int instanceId,
    const std::vector<int>& featuresTau2,
    const std::vector<int>& featuresTau1,
    const std::vector<int>& featuresTau,
    const std::vector<std::array<int, 3>>& tracklets) const
{
    InstanceTrackletStabilityDiagnostics diag;
    diag.featuresTau2 = static_cast<int>(featuresTau2.size());
    diag.featuresTau1 = static_cast<int>(featuresTau1.size());
    diag.featuresTau = static_cast<int>(featuresTau.size());
    diag.tracklets = static_cast<int>(tracklets.size());

    const InstanceObservation* obsTau2 = FindInstanceObservation(frameTau2, instanceId);
    const InstanceObservation* obsTau1 = FindInstanceObservation(frameTau1, instanceId);
    const InstanceObservation* obsTau = FindInstanceObservation(frameTau, instanceId);
    if(!obsTau2 || !obsTau1 || !obsTau)
    {
        diag.valid = false;
        diag.stable = false;
        diag.rejectReason = "missing_instance_observation";
        return diag;
    }

    diag.valid = true;
    diag.semanticTau2 = obsTau2->semanticId;
    diag.semanticTau1 = obsTau1->semanticId;
    diag.semanticTau = obsTau->semanticId;
    diag.bboxIoUTau2Tau1 = SafeRectIoU(obsTau2->bbox, obsTau1->bbox);
    diag.bboxIoUTau1Tau = SafeRectIoU(obsTau1->bbox, obsTau->bbox);

    auto center = [](const cv::Rect& rect) -> cv::Point2d
    {
        return cv::Point2d(rect.x + 0.5 * rect.width, rect.y + 0.5 * rect.height);
    };
    const double imageWidth = std::max(1.0, static_cast<double>(Frame::mnMaxX - Frame::mnMinX));
    const double imageHeight = std::max(1.0, static_cast<double>(Frame::mnMaxY - Frame::mnMinY));
    const double imageDiagonal = std::max(1.0, std::sqrt(imageWidth * imageWidth + imageHeight * imageHeight));
    diag.bboxCenterShiftTau2Tau1 =
        cv::norm(center(obsTau2->bbox) - center(obsTau1->bbox)) / imageDiagonal;
    diag.bboxCenterShiftTau1Tau =
        cv::norm(center(obsTau1->bbox) - center(obsTau->bbox)) / imageDiagonal;

    auto safeRatio3 = [](double a, double b, double c) -> double
    {
        const double minValue = std::max(1.0, std::min(a, std::min(b, c)));
        const double maxValue = std::max(a, std::max(b, c));
        return maxValue / minValue;
    };
    diag.maskAreaRatio =
        safeRatio3(static_cast<double>(obsTau2->area),
                   static_cast<double>(obsTau1->area),
                   static_cast<double>(obsTau->area));
    diag.bboxAreaRatio =
        safeRatio3(static_cast<double>(obsTau2->bbox.area()),
                   static_cast<double>(obsTau1->bbox.area()),
                   static_cast<double>(obsTau->bbox.area()));

    const double densityTau2 =
        obsTau2->area > 0 ? static_cast<double>(featuresTau2.size()) / static_cast<double>(obsTau2->area) : 0.0;
    const double densityTau1 =
        obsTau1->area > 0 ? static_cast<double>(featuresTau1.size()) / static_cast<double>(obsTau1->area) : 0.0;
    const double densityTau =
        obsTau->area > 0 ? static_cast<double>(featuresTau.size()) / static_cast<double>(obsTau->area) : 0.0;
    diag.featureDensityRatio = safeRatio3(densityTau2 * 1000000.0,
                                          densityTau1 * 1000000.0,
                                          densityTau * 1000000.0);

    const int minFeatureCount =
        std::min(diag.featuresTau2, std::min(diag.featuresTau1, diag.featuresTau));
    diag.trackletRetention =
        minFeatureCount > 0 ?
        static_cast<double>(tracklets.size()) / static_cast<double>(minFeatureCount) : 0.0;

    if(!tracklets.empty() && obsTau->bbox.area() > 0)
    {
        double minX = std::numeric_limits<double>::infinity();
        double minY = std::numeric_limits<double>::infinity();
        double maxX = -std::numeric_limits<double>::infinity();
        double maxY = -std::numeric_limits<double>::infinity();
        int validCurrentTracklets = 0;
        for(size_t i = 0; i < tracklets.size(); ++i)
        {
            const int currentIdx = tracklets[i][2];
            if(currentIdx < 0 || currentIdx >= static_cast<int>(frameTau.mvKeysUn.size()))
                continue;
            const cv::Point2f& pt = frameTau.mvKeysUn[currentIdx].pt;
            minX = std::min(minX, static_cast<double>(pt.x));
            minY = std::min(minY, static_cast<double>(pt.y));
            maxX = std::max(maxX, static_cast<double>(pt.x));
            maxY = std::max(maxY, static_cast<double>(pt.y));
            ++validCurrentTracklets;
        }
        if(validCurrentTracklets >= 2 && std::isfinite(minX) && std::isfinite(minY))
        {
            const double spreadX = std::max(0.0, maxX - minX);
            const double spreadY = std::max(0.0, maxY - minY);
            diag.currentTrackletSpreadX =
                obsTau->bbox.width > 0 ? spreadX / static_cast<double>(obsTau->bbox.width) : 0.0;
            diag.currentTrackletSpreadY =
                obsTau->bbox.height > 0 ? spreadY / static_cast<double>(obsTau->bbox.height) : 0.0;
            diag.currentTrackletCoverage =
                (spreadX * spreadY) / static_cast<double>(obsTau->bbox.area());
        }
    }

    if(!tracklets.empty())
    {
        std::vector<double> flowTau2Tau1Norms;
        std::vector<double> flowTau1TauNorms;
        std::vector<double> accelerationNorms;
        std::vector<double> directionCosines;
        flowTau2Tau1Norms.reserve(tracklets.size());
        flowTau1TauNorms.reserve(tracklets.size());
        accelerationNorms.reserve(tracklets.size());
        directionCosines.reserve(tracklets.size());

        const double bboxDiagonal =
            std::sqrt(static_cast<double>(obsTau->bbox.width) * static_cast<double>(obsTau->bbox.width) +
                      static_cast<double>(obsTau->bbox.height) * static_cast<double>(obsTau->bbox.height));
        const double geometryScale = std::max(1.0, bboxDiagonal);

        for(size_t i = 0; i < tracklets.size(); ++i)
        {
            const int idxTau2 = tracklets[i][0];
            const int idxTau1 = tracklets[i][1];
            const int idxTau = tracklets[i][2];
            if(idxTau2 < 0 || idxTau2 >= static_cast<int>(frameTau2.mvKeysUn.size()) ||
               idxTau1 < 0 || idxTau1 >= static_cast<int>(frameTau1.mvKeysUn.size()) ||
               idxTau < 0 || idxTau >= static_cast<int>(frameTau.mvKeysUn.size()))
            {
                continue;
            }

            const cv::Point2f& pTau2 = frameTau2.mvKeysUn[idxTau2].pt;
            const cv::Point2f& pTau1 = frameTau1.mvKeysUn[idxTau1].pt;
            const cv::Point2f& pTau = frameTau.mvKeysUn[idxTau].pt;
            const cv::Point2d flowTau2Tau1(static_cast<double>(pTau1.x - pTau2.x),
                                           static_cast<double>(pTau1.y - pTau2.y));
            const cv::Point2d flowTau1Tau(static_cast<double>(pTau.x - pTau1.x),
                                          static_cast<double>(pTau.y - pTau1.y));
            const double normTau2Tau1 = cv::norm(flowTau2Tau1);
            const double normTau1Tau = cv::norm(flowTau1Tau);
            flowTau2Tau1Norms.push_back(normTau2Tau1);
            flowTau1TauNorms.push_back(normTau1Tau);
            accelerationNorms.push_back(cv::norm(flowTau1Tau - flowTau2Tau1));

            if(normTau2Tau1 > 1e-3 && normTau1Tau > 1e-3)
            {
                double cosine =
                    (flowTau2Tau1.x * flowTau1Tau.x + flowTau2Tau1.y * flowTau1Tau.y) /
                    (normTau2Tau1 * normTau1Tau);
                cosine = std::max(-1.0, std::min(1.0, cosine));
                directionCosines.push_back(cosine);
            }
        }

        if(!flowTau2Tau1Norms.empty())
            diag.medianFlowTau2Tau1Px = MedianValue(flowTau2Tau1Norms);
        if(!flowTau1TauNorms.empty())
            diag.medianFlowTau1TauPx = MedianValue(flowTau1TauNorms);
        if(!accelerationNorms.empty())
        {
            diag.medianFlowAccelerationPx = MedianValue(accelerationNorms);
            diag.medianFlowAccelerationNormalized =
                diag.medianFlowAccelerationPx / geometryScale;
        }
        if(!directionCosines.empty())
            diag.medianFlowDirectionCosine = MedianValue(directionCosines);

        const int sampleLimit = GetInstanceIdGeometryTriangulationSampleLimit();
        const size_t sampleStep =
            sampleLimit > 0 && tracklets.size() > static_cast<size_t>(sampleLimit) ?
            static_cast<size_t>(std::ceil(static_cast<double>(tracklets.size()) /
                                          static_cast<double>(sampleLimit))) :
            static_cast<size_t>(1);
        int sampledTracklets = 0;
        int successTau2Tau1 = 0;
        int successTau1Tau = 0;
        int invalidTriangulations = 0;
        int highReprojectionTriangulations = 0;
        for(size_t i = 0; i < tracklets.size(); i += sampleStep)
        {
            if(sampleLimit > 0 && sampledTracklets >= sampleLimit)
                break;

            Eigen::Vector3f pointWorld;
            TriangulationQuality qualityTau2Tau1;
            TriangulationQuality qualityTau1Tau;
            const bool okTau2Tau1 =
                TriangulateMatchedFeatures(frameTau2, tracklets[i][0],
                                           frameTau1, tracklets[i][1],
                                           pointWorld, &qualityTau2Tau1);
            const bool okTau1Tau =
                TriangulateMatchedFeatures(frameTau1, tracklets[i][1],
                                           frameTau, tracklets[i][2],
                                           pointWorld, &qualityTau1Tau);
            if(okTau2Tau1)
                ++successTau2Tau1;
            else if(qualityTau2Tau1.rejectReason == "nonpositive_depth")
                ++invalidTriangulations;
            else if(qualityTau2Tau1.rejectReason == "high_reprojection_error")
                ++highReprojectionTriangulations;

            if(okTau1Tau)
                ++successTau1Tau;
            else if(qualityTau1Tau.rejectReason == "nonpositive_depth")
                ++invalidTriangulations;
            else if(qualityTau1Tau.rejectReason == "high_reprojection_error")
                ++highReprojectionTriangulations;

            ++sampledTracklets;
        }

        diag.triangulationSamples = sampledTracklets;
        if(sampledTracklets > 0)
        {
            diag.triangulationSuccessRatioTau2Tau1 =
                static_cast<double>(successTau2Tau1) / static_cast<double>(sampledTracklets);
            diag.triangulationSuccessRatioTau1Tau =
                static_cast<double>(successTau1Tau) / static_cast<double>(sampledTracklets);
            const double totalTriangulations = 2.0 * static_cast<double>(sampledTracklets);
            diag.triangulationNonpositiveRatio =
                static_cast<double>(invalidTriangulations) / totalTriangulations;
            diag.triangulationHighReprojectionRatio =
                static_cast<double>(highReprojectionTriangulations) / totalTriangulations;
            diag.triangulationInvalidGeometryRatio =
                diag.triangulationNonpositiveRatio + diag.triangulationHighReprojectionRatio;
        }
    }

    std::vector<std::string> reasons;
    std::vector<std::string> hardRejectReasons;
    if(diag.semanticTau2 != diag.semanticTau1 || diag.semanticTau1 != diag.semanticTau)
    {
        reasons.push_back("semantic_mismatch");
        hardRejectReasons.push_back("semantic_mismatch");
    }
    if(diag.trackletRetention < GetInstanceIdMinTrackletRetention())
    {
        reasons.push_back("low_tracklet_retention");
        hardRejectReasons.push_back("low_tracklet_retention");
    }
    if(std::max(diag.bboxCenterShiftTau2Tau1, diag.bboxCenterShiftTau1Tau) >
       GetInstanceIdMaxBboxCenterShift())
    {
        reasons.push_back("bbox_center_jump");
        hardRejectReasons.push_back("bbox_center_jump");
    }
    if(std::max(diag.maskAreaRatio, diag.bboxAreaRatio) > GetInstanceIdMaxAreaRatio())
    {
        reasons.push_back("area_jump");
        const double minBboxIoU = std::min(diag.bboxIoUTau2Tau1, diag.bboxIoUTau1Tau);
        const double maxCenterShift = std::max(diag.bboxCenterShiftTau2Tau1,
                                               diag.bboxCenterShiftTau1Tau);
        if(diag.bboxAreaRatio > GetInstanceIdMaxAreaRatio() &&
           minBboxIoU < GetInstanceIdBboxAreaJumpHardRejectMaxIoU() &&
           diag.trackletRetention >= GetInstanceIdAreaJumpHardRejectMinRetention() &&
           maxCenterShift < GetInstanceIdBboxAreaJumpHardRejectMaxCenterShift())
        {
            hardRejectReasons.push_back("bbox_area_jump");
        }
    }
    if(diag.featureDensityRatio > GetInstanceIdMaxFeatureDensityRatio())
        reasons.push_back("feature_density_jump");
    if(diag.tracklets >= mnInstanceInitializationMinTracklets &&
       diag.currentTrackletCoverage < GetInstanceIdMinTrackletCoverage())
        reasons.push_back("degenerate_tracklet_spread");
    if(diag.tracklets >= mnInstanceInitializationMinTracklets &&
       diag.medianFlowAccelerationNormalized > GetInstanceIdMaxFlowAccelerationNormalized())
    {
        reasons.push_back("flow_acceleration_diagnostic");
        if(diag.medianFlowDirectionCosine < GetInstanceIdMinFlowDirectionCosine())
            reasons.push_back("flow_direction_flip_diagnostic");
    }
    if(diag.triangulationSamples >= mnInstanceInitializationMinTracklets)
    {
        const double minSuccessRatio =
            std::min(diag.triangulationSuccessRatioTau2Tau1,
                     diag.triangulationSuccessRatioTau1Tau);
        if(minSuccessRatio < GetInstanceIdMinTriangulationSuccessRatio())
        {
            reasons.push_back("low_geometric_triangulation_support");
            if(diag.triangulationInvalidGeometryRatio > GetInstanceIdMaxTriangulationInvalidRatio())
                hardRejectReasons.push_back("invalid_triframe_geometry");
        }
    }

    diag.stable = reasons.empty();
    diag.hardReject = !hardRejectReasons.empty();
    if(!reasons.empty())
    {
        std::ostringstream oss;
        for(size_t i = 0; i < reasons.size(); ++i)
        {
            if(i > 0)
                oss << ",";
            oss << reasons[i];
        }
        diag.rejectReason = oss.str();
    }
    if(!hardRejectReasons.empty())
    {
        std::ostringstream oss;
        for(size_t i = 0; i < hardRejectReasons.size(); ++i)
        {
            if(i > 0)
                oss << ",";
            oss << hardRejectReasons[i];
        }
        diag.hardRejectReason = oss.str();
    }
    return diag;
}

std::vector<std::array<int, 3>> Tracking::CollectInstanceTracklets(const Frame& frameTau2,
                                                                   const Frame& frameTau1,
                                                                   const Frame& frameTau,
                                                                   int instanceId) const
{
    const std::vector<int> featuresTau2 = CollectFeatureIndicesForInstance(frameTau2, instanceId);
    const std::vector<int> featuresTau1 = CollectFeatureIndicesForInstance(frameTau1, instanceId);
    const std::vector<int> featuresTau = CollectFeatureIndicesForInstance(frameTau, instanceId);

    if(featuresTau2.empty() || featuresTau1.empty() || featuresTau.empty())
        return {};

    const cv::Mat descriptorsTau2 = ExtractDescriptorRows(frameTau2.mDescriptors, featuresTau2);
    const cv::Mat descriptorsTau1 = ExtractDescriptorRows(frameTau1.mDescriptors, featuresTau1);
    const cv::Mat descriptorsTau = ExtractDescriptorRows(frameTau.mDescriptors, featuresTau);
    if(descriptorsTau2.empty() || descriptorsTau1.empty() || descriptorsTau.empty())
        return {};

    std::vector<cv::DMatch> matchesTau2Tau1;
    std::vector<cv::DMatch> matchesTau1Tau;
    const bool strictDescriptorGate = EnableInstanceTrackletStrictDescriptorGate();
    const int descriptorMaxDistance = GetInstanceTrackletDescriptorMaxDistance();
    const double descriptorRatio = GetInstanceTrackletDescriptorRatio();
    auto matchInstanceDescriptors = [&](const cv::Mat& descriptorsA,
                                        const cv::Mat& descriptorsB,
                                        std::vector<cv::DMatch>& matches)
    {
        matches.clear();
        if(descriptorsA.empty() || descriptorsB.empty())
            return;

        if(!strictDescriptorGate)
        {
            cv::BFMatcher matcher(cv::NORM_HAMMING, true);
            matcher.match(descriptorsA, descriptorsB, matches);
            return;
        }

        cv::BFMatcher matcher(cv::NORM_HAMMING, false);
        std::vector<std::vector<cv::DMatch>> forwardMatches;
        std::vector<std::vector<cv::DMatch>> backwardMatches;
        matcher.knnMatch(descriptorsA, descriptorsB, forwardMatches, 2);
        matcher.knnMatch(descriptorsB, descriptorsA, backwardMatches, 2);

        std::vector<int> backwardBest(descriptorsB.rows, -1);
        std::vector<int> backwardDistance(descriptorsB.rows, 256);
        for(size_t i = 0; i < backwardMatches.size(); ++i)
        {
            if(backwardMatches[i].empty())
                continue;
            const cv::DMatch& best = backwardMatches[i][0];
            const bool passDistance = best.distance <= descriptorMaxDistance;
            bool passRatio = true;
            if(descriptorRatio > 0.0 && backwardMatches[i].size() > 1)
                passRatio = static_cast<double>(best.distance) <
                            descriptorRatio * static_cast<double>(backwardMatches[i][1].distance);
            if(passDistance && passRatio)
            {
                backwardBest[best.queryIdx] = best.trainIdx;
                backwardDistance[best.queryIdx] = static_cast<int>(best.distance);
            }
        }

        for(size_t i = 0; i < forwardMatches.size(); ++i)
        {
            if(forwardMatches[i].empty())
                continue;
            const cv::DMatch& best = forwardMatches[i][0];
            const bool passDistance = best.distance <= descriptorMaxDistance;
            bool passRatio = true;
            if(descriptorRatio > 0.0 && forwardMatches[i].size() > 1)
                passRatio = static_cast<double>(best.distance) <
                            descriptorRatio * static_cast<double>(forwardMatches[i][1].distance);
            if(!passDistance || !passRatio)
                continue;
            if(best.trainIdx < 0 || best.trainIdx >= static_cast<int>(backwardBest.size()))
                continue;
            if(backwardBest[best.trainIdx] != best.queryIdx)
                continue;
            if(backwardDistance[best.trainIdx] > descriptorMaxDistance)
                continue;
            matches.push_back(best);
        }
    };
    matchInstanceDescriptors(descriptorsTau2, descriptorsTau1, matchesTau2Tau1);
    matchInstanceDescriptors(descriptorsTau1, descriptorsTau, matchesTau1Tau);

    std::map<int, int> bridgeTau1ToTau2;
    std::map<int, int> bridgeTau1ToTau;
    for(const cv::DMatch& match : matchesTau2Tau1)
        bridgeTau1ToTau2[match.trainIdx] = match.queryIdx;
    for(const cv::DMatch& match : matchesTau1Tau)
        bridgeTau1ToTau[match.queryIdx] = match.trainIdx;

    std::vector<std::array<int, 3>> tracklets;
    tracklets.reserve(featuresTau1.size());
    for(const auto& bridgeEntry : bridgeTau1ToTau2)
    {
        const auto currentIt = bridgeTau1ToTau.find(bridgeEntry.first);
        if(currentIt == bridgeTau1ToTau.end())
            continue;

        tracklets.push_back(
            {featuresTau2[bridgeEntry.second], featuresTau1[bridgeEntry.first], featuresTau[currentIt->second]});
    }

    const size_t rawTrackletCount = tracklets.size();
    int motionCoherenceRejected = 0;
    double motionCoherenceGateTau2Tau1 = 0.0;
    double motionCoherenceGateTau1Tau = 0.0;
    if(EnableInstanceTrackletMotionCoherenceGate() &&
       static_cast<int>(tracklets.size()) >= std::max(mnInstanceInitializationMinTracklets * 2,
                                                      mnInstanceInitializationMinTracklets + 4))
    {
        std::vector<double> dxTau2Tau1;
        std::vector<double> dyTau2Tau1;
        std::vector<double> dxTau1Tau;
        std::vector<double> dyTau1Tau;
        dxTau2Tau1.reserve(tracklets.size());
        dyTau2Tau1.reserve(tracklets.size());
        dxTau1Tau.reserve(tracklets.size());
        dyTau1Tau.reserve(tracklets.size());
        for(const std::array<int, 3>& tracklet : tracklets)
        {
            const cv::Point2f& pTau2 = frameTau2.mvKeysUn[tracklet[0]].pt;
            const cv::Point2f& pTau1 = frameTau1.mvKeysUn[tracklet[1]].pt;
            const cv::Point2f& pTau = frameTau.mvKeysUn[tracklet[2]].pt;
            dxTau2Tau1.push_back(static_cast<double>(pTau1.x - pTau2.x));
            dyTau2Tau1.push_back(static_cast<double>(pTau1.y - pTau2.y));
            dxTau1Tau.push_back(static_cast<double>(pTau.x - pTau1.x));
            dyTau1Tau.push_back(static_cast<double>(pTau.y - pTau1.y));
        }

        const cv::Point2d medianFlowTau2Tau1(MedianValue(dxTau2Tau1),
                                             MedianValue(dyTau2Tau1));
        const cv::Point2d medianFlowTau1Tau(MedianValue(dxTau1Tau),
                                            MedianValue(dyTau1Tau));
        std::vector<double> residualsTau2Tau1;
        std::vector<double> residualsTau1Tau;
        residualsTau2Tau1.reserve(tracklets.size());
        residualsTau1Tau.reserve(tracklets.size());
        for(size_t i = 0; i < tracklets.size(); ++i)
        {
            const cv::Point2d flowTau2Tau1(dxTau2Tau1[i], dyTau2Tau1[i]);
            const cv::Point2d flowTau1Tau(dxTau1Tau[i], dyTau1Tau[i]);
            residualsTau2Tau1.push_back(cv::norm(flowTau2Tau1 - medianFlowTau2Tau1));
            residualsTau1Tau.push_back(cv::norm(flowTau1Tau - medianFlowTau1Tau));
        }

        const double madScale = 1.4826 * GetInstanceTrackletMotionMadScale();
        motionCoherenceGateTau2Tau1 =
            std::max(GetInstanceTrackletMotionMinGatePx(),
                     madScale * MedianValue(residualsTau2Tau1));
        motionCoherenceGateTau1Tau =
            std::max(GetInstanceTrackletMotionMinGatePx(),
                     madScale * MedianValue(residualsTau1Tau));

        std::vector<std::array<int, 3>> coherentTracklets;
        coherentTracklets.reserve(tracklets.size());
        for(size_t i = 0; i < tracklets.size(); ++i)
        {
            if(residualsTau2Tau1[i] <= motionCoherenceGateTau2Tau1 &&
               residualsTau1Tau[i] <= motionCoherenceGateTau1Tau)
            {
                coherentTracklets.push_back(tracklets[i]);
            }
            else
            {
                ++motionCoherenceRejected;
            }
        }

        if(static_cast<int>(coherentTracklets.size()) >= mnInstanceInitializationMinTracklets)
            tracklets.swap(coherentTracklets);
    }

    std::sort(tracklets.begin(), tracklets.end());
    const InstanceTrackletStabilityDiagnostics stability =
        EvaluateInstanceTrackletStability(frameTau2,
                                          frameTau1,
                                          frameTau,
                                          instanceId,
                                          featuresTau2,
                                          featuresTau1,
                                          featuresTau,
                                          tracklets);
    if(DebugInstanceTracklets() || DebugFocusFrame(frameTau.mnId))
    {
        const InstanceObservation* obsTau2 = FindInstanceObservation(frameTau2, instanceId);
        const InstanceObservation* obsTau1 = FindInstanceObservation(frameTau1, instanceId);
        const InstanceObservation* obsTau = FindInstanceObservation(frameTau, instanceId);
        const double imageArea = EstimateFrameImageArea();

        auto matchMeanDistance = [](const std::vector<cv::DMatch>& matches) -> double
        {
            if(matches.empty())
                return 0.0;
            double sum = 0.0;
            for(size_t i = 0; i < matches.size(); ++i)
                sum += matches[i].distance;
            return sum / static_cast<double>(matches.size());
        };

        auto matchMaxDistance = [](const std::vector<cv::DMatch>& matches) -> double
        {
            double maxDistance = 0.0;
            for(size_t i = 0; i < matches.size(); ++i)
                maxDistance = std::max(maxDistance, static_cast<double>(matches[i].distance));
            return maxDistance;
        };

        auto areaOrZero = [](const InstanceObservation* obs) -> int
        {
            return obs ? obs->area : 0;
        };

        auto bboxAreaOrZero = [](const InstanceObservation* obs) -> int
        {
            return obs ? obs->bbox.area() : 0;
        };

        auto semanticOrMinusOne = [](const InstanceObservation* obs) -> int
        {
            return obs ? obs->semanticId : -1;
        };

        auto formatRawIdsForCanonical = [&](const unsigned long frameId,
                                            const int canonicalInstanceId) -> std::string
        {
            const std::map<unsigned long, std::map<int, int>>::const_iterator frameIt =
                mmFramePanopticRawToCanonicalIds.find(frameId);
            if(frameIt == mmFramePanopticRawToCanonicalIds.end())
                return "unknown";

            std::ostringstream rawIds;
            int nRawIds = 0;
            for(const std::pair<const int, int>& entry : frameIt->second)
            {
                if(entry.second != canonicalInstanceId)
                    continue;
                if(nRawIds > 0)
                    rawIds << "|";
                rawIds << entry.first;
                ++nRawIds;
            }
            return nRawIds > 0 ? rawIds.str() : "none";
        };

        auto formatCanonicalAssociationForFrame = [&](const unsigned long frameId,
                                                      const int canonicalInstanceId) -> std::string
        {
            const std::map<unsigned long, std::map<int, PanopticCanonicalAssociation>>::const_iterator frameIt =
                mmFramePanopticCanonicalAssociations.find(frameId);
            if(frameIt == mmFramePanopticCanonicalAssociations.end())
                return "unknown";

            std::ostringstream associations;
            int printed = 0;
            for(const std::pair<const int, PanopticCanonicalAssociation>& entry : frameIt->second)
            {
                const PanopticCanonicalAssociation& association = entry.second;
                if(association.canonicalInstanceId != canonicalInstanceId)
                    continue;
                if(printed > 0)
                    associations << "|";
                associations << association.rawInstanceId << "->" << association.canonicalInstanceId
                             << ":sem=" << association.semanticId
                             << ":feat=" << association.featureCount
                             << ":best=" << association.bestMatches
                             << ":second=" << association.secondBestMatches
                             << ":streak=" << association.correctionStreak
                             << ":perm=" << (association.permanentCorrection ? 1 : 0)
                             << ":frame=" << (association.frameCorrection ? 1 : 0);
                ++printed;
            }
            return printed > 0 ? associations.str() : "none";
        };

        std::cout << "[STSLAM_TRACKLET_DIAG] current_frame=" << frameTau.mnId
                  << " instance_id=" << instanceId
                  << " canonical_id=" << instanceId
                  << " raw_ids_tau2=" << formatRawIdsForCanonical(frameTau2.mnId, instanceId)
                  << " raw_ids_tau1=" << formatRawIdsForCanonical(frameTau1.mnId, instanceId)
                  << " raw_ids_tau=" << formatRawIdsForCanonical(frameTau.mnId, instanceId)
                  << " canonical_assoc_tau2=" << formatCanonicalAssociationForFrame(frameTau2.mnId, instanceId)
                  << " canonical_assoc_tau1=" << formatCanonicalAssociationForFrame(frameTau1.mnId, instanceId)
                  << " canonical_assoc_tau=" << formatCanonicalAssociationForFrame(frameTau.mnId, instanceId)
                  << " stability_gate=" << (EnableInstanceIdStabilityGate() ? 1 : 0)
                  << " stable=" << (stability.stable ? 1 : 0)
                  << " hard_reject=" << (stability.hardReject ? 1 : 0)
                  << " unstable_reason=" << (stability.rejectReason.empty() ? "none" : stability.rejectReason)
                  << " hard_reject_reason=" << (stability.hardRejectReason.empty() ? "none" : stability.hardRejectReason)
                  << " tau2_frame=" << frameTau2.mnId
                  << " tau1_frame=" << frameTau1.mnId
                  << " semantic_tau2=" << semanticOrMinusOne(obsTau2)
                  << " semantic_tau1=" << semanticOrMinusOne(obsTau1)
                  << " semantic_tau=" << semanticOrMinusOne(obsTau)
                  << " features_tau2=" << featuresTau2.size()
                  << " features_tau1=" << featuresTau1.size()
                  << " features_tau=" << featuresTau.size()
                  << " mask_area_tau2=" << areaOrZero(obsTau2)
                  << " mask_area_tau1=" << areaOrZero(obsTau1)
                  << " mask_area_tau=" << areaOrZero(obsTau)
                  << " mask_coverage_tau2=" << static_cast<double>(areaOrZero(obsTau2)) / imageArea
                  << " mask_coverage_tau1=" << static_cast<double>(areaOrZero(obsTau1)) / imageArea
                  << " mask_coverage_tau=" << static_cast<double>(areaOrZero(obsTau)) / imageArea
                  << " feature_density_per_kpix_tau2="
                  << (areaOrZero(obsTau2) > 0 ? static_cast<double>(featuresTau2.size()) * 1000.0 / areaOrZero(obsTau2) : 0.0)
                  << " feature_density_per_kpix_tau1="
                  << (areaOrZero(obsTau1) > 0 ? static_cast<double>(featuresTau1.size()) * 1000.0 / areaOrZero(obsTau1) : 0.0)
                  << " feature_density_per_kpix_tau="
                  << (areaOrZero(obsTau) > 0 ? static_cast<double>(featuresTau.size()) * 1000.0 / areaOrZero(obsTau) : 0.0)
                  << " bbox_area_tau2=" << bboxAreaOrZero(obsTau2)
                  << " bbox_area_tau1=" << bboxAreaOrZero(obsTau1)
                  << " bbox_area_tau=" << bboxAreaOrZero(obsTau)
                  << " bbox_iou_tau2_tau1=" << stability.bboxIoUTau2Tau1
                  << " bbox_iou_tau1_tau=" << stability.bboxIoUTau1Tau
                  << " bbox_center_shift_tau2_tau1=" << stability.bboxCenterShiftTau2Tau1
                  << " bbox_center_shift_tau1_tau=" << stability.bboxCenterShiftTau1Tau
                  << " bbox_area_ratio=" << stability.bboxAreaRatio
                  << " mask_area_ratio=" << stability.maskAreaRatio
                  << " feature_density_ratio=" << stability.featureDensityRatio
                  << " current_tracklet_coverage=" << stability.currentTrackletCoverage
                  << " current_tracklet_spread_x=" << stability.currentTrackletSpreadX
                  << " current_tracklet_spread_y=" << stability.currentTrackletSpreadY
                  << " median_flow_tau2_tau1_px=" << stability.medianFlowTau2Tau1Px
                  << " median_flow_tau1_tau_px=" << stability.medianFlowTau1TauPx
                  << " median_flow_accel_px=" << stability.medianFlowAccelerationPx
                  << " median_flow_accel_norm=" << stability.medianFlowAccelerationNormalized
                  << " median_flow_direction_cos=" << stability.medianFlowDirectionCosine
                  << " triangulation_samples=" << stability.triangulationSamples
                  << " triangulation_success_tau2_tau1=" << stability.triangulationSuccessRatioTau2Tau1
                  << " triangulation_success_tau1_tau=" << stability.triangulationSuccessRatioTau1Tau
                  << " triangulation_nonpositive_ratio=" << stability.triangulationNonpositiveRatio
                  << " triangulation_high_reproj_ratio=" << stability.triangulationHighReprojectionRatio
                  << " triangulation_invalid_geometry_ratio=" << stability.triangulationInvalidGeometryRatio
                  << " matches_tau2_tau1=" << matchesTau2Tau1.size()
                  << " matches_tau1_tau=" << matchesTau1Tau.size()
                  << " mean_match_distance_tau2_tau1=" << matchMeanDistance(matchesTau2Tau1)
                  << " mean_match_distance_tau1_tau=" << matchMeanDistance(matchesTau1Tau)
                  << " max_match_distance_tau2_tau1=" << matchMaxDistance(matchesTau2Tau1)
                  << " max_match_distance_tau1_tau=" << matchMaxDistance(matchesTau1Tau)
                  << " strict_descriptor_gate=" << (strictDescriptorGate ? 1 : 0)
                  << " descriptor_max_distance=" << descriptorMaxDistance
                  << " descriptor_ratio=" << descriptorRatio
                  << " bridge_tau1_to_tau2=" << bridgeTau1ToTau2.size()
                  << " bridge_tau1_to_tau=" << bridgeTau1ToTau.size()
                  << " raw_tracklets=" << rawTrackletCount
                  << " tracklets=" << tracklets.size()
                  << " motion_coherence_gate=" << (EnableInstanceTrackletMotionCoherenceGate() ? 1 : 0)
                  << " motion_coherence_rejected=" << motionCoherenceRejected
                  << " motion_gate_tau2_tau1_px=" << motionCoherenceGateTau2Tau1
                  << " motion_gate_tau1_tau_px=" << motionCoherenceGateTau1Tau
                  << " tracklet_retention=" << stability.trackletRetention
                  << std::endl;
    }
    return tracklets;
}

bool Tracking::TriangulateMatchedFeatures(const Frame& firstFrame,
                                          int firstIdx,
                                          const Frame& secondFrame,
                                          int secondIdx,
                                          Eigen::Vector3f& pointWorld,
                                          TriangulationQuality* pQuality) const
{
    const bool debugTriangulation =
        DebugTriangulationQuality() || DebugInstanceInitialization() ||
        DebugFocusFrame(mCurrentFrame.mnId);
    double qualityWeightForLog = std::numeric_limits<double>::quiet_NaN();
    const cv::Point2f invalidPoint(-1.0f, -1.0f);
    const cv::Point2f kpFirst =
        (firstIdx >= 0 && firstIdx < static_cast<int>(firstFrame.mvKeysUn.size())) ?
        firstFrame.mvKeysUn[firstIdx].pt : invalidPoint;
    const cv::Point2f kpSecond =
        (secondIdx >= 0 && secondIdx < static_cast<int>(secondFrame.mvKeysUn.size())) ?
        secondFrame.mvKeysUn[secondIdx].pt : invalidPoint;
    const float disparity = (kpFirst == invalidPoint || kpSecond == invalidPoint) ?
        -1.0f : cv::norm(kpFirst - kpSecond);
    auto logTriangulation = [&](const char* status,
                                const char* reason,
                                const float homogeneousW,
                                const Eigen::Vector3f& worldPoint,
                                const Eigen::Vector3f& firstCameraPoint,
                                const Eigen::Vector3f& secondCameraPoint,
                                const double reprojectionErrorFirst,
                                const double reprojectionErrorSecond,
                                const double parallaxDeg)
    {
        if(!debugTriangulation)
            return;

        std::cout << "[STSLAM_TRIANGULATION] status=" << status
                  << " reason=" << reason
                  << " first_frame=" << firstFrame.mnId
                  << " second_frame=" << secondFrame.mnId
                  << " first_idx=" << firstIdx
                  << " second_idx=" << secondIdx
                  << " disparity_px=" << disparity
                  << " homogeneous_w=" << homogeneousW
                  << " world_point=" << worldPoint.transpose()
                  << " first_depth=" << firstCameraPoint[2]
                  << " second_depth=" << secondCameraPoint[2]
                  << " reproj_error_first=" << reprojectionErrorFirst
                  << " reproj_error_second=" << reprojectionErrorSecond
                  << " parallax_deg=" << parallaxDeg
                  << " quality_weight=" << qualityWeightForLog
                  << std::endl;
    };

    if(firstIdx < 0 || secondIdx < 0 ||
       firstIdx >= static_cast<int>(firstFrame.mvKeysUn.size()) ||
       secondIdx >= static_cast<int>(secondFrame.mvKeysUn.size()) ||
       !firstFrame.isSet() || !secondFrame.isSet())
    {
        if(pQuality)
            pQuality->rejectReason = "invalid_input";
        logTriangulation("failed",
                         "invalid_input",
                         0.0f,
                         Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN()),
                         Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN()),
                         Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN()),
                         std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::quiet_NaN());
        return false;
    }

    const cv::Mat projectionFirst = ToProjectionMatrix(firstFrame.mK_ * firstFrame.GetPose().matrix3x4());
    const cv::Mat projectionSecond = ToProjectionMatrix(secondFrame.mK_ * secondFrame.GetPose().matrix3x4());

    std::vector<cv::Point2f> pointsFirst(1, firstFrame.mvKeysUn[firstIdx].pt);
    std::vector<cv::Point2f> pointsSecond(1, secondFrame.mvKeysUn[secondIdx].pt);
    cv::Mat points4D;
    cv::triangulatePoints(projectionFirst, projectionSecond, pointsFirst, pointsSecond, points4D);

    const float homogeneousW = points4D.at<float>(3, 0);
    if(std::fabs(homogeneousW) < 1e-6f)
    {
        if(pQuality)
            pQuality->rejectReason = "degenerate_homogeneous_w";
        logTriangulation("failed",
                         "degenerate_homogeneous_w",
                         homogeneousW,
                         Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN()),
                         Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN()),
                         Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN()),
                         std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::quiet_NaN());
        return false;
    }

    pointWorld = Eigen::Vector3f(points4D.at<float>(0, 0) / homogeneousW,
                                 points4D.at<float>(1, 0) / homogeneousW,
                                 points4D.at<float>(2, 0) / homogeneousW);
    if(!pointWorld.allFinite())
    {
        if(pQuality)
            pQuality->rejectReason = "nonfinite_world_point";
        logTriangulation("failed",
                         "nonfinite_world_point",
                         homogeneousW,
                         pointWorld,
                         Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN()),
                         Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN()),
                         std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::quiet_NaN());
        return false;
    }

    const Eigen::Vector3f firstCameraPoint = firstFrame.GetPose() * pointWorld;
    const Eigen::Vector3f secondCameraPoint = secondFrame.GetPose() * pointWorld;
    if(firstCameraPoint[2] <= 0.0f || secondCameraPoint[2] <= 0.0f)
    {
        if(pQuality)
            pQuality->rejectReason = "nonpositive_depth";
        logTriangulation("failed",
                         "nonpositive_depth",
                         homogeneousW,
                         pointWorld,
                         firstCameraPoint,
                         secondCameraPoint,
                         std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::quiet_NaN());
        return false;
    }

    double reprojectionErrorFirst = std::numeric_limits<double>::quiet_NaN();
    double reprojectionErrorSecond = std::numeric_limits<double>::quiet_NaN();
    if(firstFrame.mpCamera && secondFrame.mpCamera)
    {
        const Eigen::Vector3d firstCameraPointD = firstCameraPoint.cast<double>();
        const Eigen::Vector3d secondCameraPointD = secondCameraPoint.cast<double>();
        const Eigen::Vector2d firstProjection =
            firstFrame.mpCamera->project(firstCameraPointD);
        const Eigen::Vector2d secondProjection =
            secondFrame.mpCamera->project(secondCameraPointD);
        if(firstProjection.allFinite())
        {
            const Eigen::Vector2d firstObservation(kpFirst.x, kpFirst.y);
            reprojectionErrorFirst = (firstObservation - firstProjection).norm();
        }
        if(secondProjection.allFinite())
        {
            const Eigen::Vector2d secondObservation(kpSecond.x, kpSecond.y);
            reprojectionErrorSecond = (secondObservation - secondProjection).norm();
        }
    }

    double parallaxDeg = std::numeric_limits<double>::quiet_NaN();
    const Eigen::Vector3f firstRay = pointWorld - firstFrame.GetOw();
    const Eigen::Vector3f secondRay = pointWorld - secondFrame.GetOw();
    const float firstRayNorm = firstRay.norm();
    const float secondRayNorm = secondRay.norm();
    if(firstRayNorm > 1e-6f && secondRayNorm > 1e-6f)
    {
        double cosine =
            static_cast<double>(firstRay.dot(secondRay)) /
            static_cast<double>(firstRayNorm * secondRayNorm);
        cosine = std::max(-1.0, std::min(1.0, cosine));
        parallaxDeg = std::acos(cosine) * 180.0 / 3.14159265358979323846;
    }

    TriangulationQuality quality;
    quality.minDepth =
        std::min(static_cast<double>(firstCameraPoint[2]),
                 static_cast<double>(secondCameraPoint[2]));
    quality.maxReprojectionError =
        std::max(reprojectionErrorFirst, reprojectionErrorSecond);
    quality.parallaxDeg = parallaxDeg;

    const double minDepth = GetTriangulationMinDepth();
    const double maxReprojectionError = GetTriangulationMaxReprojectionError();
    const double minParallaxDeg = GetTriangulationMinParallaxDeg();
    const double maxParallaxDeg = GetTriangulationMaxParallaxDeg();
    const double maxDisparity = GetTriangulationMaxDisparity();
    const double minQualityWeight = GetTriangulationMinQualityWeight();
    const double qualityParallaxDeg = GetTriangulationQualityParallaxDeg();
    double qualityWeight = 1.0;
    auto applyQualityPenalty = [&](double ratio)
    {
        if(!EnableTriangulationQualityWeights())
            return;
        if(!std::isfinite(ratio))
        {
            qualityWeight *= minQualityWeight;
            return;
        }
        qualityWeight *= std::max(minQualityWeight, std::min(1.0, ratio));
    };

    if(minDepth > 0.0 && quality.minDepth < minDepth)
    {
        quality.lowDepth = true;
        applyQualityPenalty(quality.minDepth / minDepth);
    }

    if(maxReprojectionError > 0.0 &&
       (!std::isfinite(quality.maxReprojectionError) ||
        quality.maxReprojectionError > maxReprojectionError))
    {
        quality.highReprojectionError = true;
        applyQualityPenalty(maxReprojectionError / quality.maxReprojectionError);
    }

    if(minParallaxDeg > 0.0 &&
       (!std::isfinite(quality.parallaxDeg) || quality.parallaxDeg < minParallaxDeg))
    {
        quality.lowParallax = true;
        applyQualityPenalty(quality.parallaxDeg / minParallaxDeg);
    }

    if(maxParallaxDeg > 0.0 &&
       std::isfinite(quality.parallaxDeg) &&
       quality.parallaxDeg > maxParallaxDeg)
    {
        quality.highParallax = true;
        applyQualityPenalty(maxParallaxDeg / quality.parallaxDeg);
    }

    if(maxDisparity > 0.0 &&
       std::isfinite(static_cast<double>(disparity)) &&
       static_cast<double>(disparity) > maxDisparity)
    {
        quality.highDisparity = true;
        applyQualityPenalty(maxDisparity / static_cast<double>(disparity));
    }

    if(qualityParallaxDeg > 0.0 &&
       std::isfinite(quality.parallaxDeg) &&
       quality.parallaxDeg < qualityParallaxDeg)
    {
        applyQualityPenalty(quality.parallaxDeg / qualityParallaxDeg);
    }

    quality.weight = std::max(minQualityWeight, std::min(1.0, qualityWeight));
    qualityWeightForLog = quality.weight;
    if(pQuality)
        *pQuality = quality;

    if(EnableTriangulationQualityGate())
    {
        if(quality.lowDepth)
        {
            quality.rejectReason = "low_depth";
            if(pQuality)
                *pQuality = quality;
            logTriangulation("rejected",
                             "low_depth",
                             homogeneousW,
                             pointWorld,
                             firstCameraPoint,
                             secondCameraPoint,
                             reprojectionErrorFirst,
                             reprojectionErrorSecond,
                             parallaxDeg);
            return false;
        }

        if(quality.highReprojectionError)
        {
            quality.rejectReason = "high_reprojection_error";
            if(pQuality)
                *pQuality = quality;
            logTriangulation("rejected",
                             "high_reprojection_error",
                             homogeneousW,
                             pointWorld,
                             firstCameraPoint,
                             secondCameraPoint,
                             reprojectionErrorFirst,
                             reprojectionErrorSecond,
                             parallaxDeg);
            return false;
        }

        if(quality.lowParallax)
        {
            quality.rejectReason = "low_parallax";
            if(pQuality)
                *pQuality = quality;
            logTriangulation("rejected",
                             "low_parallax",
                             homogeneousW,
                             pointWorld,
                             firstCameraPoint,
                             secondCameraPoint,
                             reprojectionErrorFirst,
                             reprojectionErrorSecond,
                             parallaxDeg);
            return false;
        }

        if(quality.highParallax)
        {
            quality.rejectReason = "high_parallax";
            if(pQuality)
                *pQuality = quality;
            logTriangulation("rejected",
                             "high_parallax",
                             homogeneousW,
                             pointWorld,
                             firstCameraPoint,
                             secondCameraPoint,
                             reprojectionErrorFirst,
                             reprojectionErrorSecond,
                             parallaxDeg);
            return false;
        }

        if(quality.highDisparity)
        {
            quality.rejectReason = "high_disparity";
            if(pQuality)
                *pQuality = quality;
            logTriangulation("rejected",
                             "high_disparity",
                             homogeneousW,
                             pointWorld,
                             firstCameraPoint,
                             secondCameraPoint,
                             reprojectionErrorFirst,
                             reprojectionErrorSecond,
                             parallaxDeg);
            return false;
        }
    }

    quality.success = true;
    quality.rejectReason.clear();
    if(pQuality)
        *pQuality = quality;

    logTriangulation("ok",
                     "success",
                     homogeneousW,
                     pointWorld,
                     firstCameraPoint,
                     secondCameraPoint,
                     reprojectionErrorFirst,
                     reprojectionErrorSecond,
                     parallaxDeg);
    return true;
}

Tracking::TriFrameConsistencyQuality Tracking::EvaluateTriFrameConsistency(
    const Frame& frameTau2,
    int idxTau2,
    const Frame& frameTau1,
    int idxTau1,
    const Frame& frameTau,
    int idxTau,
    const Sophus::SE3f& velocity,
    const Eigen::Vector3f& pointTau2,
    const Eigen::Vector3f& pointTau1) const
{
    TriFrameConsistencyQuality quality;
    if(idxTau2 < 0 || idxTau1 < 0 || idxTau < 0 ||
       idxTau2 >= static_cast<int>(frameTau2.mvKeysUn.size()) ||
       idxTau1 >= static_cast<int>(frameTau1.mvKeysUn.size()) ||
       idxTau >= static_cast<int>(frameTau.mvKeysUn.size()) ||
       !frameTau2.HasPose() || !frameTau1.HasPose() || !frameTau.HasPose() ||
       !frameTau2.mpCamera || !frameTau1.mpCamera || !frameTau.mpCamera ||
       !IsFiniteSE3(velocity) || !pointTau2.allFinite() || !pointTau1.allFinite())
    {
        return quality;
    }

    auto reprojectionError = [](const Frame& frame,
                                const int featureIdx,
                                const Eigen::Vector3f& pointWorld,
                                bool& negativeDepth) -> double
    {
        if(featureIdx < 0 || featureIdx >= static_cast<int>(frame.mvKeysUn.size()) ||
           !frame.HasPose() || !frame.mpCamera || !pointWorld.allFinite())
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const Eigen::Vector3f pointCamera = frame.GetPose() * pointWorld;
        if(!pointCamera.allFinite() || pointCamera[2] <= 0.0f)
        {
            negativeDepth = true;
            return std::numeric_limits<double>::quiet_NaN();
        }

        const Eigen::Vector3d pointCameraD = pointCamera.cast<double>();
        const Eigen::Vector2d projection =
            frame.mpCamera->project(pointCameraD);
        if(!projection.allFinite())
            return std::numeric_limits<double>::quiet_NaN();

        const cv::Point2f& observation = frame.mvKeysUn[featureIdx].pt;
        return (Eigen::Vector2d(observation.x, observation.y) - projection).norm();
    };

    bool negativeDepth = false;
    const double tau2PairError =
        reprojectionError(frameTau2, idxTau2, pointTau2, negativeDepth);
    const double tau1FromTau2PairError =
        reprojectionError(frameTau1, idxTau1, pointTau2, negativeDepth);
    const double tau1PairError =
        reprojectionError(frameTau1, idxTau1, pointTau1, negativeDepth);
    const double tauFromTau1PairError =
        reprojectionError(frameTau, idxTau, pointTau1, negativeDepth);
    const Eigen::Vector3f dynamicPointTau = velocity * pointTau2;
    const double dynamicTauError =
        reprojectionError(frameTau, idxTau, dynamicPointTau, negativeDepth);

    quality.valid = true;
    quality.negativeDepth = negativeDepth;
    quality.pairTau2Tau1ReprojectionError =
        std::max(std::isfinite(tau2PairError) ? tau2PairError : 0.0,
                 std::isfinite(tau1FromTau2PairError) ? tau1FromTau2PairError : 0.0);
    quality.pairTau1TauReprojectionError =
        std::max(std::isfinite(tau1PairError) ? tau1PairError : 0.0,
                 std::isfinite(tauFromTau1PairError) ? tauFromTau1PairError : 0.0);
    quality.dynamicReprojectionError =
        std::isfinite(dynamicTauError) ? dynamicTauError :
        std::numeric_limits<double>::infinity();
    quality.motion3dResidual = (dynamicPointTau - pointTau1).cast<double>().norm();
    quality.maxReprojectionError =
        std::max(quality.dynamicReprojectionError,
                 std::max(quality.pairTau2Tau1ReprojectionError,
                          quality.pairTau1TauReprojectionError));

    const double maxReprojection = GetTriFrameConsistencyMaxReprojectionPx();
    const double minWeight = GetTriFrameConsistencyMinWeight();
    if(quality.negativeDepth || !std::isfinite(quality.maxReprojectionError))
    {
        quality.weight = minWeight;
    }
    else if(maxReprojection > 0.0 && quality.maxReprojectionError > maxReprojection)
    {
        quality.weight =
            std::max(minWeight, std::min(1.0, maxReprojection / quality.maxReprojectionError));
    }
    else
    {
        quality.weight = 1.0;
    }

    return quality;
}

Tracking::TriFrameTrackletCandidateQuality Tracking::EvaluateTriFrameTrackletCandidate(
    const Frame& frameTau2,
    int idxTau2,
    const Frame& frameTau1,
    int idxTau1,
    const Frame& frameTau,
    int idxTau,
    const TriangulationQuality& qualityTau2Tau1,
    const TriangulationQuality& qualityTau1Tau) const
{
    TriFrameTrackletCandidateQuality quality;
    if(idxTau2 < 0 || idxTau1 < 0 || idxTau < 0 ||
       idxTau2 >= static_cast<int>(frameTau2.mvKeysUn.size()) ||
       idxTau1 >= static_cast<int>(frameTau1.mvKeysUn.size()) ||
       idxTau >= static_cast<int>(frameTau.mvKeysUn.size()))
    {
        quality.valid = false;
        quality.structureEligible = false;
        quality.weight = GetTriFrameConsistencyMinWeight();
        return quality;
    }

    quality.valid = true;
    quality.lowDepth = qualityTau2Tau1.lowDepth || qualityTau1Tau.lowDepth;
    quality.highReprojectionError =
        qualityTau2Tau1.highReprojectionError || qualityTau1Tau.highReprojectionError;
    quality.pairMaxReprojectionError =
        std::max(std::isfinite(qualityTau2Tau1.maxReprojectionError) ?
                 qualityTau2Tau1.maxReprojectionError : 0.0,
                 std::isfinite(qualityTau1Tau.maxReprojectionError) ?
                 qualityTau1Tau.maxReprojectionError : 0.0);

    const cv::Point2f& pTau2 = frameTau2.mvKeysUn[idxTau2].pt;
    const cv::Point2f& pTau1 = frameTau1.mvKeysUn[idxTau1].pt;
    const cv::Point2f& pTau = frameTau.mvKeysUn[idxTau].pt;
    const cv::Point2d flowTau2Tau1(static_cast<double>(pTau1.x - pTau2.x),
                                   static_cast<double>(pTau1.y - pTau2.y));
    const cv::Point2d flowTau1Tau(static_cast<double>(pTau.x - pTau1.x),
                                  static_cast<double>(pTau.y - pTau1.y));
    quality.flowAccelerationPx = cv::norm(flowTau1Tau - flowTau2Tau1);
    const double flowScale =
        std::max(1.0, std::max(cv::norm(flowTau2Tau1), cv::norm(flowTau1Tau)));
    quality.flowAccelerationNormalized = quality.flowAccelerationPx / flowScale;

    double weight =
        std::min(std::isfinite(qualityTau2Tau1.weight) ? qualityTau2Tau1.weight : 1.0,
                 std::isfinite(qualityTau1Tau.weight) ? qualityTau1Tau.weight : 1.0);
    const double maxReprojection = GetTriFrameConsistencyMaxReprojectionPx();
    if(maxReprojection > 0.0 &&
       std::isfinite(quality.pairMaxReprojectionError) &&
       quality.pairMaxReprojectionError > maxReprojection)
    {
        quality.highReprojectionError = true;
        weight *= std::max(GetTriFrameConsistencyMinWeight(),
                           maxReprojection / quality.pairMaxReprojectionError);
    }

    const double maxFlowAccelNorm = GetTriFrameCandidateMaxFlowAccelerationNormalized();
    if(maxFlowAccelNorm > 0.0 &&
       std::isfinite(quality.flowAccelerationNormalized) &&
       quality.flowAccelerationNormalized > maxFlowAccelNorm)
    {
        quality.highFlowAcceleration = true;
        weight *= std::max(GetTriFrameConsistencyMinWeight(),
                           maxFlowAccelNorm / quality.flowAccelerationNormalized);
    }

    if(quality.lowDepth)
        weight *= GetTriFrameConsistencyMinWeight();

    quality.weight =
        std::max(GetTriFrameConsistencyMinWeight(), std::min(1.0, weight));
    quality.structureEligible =
        !quality.lowDepth &&
        !quality.highReprojectionError &&
        !quality.highFlowAcceleration &&
        quality.weight >= GetTriFrameCandidateMinStructureWeight();

    return quality;
}

Sophus::SE3f Tracking::SolveRigidTransformSVD(const std::vector<Eigen::Vector3f>& src,
                                              const std::vector<Eigen::Vector3f>& dst) const
{
    if(src.size() != dst.size() || src.empty())
        return Sophus::SE3f();

    Eigen::Vector3f srcCentroid = Eigen::Vector3f::Zero();
    Eigen::Vector3f dstCentroid = Eigen::Vector3f::Zero();
    for(size_t idx = 0; idx < src.size(); ++idx)
    {
        srcCentroid += src[idx];
        dstCentroid += dst[idx];
    }
    srcCentroid /= static_cast<float>(src.size());
    dstCentroid /= static_cast<float>(dst.size());
    if(!srcCentroid.allFinite() || !dstCentroid.allFinite())
        return Sophus::SE3f();

    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for(size_t idx = 0; idx < src.size(); ++idx)
    {
        covariance += (src[idx] - srcCentroid) * (dst[idx] - dstCentroid).transpose();
    }
    if(!covariance.allFinite())
        return Sophus::SE3f();

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    if(!svd.matrixU().allFinite() || !svd.matrixV().allFinite())
        return Sophus::SE3f();
    Eigen::Matrix3f rotation = svd.matrixV() * svd.matrixU().transpose();
    if(rotation.determinant() < 0.0f)
    {
        Eigen::Matrix3f correction = Eigen::Matrix3f::Identity();
        correction(2, 2) = -1.0f;
        rotation = svd.matrixV() * correction * svd.matrixU().transpose();
    }
    if(!rotation.allFinite())
        return Sophus::SE3f();

    const Eigen::Vector3f translation = dstCentroid - rotation * srcCentroid;
    if(!translation.allFinite())
        return Sophus::SE3f();
    return Sophus::SE3f(rotation, translation);
}

Sophus::SE3f Tracking::SolveWeightedRigidTransformSVD(const std::vector<Eigen::Vector3f>& src,
                                                       const std::vector<Eigen::Vector3f>& dst,
                                                       const std::vector<double>& weights) const
{
    if(src.size() != dst.size() || src.empty() || weights.size() != src.size())
        return SolveRigidTransformSVD(src, dst);

    double totalWeight = 0.0;
    Eigen::Vector3d srcCentroid = Eigen::Vector3d::Zero();
    Eigen::Vector3d dstCentroid = Eigen::Vector3d::Zero();
    for(size_t idx = 0; idx < src.size(); ++idx)
    {
        const double weight = weights[idx];
        if(!std::isfinite(weight) || weight <= 0.0)
            continue;
        srcCentroid += weight * src[idx].cast<double>();
        dstCentroid += weight * dst[idx].cast<double>();
        totalWeight += weight;
    }

    if(totalWeight <= 0.0)
        return Sophus::SE3f();

    srcCentroid /= totalWeight;
    dstCentroid /= totalWeight;
    if(!srcCentroid.allFinite() || !dstCentroid.allFinite())
        return Sophus::SE3f();

    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for(size_t idx = 0; idx < src.size(); ++idx)
    {
        const double weight = weights[idx];
        if(!std::isfinite(weight) || weight <= 0.0)
            continue;
        covariance += weight *
            (src[idx].cast<double>() - srcCentroid) *
            (dst[idx].cast<double>() - dstCentroid).transpose();
    }
    if(!covariance.allFinite())
        return Sophus::SE3f();

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    if(!svd.matrixU().allFinite() || !svd.matrixV().allFinite())
        return Sophus::SE3f();

    Eigen::Matrix3d rotation = svd.matrixV() * svd.matrixU().transpose();
    if(rotation.determinant() < 0.0)
    {
        Eigen::Matrix3d correction = Eigen::Matrix3d::Identity();
        correction(2, 2) = -1.0;
        rotation = svd.matrixV() * correction * svd.matrixU().transpose();
    }
    if(!rotation.allFinite())
        return Sophus::SE3f();

    const Eigen::Vector3d translation = dstCentroid - rotation * srcCentroid;
    if(!translation.allFinite())
        return Sophus::SE3f();
    return Sophus::SE3f(rotation.cast<float>(), translation.cast<float>());
}


void Tracking::GrabImuData(const IMU::Point &imuMeasurement)
{
    unique_lock<mutex> lock(mMutexImuQueue);
    mlQueueImuData.push_back(imuMeasurement);
}

void Tracking::PreintegrateIMU()
{

    if(!mCurrentFrame.mpPrevFrame)
    {
        Verbose::PrintMess("non prev frame ", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated();
        return;
    }

    mvImuFromLastFrame.clear();
    mvImuFromLastFrame.reserve(mlQueueImuData.size());
    if(mlQueueImuData.size() == 0)
    {
        Verbose::PrintMess("Not IMU data in mlQueueImuData!!", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated();
        return;
    }

    while(true)
    {
        bool bSleep = false;
        {
            unique_lock<mutex> lock(mMutexImuQueue);
            if(!mlQueueImuData.empty())
            {
                IMU::Point* m = &mlQueueImuData.front();
                cout.precision(17);
                if(m->t<mCurrentFrame.mpPrevFrame->mTimeStamp-mImuPer)
                {
                    mlQueueImuData.pop_front();
                }
                else if(m->t<mCurrentFrame.mTimeStamp-mImuPer)
                {
                    mvImuFromLastFrame.push_back(*m);
                    mlQueueImuData.pop_front();
                }
                else
                {
                    mvImuFromLastFrame.push_back(*m);
                    break;
                }
            }
            else
            {
                break;
                bSleep = true;
            }
        }
        if(bSleep)
            usleep(500);
    }

    const int n = mvImuFromLastFrame.size()-1;
    if(n==0){
        cout << "Empty IMU measurements vector!!!\n";
        return;
    }

    IMU::Preintegrated* pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias,mCurrentFrame.mImuCalib);

    for(int i=0; i<n; i++)
    {
        float tstep;
        Eigen::Vector3f acc, angVel;
        if((i==0) && (i<(n-1)))
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tini = mvImuFromLastFrame[i].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tini/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tini/tab))*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
        }
        else if(i<(n-1))
        {
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a)*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w)*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
        }
        else if((i>0) && (i==(n-1)))
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tend = mvImuFromLastFrame[i+1].t-mCurrentFrame.mTimeStamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tend/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tend/tab))*0.5f;
            tstep = mCurrentFrame.mTimeStamp-mvImuFromLastFrame[i].t;
        }
        else if((i==0) && (i==(n-1)))
        {
            acc = mvImuFromLastFrame[i].a;
            angVel = mvImuFromLastFrame[i].w;
            tstep = mCurrentFrame.mTimeStamp-mCurrentFrame.mpPrevFrame->mTimeStamp;
        }

        if (!mpImuPreintegratedFromLastKF)
            cout << "mpImuPreintegratedFromLastKF does not exist" << endl;
        mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc,angVel,tstep);
        pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc,angVel,tstep);
    }

    mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
    mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
    mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame;

    mCurrentFrame.setIntegrated();

    //Verbose::PrintMess("Preintegration is finished!! ", Verbose::VERBOSITY_DEBUG);
}


bool Tracking::PredictStateIMU()
{
    if(!mCurrentFrame.mpPrevFrame)
    {
        Verbose::PrintMess("No last frame", Verbose::VERBOSITY_NORMAL);
        return false;
    }

    if(mbMapUpdated && mpLastKeyFrame)
    {
        const Eigen::Vector3f twb1 = mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mpLastKeyFrame->GetVelocity();

        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float t12 = mpImuPreintegratedFromLastKF->dT;

        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
        Eigen::Vector3f twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
        Eigen::Vector3f Vwb2 = Vwb1 + t12*Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());
        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2);

        mCurrentFrame.mImuBias = mpLastKeyFrame->GetImuBias();
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    }
    else if(!mbMapUpdated)
    {
        const Eigen::Vector3f twb1 = mLastFrame.GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame.GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.GetVelocity();
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float t12 = mCurrentFrame.mpImuPreintegratedFrame->dT;

        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaRotation(mLastFrame.mImuBias));
        Eigen::Vector3f twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaPosition(mLastFrame.mImuBias);
        Eigen::Vector3f Vwb2 = Vwb1 + t12*Gz + Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaVelocity(mLastFrame.mImuBias);

        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2);

        mCurrentFrame.mImuBias = mLastFrame.mImuBias;
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    }
    else
        cout << "not IMU prediction!!" << endl;

    return false;
}

void Tracking::ResetFrameIMU()
{
    // TODO To implement...
}


void Tracking::Track()
{

    if (bStepByStep)
    {
        std::cout << "Tracking: Waiting to the next step" << std::endl;
        while(!mbStep && bStepByStep)
            usleep(500);
        mbStep = false;
    }

    if(mpLocalMapper->mbBadImu)
    {
        cout << "TRACK: Reset map because local mapper set the bad imu flag " << endl;
        mpSystem->ResetActiveMap();
        return;
    }

    Map* pCurrentMap = mpAtlas->GetCurrentMap();
    if(!pCurrentMap)
    {
        cout << "ERROR: There is not an active map in the atlas" << endl;
    }

    if(mState!=NO_IMAGES_YET)
    {
        if(mLastFrame.mTimeStamp>mCurrentFrame.mTimeStamp)
        {
            cerr << "ERROR: Frame with a timestamp older than previous frame detected!" << endl;
            unique_lock<mutex> lock(mMutexImuQueue);
            mlQueueImuData.clear();
            CreateMapInAtlas();
            return;
        }
        else if(mCurrentFrame.mTimeStamp>mLastFrame.mTimeStamp+1.0)
        {
            // cout << mCurrentFrame.mTimeStamp << ", " << mLastFrame.mTimeStamp << endl;
            // cout << "id last: " << mLastFrame.mnId << "    id curr: " << mCurrentFrame.mnId << endl;
            if(mpAtlas->isInertial())
            {

                if(mpAtlas->isImuInitialized())
                {
                    cout << "Timestamp jump detected. State set to LOST. Reseting IMU integration..." << endl;
                    if(!pCurrentMap->GetIniertialBA2())
                    {
                        mpSystem->ResetActiveMap();
                    }
                    else
                    {
                        CreateMapInAtlas();
                    }
                }
                else
                {
                    cout << "Timestamp jump detected, before IMU initialization. Reseting..." << endl;
                    mpSystem->ResetActiveMap();
                }
                return;
            }

        }
    }


    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpLastKeyFrame)
        mCurrentFrame.SetNewBias(mpLastKeyFrame->GetImuBias());

    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mbCreatedMap)
    {
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartPreIMU = std::chrono::steady_clock::now();
#endif
        PreintegrateIMU();
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndPreIMU = std::chrono::steady_clock::now();

        double timePreImu = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPreIMU - time_StartPreIMU).count();
        vdIMUInteg_ms.push_back(timePreImu);
#endif

    }
    mbCreatedMap = false;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);

    mbMapUpdated = false;

    int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex();
    int nMapChangeIndex = pCurrentMap->GetLastMapChange();
    if(nCurMapChangeIndex>nMapChangeIndex)
    {
        pCurrentMap->SetLastMapChange(nCurMapChangeIndex);
        mbMapUpdated = true;
    }


    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD)
        {
            StereoInitialization();
        }
        else
        {
            MonocularInitialization();
        }

        //mpFrameDrawer->Update(this);

        if(mState!=OK) // If rightly initialized, mState=OK
        {
            mLastFrame = Frame(mCurrentFrame);
            return;
        }

        if(mpAtlas->GetAllMaps().size() == 1)
        {
            mnFirstFrameId = mCurrentFrame.mnId;
        }
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK = false;

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartPosePred = std::chrono::steady_clock::now();
#endif

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)
        {

            // State OK
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            if(mState==OK)
            {

                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                if((!mbVelocity && !pCurrentMap->isImuInitialized()) || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    Verbose::PrintMess("TRACK: Track with respect to the reference KF ", Verbose::VERBOSITY_DEBUG);
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    Verbose::PrintMess("TRACK: Track with motion model", Verbose::VERBOSITY_DEBUG);
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();
                }


                if (!bOK)
                {
                    if ( mCurrentFrame.mnId<=(mnLastRelocFrameId+mnFramesToResetIMU) &&
                         (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD))
                    {
                        mState = LOST;
                    }
                    else if(pCurrentMap->KeyFramesInMap()>10)
                    {
                        // cout << "KF in map: " << pCurrentMap->KeyFramesInMap() << endl;
                        mState = RECENTLY_LOST;
                        mTimeStampLost = mCurrentFrame.mTimeStamp;
                    }
                    else
                    {
                        mState = LOST;
                    }
                }
            }
            else
            {

                if (mState == RECENTLY_LOST)
                {
                    Verbose::PrintMess("Lost for a short time", Verbose::VERBOSITY_NORMAL);

                    bOK = true;
                    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))
                    {
                        if(pCurrentMap->isImuInitialized())
                            PredictStateIMU();
                        else
                            bOK = false;

                        if (mCurrentFrame.mTimeStamp-mTimeStampLost>time_recently_lost)
                        {
                            mState = LOST;
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK=false;
                        }
                    }
                    else
                    {
                        // Relocalization
                        bOK = Relocalization();
                        //std::cout << "mCurrentFrame.mTimeStamp:" << to_string(mCurrentFrame.mTimeStamp) << std::endl;
                        //std::cout << "mTimeStampLost:" << to_string(mTimeStampLost) << std::endl;
                        if(mCurrentFrame.mTimeStamp-mTimeStampLost>3.0f && !bOK)
                        {
                            mState = LOST;
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK=false;
                        }
                    }
                }
                else if (mState == LOST)
                {

                    Verbose::PrintMess("A new map is started...", Verbose::VERBOSITY_NORMAL);

                    if (pCurrentMap->KeyFramesInMap()<10)
                    {
                        mpSystem->ResetActiveMap();
                        Verbose::PrintMess("Reseting current map...", Verbose::VERBOSITY_NORMAL);
                    }else
                        CreateMapInAtlas();

                    if(mpLastKeyFrame)
                        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

                    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

                    return;
                }
            }

        }
        else
        {
            // Localization Mode: Local Mapping is deactivated (TODO Not available in inertial mode)
            if(mState==LOST)
            {
                if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                    Verbose::PrintMess("IMU. State LOST", Verbose::VERBOSITY_NORMAL);
                bOK = Relocalization();
            }
            else
            {
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map
                    if(mbVelocity)
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    Sophus::SE3f TcwMM;
                    if(mbVelocity)
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.GetPose();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        bool bPanopticPoseApplied = false;
        Sophus::SE3f panopticFallbackPose;
        std::vector<bool> panopticFallbackOutliers;

        if(bOK &&
           mCurrentFrame.HasPanopticObservation() &&
           !EnablePanopticSideChannelOnly())
        {
            int currentMapAgeFrames = std::numeric_limits<int>::max();
            bool isAdditionalMap = false;
            Map* pCurrentMap = mpAtlas ? mpAtlas->GetCurrentMap() : static_cast<Map*>(NULL);
            if(pCurrentMap && pCurrentMap->GetOriginKF())
            {
                const long originFrameId =
                    static_cast<long>(pCurrentMap->GetOriginKF()->mnFrameId);
                isAdditionalMap = pCurrentMap->GetOriginKF()->mnId > 0;
                currentMapAgeFrames =
                    std::max(0, static_cast<int>(static_cast<long>(mCurrentFrame.mnId) - originFrameId));
            }
            const int instanceMapWarmupFrames = GetInstanceMapWarmupFrames();
            const bool mapWarmupFinished =
                !isAdditionalMap || currentMapAgeFrames >= instanceMapWarmupFrames;

            if(mapWarmupFinished)
            {
                ProcessInstances();
            }
            else if(DebugInstanceInitialization() || DebugInstanceTracklets() ||
                    DebugFocusFrame(mCurrentFrame.mnId))
            {
                std::cout << "[STSLAM_INSTANCE_WARMUP] frame=" << mCurrentFrame.mnId
                          << " skipped=1"
                          << " additional_map=" << (isAdditionalMap ? 1 : 0)
                          << " map_age_frames=" << currentMapAgeFrames
                          << " required=" << instanceMapWarmupFrames
                          << std::endl;
            }

            static const int kPanopticPoseWarmupFrames = []()
            {
                const char* envValue = std::getenv("STSLAM_PANOPTIC_POSE_WARMUP_FRAMES");
                return envValue ? std::max(0, std::atoi(envValue)) : 5;
            }();

            if(mapWarmupFinished &&
               !mCurrentFrame.mmPredictedInstanceMotions.empty() &&
               mCurrentFrame.mnId >= mnFirstFrameId + kPanopticPoseWarmupFrames)
            {
                panopticFallbackPose = mCurrentFrame.GetPose();
                panopticFallbackOutliers = mCurrentFrame.mvbOutlier;
                if(OptimizePoseWithPanoptic())
                    bPanopticPoseApplied = true;
                else
                {
                    mCurrentFrame.SetPose(panopticFallbackPose);
                    mCurrentFrame.mvbOutlier = panopticFallbackOutliers;
                }
            }
        }

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndPosePred = std::chrono::steady_clock::now();

        double timePosePred = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPosePred - time_StartPosePred).count();
        vdPosePred_ms.push_back(timePosePred);
#endif


#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartLMTrack = std::chrono::steady_clock::now();
#endif
        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)
        {
            if(bOK)
            {
                if(DebugFocusFrame(mCurrentFrame.mnId))
                {
                    std::cout << "[STSLAM_FOCUS] frame=" << mCurrentFrame.mnId
                              << " stage=before_local_map"
                              << " coarse_tracking_ok=1"
                              << " predicted_instances=" << mCurrentFrame.mmPredictedInstanceMotions.size()
                              << " current_matches=" << CountTrackedMapPoints(mCurrentFrame)
                              << " reference_kf=" << (mCurrentFrame.mpReferenceKF ? static_cast<long>(mCurrentFrame.mpReferenceKF->mnId) : -1)
                              << std::endl;
                }
                if(!EnablePanopticSideChannelOnly())
                    SplitRgbdDynamicFeatureMatches("before_local_map", true);
                ForceFilterDetectedDynamicFeatureMatches(mCurrentFrame, "before_local_map");
                bOK = TrackLocalMap();
                const bool bTrackLocalMapRecoveredByFallback = !bOK && bPanopticPoseApplied;
                if(!bOK && bPanopticPoseApplied)
                {
                    mCurrentFrame.SetPose(panopticFallbackPose);
                    mCurrentFrame.mvbOutlier = panopticFallbackOutliers;
                    if(!EnablePanopticSideChannelOnly())
                        SplitRgbdDynamicFeatureMatches("before_local_map_fallback", true);
                    ForceFilterDetectedDynamicFeatureMatches(mCurrentFrame, "before_local_map_fallback");
                    bOK = TrackLocalMap();
                }
                if(DebugPanopticFallback() && bPanopticPoseApplied)
                {
                    std::cout << "[STSLAM_DEBUG] frame=" << mCurrentFrame.mnId
                              << " predicted_instances=" << mCurrentFrame.mmPredictedInstanceMotions.size()
                              << " panoptic_pose_applied=1"
                              << " first_local_map_ok=" << (bTrackLocalMapRecoveredByFallback ? 0 : 1)
                              << " final_local_map_ok=" << (bOK ? 1 : 0)
                              << std::endl;
                }

            }
            if(!bOK)
            {
                if(DebugPanopticFallback())
                {
                    std::cout << "[STSLAM_DEBUG] frame=" << mCurrentFrame.mnId
                              << " predicted_instances=" << mCurrentFrame.mmPredictedInstanceMotions.size()
                              << " panoptic_pose_applied=" << (bPanopticPoseApplied ? 1 : 0)
                              << " local_map_failed=1" << std::endl;
                }
                cout << "Fail to track local map!"
                     << " frame=" << mCurrentFrame.mnId
                     << " current_matches=" << CountTrackedMapPoints(mCurrentFrame)
                     << " reference_kf=" << (mCurrentFrame.mpReferenceKF ? static_cast<long>(mCurrentFrame.mpReferenceKF->mnId) : -1)
                     << endl;
            }
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if(bOK)
            mState = OK;
        else if (mState == OK)
        {
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            {
                Verbose::PrintMess("Track lost for less than one second...", Verbose::VERBOSITY_NORMAL);
                if(!pCurrentMap->isImuInitialized() || !pCurrentMap->GetIniertialBA2())
                {
                    cout << "IMU is not or recently initialized. Reseting active map..." << endl;
                    mpSystem->ResetActiveMap();
                }

                mState=RECENTLY_LOST;
            }
            else
                mState=RECENTLY_LOST; // visual to lost

            /*if(mCurrentFrame.mnId>mnLastRelocFrameId+mMaxFrames)
            {*/
                mTimeStampLost = mCurrentFrame.mTimeStamp;
            //}
        }

        // Save frame if recent relocalization, since they are used for IMU reset (as we are making copy, it shluld be once mCurrFrame is completely modified)
        if((mCurrentFrame.mnId<(mnLastRelocFrameId+mnFramesToResetIMU)) && (mCurrentFrame.mnId > mnFramesToResetIMU) &&
           (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && pCurrentMap->isImuInitialized())
        {
            // TODO check this situation
            Verbose::PrintMess("Saving pointer to frame. imu needs reset...", Verbose::VERBOSITY_NORMAL);
            Frame* pF = new Frame(mCurrentFrame);
            pF->mpPrevFrame = new Frame(mLastFrame);

            // Load preintegration
            pF->mpImuPreintegratedFrame = new IMU::Preintegrated(mCurrentFrame.mpImuPreintegratedFrame);
        }

        if(pCurrentMap->isImuInitialized())
        {
            if(bOK)
            {
                if(mCurrentFrame.mnId==(mnLastRelocFrameId+mnFramesToResetIMU))
                {
                    cout << "RESETING FRAME!!!" << endl;
                    ResetFrameIMU();
                }
                else if(mCurrentFrame.mnId>(mnLastRelocFrameId+30))
                    mLastBias = mCurrentFrame.mImuBias;
            }
        }

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndLMTrack = std::chrono::steady_clock::now();

        double timeLMTrack = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLMTrack - time_StartLMTrack).count();
        vdLMTrack_ms.push_back(timeLMTrack);
#endif

        // Update drawer
        mpFrameDrawer->Update(this);
        if(mCurrentFrame.isSet())
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

        if(bOK || mState==RECENTLY_LOST)
        {
            // Update motion model
            if(mLastFrame.isSet() && mCurrentFrame.isSet())
            {
                Sophus::SE3f LastTwc = mLastFrame.GetPose().inverse();
                mVelocity = mCurrentFrame.GetPose() * LastTwc;
                mbVelocity = true;
            }
            else {
                mbVelocity = false;
            }

            if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                {
                    bool keepDynamicInstancePointForKeyFrame = false;
                    const int pointInstanceId = pMP->GetInstanceId();
                    if(pointInstanceId > 0)
                    {
                        const std::map<int, Sophus::SE3f>::const_iterator itMotion =
                            mCurrentFrame.mmPredictedInstanceMotions.find(pointInstanceId);
                        keepDynamicInstancePointForKeyFrame =
                            itMotion != mCurrentFrame.mmPredictedInstanceMotions.end() &&
                            !IsNearlyIdentityInstanceMotion(itMotion->second);
                    }

                    if(pMP->Observations()<1 && !keepDynamicInstancePointForKeyFrame)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
                }
            }

            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            SplitRgbdDynamicFeatureMatches("before_need_keyframe", true);
            SupplyRgbdDepthBackedDynamicObservations("before_need_keyframe");

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartNewKF = std::chrono::steady_clock::now();
#endif
            bool bNeedKF = NeedNewKeyFrame();

            // Check if we need to insert a new keyframe
            // if(bNeedKF && bOK)
            if(bNeedKF && (bOK || (mInsertKFsLost && mState==RECENTLY_LOST &&
                                   (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))))
            {
                if(!EnablePanopticSideChannelOnly())
                {
                    SplitRgbdDynamicFeatureMatches("before_create_keyframe", true);
                    SupplyRgbdDepthBackedDynamicObservations("before_create_keyframe");
                }
                ForceFilterDetectedDynamicFeatureMatches(mCurrentFrame, "before_create_keyframe");
                CreateNewKeyFrame();
            }

            for(int i = 0; i < mCurrentFrame.N; ++i)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP && pMP->Observations() < 1 && pMP->GetInstanceId() > 0)
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
            }

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndNewKF = std::chrono::steady_clock::now();

            double timeNewKF = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndNewKF - time_StartNewKF).count();
            vdNewKF_ms.push_back(timeNewKF);
#endif

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame. Only has effect if lastframe is tracked
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(pCurrentMap->KeyFramesInMap()<=10)
            {
                mpSystem->ResetActiveMap();
                return;
            }
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                if (!pCurrentMap->isImuInitialized())
                {
                    Verbose::PrintMess("Track lost before IMU initialisation, reseting...", Verbose::VERBOSITY_QUIET);
                    mpSystem->ResetActiveMap();
                    return;
                }

            CreateMapInAtlas();

            return;
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }




    if(mState==OK || mState==RECENTLY_LOST)
    {
        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        if(mCurrentFrame.isSet())
        {
            Sophus::SE3f Tcr_ = mCurrentFrame.GetPose() * mCurrentFrame.mpReferenceKF->GetPoseInverse();
            mlRelativeFramePoses.push_back(Tcr_);
            mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState==LOST);
        }
        else
        {
            // This can happen if tracking is lost
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState==LOST);
        }

    }

    if(EnableObservabilityLogging() && mCurrentFrame.N > 0)
    {
        WriteObservabilityFrameStats(f_observability_stats,
                                     mCurrentFrame,
                                     mState,
                                     mSensor,
                                     mbCurrentFrameCreatedKeyFrame);
    }

#ifdef REGISTER_LOOP
    if (Stop()) {

        // Safe area to stop
        while(isStopped())
        {
            usleep(3000);
        }
    }
#endif
}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            if (!mCurrentFrame.mpImuPreintegrated || !mLastFrame.mpImuPreintegrated)
            {
                cout << "not IMU meas" << endl;
                return;
            }

            if (!mFastInit && (mCurrentFrame.mpImuPreintegratedFrame->avgA-mLastFrame.mpImuPreintegratedFrame->avgA).norm()<0.5)
            {
                cout << "not enough acceleration" << endl;
                return;
            }

            if(mpImuPreintegratedFromLastKF)
                delete mpImuPreintegratedFromLastKF;

            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
            mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
        }

        // Set Frame pose to the origin (In case of inertial SLAM to imu)
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            Eigen::Matrix3f Rwb0 = mCurrentFrame.mImuCalib.mTcb.rotationMatrix();
            Eigen::Vector3f twb0 = mCurrentFrame.mImuCalib.mTcb.translation();
            Eigen::Vector3f Vwb0;
            Vwb0.setZero();
            mCurrentFrame.SetImuPoseVelocity(Rwb0, twb0, Vwb0);
        }
        else
            mCurrentFrame.SetPose(Sophus::SE3f());

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpAtlas->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        if(!mpCamera2){
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                float z = mCurrentFrame.mvDepth[i];
                if(z>0)
                {
                    Eigen::Vector3f x3D;
                    mCurrentFrame.UnprojectStereo(i, x3D);
                    MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());
                    pNewMP->AddObservation(pKFini,i);
                    pKFini->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                }
            }
        } else{
            for(int i = 0; i < mCurrentFrame.Nleft; i++){
                int rightIndex = mCurrentFrame.mvLeftToRightMatch[i];
                if(rightIndex != -1){
                    Eigen::Vector3f x3D = mCurrentFrame.mvStereo3Dpoints[i];

                    MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());

                    pNewMP->AddObservation(pKFini,i);
                    pNewMP->AddObservation(pKFini,rightIndex + mCurrentFrame.Nleft);

                    pKFini->AddMapPoint(pNewMP,i);
                    pKFini->AddMapPoint(pNewMP,rightIndex + mCurrentFrame.Nleft);

                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    mCurrentFrame.mvpMapPoints[rightIndex + mCurrentFrame.Nleft]=pNewMP;
                }
            }
        }

        Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);

        //cout << "Active map: " << mpAtlas->GetCurrentMap()->GetId() << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;
        //mnLastRelocFrameId = mCurrentFrame.mnId;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpAtlas->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

        mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

        mState=OK;
    }
}


void Tracking::MonocularInitialization()
{

    if(!mbReadyToInitializate)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)
        {

            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            if (mSensor == System::IMU_MONOCULAR)
            {
                if(mpImuPreintegratedFromLastKF)
                {
                    delete mpImuPreintegratedFromLastKF;
                }
                mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
                mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;

            }

            mbReadyToInitializate = true;

            return;
        }
    }
    else
    {
        if (((int)mCurrentFrame.mvKeys.size()<=100)||((mSensor == System::IMU_MONOCULAR)&&(mLastFrame.mTimeStamp-mInitialFrame.mTimeStamp>1.0)))
        {
            mbReadyToInitializate = false;

            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        if(nmatches<100)
        {
            mbReadyToInitializate = false;
            return;
        }

        Sophus::SE3f Tcw;
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if(mpCamera->ReconstructWithTwoViews(mInitialFrame.mvKeysUn,mCurrentFrame.mvKeysUn,mvIniMatches,Tcw,mvIniP3D,vbTriangulated))
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(Sophus::SE3f());
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
    }
}



void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

    if(mSensor == System::IMU_MONOCULAR)
        pKFini->mpImuPreintegrated = (IMU::Preintegrated*)(NULL);


    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpAtlas->AddKeyFrame(pKFini);
    mpAtlas->AddKeyFrame(pKFcur);

    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        Eigen::Vector3f worldPos;
        worldPos << mvIniP3D[i].x, mvIniP3D[i].y, mvIniP3D[i].z;
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpAtlas->GetCurrentMap());

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpAtlas->AddMapPoint(pMP);
    }


    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    std::set<MapPoint*> sMPs;
    sMPs = pKFini->GetMapPoints();

    // Bundle Adjustment
    Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);
    Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(),20);

    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth;
    if(mSensor == System::IMU_MONOCULAR)
        invMedianDepth = 4.0f/medianDepth; // 4.0f
    else
        invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<50) // TODO Check, originally 100 tracks
    {
        Verbose::PrintMess("Wrong initialization, reseting...", Verbose::VERBOSITY_QUIET);
        mpSystem->ResetActiveMap();
        return;
    }

    // Scale initial baseline
    Sophus::SE3f Tc2w = pKFcur->GetPose();
    Tc2w.translation() *= invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
            pMP->UpdateNormalAndDepth();
        }
    }

    if (mSensor == System::IMU_MONOCULAR)
    {
        pKFcur->mPrevKF = pKFini;
        pKFini->mNextKF = pKFcur;
        pKFcur->mpImuPreintegrated = mpImuPreintegratedFromLastKF;

        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKFcur->mpImuPreintegrated->GetUpdatedBias(),pKFcur->mImuCalib);
    }


    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);
    mpLocalMapper->mFirstTs=pKFcur->mTimeStamp;

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;
    //mnLastRelocFrameId = mInitialFrame.mnId;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpAtlas->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    // Compute here initial velocity
    vector<KeyFrame*> vKFs = mpAtlas->GetAllKeyFrames();

    Sophus::SE3f deltaT = vKFs.back()->GetPose() * vKFs.front()->GetPoseInverse();
    mbVelocity = false;
    Eigen::Vector3f phi = deltaT.so3().log();

    double aux = (mCurrentFrame.mTimeStamp-mLastFrame.mTimeStamp)/(mCurrentFrame.mTimeStamp-mInitialFrame.mTimeStamp);
    phi *= aux;

    mLastFrame = Frame(mCurrentFrame);

    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;

    initID = pKFcur->mnId;
}


void Tracking::CreateMapInAtlas()
{
    mnLastInitFrameId = mCurrentFrame.mnId;
    mpAtlas->CreateNewMap();
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
        mpAtlas->SetInertialSensor();
    mbSetInit=false;

    mnInitialFrameId = mCurrentFrame.mnId+1;
    mState = NO_IMAGES_YET;

    // Restart the variable with information about the last KF
    mbVelocity = false;
    //mnLastRelocFrameId = mnLastInitFrameId; // The last relocation KF_id is the current id, because it is the new starting point for new map
    Verbose::PrintMess("First frame id in map: " + to_string(mnLastInitFrameId+1), Verbose::VERBOSITY_NORMAL);
    mbVO = false; // Init value for know if there are enough MapPoints in the last KF
    if(mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
    {
        mbReadyToInitializate = false;
    }

    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpImuPreintegratedFromLastKF)
    {
        delete mpImuPreintegratedFromLastKF;
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
    }

    if(mpLastKeyFrame)
        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

    if(mpReferenceKF)
        mpReferenceKF = static_cast<KeyFrame*>(NULL);

    mRecentPanopticFrames.clear();
    mvFramesSinceLastKeyFrame.clear();
    mbCurrentFrameCreatedKeyFrame = false;

    mLastFrame = Frame();
    mCurrentFrame = Frame();
    mvIniMatches.clear();

    mbCreatedMap = true;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}


bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<15)
    {
        cout << "TRACK_REF_KF: Less than 15 matches!!\n";
        return false;
    }

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.GetPose());

    //mCurrentFrame.PrintPointDistribution();


    // cout << " TrackReferenceKeyFrame mLastFrame.mTcw:  " << mLastFrame.mTcw << endl;
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        //if(i >= mCurrentFrame.Nleft) break;
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                if(i < mCurrentFrame.Nleft){
                    pMP->mbTrackInView = false;
                }
                else{
                    pMP->mbTrackInViewR = false;
                }
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        return true;
    else
        return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    Sophus::SE3f Tlr = mlRelativeFramePoses.back();
    mLastFrame.SetPose(Tlr * pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    const int Nfeat = mLastFrame.Nleft == -1? mLastFrame.N : mLastFrame.Nleft;
    vDepthIdx.reserve(Nfeat);
    for(int i=0; i<Nfeat;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
            bCreateNew = true;

        if(bCreateNew)
        {
            Eigen::Vector3f x3D;

            if(mLastFrame.Nleft == -1){
                mLastFrame.UnprojectStereo(i, x3D);
            }
            else{
                x3D = mLastFrame.UnprojectStereoFishEye(i);
            }

            MapPoint* pNewMP = new MapPoint(x3D,mpAtlas->GetCurrentMap(),&mLastFrame,i);
            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;

    }
}

bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    if (mpAtlas->isImuInitialized() && (mCurrentFrame.mnId>mnLastRelocFrameId+mnFramesToResetIMU))
    {
        // Predict state with IMU if it is initialized and it doesnt need reset
        PredictStateIMU();
        return true;
    }
    else
    {
        mCurrentFrame.SetPose(mVelocity * mLastFrame.GetPose());
    }




    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;

    if(mSensor==System::STEREO)
        th=7;
    else
        th=15;

    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        Verbose::PrintMess("Not enough matches, wider window search!!", Verbose::VERBOSITY_NORMAL);
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);
        Verbose::PrintMess("Matches with wider search: " + to_string(nmatches), Verbose::VERBOSITY_NORMAL);

    }

    if(nmatches<20)
    {
        Verbose::PrintMess("Not enough matches!!", Verbose::VERBOSITY_NORMAL);
        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            return true;
        else
            return false;
    }

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                if(i < mCurrentFrame.Nleft){
                    pMP->mbTrackInView = false;
                }
                else{
                    pMP->mbTrackInViewR = false;
                }
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        return true;
    else
        return nmatchesMap>=10;
}

bool Tracking::TrackLocalMap()
{

    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    mTrackedFr++;

    if(DebugFocusFrame(mCurrentFrame.mnId))
    {
        const FrameFeatureDebugStats seedStats =
            CollectFrameFeatureDebugStats(mCurrentFrame, true, false, false);
        std::cout << "[STSLAM_FOCUS] frame=" << mCurrentFrame.mnId
                  << " stage=track_local_map_enter"
                  << " state=" << static_cast<int>(mState)
                  << " seed_matches=" << CountTrackedMapPoints(mCurrentFrame)
                  << " reference_kf=" << (mpReferenceKF ? static_cast<long>(mpReferenceKF->mnId) : -1)
                  << " seed_match_stats={" << FormatFrameFeatureDebugStats(seedStats) << "}"
                  << std::endl;
    }

    UpdateLocalMap();
    SearchLocalPoints();
    if(!EnablePanopticSideChannelOnly())
        SplitRgbdDynamicFeatureMatches("track_local_map_pre_pose", true);
    ForceFilterDetectedDynamicFeatureMatches(mCurrentFrame, "track_local_map_pre_pose");

    // TOO check outliers before PO
    int aux1 = 0, aux2=0;
    for(int i=0; i<mCurrentFrame.N; i++)
        if( mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            if(mCurrentFrame.mvbOutlier[i])
                aux2++;
        }

    int inliers;
    if (!mpAtlas->isImuInitialized())
        Optimizer::PoseOptimization(&mCurrentFrame);
    else
    {
        if(mCurrentFrame.mnId<=mnLastRelocFrameId+mnFramesToResetIMU)
        {
            Verbose::PrintMess("TLM: PoseOptimization ", Verbose::VERBOSITY_DEBUG);
            Optimizer::PoseOptimization(&mCurrentFrame);
        }
        else
        {
            // if(!mbMapUpdated && mState == OK) //  && (mnMatchesInliers>30))
            if(!mbMapUpdated) //  && (mnMatchesInliers>30))
            {
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastFrame ", Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
            else
            {
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastKeyFrame ", Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastKeyFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
        }
    }

    aux1 = 0, aux2 = 0;
    for(int i=0; i<mCurrentFrame.N; i++)
        if( mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            if(mCurrentFrame.mvbOutlier[i])
                aux2++;
        }

    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    mpLocalMapper->mnMatchesInliers=mnMatchesInliers;
    if(DebugFocusFrame(mCurrentFrame.mnId))
    {
        const FrameFeatureDebugStats matchedStats =
            CollectFrameFeatureDebugStats(mCurrentFrame, true, false, false);
        const FrameFeatureDebugStats inlierStats =
            CollectFrameFeatureDebugStats(mCurrentFrame, true, true, false);
        const FrameFeatureDebugStats outlierStats =
            CollectFrameFeatureDebugStats(mCurrentFrame, true, false, true);
        std::cout << "[STSLAM_FOCUS] frame=" << mCurrentFrame.mnId
                  << " stage=track_local_map_exit"
                  << " total_matches=" << aux1
                  << " outliers=" << aux2
                  << " inlier_map_matches=" << mnMatchesInliers
                  << " recent_reloc_guard=" << ((mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames) ? 1 : 0)
                  << " matched_stats={" << FormatFrameFeatureDebugStats(matchedStats) << "}"
                  << " inlier_stats={" << FormatFrameFeatureDebugStats(inlierStats) << "}"
                  << " outlier_stats={" << FormatFrameFeatureDebugStats(outlierStats) << "}"
                  << std::endl;
    }
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if((mnMatchesInliers>10)&&(mState==RECENTLY_LOST))
        return true;


    if (mSensor == System::IMU_MONOCULAR)
    {
        if((mnMatchesInliers<15 && mpAtlas->isImuInitialized())||(mnMatchesInliers<50 && !mpAtlas->isImuInitialized()))
        {
            return false;
        }
        else
            return true;
    }
    else if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
    {
        if(mnMatchesInliers<15)
        {
            return false;
        }
        else
            return true;
    }
    else
    {
        if(mnMatchesInliers<30)
            return false;
        else
            return true;
    }
}

bool Tracking::NeedNewKeyFrame()
{
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mpAtlas->GetCurrentMap()->isImuInitialized())
    {
        if (mSensor == System::IMU_MONOCULAR && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25)
            return true;
        else if ((mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25)
            return true;
        else
            return false;
    }

    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) {
        /*if(mSensor == System::MONOCULAR)
        {
            std::cout << "NeedNewKeyFrame: localmap stopped" << std::endl;
        }*/
        return false;
    }

    const int nKFs = mpAtlas->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
    {
        return false;
    }

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;

    if(mSensor!=System::MONOCULAR && mSensor!=System::IMU_MONOCULAR)
    {
        int N = (mCurrentFrame.Nleft == -1) ? mCurrentFrame.N : mCurrentFrame.Nleft;
        for(int i =0; i<N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;

            }
        }
        //Verbose::PrintMess("[NEEDNEWKF]-> closed points: " + to_string(nTrackedClose) + "; non tracked closed points: " + to_string(nNonTrackedClose), Verbose::VERBOSITY_NORMAL);// Verbose::VERBOSITY_DEBUG);
    }

    bool bNeedToInsertClose;
    bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    /*int nClosedPoints = nTrackedClose + nNonTrackedClose;
    const int thStereoClosedPoints = 15;
    if(nClosedPoints < thStereoClosedPoints && (mSensor==System::STEREO || mSensor==System::IMU_STEREO))
    {
        //Pseudo-monocular, there are not enough close points to be confident about the stereo observations.
        thRefRatio = 0.9f;
    }*/

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    if(mpCamera2) thRefRatio = 0.75f;

    if(mSensor==System::IMU_MONOCULAR)
    {
        if(mnMatchesInliers>350) // Points tracked from the local map
            thRefRatio = 0.75f;
        else
            thRefRatio = 0.90f;
    }

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = ((mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames) && bLocalMappingIdle); //mpLocalMapper->KeyframesInQueue() < 2);
    //Condition 1c: tracking is weak
    const bool c1c = mSensor!=System::MONOCULAR && mSensor!=System::IMU_MONOCULAR && mSensor!=System::IMU_STEREO && mSensor!=System::IMU_RGBD && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = (((mnMatchesInliers<nRefMatches*thRefRatio || bNeedToInsertClose)) && mnMatchesInliers>15);

    //std::cout << "NeedNewKF: c1a=" << c1a << "; c1b=" << c1b << "; c1c=" << c1c << "; c2=" << c2 << std::endl;
    // Temporal condition for Inertial cases
    bool c3 = false;
    if(mpLastKeyFrame)
    {
        if (mSensor==System::IMU_MONOCULAR)
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                c3 = true;
        }
        else if (mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                c3 = true;
        }
    }

    bool c4 = false;
    if ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && (mSensor == System::IMU_MONOCULAR)) // MODIFICATION_2, originally ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && ((mSensor == System::IMU_MONOCULAR)))
        c4=true;
    else
        c4=false;

    if(((c1a||c1b||c1c) && c2)||c3 ||c4)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle || mpLocalMapper->IsInitializing())
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR  && mSensor!=System::IMU_MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
            {
                //std::cout << "NeedNewKeyFrame: localmap is busy" << std::endl;
                return false;
            }
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(mpLocalMapper->IsInitializing() && !mpAtlas->isImuInitialized())
        return;

    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);
    if(KeepInstanceStructurePointsOutOfStaticKeyFrameSlots())
    {
        int removedStructurePointsFromStaticSlots = 0;
        const std::vector<MapPoint*> vpCopiedMapPoints = pKF->GetMapPointMatches();
        for(size_t idx = 0; idx < vpCopiedMapPoints.size(); ++idx)
        {
            MapPoint* pMP = vpCopiedMapPoints[idx];
            if(!pMP || pMP->isBad() || !pMP->IsInstanceStructurePoint())
                continue;

            pKF->EraseMapPointMatch(static_cast<int>(idx));
            ++removedStructurePointsFromStaticSlots;
        }

        if(removedStructurePointsFromStaticSlots > 0)
        {
            std::cout << "[STSLAM_INSTANCE_LIFECYCLE] frame=" << mCurrentFrame.mnId
                      << " keyframe_id=" << pKF->mnId
                      << " static_keyframe_structure_points_removed="
                      << removedStructurePointsFromStaticSlots
                      << " dynamic_observations_preserved="
                      << pKF->GetDynamicInstancePointObservations().size()
                      << std::endl;
        }
    }

    std::vector<WindowFrameSnapshot> vImageFrameWindow = mvFramesSinceLastKeyFrame;
    auto appendUniqueWindowFrame = [&](const WindowFrameSnapshot& snapshot)
    {
        if(!snapshot.HasPose())
            return;

        for(size_t i = 0; i < vImageFrameWindow.size(); ++i)
        {
            if(vImageFrameWindow[i].mnFrameId == snapshot.mnFrameId)
            {
                vImageFrameWindow[i] = snapshot;
                return;
            }
        }

        vImageFrameWindow.push_back(snapshot);
    };

    for(size_t i = 0; i < mRecentPanopticFrames.size(); ++i)
    {
        const Frame& recentFrame = mRecentPanopticFrames[i];
        if(!recentFrame.HasPanopticObservation() || !recentFrame.isSet())
            continue;

        appendUniqueWindowFrame(WindowFrameSnapshot(recentFrame));
    }

    pKF->SetImageFrameWindow(vImageFrameWindow);
    mvFramesSinceLastKeyFrame.clear();
    mbCurrentFrameCreatedKeyFrame = true;

    if(mpAtlas->isImuInitialized()) //  || mpLocalMapper->IsInitializing())
        pKF->bImu = true;

    pKF->SetNewBias(mCurrentFrame.mImuBias);
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mpLastKeyFrame)
    {
        pKF->mPrevKF = mpLastKeyFrame;
        mpLastKeyFrame->mNextKF = pKF;
    }
    else
        Verbose::PrintMess("No last KF in KF creation!!", Verbose::VERBOSITY_NORMAL);

    // Reset preintegration from last KF (Create new object)
    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
    {
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKF->GetImuBias(),pKF->mImuCalib);
    }

    if(mSensor!=System::MONOCULAR && mSensor != System::IMU_MONOCULAR) // TODO check if incluide imu_stereo
    {
        mCurrentFrame.UpdatePoseMatrices();
        // cout << "create new MPs" << endl;
        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        int maxPoint = 100;
        if(mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            maxPoint = 100;

        vector<pair<float,int> > vDepthIdx;
        int N = (mCurrentFrame.Nleft != -1) ? mCurrentFrame.Nleft : mCurrentFrame.N;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<N; i++)
        {
            if(EnableRgbdDynamicFrontendSplit() &&
               mCurrentFrame.HasPanopticObservation() &&
               mCurrentFrame.GetFeatureInstanceId(static_cast<size_t>(i)) > 0)
            {
                Map* pCurrentMap = mpAtlas ? mpAtlas->GetCurrentMap() : static_cast<Map*>(NULL);
                Instance* pInstance =
                    pCurrentMap ?
                    pCurrentMap->GetInstance(
                        mCurrentFrame.GetFeatureInstanceId(static_cast<size_t>(i))) :
                    static_cast<Instance*>(NULL);
                if(ShouldDetachRgbdInstanceFromStaticPath(pInstance))
                    continue;
            }
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    Eigen::Vector3f x3D;

                    if(mCurrentFrame.Nleft == -1){
                        mCurrentFrame.UnprojectStereo(i, x3D);
                    }
                    else{
                        x3D = mCurrentFrame.UnprojectStereoFishEye(i);
                    }

                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpAtlas->GetCurrentMap());
                    pNewMP->AddObservation(pKF,i);

                    //Check if it is a stereo observation in order to not
                    //duplicate mappoints
                    if(mCurrentFrame.Nleft != -1 && mCurrentFrame.mvLeftToRightMatch[i] >= 0){
                        mCurrentFrame.mvpMapPoints[mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]]=pNewMP;
                        pNewMP->AddObservation(pKF,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                        pKF->AddMapPoint(pNewMP,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                    }

                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>maxPoint)
                {
                    break;
                }
            }
            //Verbose::PrintMess("new mps for stereo KF: " + to_string(nPoints), Verbose::VERBOSITY_NORMAL);
        }
    }


    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    FrameFeatureDebugStats seedFrameStats;
    if(DebugFocusFrame(mCurrentFrame.mnId))
        seedFrameStats = CollectFrameFeatureDebugStats(mCurrentFrame, true, false, false);

    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
                pMP->mbTrackInViewR = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
        if(pMP->mbTrackInView)
        {
            mCurrentFrame.mmProjectPoints[pMP->mnId] = cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY);
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD || mSensor==System::IMU_RGBD)
            th=3;
        if(mpAtlas->isImuInitialized())
        {
            if(mpAtlas->GetCurrentMap()->GetIniertialBA2())
                th=2;
            else
                th=6;
        }
        else if(!mpAtlas->isImuInitialized() && (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD))
        {
            th=10;
        }

        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;

        if(mState==LOST || mState==RECENTLY_LOST) // Lost for less than 1 second
            th=15; // 15

        int matches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th, mpLocalMapper->mbFarPoints, mpLocalMapper->mThFarPoints);
        if(DebugFocusFrame(mCurrentFrame.mnId))
        {
            const MapPointDebugStats localMapStats =
                CollectMapPointDebugStats(mvpLocalMapPoints, false);
            const MapPointDebugStats projectedStats =
                CollectMapPointDebugStats(mvpLocalMapPoints, true);
            const FrameFeatureDebugStats matchedFrameStats =
                CollectFrameFeatureDebugStats(mCurrentFrame, true, false, false);
            std::cout << "[STSLAM_FOCUS] frame=" << mCurrentFrame.mnId
                      << " stage=search_local_points"
                      << " local_map_points=" << mvpLocalMapPoints.size()
                      << " projected_candidates=" << nToMatch
                      << " projection_match_threshold=" << th
                      << " projection_matches=" << matches
                      << " local_map_stats={" << FormatMapPointDebugStats(localMapStats) << "}"
                      << " projected_stats={" << FormatMapPointDebugStats(projectedStats) << "}"
                      << " seed_match_stats={" << FormatFrameFeatureDebugStats(seedFrameStats) << "}"
                      << " matched_frame_stats={" << FormatFrameFeatureDebugStats(matchedFrameStats) << "}"
                      << std::endl;
        }
    }
    else if(DebugFocusFrame(mCurrentFrame.mnId))
    {
        const MapPointDebugStats localMapStats =
            CollectMapPointDebugStats(mvpLocalMapPoints, false);
        std::cout << "[STSLAM_FOCUS] frame=" << mCurrentFrame.mnId
                  << " stage=search_local_points"
                  << " local_map_points=" << mvpLocalMapPoints.size()
                  << " projected_candidates=0"
                  << " local_map_stats={" << FormatMapPointDebugStats(localMapStats) << "}"
                  << " seed_match_stats={" << FormatFrameFeatureDebugStats(seedFrameStats) << "}"
                  << std::endl;
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    int count_pts = 0;

    for(vector<KeyFrame*>::const_reverse_iterator itKF=mvpLocalKeyFrames.rbegin(), itEndKF=mvpLocalKeyFrames.rend(); itKF!=itEndKF; ++itKF)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {

            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                count_pts++;
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}


void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    if(!mpAtlas->isImuInitialized() || (mCurrentFrame.mnId<mnLastRelocFrameId+2))
    {
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();
                    for(map<KeyFrame*,tuple<int,int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                        keyframeCounter[it->first]++;
                }
                else
                {
                    mCurrentFrame.mvpMapPoints[i]=NULL;
                }
            }
        }
    }
    else
    {
        for(int i=0; i<mLastFrame.N; i++)
        {
            // Using lastframe since current frame has not matches yet
            if(mLastFrame.mvpMapPoints[i])
            {
                MapPoint* pMP = mLastFrame.mvpMapPoints[i];
                if(!pMP)
                    continue;
                if(!pMP->isBad())
                {
                    const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();
                    for(map<KeyFrame*,tuple<int,int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                        keyframeCounter[it->first]++;
                }
                else
                {
                    // MODIFICATION
                    mLastFrame.mvpMapPoints[i]=NULL;
                }
            }
        }
    }


    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(!pKFmax || it->second>max || (it->second==max && pKF->mnId > pKFmax->mnId))
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(pKF);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80) // 80
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);


        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }
    }

    // Add 10 last temporal KFs (mainly for IMU)
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) &&mvpLocalKeyFrames.size()<80)
    {
        KeyFrame* tempKeyFrame = mCurrentFrame.mpLastKeyFrame;

        const int Nd = 20;
        for(int i=0; i<Nd; i++){
            if (!tempKeyFrame)
                break;
            if(tempKeyFrame->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(tempKeyFrame);
                tempKeyFrame->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                tempKeyFrame=tempKeyFrame->mPrevKF;
            }
        }
    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }

    if(DebugFocusFrame(mCurrentFrame.mnId))
    {
        std::vector<std::pair<KeyFrame*, int>> topKeyframes(keyframeCounter.begin(), keyframeCounter.end());
        std::sort(topKeyframes.begin(), topKeyframes.end(),
                  [](const std::pair<KeyFrame*, int>& lhs, const std::pair<KeyFrame*, int>& rhs)
                  {
                      if(lhs.second != rhs.second)
                          return lhs.second > rhs.second;
                      return lhs.first->mnId < rhs.first->mnId;
                  });
        std::ostringstream topVotes;
        const size_t topCount = std::min<size_t>(5, topKeyframes.size());
        for(size_t i = 0; i < topCount; ++i)
        {
            if(i > 0)
                topVotes << ",";
            topVotes << topKeyframes[i].first->mnId << ":" << topKeyframes[i].second;
        }
        std::cout << "[STSLAM_FOCUS] frame=" << mCurrentFrame.mnId
                  << " stage=update_local_map"
                  << " voted_keyframes=" << keyframeCounter.size()
                  << " local_keyframes=" << mvpLocalKeyFrames.size()
                  << " local_map_points=" << mvpLocalMapPoints.size()
                  << " reference_kf=" << (mpReferenceKF ? static_cast<long>(mpReferenceKF->mnId) : -1)
                  << " seed_matches=" << CountTrackedMapPoints(mCurrentFrame)
                  << " top_votes=" << topVotes.str()
                  << std::endl;
    }
}

bool Tracking::Relocalization()
{
    Verbose::PrintMess("Starting relocalization", Verbose::VERBOSITY_NORMAL);
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame, mpAtlas->GetCurrentMap());

    if(vpCandidateKFs.empty()) {
        Verbose::PrintMess("There are not candidates", Verbose::VERBOSITY_NORMAL);
        return false;
    }

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<MLPnPsolver*> vpMLPnPsolvers;
    vpMLPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                MLPnPsolver* pSolver = new MLPnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,6,0.5,5.991);  //This solver needs at least 6 points
                vpMLPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            MLPnPsolver* pSolver = vpMLPnPsolvers[i];
            Eigen::Matrix4f eigTcw;
            bool bTcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers, eigTcw);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(bTcw)
            {
                Sophus::SE3f Tcw(eigTcw);
                mCurrentFrame.SetPose(Tcw);
                // Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        cout << "Relocalized!!" << endl;
        return true;
    }

}

void Tracking::Reset(bool bLocMap)
{
    Verbose::PrintMess("System Reseting", Verbose::VERBOSITY_NORMAL);

    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
        mpLocalMapper->RequestReset();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
    }


    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestReset();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database...", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clear();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearAtlas();
    mpAtlas->CreateNewMap();
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
        mpAtlas->SetInertialSensor();
    mnInitialFrameId = 0;

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    mbReadyToInitializate = false;
    mbSetInit=false;

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
    mCurrentFrame = Frame();
    mnLastRelocFrameId = 0;
    mLastFrame = Frame();
    mPreviousImGrayForSparseFlow.release();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();
    mRecentPanopticFrames.clear();
    mvFramesSinceLastKeyFrame.clear();
    mbCurrentFrameCreatedKeyFrame = false;

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

void Tracking::ResetActiveMap(bool bLocMap)
{
    Verbose::PrintMess("Active map Reseting", Verbose::VERBOSITY_NORMAL);
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    Map* pMap = mpAtlas->GetCurrentMap();

    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_VERY_VERBOSE);
        mpLocalMapper->RequestResetActiveMap(pMap);
        Verbose::PrintMess("done", Verbose::VERBOSITY_VERY_VERBOSE);
    }

    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestResetActiveMap(pMap);
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clearMap(pMap); // Only clear the active map references
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearMap();


    //KeyFrame::nNextId = mpAtlas->GetLastInitKFid();
    //Frame::nNextId = mnLastInitFrameId;
    mnLastInitFrameId = Frame::nNextId;
    //mnLastRelocFrameId = mnLastInitFrameId;
    mState = NO_IMAGES_YET; //NOT_INITIALIZED;

    mbReadyToInitializate = false;

    list<bool> lbLost;
    // lbLost.reserve(mlbLost.size());
    unsigned int index = mnFirstFrameId;
    cout << "mnFirstFrameId = " << mnFirstFrameId << endl;
    for(Map* pMap : mpAtlas->GetAllMaps())
    {
        if(pMap->GetAllKeyFrames().size() > 0)
        {
            if(index > pMap->GetLowerKFID())
                index = pMap->GetLowerKFID();
        }
    }

    //cout << "First Frame id: " << index << endl;
    int num_lost = 0;
    cout << "mnInitialFrameId = " << mnInitialFrameId << endl;

    for(list<bool>::iterator ilbL = mlbLost.begin(); ilbL != mlbLost.end(); ilbL++)
    {
        if(index < mnInitialFrameId)
            lbLost.push_back(*ilbL);
        else
        {
            lbLost.push_back(true);
            num_lost += 1;
        }

        index++;
    }
    cout << num_lost << " Frames set to lost" << endl;

    mlbLost = lbLost;

    mnInitialFrameId = mCurrentFrame.mnId;
    mnLastRelocFrameId = mCurrentFrame.mnId;

    mCurrentFrame = Frame();
    mLastFrame = Frame();
    mPreviousImGrayForSparseFlow.release();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();
    mRecentPanopticFrames.clear();
    mvFramesSinceLastKeyFrame.clear();
    mbCurrentFrameCreatedKeyFrame = false;

    mbVelocity = false;

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

vector<MapPoint*> Tracking::GetLocalMapMPS()
{
    return mvpLocalMapPoints;
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    mK_.setIdentity();
    mK_(0,0) = fx;
    mK_(1,1) = fy;
    mK_(0,2) = cx;
    mK_(1,2) = cy;

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

void Tracking::UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame)
{
    Map * pMap = pCurrentKeyFrame->GetMap();
    unsigned int index = mnFirstFrameId;
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mlpReferences.begin();
    list<bool>::iterator lbL = mlbLost.begin();
    for(auto lit=mlRelativeFramePoses.begin(),lend=mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        while(pKF->isBad())
        {
            pKF = pKF->GetParent();
        }

        if(pKF->GetMap() == pMap)
        {
            (*lit).translation() *= s;
        }
    }

    mLastBias = b;

    mpLastKeyFrame = pCurrentKeyFrame;

    mLastFrame.SetNewBias(mLastBias);
    mCurrentFrame.SetNewBias(mLastBias);

    while(!mCurrentFrame.imuIsPreintegrated())
    {
        usleep(500);
    }


    if(mLastFrame.mnId == mLastFrame.mpLastKeyFrame->mnFrameId)
    {
        mLastFrame.SetImuPoseVelocity(mLastFrame.mpLastKeyFrame->GetImuRotation(),
                                      mLastFrame.mpLastKeyFrame->GetImuPosition(),
                                      mLastFrame.mpLastKeyFrame->GetVelocity());
    }
    else
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const Eigen::Vector3f twb1 = mLastFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.mpLastKeyFrame->GetVelocity();
        float t12 = mLastFrame.mpImuPreintegrated->dT;

        mLastFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    if (mCurrentFrame.mpImuPreintegrated)
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);

        const Eigen::Vector3f twb1 = mCurrentFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mCurrentFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mCurrentFrame.mpLastKeyFrame->GetVelocity();
        float t12 = mCurrentFrame.mpImuPreintegrated->dT;

        mCurrentFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    mnFirstImuFrameId = mCurrentFrame.mnId;
}

void Tracking::NewDataset()
{
    mnNumDataset++;
    mPreviousImGrayForSparseFlow.release();
}

int Tracking::GetNumberDataset()
{
    return mnNumDataset;
}

int Tracking::GetMatchesInliers()
{
    return mnMatchesInliers;
}

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, string strFolder)
{
    mpSystem->SaveTrajectoryEuRoC(strFolder + strNameFile_frames);
    //mpSystem->SaveKeyFrameTrajectoryEuRoC(strFolder + strNameFile_kf);
}

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, Map* pMap)
{
    mpSystem->SaveTrajectoryEuRoC(strNameFile_frames, pMap);
    if(!strNameFile_kf.empty())
        mpSystem->SaveKeyFrameTrajectoryEuRoC(strNameFile_kf, pMap);
}

float Tracking::GetImageScale()
{
    return mImageScale;
}

#ifdef REGISTER_LOOP
void Tracking::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
}

bool Tracking::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Tracking STOP" << endl;
        return true;
    }

    return false;
}

bool Tracking::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

bool Tracking::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

void Tracking::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
    mbStopRequested = false;
}
#endif

} //namespace ORB_SLAM
