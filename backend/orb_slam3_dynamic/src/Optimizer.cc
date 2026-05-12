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


#include "Optimizer.h"


#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>

#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "Thirdparty/g2o/g2o/core/sparse_block_matrix.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "G2oTypes.h"
#include "Converter.h"
#include "Instance.h"

#include<mutex>

#include "OptimizableTypes.h"


namespace ORB_SLAM3
{
bool sortByVal(const pair<MapPoint*, int> &a, const pair<MapPoint*, int> &b)
{
    return (a.second < b.second);
}

namespace
{

struct DynamicVertexKey
{
    DynamicVertexKey(MapPoint* pMP_ = NULL, unsigned long frameId_ = 0)
        : pMP(pMP_), frameId(frameId_)
    {
    }

    bool operator<(const DynamicVertexKey& other) const
    {
        if(pMP != other.pMP)
            return pMP < other.pMP;
        return frameId < other.frameId;
    }

    MapPoint* pMP;
    unsigned long frameId;
};

struct InstanceFrameKey
{
    InstanceFrameKey(int instanceId_ = -1, unsigned long frameId_ = 0)
        : instanceId(instanceId_), frameId(frameId_)
    {
    }

    bool operator<(const InstanceFrameKey& other) const
    {
        if(instanceId != other.instanceId)
            return instanceId < other.instanceId;
        return frameId < other.frameId;
    }

    int instanceId;
    unsigned long frameId;
};

struct FrameObservationIndex
{
    int leftIndex = -1;
    int rightIndex = -1;
};

struct ImageFrameHandle
{
    unsigned long frameId = 0;
    double timeStamp = 0.0;
    int poseVertexId = -1;
    bool isKeyFrame = false;
    bool fixed = false;
    bool inCurrentInterval = false;
    KeyFrame* pKeyFrame = NULL;
    WindowFrameSnapshot* pWindowFrame = NULL;
};

bool SortImageFramesByTimestamp(const ImageFrameHandle& a, const ImageFrameHandle& b)
{
    if(a.timeStamp != b.timeStamp)
        return a.timeStamp < b.timeStamp;
    return a.frameId < b.frameId;
}

const std::vector<cv::KeyPoint>& GetFrameKeysUn(const ImageFrameHandle& frame)
{
    return frame.isKeyFrame ? frame.pKeyFrame->mvKeysUn : frame.pWindowFrame->mvKeysUn;
}

const std::vector<cv::KeyPoint>& GetFrameKeysRight(const ImageFrameHandle& frame)
{
    return frame.isKeyFrame ? frame.pKeyFrame->mvKeysRight : frame.pWindowFrame->mvKeysRight;
}

const std::vector<float>& GetFrameInvLevelSigma2(const ImageFrameHandle& frame)
{
    return frame.isKeyFrame ? frame.pKeyFrame->mvInvLevelSigma2 : frame.pWindowFrame->mvInvLevelSigma2;
}

const std::vector<float>& GetFrameRightCoordinates(const ImageFrameHandle& frame)
{
    return frame.isKeyFrame ? frame.pKeyFrame->mvuRight : frame.pWindowFrame->mvuRight;
}

GeometricCamera* GetFrameCamera(const ImageFrameHandle& frame)
{
    return frame.isKeyFrame ? frame.pKeyFrame->mpCamera : frame.pWindowFrame->mpCamera;
}

GeometricCamera* GetFrameCamera2(const ImageFrameHandle& frame)
{
    return frame.isKeyFrame ? frame.pKeyFrame->mpCamera2 : frame.pWindowFrame->mpCamera2;
}

Sophus::SE3f GetFrameTrl(const ImageFrameHandle& frame)
{
    return frame.isKeyFrame ? frame.pKeyFrame->GetRelativePoseTrl() : frame.pWindowFrame->mTrl;
}

Sophus::SE3f GetFramePose(const ImageFrameHandle& frame)
{
    return frame.isKeyFrame ? frame.pKeyFrame->GetPose() : frame.pWindowFrame->mTcw;
}

bool IsNearlyZeroInstanceMotion(const Sophus::SE3f& motion)
{
    const double translationNorm = motion.translation().cast<double>().norm();
    const Eigen::AngleAxisd angleAxis(motion.rotationMatrix().cast<double>());
    return translationNorm < 1e-6 && std::abs(angleAxis.angle()) < 1e-6;
}

bool FrameUsesDynamicInstanceMotion(const ImageFrameHandle& frame, const int instanceId)
{
    if(instanceId <= 0)
        return false;

    if(frame.isKeyFrame)
    {
        if(!frame.pKeyFrame)
            return false;
        Sophus::SE3f motion;
        return frame.pKeyFrame->GetPredictedInstanceMotion(instanceId, motion) &&
               !IsNearlyZeroInstanceMotion(motion);
    }

    if(!frame.pWindowFrame)
        return false;

    const std::map<int, Sophus::SE3f>::const_iterator itMotion =
        frame.pWindowFrame->mmPredictedInstanceMotions.find(instanceId);
    return itMotion != frame.pWindowFrame->mmPredictedInstanceMotions.end() &&
           !IsNearlyZeroInstanceMotion(itMotion->second);
}

void SetFramePose(ImageFrameHandle& frame, const Sophus::SE3f& pose)
{
    if(frame.isKeyFrame)
        frame.pKeyFrame->SetPose(pose);
    else if(frame.pWindowFrame)
    {
        frame.pWindowFrame->mTcw = pose;
        frame.pWindowFrame->mbHasPose = true;
    }
}

int GetFrameFeatureInstanceId(const ImageFrameHandle& frame, size_t idx)
{
    return frame.isKeyFrame ? frame.pKeyFrame->GetFeatureInstanceId(idx)
                            : frame.pWindowFrame->GetFeatureInstanceId(idx);
}

int GetFrameFeatureSemanticLabel(const ImageFrameHandle& frame, size_t idx)
{
    return frame.isKeyFrame ? frame.pKeyFrame->GetFeatureSemanticLabel(idx)
                            : frame.pWindowFrame->GetFeatureSemanticLabel(idx);
}

int GetFrameFeatureCount(const ImageFrameHandle& frame)
{
    return frame.isKeyFrame ? frame.pKeyFrame->N : frame.pWindowFrame->GetNumFeatures();
}

int GetFrameLeftFeatureCount(const ImageFrameHandle& frame)
{
    return frame.isKeyFrame
        ? ((frame.pKeyFrame->NLeft != -1) ? frame.pKeyFrame->NLeft : frame.pKeyFrame->N)
        : ((frame.pWindowFrame->GetLeftFeatureCount() != -1) ? frame.pWindowFrame->GetLeftFeatureCount()
                                                              : frame.pWindowFrame->GetNumFeatures());
}

bool FrameObservationIsOutlier(const ImageFrameHandle& frame, int idx)
{
    if(frame.isKeyFrame)
        return false;
    return idx >= 0 &&
           idx < static_cast<int>(frame.pWindowFrame->mvbOutlier.size()) &&
           frame.pWindowFrame->mvbOutlier[idx];
}

const std::vector<DynamicInstancePointObservation>& GetFrameDynamicInstancePointObservations(
    const ImageFrameHandle& frame)
{
    return frame.isKeyFrame ? frame.pKeyFrame->GetDynamicInstancePointObservations()
                            : frame.pWindowFrame->mvDynamicInstancePointObservations;
}

bool SortKeyFramesByTimestamp(KeyFrame* pA, KeyFrame* pB)
{
    if(pA->mTimeStamp != pB->mTimeStamp)
        return pA->mTimeStamp < pB->mTimeStamp;
    return pA->mnId < pB->mnId;
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

bool UseStrictPaperArchitectureDefaults()
{
    const char* envValue = std::getenv("STSLAM_MODULE8_PROFILE");
    if(!envValue || std::string(envValue).empty())
        return true;

    const std::string profile(envValue);
    return profile == "paper_strict" || profile == "paper_eq16";
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

size_t GetShapeTripletSampleLimit()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_MAX_SHAPE_TRIPLETS_PER_FRAME", 50, 0);
    return static_cast<size_t>(value);
}

bool UseStrictShapeScaleTripletFactors()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_STRICT_SHAPE_TRIPLET_FACTORS",
                            UseStrictPaperArchitectureDefaults());
    return value;
}

size_t GetRigidityPairSampleLimit()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_MAX_RIGIDITY_PAIRS_PER_TIMEPAIR", 0, 0);
    return static_cast<size_t>(value);
}

std::string GetModule8ProfileName()
{
    const char* envValue = std::getenv("STSLAM_MODULE8_PROFILE");
    if(envValue && std::string(envValue).size() > 0)
        return std::string(envValue);
    return "paper_eq16";
}

int GetStrictEq16IterationCount()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_STRICT_EQ16_ITERATIONS", 10, 1);
    return value;
}

double GetDynamicPanopticFactorWeight()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_PANOPTIC_FACTOR_WEIGHT", 1.0, 0.0);
    return value;
}

double GetDynamicShapeFactorWeight()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_SHAPE_FACTOR_WEIGHT", 1.0, 0.0);
    return value;
}

double GetDynamicRigidityFactorWeight()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_RIGIDITY_FACTOR_WEIGHT", 1.0, 0.0);
    return value;
}

int GetArticulatedSemanticLabel()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_ARTICULATED_SEMANTIC_LABEL", 11, 0);
    return value;
}

double GetArticulatedShapeFactorScale()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_ARTICULATED_SHAPE_FACTOR_SCALE", 1.0, 0.0);
    return value;
}

double GetArticulatedRigidityFactorScale()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_ARTICULATED_RIGIDITY_FACTOR_SCALE", 0.1, 0.0);
    return value;
}

bool IsArticulatedSemanticLabel(const int semanticLabel)
{
    const int articulatedSemanticLabel = GetArticulatedSemanticLabel();
    return articulatedSemanticLabel > 0 && semanticLabel == articulatedSemanticLabel;
}

double GetShapeHuberDelta()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_SHAPE_HUBER_DELTA", 1.0, 0.0);
    return value;
}

double GetRigidityHuberDelta()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_RIGIDITY_HUBER_DELTA", 0.5, 0.0);
    return value;
}

size_t GetRigidityMaxFrameGap()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_RIGIDITY_MAX_FRAME_GAP", 1, 0);
    return static_cast<size_t>(value);
}

size_t GetRigidityMaxPairsPerPoint()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_MAX_RIGIDITY_PAIRS_PER_POINT", 0, 0);
    return static_cast<size_t>(value);
}

double GetRigidityPairMinQuality()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_RIGIDITY_PAIR_MIN_QUALITY", 0.2, 0.0);
    return std::min(1.0, value);
}

int GetRigidityPairMinKnownObservations()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_RIGIDITY_PAIR_MIN_KNOWN_OBS", 4, 0);
    return std::min(4, value);
}

bool UseQualityGatedRigidityPairSampler()
{
    const char* envValue = std::getenv("STSLAM_RIGIDITY_PAIR_SAMPLER");
    return envValue &&
           (std::string(envValue) == "quality_gated" ||
            std::string(envValue) == "quality_coverage");
}

bool UseQualityCoverageRigidityPairSampler()
{
    const char* envValue = std::getenv("STSLAM_RIGIDITY_PAIR_SAMPLER");
    return envValue && std::string(envValue) == "quality_coverage";
}

std::string GetRigidityPairSamplerName(bool useQualityGatedSampler,
                                       bool useQualityCoverageSampler,
                                       size_t maxPairs)
{
    if(maxPairs == 0)
        return "full_frobenius";
    if(useQualityCoverageSampler)
        return "quality_coverage";
    if(useQualityGatedSampler)
        return "quality_gated";
    return "legacy";
}

int GetDynamicBackendMinFrameSupport()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_BACKEND_MIN_FRAME_POINTS", 3, 1);
    return value;
}

bool UseStrictEq16ImageWindowForLBA()
{
    const char* envValue = std::getenv("STSLAM_STRICT_EQ16_WINDOW");
    return !envValue || std::string(envValue) != "0";
}

bool EnableInstanceStructureProxyForLBA()
{
    const char* envValue = std::getenv("STSLAM_ENABLE_INSTANCE_STRUCTURE_PROXY");
    // Main Module 8 path: couple instance pose and canonical instance
    // structure in the graph. Set STSLAM_ENABLE_INSTANCE_STRUCTURE_PROXY=0
    // only for the strict implicit Eq.(16) ablation.
    return !envValue || std::string(envValue) != "0";
}

bool EnableBackendInstanceObservationQualityGate()
{
    return GetEnvFlagOrDefault("STSLAM_BACKEND_INSTANCE_OBSERVATION_QUALITY_GATE",
                               true);
}

bool EnableBackendInstanceObservationQualityGateHardReject()
{
    return GetEnvFlagOrDefault(
        "STSLAM_BACKEND_INSTANCE_OBSERVATION_QUALITY_GATE_HARD_REJECT",
        !UseStrictPaperArchitectureDefaults());
}

bool EnableRgbdDepthBackedStrictGeometryGate()
{
    const bool depthBackedDynamicObservations =
        GetEnvFlagOrDefault("STSLAM_RGBD_DEPTH_BACKED_DYNAMIC_OBSERVATIONS", false);
    return GetEnvFlagOrDefault("STSLAM_RGBD_DEPTH_BACKED_STRICT_GEOMETRY_GATE",
                               depthBackedDynamicObservations);
}

double GetBackendInstanceGateMaxMeanChi2()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_BACKEND_INSTANCE_GATE_MAX_MEAN_CHI2", 10.0, 0.0);
    return value;
}

double GetBackendInstanceGateMaxHighChi2Ratio()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_BACKEND_INSTANCE_GATE_MAX_HIGH_CHI2_RATIO", 0.3, 0.0);
    return std::min(1.0, value);
}

double GetBackendInstanceGateMaxDepthFailureRatio()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_BACKEND_INSTANCE_GATE_MAX_DEPTH_FAILURE_RATIO", 0.0, 0.0);
    return std::min(1.0, value);
}

double GetBackendInstanceGateMaxReconError()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_BACKEND_INSTANCE_GATE_MAX_RECON_ERROR", 0.5, 0.0);
    return value;
}

bool EnableDynamicObservationPointWorldRefinement()
{
    return GetEnvFlagOrDefault("STSLAM_DYNAMIC_OBSERVATION_POINTWORLD_REFINEMENT",
                               UseStrictPaperArchitectureDefaults());
}

double GetDynamicObservationPointWorldMaxReprojectionErrorPx()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_OBSERVATION_POINTWORLD_MAX_REPROJ_PX",
                              8.0,
                              0.0);
    return value;
}

double GetDynamicObservationPointWorldRefinedWeight()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_OBSERVATION_POINTWORLD_REFINED_WEIGHT",
                              0.25,
                              0.01);
    return std::min(1.0, value);
}

bool EnableCentroidMotionFallbackForLBA()
{
    const char* envValue = std::getenv("STSLAM_ENABLE_CENTROID_MOTION_FALLBACK");
    return envValue && std::string(envValue) != "0";
}

bool UseTranslationOnlyInstanceMotionWritebackForLBA()
{
    const char* envValue = std::getenv("STSLAM_INSTANCE_MOTION_TRANSLATION_ONLY");
    return envValue && std::string(envValue) != "0";
}

bool UseTranslationOnlyInstanceStructureWritebackForLBA(const int semanticLabel)
{
    (void)semanticLabel;
    const char* envValue = std::getenv("STSLAM_INSTANCE_STRUCTURE_TRANSLATION_ONLY");
    if(envValue)
        return std::string(envValue) != "0";

    return false;
}

bool AllowImmatureInstanceStructureWritebackForLBA()
{
    const char* envValue = std::getenv("STSLAM_ALLOW_IMMATURE_INSTANCE_STRUCTURE_WRITEBACK");
    return envValue && std::string(envValue) != "0";
}

bool EnableBackendInstanceStructureWritebackForLBA()
{
    const char* envValue = std::getenv("STSLAM_ENABLE_BACKEND_INSTANCE_STRUCTURE_WRITEBACK");
    return envValue && std::string(envValue) != "0";
}

bool AllowInstanceStructureOverwriteForLBA()
{
    const char* envValue = std::getenv("STSLAM_ALLOW_INSTANCE_STRUCTURE_OVERWRITE");
    return envValue && std::string(envValue) != "0";
}

float GetBackendImmatureMotionMaxTranslation()
{
    const char* envValue = std::getenv("STSLAM_BACKEND_IMMATURE_MAX_TRANSLATION");
    if(!envValue)
        return 0.08f;

    char* endPtr = NULL;
    const double value = std::strtod(envValue, &endPtr);
    if(endPtr == envValue || value < 0.0 || !std::isfinite(value))
        return 0.08f;

    return static_cast<float>(value);
}

float GetBackendMatureMotionMaxTranslation()
{
    const char* envValue = std::getenv("STSLAM_BACKEND_MATURE_MAX_TRANSLATION");
    if(!envValue)
        return 0.02f;

    char* endPtr = NULL;
    const double value = std::strtod(envValue, &endPtr);
    if(endPtr == envValue || value < 0.0 || !std::isfinite(value))
        return 0.02f;

    return static_cast<float>(value);
}

float GetBackendImmatureMotionMaxRotationDeg()
{
    const char* envValue = std::getenv("STSLAM_BACKEND_IMMATURE_MAX_ROTATION_DEG");
    if(!envValue)
        return 15.0f;

    char* endPtr = NULL;
    const double value = std::strtod(envValue, &endPtr);
    if(endPtr == envValue || value < 0.0 || !std::isfinite(value))
        return 15.0f;

    return static_cast<float>(value);
}

float GetBackendMatureMotionMaxRotationDeg()
{
    const char* envValue = std::getenv("STSLAM_BACKEND_MATURE_MAX_ROTATION_DEG");
    if(!envValue)
        return 5.0f;

    char* endPtr = NULL;
    const double value = std::strtod(envValue, &endPtr);
    if(endPtr == envValue || value < 0.0 || !std::isfinite(value))
        return 5.0f;

    return static_cast<float>(value);
}

int GetBackendImmatureMotionConfirmFrames()
{
    return GetEnvIntOrDefault("STSLAM_BACKEND_IMMATURE_CONFIRM_FRAMES", 2, 1);
}

bool EnableBackendMatureMotionGate()
{
    const char* envValue = std::getenv("STSLAM_BACKEND_MATURE_MOTION_GATE");
    return envValue && std::string(envValue) != "0";
}

int GetBackendMotionMaxFrameGap()
{
    return GetEnvIntOrDefault("STSLAM_BACKEND_MOTION_MAX_FRAME_GAP", 0, 0);
}

bool EnableInstancePoseMotionPrior()
{
    return GetEnvFlagOrDefault("STSLAM_INSTANCE_POSE_MOTION_PRIOR",
                               UseStrictPaperArchitectureDefaults());
}

double GetInstancePoseMotionPriorInvSigma2()
{
    return GetEnvDoubleOrDefault("STSLAM_INSTANCE_POSE_MOTION_PRIOR_INV_SIGMA2",
                                 0.05, 0.0);
}

double GetInstanceStructureInvSigma2()
{
    const char* envValue = std::getenv("STSLAM_INSTANCE_STRUCTURE_INV_SIGMA2");
    if(!envValue)
        return 1.0;

    char* endPtr = NULL;
    const double value = std::strtod(envValue, &endPtr);
    if(endPtr == envValue || value <= 0.0 || !std::isfinite(value))
        return 1.0;

    return value;
}

bool HasMatureInstanceBackendState(Instance* pInstance, KeyFrame* pKF)
{
    if(!pInstance || !pKF || !pInstance->IsInitialized())
        return false;

    const int initializedFrame = pInstance->GetInitializedFrame();
    return initializedFrame >= 0 &&
           static_cast<int>(pKF->mnFrameId) >= initializedFrame + GetInstancePredictionWarmupFrames() &&
           static_cast<int>(pInstance->NumMapPoints()) >= GetInstancePredictionMinSupport();
}

struct ShapeScaleTripletMeasurement
{
    size_t indexA = 0;
    size_t indexB = 0;
    size_t indexC = 0;
    Vector6d measurement = Vector6d::Zero();
    double invSigma2 = 1.0;
};

struct DynamicEdgeObservation
{
    DynamicEdgeObservation(MapPoint* pMP_ = NULL, unsigned long frameId_ = 0)
        : pMP(pMP_), frameId(frameId_)
    {
    }

    MapPoint* pMP;
    unsigned long frameId;
};

struct FactorResidualStats
{
    size_t records = 0;
    size_t logicalTerms = 0;
    size_t highChi2 = 0;
    double chi2Sum = 0.0;
    double chi2Max = 0.0;

    void Add(const double chi2, const size_t terms, const double chi2Gate)
    {
        if(!std::isfinite(chi2))
            return;
        ++records;
        logicalTerms += terms;
        chi2Sum += chi2;
        chi2Max = std::max(chi2Max, chi2);
        if(chi2Gate > 0.0 && chi2 > chi2Gate)
            ++highChi2;
    }

    double MeanChi2() const
    {
        return records > 0 ? chi2Sum / static_cast<double>(records) : 0.0;
    }
};

template<typename EdgeT>
void AccumulateFactorResidual(EdgeT* edge,
                              FactorResidualStats& stats,
                              const size_t logicalTerms,
                              const double chi2Gate)
{
    if(!edge)
        return;
    edge->computeError();
    stats.Add(edge->chi2(), logicalTerms, chi2Gate);
}

class EdgeShapeScaleFrame : public g2o::BaseMultiEdge<2, Eigen::Vector2d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit EdgeShapeScaleFrame(const size_t nVertices = 0)
    {
        resize(nVertices);
    }

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void AddTriplet(const size_t indexA,
                    const size_t indexB,
                    const size_t indexC,
                    const Vector6d& measurement,
                    const double invSigma2)
    {
        ShapeScaleTripletMeasurement triplet;
        triplet.indexA = indexA;
        triplet.indexB = indexB;
        triplet.indexC = indexC;
        triplet.measurement = measurement;
        triplet.invSigma2 = invSigma2;
        mvTriplets.push_back(triplet);
        mInvSigma2Sum += invSigma2;
    }

    size_t NumTriplets() const
    {
        return mvTriplets.size();
    }

    double MeanInvSigma2() const
    {
        return mvTriplets.empty() ? 1.0 : (mInvSigma2Sum / static_cast<double>(mvTriplets.size()));
    }

    void computeError()
    {
        double accumulatedDist = 0.0;
        double accumulatedAngle = 0.0;
        for(size_t i = 0; i < mvTriplets.size(); ++i)
        {
            const ShapeScaleTripletMeasurement& triplet = mvTriplets[i];
            const g2o::VertexSBAPointXYZ* vA =
                static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[triplet.indexA]);
            const g2o::VertexSBAPointXYZ* vB =
                static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[triplet.indexB]);
            const g2o::VertexSBAPointXYZ* vC =
                static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[triplet.indexC]);

            const Eigen::Vector3d a = vA->estimate();
            const Eigen::Vector3d b = vB->estimate();
            const Eigen::Vector3d c = vC->estimate();

            const Eigen::Vector2d pa = triplet.measurement.segment<2>(0);
            const Eigen::Vector2d pb = triplet.measurement.segment<2>(2);
            const Eigen::Vector2d pc = triplet.measurement.segment<2>(4);

            const double distPab = std::max(1e-6, (pa - pb).norm());
            const double distPbc = std::max(1e-6, (pb - pc).norm());
            const double distMab = std::max(1e-6, (a - b).norm());
            const double distMbc = std::max(1e-6, (b - c).norm());
            const double eDist = distPab * distMbc - distMab * distPbc;

            const Eigen::Vector2d vPab = pa - pb;
            const Eigen::Vector2d vPbc = pb - pc;
            const Eigen::Vector3d vMab = a - b;
            const Eigen::Vector3d vMbc = b - c;

            const double denomP = std::max(1e-6, vPab.norm() * vPbc.norm());
            const double denomM = std::max(1e-6, vMab.norm() * vMbc.norm());
            const double cosP = std::max(-1.0, std::min(1.0, vPab.dot(vPbc) / denomP));
            const double cosM = std::max(-1.0, std::min(1.0, vMab.dot(vMbc) / denomM));
            const double eAngle = std::acos(cosP) - std::acos(cosM);

            accumulatedDist += std::abs(eDist);
            accumulatedAngle += std::abs(eAngle);
        }

        const double normalizer =
            mvTriplets.empty() ? 1.0 : static_cast<double>(mvTriplets.size());
        _error[0] = accumulatedDist / normalizer;
        _error[1] = accumulatedAngle / normalizer;
    }

private:
    std::vector<ShapeScaleTripletMeasurement> mvTriplets;
    double mInvSigma2Sum = 0.0;
};

class EdgeShapeScaleTriplet : public g2o::BaseMultiEdge<2, Vector6d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeShapeScaleTriplet()
    {
        resize(3);
    }

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError()
    {
        const g2o::VertexSBAPointXYZ* vA =
            static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const g2o::VertexSBAPointXYZ* vB =
            static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[1]);
        const g2o::VertexSBAPointXYZ* vC =
            static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[2]);

        const Eigen::Vector3d a = vA->estimate();
        const Eigen::Vector3d b = vB->estimate();
        const Eigen::Vector3d c = vC->estimate();

        const Eigen::Vector2d pa = _measurement.segment<2>(0);
        const Eigen::Vector2d pb = _measurement.segment<2>(2);
        const Eigen::Vector2d pc = _measurement.segment<2>(4);

        const double distPab = std::max(1e-6, (pa - pb).norm());
        const double distPbc = std::max(1e-6, (pb - pc).norm());
        const double distMab = std::max(1e-6, (a - b).norm());
        const double distMbc = std::max(1e-6, (b - c).norm());
        _error[0] = distPab * distMbc - distMab * distPbc;

        const Eigen::Vector2d vPab = pa - pb;
        const Eigen::Vector2d vPbc = pb - pc;
        const Eigen::Vector3d vMab = a - b;
        const Eigen::Vector3d vMbc = b - c;

        const double denomP = std::max(1e-6, vPab.norm() * vPbc.norm());
        const double denomM = std::max(1e-6, vMab.norm() * vMbc.norm());
        const double cosP = std::max(-1.0, std::min(1.0, vPab.dot(vPbc) / denomP));
        const double cosM = std::max(-1.0, std::min(1.0, vMab.dot(vMbc) / denomM));
        _error[1] = std::acos(cosP) - std::acos(cosM);
    }
};

class EdgePanopticProjection
    : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgePanopticProjection()
    {
    }

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError()
    {
        const g2o::VertexSE3Expmap* vPose =
            static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
        const g2o::VertexSBAPointXYZ* vPoint =
            static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);

        const Eigen::Vector3d pointCam = vPose->estimate().map(vPoint->estimate());
        if(pointCam[2] <= 0.0)
        {
            _error << 1e6, 1e6;
            return;
        }

        _error = _measurement - pCamera->project(pointCam);
    }

    bool isDepthPositive() const
    {
        const g2o::VertexSE3Expmap* vPose =
            static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
        const g2o::VertexSBAPointXYZ* vPoint =
            static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        return vPose->estimate().map(vPoint->estimate())[2] > 0.0;
    }

    virtual void linearizeOplus()
    {
        g2o::VertexSE3Expmap* vPose = static_cast<g2o::VertexSE3Expmap*>(_vertices[1]);
        g2o::SE3Quat Tcw(vPose->estimate());
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const Eigen::Vector3d worldPoint = vPoint->estimate();
        const Eigen::Vector3d pointCam = Tcw.map(worldPoint);

        Eigen::Matrix<double,2,3> projectJac = (-pCamera->projectJac(pointCam)).eval();
        _jacobianOplusXi = projectJac * Tcw.rotation().toRotationMatrix();

        const double x = pointCam[0];
        const double y = pointCam[1];
        const double z = pointCam[2];
        Eigen::Matrix<double,3,6> se3Deriv;
        se3Deriv << 0.0, z,   -y, 1.0, 0.0, 0.0,
                    -z,  0.0, x, 0.0, 1.0, 0.0,
                     y,  -x, 0.0, 0.0, 0.0, 1.0;
        _jacobianOplusXj = projectJac * se3Deriv;
    }

    GeometricCamera* pCamera = NULL;
};

class EdgePanopticInstanceProjection
    : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgePanopticInstanceProjection()
        : mLocalPoint(Eigen::Vector3d::Zero()), pCamera(NULL)
    {
    }

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void SetLocalPoint(const Eigen::Vector3d& localPoint)
    {
        mLocalPoint = localPoint;
    }

    void computeError()
    {
        const g2o::VertexSE3Expmap* vInstancePose =
            static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        const g2o::VertexSE3Expmap* vCameraPose =
            static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);

        const Eigen::Vector3d pointWorld = vInstancePose->estimate().map(mLocalPoint);
        const Eigen::Vector3d pointCam = vCameraPose->estimate().map(pointWorld);
        if(pointCam[2] <= 0.0)
        {
            _error << 1e6, 1e6;
            return;
        }

        _error = _measurement - pCamera->project(pointCam);
    }

    bool isDepthPositive() const
    {
        const g2o::VertexSE3Expmap* vInstancePose =
            static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        const g2o::VertexSE3Expmap* vCameraPose =
            static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
        const Eigen::Vector3d pointWorld = vInstancePose->estimate().map(mLocalPoint);
        return vCameraPose->estimate().map(pointWorld)[2] > 0.0;
    }

    virtual void linearizeOplus()
    {
        g2o::VertexSE3Expmap* vInstancePose =
            static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
        g2o::VertexSE3Expmap* vCameraPose =
            static_cast<g2o::VertexSE3Expmap*>(_vertices[1]);

        const g2o::SE3Quat Tow(vInstancePose->estimate());
        const g2o::SE3Quat Tcw(vCameraPose->estimate());
        const Eigen::Vector3d pointWorld = Tow.map(mLocalPoint);
        const Eigen::Vector3d pointCam = Tcw.map(pointWorld);

        Eigen::Matrix<double,2,3> projectJac = (-pCamera->projectJac(pointCam)).eval();
        const Eigen::Matrix3d rotationCamera = Tcw.rotation().toRotationMatrix();

        const double wx = pointWorld[0];
        const double wy = pointWorld[1];
        const double wz = pointWorld[2];
        Eigen::Matrix<double,3,6> instanceDeriv;
        instanceDeriv << 0.0, wz,  -wy, 1.0, 0.0, 0.0,
                        -wz,  0.0, wx, 0.0, 1.0, 0.0,
                         wy, -wx, 0.0, 0.0, 0.0, 1.0;
        _jacobianOplusXi = projectJac * rotationCamera * instanceDeriv;

        const double x = pointCam[0];
        const double y = pointCam[1];
        const double z = pointCam[2];
        Eigen::Matrix<double,3,6> cameraDeriv;
        cameraDeriv << 0.0, z,   -y, 1.0, 0.0, 0.0,
                      -z,  0.0, x, 0.0, 1.0, 0.0,
                       y,  -x, 0.0, 0.0, 0.0, 1.0;
        _jacobianOplusXj = projectJac * cameraDeriv;
    }

    Eigen::Vector3d mLocalPoint;
    GeometricCamera* pCamera;
};

class EdgeRigidityPair : public g2o::BaseMultiEdge<1, double>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeRigidityPair()
    {
        resize(4);
    }

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError()
    {
        const g2o::VertexSBAPointXYZ* vA0 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const g2o::VertexSBAPointXYZ* vB0 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[1]);
        const g2o::VertexSBAPointXYZ* vA1 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[2]);
        const g2o::VertexSBAPointXYZ* vB1 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[3]);

        const double dist0 = (vA0->estimate() - vB0->estimate()).norm();
        const double dist1 = (vA1->estimate() - vB1->estimate()).norm();
        _error[0] = dist0 - dist1;
    }
};

class EdgeRigidityFramePair : public g2o::BaseMultiEdge<1, double>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit EdgeRigidityFramePair(const size_t nCommonPoints = 0)
        : mnCommonPoints(nCommonPoints)
    {
        resize(2 * mnCommonPoints);
    }

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError()
    {
        if(mnCommonPoints < 2)
        {
            _error[0] = 0.0;
            return;
        }

        double sumSq = 0.0;
        size_t nPairs = 0;
        for(size_t i = 0; i < mnCommonPoints; ++i)
        {
            const g2o::VertexSBAPointXYZ* vPrevI =
                static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[i]);
            const g2o::VertexSBAPointXYZ* vCurrI =
                static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[mnCommonPoints + i]);
            for(size_t j = i + 1; j < mnCommonPoints; ++j)
            {
                const g2o::VertexSBAPointXYZ* vPrevJ =
                    static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[j]);
                const g2o::VertexSBAPointXYZ* vCurrJ =
                    static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[mnCommonPoints + j]);
                const double distPrev = (vPrevI->estimate() - vPrevJ->estimate()).norm();
                const double distCurr = (vCurrI->estimate() - vCurrJ->estimate()).norm();
                const double diff = distCurr - distPrev;
                sumSq += diff * diff;
                ++nPairs;
            }
        }

        if(nPairs == 0)
        {
            _error[0] = 0.0;
            return;
        }

        // A scalar g2o edge represents the Eq. (15) Frobenius matrix residual.
        // D_k is symmetric with a zero diagonal, so ||D_k||_F^2 = 2 * sum_{i<j} d_ij^2.
        _error[0] = std::sqrt(2.0 * sumSq);
    }

private:
    size_t mnCommonPoints;
};

class EdgePointXYZPrior : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSBAPointXYZ>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError()
    {
        const g2o::VertexSBAPointXYZ* vPoint = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        _error = vPoint->estimate() - _measurement;
    }

    virtual void linearizeOplus()
    {
        _jacobianOplusXi = Eigen::Matrix3d::Identity();
    }
};

class EdgeInstancePointStructure
    : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError()
    {
        const g2o::VertexSBAPointXYZ* vPoint =
            static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const g2o::VertexSE3Expmap* vInstancePose =
            static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);

        _error = vPoint->estimate() - vInstancePose->estimate().map(_measurement);
    }
};

bool SortMapPointsById(MapPoint* pA, MapPoint* pB)
{
    if(pA == pB)
        return false;
    if(!pA)
        return false;
    if(!pB)
        return true;
    return pA->mnId < pB->mnId;
}

std::vector<std::tuple<MapPoint*, MapPoint*, MapPoint*> > SampleTripletsInFrame(
    const std::vector<MapPoint*>& vPoints,
    const std::map<MapPoint*, int>& mLeftIndices,
    const ImageFrameHandle& frame,
    size_t maxSamples)
{
    std::vector<std::tuple<MapPoint*, MapPoint*, MapPoint*> > vTriplets;
    if(vPoints.size() < 3)
        return vTriplets;

    const std::vector<cv::KeyPoint>& vKeysUn = GetFrameKeysUn(frame);

    std::vector<MapPoint*> vValidPoints;
    vValidPoints.reserve(vPoints.size());
    for(size_t i = 0; i < vPoints.size(); ++i)
    {
        if(vPoints[i] && mLeftIndices.count(vPoints[i]))
            vValidPoints.push_back(vPoints[i]);
    }
    if(vValidPoints.size() < 3)
        return vTriplets;

    std::sort(vValidPoints.begin(), vValidPoints.end(), SortMapPointsById);

    if(maxSamples == 0)
    {
        const size_t totalTriplets =
            (vValidPoints.size() * (vValidPoints.size() - 1) * (vValidPoints.size() - 2)) / 6;
        vTriplets.reserve(totalTriplets);
        for(size_t i = 0; i < vValidPoints.size(); ++i)
        {
            for(size_t j = i + 1; j < vValidPoints.size(); ++j)
            {
                for(size_t k = j + 1; k < vValidPoints.size(); ++k)
                {
                    vTriplets.push_back(std::make_tuple(vValidPoints[i], vValidPoints[j], vValidPoints[k]));
                }
            }
        }
        return vTriplets;
    }

    const size_t nAnchors = (maxSamples == 0) ? vValidPoints.size() : std::min(maxSamples, vValidPoints.size());
    std::set<std::tuple<unsigned long, unsigned long, unsigned long> > sSeenTriplets;
    vTriplets.reserve(nAnchors);
    for(size_t i = 0; i < nAnchors; ++i)
    {
        MapPoint* pAnchor = vValidPoints[i];
        const cv::Point2f anchorPt = vKeysUn[mLeftIndices.at(pAnchor)].pt;

        double bestDist1 = -1.0;
        MapPoint* pBest1 = NULL;
        for(size_t j = 0; j < vValidPoints.size(); ++j)
        {
            MapPoint* pCandidate = vValidPoints[j];
            if(pCandidate == pAnchor)
                continue;

            const cv::Point2f candidatePt = vKeysUn[mLeftIndices.at(pCandidate)].pt;
            const double dx = static_cast<double>(anchorPt.x - candidatePt.x);
            const double dy = static_cast<double>(anchorPt.y - candidatePt.y);
            const double dist2 = dx * dx + dy * dy;
            if(dist2 > bestDist1)
            {
                bestDist1 = dist2;
                pBest1 = pCandidate;
            }
        }

        if(!pBest1)
            continue;

        const cv::Point2f best1Pt = vKeysUn[mLeftIndices.at(pBest1)].pt;
        const double baseX = static_cast<double>(best1Pt.x - anchorPt.x);
        const double baseY = static_cast<double>(best1Pt.y - anchorPt.y);
        double bestArea2 = -1.0;
        double bestFallbackDist = -1.0;
        MapPoint* pBest2 = NULL;
        for(size_t j = 0; j < vValidPoints.size(); ++j)
        {
            MapPoint* pCandidate = vValidPoints[j];
            if(pCandidate == pAnchor || pCandidate == pBest1)
                continue;

            const cv::Point2f candidatePt = vKeysUn[mLeftIndices.at(pCandidate)].pt;
            const double candX = static_cast<double>(candidatePt.x - anchorPt.x);
            const double candY = static_cast<double>(candidatePt.y - anchorPt.y);
            const double cross = baseX * candY - baseY * candX;
            const double area2 = cross * cross;
            const double dist2 = candX * candX + candY * candY;
            if(area2 > bestArea2)
            {
                bestArea2 = area2;
                pBest2 = pCandidate;
            }
            if(dist2 > bestFallbackDist)
            {
                bestFallbackDist = dist2;
                if(bestArea2 <= 1e-12)
                    pBest2 = pCandidate;
            }
        }

        if(!pBest1 || !pBest2)
            continue;

        std::array<MapPoint*, 3> vTripletPoints = {pAnchor, pBest1, pBest2};
        std::sort(vTripletPoints.begin(), vTripletPoints.end(), SortMapPointsById);
        const std::tuple<unsigned long, unsigned long, unsigned long> tripletKey(
            vTripletPoints[0]->mnId,
            vTripletPoints[1]->mnId,
            vTripletPoints[2]->mnId);
        if(!sSeenTriplets.insert(tripletKey).second)
            continue;

        vTriplets.push_back(std::make_tuple(vTripletPoints[0], vTripletPoints[1], vTripletPoints[2]));
    }

    return vTriplets;
}

std::vector<std::pair<MapPoint*, MapPoint*> > SamplePairsInFrame(
    const std::vector<MapPoint*>& vPoints, size_t maxPairs)
{
    std::vector<std::pair<MapPoint*, MapPoint*> > vPairs;
    if(vPoints.size() < 2)
        return vPairs;

    std::vector<MapPoint*> vSortedPoints = vPoints;
    std::sort(vSortedPoints.begin(), vSortedPoints.end(), SortMapPointsById);

    const size_t totalPairs = (vSortedPoints.size() * (vSortedPoints.size() - 1)) / 2;
    const bool useAllPairs = (maxPairs == 0 || maxPairs >= totalPairs);

    vPairs.reserve(useAllPairs ? totalPairs : maxPairs);
    for(size_t i = 0; i < vSortedPoints.size() && (useAllPairs || vPairs.size() < maxPairs); ++i)
    {
        for(size_t j = i + 1; j < vSortedPoints.size() && (useAllPairs || vPairs.size() < maxPairs); ++j)
        {
            vPairs.push_back(std::make_pair(vSortedPoints[i], vSortedPoints[j]));
        }
    }

    return vPairs;
}

Eigen::Vector3d AveragePointEstimate(g2o::SparseOptimizer& optimizer, const std::vector<int>& vVertexIds)
{
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    size_t nValid = 0;

    for(size_t i = 0; i < vVertexIds.size(); ++i)
    {
        g2o::VertexSBAPointXYZ* vPoint =
            static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(vVertexIds[i]));
        if(!vPoint)
            continue;

        mean += vPoint->estimate();
        ++nValid;
    }

    if(nValid > 0)
        mean /= static_cast<double>(nValid);

    return mean;
}

Eigen::Vector3d EstimateThingVertexInitialization(MapPoint* pMP, Instance* pInstance, KeyFrame* pKF)
{
    Eigen::Vector3f initPos = pMP->GetWorldPos();
    if(pInstance && pInstance->IsInitialized() && pInstance->HasReliableInitializationMotion() && pKF)
        initPos = pInstance->GetMotionPriorForKeyFrame(pKF) * initPos;

    return initPos.cast<double>();
}

Eigen::Vector3d EstimateThingVertexInitialization(MapPoint* pMP,
                                                  Instance* pInstance,
                                                  const ImageFrameHandle& frame)
{
    if(frame.isKeyFrame)
        return EstimateThingVertexInitialization(pMP, pInstance, frame.pKeyFrame);

    Eigen::Vector3f initPos = pMP->GetWorldPos();
    if(frame.pWindowFrame)
    {
        const int instanceId = pMP->GetInstanceId();
        const std::map<int, Sophus::SE3f>::const_iterator itMotion =
            frame.pWindowFrame->mmPredictedInstanceMotions.find(instanceId);
        if(itMotion != frame.pWindowFrame->mmPredictedInstanceMotions.end())
            initPos = itMotion->second * initPos;
        else if(pInstance && pInstance->IsInitialized() && pInstance->HasReliableInitializationMotion())
            initPos = pInstance->GetVelocity() * initPos;
    }

    return initPos.cast<double>();
}

Vector6d BuildTripletMeasurement(const std::map<MapPoint*, int>& mLeftIndices,
                                 const ImageFrameHandle& frame,
                                 MapPoint* pA,
                                 MapPoint* pB,
                                 MapPoint* pC)
{
    Vector6d measurement = Vector6d::Zero();
    const std::vector<cv::KeyPoint>& vKeysUn = GetFrameKeysUn(frame);
    const cv::KeyPoint& kpA = vKeysUn[mLeftIndices.at(pA)];
    const cv::KeyPoint& kpB = vKeysUn[mLeftIndices.at(pB)];
    const cv::KeyPoint& kpC = vKeysUn[mLeftIndices.at(pC)];

    measurement << kpA.pt.x, kpA.pt.y,
                   kpB.pt.x, kpB.pt.y,
                   kpC.pt.x, kpC.pt.y;
    return measurement;
}

double ComputeTripletInvSigma2(const std::map<MapPoint*, int>& mLeftIndices,
                               const ImageFrameHandle& frame,
                               MapPoint* pA,
                               MapPoint* pB,
                               MapPoint* pC)
{
    const int idxA = mLeftIndices.at(pA);
    const int idxB = mLeftIndices.at(pB);
    const int idxC = mLeftIndices.at(pC);

    const std::vector<float>& vInvLevelSigma2 = GetFrameInvLevelSigma2(frame);
    const std::vector<cv::KeyPoint>& vKeysUn = GetFrameKeysUn(frame);
    const double invSigma2A = vInvLevelSigma2[vKeysUn[idxA].octave];
    const double invSigma2B = vInvLevelSigma2[vKeysUn[idxB].octave];
    const double invSigma2C = vInvLevelSigma2[vKeysUn[idxC].octave];
    return (invSigma2A + invSigma2B + invSigma2C) / 3.0;
}

struct ClassRadiusStats
{
    double sumRadius = 0.0;
    double sumRadiusSq = 0.0;
    int count = 0;
};

std::map<int, ClassRadiusStats> BuildClassRadiusStats(Map* pMap, const std::set<int>& sExcludedInstanceIds)
{
    std::map<int, ClassRadiusStats> mStats;
    if(!pMap)
        return mStats;

    const std::vector<Instance*> vInstances = pMap->GetAllInstances();
    for(size_t i = 0; i < vInstances.size(); ++i)
    {
        Instance* pInstance = vInstances[i];
        if(!pInstance || pInstance->GetSemanticLabel() <= 0)
            continue;
        const std::vector<Eigen::Vector3f> vShapeTemplate = pInstance->GetShapeTemplate();
        if(static_cast<int>(vShapeTemplate.size()) < GetDynamicBackendMinFrameSupport())
            continue;
        if(sExcludedInstanceIds.count(pInstance->GetId()))
            continue;

        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        for(size_t j = 0; j < vShapeTemplate.size(); ++j)
            centroid += vShapeTemplate[j];
        centroid /= static_cast<float>(vShapeTemplate.size());

        double meanRadius = 0.0;
        for(size_t j = 0; j < vShapeTemplate.size(); ++j)
            meanRadius += (vShapeTemplate[j] - centroid).norm();
        meanRadius /= static_cast<double>(vShapeTemplate.size());

        ClassRadiusStats& stats = mStats[pInstance->GetSemanticLabel()];
        stats.sumRadius += meanRadius;
        stats.sumRadiusSq += meanRadius * meanRadius;
        ++stats.count;
    }

    const std::map<int, Map::InstanceClassSizePrior> mPersistentPriors =
        pMap->GetAllInstanceClassSizePriors();
    for(std::map<int, Map::InstanceClassSizePrior>::const_iterator itPrior = mPersistentPriors.begin();
        itPrior != mPersistentPriors.end(); ++itPrior)
    {
        ClassRadiusStats& stats = mStats[itPrior->first];
        if(!itPrior->second.perInstance.empty())
        {
            for(std::map<int, Map::InstanceClassSizePrior::RadiusAccumulator>::const_iterator
                    itInstancePrior = itPrior->second.perInstance.begin();
                itInstancePrior != itPrior->second.perInstance.end(); ++itInstancePrior)
            {
                if(sExcludedInstanceIds.count(itInstancePrior->first) ||
                   itInstancePrior->second.count <= 0)
                    continue;

                const double meanRadius =
                    itInstancePrior->second.sumRadius /
                    static_cast<double>(itInstancePrior->second.count);
                stats.sumRadius += meanRadius;
                stats.sumRadiusSq += meanRadius * meanRadius;
                ++stats.count;
            }
            continue;
        }

        if(itPrior->second.count > 0)
        {
            stats.sumRadius += itPrior->second.sumRadius;
            stats.sumRadiusSq += itPrior->second.sumRadiusSq;
            stats.count += itPrior->second.count;
        }
    }

    return mStats;
}

bool GetClassRadiusPrior(const std::map<int, ClassRadiusStats>& mStats,
                         int semanticLabel,
                         double& meanRadius,
                         double& stdRadius,
                         int& sampleCount)
{
    const std::map<int, ClassRadiusStats>::const_iterator itStats = mStats.find(semanticLabel);
    if(itStats == mStats.end() || itStats->second.count <= 0)
        return false;

    sampleCount = itStats->second.count;
    meanRadius = itStats->second.sumRadius / static_cast<double>(sampleCount);
    const double meanSq = itStats->second.sumRadiusSq / static_cast<double>(sampleCount);
    stdRadius = std::sqrt(std::max(0.0, meanSq - meanRadius * meanRadius));
    return true;
}

bool DisableDynamicBackendForLBA()
{
    const char* envValue = std::getenv("STSLAM_DISABLE_DYNAMIC_BACKEND");
    return envValue && std::string(envValue) != "0";
}

bool EnableDynamicLBADebug()
{
    const char* envValue = std::getenv("STSLAM_DEBUG_DYNAMIC_LBA");
    return envValue && std::string(envValue) != "0";
}

bool DisableDynamicShapeForLBA()
{
    const char* envValue = std::getenv("STSLAM_DISABLE_DYNAMIC_SHAPE");
    return envValue && std::string(envValue) != "0";
}

bool DisableDynamicRigidityForLBA()
{
    const char* envValue = std::getenv("STSLAM_DISABLE_DYNAMIC_RIGIDITY");
    return envValue && std::string(envValue) != "0";
}

bool DisableEq17SizePriorForLBA()
{
    const char* envValue = std::getenv("STSLAM_DISABLE_EQ17_SIZE_PRIOR");
    return envValue && std::string(envValue) != "0";
}

bool EnableDynamicPointPriorForLBA()
{
    const char* envValue = std::getenv("STSLAM_ENABLE_DYNAMIC_POINT_PRIOR");
    return envValue && std::string(envValue) != "0";
}

double RotationAngleDeg(const Eigen::Matrix3f& rotation)
{
    const Eigen::AngleAxisf angleAxis(rotation);
    return static_cast<double>(std::abs(angleAxis.angle())) * 180.0 / 3.14159265358979323846;
}

bool IsFiniteSE3Estimate(const Sophus::SE3f& pose)
{
    return pose.rotationMatrix().allFinite() && pose.translation().allFinite();
}

Instance::DynamicEntityMotionState ClassifyDynamicEntityMotionStateFromVelocity(
    const Sophus::SE3f& velocity,
    bool reliable,
    int semanticLabel)
{
    if(!IsFiniteSE3Estimate(velocity))
        return Instance::kUncertainDynamicEntity;

    const double translationNorm = velocity.translation().cast<double>().norm();
    const double rotationDeg = RotationAngleDeg(velocity.rotationMatrix());
    const double zeroTranslation =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_ENTITY_ZERO_TRANSLATION", 0.03, 0.0);
    const double zeroRotationDeg =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_ENTITY_ZERO_ROTATION_DEG", 2.0, 0.0);
    if(IsArticulatedSemanticLabel(semanticLabel))
    {
        if(translationNorm <= zeroTranslation)
            return Instance::kZeroVelocityDynamicEntity;
        return reliable ? Instance::kMovingDynamicEntity : Instance::kUncertainDynamicEntity;
    }

    if(translationNorm <= zeroTranslation && rotationDeg <= zeroRotationDeg)
        return Instance::kZeroVelocityDynamicEntity;

    return reliable ? Instance::kMovingDynamicEntity : Instance::kUncertainDynamicEntity;
}

Sophus::SE3f CanonicalVelocityForDynamicEntityState(
    const Sophus::SE3f& velocity,
    Instance::DynamicEntityMotionState state)
{
    if(state == Instance::kZeroVelocityDynamicEntity)
        return Sophus::SE3f();
    return IsFiniteSE3Estimate(velocity) ? velocity : Sophus::SE3f();
}

const char* DynamicEntityMotionStateName(Instance::DynamicEntityMotionState state)
{
    if(state == Instance::kZeroVelocityDynamicEntity)
        return "zero_velocity_dynamic_entity";
    if(state == Instance::kMovingDynamicEntity)
        return "moving_dynamic_entity";
    if(state == Instance::kUncertainDynamicEntity)
        return "uncertain_dynamic_entity";
    return "unknown";
}

Sophus::SE3f NormalizeMotionByFrameGap(const Sophus::SE3f& motion, int frameGap)
{
    if(frameGap <= 1)
        return motion;

    Sophus::SE3f::Tangent tangent = motion.log();
    if(!tangent.allFinite())
        return motion;

    tangent /= static_cast<float>(frameGap);
    const Sophus::SE3f normalized = Sophus::SE3f::exp(tangent);
    return IsFiniteSE3Estimate(normalized) ? normalized : motion;
}

Sophus::SE3f CompoundMotionByFrameGap(const Sophus::SE3f& motion, int frameGap)
{
    if(frameGap <= 1 || !IsFiniteSE3Estimate(motion))
        return motion;

    Sophus::SE3f::Tangent tangent = motion.log();
    if(!tangent.allFinite())
        return motion;

    tangent *= static_cast<float>(frameGap);
    const Sophus::SE3f compounded = Sophus::SE3f::exp(tangent);
    return IsFiniteSE3Estimate(compounded) ? compounded : motion;
}

int GetInstancePoseStatePredictionMaxFrameGap()
{
    return GetEnvIntOrDefault("STSLAM_INSTANCE_POSE_STATE_PREDICTION_MAX_GAP", 10, 0);
}

bool CanUseMotionStateVelocityForPoseInit(const Instance::InstanceMotionStateRecord& record)
{
    if(!IsFiniteSE3Estimate(record.velocity))
        return false;

    if(record.state == Instance::kMovingDynamicEntity)
        return true;

    if(record.state == Instance::kZeroVelocityDynamicEntity)
        return true;

    // SVD/VPS can be uncertain at initialization.  Use it only as an
    // initialization prior; the STSLAM factors still decide the optimized state.
    return record.state == Instance::kUncertainDynamicEntity &&
           record.confidence >= 0.30;
}

bool PredictInstancePoseFromStateChain(Instance* pInstance,
                                       unsigned long frameId,
                                       Sophus::SE3f& pose,
                                       std::string& source)
{
    if(!pInstance || !pInstance->IsInitialized())
        return false;

    Instance::InstanceMotionStateRecord exactRecord;
    if(pInstance->GetInstanceMotionState(frameId, exactRecord) &&
       IsFiniteSE3Estimate(exactRecord.pose))
    {
        pose = exactRecord.pose;
        source = "state_exact";
        return true;
    }

    Instance::InstanceMotionStateRecord latestRecord;
    if(!pInstance->GetLatestInstanceMotionState(latestRecord) ||
       !IsFiniteSE3Estimate(latestRecord.pose) ||
       latestRecord.frameId >= frameId)
    {
        return false;
    }

    const unsigned long frameGap = frameId - latestRecord.frameId;
    const int maxFrameGap = GetInstancePoseStatePredictionMaxFrameGap();
    if(maxFrameGap > 0 && frameGap > static_cast<unsigned long>(maxFrameGap))
        return false;

    if(!CanUseMotionStateVelocityForPoseInit(latestRecord))
        return false;

    Sophus::SE3f predictedPose = latestRecord.pose;
    for(unsigned long step = 0; step < frameGap; ++step)
    {
        predictedPose = latestRecord.velocity * predictedPose;
        if(!IsFiniteSE3Estimate(predictedPose))
            return false;
    }

    pose = predictedPose;
    source = latestRecord.state == Instance::kMovingDynamicEntity ?
        "state_predicted_moving" :
        (latestRecord.state == Instance::kZeroVelocityDynamicEntity ?
             "state_predicted_zero_velocity" :
             "state_predicted_uncertain");
    return true;
}

bool GetInstanceMotionPriorForFrame(Instance* pInstance,
                                    unsigned long frameId,
                                    int frameGap,
                                    Sophus::SE3f& motion,
                                    std::string& source)
{
    if(!pInstance || !pInstance->IsInitialized())
        return false;

    Instance::InstanceMotionStateRecord exactRecord;
    if(pInstance->GetInstanceMotionState(frameId, exactRecord) &&
       CanUseMotionStateVelocityForPoseInit(exactRecord))
    {
        motion = CompoundMotionByFrameGap(exactRecord.velocity, frameGap);
        source = "state_exact_velocity";
        return IsFiniteSE3Estimate(motion);
    }

    Instance::InstanceMotionStateRecord latestRecord;
    if(!pInstance->GetLatestInstanceMotionState(latestRecord) ||
       latestRecord.frameId > frameId ||
       !CanUseMotionStateVelocityForPoseInit(latestRecord))
    {
        return false;
    }

    motion = CompoundMotionByFrameGap(latestRecord.velocity, frameGap);
    source = latestRecord.state == Instance::kMovingDynamicEntity ?
        "state_latest_moving_velocity" :
        (latestRecord.state == Instance::kZeroVelocityDynamicEntity ?
             "state_latest_zero_velocity" :
             "state_latest_uncertain_velocity");
    return IsFiniteSE3Estimate(motion);
}

bool UsePanopticPredictionResidualGate()
{
    return GetEnvFlagOrDefault("STSLAM_PANOPTIC_PREDICTION_RESIDUAL_GATE",
                               !UseStrictPaperArchitectureDefaults());
}

bool EstimateCommonPointTranslation(const std::vector<Eigen::Vector3f>& vPrevPoints,
                                    const std::vector<Eigen::Vector3f>& vCurrPoints,
                                    Sophus::SE3f& motion)
{
    if(vPrevPoints.size() != vCurrPoints.size() || vPrevPoints.size() < 3)
        return false;

    Eigen::Vector3f prevCentroid = Eigen::Vector3f::Zero();
    Eigen::Vector3f currCentroid = Eigen::Vector3f::Zero();
    for(size_t i = 0; i < vPrevPoints.size(); ++i)
    {
        if(!vPrevPoints[i].allFinite() || !vCurrPoints[i].allFinite())
            return false;
        prevCentroid += vPrevPoints[i];
        currCentroid += vCurrPoints[i];
    }
    prevCentroid /= static_cast<float>(vPrevPoints.size());
    currCentroid /= static_cast<float>(vCurrPoints.size());

    const Eigen::Vector3f translation = currCentroid - prevCentroid;
    motion = Sophus::SE3f(Eigen::Matrix3f::Identity(), translation);
    return IsFiniteSE3Estimate(motion);
}

double GetDynamicObservationWeight(Map* pMap, MapPoint* pMP, unsigned long frameId)
{
    if(!pMap || !pMP || pMP->GetInstanceId() <= 0)
        return 1.0;

    Instance* pInstance = pMap->GetInstance(pMP->GetInstanceId());
    if(!pInstance)
        return 1.0;

    double weight = pInstance->GetBackendOutlierWeight(frameId, pMP);
    if(pInstance->HasObservationQualityWeight(frameId, pMP))
    {
        const double qualityWeight = pInstance->GetObservationQualityWeight(frameId, pMP);
        if(std::isfinite(qualityWeight))
            weight *= std::max(0.0, std::min(1.0, qualityWeight));
    }
    return std::max(0.0, std::min(1.0, weight));
}

int GetInstanceSemanticLabel(Map* pMap, const int instanceId)
{
    if(!pMap || instanceId <= 0)
        return 0;
    Instance* pInstance = pMap->GetInstance(instanceId);
    return pInstance ? pInstance->GetSemanticLabel() : 0;
}

double GetInstanceShapeFactorScale(Map* pMap, const int instanceId)
{
    const int semanticLabel = GetInstanceSemanticLabel(pMap, instanceId);
    return IsArticulatedSemanticLabel(semanticLabel) ? GetArticulatedShapeFactorScale() : 1.0;
}

double GetInstanceRigidityFactorScale(Map* pMap, const int instanceId)
{
    const int semanticLabel = GetInstanceSemanticLabel(pMap, instanceId);
    return IsArticulatedSemanticLabel(semanticLabel) ? GetArticulatedRigidityFactorScale() : 1.0;
}

double GetDynamicObservationQuality(Map* pMap,
                                    MapPoint* pMP,
                                    unsigned long frameId,
                                    bool* pHasQuality)
{
    if(pHasQuality)
        *pHasQuality = false;

    if(!pMap || !pMP || pMP->GetInstanceId() <= 0)
        return 0.0;

    Instance* pInstance = pMap->GetInstance(pMP->GetInstanceId());
    if(!pInstance || !pInstance->HasObservationQualityWeight(frameId, pMP))
        return 0.0;

    if(pHasQuality)
        *pHasQuality = true;
    return pInstance->GetObservationQualityWeight(frameId, pMP);
}

double GetMeanDynamicObservationWeight(Map* pMap,
                                       const std::vector<std::pair<MapPoint*, unsigned long> >& vObservations)
{
    if(vObservations.empty())
        return 1.0;

    double weight = 0.0;
    size_t count = 0;
    for(size_t i = 0; i < vObservations.size(); ++i)
    {
        weight += GetDynamicObservationWeight(pMap, vObservations[i].first, vObservations[i].second);
        ++count;
    }
    return count > 0 ? (weight / static_cast<double>(count)) : 1.0;
}

double GetMeanDynamicObservationWeight(Map* pMap,
                                       const std::vector<DynamicEdgeObservation>& vObservations)
{
    if(vObservations.empty())
        return 1.0;

    double weight = 0.0;
    size_t count = 0;
    for(size_t i = 0; i < vObservations.size(); ++i)
    {
        weight += GetDynamicObservationWeight(pMap, vObservations[i].pMP, vObservations[i].frameId);
        ++count;
    }
    return count > 0 ? (weight / static_cast<double>(count)) : 1.0;
}

std::vector<std::pair<MapPoint*, MapPoint*> > SampleQualityWeightedPairsInFrame(
    Map* pMap,
    const std::vector<MapPoint*>& vPoints,
    const unsigned long prevFrameId,
    const unsigned long currFrameId,
    const size_t maxPairs,
    const size_t maxPairsPerPoint,
    const bool preferSpatialCoverage)
{
    std::vector<std::pair<MapPoint*, MapPoint*> > vPairs;
    if(vPoints.size() < 2)
        return vPairs;

    const size_t totalPairs = (vPoints.size() * (vPoints.size() - 1)) / 2;
    if(maxPairs == 0 || maxPairs >= totalPairs)
        return SamplePairsInFrame(vPoints, maxPairs);

    std::vector<MapPoint*> vSortedPoints = vPoints;
    std::sort(vSortedPoints.begin(), vSortedPoints.end(), SortMapPointsById);

    const double minPairQuality = GetRigidityPairMinQuality();
    const int minKnownObservations = GetRigidityPairMinKnownObservations();
    const size_t autoDegreeLimit =
        vSortedPoints.empty() ? 0 :
        std::max<size_t>(
            4,
            std::min<size_t>(
                32,
                (2 * maxPairs + vSortedPoints.size() - 1) / vSortedPoints.size() + 1));
    const size_t degreeLimit = maxPairsPerPoint > 0 ? maxPairsPerPoint : autoDegreeLimit;
    std::map<MapPoint*, size_t> mPointDegrees;

    struct PairCandidate
    {
        MapPoint* pA;
        MapPoint* pB;
        double quality;
        int knownObservations;
        double spatialDistance;
    };

    std::vector<PairCandidate> vCandidates;
    vCandidates.reserve(totalPairs);
    size_t nRejectedByKnownObs = 0;
    size_t nRejectedByQuality = 0;

    for(size_t i = 0; i < vSortedPoints.size(); ++i)
    {
        MapPoint* pA = vSortedPoints[i];
        if(!pA)
            continue;

        for(size_t j = i + 1; j < vSortedPoints.size(); ++j)
        {
            MapPoint* pB = vSortedPoints[j];
            if(!pB)
                continue;

            bool hasQuality[4] = {false, false, false, false};
            const double quality[4] = {
                GetDynamicObservationQuality(pMap, pA, prevFrameId, &hasQuality[0]),
                GetDynamicObservationQuality(pMap, pA, currFrameId, &hasQuality[1]),
                GetDynamicObservationQuality(pMap, pB, prevFrameId, &hasQuality[2]),
                GetDynamicObservationQuality(pMap, pB, currFrameId, &hasQuality[3])
            };

            int knownObservations = 0;
            double pairQuality = 1.0;
            for(size_t k = 0; k < 4; ++k)
            {
                if(!hasQuality[k])
                    continue;
                ++knownObservations;
                pairQuality = std::min(pairQuality, quality[k]);
            }

            if(knownObservations < minKnownObservations)
            {
                ++nRejectedByKnownObs;
                continue;
            }

            if(!std::isfinite(pairQuality) || pairQuality < minPairQuality)
            {
                ++nRejectedByQuality;
                continue;
            }

            const Eigen::Vector3f pointA = pA->GetWorldPos();
            const Eigen::Vector3f pointB = pB->GetWorldPos();
            const double spatialDistance =
                (pointA.allFinite() && pointB.allFinite()) ?
                static_cast<double>((pointA - pointB).norm()) : 0.0;

            PairCandidate candidate;
            candidate.pA = pA;
            candidate.pB = pB;
            candidate.quality = pairQuality;
            candidate.knownObservations = knownObservations;
            candidate.spatialDistance = std::isfinite(spatialDistance) ? spatialDistance : 0.0;
            vCandidates.push_back(candidate);
        }
    }

    std::vector<double> vSelectedQualities;
    std::vector<double> vSelectedDistances;
    vSelectedQualities.reserve(maxPairs);
    vSelectedDistances.reserve(maxPairs);

    std::sort(vCandidates.begin(), vCandidates.end(),
              [](const PairCandidate& lhs, const PairCandidate& rhs)
              {
                  if(lhs.spatialDistance != rhs.spatialDistance)
                      return lhs.spatialDistance > rhs.spatialDistance;
                  if(lhs.quality != rhs.quality)
                      return lhs.quality > rhs.quality;
                  if(lhs.knownObservations != rhs.knownObservations)
                      return lhs.knownObservations > rhs.knownObservations;
                  if(lhs.pA->mnId != rhs.pA->mnId)
                      return lhs.pA->mnId < rhs.pA->mnId;
                  return lhs.pB->mnId < rhs.pB->mnId;
              });

    vPairs.reserve(maxPairs);
    size_t nRejectedByDegree = 0;
    std::set<MapPoint*> sSelectedPoints;
    std::set<std::pair<unsigned long, unsigned long> > sSelectedPairIds;
    auto getPointDegree = [&mPointDegrees](MapPoint* pMP) -> size_t
    {
        const std::map<MapPoint*, size_t>::const_iterator it = mPointDegrees.find(pMP);
        return it == mPointDegrees.end() ? 0 : it->second;
    };

    if(preferSpatialCoverage)
    {
        for(size_t i = 0; i < vCandidates.size() && vPairs.size() < maxPairs; ++i)
        {
            MapPoint* pA = vCandidates[i].pA;
            MapPoint* pB = vCandidates[i].pB;
            const std::pair<unsigned long, unsigned long> pairId(pA->mnId, pB->mnId);
            if(sSelectedPairIds.count(pairId) > 0)
                continue;
            if(sSelectedPoints.count(pA) > 0 && sSelectedPoints.count(pB) > 0)
                continue;
            if(degreeLimit > 0 &&
               (getPointDegree(pA) >= degreeLimit || getPointDegree(pB) >= degreeLimit))
            {
                ++nRejectedByDegree;
                continue;
            }

            vPairs.push_back(std::make_pair(pA, pB));
            sSelectedPairIds.insert(pairId);
            vSelectedQualities.push_back(vCandidates[i].quality);
            vSelectedDistances.push_back(vCandidates[i].spatialDistance);
            ++mPointDegrees[pA];
            ++mPointDegrees[pB];
            sSelectedPoints.insert(pA);
            sSelectedPoints.insert(pB);
        }
    }

    if(!preferSpatialCoverage)
    {
        std::sort(vCandidates.begin(), vCandidates.end(),
                  [](const PairCandidate& lhs, const PairCandidate& rhs)
                  {
                      if(lhs.quality != rhs.quality)
                          return lhs.quality > rhs.quality;
                      if(lhs.knownObservations != rhs.knownObservations)
                          return lhs.knownObservations > rhs.knownObservations;
                      if(lhs.spatialDistance != rhs.spatialDistance)
                          return lhs.spatialDistance > rhs.spatialDistance;
                      if(lhs.pA->mnId != rhs.pA->mnId)
                          return lhs.pA->mnId < rhs.pA->mnId;
                      return lhs.pB->mnId < rhs.pB->mnId;
                  });
    }

    for(size_t i = 0; i < vCandidates.size() && vPairs.size() < maxPairs; ++i)
    {
        MapPoint* pA = vCandidates[i].pA;
        MapPoint* pB = vCandidates[i].pB;
        const std::pair<unsigned long, unsigned long> pairId(pA->mnId, pB->mnId);
        if(sSelectedPairIds.count(pairId) > 0)
            continue;
        if(preferSpatialCoverage && getPointDegree(pA) > 0 && getPointDegree(pB) > 0)
        {
            const size_t maxDegreeGap =
                degreeLimit > 0 ? degreeLimit : std::numeric_limits<size_t>::max();
            if(getPointDegree(pA) >= maxDegreeGap || getPointDegree(pB) >= maxDegreeGap)
            {
                ++nRejectedByDegree;
                continue;
            }
        }
        if(degreeLimit > 0 &&
           (getPointDegree(pA) >= degreeLimit || getPointDegree(pB) >= degreeLimit))
        {
            ++nRejectedByDegree;
            continue;
        }

        vPairs.push_back(std::make_pair(pA, pB));
        sSelectedPairIds.insert(pairId);
        vSelectedQualities.push_back(vCandidates[i].quality);
        vSelectedDistances.push_back(vCandidates[i].spatialDistance);
        ++mPointDegrees[pA];
        ++mPointDegrees[pB];
        sSelectedPoints.insert(pA);
        sSelectedPoints.insert(pB);
    }

    if(vPairs.empty())
    {
        if(std::getenv("STSLAM_DEBUG_DYNAMIC_LBA"))
        {
            std::cerr << "[STSLAM_DYNAMIC_LBA] rigidity_quality_sampler_fallback"
                      << " prev_frame_id=" << prevFrameId
                      << " curr_frame_id=" << currFrameId
                      << " points=" << vPoints.size()
                      << " total_pairs=" << totalPairs
                      << " candidates=" << vCandidates.size()
                      << " rejected_known_obs=" << nRejectedByKnownObs
                      << " rejected_quality=" << nRejectedByQuality
                      << " rejected_degree=" << nRejectedByDegree
                      << " min_known_obs=" << minKnownObservations
                      << " min_quality=" << minPairQuality
                      << std::endl;
        }
        return SamplePairsInFrame(vPoints, maxPairs);
    }

    if(std::getenv("STSLAM_DEBUG_DYNAMIC_LBA"))
    {
        const double pointCoverageRatio =
            vPoints.empty() ? 0.0 :
            static_cast<double>(sSelectedPoints.size()) / static_cast<double>(vPoints.size());
        size_t minDegree = std::numeric_limits<size_t>::max();
        size_t maxDegree = 0;
        double meanDegree = 0.0;
        for(std::map<MapPoint*, size_t>::const_iterator it = mPointDegrees.begin();
            it != mPointDegrees.end(); ++it)
        {
            minDegree = std::min(minDegree, it->second);
            maxDegree = std::max(maxDegree, it->second);
            meanDegree += static_cast<double>(it->second);
        }
        if(mPointDegrees.empty())
            minDegree = 0;
        else
            meanDegree /= static_cast<double>(mPointDegrees.size());

        std::vector<double> vSortedSelectedQualities = vSelectedQualities;
        std::vector<double> vSortedSelectedDistances = vSelectedDistances;
        std::vector<double> vCandidateQualities;
        vCandidateQualities.reserve(vCandidates.size());
        for(size_t i = 0; i < vCandidates.size(); ++i)
            vCandidateQualities.push_back(vCandidates[i].quality);
        std::sort(vSortedSelectedQualities.begin(), vSortedSelectedQualities.end());
        std::sort(vSortedSelectedDistances.begin(), vSortedSelectedDistances.end());
        std::sort(vCandidateQualities.begin(), vCandidateQualities.end());
        const size_t midQuality =
            vSortedSelectedQualities.empty() ? 0 : vSortedSelectedQualities.size() / 2;
        const size_t midDistance =
            vSortedSelectedDistances.empty() ? 0 : vSortedSelectedDistances.size() / 2;
        const size_t midCandidateQuality =
            vCandidateQualities.empty() ? 0 : vCandidateQualities.size() / 2;
        const double minCandidateQuality =
            vCandidateQualities.empty() ? 0.0 : vCandidateQualities.front();
        const double medianCandidateQuality =
            vCandidateQualities.empty() ? 0.0 : vCandidateQualities[midCandidateQuality];
        const double maxCandidateQuality =
            vCandidateQualities.empty() ? 0.0 : vCandidateQualities.back();
        const double minSelectedQuality =
            vSortedSelectedQualities.empty() ? 0.0 : vSortedSelectedQualities.front();
        const double medianSelectedQuality =
            vSortedSelectedQualities.empty() ? 0.0 : vSortedSelectedQualities[midQuality];
        const double maxSelectedQuality =
            vSortedSelectedQualities.empty() ? 0.0 : vSortedSelectedQualities.back();
        const double minSelectedDistance =
            vSortedSelectedDistances.empty() ? 0.0 : vSortedSelectedDistances.front();
        const double medianSelectedDistance =
            vSortedSelectedDistances.empty() ? 0.0 : vSortedSelectedDistances[midDistance];
        const double maxSelectedDistance =
            vSortedSelectedDistances.empty() ? 0.0 : vSortedSelectedDistances.back();

        std::cerr << "[STSLAM_DYNAMIC_LBA] rigidity_quality_sampler"
                  << " prev_frame_id=" << prevFrameId
                  << " curr_frame_id=" << currFrameId
                  << " mode=" << (preferSpatialCoverage ? "quality_coverage" : "quality_gated")
                  << " points=" << vPoints.size()
                  << " total_pairs=" << totalPairs
                  << " candidates=" << vCandidates.size()
                  << " selected=" << vPairs.size()
                  << " selected_unique_points=" << sSelectedPoints.size()
                  << " selected_point_coverage=" << pointCoverageRatio
                  << " rejected_known_obs=" << nRejectedByKnownObs
                  << " rejected_quality=" << nRejectedByQuality
                  << " rejected_degree=" << nRejectedByDegree
                  << " degree_limit=" << degreeLimit
                  << " selected_degree_min=" << minDegree
                  << " selected_degree_mean=" << meanDegree
                  << " selected_degree_max=" << maxDegree
                  << " min_known_obs=" << minKnownObservations
                  << " min_quality=" << minPairQuality
                  << " candidate_quality_min=" << minCandidateQuality
                  << " candidate_quality_median=" << medianCandidateQuality
                  << " candidate_quality_max=" << maxCandidateQuality
                  << " selected_quality_min=" << minSelectedQuality
                  << " selected_quality_median=" << medianSelectedQuality
                  << " selected_quality_max=" << maxSelectedQuality
                  << " selected_distance_min=" << minSelectedDistance
                  << " selected_distance_median=" << medianSelectedDistance
                  << " selected_distance_max=" << maxSelectedDistance
                  << std::endl;
    }

    return vPairs;
}

void MarkDynamicBackendOutlier(Map* pMap, MapPoint* pMP, unsigned long frameId)
{
    if(!pMap || !pMP || pMP->GetInstanceId() <= 0)
        return;

    Instance* pInstance = pMap->GetInstance(pMP->GetInstanceId());
    if(pInstance)
        pInstance->MarkBackendOutlier(frameId, pMP);
}

void ClearDynamicBackendOutlier(Map* pMap, MapPoint* pMP, unsigned long frameId)
{
    if(!pMap || !pMP || pMP->GetInstanceId() <= 0)
        return;

    Instance* pInstance = pMap->GetInstance(pMP->GetInstanceId());
    if(pInstance)
        pInstance->ClearBackendOutlier(frameId, pMP);
}

bool LocalWindowHasThingPoints(KeyFrame* pKF, Map* pCurrentMap)
{
    if(!pKF || !pCurrentMap)
        return false;

    auto frameHasThingPoint = [&](const std::vector<MapPoint*>& vpMPs) -> bool
    {
        for(size_t i = 0; i < vpMPs.size(); ++i)
        {
            MapPoint* pMP = vpMPs[i];
            if(!pMP || pMP->isBad() || pMP->GetMap() != pCurrentMap)
                continue;
            if(pMP->GetInstanceId() > 0)
                return true;
        }
        return false;
    };

    if(UseStrictEq16ImageWindowForLBA())
    {
        if(frameHasThingPoint(pKF->GetMapPointMatches()))
            return true;

        KeyFrame* pPreviousKeyFrame = pKF->mPrevKF;
        if(pPreviousKeyFrame &&
           !pPreviousKeyFrame->isBad() &&
           pPreviousKeyFrame->GetMap() == pCurrentMap &&
           frameHasThingPoint(pPreviousKeyFrame->GetMapPointMatches()))
        {
            return true;
        }

        std::vector<WindowFrameSnapshot>& vCurrentIntervalFrames = pKF->GetMutableImageFrameWindow();
        for(size_t frameIdx = 0; frameIdx < vCurrentIntervalFrames.size(); ++frameIdx)
        {
            WindowFrameSnapshot& frame = vCurrentIntervalFrames[frameIdx];
            if(!frame.HasPose())
                continue;
            if(frameHasThingPoint(frame.mvpMapPoints))
                return true;
        }

        return false;
    }

    std::set<KeyFrame*> sLocalKeyFrames;
    sLocalKeyFrames.insert(pKF);

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(size_t i = 0; i < vNeighKFs.size(); ++i)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        if(!pKFi || pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
            continue;
        sLocalKeyFrames.insert(pKFi);
    }

    for(std::set<KeyFrame*>::const_iterator sit = sLocalKeyFrames.begin(); sit != sLocalKeyFrames.end(); ++sit)
    {
        if(frameHasThingPoint((*sit)->GetMapPointMatches()))
            return true;
    }

    return false;
}

bool EnableBundleAdjustmentDebugProbe()
{
    const char* envValue = std::getenv("STSLAM_DEBUG_BA_PROBE");
    return envValue && std::string(envValue) != "0";
}

void PrintPoseVertexDebugInfo(const char* stage, g2o::VertexSE3Expmap* vSE3)
{
    if(!vSE3)
    {
        std::cerr << "[STSLAM_DEBUG] " << stage << " null pose vertex" << std::endl;
        return;
    }

    const g2o::SE3Quat estimate = vSE3->estimate();
    const Eigen::Vector3d translation = estimate.translation();
    const Eigen::Quaterniond rotation = estimate.rotation();

    std::cerr << "[STSLAM_DEBUG] " << stage
              << " id=" << vSE3->id()
              << " ptr=" << static_cast<const void*>(vSE3)
              << " fixed=" << vSE3->fixed()
              << " t=" << translation.transpose()
              << " q=" << rotation.coeffs().transpose()
              << std::endl;
}

Eigen::Matrix<double,3,1> ResolvePoseOnlyWorldPoint(Frame* pFrame,
                                                    MapPoint* pMP,
                                                    const size_t idx,
                                                    const bool usePanopticPrediction)
{
    const Eigen::Vector3f basePoint = pMP->GetWorldPos();
    if(!usePanopticPrediction || !pFrame)
        return basePoint.cast<double>();

    int instanceId = pMP->GetInstanceId();
    if(instanceId <= 0)
        instanceId = pFrame->GetFeatureInstanceId(idx);

    if(instanceId <= 0)
        return basePoint.cast<double>();

    const auto motionIt = pFrame->mmPredictedInstanceMotions.find(instanceId);
    if(motionIt == pFrame->mmPredictedInstanceMotions.end())
        return basePoint.cast<double>();

    const Eigen::Vector3f predictedPoint = motionIt->second * basePoint;
    if(!predictedPoint.allFinite())
        return basePoint.cast<double>();

    if(UsePanopticPredictionResidualGate() &&
       idx < pFrame->mvKeysUn.size() &&
       pFrame->mpCamera)
    {
        const Sophus::SE3f Tcw = pFrame->GetPose();
        const Eigen::Vector3f baseCam = Tcw * basePoint;
        const Eigen::Vector3f predictedCam = Tcw * predictedPoint;
        if(baseCam[2] > 0.0f && predictedCam[2] > 0.0f)
        {
            const Eigen::Vector2d obs(pFrame->mvKeysUn[idx].pt.x,
                                      pFrame->mvKeysUn[idx].pt.y);
            const Eigen::Vector3d baseCamD = baseCam.cast<double>();
            const Eigen::Vector3d predictedCamD = predictedCam.cast<double>();
            const Eigen::Vector2d baseProjection =
                pFrame->mpCamera->project(baseCamD);
            const Eigen::Vector2d predictedProjection =
                pFrame->mpCamera->project(predictedCamD);
            const double baseError = (obs - baseProjection).squaredNorm();
            const double predictedError = (obs - predictedProjection).squaredNorm();
            if(predictedError >= baseError)
                return basePoint.cast<double>();
        }
    }

    return predictedPoint.cast<double>();
}

} // namespace

void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
    BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag, nLoopKF, bRobust);
}


void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    Map* pMap = vpKFs[0]->GetMap();

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    const int nExpectedSize = (vpKFs.size())*vpMP.size();

    vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody*> vpEdgesBody;
    vpEdgesBody.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFBody;
    vpEdgeKFBody.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeBody;
    vpMapPointEdgeBody.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);


    // Set KeyFrame vertices

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKF->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
        vSE3->setId(pKF->mnId);
        vSE3->setFixed(pKF->mnId==pMap->GetInitKFid());
        optimizer.addVertex(vSE3);
        if(pKF->mnId>maxKFid)
            maxKFid=pKF->mnId;
    }

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices
    for(size_t i=0; i<vpMP.size(); i++)
    {
        MapPoint* pMP = vpMP[i];
        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        const int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

       const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

        int nEdges = 0;
        //SET EDGES
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {
            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid)
                continue;
            if(optimizer.vertex(id) == NULL || optimizer.vertex(pKF->mnId) == NULL)
                continue;
            nEdges++;

            const int leftIndex = get<0>(mit->second);

            if(leftIndex != -1 && pKF->mvuRight[get<0>(mit->second)]<0)
            {
                const cv::KeyPoint &kpUn = pKF->mvKeysUn[leftIndex];

                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                e->pCamera = pKF->mpCamera;

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vpEdgeKFMono.push_back(pKF);
                vpMapPointEdgeMono.push_back(pMP);
            }
            else if(leftIndex != -1 && pKF->mvuRight[leftIndex] >= 0) //Stereo observation
            {
                const cv::KeyPoint &kpUn = pKF->mvKeysUn[leftIndex];

                Eigen::Matrix<double,3,1> obs;
                const float kp_ur = pKF->mvuRight[get<0>(mit->second)];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vpEdgeKFStereo.push_back(pKF);
                vpMapPointEdgeStereo.push_back(pMP);
            }

            if(pKF->mpCamera2){
                int rightIndex = get<1>(mit->second);

                if(rightIndex != -1 && rightIndex < pKF->mvKeysRight.size()){
                    rightIndex -= pKF->NLeft;

                    Eigen::Matrix<double,2,1> obs;
                    cv::KeyPoint kp = pKF->mvKeysRight[rightIndex];
                    obs << kp.pt.x, kp.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKF->mvInvLevelSigma2[kp.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);

                    Sophus::SE3f Trl = pKF-> GetRelativePoseTrl();
                    e->mTrl = g2o::SE3Quat(Trl.unit_quaternion().cast<double>(), Trl.translation().cast<double>());

                    e->pCamera = pKF->mpCamera2;

                    optimizer.addEdge(e);
                    vpEdgesBody.push_back(e);
                    vpEdgeKFBody.push_back(pKF);
                    vpMapPointEdgeBody.push_back(pMP);
                }
            }
        }



        if(nEdges==0)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
        else
        {
            vbNotIncludedMP[i]=false;
        }
    }

    // Optimize!
    optimizer.setVerbose(false);
    const bool debugBAProbe = EnableBundleAdjustmentDebugProbe();
    if(debugBAProbe)
    {
        std::cerr << "[STSLAM_DEBUG] BA probe: vertices before initializeOptimization="
                  << optimizer.vertices().size() << std::endl;
        for(size_t i = 0; i < vpKFs.size(); ++i)
        {
            KeyFrame* pKF = vpKFs[i];
            if(!pKF || pKF->isBad())
                continue;

            g2o::VertexSE3Expmap* vSE3 =
                static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
            PrintPoseVertexDebugInfo("pre-init", vSE3);
            vSE3->push();
            vSE3->pop();
            std::cerr << "[STSLAM_DEBUG] pre-init push/pop ok for id=" << pKF->mnId << std::endl;
        }
    }
    optimizer.initializeOptimization();
    if(debugBAProbe)
    {
        const g2o::SparseOptimizer::VertexContainer& vActiveVertices = optimizer.activeVertices();
        std::cerr << "[STSLAM_DEBUG] BA probe: active vertices after initializeOptimization="
                  << vActiveVertices.size() << std::endl;
        for(size_t i = 0; i < vActiveVertices.size(); ++i)
        {
            g2o::OptimizableGraph::Vertex* pVertex = vActiveVertices[i];
            if(!pVertex)
            {
                std::cerr << "[STSLAM_DEBUG] active vertex[" << i << "] is null" << std::endl;
                continue;
            }

            std::cerr << "[STSLAM_DEBUG] active vertex[" << i << "] id=" << pVertex->id()
                      << " dim=" << pVertex->dimension()
                      << " fixed=" << pVertex->fixed()
                      << " ptr=" << static_cast<const void*>(pVertex)
                      << std::endl;

            if(pVertex->dimension() == 6)
            {
                g2o::VertexSE3Expmap* vSE3 = dynamic_cast<g2o::VertexSE3Expmap*>(pVertex);
                PrintPoseVertexDebugInfo("post-init", vSE3);
                if(vSE3)
                {
                    vSE3->push();
                    vSE3->pop();
                    std::cerr << "[STSLAM_DEBUG] post-init push/pop ok for id=" << vSE3->id() << std::endl;
                }
            }
        }
        std::cerr << "[STSLAM_DEBUG] BA probe: entering optimizer.optimize(" << nIterations << ")"
                  << std::endl;
    }
    optimizer.optimize(nIterations);
    Verbose::PrintMess("BA: End of the optimization", Verbose::VERBOSITY_NORMAL);

    // Recover optimized data
    //Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));

        g2o::SE3Quat SE3quat = vSE3->estimate();
        if(nLoopKF==pMap->GetOriginKF()->mnId)
        {
            pKF->SetPose(Sophus::SE3f(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>()));
        }
        else
        {
            pKF->mTcwGBA = Sophus::SE3d(SE3quat.rotation(),SE3quat.translation()).cast<float>();
            pKF->mnBAGlobalForKF = nLoopKF;

            Sophus::SE3f mTwc = pKF->GetPoseInverse();
            Sophus::SE3f mTcGBA_c = pKF->mTcwGBA * mTwc;
            Eigen::Vector3f vector_dist =  mTcGBA_c.translation();
            double dist = vector_dist.norm();
            if(dist > 1)
            {
                int numMonoBadPoints = 0, numMonoOptPoints = 0;
                int numStereoBadPoints = 0, numStereoOptPoints = 0;
                vector<MapPoint*> vpMonoMPsOpt, vpStereoMPsOpt;

                for(size_t i2=0, iend=vpEdgesMono.size(); i2<iend;i2++)
                {
                    ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i2];
                    MapPoint* pMP = vpMapPointEdgeMono[i2];
                    KeyFrame* pKFedge = vpEdgeKFMono[i2];

                    if(pKF != pKFedge)
                    {
                        continue;
                    }

                    if(pMP->isBad())
                        continue;

                    if(e->chi2()>5.991 || !e->isDepthPositive())
                    {
                        numMonoBadPoints++;

                    }
                    else
                    {
                        numMonoOptPoints++;
                        vpMonoMPsOpt.push_back(pMP);
                    }

                }

                for(size_t i2=0, iend=vpEdgesStereo.size(); i2<iend;i2++)
                {
                    g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i2];
                    MapPoint* pMP = vpMapPointEdgeStereo[i2];
                    KeyFrame* pKFedge = vpEdgeKFMono[i2];

                    if(pKF != pKFedge)
                    {
                        continue;
                    }

                    if(pMP->isBad())
                        continue;

                    if(e->chi2()>7.815 || !e->isDepthPositive())
                    {
                        numStereoBadPoints++;
                    }
                    else
                    {
                        numStereoOptPoints++;
                        vpStereoMPsOpt.push_back(pMP);
                    }
                }
            }
        }
    }

    //Points
    for(size_t i=0; i<vpMP.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMP[i];

        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));

        if(nLoopKF==pMap->GetOriginKF()->mnId)
        {
            pMP->SetWorldPos(vPoint->estimate().cast<float>());
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            pMP->mPosGBA = vPoint->estimate().cast<float>();
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }
}

void Optimizer::FullInertialBA(Map *pMap, int its, const bool bFixLocal, const long unsigned int nLoopId, bool *pbStopFlag, bool bInit, float priorG, float priorA, Eigen::VectorXd *vSingVal, bool *bHess)
{
    long unsigned int maxKFid = pMap->GetMaxKFid();
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    solver->setUserLambdaInit(1e-5);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    int nNonFixed = 0;

    // Set KeyFrame vertices
    KeyFrame* pIncKF;
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        pIncKF=pKFi;
        bool bFixed = false;
        if(bFixLocal)
        {
            bFixed = (pKFi->mnBALocalForKF>=(maxKFid-1)) || (pKFi->mnBAFixedForKF>=(maxKFid-1));
            if(!bFixed)
                nNonFixed++;
            VP->setFixed(bFixed);
        }
        optimizer.addVertex(VP);

        if(pKFi->bImu)
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+3*(pKFi->mnId)+1);
            VV->setFixed(bFixed);
            optimizer.addVertex(VV);
            if (!bInit)
            {
                VertexGyroBias* VG = new VertexGyroBias(pKFi);
                VG->setId(maxKFid+3*(pKFi->mnId)+2);
                VG->setFixed(bFixed);
                optimizer.addVertex(VG);
                VertexAccBias* VA = new VertexAccBias(pKFi);
                VA->setId(maxKFid+3*(pKFi->mnId)+3);
                VA->setFixed(bFixed);
                optimizer.addVertex(VA);
            }
        }
    }

    if (bInit)
    {
        VertexGyroBias* VG = new VertexGyroBias(pIncKF);
        VG->setId(4*maxKFid+2);
        VG->setFixed(false);
        optimizer.addVertex(VG);
        VertexAccBias* VA = new VertexAccBias(pIncKF);
        VA->setId(4*maxKFid+3);
        VA->setFixed(false);
        optimizer.addVertex(VA);
    }

    if(bFixLocal)
    {
        if(nNonFixed<3)
            return;
    }

    // IMU links
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        if(!pKFi->mPrevKF)
        {
            Verbose::PrintMess("NOT INERTIAL LINK TO PREVIOUS FRAME!", Verbose::VERBOSITY_NORMAL);
            continue;
        }

        if(pKFi->mPrevKF && pKFi->mnId<=maxKFid)
        {
            if(pKFi->isBad() || pKFi->mPrevKF->mnId>maxKFid)
                continue;
            if(pKFi->bImu && pKFi->mPrevKF->bImu)
            {
                pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
                g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
                g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+1);

                g2o::HyperGraph::Vertex* VG1;
                g2o::HyperGraph::Vertex* VA1;
                g2o::HyperGraph::Vertex* VG2;
                g2o::HyperGraph::Vertex* VA2;
                if (!bInit)
                {
                    VG1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+2);
                    VA1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+3);
                    VG2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+2);
                    VA2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+3);
                }
                else
                {
                    VG1 = optimizer.vertex(4*maxKFid+2);
                    VA1 = optimizer.vertex(4*maxKFid+3);
                }

                g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
                g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+1);

                if (!bInit)
                {
                    if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
                    {
                        cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;
                        continue;
                    }
                }
                else
                {
                    if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2)
                    {
                        cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<endl;
                        continue;
                    }
                }

                EdgeInertial* ei = new EdgeInertial(pKFi->mpImuPreintegrated);
                ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
                ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
                ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
                ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
                ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
                ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

                g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
                ei->setRobustKernel(rki);
                rki->setDelta(sqrt(16.92));

                optimizer.addEdge(ei);

                if (!bInit)
                {
                    EdgeGyroRW* egr= new EdgeGyroRW();
                    egr->setVertex(0,VG1);
                    egr->setVertex(1,VG2);
                    Eigen::Matrix3d InfoG = pKFi->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse();
                    egr->setInformation(InfoG);
                    egr->computeError();
                    optimizer.addEdge(egr);

                    EdgeAccRW* ear = new EdgeAccRW();
                    ear->setVertex(0,VA1);
                    ear->setVertex(1,VA2);
                    Eigen::Matrix3d InfoA = pKFi->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse();
                    ear->setInformation(InfoA);
                    ear->computeError();
                    optimizer.addEdge(ear);
                }
            }
            else
                cout << pKFi->mnId << " or " << pKFi->mPrevKF->mnId << " no imu" << endl;
        }
    }

    if (bInit)
    {
        g2o::HyperGraph::Vertex* VG = optimizer.vertex(4*maxKFid+2);
        g2o::HyperGraph::Vertex* VA = optimizer.vertex(4*maxKFid+3);

        // Add prior to comon biases
        Eigen::Vector3f bprior;
        bprior.setZero();

        EdgePriorAcc* epa = new EdgePriorAcc(bprior);
        epa->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
        double infoPriorA = priorA; //
        epa->setInformation(infoPriorA*Eigen::Matrix3d::Identity());
        optimizer.addEdge(epa);

        EdgePriorGyro* epg = new EdgePriorGyro(bprior);
        epg->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
        double infoPriorG = priorG; //
        epg->setInformation(infoPriorG*Eigen::Matrix3d::Identity());
        optimizer.addEdge(epg);
    }

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    const unsigned long iniMPid = maxKFid*5;

    vector<bool> vbNotIncludedMP(vpMPs.size(),false);

    for(size_t i=0; i<vpMPs.size(); i++)
    {
        MapPoint* pMP = vpMPs[i];
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        unsigned long id = pMP->mnId+iniMPid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();


        bool bAllFixed = true;

        //Set edges
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnId>maxKFid)
                continue;

            if(!pKFi->isBad())
            {
                const int leftIndex = get<0>(mit->second);
                cv::KeyPoint kpUn;

                if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]<0) // Monocular observation
                {
                    kpUn = pKFi->mvKeysUn[leftIndex];
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMono* e = new EdgeMono(0);

                    g2o::OptimizableGraph::Vertex* VP = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId));
                    if(bAllFixed)
                        if(!VP->fixed())
                            bAllFixed=false;

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, VP);
                    e->setMeasurement(obs);
                    const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];

                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);
                }
                else if(leftIndex != -1 && pKFi->mvuRight[leftIndex] >= 0) // stereo observation
                {
                    kpUn = pKFi->mvKeysUn[leftIndex];
                    const float kp_ur = pKFi->mvuRight[leftIndex];
                    Eigen::Matrix<double,3,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    EdgeStereo* e = new EdgeStereo(0);

                    g2o::OptimizableGraph::Vertex* VP = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId));
                    if(bAllFixed)
                        if(!VP->fixed())
                            bAllFixed=false;

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, VP);
                    e->setMeasurement(obs);
                    const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];

                    e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);
                }

                if(pKFi->mpCamera2){ // Monocular right observation
                    int rightIndex = get<1>(mit->second);

                    if(rightIndex != -1 && rightIndex < pKFi->mvKeysRight.size()){
                        rightIndex -= pKFi->NLeft;

                        Eigen::Matrix<double,2,1> obs;
                        kpUn = pKFi->mvKeysRight[rightIndex];
                        obs << kpUn.pt.x, kpUn.pt.y;

                        EdgeMono *e = new EdgeMono(1);

                        g2o::OptimizableGraph::Vertex* VP = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId));
                        if(bAllFixed)
                            if(!VP->fixed())
                                bAllFixed=false;

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, VP);
                        e->setMeasurement(obs);
                        const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);
                    }
                }
            }
        }

        if(bAllFixed)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
    }

    if(pbStopFlag)
        if(*pbStopFlag)
            return;


    optimizer.initializeOptimization();
    optimizer.optimize(its);


    // Recover optimized data
    //Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;
        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        if(nLoopId==0)
        {
            Sophus::SE3f Tcw(VP->estimate().Rcw[0].cast<float>(), VP->estimate().tcw[0].cast<float>());
            pKFi->SetPose(Tcw);
        }
        else
        {
            pKFi->mTcwGBA = Sophus::SE3f(VP->estimate().Rcw[0].cast<float>(),VP->estimate().tcw[0].cast<float>());
            pKFi->mnBAGlobalForKF = nLoopId;

        }
        if(pKFi->bImu)
        {
            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+1));
            if(nLoopId==0)
            {
                pKFi->SetVelocity(VV->estimate().cast<float>());
            }
            else
            {
                pKFi->mVwbGBA = VV->estimate().cast<float>();
            }

            VertexGyroBias* VG;
            VertexAccBias* VA;
            if (!bInit)
            {
                VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+2));
                VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+3));
            }
            else
            {
                VG = static_cast<VertexGyroBias*>(optimizer.vertex(4*maxKFid+2));
                VA = static_cast<VertexAccBias*>(optimizer.vertex(4*maxKFid+3));
            }

            Vector6d vb;
            vb << VG->estimate(), VA->estimate();
            IMU::Bias b (vb[3],vb[4],vb[5],vb[0],vb[1],vb[2]);
            if(nLoopId==0)
            {
                pKFi->SetNewBias(b);
            }
            else
            {
                pKFi->mBiasGBA = b;
            }
        }
    }

    //Points
    for(size_t i=0; i<vpMPs.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMPs[i];
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+iniMPid+1));

        if(nLoopId==0)
        {
            pMP->SetWorldPos(vPoint->estimate().cast<float>());
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            pMP->mPosGBA = vPoint->estimate().cast<float>();
            pMP->mnBAGlobalForKF = nLoopId;
        }

    }

    pMap->IncreaseChangeIndex();
}


int Optimizer::PoseOptimization(Frame *pFrame)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    // Set Frame vertex
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    Sophus::SE3<float> Tcw = pFrame->GetPose();
    vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;

    vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *> vpEdgesMono_FHR;
    vector<size_t> vnIndexEdgeMono, vnIndexEdgeRight;
    vpEdgesMono.reserve(N);
    vpEdgesMono_FHR.reserve(N);
    vnIndexEdgeMono.reserve(N);
    vnIndexEdgeRight.reserve(N);

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);

    {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        if(pMP)
        {
            //Conventional SLAM
            if(!pFrame->mpCamera2){
                // Monocular observation
                if(pFrame->mvuRight[i]<0)
                {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double,2,1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    obs << kpUn.pt.x, kpUn.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose* e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->pCamera = pFrame->mpCamera;
                    e->Xw = pMP->GetWorldPos().cast<double>();

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                else  // Stereo observation
                {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double,3,1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    const float &kp_ur = pFrame->mvuRight[i];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaStereo);

                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;
                    e->bf = pFrame->mbf;
                    e->Xw = pMP->GetWorldPos().cast<double>();

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }
            }
            //SLAM with respect a rigid body
            else{
                nInitialCorrespondences++;

                cv::KeyPoint kpUn;

                if (i < pFrame->Nleft) {    //Left camera observation
                    kpUn = pFrame->mvKeys[i];

                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose *e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->pCamera = pFrame->mpCamera;
                    e->Xw = pMP->GetWorldPos().cast<double>();

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                else {
                    kpUn = pFrame->mvKeysRight[i - pFrame->Nleft];

                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    pFrame->mvbOutlier[i] = false;

                    ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->pCamera = pFrame->mpCamera2;
                    e->Xw = pMP->GetWorldPos().cast<double>();

                    e->mTrl = g2o::SE3Quat(pFrame->GetRelativePoseTrl().unit_quaternion().cast<double>(), pFrame->GetRelativePoseTrl().translation().cast<double>());

                    optimizer.addEdge(e);

                    vpEdgesMono_FHR.push_back(e);
                    vnIndexEdgeRight.push_back(i);
                }
            }
        }
    }
    }

    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};    

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {
        Tcw = pFrame->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));

        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {                
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesMono_FHR.size(); i<iend; i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody* e = vpEdgesMono_FHR[i];

            const size_t idx = vnIndexEdgeRight[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {                
                e->setLevel(0);
                pFrame->mvbOutlier[idx]=false;
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        if(optimizer.edges().size()<10)
            break;
    }    

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    Sophus::SE3<float> pose(SE3quat_recov.rotation().cast<float>(),
            SE3quat_recov.translation().cast<float>());
    pFrame->SetPose(pose);

    return nInitialCorrespondences-nBad;
}

int Optimizer::PoseOptimizationPanoptic(Frame *pFrame)
{
    if(!pFrame || pFrame->mmPredictedInstanceMotions.empty())
        return PoseOptimization(pFrame);

    bool hasNonZeroInstanceMotion = false;
    for(std::map<int, Sophus::SE3f>::const_iterator itMotion =
            pFrame->mmPredictedInstanceMotions.begin();
        itMotion != pFrame->mmPredictedInstanceMotions.end(); ++itMotion)
    {
        if(!IsNearlyZeroInstanceMotion(itMotion->second))
        {
            hasNonZeroInstanceMotion = true;
            break;
        }
    }
    if(!hasNonZeroInstanceMotion)
        return PoseOptimization(pFrame);

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    Sophus::SE3<float> Tcw = pFrame->GetPose();
    vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    const int N = pFrame->N;

    vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *> vpEdgesMono_FHR;
    vector<size_t> vnIndexEdgeMono, vnIndexEdgeRight;
    vpEdgesMono.reserve(N);
    vpEdgesMono_FHR.reserve(N);
    vnIndexEdgeMono.reserve(N);
    vnIndexEdgeRight.reserve(N);

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);

    {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        if(pMP)
        {
            if(!pFrame->mpCamera2){
                if(pFrame->mvuRight[i]<0)
                {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double,2,1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    obs << kpUn.pt.x, kpUn.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose* e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->pCamera = pFrame->mpCamera;
                    e->Xw = ResolvePoseOnlyWorldPoint(pFrame, pMP, i, true);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                else
                {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double,3,1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    const float &kp_ur = pFrame->mvuRight[i];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaStereo);

                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;
                    e->bf = pFrame->mbf;
                    e->Xw = ResolvePoseOnlyWorldPoint(pFrame, pMP, i, true);

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }
            }
            else{
                nInitialCorrespondences++;

                cv::KeyPoint kpUn;

                if (i < pFrame->Nleft) {
                    kpUn = pFrame->mvKeys[i];

                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose *e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->pCamera = pFrame->mpCamera;
                    e->Xw = ResolvePoseOnlyWorldPoint(pFrame, pMP, i, true);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                else {
                    kpUn = pFrame->mvKeysRight[i - pFrame->Nleft];

                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    pFrame->mvbOutlier[i] = false;

                    ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->pCamera = pFrame->mpCamera2;
                    e->Xw = ResolvePoseOnlyWorldPoint(pFrame, pMP, i, true);

                    e->mTrl = g2o::SE3Quat(pFrame->GetRelativePoseTrl().unit_quaternion().cast<double>(), pFrame->GetRelativePoseTrl().translation().cast<double>());

                    optimizer.addEdge(e);

                    vpEdgesMono_FHR.push_back(e);
                    vnIndexEdgeRight.push_back(i);
                }
            }
        }
    }
    }

    if(nInitialCorrespondences<3)
        return 0;

    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {
        Tcw = pFrame->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));

        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesMono_FHR.size(); i<iend; i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody* e = vpEdgesMono_FHR[i];

            const size_t idx = vnIndexEdgeRight[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                e->setLevel(0);
                pFrame->mvbOutlier[idx]=false;
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        if(optimizer.edges().size()<10)
            break;
    }

    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    Sophus::SE3<float> pose(SE3quat_recov.rotation().cast<float>(),
            SE3quat_recov.translation().cast<float>());
    pFrame->SetPose(pose);

    return nInitialCorrespondences-nBad;
}

void RunVanillaLocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap, int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges)
{
    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    Map* pCurrentMap = pKF->GetMap();

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames
    num_fixedKF = 0;
    list<MapPoint*> lLocalMapPoints;
    set<MapPoint*> sNumObsMP;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        if(pKFi->mnId==pMap->GetInitKFid())
        {
            num_fixedKF = 1;
        }
        vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad() && pMP->GetMap() == pCurrentMap)
                {

                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
                }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    list<KeyFrame*> lFixedCameras;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,tuple<int,int>> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId )
            {                
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                    lFixedCameras.push_back(pKFi);
            }
        }
    }
    num_fixedKF = lFixedCameras.size() + num_fixedKF;


    if(num_fixedKF == 0)
    {
        Verbose::PrintMess("LM-LBA: There are 0 fixed KF in the optimizations, LBA aborted", Verbose::VERBOSITY_NORMAL);
        return;
    }

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    if (pMap->IsInertial())
        solver->setUserLambdaInit(100.0);

    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // DEBUG LBA
    pCurrentMap->msOptKFs.clear();
    pCurrentMap->msFixedKFs.clear();

    // Set Local KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId==pMap->GetInitKFid());
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
        // DEBUG LBA
        pCurrentMap->msOptKFs.insert(pKFi->mnId);
    }
    num_OptKF = lLocalKeyFrames.size();

    // Set Fixed KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
        // DEBUG LBA
        pCurrentMap->msFixedKFs.insert(pKFi->mnId);
    }

    // Set MapPoint vertices
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

    vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody*> vpEdgesBody;
    vpEdgesBody.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFBody;
    vpEdgeKFBody.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeBody;
    vpMapPointEdgeBody.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    int nPoints = 0;

    int nEdges = 0;

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        nPoints++;

        const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

        //Set edges
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            {
                const int leftIndex = get<0>(mit->second);

                // Monocular observation
                if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]<0)
                {
                    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->pCamera = pKFi->mpCamera;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);

                    nEdges++;
                }
                else if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]>=0)// Stereo observation
                {
                    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                    Eigen::Matrix<double,3,1> obs;
                    const float kp_ur = pKFi->mvuRight[get<0>(mit->second)];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);

                    nEdges++;
                }

                if(pKFi->mpCamera2){
                    int rightIndex = get<1>(mit->second);

                    if(rightIndex != -1 ){
                        rightIndex -= pKFi->NLeft;

                        Eigen::Matrix<double,2,1> obs;
                        cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
                        obs << kp.pt.x, kp.pt.y;

                        ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kp.octave];
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        Sophus::SE3f Trl = pKFi-> GetRelativePoseTrl();
                        e->mTrl = g2o::SE3Quat(Trl.unit_quaternion().cast<double>(), Trl.translation().cast<double>());

                        e->pCamera = pKFi->mpCamera2;

                        optimizer.addEdge(e);
                        vpEdgesBody.push_back(e);
                        vpEdgeKFBody.push_back(pKFi);
                        vpMapPointEdgeBody.push_back(pMP);

                        nEdges++;
                    }
                }
            }
        }
    }
    num_edges = nEdges;

    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesBody.size()+vpEdgesStereo.size());

    // Check inlier observations       
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    for(size_t i=0, iend=vpEdgesBody.size(); i<iend;i++)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZToBody* e = vpEdgesBody[i];
        MapPoint* pMP = vpMapPointEdgeBody[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFBody[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }


    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data
    //Keyframes
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        Sophus::SE3f Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());
        pKFi->SetPose(Tiw);
    }

    //Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
        pMP->UpdateNormalAndDepth();
    }

    pMap->IncreaseChangeIndex();
}

void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap, int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges)
{
    if(DisableDynamicBackendForLBA() || !LocalWindowHasThingPoints(pKF, pKF ? pKF->GetMap() : NULL))
    {
        RunVanillaLocalBundleAdjustment(pKF, pbStopFlag, pMap, num_fixedKF, num_OptKF, num_MPs, num_edges);
        return;
    }

    const bool debugDynamicLBA = EnableDynamicLBADebug();
    const bool disableShapeEdges = DisableDynamicShapeForLBA();
    const bool disableRigidityEdges = DisableDynamicRigidityForLBA();
    const bool enableDynamicPointPrior = EnableDynamicPointPriorForLBA();
    const bool enableInstanceStructureProxy = EnableInstanceStructureProxyForLBA();
    const bool enableCentroidMotionFallback = EnableCentroidMotionFallbackForLBA();
    const bool translationOnlyInstanceMotionWriteback =
        UseTranslationOnlyInstanceMotionWritebackForLBA();
    const bool strictEq16ImageWindow = UseStrictEq16ImageWindowForLBA();
    const std::string module8Profile = GetModule8ProfileName();
    const double instanceStructureInvSigma2 = GetInstanceStructureInvSigma2();
    const double panopticFactorWeight = GetDynamicPanopticFactorWeight();
    const double shapeFactorWeight = GetDynamicShapeFactorWeight();
    const double rigidityFactorWeight = GetDynamicRigidityFactorWeight();
    const float backendImmatureMaxTranslation = GetBackendImmatureMotionMaxTranslation();
    const float backendImmatureMaxRotationDeg = GetBackendImmatureMotionMaxRotationDeg();
    const int backendImmatureConfirmFrames = GetBackendImmatureMotionConfirmFrames();
    const bool backendMatureMotionGate = EnableBackendMatureMotionGate();
    const float backendMatureMaxTranslation = GetBackendMatureMotionMaxTranslation();
    const float backendMatureMaxRotationDeg = GetBackendMatureMotionMaxRotationDeg();
    const int backendMotionMaxFrameGap = GetBackendMotionMaxFrameGap();
    const int nStrictEq16Iterations = GetStrictEq16IterationCount();
    const bool enableInstancePoseMotionPrior = EnableInstancePoseMotionPrior();
    const double instancePoseMotionPriorInvSigma2 =
        GetInstancePoseMotionPriorInvSigma2();
    int nShapeEdges = 0;
    int nRigidityEdges = 0;
    int nRigidityPointPairs = 0;
    int nDynamicPriorEdges = 0;
    int nInstancePoseMotionPriorEdges = 0;

    if(debugDynamicLBA)
        std::cerr << "[STSLAM_DYNAMIC_LBA] enter"
                  << " kf=" << (pKF ? pKF->mnId : static_cast<unsigned long>(-1))
                  << " module8_profile=" << module8Profile
                  << " eq16_terms=stuff_reprojection,thing_panoptic_reprojection,shape_scale,rigidity"
                  << " strict_eq16_image_window=" << strictEq16ImageWindow
                  << " strict_eq16_iterations=" << nStrictEq16Iterations
                  << std::endl;

    list<KeyFrame*> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    Map* pCurrentMap = pKF->GetMap();

    if(!strictEq16ImageWindow)
    {
        const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
        for(int i = 0, iend = vNeighKFs.size(); i < iend; ++i)
        {
            KeyFrame* pKFi = vNeighKFs[i];
            pKFi->mnBALocalForKF = pKF->mnId;
            if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                lLocalKeyFrames.push_back(pKFi);
        }
    }
    lLocalKeyFrames.sort(SortKeyFramesByTimestamp);

    num_fixedKF = 0;
    list<MapPoint*> lLocalMapPoints;
    int nCurrentIntervalWindowMapPointsAdded = 0;
    for(list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; ++lit)
    {
        KeyFrame* pKFi = *lit;
        if(pKFi->mnId == pMap->GetInitKFid())
            num_fixedKF = 1;

        vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; ++vit)
        {
            MapPoint* pMP = *vit;
            if(!pMP || pMP->isBad() || pMP->GetMap() != pCurrentMap)
                continue;

            if(pMP->mnBALocalForKF != pKF->mnId)
            {
                lLocalMapPoints.push_back(pMP);
                pMP->mnBALocalForKF = pKF->mnId;
            }
        }
    }

    if(pKF)
    {
        int nCurrentIntervalDynamicBufferMapPointsAdded = 0;
        int nPersistentDynamicBufferMapPointsAdded = 0;
        auto addDynamicBufferMapPoint = [&](MapPoint* pMP) -> bool
        {
            if(!pMP || pMP->isBad() || pMP->GetMap() != pCurrentMap)
                return false;

            if(pMP->mnBALocalForKF != pKF->mnId)
            {
                lLocalMapPoints.push_back(pMP);
                pMP->mnBALocalForKF = pKF->mnId;
                return true;
            }
            return false;
        };

        const std::vector<DynamicInstancePointObservation>& vKeyFrameDynamicObservations =
            pKF->GetDynamicInstancePointObservations();
        for(size_t dynIdx = 0; dynIdx < vKeyFrameDynamicObservations.size(); ++dynIdx)
        {
            if(addDynamicBufferMapPoint(vKeyFrameDynamicObservations[dynIdx].pBackendPoint))
                ++nCurrentIntervalDynamicBufferMapPointsAdded;
        }

        std::vector<WindowFrameSnapshot>& vCurrentIntervalFrames = pKF->GetMutableImageFrameWindow();
        for(size_t frameIdx = 0; frameIdx < vCurrentIntervalFrames.size(); ++frameIdx)
        {
            WindowFrameSnapshot& frame = vCurrentIntervalFrames[frameIdx];
            if(!frame.HasPose())
                continue;

            for(size_t pointIdx = 0; pointIdx < frame.mvpMapPoints.size(); ++pointIdx)
            {
                MapPoint* pMP = frame.mvpMapPoints[pointIdx];
                if(!pMP || pMP->isBad() || pMP->GetMap() != pCurrentMap)
                    continue;

                if(pMP->mnBALocalForKF != pKF->mnId)
                {
                    lLocalMapPoints.push_back(pMP);
                    pMP->mnBALocalForKF = pKF->mnId;
                    ++nCurrentIntervalWindowMapPointsAdded;
                }
            }

            for(size_t dynIdx = 0; dynIdx < frame.mvDynamicInstancePointObservations.size(); ++dynIdx)
            {
                if(addDynamicBufferMapPoint(frame.mvDynamicInstancePointObservations[dynIdx].pBackendPoint))
                    ++nCurrentIntervalDynamicBufferMapPointsAdded;
            }
        }

        std::set<unsigned long> sPersistentDynamicFrameIds;
        sPersistentDynamicFrameIds.insert(pKF->mnFrameId);
        for(size_t frameIdx = 0; frameIdx < vCurrentIntervalFrames.size(); ++frameIdx)
        {
            if(vCurrentIntervalFrames[frameIdx].HasPose())
                sPersistentDynamicFrameIds.insert(vCurrentIntervalFrames[frameIdx].mnFrameId);
        }

        const std::vector<Instance*> vInstances = pCurrentMap->GetAllInstances();
        for(size_t instanceIdx = 0; instanceIdx < vInstances.size(); ++instanceIdx)
        {
            Instance* pInstance = vInstances[instanceIdx];
            if(!pInstance)
                continue;

            for(std::set<unsigned long>::const_iterator itFrameId = sPersistentDynamicFrameIds.begin();
                itFrameId != sPersistentDynamicFrameIds.end(); ++itFrameId)
            {
                const std::vector<Instance::DynamicObservationRecord> vRecords =
                    pInstance->GetDynamicObservationsForFrame(*itFrameId);
                for(size_t recordIdx = 0; recordIdx < vRecords.size(); ++recordIdx)
                {
                    if(addDynamicBufferMapPoint(vRecords[recordIdx].pBackendPoint))
                        ++nPersistentDynamicBufferMapPointsAdded;
                }
            }
        }
        nCurrentIntervalWindowMapPointsAdded += nCurrentIntervalDynamicBufferMapPointsAdded;
        nCurrentIntervalWindowMapPointsAdded += nPersistentDynamicBufferMapPointsAdded;
    }
    lLocalMapPoints.sort(SortMapPointsById);

    list<KeyFrame*> lFixedCameras;
    if(!strictEq16ImageWindow)
    {
        for(list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; ++lit)
        {
            map<KeyFrame*, tuple<int,int> > observations = (*lit)->GetObservations();
            std::vector<std::pair<KeyFrame*, tuple<int,int> > > vObservations(observations.begin(), observations.end());
            std::sort(vObservations.begin(), vObservations.end(),
                      [](const std::pair<KeyFrame*, tuple<int,int> >& lhs,
                         const std::pair<KeyFrame*, tuple<int,int> >& rhs)
                      {
                          return SortKeyFramesByTimestamp(lhs.first, rhs.first);
                      });
            for(size_t obsIdx = 0; obsIdx < vObservations.size(); ++obsIdx)
            {
                KeyFrame* pKFi = vObservations[obsIdx].first;
                if(pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
                {
                    pKFi->mnBAFixedForKF = pKF->mnId;
                    if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                        lFixedCameras.push_back(pKFi);
                }
            }
        }
    }

    KeyFrame* pPreviousKeyFrame = pKF ? pKF->mPrevKF : NULL;
    if(pPreviousKeyFrame &&
       !pPreviousKeyFrame->isBad() &&
       pPreviousKeyFrame->GetMap() == pCurrentMap &&
       pPreviousKeyFrame->mnBALocalForKF != pKF->mnId &&
       pPreviousKeyFrame->mnBAFixedForKF != pKF->mnId)
    {
        pPreviousKeyFrame->mnBAFixedForKF = pKF->mnId;
        lFixedCameras.push_back(pPreviousKeyFrame);
    }
    lFixedCameras.sort(SortKeyFramesByTimestamp);
    num_fixedKF = static_cast<int>(lFixedCameras.size()) + num_fixedKF;

    if(debugDynamicLBA)
    {
        std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
                  << " local_window_ready local_kf=" << lLocalKeyFrames.size()
                  << " fixed_kf=" << lFixedCameras.size()
                  << " local_mp=" << lLocalMapPoints.size()
                  << " current_interval_window_mp_added=" << nCurrentIntervalWindowMapPointsAdded
                  << " strict_eq16_image_window=" << strictEq16ImageWindow
                  << std::endl;
    }

    if(num_fixedKF == 0)
    {
        Verbose::PrintMess("LM-LBA: There are 0 fixed KF in the optimizations, LBA aborted", Verbose::VERBOSITY_NORMAL);
        return;
    }

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType* linearSolver =
        new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX* solver_ptr = new g2o::BlockSolverX(linearSolver);
    solver_ptr->setSchur(false);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    if(pMap->IsInertial())
        solver->setUserLambdaInit(100.0);

    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;
    pCurrentMap->msOptKFs.clear();
    pCurrentMap->msFixedKFs.clear();

    std::set<KeyFrame*> sWindowKeyFrames;
    std::map<unsigned long, ImageFrameHandle> mImageFrames;
    int nWindowFrameVertices = 0;
    int nFixedWindowFrameVertices = 0;
    int nTauImageFrames = 0;

    for(list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; ++lit)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId == pMap->GetInitKFid());
        optimizer.addVertex(vSE3);
        if(pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
        pCurrentMap->msOptKFs.insert(pKFi->mnId);
        sWindowKeyFrames.insert(pKFi);

        ImageFrameHandle handle;
        handle.frameId = pKFi->mnFrameId;
        handle.timeStamp = pKFi->mTimeStamp;
        handle.poseVertexId = static_cast<int>(pKFi->mnId);
        handle.isKeyFrame = true;
        handle.fixed = (pKFi->mnId == pMap->GetInitKFid());
        handle.inCurrentInterval = (pKFi == pKF || pKFi == pPreviousKeyFrame);
        handle.pKeyFrame = pKFi;
        mImageFrames[handle.frameId] = handle;
    }
    num_OptKF = static_cast<int>(lLocalKeyFrames.size());

    for(list<KeyFrame*>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; ++lit)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
        pCurrentMap->msFixedKFs.insert(pKFi->mnId);
        sWindowKeyFrames.insert(pKFi);

        ImageFrameHandle handle;
        handle.frameId = pKFi->mnFrameId;
        handle.timeStamp = pKFi->mTimeStamp;
        handle.poseVertexId = static_cast<int>(pKFi->mnId);
        handle.isKeyFrame = true;
        handle.fixed = true;
        handle.inCurrentInterval = (pKFi == pKF || pKFi == pPreviousKeyFrame);
        handle.pKeyFrame = pKFi;
        mImageFrames[handle.frameId] = handle;
    }

    int nextFramePoseVertexId = static_cast<int>(maxKFid + 1);
    auto addWindowFramesForKeyFrame = [&](KeyFrame* pKFi, const bool fixedPose)
    {
        std::vector<WindowFrameSnapshot>& vWindowFrames = pKFi->GetMutableImageFrameWindow();
        std::sort(vWindowFrames.begin(), vWindowFrames.end(),
                  [](const WindowFrameSnapshot& lhs, const WindowFrameSnapshot& rhs)
                  {
                      if(lhs.mTimeStamp != rhs.mTimeStamp)
                          return lhs.mTimeStamp < rhs.mTimeStamp;
                      return lhs.mnFrameId < rhs.mnFrameId;
                  });

        for(size_t i = 0; i < vWindowFrames.size(); ++i)
        {
            WindowFrameSnapshot& frame = vWindowFrames[i];
            if(!frame.HasPose() || mImageFrames.count(frame.mnFrameId))
                continue;

            g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
            const Sophus::SE3f Tcw = frame.mTcw;
            vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
            vSE3->setId(nextFramePoseVertexId);
            vSE3->setFixed(fixedPose);
            optimizer.addVertex(vSE3);

            ImageFrameHandle handle;
            handle.frameId = frame.mnFrameId;
            handle.timeStamp = frame.mTimeStamp;
            handle.poseVertexId = nextFramePoseVertexId;
            handle.isKeyFrame = false;
            handle.fixed = fixedPose;
            handle.inCurrentInterval = (pKFi == pKF);
            handle.pKeyFrame = NULL;
            handle.pWindowFrame = &frame;
            mImageFrames[handle.frameId] = handle;

            ++nextFramePoseVertexId;
            ++nWindowFrameVertices;
            if(fixedPose)
                ++nFixedWindowFrameVertices;
        }
    };

    if(strictEq16ImageWindow)
    {
        addWindowFramesForKeyFrame(pKF, false);
    }
    else
    {
        for(list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; ++lit)
            addWindowFramesForKeyFrame(*lit, false);
        for(list<KeyFrame*>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; ++lit)
            addWindowFramesForKeyFrame(*lit, true);
    }

    for(std::map<unsigned long, ImageFrameHandle>::const_iterator itFrame = mImageFrames.begin();
        itFrame != mImageFrames.end(); ++itFrame)
    {
        if(itFrame->second.inCurrentInterval)
            ++nTauImageFrames;
    }

    const int nExpectedSize = static_cast<int>(mImageFrames.size() * lLocalMapPoints.size());
    vector<EdgePanopticProjection*> vpEdgesPanopticMono;
    vpEdgesPanopticMono.reserve(nExpectedSize);
    vector<double> vpEdgePanopticMonoBaseInvSigma2;
    vpEdgePanopticMonoBaseInvSigma2.reserve(nExpectedSize);
    vector<EdgePanopticInstanceProjection*> vpEdgesPanopticInstanceMono;
    vpEdgesPanopticInstanceMono.reserve(nExpectedSize);
    vector<double> vpEdgePanopticInstanceMonoBaseInvSigma2;
    vpEdgePanopticInstanceMonoBaseInvSigma2.reserve(nExpectedSize);
    vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody*> vpEdgesBody;
    vpEdgesBody.reserve(nExpectedSize);
    vector<double> vpEdgeBodyBaseInvSigma2;
    vpEdgeBodyBaseInvSigma2.reserve(nExpectedSize);
    vector<KeyFrame*> vpEdgeKFPanopticMono;
    vpEdgeKFPanopticMono.reserve(nExpectedSize);
    vector<KeyFrame*> vpEdgeKFBody;
    vpEdgeKFBody.reserve(nExpectedSize);
    vector<unsigned long> vpEdgeFramePanopticMono;
    vpEdgeFramePanopticMono.reserve(nExpectedSize);
    vector<unsigned long> vpEdgeFrameBody;
    vpEdgeFrameBody.reserve(nExpectedSize);
    vector<MapPoint*> vpMapPointEdgePanopticMono;
    vpMapPointEdgePanopticMono.reserve(nExpectedSize);
    vector<MapPoint*> vpMapPointEdgeBody;
    vpMapPointEdgeBody.reserve(nExpectedSize);
    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);
    vector<double> vpEdgeStereoBaseInvSigma2;
    vpEdgeStereoBaseInvSigma2.reserve(nExpectedSize);
    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);
    vector<unsigned long> vpEdgeFrameStereo;
    vpEdgeFrameStereo.reserve(nExpectedSize);
    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991f);
    const float thHuberStereo = sqrt(7.815f);
    const float thHuberShape = static_cast<float>(GetShapeHuberDelta());
    const float thHuberRigidity = static_cast<float>(GetRigidityHuberDelta());
    const double kDynamicPointPriorInvSigma2 = 0.1;
    const size_t kMaxShapeTripletsPerFrame = GetShapeTripletSampleLimit();
    const bool useStrictShapeTripletFactors = UseStrictShapeScaleTripletFactors();
    const size_t kMaxRigidityPairsPerNeighbor = GetRigidityPairSampleLimit();
    const size_t kMaxRigidityPairsPerPoint = GetRigidityMaxPairsPerPoint();
    const double kRigidityPairMinQuality = GetRigidityPairMinQuality();
    const int kRigidityPairMinKnownObservations = GetRigidityPairMinKnownObservations();
    const bool useQualityGatedRigidityPairSampler = UseQualityGatedRigidityPairSampler();
    const bool useQualityCoverageRigidityPairSampler = UseQualityCoverageRigidityPairSampler();
    const size_t kMaxRigidityFrameGap = GetRigidityMaxFrameGap();
    const int kDynamicBackendMinFrameSupport = GetDynamicBackendMinFrameSupport();
    const bool disableEq17SizePrior = DisableEq17SizePriorForLBA();

    unsigned long maxMapPointId = 0;
    for(list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; ++lit)
    {
        if((*lit)->mnId > maxMapPointId)
            maxMapPointId = (*lit)->mnId;
    }

    const int pointVertexBaseId = nextFramePoseVertexId;
    int nextDynamicPointId = pointVertexBaseId + static_cast<int>(maxMapPointId) + 1;
    int nPoints = 0;
    int nEdges = 0;
    int nPanopticInstanceEdges = 0;
    int nInstancePoseVertices = 0;
    int nInstanceStructureEdges = 0;
    int nDynamicReweightedObservations = 0;
    int nBackendOutlierMarks = 0;

    std::map<DynamicVertexKey, int> mThingVertexIds;
    std::map<MapPoint*, std::vector<int> > mmThingVertexIdsByPoint;
    std::map<MapPoint*, int> mThingCurrentVertexIds;
    std::map<int, std::vector<int> > mmInstanceCurrentVertexIds;
    std::map<int, std::map<unsigned long, std::vector<MapPoint*> > > mmInstanceFramePoints;
    std::map<InstanceFrameKey, int> mInstancePoseVertexIds;
    std::map<unsigned long, std::map<MapPoint*, int> > mmFrameLeftIndices;
    std::map<int, Instance*> mInstances;
    std::map<int, bool> mInstanceBackendMature;
    std::map<unsigned long, std::map<MapPoint*, FrameObservationIndex> > mmFrameObservations;
    std::map<MapPoint*, std::vector<unsigned long> > mmPointFrameIds;
    std::map<MapPoint*, int> mDynamicObservationInstanceIds;
    std::map<DynamicVertexKey, Eigen::Vector3d> mDynamicObservationPointWorlds;
    std::set<DynamicVertexKey> sAcceptedDynamicObservationKeys;
    std::set<DynamicVertexKey> sPersistentDynamicObservationKeys;

    struct InstanceConstraintDebug
    {
        int semanticLabel = 0;
        int currentPoints = 0;
        int maxFramePoints = 0;
        int windowFrames = 0;
        int backendMature = 0;
        int currentSupportOk = 0;
        int windowSupportOk = 0;
        int shapeCandidateFrames = 0;
        int shapeCandidatePoints = 0;
        int shapeTriplets = 0;
        int shapeValidTriplets = 0;
        int rigidityCandidateFramePairs = 0;
        int rigidityCommonPoints = 0;
        int rigidityFramePairsTested = 0;
        int rigidityInsufficientCommonPairs = 0;
        int rigidityMissingVertexPairs = 0;
        int shapeEdges = 0;
        int rigidityEdges = 0;
        int rigidityPointPairs = 0;
        int dynamicObservations = 0;
        int acceptedDynamicObservations = 0;
        int rejectedDynamicNullPoint = 0;
        int rejectedDynamicBadOrMap = 0;
        int rejectedDynamicInvalidInstance = 0;
        int rejectedDynamicFeatureRange = 0;
        int rejectedDynamicRightFeature = 0;
        int pointWorldConsistencyChecked = 0;
        int pointWorldConsistencyBad = 0;
        int pointWorldConsistencyRefined = 0;
        int pointWorldConsistencyNegativeDepth = 0;
        double pointWorldReprojectionErrorSum = 0.0;
        double pointWorldReprojectionErrorMax = 0.0;
        int pointTrackLen1 = 0;
        int pointTrackLen2 = 0;
        int pointTrackLen3Plus = 0;
        int maxPointTrackLen = 0;
        int qualityGateAcceptedFrames = 0;
        int qualityGateRejectedFrames = 0;
        int qualityGateSparseFrames = 0;
    };

    struct DynamicInitializationDebug
    {
        int identity = 0;
        int keyframeMotionPrior = 0;
        int snapshotMotionPrior = 0;
        int velocityFallback = 0;
        int observationPointWorld = 0;
        int observationPointWorldMissing = 0;
        double observationPointWorldDeltaSum = 0.0;
    };

    struct DynamicObservationSummaryDebug
    {
        int framesWithDynamicObservations = 0;
        int dynamicObservations = 0;
        int acceptedDynamicObservations = 0;
        int rejectedDynamicNullPoint = 0;
        int rejectedDynamicBadOrMap = 0;
        int rejectedDynamicInvalidInstance = 0;
        int rejectedDynamicFeatureRange = 0;
        int rejectedDynamicRightFeature = 0;
        int pointWorldConsistencyChecked = 0;
        int pointWorldConsistencyBad = 0;
        int pointWorldConsistencyRefined = 0;
        int pointWorldConsistencyNegativeDepth = 0;
        double pointWorldReprojectionErrorSum = 0.0;
        double pointWorldReprojectionErrorMax = 0.0;
    };

    std::map<int, InstanceConstraintDebug> mInstanceConstraintDebug;
    DynamicInitializationDebug dynamicInitDebug;
    DynamicObservationSummaryDebug dynamicObservationDebug;

    struct ShapeScaleEdgeRecord
    {
        EdgeShapeScaleFrame* frameEdge = NULL;
        EdgeShapeScaleTriplet* tripletEdge = NULL;
        int instanceId = -1;
        double baseInvSigma2 = 1.0;
        double factorScale = 1.0;
        size_t logicalTerms = 0;
        std::vector<DynamicEdgeObservation> observations;
    };

    struct RigidityEdgeRecord
    {
        EdgeRigidityFramePair* frameEdge = NULL;
        EdgeRigidityPair* pairEdge = NULL;
        int instanceId = -1;
        double baseInvSigma2 = 1.0;
        double factorScale = 1.0;
        size_t logicalTerms = 0;
        std::vector<DynamicEdgeObservation> observations;
    };

    struct PanopticInstanceEdgeRecord
    {
        EdgePanopticInstanceProjection* edge = NULL;
        MapPoint* pMP = NULL;
        int instanceId = -1;
        int instancePoseVertexId = -1;
        unsigned long frameId = 0;
        double baseInvSigma2 = 1.0;
    };

    struct InstanceProjectionDebugStats
    {
        size_t records = 0;
        size_t highChi2 = 0;
        size_t depthFailures = 0;
        double chi2Sum = 0.0;
        double chi2Max = 0.0;
        double directChi2Sum = 0.0;
        double directChi2Max = 0.0;
        double reconErrorSum = 0.0;
        double reconErrorMax = 0.0;
        double localNormSum = 0.0;
        double localNormMax = 0.0;

        void Add(double chi2,
                 double directChi2,
                 double reconError,
                 double localNorm,
                 bool depthPositive)
        {
            ++records;
            chi2Sum += chi2;
            chi2Max = std::max(chi2Max, chi2);
            directChi2Sum += directChi2;
            directChi2Max = std::max(directChi2Max, directChi2);
            reconErrorSum += reconError;
            reconErrorMax = std::max(reconErrorMax, reconError);
            localNormSum += localNorm;
            localNormMax = std::max(localNormMax, localNorm);
            if(chi2 > 5.991)
                ++highChi2;
            if(!depthPositive)
                ++depthFailures;
        }

        double MeanChi2() const { return records > 0 ? chi2Sum / static_cast<double>(records) : 0.0; }
        double MeanDirectChi2() const { return records > 0 ? directChi2Sum / static_cast<double>(records) : 0.0; }
        double MeanReconError() const { return records > 0 ? reconErrorSum / static_cast<double>(records) : 0.0; }
        double MeanLocalNorm() const { return records > 0 ? localNormSum / static_cast<double>(records) : 0.0; }
    };

    std::vector<ShapeScaleEdgeRecord> vShapeScaleEdgeRecords;
    std::vector<RigidityEdgeRecord> vRigidityEdgeRecords;
    std::vector<PanopticInstanceEdgeRecord> vPanopticInstanceEdgeRecords;
    std::map<int, InstanceProjectionDebugStats> mPanopticInstanceInitialDebug;
    std::set<InstanceFrameKey> sRejectedInstanceFramesByQualityGate;
    int nInstanceObservationQualityGateAccepted = 0;
    int nInstanceObservationQualityGateRejected = 0;
    int nInstanceObservationQualityGateSparse = 0;

    auto addDynamicFrameObservation =
        [&](const ImageFrameHandle& frame,
            MapPoint* pMP,
            const int featureIdx,
            const int instanceId,
            const int semanticLabel,
            const int nFeatures,
            const int leftFeatureCount,
            const bool persistentObservation,
            const Eigen::Vector3f* pObservationPointWorld)
    {
        ++dynamicObservationDebug.dynamicObservations;
        InstanceConstraintDebug* pObsDebug = NULL;
        if(instanceId > 0)
        {
            pObsDebug = &mInstanceConstraintDebug[instanceId];
            ++pObsDebug->dynamicObservations;
            pObsDebug->semanticLabel = semanticLabel;
        }

        if(instanceId <= 0)
        {
            ++dynamicObservationDebug.rejectedDynamicInvalidInstance;
            if(pObsDebug)
                ++pObsDebug->rejectedDynamicInvalidInstance;
            return;
        }
        if(!pMP)
        {
            ++dynamicObservationDebug.rejectedDynamicNullPoint;
            if(pObsDebug)
                ++pObsDebug->rejectedDynamicNullPoint;
            return;
        }
        if(pMP->isBad() || pMP->GetMap() != pCurrentMap)
        {
            ++dynamicObservationDebug.rejectedDynamicBadOrMap;
            if(pObsDebug)
                ++pObsDebug->rejectedDynamicBadOrMap;
            return;
        }
        if(featureIdx < 0 || featureIdx >= nFeatures)
        {
            ++dynamicObservationDebug.rejectedDynamicFeatureRange;
            if(pObsDebug)
                ++pObsDebug->rejectedDynamicFeatureRange;
            return;
        }
        if(featureIdx >= leftFeatureCount)
        {
            ++dynamicObservationDebug.rejectedDynamicRightFeature;
            if(pObsDebug)
                ++pObsDebug->rejectedDynamicRightFeature;
            return;
        }

        Eigen::Vector3f refinedPointWorld =
            pObservationPointWorld ? *pObservationPointWorld :
            Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN());
        double refinedQualityMultiplier = 1.0;
        if(EnableDynamicObservationPointWorldRefinement() &&
           pObservationPointWorld &&
           pObservationPointWorld->allFinite())
        {
            const std::vector<cv::KeyPoint>& vKeysUn = GetFrameKeysUn(frame);
            GeometricCamera* pCamera = GetFrameCamera(frame);
            if(featureIdx >= 0 &&
               featureIdx < static_cast<int>(vKeysUn.size()) &&
               pCamera)
            {
                const Eigen::Vector3f pointCamera =
                    GetFramePose(frame) * (*pObservationPointWorld);
                if(pointCamera.allFinite() && pointCamera[2] > 0.0f)
                {
                    const Eigen::Vector2f projection =
                        pCamera->project(pointCamera);
                    const Eigen::Vector2f observed(vKeysUn[featureIdx].pt.x,
                                                   vKeysUn[featureIdx].pt.y);
                    const double reprojectionError =
                        projection.allFinite() ?
                        static_cast<double>((observed - projection).norm()) :
                        std::numeric_limits<double>::infinity();

                    ++dynamicObservationDebug.pointWorldConsistencyChecked;
                    dynamicObservationDebug.pointWorldReprojectionErrorSum +=
                        std::isfinite(reprojectionError) ? reprojectionError : 0.0;
                    if(std::isfinite(reprojectionError))
                    {
                        dynamicObservationDebug.pointWorldReprojectionErrorMax =
                            std::max(dynamicObservationDebug.pointWorldReprojectionErrorMax,
                                     reprojectionError);
                    }

                    if(pObsDebug)
                    {
                        ++pObsDebug->pointWorldConsistencyChecked;
                        pObsDebug->pointWorldReprojectionErrorSum +=
                            std::isfinite(reprojectionError) ? reprojectionError : 0.0;
                        if(std::isfinite(reprojectionError))
                        {
                            pObsDebug->pointWorldReprojectionErrorMax =
                                std::max(pObsDebug->pointWorldReprojectionErrorMax,
                                         reprojectionError);
                        }
                    }

                    if(!std::isfinite(reprojectionError) ||
                       reprojectionError >
                           GetDynamicObservationPointWorldMaxReprojectionErrorPx())
                    {
                        ++dynamicObservationDebug.pointWorldConsistencyBad;
                        if(pObsDebug)
                            ++pObsDebug->pointWorldConsistencyBad;

                        Eigen::Vector3f ray = pCamera->unprojectEig(vKeysUn[featureIdx].pt);
                        if(ray.allFinite() && std::fabs(ray[2]) > 1e-6f)
                        {
                            const Eigen::Vector3f refinedCameraPoint =
                                ray * (pointCamera[2] / ray[2]);
                            const Eigen::Vector3f candidateWorld =
                                GetFramePose(frame).inverse() * refinedCameraPoint;
                            if(candidateWorld.allFinite())
                            {
                                refinedPointWorld = candidateWorld;
                                refinedQualityMultiplier =
                                    GetDynamicObservationPointWorldRefinedWeight();
                                ++dynamicObservationDebug.pointWorldConsistencyRefined;
                                if(pObsDebug)
                                    ++pObsDebug->pointWorldConsistencyRefined;
                            }
                        }
                    }
                }
                else
                {
                    ++dynamicObservationDebug.pointWorldConsistencyNegativeDepth;
                    if(pObsDebug)
                        ++pObsDebug->pointWorldConsistencyNegativeDepth;
                }
            }
        }

        FrameObservationIndex& obs = mmFrameObservations[frame.frameId][pMP];
        obs.leftIndex = featureIdx;
        const DynamicVertexKey dynamicKey(pMP, frame.frameId);
        sAcceptedDynamicObservationKeys.insert(dynamicKey);
        if(persistentObservation)
            sPersistentDynamicObservationKeys.insert(dynamicKey);
        if(refinedPointWorld.allFinite())
            mDynamicObservationPointWorlds[dynamicKey] = refinedPointWorld.cast<double>();
        mDynamicObservationInstanceIds[pMP] = instanceId;
        if(mInstances.count(instanceId) == 0)
            mInstances[instanceId] = pCurrentMap->GetInstance(instanceId);
        if(mInstances[instanceId] && refinedQualityMultiplier < 1.0)
        {
            const double existingWeight =
                mInstances[instanceId]->GetObservationQualityWeight(frame.frameId, pMP);
            mInstances[instanceId]->SetObservationQualityWeight(
                frame.frameId,
                pMP,
                std::max(0.05, std::min(existingWeight, refinedQualityMultiplier)));
        }

        ++dynamicObservationDebug.acceptedDynamicObservations;
        if(pObsDebug)
            ++pObsDebug->acceptedDynamicObservations;
    };

    const std::vector<Instance*> vPersistentDynamicInstances = pCurrentMap->GetAllInstances();

    for(std::map<unsigned long, ImageFrameHandle>::iterator itFrame = mImageFrames.begin();
        itFrame != mImageFrames.end(); ++itFrame)
    {
        const ImageFrameHandle& frame = itFrame->second;
        if(!frame.inCurrentInterval)
            continue;

        const int nFeatures = GetFrameFeatureCount(frame);
        const int leftFeatureCount = GetFrameLeftFeatureCount(frame);
        if(frame.isKeyFrame)
        {
            const std::vector<MapPoint*> vpMapPointMatches = frame.pKeyFrame->GetMapPointMatches();
            const int nUsable = std::min(nFeatures, static_cast<int>(vpMapPointMatches.size()));
            for(int i = 0; i < nUsable; ++i)
            {
                MapPoint* pMP = vpMapPointMatches[i];
                if(!pMP || pMP->isBad() || pMP->GetMap() != pCurrentMap)
                    continue;

                FrameObservationIndex& obs = mmFrameObservations[frame.frameId][pMP];
                if(i < leftFeatureCount)
                    obs.leftIndex = i;
                else
                    obs.rightIndex = i;
            }
        }
        else if(frame.pWindowFrame)
        {
            const std::vector<MapPoint*>& vpMapPointMatches = frame.pWindowFrame->mvpMapPoints;
            const int nUsable = std::min(nFeatures, static_cast<int>(vpMapPointMatches.size()));
            for(int i = 0; i < nUsable; ++i)
            {
                if(FrameObservationIsOutlier(frame, i))
                    continue;

                MapPoint* pMP = vpMapPointMatches[i];
                if(!pMP || pMP->isBad() || pMP->GetMap() != pCurrentMap)
                    continue;

                FrameObservationIndex& obs = mmFrameObservations[frame.frameId][pMP];
                if(i < leftFeatureCount)
                    obs.leftIndex = i;
                else
                    obs.rightIndex = i;
            }
        }

        const std::vector<DynamicInstancePointObservation>& vDynamicObservations =
            GetFrameDynamicInstancePointObservations(frame);
        bool frameHasDynamicObservations = !vDynamicObservations.empty();
        for(size_t instanceIdx = 0; instanceIdx < vPersistentDynamicInstances.size(); ++instanceIdx)
        {
            Instance* pInstance = vPersistentDynamicInstances[instanceIdx];
            if(!pInstance)
                continue;
            if(!pInstance->GetDynamicObservationsForFrame(frame.frameId).empty())
            {
                frameHasDynamicObservations = true;
                break;
            }
        }
        if(frameHasDynamicObservations)
            ++dynamicObservationDebug.framesWithDynamicObservations;
        for(size_t dynIdx = 0; dynIdx < vDynamicObservations.size(); ++dynIdx)
        {
            const DynamicInstancePointObservation& dynamicObservation = vDynamicObservations[dynIdx];
            addDynamicFrameObservation(frame,
                                       dynamicObservation.pBackendPoint,
                                       dynamicObservation.featureIdx,
                                       dynamicObservation.instanceId,
                                       dynamicObservation.semanticLabel,
                                       nFeatures,
                                       leftFeatureCount,
                                       false,
                                       &dynamicObservation.pointWorld);
        }

        for(size_t instanceIdx = 0; instanceIdx < vPersistentDynamicInstances.size(); ++instanceIdx)
        {
            Instance* pInstance = vPersistentDynamicInstances[instanceIdx];
            if(!pInstance)
                continue;

            const std::vector<Instance::DynamicObservationRecord> vRecords =
                pInstance->GetDynamicObservationsForFrame(frame.frameId);
            for(size_t recordIdx = 0; recordIdx < vRecords.size(); ++recordIdx)
            {
                addDynamicFrameObservation(frame,
                                           vRecords[recordIdx].pBackendPoint,
                                           vRecords[recordIdx].featureIdx,
                                           pInstance->GetId(),
                                           pInstance->GetSemanticLabel(),
                                           nFeatures,
                                           leftFeatureCount,
                                           true,
                                           &vRecords[recordIdx].pointWorld);
            }
        }

        std::map<MapPoint*, FrameObservationIndex>& mObservations = mmFrameObservations[frame.frameId];
        for(std::map<MapPoint*, FrameObservationIndex>::iterator itObs = mObservations.begin();
            itObs != mObservations.end(); ++itObs)
        {
            mmPointFrameIds[itObs->first].push_back(frame.frameId);
        }
    }

    for(std::map<MapPoint*, std::vector<unsigned long> >::iterator itPoint = mmPointFrameIds.begin();
        itPoint != mmPointFrameIds.end(); ++itPoint)
    {
        std::vector<unsigned long>& vFrameIds = itPoint->second;
        std::sort(vFrameIds.begin(), vFrameIds.end(),
                  [&](const unsigned long lhs, const unsigned long rhs)
                  {
                      return SortImageFramesByTimestamp(mImageFrames[lhs], mImageFrames[rhs]);
                  });
        vFrameIds.erase(std::unique(vFrameIds.begin(), vFrameIds.end()), vFrameIds.end());
    }

    auto addPanopticProjectionEdge = [&](int pointVertexId,
                                         const ImageFrameHandle& frame,
                                         MapPoint* pMP,
                                         int leftIndex)
    {
        const std::vector<cv::KeyPoint>& vKeysUn = GetFrameKeysUn(frame);
        const std::vector<float>& vInvLevelSigma2 = GetFrameInvLevelSigma2(frame);
        const cv::KeyPoint& kpUn = vKeysUn[leftIndex];
        Eigen::Matrix<double,2,1> obs;
        obs << kpUn.pt.x, kpUn.pt.y;

        EdgePanopticProjection* e = new EdgePanopticProjection();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pointVertexId)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(frame.poseVertexId)));
        e->setMeasurement(obs);
        const double invSigma2 = vInvLevelSigma2[kpUn.octave];
        const double dynamicWeight = GetDynamicObservationWeight(pCurrentMap, pMP, frame.frameId);
        e->setInformation(Eigen::Matrix2d::Identity() *
                          invSigma2 * dynamicWeight * panopticFactorWeight);
        if(dynamicWeight < 1.0)
            ++nDynamicReweightedObservations;

        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(thHuberMono);
        e->pCamera = GetFrameCamera(frame);

        optimizer.addEdge(e);
        vpEdgesPanopticMono.push_back(e);
        vpEdgePanopticMonoBaseInvSigma2.push_back(invSigma2);
        vpEdgeKFPanopticMono.push_back(frame.isKeyFrame ? frame.pKeyFrame : static_cast<KeyFrame*>(NULL));
        vpEdgeFramePanopticMono.push_back(frame.frameId);
        vpMapPointEdgePanopticMono.push_back(pMP);
        ++nEdges;
    };

    auto addPanopticInstanceProjectionEdge = [&](const int instancePoseVertexId,
                                                 const Eigen::Vector3d& localPoint,
                                                 const ImageFrameHandle& frame,
                                                 MapPoint* pMP,
                                                 int leftIndex,
                                                 const int instanceId)
    {
        const std::vector<cv::KeyPoint>& vKeysUn = GetFrameKeysUn(frame);
        const std::vector<float>& vInvLevelSigma2 = GetFrameInvLevelSigma2(frame);
        const cv::KeyPoint& kpUn = vKeysUn[leftIndex];
        Eigen::Matrix<double,2,1> obs;
        obs << kpUn.pt.x, kpUn.pt.y;

        g2o::VertexSE3Expmap* vInitialInstancePose =
            static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(instancePoseVertexId));
        g2o::VertexSE3Expmap* vInitialCameraPose =
            static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(frame.poseVertexId));
        GeometricCamera* pInitialCamera = GetFrameCamera(frame);
        if(!vInitialInstancePose || !vInitialCameraPose || !pInitialCamera)
            return;

        const Eigen::Vector3d initialWorldPoint =
            vInitialInstancePose->estimate().map(localPoint);
        const Eigen::Vector3d initialCameraPoint =
            vInitialCameraPose->estimate().map(initialWorldPoint);
        if(!initialCameraPoint.allFinite() || initialCameraPoint[2] <= 0.0)
            return;
        const Eigen::Vector2d initialProjection =
            pInitialCamera->project(initialCameraPoint);
        if(!initialProjection.allFinite())
            return;

        EdgePanopticInstanceProjection* e = new EdgePanopticInstanceProjection();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(instancePoseVertexId)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(frame.poseVertexId)));
        e->setMeasurement(obs);
        e->SetLocalPoint(localPoint);
        const double invSigma2 = vInvLevelSigma2[kpUn.octave];
        const double dynamicWeight = GetDynamicObservationWeight(pCurrentMap, pMP, frame.frameId);
        e->setInformation(Eigen::Matrix2d::Identity() *
                          invSigma2 * dynamicWeight * panopticFactorWeight);
        if(dynamicWeight < 1.0)
            ++nDynamicReweightedObservations;

        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(thHuberMono);
        e->pCamera = pInitialCamera;

        optimizer.addEdge(e);

        e->computeError();
        double directChi2 = 0.0;
        double reconError = 0.0;
        const std::map<DynamicVertexKey, int>::const_iterator itPointVertex =
            mThingVertexIds.find(DynamicVertexKey(pMP, frame.frameId));
        g2o::VertexSBAPointXYZ* vPoint = NULL;
        if(itPointVertex != mThingVertexIds.end())
            vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(itPointVertex->second));
        g2o::VertexSE3Expmap* vInstancePose =
            static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(instancePoseVertexId));
        g2o::VertexSE3Expmap* vCameraPose =
            static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(frame.poseVertexId));
        if(vPoint && vInstancePose)
        {
            const Eigen::Vector3d instanceWorldPoint = vInstancePose->estimate().map(localPoint);
            reconError = (instanceWorldPoint - vPoint->estimate()).norm();
        }
        if(vPoint && vCameraPose && e->pCamera)
        {
            const Eigen::Vector3d pointCam = vCameraPose->estimate().map(vPoint->estimate());
            if(pointCam[2] > 0.0)
            {
                const Eigen::Vector2d directResidual = obs - e->pCamera->project(pointCam);
                directChi2 = directResidual.squaredNorm() *
                             invSigma2 * dynamicWeight * panopticFactorWeight;
            }
            else
            {
                directChi2 = 1e12;
            }
        }
        mPanopticInstanceInitialDebug[instanceId].Add(e->chi2(),
                                                      directChi2,
                                                      reconError,
                                                      localPoint.norm(),
                                                      e->isDepthPositive());

        vpEdgesPanopticInstanceMono.push_back(e);
        vpEdgePanopticInstanceMonoBaseInvSigma2.push_back(invSigma2);
        PanopticInstanceEdgeRecord record;
        record.edge = e;
        record.pMP = pMP;
        record.instanceId = instanceId;
        record.instancePoseVertexId = instancePoseVertexId;
        record.frameId = frame.frameId;
        record.baseInvSigma2 = invSigma2;
        vPanopticInstanceEdgeRecords.push_back(record);
        ++nEdges;
        ++nPanopticInstanceEdges;
    };

    auto addStereoEdge = [&](int pointVertexId,
                             const ImageFrameHandle& frame,
                             MapPoint* pMP,
                             int leftIndex)
    {
        const std::vector<cv::KeyPoint>& vKeysUn = GetFrameKeysUn(frame);
        const std::vector<float>& vRightCoordinates = GetFrameRightCoordinates(frame);
        const std::vector<float>& vInvLevelSigma2 = GetFrameInvLevelSigma2(frame);
        const cv::KeyPoint& kpUn = vKeysUn[leftIndex];
        Eigen::Matrix<double,3,1> obs;
        obs << kpUn.pt.x, kpUn.pt.y, vRightCoordinates[leftIndex];

        g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pointVertexId)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(frame.poseVertexId)));
        e->setMeasurement(obs);
        const double invSigma2 = vInvLevelSigma2[kpUn.octave];
        const double dynamicWeight = GetDynamicObservationWeight(pCurrentMap, pMP, frame.frameId);
        e->setInformation(Eigen::Matrix3d::Identity() * invSigma2 * dynamicWeight);
        if(dynamicWeight < 1.0)
            ++nDynamicReweightedObservations;

        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(thHuberStereo);

        e->fx = frame.isKeyFrame ? frame.pKeyFrame->fx : frame.pWindowFrame->fx;
        e->fy = frame.isKeyFrame ? frame.pKeyFrame->fy : frame.pWindowFrame->fy;
        e->cx = frame.isKeyFrame ? frame.pKeyFrame->cx : frame.pWindowFrame->cx;
        e->cy = frame.isKeyFrame ? frame.pKeyFrame->cy : frame.pWindowFrame->cy;
        e->bf = frame.isKeyFrame ? frame.pKeyFrame->mbf : frame.pWindowFrame->mbf;

        optimizer.addEdge(e);
        vpEdgesStereo.push_back(e);
        vpEdgeStereoBaseInvSigma2.push_back(invSigma2);
        vpEdgeKFStereo.push_back(frame.isKeyFrame ? frame.pKeyFrame : static_cast<KeyFrame*>(NULL));
        vpEdgeFrameStereo.push_back(frame.frameId);
        vpMapPointEdgeStereo.push_back(pMP);
        ++nEdges;
    };

    auto addBodyEdge = [&](int pointVertexId,
                           const ImageFrameHandle& frame,
                           MapPoint* pMP,
                           int rightIndex)
    {
        if(!GetFrameCamera2(frame))
            return;

        const int rightOffset = rightIndex - GetFrameLeftFeatureCount(frame);
        const std::vector<cv::KeyPoint>& vKeysRight = GetFrameKeysRight(frame);
        if(rightOffset < 0 || rightOffset >= static_cast<int>(vKeysRight.size()))
            return;

        Eigen::Matrix<double,2,1> obs;
        const cv::KeyPoint& kp = vKeysRight[rightOffset];
        obs << kp.pt.x, kp.pt.y;

        ORB_SLAM3::EdgeSE3ProjectXYZToBody* e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pointVertexId)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(frame.poseVertexId)));
        e->setMeasurement(obs);
        const std::vector<float>& vInvLevelSigma2 = GetFrameInvLevelSigma2(frame);
        const double invSigma2 = vInvLevelSigma2[kp.octave];
        const double dynamicWeight = GetDynamicObservationWeight(pCurrentMap, pMP, frame.frameId);
        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2 * dynamicWeight);
        if(dynamicWeight < 1.0)
            ++nDynamicReweightedObservations;

        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(thHuberMono);

        Sophus::SE3f Trl = GetFrameTrl(frame);
        e->mTrl = g2o::SE3Quat(Trl.unit_quaternion().cast<double>(), Trl.translation().cast<double>());
        e->pCamera = GetFrameCamera2(frame);

        optimizer.addEdge(e);
        vpEdgesBody.push_back(e);
        vpEdgeBodyBaseInvSigma2.push_back(invSigma2);
        vpEdgeKFBody.push_back(frame.isKeyFrame ? frame.pKeyFrame : static_cast<KeyFrame*>(NULL));
        vpEdgeFrameBody.push_back(frame.frameId);
        vpMapPointEdgeBody.push_back(pMP);
        ++nEdges;
    };

    for(list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; ++lit)
    {
        MapPoint* pMP = *lit;
        int instanceId = pMP->GetInstanceId();
        if(instanceId <= 0 && mDynamicObservationInstanceIds.count(pMP))
            instanceId = mDynamicObservationInstanceIds[pMP];

        const std::map<MapPoint*, std::vector<unsigned long> >::const_iterator itPointFrames =
            mmPointFrameIds.find(pMP);
        if(itPointFrames == mmPointFrameIds.end())
            continue;

        bool bThingPoint = false;
        if(instanceId > 0)
        {
            for(size_t frameIdx = 0; frameIdx < itPointFrames->second.size(); ++frameIdx)
            {
                const std::map<unsigned long, ImageFrameHandle>::const_iterator itFrame =
                    mImageFrames.find(itPointFrames->second[frameIdx]);
                const DynamicVertexKey dynamicKey(pMP, itPointFrames->second[frameIdx]);
                if(itFrame != mImageFrames.end() &&
                   (FrameUsesDynamicInstanceMotion(itFrame->second, instanceId) ||
                    sAcceptedDynamicObservationKeys.count(dynamicKey) > 0))
                {
                    bThingPoint = true;
                    break;
                }
            }
        }

        int stuffVertexId = -1;
        auto ensureStaticPointVertex = [&]() -> int
        {
            if(stuffVertexId >= 0)
                return stuffVertexId;

            g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
            stuffVertexId = pointVertexBaseId + static_cast<int>(pMP->mnId);
            vPoint->setId(stuffVertexId);
            vPoint->setMarginalized(false);
            optimizer.addVertex(vPoint);
            ++nPoints;

            return stuffVertexId;
        };

        if(!bThingPoint)
            ensureStaticPointVertex();
        else if(mInstances.count(instanceId) == 0)
        {
            mInstances[instanceId] = pCurrentMap->GetInstance(instanceId);
        }

        for(size_t obsIdx = 0; obsIdx < itPointFrames->second.size(); ++obsIdx)
        {
            const unsigned long frameId = itPointFrames->second[obsIdx];
            const std::map<unsigned long, ImageFrameHandle>::const_iterator itFrame = mImageFrames.find(frameId);
            if(itFrame == mImageFrames.end())
                continue;

                const ImageFrameHandle& frame = itFrame->second;
                const FrameObservationIndex& observation = mmFrameObservations[frameId][pMP];
                const int leftIndex = observation.leftIndex;
                const int rightIndex = observation.rightIndex;

                int pointVertexId = stuffVertexId;
                const DynamicVertexKey dynamicKey(pMP, frameId);
                const bool useDynamicObservation =
                    bThingPoint &&
                    (FrameUsesDynamicInstanceMotion(frame, instanceId) ||
                     sAcceptedDynamicObservationKeys.count(dynamicKey) > 0);
                if(useDynamicObservation)
                {
                const DynamicVertexKey key = dynamicKey;
                map<DynamicVertexKey, int>::iterator itVertex = mThingVertexIds.find(key);
                if(itVertex == mThingVertexIds.end())
                {
                    if(frame.isKeyFrame)
                    {
                        if(mInstances[instanceId] &&
                           mInstances[instanceId]->IsInitialized() &&
                           mInstances[instanceId]->HasReliableInitializationMotion())
                            ++dynamicInitDebug.keyframeMotionPrior;
                        else
                            ++dynamicInitDebug.identity;
                    }
                    else if(frame.pWindowFrame &&
                            frame.pWindowFrame->mmPredictedInstanceMotions.count(instanceId))
                    {
                        ++dynamicInitDebug.snapshotMotionPrior;
                    }
                    else if(mInstances[instanceId] &&
                            mInstances[instanceId]->IsInitialized() &&
                            mInstances[instanceId]->HasReliableInitializationMotion())
                    {
                        ++dynamicInitDebug.velocityFallback;
                    }
                    else
                    {
                        ++dynamicInitDebug.identity;
                    }

                    g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
                    const std::map<DynamicVertexKey, Eigen::Vector3d>::const_iterator itObservationPointWorld =
                        mDynamicObservationPointWorlds.find(key);
                    Eigen::Vector3d initPointWorld;
                    if(itObservationPointWorld != mDynamicObservationPointWorlds.end() &&
                       itObservationPointWorld->second.allFinite())
                    {
                        initPointWorld = itObservationPointWorld->second;
                        ++dynamicInitDebug.observationPointWorld;
                        dynamicInitDebug.observationPointWorldDeltaSum +=
                            (initPointWorld - pMP->GetWorldPos().cast<double>()).norm();
                    }
                    else
                    {
                        initPointWorld = EstimateThingVertexInitialization(pMP, mInstances[instanceId], frame);
                        ++dynamicInitDebug.observationPointWorldMissing;
                    }
                    vPoint->setEstimate(initPointWorld);
                    vPoint->setId(nextDynamicPointId);
                    vPoint->setMarginalized(false);
                    optimizer.addVertex(vPoint);

                    if(enableDynamicPointPrior)
                    {
                        EdgePointXYZPrior* ePrior = new EdgePointXYZPrior();
                        ePrior->setVertex(0, vPoint);
                        ePrior->setMeasurement(vPoint->estimate());
                        ePrior->setInformation(Eigen::Matrix3d::Identity() * kDynamicPointPriorInvSigma2);
                        optimizer.addEdge(ePrior);
                        ++nEdges;
                        ++nDynamicPriorEdges;
                    }

                    pointVertexId = nextDynamicPointId;
                    mThingVertexIds[key] = pointVertexId;
                    mmThingVertexIdsByPoint[pMP].push_back(pointVertexId);
                    mmInstanceFramePoints[instanceId][frameId].push_back(pMP);
                    if(frameId == pKF->mnFrameId)
                    {
                        mThingCurrentVertexIds[pMP] = pointVertexId;
                        mmInstanceCurrentVertexIds[instanceId].push_back(pointVertexId);
                    }

                    ++nextDynamicPointId;
                    ++nPoints;
                }
                else
                {
                    pointVertexId = itVertex->second;
                }

                if(leftIndex != -1)
                    mmFrameLeftIndices[frameId][pMP] = leftIndex;
            }
            else
            {
                pointVertexId = ensureStaticPointVertex();
            }

            if(leftIndex != -1)
            {
                if(GetFrameRightCoordinates(frame)[leftIndex] < 0)
                {
                    // Default Module 8 path: instance-pose / instance-structure
                    // coupling owns thing reprojection. The pure timestamped
                    // point projection is kept as a strict implicit Eq.(16)
                    // ablation when the coupling is disabled.
                    if(!(enableInstanceStructureProxy && useDynamicObservation && instanceId > 0))
                        addPanopticProjectionEdge(pointVertexId, frame, pMP, leftIndex);
                }
                else
                    addStereoEdge(pointVertexId, frame, pMP, leftIndex);
            }

            if(rightIndex != -1)
                addBodyEdge(pointVertexId, frame, pMP, rightIndex);
        }
    }

    int nextInstancePoseId = nextDynamicPointId;
    if(enableInstanceStructureProxy)
    {
        std::map<int, Eigen::Vector3d> mInstanceCanonicalCentroids;
        std::map<std::pair<int, MapPoint*>, Eigen::Vector3d> mInstanceCanonicalLocalPoints;

        auto getThingPointEstimate = [&](MapPoint* pMP,
                                         const unsigned long frameId,
                                         Eigen::Vector3d& estimate) -> bool
        {
            const std::map<DynamicVertexKey, int>::const_iterator itVertexId =
                mThingVertexIds.find(DynamicVertexKey(pMP, frameId));
            if(itVertexId == mThingVertexIds.end())
                return false;

            g2o::VertexSBAPointXYZ* vPoint =
                static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(itVertexId->second));
            if(!vPoint)
                return false;

            estimate = vPoint->estimate();
            return estimate.allFinite();
        };

        auto buildCanonicalInstanceShape =
            [&](const int instanceId,
                std::map<unsigned long, std::vector<MapPoint*> >& mFramePoints) -> bool
        {
            if(mInstanceCanonicalCentroids.count(instanceId))
                return true;

            std::vector<unsigned long> vFrameIds;
            vFrameIds.reserve(mFramePoints.size());
            for(std::map<unsigned long, std::vector<MapPoint*> >::const_iterator itFrame = mFramePoints.begin();
                itFrame != mFramePoints.end(); ++itFrame)
            {
                vFrameIds.push_back(itFrame->first);
            }
            std::sort(vFrameIds.begin(), vFrameIds.end(),
                      [&](const unsigned long lhs, const unsigned long rhs)
                      {
                          return SortImageFramesByTimestamp(mImageFrames[lhs], mImageFrames[rhs]);
                      });

            Instance* pInstanceForStructure =
                (mInstances.count(instanceId) > 0) ? mInstances[instanceId] :
                    pCurrentMap->GetInstance(instanceId);
            if(pInstanceForStructure)
            {
                Eigen::Vector3d poseTranslation = Eigen::Vector3d::Zero();
                int poseTranslationSupport = 0;
                for(size_t frameIdx = 0; frameIdx < vFrameIds.size(); ++frameIdx)
                {
                    const unsigned long frameId = vFrameIds[frameIdx];
                    const std::vector<MapPoint*>& vFramePoints = mFramePoints[frameId];
                    for(size_t pointIdx = 0; pointIdx < vFramePoints.size(); ++pointIdx)
                    {
                        MapPoint* pMP = vFramePoints[pointIdx];
                        Eigen::Vector3f localPointF;
                        Eigen::Vector3d estimate;
                        if(!pInstanceForStructure->GetStructureLocalPoint(pMP, localPointF) ||
                           !getThingPointEstimate(pMP, frameId, estimate))
                        {
                            continue;
                        }

                        poseTranslation += estimate - localPointF.cast<double>();
                        ++poseTranslationSupport;
                    }
                }

                if(poseTranslationSupport >= kDynamicBackendMinFrameSupport)
                {
                    poseTranslation /= static_cast<double>(poseTranslationSupport);
                    mInstanceCanonicalCentroids[instanceId] = poseTranslation;

                    for(std::map<unsigned long, std::vector<MapPoint*> >::const_iterator itFrame =
                            mFramePoints.begin();
                        itFrame != mFramePoints.end(); ++itFrame)
                    {
                        const unsigned long frameId = itFrame->first;
                        const std::vector<MapPoint*>& vFramePoints = itFrame->second;
                        for(size_t pointIdx = 0; pointIdx < vFramePoints.size(); ++pointIdx)
                        {
                            MapPoint* pMP = vFramePoints[pointIdx];
                            const std::pair<int, MapPoint*> key(instanceId, pMP);
                            if(mInstanceCanonicalLocalPoints.count(key))
                                continue;

                            Eigen::Vector3f localPointF;
                            if(pInstanceForStructure->GetStructureLocalPoint(pMP, localPointF))
                            {
                                mInstanceCanonicalLocalPoints[key] = localPointF.cast<double>();
                                continue;
                            }

                            Eigen::Vector3d estimate;
                            if(!getThingPointEstimate(pMP, frameId, estimate))
                                estimate = pMP->GetWorldPos().cast<double>();
                            mInstanceCanonicalLocalPoints[key] = estimate - poseTranslation;
                        }
                    }

                    return true;
                }
            }

            std::vector<MapPoint*> vReferencePoints;
            std::vector<Eigen::Vector3d> vReferenceEstimates;
            for(size_t frameIdx = 0; frameIdx < vFrameIds.size(); ++frameIdx)
            {
                const unsigned long frameId = vFrameIds[frameIdx];
                std::vector<MapPoint*> vCandidatePoints;
                std::vector<Eigen::Vector3d> vCandidateEstimates;
                const std::vector<MapPoint*>& vFramePoints = mFramePoints[frameId];
                for(size_t pointIdx = 0; pointIdx < vFramePoints.size(); ++pointIdx)
                {
                    Eigen::Vector3d estimate;
                    if(!getThingPointEstimate(vFramePoints[pointIdx], frameId, estimate))
                        continue;
                    vCandidatePoints.push_back(vFramePoints[pointIdx]);
                    vCandidateEstimates.push_back(estimate);
                }

                if(static_cast<int>(vCandidateEstimates.size()) >= kDynamicBackendMinFrameSupport)
                {
                    vReferencePoints.swap(vCandidatePoints);
                    vReferenceEstimates.swap(vCandidateEstimates);
                    break;
                }
            }

            if(static_cast<int>(vReferenceEstimates.size()) < kDynamicBackendMinFrameSupport)
                return false;

            Eigen::Vector3d canonicalCentroid = Eigen::Vector3d::Zero();
            for(size_t i = 0; i < vReferenceEstimates.size(); ++i)
                canonicalCentroid += vReferenceEstimates[i];
            canonicalCentroid /= static_cast<double>(vReferenceEstimates.size());
            mInstanceCanonicalCentroids[instanceId] = canonicalCentroid;

            for(size_t i = 0; i < vReferencePoints.size(); ++i)
            {
                mInstanceCanonicalLocalPoints[std::make_pair(instanceId, vReferencePoints[i])] =
                    vReferenceEstimates[i] - canonicalCentroid;
            }

            for(std::map<unsigned long, std::vector<MapPoint*> >::const_iterator itFrame = mFramePoints.begin();
                itFrame != mFramePoints.end(); ++itFrame)
            {
                const unsigned long frameId = itFrame->first;
                const std::vector<MapPoint*>& vFramePoints = itFrame->second;
                for(size_t pointIdx = 0; pointIdx < vFramePoints.size(); ++pointIdx)
                {
                    MapPoint* pMP = vFramePoints[pointIdx];
                    const std::pair<int, MapPoint*> key(instanceId, pMP);
                    if(mInstanceCanonicalLocalPoints.count(key))
                        continue;

                    Eigen::Vector3d estimate;
                    if(!getThingPointEstimate(pMP, frameId, estimate))
                        estimate = pMP->GetWorldPos().cast<double>();
                    mInstanceCanonicalLocalPoints[key] = estimate - canonicalCentroid;
                }
            }

            return true;
        };

        for(std::map<int, std::map<unsigned long, std::vector<MapPoint*> > >::iterator itInstance = mmInstanceFramePoints.begin();
            itInstance != mmInstanceFramePoints.end(); ++itInstance)
        {
            const int instanceId = itInstance->first;
            if(mInstances.count(instanceId) == 0)
                mInstances[instanceId] = pCurrentMap->GetInstance(instanceId);
            if(!buildCanonicalInstanceShape(instanceId, itInstance->second))
                continue;

            for(std::map<unsigned long, std::vector<MapPoint*> >::iterator itFramePoints = itInstance->second.begin();
                itFramePoints != itInstance->second.end(); ++itFramePoints)
            {
                const unsigned long frameId = itFramePoints->first;
                const std::map<unsigned long, ImageFrameHandle>::const_iterator itFrameHandle =
                    mImageFrames.find(frameId);
                if(itFrameHandle == mImageFrames.end())
                    continue;
                const ImageFrameHandle& frame = itFrameHandle->second;
                const std::vector<MapPoint*>& vFramePoints = itFramePoints->second;
                std::vector<int> vPointVertexIds;
                std::vector<MapPoint*> vValidPoints;
                std::vector<Eigen::Vector3d> vPointEstimates;
                vPointVertexIds.reserve(vFramePoints.size());
                vValidPoints.reserve(vFramePoints.size());
                vPointEstimates.reserve(vFramePoints.size());

                for(size_t i = 0; i < vFramePoints.size(); ++i)
                {
                    MapPoint* pMP = vFramePoints[i];
                    const DynamicVertexKey key(pMP, frameId);
                    const std::map<DynamicVertexKey, int>::const_iterator itVertexId = mThingVertexIds.find(key);
                    if(itVertexId == mThingVertexIds.end())
                        continue;

                    g2o::VertexSBAPointXYZ* vPoint =
                        static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(itVertexId->second));
                    if(!vPoint)
                        continue;

                    vPointVertexIds.push_back(itVertexId->second);
                    vValidPoints.push_back(pMP);
                    vPointEstimates.push_back(vPoint->estimate());
                }

                if(static_cast<int>(vPointVertexIds.size()) < kDynamicBackendMinFrameSupport)
                    continue;

                Eigen::Vector3d translation = Eigen::Vector3d::Zero();
                int translationSupport = 0;
                for(size_t i = 0; i < vValidPoints.size(); ++i)
                {
                    const std::pair<int, MapPoint*> key(instanceId, vValidPoints[i]);
                    const std::map<std::pair<int, MapPoint*>, Eigen::Vector3d>::const_iterator itLocal =
                        mInstanceCanonicalLocalPoints.find(key);
                    if(itLocal == mInstanceCanonicalLocalPoints.end())
                        continue;
                    translation += vPointEstimates[i] - itLocal->second;
                    ++translationSupport;
                }
                if(translationSupport < kDynamicBackendMinFrameSupport)
                    continue;
                translation /= static_cast<double>(translationSupport);

                Instance* pInstanceForPoseInit =
                    mInstances.count(instanceId) ? mInstances[instanceId] :
                    static_cast<Instance*>(NULL);
                const Sophus::SE3f centroidStructurePoseEstimate(
                    Eigen::Matrix3f::Identity(),
                    translation.cast<float>());
                std::string instancePoseInitSource = "centroid_structure";
                Sophus::SE3f stateChainPoseEstimate;
                std::string stateChainPoseSource = "none";
                const bool hasStateChainPoseEstimate =
                    PredictInstancePoseFromStateChain(pInstanceForPoseInit,
                                                      frameId,
                                                      stateChainPoseEstimate,
                                                      stateChainPoseSource);
                Sophus::SE3f instancePoseEstimate = centroidStructurePoseEstimate;

                InstanceProjectionDebugStats instanceFrameGateStats;
                if(EnableBackendInstanceObservationQualityGate())
                {
                    const std::map<unsigned long, std::map<MapPoint*, int> >::const_iterator itLeftMap =
                        mmFrameLeftIndices.find(frameId);
                    g2o::VertexSE3Expmap* vCameraPose =
                        static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(frame.poseVertexId));
                    GeometricCamera* pCamera = GetFrameCamera(frame);

                    auto evaluateInstancePoseCandidate =
                        [&](const Sophus::SE3f& candidatePose) -> InstanceProjectionDebugStats
                    {
                        InstanceProjectionDebugStats stats;
                        if(itLeftMap == mmFrameLeftIndices.end() || !vCameraPose || !pCamera)
                            return stats;

                        const std::vector<cv::KeyPoint>& vKeysUn = GetFrameKeysUn(frame);
                        const std::vector<float>& vRightCoordinates = GetFrameRightCoordinates(frame);
                        const std::vector<float>& vInvLevelSigma2 = GetFrameInvLevelSigma2(frame);
                        for(size_t i = 0; i < vValidPoints.size(); ++i)
                        {
                            MapPoint* pMP = vValidPoints[i];
                            const std::map<MapPoint*, int>::const_iterator itLeft =
                                itLeftMap->second.find(pMP);
                            if(itLeft == itLeftMap->second.end())
                                continue;

                            const int leftIndex = itLeft->second;
                            if(leftIndex < 0 ||
                               leftIndex >= static_cast<int>(vKeysUn.size()) ||
                               leftIndex >= static_cast<int>(vRightCoordinates.size()))
                            {
                                continue;
                            }
                            if(vRightCoordinates[leftIndex] >= 0)
                                continue;

                            const std::pair<int, MapPoint*> localKey(instanceId, pMP);
                            const std::map<std::pair<int, MapPoint*>, Eigen::Vector3d>::const_iterator itLocal =
                                mInstanceCanonicalLocalPoints.find(localKey);
                            if(itLocal == mInstanceCanonicalLocalPoints.end())
                                continue;

                            const Eigen::Vector3d instanceWorldPoint =
                                candidatePose.cast<double>() * itLocal->second;
                            const Eigen::Vector3d pointCamera =
                                vCameraPose->estimate().map(instanceWorldPoint);
                            const bool depthPositive =
                                pointCamera.allFinite() && pointCamera[2] > 0.0;
                            double chi2 = 1e12;
                            if(depthPositive)
                            {
                                const Eigen::Vector2d projection = pCamera->project(pointCamera);
                                const Eigen::Vector2d observation(vKeysUn[leftIndex].pt.x,
                                                                  vKeysUn[leftIndex].pt.y);
                                if(projection.allFinite())
                                {
                                    const double invSigma2 =
                                        vInvLevelSigma2[vKeysUn[leftIndex].octave];
                                    const double dynamicWeight =
                                        GetDynamicObservationWeight(pCurrentMap, pMP, frameId);
                                    chi2 = (observation - projection).squaredNorm() *
                                           invSigma2 * dynamicWeight * panopticFactorWeight;
                                }
                            }

                            const double reconError =
                                (instanceWorldPoint - vPointEstimates[i]).norm();
                            stats.Add(chi2,
                                      0.0,
                                      reconError,
                                      itLocal->second.norm(),
                                      depthPositive);
                        }
                        return stats;
                    };

                    const int minGateSupport = kDynamicBackendMinFrameSupport;
                    const InstanceProjectionDebugStats centroidGateStats =
                        evaluateInstancePoseCandidate(centroidStructurePoseEstimate);
                    InstanceProjectionDebugStats stateChainGateStats;
                    bool hasStateChainGateStats = false;
                    if(hasStateChainPoseEstimate)
                    {
                        stateChainGateStats =
                            evaluateInstancePoseCandidate(stateChainPoseEstimate);
                        hasStateChainGateStats =
                            static_cast<int>(stateChainGateStats.records) >= minGateSupport;
                    }

                    auto highChi2RatioForStats =
                        [](const InstanceProjectionDebugStats& stats) -> double
                    {
                        return stats.records > 0 ?
                            static_cast<double>(stats.highChi2) /
                                static_cast<double>(stats.records) : 1.0;
                    };
                    auto depthFailureRatioForStats =
                        [](const InstanceProjectionDebugStats& stats) -> double
                    {
                        return stats.records > 0 ?
                            static_cast<double>(stats.depthFailures) /
                                static_cast<double>(stats.records) : 1.0;
                    };
                    auto qualityGatePassesStats =
                        [&](const InstanceProjectionDebugStats& stats) -> bool
                    {
                        return static_cast<int>(stats.records) >= minGateSupport &&
                               stats.MeanChi2() <= GetBackendInstanceGateMaxMeanChi2() &&
                               highChi2RatioForStats(stats) <=
                                   GetBackendInstanceGateMaxHighChi2Ratio() &&
                               depthFailureRatioForStats(stats) <=
                                   GetBackendInstanceGateMaxDepthFailureRatio() &&
                               stats.MeanReconError() <=
                                   GetBackendInstanceGateMaxReconError();
                    };

                    instanceFrameGateStats = centroidGateStats;
                    const bool centroidGatePassed =
                        qualityGatePassesStats(centroidGateStats);
                    const bool stateChainGatePassed =
                        hasStateChainGateStats &&
                        qualityGatePassesStats(stateChainGateStats);
                    const bool strictRgbdDepthBackedGeometryGate =
                        EnableRgbdDepthBackedStrictGeometryGate();
                    if(hasStateChainGateStats &&
                       (stateChainGatePassed ||
                        (!strictRgbdDepthBackedGeometryGate &&
                         !centroidGatePassed &&
                         stateChainGateStats.MeanChi2() < centroidGateStats.MeanChi2())))
                    {
                        instanceFrameGateStats = stateChainGateStats;
                        instancePoseEstimate = stateChainPoseEstimate;
                        instancePoseInitSource = stateChainPoseSource;
                    }
                    else
                    {
                        instancePoseEstimate = centroidStructurePoseEstimate;
                        instancePoseInitSource =
                            hasStateChainPoseEstimate ?
                            "centroid_structure_state_chain_rejected" :
                            "centroid_structure";
                    }

                    if(static_cast<int>(instanceFrameGateStats.records) >= minGateSupport)
                    {
                        const double highChi2Ratio =
                            highChi2RatioForStats(instanceFrameGateStats);
                        const double depthFailureRatio =
                            depthFailureRatioForStats(instanceFrameGateStats);
                        const bool qualityGatePassed =
                            qualityGatePassesStats(instanceFrameGateStats);

                        if(!qualityGatePassed)
                        {
                            const bool hardRejectByQualityGate =
                                EnableBackendInstanceObservationQualityGateHardReject();
                            const bool rejectByStrictRgbdDepthBackedGeometry =
                                strictRgbdDepthBackedGeometryGate &&
                                !centroidGatePassed &&
                                !stateChainGatePassed;
                            if(hardRejectByQualityGate ||
                               rejectByStrictRgbdDepthBackedGeometry)
                            {
                                sRejectedInstanceFramesByQualityGate.insert(
                                    InstanceFrameKey(instanceId, frameId));
                            }
                            ++nInstanceObservationQualityGateRejected;
                            ++mInstanceConstraintDebug[instanceId].qualityGateRejectedFrames;
                            if(debugDynamicLBA)
                            {
                                std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
                                          << " instance_observation_quality_gate"
                                          << " action="
                                          << ((hardRejectByQualityGate ||
                                               rejectByStrictRgbdDepthBackedGeometry) ?
                                              "reject" : "diagnostic_reject")
                                          << " instance_id=" << instanceId
                                          << " frame_id=" << frameId
                                          << " records=" << instanceFrameGateStats.records
                                          << " chi2_mean=" << instanceFrameGateStats.MeanChi2()
                                          << " chi2_max=" << instanceFrameGateStats.chi2Max
                                          << " high_chi2_ratio=" << highChi2Ratio
                                          << " depth_failure_ratio=" << depthFailureRatio
                                          << " recon_error_mean=" << instanceFrameGateStats.MeanReconError()
                                          << " recon_error_max=" << instanceFrameGateStats.reconErrorMax
                                          << " pose_source=" << instancePoseInitSource
                                          << " centroid_chi2_mean=" << centroidGateStats.MeanChi2()
                                          << " centroid_recon_error_mean="
                                          << centroidGateStats.MeanReconError()
                                          << " state_chain_available="
                                          << (hasStateChainPoseEstimate ? 1 : 0)
                                          << " state_chain_source=" << stateChainPoseSource
                                          << " state_chain_records="
                                          << stateChainGateStats.records
                                          << " state_chain_chi2_mean="
                                          << stateChainGateStats.MeanChi2()
                                          << " state_chain_recon_error_mean="
                                          << stateChainGateStats.MeanReconError()
                                          << " max_mean_chi2=" << GetBackendInstanceGateMaxMeanChi2()
                                          << " max_high_chi2_ratio=" << GetBackendInstanceGateMaxHighChi2Ratio()
                                          << " max_depth_failure_ratio=" << GetBackendInstanceGateMaxDepthFailureRatio()
                                          << " max_recon_error=" << GetBackendInstanceGateMaxReconError()
                                          << " hard_reject=" << (hardRejectByQualityGate ? 1 : 0)
                                          << " strict_rgbd_depth_backed_geometry_gate="
                                          << (strictRgbdDepthBackedGeometryGate ? 1 : 0)
                                          << " strict_rgbd_depth_backed_reject="
                                          << (rejectByStrictRgbdDepthBackedGeometry ? 1 : 0)
                                          << std::endl;
                            }
                            if(hardRejectByQualityGate ||
                               rejectByStrictRgbdDepthBackedGeometry)
                                continue;
                        }
                        else
                        {
                            ++nInstanceObservationQualityGateAccepted;
                            ++mInstanceConstraintDebug[instanceId].qualityGateAcceptedFrames;
                        }
                    }
                    else
                    {
                        ++nInstanceObservationQualityGateSparse;
                        ++mInstanceConstraintDebug[instanceId].qualityGateSparseFrames;
                    }
                }

                g2o::VertexSE3Expmap* vInstancePose = new g2o::VertexSE3Expmap();
                const g2o::SE3Quat instancePoseInit(
                    instancePoseEstimate.unit_quaternion().cast<double>(),
                    instancePoseEstimate.translation().cast<double>());
                const int instancePoseVertexId = nextInstancePoseId;
                vInstancePose->setEstimate(instancePoseInit);
                vInstancePose->setId(instancePoseVertexId);
                vInstancePose->setFixed(false);
                optimizer.addVertex(vInstancePose);
                mInstancePoseVertexIds[InstanceFrameKey(instanceId, frameId)] = instancePoseVertexId;
                ++nextInstancePoseId;
                ++nInstancePoseVertices;
                if(debugDynamicLBA)
                {
                    std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
                              << " instance_pose_vertex_init"
                              << " instance_id=" << instanceId
                              << " frame_id=" << frameId
                              << " source=" << instancePoseInitSource
                              << " translation_norm="
                              << instancePoseEstimate.translation().norm()
                              << " rotation_deg="
                              << RotationAngleDeg(instancePoseEstimate.rotationMatrix())
                              << " structure_points=" << vValidPoints.size()
                              << " translation_support=" << translationSupport
                              << std::endl;
                }

                if(enableInstancePoseMotionPrior &&
                   pInstanceForPoseInit &&
                   instancePoseMotionPriorInvSigma2 > 0.0)
                {
                    std::map<unsigned long, std::vector<MapPoint*> >::iterator itPrevFrame =
                        itFramePoints;
                    bool foundPreviousInstancePose = false;
                    unsigned long prevFrameId = 0;
                    int prevInstancePoseVertexId = -1;
                    while(itPrevFrame != itInstance->second.begin())
                    {
                        --itPrevFrame;
                        const int frameGap =
                            frameId > itPrevFrame->first ?
                            static_cast<int>(frameId - itPrevFrame->first) : 0;
                        if(frameGap <= 0)
                            continue;

                        const int maxFrameGap = GetInstancePoseStatePredictionMaxFrameGap();
                        if(maxFrameGap > 0 && frameGap > maxFrameGap)
                            break;

                        const std::map<InstanceFrameKey, int>::const_iterator itPrevPose =
                            mInstancePoseVertexIds.find(
                                InstanceFrameKey(instanceId, itPrevFrame->first));
                        if(itPrevPose == mInstancePoseVertexIds.end())
                            continue;

                        foundPreviousInstancePose = true;
                        prevFrameId = itPrevFrame->first;
                        prevInstancePoseVertexId = itPrevPose->second;
                        break;
                    }

                    if(foundPreviousInstancePose)
                    {
                        const int frameGap =
                            static_cast<int>(frameId - prevFrameId);
                        Sophus::SE3f motionPrior;
                        std::string motionPriorSource = "none";
                        if(GetInstanceMotionPriorForFrame(pInstanceForPoseInit,
                                                          frameId,
                                                          frameGap,
                                                          motionPrior,
                                                          motionPriorSource))
                        {
                            g2o::EdgeSE3* eMotionPrior = new g2o::EdgeSE3();
                            eMotionPrior->setVertex(
                                0, optimizer.vertex(instancePoseVertexId));
                            eMotionPrior->setVertex(
                                1, optimizer.vertex(prevInstancePoseVertexId));
                            const Sophus::SE3f inverseMotionPrior =
                                motionPrior.inverse();
                            eMotionPrior->setMeasurement(
                                g2o::SE3Quat(
                                    inverseMotionPrior.unit_quaternion().cast<double>(),
                                    inverseMotionPrior.translation().cast<double>()));
                            eMotionPrior->setInformation(
                                Eigen::Matrix<double, 6, 6>::Identity() *
                                instancePoseMotionPriorInvSigma2);
                            g2o::RobustKernelHuber* rkMotion =
                                new g2o::RobustKernelHuber;
                            eMotionPrior->setRobustKernel(rkMotion);
                            rkMotion->setDelta(1.0);
                            optimizer.addEdge(eMotionPrior);
                            ++nEdges;
                            ++nInstancePoseMotionPriorEdges;
                            if(debugDynamicLBA)
                            {
                                std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
                                          << " instance_pose_motion_prior"
                                          << " instance_id=" << instanceId
                                          << " prev_frame_id=" << prevFrameId
                                          << " frame_id=" << frameId
                                          << " frame_gap=" << frameGap
                                          << " source=" << motionPriorSource
                                          << " translation_norm="
                                          << motionPrior.translation().norm()
                                          << " rotation_deg="
                                          << RotationAngleDeg(motionPrior.rotationMatrix())
                                          << " inv_sigma2="
                                          << instancePoseMotionPriorInvSigma2
                                          << std::endl;
                            }
                        }
                    }
                }

                for(size_t i = 0; i < vPointVertexIds.size(); ++i)
                {
                    g2o::VertexSBAPointXYZ* vPoint =
                        static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(vPointVertexIds[i]));
                    if(!vPoint)
                        continue;

                    EdgeInstancePointStructure* e = new EdgeInstancePointStructure();
                    e->setVertex(0, optimizer.vertex(vPointVertexIds[i]));
                    e->setVertex(1, vInstancePose);
                    const std::pair<int, MapPoint*> key(instanceId, vValidPoints[i]);
                    e->setMeasurement(mInstanceCanonicalLocalPoints[key]);
                    const double dynamicWeight = GetDynamicObservationWeight(pCurrentMap, vValidPoints[i], frameId);
                    e->setInformation(Eigen::Matrix3d::Identity() *
                                      instanceStructureInvSigma2 * dynamicWeight);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(1.0);

                    optimizer.addEdge(e);
                    ++nEdges;
                    ++nInstanceStructureEdges;
                }

                const std::map<unsigned long, std::map<MapPoint*, int> >::const_iterator itLeftMap =
                    mmFrameLeftIndices.find(frameId);
                if(itLeftMap != mmFrameLeftIndices.end())
                {
                    const std::map<unsigned long, ImageFrameHandle>::const_iterator itFrameHandle =
                        mImageFrames.find(frameId);
                    if(itFrameHandle != mImageFrames.end())
                    {
                        const ImageFrameHandle& frame = itFrameHandle->second;
                        for(size_t i = 0; i < vValidPoints.size(); ++i)
                        {
                            MapPoint* pMP = vValidPoints[i];
                            const std::map<MapPoint*, int>::const_iterator itLeft =
                                itLeftMap->second.find(pMP);
                            if(itLeft == itLeftMap->second.end())
                                continue;

                            const int leftIndex = itLeft->second;
                            if(leftIndex < 0)
                                continue;
                            if(GetFrameRightCoordinates(frame)[leftIndex] >= 0)
                                continue;

                            const std::pair<int, MapPoint*> key(instanceId, pMP);
                            const std::map<std::pair<int, MapPoint*>, Eigen::Vector3d>::const_iterator
                                itLocalPoint = mInstanceCanonicalLocalPoints.find(key);
                            if(itLocalPoint == mInstanceCanonicalLocalPoints.end())
                                continue;

                            addPanopticInstanceProjectionEdge(instancePoseVertexId,
                                                              itLocalPoint->second,
                                                              frame,
                                                              pMP,
                                                              leftIndex,
                                                              instanceId);
                        }
                    }
                }
            }
        }
    }

    if(debugDynamicLBA)
    {
        size_t nCurrentIntervalSnapshots = 0;
        size_t nCurrentIntervalActiveFrames = 0;
        std::vector<WindowFrameSnapshot>& vCurrentIntervalFrames = pKF->GetMutableImageFrameWindow();
        for(size_t i = 0; i < vCurrentIntervalFrames.size(); ++i)
        {
            if(!vCurrentIntervalFrames[i].HasPose())
                continue;
            ++nCurrentIntervalSnapshots;
            if(mImageFrames.count(vCurrentIntervalFrames[i].mnFrameId))
                ++nCurrentIntervalActiveFrames;
        }

        std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
                  << " reprojection_graph_ready opt_points=" << nPoints
                  << " total_edges=" << nEdges
                  << " image_frames=" << mImageFrames.size()
                  << " tau_image_frames=" << nTauImageFrames
                  << " window_frame_vertices=" << nWindowFrameVertices
                  << " fixed_window_frame_vertices=" << nFixedWindowFrameVertices
                  << " current_interval_snapshots=" << nCurrentIntervalSnapshots
                  << " current_interval_active_frames=" << nCurrentIntervalActiveFrames
                  << " instance_pose_vertices=" << nInstancePoseVertices
                  << " instance_structure_edges=" << nInstanceStructureEdges
                  << " instance_structure_inv_sigma2=" << instanceStructureInvSigma2
                  << " instance_pose_structure_coupling=" << enableInstanceStructureProxy
                  << " strict_eq16_instance_structure_proxy=" << enableInstanceStructureProxy
                  << " centroid_motion_fallback=" << enableCentroidMotionFallback
                  << " instance_motion_translation_only=" << translationOnlyInstanceMotionWriteback
                  << " backend_mature_motion_gate=" << (backendMatureMotionGate ? 1 : 0)
                  << " backend_mature_max_translation=" << backendMatureMaxTranslation
                  << " backend_mature_max_rotation_deg=" << backendMatureMaxRotationDeg
                  << " strict_eq16_image_window=" << strictEq16ImageWindow
                  << std::endl;
    }

    for(std::map<int, std::map<unsigned long, std::vector<MapPoint*> > >::iterator itInstance = mmInstanceFramePoints.begin();
        itInstance != mmInstanceFramePoints.end(); ++itInstance)
    {
        const int instanceId = itInstance->first;
        std::map<unsigned long, std::vector<MapPoint*> >& mFramePoints = itInstance->second;
        InstanceConstraintDebug& instanceDebug = mInstanceConstraintDebug[instanceId];
        Instance* pDebugInstance = pCurrentMap->GetInstance(instanceId);
        if(pDebugInstance)
        {
            instanceDebug.semanticLabel = pDebugInstance->GetSemanticLabel();
            instanceDebug.backendMature = HasMatureInstanceBackendState(pDebugInstance, pKF) ? 1 : 0;
        }
        const std::map<unsigned long, std::vector<MapPoint*> >::const_iterator itCurrentDebug =
            mFramePoints.find(pKF->mnFrameId);
        if(itCurrentDebug != mFramePoints.end())
            instanceDebug.currentPoints = static_cast<int>(itCurrentDebug->second.size());
        instanceDebug.currentSupportOk =
            (instanceDebug.currentPoints >= kDynamicBackendMinFrameSupport) ? 1 : 0;

        std::vector<unsigned long> vInstanceFrames;
        vInstanceFrames.reserve(mFramePoints.size());
        for(std::map<unsigned long, std::vector<MapPoint*> >::iterator itFrame = mFramePoints.begin();
            itFrame != mFramePoints.end(); ++itFrame)
        {
            if(sRejectedInstanceFramesByQualityGate.count(
                   InstanceFrameKey(instanceId, itFrame->first)) > 0)
            {
                continue;
            }
            vInstanceFrames.push_back(itFrame->first);
            instanceDebug.maxFramePoints =
                std::max(instanceDebug.maxFramePoints,
                         static_cast<int>(itFrame->second.size()));
        }
        std::sort(vInstanceFrames.begin(), vInstanceFrames.end(),
                  [&](const unsigned long lhs, const unsigned long rhs)
                  {
                      return SortImageFramesByTimestamp(mImageFrames[lhs], mImageFrames[rhs]);
                  });
        instanceDebug.windowFrames = static_cast<int>(vInstanceFrames.size());
        instanceDebug.windowSupportOk =
            (instanceDebug.maxFramePoints >= kDynamicBackendMinFrameSupport) ? 1 : 0;
        mInstanceBackendMature[instanceId] = (instanceDebug.windowSupportOk != 0);

        if(!mInstanceBackendMature[instanceId])
            continue;

        for(size_t frameIdx = 0; frameIdx < vInstanceFrames.size(); ++frameIdx)
        {
            const unsigned long frameId = vInstanceFrames[frameIdx];
            const std::map<unsigned long, ImageFrameHandle>::iterator itFrameHandle = mImageFrames.find(frameId);
            if(itFrameHandle == mImageFrames.end())
                continue;
            const ImageFrameHandle& frame = itFrameHandle->second;

            const std::map<unsigned long, std::map<MapPoint*, int> >::iterator itLeftMap = mmFrameLeftIndices.find(frameId);
            if(itLeftMap == mmFrameLeftIndices.end())
                continue;

            std::vector<MapPoint*> vPointsWithLeftObs;
            const std::vector<MapPoint*>& vFramePoints = mFramePoints[frameId];
            for(size_t i = 0; i < vFramePoints.size(); ++i)
            {
                if(itLeftMap->second.count(vFramePoints[i]))
                    vPointsWithLeftObs.push_back(vFramePoints[i]);
            }
            if(!vPointsWithLeftObs.empty())
            {
                ++instanceDebug.shapeCandidateFrames;
                instanceDebug.shapeCandidatePoints += static_cast<int>(vPointsWithLeftObs.size());
            }

            const std::vector<std::tuple<MapPoint*, MapPoint*, MapPoint*> > vTriplets =
                SampleTripletsInFrame(vPointsWithLeftObs, itLeftMap->second, frame, kMaxShapeTripletsPerFrame);
            instanceDebug.shapeTriplets += static_cast<int>(vTriplets.size());

            if(!disableShapeEdges)
            {
                std::vector<MapPoint*> vShapePoints;
                std::map<MapPoint*, size_t> mShapePointLocalIds;
                std::vector<std::tuple<MapPoint*, MapPoint*, MapPoint*> > vValidTriplets;
                std::vector<Vector6d> vTripletMeasurements;
                std::vector<double> vTripletInvSigma2;
                std::vector<DynamicEdgeObservation> vShapeObservations;

                for(size_t i = 0; i < vTriplets.size(); ++i)
                {
                    MapPoint* pA = std::get<0>(vTriplets[i]);
                    MapPoint* pB = std::get<1>(vTriplets[i]);
                    MapPoint* pC = std::get<2>(vTriplets[i]);
                    const DynamicVertexKey keyA(pA, frameId);
                    const DynamicVertexKey keyB(pB, frameId);
                    const DynamicVertexKey keyC(pC, frameId);
                    if(mThingVertexIds.count(keyA) == 0 ||
                       mThingVertexIds.count(keyB) == 0 ||
                       mThingVertexIds.count(keyC) == 0)
                        continue;

                    const std::array<MapPoint*, 3> tripletPoints = {pA, pB, pC};
                    for(size_t pointIdx = 0; pointIdx < tripletPoints.size(); ++pointIdx)
                    {
                        MapPoint* pPoint = tripletPoints[pointIdx];
                        if(mShapePointLocalIds.count(pPoint))
                            continue;
                        const size_t localId = vShapePoints.size();
                        mShapePointLocalIds[pPoint] = localId;
                        vShapePoints.push_back(pPoint);
                        vShapeObservations.push_back(DynamicEdgeObservation(pPoint, frameId));
                    }

                    vValidTriplets.push_back(vTriplets[i]);
                    vTripletMeasurements.push_back(
                        BuildTripletMeasurement(itLeftMap->second, frame, pA, pB, pC));
                    vTripletInvSigma2.push_back(
                        ComputeTripletInvSigma2(itLeftMap->second, frame, pA, pB, pC));
                }
                instanceDebug.shapeValidTriplets += static_cast<int>(vValidTriplets.size());

                if(!vValidTriplets.empty() && vShapePoints.size() >= 3)
                {
                    const double instanceShapeScale =
                        GetInstanceShapeFactorScale(pCurrentMap, instanceId);
                    if(useStrictShapeTripletFactors)
                    {
                        for(size_t i = 0; i < vValidTriplets.size(); ++i)
                        {
                            MapPoint* pA = std::get<0>(vValidTriplets[i]);
                            MapPoint* pB = std::get<1>(vValidTriplets[i]);
                            MapPoint* pC = std::get<2>(vValidTriplets[i]);
                            const DynamicVertexKey keyA(pA, frameId);
                            const DynamicVertexKey keyB(pB, frameId);
                            const DynamicVertexKey keyC(pC, frameId);
                            if(mThingVertexIds.count(keyA) == 0 ||
                               mThingVertexIds.count(keyB) == 0 ||
                               mThingVertexIds.count(keyC) == 0)
                            {
                                continue;
                            }

                            EdgeShapeScaleTriplet* e = new EdgeShapeScaleTriplet();
                            e->setVertex(0, optimizer.vertex(mThingVertexIds[keyA]));
                            e->setVertex(1, optimizer.vertex(mThingVertexIds[keyB]));
                            e->setVertex(2, optimizer.vertex(mThingVertexIds[keyC]));
                            e->setMeasurement(vTripletMeasurements[i]);

                            std::vector<DynamicEdgeObservation> vTripletObservations;
                            vTripletObservations.reserve(3);
                            vTripletObservations.push_back(DynamicEdgeObservation(pA, frameId));
                            vTripletObservations.push_back(DynamicEdgeObservation(pB, frameId));
                            vTripletObservations.push_back(DynamicEdgeObservation(pC, frameId));
                            const double dynamicWeight =
                                GetMeanDynamicObservationWeight(pCurrentMap, vTripletObservations);
                            const double baseInvSigma2 = vTripletInvSigma2[i];
                            e->setInformation(Eigen::Matrix2d::Identity() *
                                              baseInvSigma2 * dynamicWeight * shapeFactorWeight *
                                              instanceShapeScale);

                            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                            e->setRobustKernel(rk);
                            rk->setDelta(thHuberShape);

                            optimizer.addEdge(e);
                            ShapeScaleEdgeRecord record;
                            record.tripletEdge = e;
                            record.instanceId = instanceId;
                            record.baseInvSigma2 = baseInvSigma2;
                            record.factorScale = instanceShapeScale;
                            record.logicalTerms = 1;
                            record.observations.swap(vTripletObservations);
                            vShapeScaleEdgeRecords.push_back(record);
                            ++nEdges;
                            ++nShapeEdges;
                            ++instanceDebug.shapeEdges;
                        }
                    }
                    else
                    {
                        EdgeShapeScaleFrame* e = new EdgeShapeScaleFrame(vShapePoints.size());
                        for(size_t pointIdx = 0; pointIdx < vShapePoints.size(); ++pointIdx)
                        {
                            const DynamicVertexKey key(vShapePoints[pointIdx], frameId);
                            e->setVertex(pointIdx, optimizer.vertex(mThingVertexIds[key]));
                        }

                        for(size_t i = 0; i < vValidTriplets.size(); ++i)
                        {
                            MapPoint* pA = std::get<0>(vValidTriplets[i]);
                            MapPoint* pB = std::get<1>(vValidTriplets[i]);
                            MapPoint* pC = std::get<2>(vValidTriplets[i]);
                            e->AddTriplet(mShapePointLocalIds[pA],
                                          mShapePointLocalIds[pB],
                                          mShapePointLocalIds[pC],
                                          vTripletMeasurements[i],
                                          vTripletInvSigma2[i]);
                        }

                        const double baseInvSigma2 = e->MeanInvSigma2();
                        const double dynamicWeight =
                            GetMeanDynamicObservationWeight(pCurrentMap, vShapeObservations);
                        e->setInformation(Eigen::Matrix2d::Identity() *
                                          baseInvSigma2 * dynamicWeight * shapeFactorWeight *
                                          instanceShapeScale);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberShape);

                        optimizer.addEdge(e);
                        ShapeScaleEdgeRecord record;
                        record.frameEdge = e;
                        record.instanceId = instanceId;
                        record.baseInvSigma2 = baseInvSigma2;
                        record.factorScale = instanceShapeScale;
                        record.logicalTerms = vValidTriplets.size();
                        record.observations.swap(vShapeObservations);
                        vShapeScaleEdgeRecords.push_back(record);
                        ++nEdges;
                        nShapeEdges += static_cast<int>(vValidTriplets.size());
                        instanceDebug.shapeEdges += static_cast<int>(vValidTriplets.size());
                    }
                }
            }
        }

        for(size_t ta = 0; ta < vInstanceFrames.size(); ++ta)
        {
            for(size_t tb = ta + 1; tb < vInstanceFrames.size(); ++tb)
            {
                if(kMaxRigidityFrameGap > 0 && tb - ta > kMaxRigidityFrameGap)
                    break;

                const unsigned long prevFrameId = vInstanceFrames[ta];
                const unsigned long currFrameId = vInstanceFrames[tb];
                const std::vector<MapPoint*>& vPrevPoints = mFramePoints[prevFrameId];
                const std::vector<MapPoint*>& vCurrPoints = mFramePoints[currFrameId];
                std::set<MapPoint*> sCurrPoints(vCurrPoints.begin(), vCurrPoints.end());

                std::vector<MapPoint*> vCommonPoints;
                for(size_t j = 0; j < vPrevPoints.size(); ++j)
                {
                    if(sCurrPoints.count(vPrevPoints[j]))
                    {
                        vCommonPoints.push_back(vPrevPoints[j]);
                    }
                }
                ++instanceDebug.rigidityFramePairsTested;
                if(!vCommonPoints.empty())
                {
                    ++instanceDebug.rigidityCandidateFramePairs;
                    instanceDebug.rigidityCommonPoints += static_cast<int>(vCommonPoints.size());
                }
                if(vCommonPoints.size() < 2)
                    ++instanceDebug.rigidityInsufficientCommonPairs;

                const double frameInterval = std::max(1.0, static_cast<double>(tb - ta));
                const double invSigma2 = 1.0 / (frameInterval * frameInterval);

                if(!disableRigidityEdges && vCommonPoints.size() >= 2)
                {
                    const size_t totalRigidityPairs =
                        (vCommonPoints.size() * (vCommonPoints.size() - 1)) / 2;
                    if(kMaxRigidityPairsPerNeighbor > 0 &&
                       totalRigidityPairs > kMaxRigidityPairsPerNeighbor)
                    {
                        const std::vector<std::pair<MapPoint*, MapPoint*> > vPairs =
                            useQualityGatedRigidityPairSampler ?
                            SampleQualityWeightedPairsInFrame(
                                pCurrentMap,
                                vCommonPoints,
                                prevFrameId,
                                currFrameId,
                                kMaxRigidityPairsPerNeighbor,
                                kMaxRigidityPairsPerPoint,
                                useQualityCoverageRigidityPairSampler) :
                            SamplePairsInFrame(vCommonPoints, kMaxRigidityPairsPerNeighbor);
                        for(size_t j = 0; j < vPairs.size(); ++j)
                        {
                            MapPoint* pA = vPairs[j].first;
                            MapPoint* pB = vPairs[j].second;
                            const DynamicVertexKey keyPrevA(pA, prevFrameId);
                            const DynamicVertexKey keyPrevB(pB, prevFrameId);
                            const DynamicVertexKey keyCurrA(pA, currFrameId);
                            const DynamicVertexKey keyCurrB(pB, currFrameId);
                            if(mThingVertexIds.count(keyPrevA) == 0 ||
                               mThingVertexIds.count(keyPrevB) == 0 ||
                               mThingVertexIds.count(keyCurrA) == 0 ||
                               mThingVertexIds.count(keyCurrB) == 0)
                            {
                                ++instanceDebug.rigidityMissingVertexPairs;
                                continue;
                            }

                            EdgeRigidityPair* e = new EdgeRigidityPair();
                            e->setVertex(0, optimizer.vertex(mThingVertexIds[keyPrevA]));
                            e->setVertex(1, optimizer.vertex(mThingVertexIds[keyPrevB]));
                            e->setVertex(2, optimizer.vertex(mThingVertexIds[keyCurrA]));
                            e->setVertex(3, optimizer.vertex(mThingVertexIds[keyCurrB]));
                            e->setMeasurement(0.0);

                            std::vector<DynamicEdgeObservation> vRigidityObservations;
                            vRigidityObservations.reserve(4);
                            vRigidityObservations.push_back(DynamicEdgeObservation(pA, prevFrameId));
                            vRigidityObservations.push_back(DynamicEdgeObservation(pB, prevFrameId));
                            vRigidityObservations.push_back(DynamicEdgeObservation(pA, currFrameId));
                            vRigidityObservations.push_back(DynamicEdgeObservation(pB, currFrameId));
                            const double dynamicWeight =
                                GetMeanDynamicObservationWeight(pCurrentMap, vRigidityObservations);
                            const double instanceRigidityScale =
                                GetInstanceRigidityFactorScale(pCurrentMap, instanceId);
                            e->setInformation(Eigen::Matrix<double,1,1>::Identity() *
                                              invSigma2 * dynamicWeight * rigidityFactorWeight *
                                              instanceRigidityScale);

                            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                            e->setRobustKernel(rk);
                            rk->setDelta(thHuberRigidity);

                            optimizer.addEdge(e);
                            RigidityEdgeRecord record;
                            record.pairEdge = e;
                            record.instanceId = instanceId;
                            record.baseInvSigma2 = invSigma2;
                            record.factorScale = instanceRigidityScale;
                            record.logicalTerms = 1;
                            record.observations.swap(vRigidityObservations);
                            vRigidityEdgeRecords.push_back(record);
                            ++nEdges;
                            ++nRigidityEdges;
                            ++nRigidityPointPairs;
                            ++instanceDebug.rigidityEdges;
                            ++instanceDebug.rigidityPointPairs;
                        }
                    }
                    else
                    {
                        EdgeRigidityFramePair* e = new EdgeRigidityFramePair(vCommonPoints.size());
                        std::vector<DynamicEdgeObservation> vRigidityObservations;
                        vRigidityObservations.reserve(vCommonPoints.size() * 2);
                        for(size_t j = 0; j < vCommonPoints.size(); ++j)
                        {
                            MapPoint* pMP = vCommonPoints[j];
                            const DynamicVertexKey keyPrev(pMP, prevFrameId);
                            const DynamicVertexKey keyCurr(pMP, currFrameId);
                            if(mThingVertexIds.count(keyPrev) == 0 ||
                               mThingVertexIds.count(keyCurr) == 0)
                            {
                                ++instanceDebug.rigidityMissingVertexPairs;
                                delete e;
                                e = NULL;
                                break;
                            }

                            e->setVertex(j, optimizer.vertex(mThingVertexIds[keyPrev]));
                            e->setVertex(vCommonPoints.size() + j,
                                         optimizer.vertex(mThingVertexIds[keyCurr]));
                            vRigidityObservations.push_back(DynamicEdgeObservation(pMP, prevFrameId));
                            vRigidityObservations.push_back(DynamicEdgeObservation(pMP, currFrameId));
                        }

                        if(e)
                        {
                            e->setMeasurement(0.0);
                            const double dynamicWeight =
                                GetMeanDynamicObservationWeight(pCurrentMap, vRigidityObservations);
                            const double instanceRigidityScale =
                                GetInstanceRigidityFactorScale(pCurrentMap, instanceId);
                            e->setInformation(Eigen::Matrix<double,1,1>::Identity() *
                                              invSigma2 * dynamicWeight * rigidityFactorWeight *
                                              instanceRigidityScale);

                            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                            e->setRobustKernel(rk);
                            rk->setDelta(thHuberRigidity);

                            optimizer.addEdge(e);
                            RigidityEdgeRecord record;
                            record.frameEdge = e;
                            record.instanceId = instanceId;
                            record.baseInvSigma2 = invSigma2;
                            record.factorScale = instanceRigidityScale;
                            record.logicalTerms = totalRigidityPairs;
                            record.observations.swap(vRigidityObservations);
                            vRigidityEdgeRecords.push_back(record);
                            ++nEdges;
                            ++nRigidityEdges;
                            nRigidityPointPairs += static_cast<int>(totalRigidityPairs);
                            ++instanceDebug.rigidityEdges;
                            instanceDebug.rigidityPointPairs += static_cast<int>(totalRigidityPairs);
                        }
                    }
                }
            }
        }

        if(mInstances.count(instanceId) == 0)
            mInstances[instanceId] = pCurrentMap->GetInstance(instanceId);
    }

    if(debugDynamicLBA)
    {
        for(std::map<int, std::map<unsigned long, std::vector<MapPoint*> > >::const_iterator itInstance =
                mmInstanceFramePoints.begin();
            itInstance != mmInstanceFramePoints.end(); ++itInstance)
        {
            InstanceConstraintDebug& dbg = mInstanceConstraintDebug[itInstance->first];
            std::map<MapPoint*, int> mPointTrackLengths;
            for(std::map<unsigned long, std::vector<MapPoint*> >::const_iterator itFrame =
                    itInstance->second.begin();
                itFrame != itInstance->second.end(); ++itFrame)
            {
                for(size_t pointIdx = 0; pointIdx < itFrame->second.size(); ++pointIdx)
                    ++mPointTrackLengths[itFrame->second[pointIdx]];
            }

            for(std::map<MapPoint*, int>::const_iterator itPoint = mPointTrackLengths.begin();
                itPoint != mPointTrackLengths.end(); ++itPoint)
            {
                dbg.maxPointTrackLen = std::max(dbg.maxPointTrackLen, itPoint->second);
                if(itPoint->second <= 1)
                    ++dbg.pointTrackLen1;
                else if(itPoint->second == 2)
                    ++dbg.pointTrackLen2;
                else
                    ++dbg.pointTrackLen3Plus;
            }
        }

        std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
                  << " dynamic_constraints_ready total_edges=" << nEdges
                  << " module8_profile=" << module8Profile
                  << " panoptic_instance_edges=" << nPanopticInstanceEdges
                  << " shape_edges=" << nShapeEdges
                  << " rigidity_edges=" << nRigidityEdges
                  << " rigidity_point_pairs=" << nRigidityPointPairs
                  << " dynamic_priors=" << nDynamicPriorEdges
                  << " instance_pose_motion_priors=" << nInstancePoseMotionPriorEdges
                  << " instance_pose_vertices=" << nInstancePoseVertices
                  << " instance_structure_edges=" << nInstanceStructureEdges
                  << " instance_structure_inv_sigma2=" << instanceStructureInvSigma2
                  << " shape_triplet_sample_limit=" << kMaxShapeTripletsPerFrame
                  << " shape_factor_mode="
                  << (useStrictShapeTripletFactors ? "per_triplet_eq13" : "aggregated_frame")
                  << " rigidity_pair_sample_limit=" << kMaxRigidityPairsPerNeighbor
                  << " rigidity_pair_max_degree=" << kMaxRigidityPairsPerPoint
                  << " rigidity_pair_min_quality=" << kRigidityPairMinQuality
                  << " rigidity_pair_min_known_obs=" << kRigidityPairMinKnownObservations
                  << " panoptic_factor_weight=" << panopticFactorWeight
                  << " shape_factor_weight=" << shapeFactorWeight
                  << " rigidity_factor_weight=" << rigidityFactorWeight
                  << " articulated_semantic_label=" << GetArticulatedSemanticLabel()
                  << " articulated_shape_factor_scale=" << GetArticulatedShapeFactorScale()
                  << " articulated_rigidity_factor_scale=" << GetArticulatedRigidityFactorScale()
                  << " shape_huber_delta=" << thHuberShape
                  << " rigidity_huber_delta=" << thHuberRigidity
                  << " rigidity_pair_sampler="
                  << GetRigidityPairSamplerName(useQualityGatedRigidityPairSampler,
                                                useQualityCoverageRigidityPairSampler,
                                                kMaxRigidityPairsPerNeighbor)
                  << " rigidity_max_frame_gap=" << kMaxRigidityFrameGap
                  << " dynamic_backend_min_frame_points=" << kDynamicBackendMinFrameSupport
                  << " rigidity_frobenius_residual=1"
                  << " rigidity_interval_unit=frame_index"
                  << " instance_pose_structure_coupling=" << enableInstanceStructureProxy
                  << " strict_eq16_instance_structure_proxy=" << enableInstanceStructureProxy
                  << " centroid_motion_fallback=" << enableCentroidMotionFallback
                  << " instance_motion_translation_only=" << translationOnlyInstanceMotionWriteback
                  << " instance_pose_motion_prior=" << (enableInstancePoseMotionPrior ? 1 : 0)
                  << " instance_pose_motion_prior_inv_sigma2="
                  << instancePoseMotionPriorInvSigma2
                  << " backend_mature_motion_gate=" << (backendMatureMotionGate ? 1 : 0)
                  << " backend_mature_max_translation=" << backendMatureMaxTranslation
                  << " backend_mature_max_rotation_deg=" << backendMatureMaxRotationDeg
                  << " strict_eq16_image_window=" << strictEq16ImageWindow
                  << " reweighted_observations=" << nDynamicReweightedObservations
                  << " backend_outlier_marks=" << nBackendOutlierMarks
                  << " instance_quality_gate=" << (EnableBackendInstanceObservationQualityGate() ? 1 : 0)
                  << " instance_quality_gate_hard_reject="
                  << (EnableBackendInstanceObservationQualityGateHardReject() ? 1 : 0)
                  << " instance_quality_gate_accepted=" << nInstanceObservationQualityGateAccepted
                  << " instance_quality_gate_rejected=" << nInstanceObservationQualityGateRejected
                  << " instance_quality_gate_sparse=" << nInstanceObservationQualityGateSparse
                  << std::endl;
        std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
                  << " dynamic_vertex_initialization"
                  << " identity=" << dynamicInitDebug.identity
                  << " keyframe_motion_prior=" << dynamicInitDebug.keyframeMotionPrior
                  << " snapshot_motion_prior=" << dynamicInitDebug.snapshotMotionPrior
                  << " velocity_fallback=" << dynamicInitDebug.velocityFallback
                  << " observation_point_world=" << dynamicInitDebug.observationPointWorld
                  << " observation_point_world_missing=" << dynamicInitDebug.observationPointWorldMissing
                  << " observation_point_world_delta_mean="
                  << (dynamicInitDebug.observationPointWorld > 0 ?
                      dynamicInitDebug.observationPointWorldDeltaSum /
                          static_cast<double>(dynamicInitDebug.observationPointWorld) : 0.0)
                  << std::endl;
        std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
                  << " dynamic_observation_summary"
                  << " frames_with_dynamic_observations=" << dynamicObservationDebug.framesWithDynamicObservations
                  << " dynamic_observations=" << dynamicObservationDebug.dynamicObservations
                  << " accepted_dynamic_observations=" << dynamicObservationDebug.acceptedDynamicObservations
                  << " rejected_dynamic_null_point=" << dynamicObservationDebug.rejectedDynamicNullPoint
                  << " rejected_dynamic_bad_or_map=" << dynamicObservationDebug.rejectedDynamicBadOrMap
                  << " rejected_dynamic_invalid_instance=" << dynamicObservationDebug.rejectedDynamicInvalidInstance
                  << " rejected_dynamic_feature_range=" << dynamicObservationDebug.rejectedDynamicFeatureRange
                  << " rejected_dynamic_right_feature=" << dynamicObservationDebug.rejectedDynamicRightFeature
                  << " pointworld_refinement="
                  << (EnableDynamicObservationPointWorldRefinement() ? 1 : 0)
                  << " pointworld_checked=" << dynamicObservationDebug.pointWorldConsistencyChecked
                  << " pointworld_bad=" << dynamicObservationDebug.pointWorldConsistencyBad
                  << " pointworld_refined=" << dynamicObservationDebug.pointWorldConsistencyRefined
                  << " pointworld_negative_depth="
                  << dynamicObservationDebug.pointWorldConsistencyNegativeDepth
                  << " pointworld_reproj_mean="
                  << (dynamicObservationDebug.pointWorldConsistencyChecked > 0 ?
                      dynamicObservationDebug.pointWorldReprojectionErrorSum /
                          static_cast<double>(dynamicObservationDebug.pointWorldConsistencyChecked) : 0.0)
                  << " pointworld_reproj_max="
                  << dynamicObservationDebug.pointWorldReprojectionErrorMax
                  << std::endl;
        for(std::map<int, InstanceConstraintDebug>::const_iterator itDebug = mInstanceConstraintDebug.begin();
            itDebug != mInstanceConstraintDebug.end(); ++itDebug)
        {
            const InstanceConstraintDebug& dbg = itDebug->second;
            std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
                      << " instance_constraints"
                      << " instance_id=" << itDebug->first
                      << " semantic=" << dbg.semanticLabel
                      << " current_points=" << dbg.currentPoints
                      << " max_frame_points=" << dbg.maxFramePoints
                      << " window_frames=" << dbg.windowFrames
                      << " backend_mature=" << dbg.backendMature
                      << " current_support_ok=" << dbg.currentSupportOk
                      << " window_support_ok=" << dbg.windowSupportOk
                      << " shape_candidate_frames=" << dbg.shapeCandidateFrames
                      << " shape_candidate_points=" << dbg.shapeCandidatePoints
                      << " shape_triplets=" << dbg.shapeTriplets
                      << " shape_valid_triplets=" << dbg.shapeValidTriplets
                      << " rigidity_candidate_frame_pairs=" << dbg.rigidityCandidateFramePairs
                      << " rigidity_common_points=" << dbg.rigidityCommonPoints
                      << " rigidity_frame_pairs_tested=" << dbg.rigidityFramePairsTested
                      << " rigidity_insufficient_common_pairs=" << dbg.rigidityInsufficientCommonPairs
                      << " rigidity_missing_vertex_pairs=" << dbg.rigidityMissingVertexPairs
                      << " shape_edges=" << dbg.shapeEdges
                      << " rigidity_edges=" << dbg.rigidityEdges
                      << " rigidity_point_pairs=" << dbg.rigidityPointPairs
                      << " dynamic_observations=" << dbg.dynamicObservations
                      << " accepted_dynamic_observations=" << dbg.acceptedDynamicObservations
                      << " rejected_dynamic_null_point=" << dbg.rejectedDynamicNullPoint
                      << " rejected_dynamic_bad_or_map=" << dbg.rejectedDynamicBadOrMap
                      << " rejected_dynamic_invalid_instance=" << dbg.rejectedDynamicInvalidInstance
                      << " rejected_dynamic_feature_range=" << dbg.rejectedDynamicFeatureRange
                      << " rejected_dynamic_right_feature=" << dbg.rejectedDynamicRightFeature
                      << " pointworld_checked=" << dbg.pointWorldConsistencyChecked
                      << " pointworld_bad=" << dbg.pointWorldConsistencyBad
                      << " pointworld_refined=" << dbg.pointWorldConsistencyRefined
                      << " pointworld_negative_depth="
                      << dbg.pointWorldConsistencyNegativeDepth
                      << " pointworld_reproj_mean="
                      << (dbg.pointWorldConsistencyChecked > 0 ?
                          dbg.pointWorldReprojectionErrorSum /
                              static_cast<double>(dbg.pointWorldConsistencyChecked) : 0.0)
                      << " pointworld_reproj_max="
                      << dbg.pointWorldReprojectionErrorMax
                      << " point_track_len1=" << dbg.pointTrackLen1
                      << " point_track_len2=" << dbg.pointTrackLen2
                      << " point_track_len3plus=" << dbg.pointTrackLen3Plus
                      << " max_point_track_len=" << dbg.maxPointTrackLen
                      << " quality_gate_accepted_frames=" << dbg.qualityGateAcceptedFrames
                      << " quality_gate_rejected_frames=" << dbg.qualityGateRejectedFrames
                      << " quality_gate_sparse_frames=" << dbg.qualityGateSparseFrames
                      << std::endl;
        }
        for(std::map<int, InstanceProjectionDebugStats>::const_iterator itDebug =
                mPanopticInstanceInitialDebug.begin();
            itDebug != mPanopticInstanceInitialDebug.end(); ++itDebug)
        {
            const InstanceProjectionDebugStats& dbg = itDebug->second;
            std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
                      << " panoptic_instance_projection_init"
                      << " instance_id=" << itDebug->first
                      << " records=" << dbg.records
                      << " chi2_mean=" << dbg.MeanChi2()
                      << " chi2_max=" << dbg.chi2Max
                      << " high_chi2=" << dbg.highChi2
                      << " depth_failures=" << dbg.depthFailures
                      << " direct_chi2_mean=" << dbg.MeanDirectChi2()
                      << " direct_chi2_max=" << dbg.directChi2Max
                      << " recon_error_mean=" << dbg.MeanReconError()
                      << " recon_error_max=" << dbg.reconErrorMax
                      << " local_norm_mean=" << dbg.MeanLocalNorm()
                      << " local_norm_max=" << dbg.localNormMax
                      << std::endl;
        }
    }

    auto computeCurrentThingOutliers = [&]() -> std::set<MapPoint*>
    {
        std::set<int> sCurrentInstanceIds;
        for(std::map<int, std::vector<int> >::const_iterator itInstance = mmInstanceCurrentVertexIds.begin();
            itInstance != mmInstanceCurrentVertexIds.end(); ++itInstance)
        {
            sCurrentInstanceIds.insert(itInstance->first);
        }

        const std::map<int, ClassRadiusStats> mClassRadiusStats =
            BuildClassRadiusStats(pCurrentMap, sCurrentInstanceIds);
        std::set<MapPoint*> sCurrentThingOutliers;
        if(disableEq17SizePrior)
            return sCurrentThingOutliers;

        const double alphaOutlier = 2.5;
        const double betaOutlier = 25.0;
        for(std::map<int, std::map<unsigned long, std::vector<MapPoint*> > >::const_iterator itInstance = mmInstanceFramePoints.begin();
            itInstance != mmInstanceFramePoints.end(); ++itInstance)
        {
            const std::map<int, Instance*>::const_iterator itInstanceHandle = mInstances.find(itInstance->first);
            if(itInstanceHandle == mInstances.end() || !itInstanceHandle->second)
                continue;

            Instance* pInstance = itInstanceHandle->second;
            const std::map<int, bool>::const_iterator itMature = mInstanceBackendMature.find(itInstance->first);
            if(itMature == mInstanceBackendMature.end() || !itMature->second)
            {
                if(debugDynamicLBA)
                {
                    std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
                              << " eq17_prior"
                              << " instance_id=" << itInstance->first
                              << " semantic=" << pInstance->GetSemanticLabel()
                              << " available=0 reason=instance_backend_not_mature"
                              << std::endl;
                }
                continue;
            }

            double meanRadius = 0.0;
            double stdRadius = 0.0;
            int sampleCount = 0;
            if(!GetClassRadiusPrior(mClassRadiusStats, pInstance->GetSemanticLabel(), meanRadius, stdRadius, sampleCount))
            {
                if(debugDynamicLBA)
                {
                    std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
                              << " eq17_prior"
                              << " instance_id=" << itInstance->first
                              << " semantic=" << pInstance->GetSemanticLabel()
                              << " available=0 reason=no_class_prior"
                              << std::endl;
                }
                continue;
            }
            if(sampleCount < 3 || stdRadius < 1e-6)
            {
                if(debugDynamicLBA)
                {
                    std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
                              << " eq17_prior"
                              << " instance_id=" << itInstance->first
                              << " semantic=" << pInstance->GetSemanticLabel()
                              << " available=0"
                              << " reason=immature_or_degenerate"
                              << " sample_count=" << sampleCount
                              << " mean_radius=" << meanRadius
                              << " std_radius=" << stdRadius
                              << std::endl;
                }
                continue;
            }

            const std::map<unsigned long, std::vector<MapPoint*> >::const_iterator itCurrFrame =
                itInstance->second.find(pKF->mnFrameId);
            if(itCurrFrame == itInstance->second.end() || itCurrFrame->second.empty())
                continue;

            std::vector<int> vCurrentVertexIds;
            vCurrentVertexIds.reserve(itCurrFrame->second.size());
            for(size_t i = 0; i < itCurrFrame->second.size(); ++i)
            {
                const DynamicVertexKey key(itCurrFrame->second[i], pKF->mnFrameId);
                const std::map<DynamicVertexKey, int>::const_iterator itVertex = mThingVertexIds.find(key);
                if(itVertex != mThingVertexIds.end())
                    vCurrentVertexIds.push_back(itVertex->second);
            }

            if(vCurrentVertexIds.empty())
                continue;

            const Eigen::Vector3d centroid = AveragePointEstimate(optimizer, vCurrentVertexIds);
            const double threshold = (alphaOutlier + std::exp(-(1.0 / betaOutlier) * sampleCount)) * stdRadius;
            int nInstanceOutliers = 0;

            for(size_t i = 0; i < itCurrFrame->second.size(); ++i)
            {
                MapPoint* pMP = itCurrFrame->second[i];
                const DynamicVertexKey key(pMP, pKF->mnFrameId);
                const std::map<DynamicVertexKey, int>::const_iterator itVertex = mThingVertexIds.find(key);
                if(itVertex == mThingVertexIds.end())
                    continue;

                g2o::VertexSBAPointXYZ* vPoint =
                    static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(itVertex->second));
                if(!vPoint)
                    continue;

                const double pointRadius = (vPoint->estimate() - centroid).norm();
                if(std::abs(pointRadius - meanRadius) > threshold)
                {
                    sCurrentThingOutliers.insert(pMP);
                    ++nInstanceOutliers;
                }
            }
            if(debugDynamicLBA)
            {
                std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
                          << " eq17_prior"
                          << " instance_id=" << itInstance->first
                          << " semantic=" << pInstance->GetSemanticLabel()
                          << " available=1"
                          << " current_points=" << itCurrFrame->second.size()
                          << " vertex_points=" << vCurrentVertexIds.size()
                          << " sample_count=" << sampleCount
                          << " mean_radius=" << meanRadius
                          << " std_radius=" << stdRadius
                          << " threshold=" << threshold
                          << " outliers=" << nInstanceOutliers
                          << std::endl;
            }
        }

        return sCurrentThingOutliers;
    };

    auto collectCurrentThingPoints = [&]() -> std::set<MapPoint*>
    {
        std::set<MapPoint*> sCurrentThingPoints;
        for(std::map<int, std::map<unsigned long, std::vector<MapPoint*> > >::const_iterator itInstance =
                mmInstanceFramePoints.begin();
            itInstance != mmInstanceFramePoints.end(); ++itInstance)
        {
            const std::map<unsigned long, std::vector<MapPoint*> >::const_iterator itCurr =
                itInstance->second.find(pKF->mnFrameId);
            if(itCurr == itInstance->second.end())
                continue;

            for(size_t i = 0; i < itCurr->second.size(); ++i)
            {
                MapPoint* pMP = itCurr->second[i];
                if(pMP && !pMP->isBad() && pMP->GetInstanceId() > 0)
                    sCurrentThingPoints.insert(pMP);
            }
        }
        return sCurrentThingPoints;
    };

    struct CurrentThingResidualOutlierStats
    {
        std::set<MapPoint*> points;
        size_t panopticInstance = 0;
        size_t panopticMono = 0;
        size_t body = 0;
        size_t stereo = 0;
    };

    auto collectCurrentThingResidualOutliers = [&]() -> CurrentThingResidualOutlierStats
    {
        CurrentThingResidualOutlierStats stats;

        for(size_t i = 0; i < vPanopticInstanceEdgeRecords.size(); ++i)
        {
            const PanopticInstanceEdgeRecord& record = vPanopticInstanceEdgeRecords[i];
            if(!record.edge || !record.pMP)
                continue;
            if(record.frameId != pKF->mnFrameId)
                continue;

            const bool bOutlier = (record.edge->chi2() > 5.991 || !record.edge->isDepthPositive());
            if(!bOutlier)
                continue;

            stats.points.insert(record.pMP);
            ++stats.panopticInstance;
        }

        for(size_t i = 0, iend = vpEdgesPanopticMono.size(); i < iend; ++i)
        {
            EdgePanopticProjection* e = vpEdgesPanopticMono[i];
            MapPoint* pMP = vpMapPointEdgePanopticMono[i];
            if(!e || !pMP || pMP->isBad() || pMP->GetInstanceId() <= 0)
                continue;
            if(i >= vpEdgeFramePanopticMono.size() || vpEdgeFramePanopticMono[i] != pKF->mnFrameId)
                continue;

            const bool bOutlier = (e->chi2() > 5.991 || !e->isDepthPositive());
            if(!bOutlier)
                continue;

            stats.points.insert(pMP);
            ++stats.panopticMono;
        }

        for(size_t i = 0, iend = vpEdgesBody.size(); i < iend; ++i)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZToBody* e = vpEdgesBody[i];
            MapPoint* pMP = vpMapPointEdgeBody[i];
            if(!e || !pMP || pMP->isBad() || pMP->GetInstanceId() <= 0)
                continue;
            if(i >= vpEdgeFrameBody.size() || vpEdgeFrameBody[i] != pKF->mnFrameId)
                continue;

            const bool bOutlier = (e->chi2() > 5.991 || !e->isDepthPositive());
            if(!bOutlier)
                continue;

            stats.points.insert(pMP);
            ++stats.body;
        }

        for(size_t i = 0, iend = vpEdgesStereo.size(); i < iend; ++i)
        {
            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
            MapPoint* pMP = vpMapPointEdgeStereo[i];
            if(!e || !pMP || pMP->isBad() || pMP->GetInstanceId() <= 0)
                continue;
            if(i >= vpEdgeFrameStereo.size() || vpEdgeFrameStereo[i] != pKF->mnFrameId)
                continue;

            const bool bOutlier = (e->chi2() > 7.815 || !e->isDepthPositive());
            if(!bOutlier)
                continue;

            stats.points.insert(pMP);
            ++stats.stereo;
        }

        return stats;
    };

    auto updateDynamicBackendOutliersForCurrentGraph = [&]() -> int
    {
        int nMarks = 0;
        const std::set<MapPoint*> sCurrentThingPoints = collectCurrentThingPoints();
        std::set<MapPoint*> sCurrentThingOutliers = computeCurrentThingOutliers();
        const CurrentThingResidualOutlierStats residualOutliers = collectCurrentThingResidualOutliers();
        sCurrentThingOutliers.insert(residualOutliers.points.begin(), residualOutliers.points.end());
        for(std::set<MapPoint*>::const_iterator sit = sCurrentThingPoints.begin();
            sit != sCurrentThingPoints.end(); ++sit)
        {
            if(sCurrentThingOutliers.count(*sit))
            {
                MarkDynamicBackendOutlier(pCurrentMap, *sit, pKF->mnFrameId);
                ++nMarks;
            }
            else
            {
                ClearDynamicBackendOutlier(pCurrentMap, *sit, pKF->mnFrameId);
            }
        }

        return nMarks;
    };

    auto refreshDynamicFactorWeights = [&]() -> int
    {
        int nUpdated = 0;
        for(size_t i = 0, iend = vpEdgesPanopticMono.size(); i < iend; ++i)
        {
            MapPoint* pMP = vpMapPointEdgePanopticMono[i];
            if(!pMP || pMP->GetInstanceId() <= 0 || i >= vpEdgePanopticMonoBaseInvSigma2.size())
                continue;

            const double weight = GetDynamicObservationWeight(pCurrentMap, pMP, vpEdgeFramePanopticMono[i]);
            vpEdgesPanopticMono[i]->setInformation(Eigen::Matrix2d::Identity() *
                                                   vpEdgePanopticMonoBaseInvSigma2[i] *
                                                   weight * panopticFactorWeight);
            ++nUpdated;
        }

        for(size_t i = 0; i < vPanopticInstanceEdgeRecords.size(); ++i)
        {
            PanopticInstanceEdgeRecord& record = vPanopticInstanceEdgeRecords[i];
            if(!record.edge || !record.pMP)
                continue;

            const double weight = GetDynamicObservationWeight(pCurrentMap, record.pMP, record.frameId);
            record.edge->setInformation(Eigen::Matrix2d::Identity() *
                                        record.baseInvSigma2 * weight * panopticFactorWeight);
            ++nUpdated;
        }

        for(size_t i = 0, iend = vpEdgesBody.size(); i < iend; ++i)
        {
            MapPoint* pMP = vpMapPointEdgeBody[i];
            if(!pMP || pMP->GetInstanceId() <= 0 || i >= vpEdgeBodyBaseInvSigma2.size())
                continue;

            const double weight = GetDynamicObservationWeight(pCurrentMap, pMP, vpEdgeFrameBody[i]);
            vpEdgesBody[i]->setInformation(Eigen::Matrix2d::Identity() *
                                            vpEdgeBodyBaseInvSigma2[i] * weight);
            ++nUpdated;
        }

        for(size_t i = 0, iend = vpEdgesStereo.size(); i < iend; ++i)
        {
            MapPoint* pMP = vpMapPointEdgeStereo[i];
            if(!pMP || pMP->GetInstanceId() <= 0 || i >= vpEdgeStereoBaseInvSigma2.size())
                continue;

            const double weight = GetDynamicObservationWeight(pCurrentMap, pMP, vpEdgeFrameStereo[i]);
            vpEdgesStereo[i]->setInformation(Eigen::Matrix3d::Identity() *
                                             vpEdgeStereoBaseInvSigma2[i] * weight);
            ++nUpdated;
        }

        for(size_t i = 0; i < vShapeScaleEdgeRecords.size(); ++i)
        {
            ShapeScaleEdgeRecord& record = vShapeScaleEdgeRecords[i];
            if(!record.frameEdge && !record.tripletEdge)
                continue;

            const double weight = GetMeanDynamicObservationWeight(pCurrentMap, record.observations);
            const Eigen::Matrix2d information =
                Eigen::Matrix2d::Identity() *
                record.baseInvSigma2 * weight * shapeFactorWeight *
                record.factorScale;
            if(record.frameEdge)
                record.frameEdge->setInformation(information);
            if(record.tripletEdge)
                record.tripletEdge->setInformation(information);
            ++nUpdated;
        }

        for(size_t i = 0; i < vRigidityEdgeRecords.size(); ++i)
        {
            RigidityEdgeRecord& record = vRigidityEdgeRecords[i];
            if(!record.frameEdge && !record.pairEdge)
                continue;

            const double weight = GetMeanDynamicObservationWeight(pCurrentMap, record.observations);
            const Eigen::Matrix<double,1,1> information =
                Eigen::Matrix<double,1,1>::Identity() *
                record.baseInvSigma2 * weight * rigidityFactorWeight *
                record.factorScale;
            if(record.frameEdge)
                record.frameEdge->setInformation(information);
            if(record.pairEdge)
                record.pairEdge->setInformation(information);
            ++nUpdated;
        }

        return nUpdated;
    };

    num_MPs = nPoints;
    num_edges = nEdges;

    if(pbStopFlag && *pbStopFlag)
        return;

    if(debugDynamicLBA)
    {
        std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
                  << " local_kf=" << lLocalKeyFrames.size()
                  << " module8_profile=" << module8Profile
                  << " fixed_kf=" << lFixedCameras.size()
                  << " local_points=" << lLocalMapPoints.size()
                  << " opt_points=" << nPoints
                  << " total_edges=" << nEdges
                  << " panoptic_instance_edges=" << nPanopticInstanceEdges
                  << " shape_edges=" << nShapeEdges
                  << " rigidity_edges=" << nRigidityEdges
                  << " rigidity_point_pairs=" << nRigidityPointPairs
                  << " dynamic_priors=" << nDynamicPriorEdges
                  << " instance_pose_motion_priors=" << nInstancePoseMotionPriorEdges
                  << " instance_pose_vertices=" << nInstancePoseVertices
                  << " instance_structure_edges=" << nInstanceStructureEdges
                  << " instance_structure_inv_sigma2=" << instanceStructureInvSigma2
                  << " shape_triplet_sample_limit=" << kMaxShapeTripletsPerFrame
                  << " shape_factor_mode="
                  << (useStrictShapeTripletFactors ? "per_triplet_eq13" : "aggregated_frame")
                  << " rigidity_pair_sample_limit=" << kMaxRigidityPairsPerNeighbor
                  << " rigidity_pair_max_degree=" << kMaxRigidityPairsPerPoint
                  << " rigidity_pair_min_quality=" << kRigidityPairMinQuality
                  << " rigidity_pair_min_known_obs=" << kRigidityPairMinKnownObservations
                  << " panoptic_factor_weight=" << panopticFactorWeight
                  << " shape_factor_weight=" << shapeFactorWeight
                  << " rigidity_factor_weight=" << rigidityFactorWeight
                  << " articulated_semantic_label=" << GetArticulatedSemanticLabel()
                  << " articulated_shape_factor_scale=" << GetArticulatedShapeFactorScale()
                  << " articulated_rigidity_factor_scale=" << GetArticulatedRigidityFactorScale()
                  << " shape_huber_delta=" << thHuberShape
                  << " rigidity_huber_delta=" << thHuberRigidity
                  << " rigidity_pair_sampler="
                  << GetRigidityPairSamplerName(useQualityGatedRigidityPairSampler,
                                                useQualityCoverageRigidityPairSampler,
                                                kMaxRigidityPairsPerNeighbor)
                  << " rigidity_max_frame_gap=" << kMaxRigidityFrameGap
                  << " dynamic_backend_min_frame_points=" << kDynamicBackendMinFrameSupport
                  << " rigidity_frobenius_residual=1"
                  << " rigidity_interval_unit=frame_index"
                  << " instance_pose_structure_coupling=" << enableInstanceStructureProxy
                  << " strict_eq16_instance_structure_proxy=" << enableInstanceStructureProxy
                  << " centroid_motion_fallback=" << enableCentroidMotionFallback
                  << " instance_motion_translation_only=" << translationOnlyInstanceMotionWriteback
                  << " instance_pose_motion_prior=" << (enableInstancePoseMotionPrior ? 1 : 0)
                  << " instance_pose_motion_prior_inv_sigma2="
                  << instancePoseMotionPriorInvSigma2
                  << " backend_mature_motion_gate=" << (backendMatureMotionGate ? 1 : 0)
                  << " backend_mature_max_translation=" << backendMatureMaxTranslation
                  << " backend_mature_max_rotation_deg=" << backendMatureMaxRotationDeg
                  << " strict_eq16_image_window=" << strictEq16ImageWindow
                  << " reweighted_observations=" << nDynamicReweightedObservations
                  << " backend_outlier_marks=" << nBackendOutlierMarks
                  << " instance_quality_gate=" << (EnableBackendInstanceObservationQualityGate() ? 1 : 0)
                  << " instance_quality_gate_hard_reject="
                  << (EnableBackendInstanceObservationQualityGateHardReject() ? 1 : 0)
                  << " instance_quality_gate_accepted=" << nInstanceObservationQualityGateAccepted
                  << " instance_quality_gate_rejected=" << nInstanceObservationQualityGateRejected
                  << " instance_quality_gate_sparse=" << nInstanceObservationQualityGateSparse
                  << " disable_shape=" << disableShapeEdges
                  << " disable_rigidity=" << disableRigidityEdges
                  << " disable_eq17_size_prior=" << disableEq17SizePrior
                  << std::endl;
    }

    int nIterativeBackendOutlierMarks = 0;
    int nIterativeDynamicFactorRefreshes = 0;
    for(int iter = 0; iter < nStrictEq16Iterations; ++iter)
    {
        optimizer.initializeOptimization();
        if(debugDynamicLBA && iter == 0)
            std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
                      << " initializeOptimization ok"
                      << " strict_eq17_iterative_reweight=1"
                      << std::endl;

        optimizer.optimize(1);
        const int iterMarks = updateDynamicBackendOutliersForCurrentGraph();
        const int iterRefreshes = refreshDynamicFactorWeights();
        nIterativeBackendOutlierMarks += iterMarks;
        nIterativeDynamicFactorRefreshes += iterRefreshes;

        if(debugDynamicLBA)
        {
            std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
                      << " optimize_iter=" << iter
                      << " backend_outlier_marks=" << iterMarks
                      << " dynamic_factor_refreshes=" << iterRefreshes
                      << std::endl;
        }
    }
    nBackendOutlierMarks += nIterativeBackendOutlierMarks;
    if(debugDynamicLBA)
    {
        std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
                  << " optimize ok"
                  << " strict_eq16_iterations=" << nStrictEq16Iterations
                  << " iterative_backend_outlier_marks=" << nIterativeBackendOutlierMarks
                  << " iterative_dynamic_factor_refreshes=" << nIterativeDynamicFactorRefreshes
                  << std::endl;

        FactorResidualStats stuffReprojectionStats;
        FactorResidualStats thingPanopticStats;
        FactorResidualStats shapeScaleStats;
        FactorResidualStats rigidityStats;

        for(size_t i = 0, iend = vpEdgesPanopticMono.size(); i < iend; ++i)
        {
            EdgePanopticProjection* e = vpEdgesPanopticMono[i];
            MapPoint* pMP = (i < vpMapPointEdgePanopticMono.size()) ? vpMapPointEdgePanopticMono[i] : NULL;
            if(!e || !pMP)
                continue;
            if(pMP->GetInstanceId() > 0)
                AccumulateFactorResidual(e, thingPanopticStats, 1, 5.991);
            else
                AccumulateFactorResidual(e, stuffReprojectionStats, 1, 5.991);
        }

        for(size_t i = 0; i < vPanopticInstanceEdgeRecords.size(); ++i)
        {
            const PanopticInstanceEdgeRecord& record = vPanopticInstanceEdgeRecords[i];
            if(!record.edge)
                continue;
            AccumulateFactorResidual(record.edge, thingPanopticStats, 1, 5.991);
        }

        for(size_t i = 0; i < vShapeScaleEdgeRecords.size(); ++i)
        {
            const ShapeScaleEdgeRecord& record = vShapeScaleEdgeRecords[i];
            if(record.frameEdge)
            {
                AccumulateFactorResidual(record.frameEdge,
                                         shapeScaleStats,
                                         std::max<size_t>(1, record.logicalTerms),
                                         thHuberShape * thHuberShape);
            }
            if(record.tripletEdge)
            {
                AccumulateFactorResidual(record.tripletEdge,
                                         shapeScaleStats,
                                         std::max<size_t>(1, record.logicalTerms),
                                         thHuberShape * thHuberShape);
            }
        }

        for(size_t i = 0; i < vRigidityEdgeRecords.size(); ++i)
        {
            const RigidityEdgeRecord& record = vRigidityEdgeRecords[i];
            if(record.frameEdge)
                AccumulateFactorResidual(record.frameEdge,
                                         rigidityStats,
                                         std::max<size_t>(1, record.logicalTerms),
                                         thHuberRigidity * thHuberRigidity);
            if(record.pairEdge)
                AccumulateFactorResidual(record.pairEdge,
                                         rigidityStats,
                                         std::max<size_t>(1, record.logicalTerms),
                                         thHuberRigidity * thHuberRigidity);
        }

        std::map<int, InstanceProjectionDebugStats> mPanopticInstanceFinalDebug;
        for(size_t i = 0; i < vPanopticInstanceEdgeRecords.size(); ++i)
        {
            const PanopticInstanceEdgeRecord& record = vPanopticInstanceEdgeRecords[i];
            if(!record.edge || !record.pMP)
                continue;

            record.edge->computeError();
            double reconError = 0.0;
            const std::map<DynamicVertexKey, int>::const_iterator itPointVertex =
                mThingVertexIds.find(DynamicVertexKey(record.pMP, record.frameId));
            g2o::VertexSBAPointXYZ* vPoint = NULL;
            if(itPointVertex != mThingVertexIds.end())
                vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(itPointVertex->second));
            g2o::VertexSE3Expmap* vInstancePose =
                static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(record.instancePoseVertexId));
            if(vPoint && vInstancePose)
            {
                const Eigen::Vector3d instanceWorldPoint =
                    vInstancePose->estimate().map(record.edge->mLocalPoint);
                reconError = (instanceWorldPoint - vPoint->estimate()).norm();
            }
            mPanopticInstanceFinalDebug[record.instanceId].Add(record.edge->chi2(),
                                                               0.0,
                                                               reconError,
                                                               record.edge->mLocalPoint.norm(),
                                                               record.edge->isDepthPositive());
        }

        std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
                  << " factor_residuals"
                  << " stuff_records=" << stuffReprojectionStats.records
                  << " stuff_terms=" << stuffReprojectionStats.logicalTerms
                  << " stuff_chi2_mean=" << stuffReprojectionStats.MeanChi2()
                  << " stuff_chi2_max=" << stuffReprojectionStats.chi2Max
                  << " stuff_high_chi2=" << stuffReprojectionStats.highChi2
                  << " thing_panoptic_records=" << thingPanopticStats.records
                  << " thing_panoptic_terms=" << thingPanopticStats.logicalTerms
                  << " thing_panoptic_chi2_mean=" << thingPanopticStats.MeanChi2()
                  << " thing_panoptic_chi2_max=" << thingPanopticStats.chi2Max
                  << " thing_panoptic_high_chi2=" << thingPanopticStats.highChi2
                  << " shape_records=" << shapeScaleStats.records
                  << " shape_terms=" << shapeScaleStats.logicalTerms
                  << " shape_chi2_mean=" << shapeScaleStats.MeanChi2()
                  << " shape_chi2_max=" << shapeScaleStats.chi2Max
                  << " shape_high_chi2=" << shapeScaleStats.highChi2
                  << " rigidity_records=" << rigidityStats.records
                  << " rigidity_terms=" << rigidityStats.logicalTerms
                  << " rigidity_chi2_mean=" << rigidityStats.MeanChi2()
                  << " rigidity_chi2_max=" << rigidityStats.chi2Max
                  << " rigidity_high_chi2=" << rigidityStats.highChi2
                  << std::endl;
        for(std::map<int, InstanceProjectionDebugStats>::const_iterator itDebug =
                mPanopticInstanceFinalDebug.begin();
            itDebug != mPanopticInstanceFinalDebug.end(); ++itDebug)
        {
            const InstanceProjectionDebugStats& dbg = itDebug->second;
            std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
                      << " panoptic_instance_projection_final"
                      << " instance_id=" << itDebug->first
                      << " records=" << dbg.records
                      << " chi2_mean=" << dbg.MeanChi2()
                      << " chi2_max=" << dbg.chi2Max
                      << " high_chi2=" << dbg.highChi2
                      << " depth_failures=" << dbg.depthFailures
                      << " recon_error_mean=" << dbg.MeanReconError()
                      << " recon_error_max=" << dbg.reconErrorMax
                      << " local_norm_mean=" << dbg.MeanLocalNorm()
                      << " local_norm_max=" << dbg.localNormMax
                      << std::endl;
        }
    }

    std::set<MapPoint*> sCurrentThingOutliers = computeCurrentThingOutliers();
    const size_t nEq17ThingOutliers = sCurrentThingOutliers.size();
    const CurrentThingResidualOutlierStats currentThingResidualOutliers =
        collectCurrentThingResidualOutliers();
    sCurrentThingOutliers.insert(currentThingResidualOutliers.points.begin(),
                                 currentThingResidualOutliers.points.end());

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesPanopticMono.size() + vpEdgesBody.size() + vpEdgesStereo.size());

    for(size_t i = 0, iend = vpEdgesPanopticMono.size(); i < iend; ++i)
    {
        EdgePanopticProjection* e = vpEdgesPanopticMono[i];
        MapPoint* pMP = vpMapPointEdgePanopticMono[i];
        if(pMP->isBad())
            continue;

        const bool bOutlier = (e->chi2() > 5.991 || !e->isDepthPositive());
        if(pMP->GetInstanceId() > 0)
        {
            continue;
        }

        if(bOutlier)
            vToErase.push_back(make_pair(vpEdgeKFPanopticMono[i], pMP));
    }

    for(size_t i = 0, iend = vpEdgesBody.size(); i < iend; ++i)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZToBody* e = vpEdgesBody[i];
        MapPoint* pMP = vpMapPointEdgeBody[i];
        if(pMP->isBad())
            continue;

        const bool bOutlier = (e->chi2() > 5.991 || !e->isDepthPositive());
        if(pMP->GetInstanceId() > 0)
        {
            continue;
        }

        if(bOutlier)
            vToErase.push_back(make_pair(vpEdgeKFBody[i], pMP));
    }

    for(size_t i = 0, iend = vpEdgesStereo.size(); i < iend; ++i)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];
        if(pMP->isBad())
            continue;

        const bool bOutlier = (e->chi2() > 7.815 || !e->isDepthPositive());
        if(pMP->GetInstanceId() > 0)
        {
            continue;
        }

        if(bOutlier)
            vToErase.push_back(make_pair(vpEdgeKFStereo[i], pMP));
    }

    const std::set<MapPoint*> sCurrentThingPoints = collectCurrentThingPoints();
    for(std::set<MapPoint*>::const_iterator sit = sCurrentThingPoints.begin(); sit != sCurrentThingPoints.end(); ++sit)
    {
        if(sCurrentThingOutliers.count(*sit))
        {
            MarkDynamicBackendOutlier(pCurrentMap, *sit, pKF->mnFrameId);
            ++nBackendOutlierMarks;
        }
        else
        {
            ClearDynamicBackendOutlier(pCurrentMap, *sit, pKF->mnFrameId);
        }
    }

    const size_t nStaticOutlierCandidates = vToErase.size();
    if(strictEq16ImageWindow)
        vToErase.clear();

    if(debugDynamicLBA)
    {
        std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
                  << " backend_outlier_update marks=" << nBackendOutlierMarks
                  << " static_outlier_candidates=" << nStaticOutlierCandidates
                  << " erase_static_observations=" << vToErase.size()
                  << " dynamic_outlier_eq17_size=" << nEq17ThingOutliers
                  << " dynamic_outlier_panoptic_instance=" << currentThingResidualOutliers.panopticInstance
                  << " dynamic_outlier_panoptic_mono=" << currentThingResidualOutliers.panopticMono
                  << " dynamic_outlier_body=" << currentThingResidualOutliers.body
                  << " dynamic_outlier_stereo=" << currentThingResidualOutliers.stereo
                  << " strict_eq16_no_static_observation_erasure=" << strictEq16ImageWindow
                  << std::endl;
    }

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    for(size_t i = 0; i < vToErase.size(); ++i)
    {
        KeyFrame* pKFi = vToErase[i].first;
        MapPoint* pMPi = vToErase[i].second;
        if(!pKFi || !pMPi)
            continue;
        pKFi->EraseMapPointMatch(pMPi);
        pMPi->EraseObservation(pKFi);
    }

    for(list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; ++lit)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        Sophus::SE3f Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());
        pKFi->SetPose(Tiw);
    }

    for(std::map<unsigned long, ImageFrameHandle>::iterator itFrame = mImageFrames.begin();
        itFrame != mImageFrames.end(); ++itFrame)
    {
        if(itFrame->second.isKeyFrame)
            continue;

        g2o::VertexSE3Expmap* vSE3 =
            static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(itFrame->second.poseVertexId));
        if(!vSE3)
            continue;

        g2o::SE3Quat SE3quat = vSE3->estimate();
        Sophus::SE3f Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());
        SetFramePose(itFrame->second, Tiw);
    }

    for(list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; ++lit)
    {
        MapPoint* pMP = *lit;

        g2o::VertexSBAPointXYZ* vStaticPoint =
            static_cast<g2o::VertexSBAPointXYZ*>(
                optimizer.vertex(pointVertexBaseId + static_cast<int>(pMP->mnId)));
        if(vStaticPoint)
        {
            pMP->SetWorldPos(vStaticPoint->estimate().cast<float>());
        }
        else
        {
            if(sCurrentThingOutliers.count(pMP))
                continue;

            Eigen::Vector3d optimizedPos = pMP->GetWorldPos().cast<double>();
            const std::map<MapPoint*, int>::iterator itCurrent = mThingCurrentVertexIds.find(pMP);
            if(itCurrent != mThingCurrentVertexIds.end())
            {
                g2o::VertexSBAPointXYZ* vPoint =
                    static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(itCurrent->second));
                if(vPoint)
                    optimizedPos = vPoint->estimate();
            }
            else
            {
                const std::map<MapPoint*, std::vector<int> >::iterator itVertexIds = mmThingVertexIdsByPoint.find(pMP);
                if(itVertexIds != mmThingVertexIdsByPoint.end())
                    optimizedPos = AveragePointEstimate(optimizer, itVertexIds->second);
            }

            pMP->SetWorldPos(optimizedPos.cast<float>());
        }

        pMP->UpdateNormalAndDepth();
    }

    for(std::map<int, std::map<unsigned long, std::vector<MapPoint*> > >::iterator itInstance =
            mmInstanceFramePoints.begin();
        itInstance != mmInstanceFramePoints.end(); ++itInstance)
    {
        Instance* pInstance = pCurrentMap->GetInstance(itInstance->first);
        if(!pInstance || itInstance->second.empty())
            continue;

        if(!pInstance->IsInitialized())
            continue;
        const bool instanceBackendMature = HasMatureInstanceBackendState(pInstance, pKF);

        const std::map<unsigned long, std::vector<MapPoint*> >& instanceFramePoints =
            itInstance->second;

        std::vector<unsigned long> vFrames;
        for(std::map<unsigned long, std::vector<MapPoint*> >::const_iterator itFrame =
                instanceFramePoints.begin();
            itFrame != instanceFramePoints.end(); ++itFrame)
        {
            if(mImageFrames.count(itFrame->first))
                vFrames.push_back(itFrame->first);
        }
        std::sort(vFrames.begin(), vFrames.end(),
                  [&](const unsigned long lhs, const unsigned long rhs)
                  {
                      return SortImageFramesByTimestamp(mImageFrames[lhs], mImageFrames[rhs]);
                  });

        unsigned long currentOptimizedFrameId = 0;
        bool foundCurrentOptimizedFrame = false;
        std::vector<Eigen::Vector3f> vShapeTemplate;
        std::vector<unsigned long>::reverse_iterator ritFrame = vFrames.rbegin();
        for(; ritFrame != vFrames.rend(); ++ritFrame)
        {
            const unsigned long frameId = *ritFrame;
            if(mInstancePoseVertexIds.count(InstanceFrameKey(itInstance->first, frameId)) == 0)
                continue;

            const std::map<unsigned long, std::vector<MapPoint*> >::const_iterator itPoints =
                instanceFramePoints.find(frameId);
            if(itPoints == instanceFramePoints.end())
                continue;

            std::vector<Eigen::Vector3f> vCandidateShape;
            vCandidateShape.reserve(itPoints->second.size());
            for(size_t i = 0; i < itPoints->second.size(); ++i)
            {
                MapPoint* pMP = itPoints->second[i];
                if(sCurrentThingOutliers.count(pMP))
                    continue;

                const DynamicVertexKey key(pMP, frameId);
                const std::map<DynamicVertexKey, int>::iterator itVertexId = mThingVertexIds.find(key);
                if(itVertexId == mThingVertexIds.end())
                    continue;

                g2o::VertexSBAPointXYZ* vPoint =
                    static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(itVertexId->second));
                if(vPoint)
                    vCandidateShape.push_back(vPoint->estimate().cast<float>());
            }

            if(static_cast<int>(vCandidateShape.size()) < kDynamicBackendMinFrameSupport)
                continue;

            currentOptimizedFrameId = frameId;
            foundCurrentOptimizedFrame = true;
            vShapeTemplate.swap(vCandidateShape);
            break;
        }

        if(!foundCurrentOptimizedFrame)
        {
            if(debugDynamicLBA)
            {
                std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
                          << " instance_motion_update_skipped"
                          << " instance_id=" << itInstance->first
                          << " reason=no_optimized_window_frame"
                          << " backend_mature=" << (instanceBackendMature ? 1 : 0)
                          << " window_frames=" << vFrames.size()
                          << std::endl;
            }
            continue;
        }

        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        for(size_t i = 0; i < vShapeTemplate.size(); ++i)
            centroid += vShapeTemplate[i];
        centroid /= static_cast<float>(vShapeTemplate.size());

        Sophus::SE3f currentInstancePose(Eigen::Matrix3f::Identity(), centroid);
        const std::map<InstanceFrameKey, int>::const_iterator itCurrentInstancePose =
            mInstancePoseVertexIds.find(InstanceFrameKey(itInstance->first, currentOptimizedFrameId));
        if(itCurrentInstancePose != mInstancePoseVertexIds.end())
        {
            g2o::VertexSE3Expmap* vInstancePose =
                static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(itCurrentInstancePose->second));
            if(vInstancePose)
            {
                const g2o::SE3Quat SE3quat = vInstancePose->estimate();
                currentInstancePose = Sophus::SE3f(SE3quat.rotation().cast<float>(),
                                                   SE3quat.translation().cast<float>());
            }
        }

        pInstance->UpdatePoseProxy(pKF, currentInstancePose);
        pInstance->SetInstanceMotionState(
            currentOptimizedFrameId,
            currentInstancePose,
            pInstance->GetVelocity(),
            pInstance->GetCurrentDynamicEntityMotionState(),
            instanceBackendMature ? 0.75 : 0.5,
            instanceBackendMature);
        pInstance->SetShapeTemplate(vShapeTemplate);

        int persistedOptimizedLocalPoints = 0;
        int persistedCurrentFrameLocalPoints = 0;
        int skippedExistingLocalPoints = 0;
        std::set<MapPoint*> sPersistedLocalPoints;
        const bool enableStructureWriteback =
            EnableBackendInstanceStructureWritebackForLBA();
        const bool allowStructureWriteback =
            enableStructureWriteback &&
            (instanceBackendMature || AllowImmatureInstanceStructureWritebackForLBA());
        const bool allowStructureOverwrite = AllowInstanceStructureOverwriteForLBA();
        const bool translationOnlyStructureWriteback =
            UseTranslationOnlyInstanceStructureWritebackForLBA(pInstance->GetSemanticLabel());
        const Sophus::SE3f persistedInstancePose =
            translationOnlyStructureWriteback ?
                Sophus::SE3f(Eigen::Matrix3f::Identity(), currentInstancePose.translation()) :
                currentInstancePose;
        const Sophus::SE3f persistedInstancePoseInv = persistedInstancePose.inverse();
        const std::map<unsigned long, std::vector<MapPoint*> >::const_iterator itCurrentFramePoints =
            instanceFramePoints.find(currentOptimizedFrameId);
        if(allowStructureWriteback && itCurrentFramePoints != instanceFramePoints.end())
        {
            const std::vector<MapPoint*>& vCurrentFramePoints = itCurrentFramePoints->second;
            for(size_t pointIdx = 0; pointIdx < vCurrentFramePoints.size(); ++pointIdx)
            {
                MapPoint* pMP = vCurrentFramePoints[pointIdx];
                if(!pMP || sCurrentThingOutliers.count(pMP))
                    continue;

                const DynamicVertexKey key(pMP, currentOptimizedFrameId);
                const std::map<DynamicVertexKey, int>::const_iterator itVertexId =
                    mThingVertexIds.find(key);
                if(itVertexId == mThingVertexIds.end())
                    continue;

                g2o::VertexSBAPointXYZ* vPoint =
                    static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(itVertexId->second));
                if(!vPoint || !vPoint->estimate().allFinite())
                    continue;

                Eigen::Vector3f existingLocalPoint;
                if(!allowStructureOverwrite &&
                   pInstance->GetStructureLocalPoint(pMP, existingLocalPoint))
                {
                    ++skippedExistingLocalPoints;
                    continue;
                }

                const Eigen::Vector3f localPoint =
                    persistedInstancePoseInv * vPoint->estimate().cast<float>();
                if(!localPoint.allFinite())
                    continue;

                pInstance->SetStructureLocalPoint(pMP, localPoint);
                sPersistedLocalPoints.insert(pMP);
                ++persistedOptimizedLocalPoints;
                ++persistedCurrentFrameLocalPoints;
            }
        }

        if(debugDynamicLBA)
        {
            std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
                      << " instance_structure_update"
                      << " instance_id=" << itInstance->first
                      << " semantic=" << pInstance->GetSemanticLabel()
                      << " current_frame_id=" << currentOptimizedFrameId
                      << " writeback_enabled=" << (enableStructureWriteback ? 1 : 0)
                      << " backend_mature=" << (instanceBackendMature ? 1 : 0)
                      << " allow_immature="
                      << (AllowImmatureInstanceStructureWritebackForLBA() ? 1 : 0)
                      << " allow_overwrite=" << (allowStructureOverwrite ? 1 : 0)
                      << " persisted_local_points=" << persistedOptimizedLocalPoints
                      << " current_frame_local_points=" << persistedCurrentFrameLocalPoints
                      << " skipped_existing_local_points=" << skippedExistingLocalPoints
                      << " translation_only=" << (translationOnlyStructureWriteback ? 1 : 0)
                      << " skipped_immature=" << (!allowStructureWriteback ? 1 : 0)
                      << " optimized_pose_rotation_deg="
                      << RotationAngleDeg(currentInstancePose.rotationMatrix())
                      << std::endl;
        }

        double meanRadius = 0.0;
        for(size_t i = 0; i < vShapeTemplate.size(); ++i)
            meanRadius += (vShapeTemplate[i] - centroid).norm();
        meanRadius /= static_cast<double>(vShapeTemplate.size());
        pCurrentMap->RecordInstanceClassSizePrior(pInstance->GetSemanticLabel(),
                                                  pInstance->GetId(),
                                                  meanRadius);

        std::vector<unsigned long>::iterator itCurrFrame =
            std::find(vFrames.begin(), vFrames.end(), currentOptimizedFrameId);
        if(itCurrFrame != vFrames.end() && itCurrFrame != vFrames.begin())
        {
            unsigned long prevFrameId = 0;
            bool foundPreviousOptimizedFrame = false;
            std::vector<unsigned long>::reverse_iterator ritPrev(itCurrFrame);
            for(; ritPrev != vFrames.rend(); ++ritPrev)
            {
                const unsigned long candidateFrameId = *ritPrev;
                const bool shortFrameGap =
                    (backendMotionMaxFrameGap == 0 ||
                     currentOptimizedFrameId <=
                         candidateFrameId + static_cast<unsigned long>(backendMotionMaxFrameGap));
                if(!shortFrameGap)
                    continue;

                const std::map<unsigned long, std::vector<MapPoint*> >::const_iterator itPrevPointsCandidate =
                    instanceFramePoints.find(candidateFrameId);
                if(itPrevPointsCandidate == instanceFramePoints.end() ||
                   static_cast<int>(itPrevPointsCandidate->second.size()) < kDynamicBackendMinFrameSupport)
                    continue;

                if(mInstancePoseVertexIds.count(InstanceFrameKey(itInstance->first, candidateFrameId)) == 0 &&
                   !enableCentroidMotionFallback)
                    continue;

                prevFrameId = candidateFrameId;
                foundPreviousOptimizedFrame = true;
                break;
            }

            if(!foundPreviousOptimizedFrame)
            {
                if(debugDynamicLBA)
                {
                    std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
                              << " instance_motion_update_skipped"
                              << " instance_id=" << itInstance->first
                              << " reason=no_previous_optimized_window_frame"
                              << " backend_mature=" << (instanceBackendMature ? 1 : 0)
                              << " current_frame_id=" << currentOptimizedFrameId
                              << std::endl;
                }
                continue;
            }

            const std::vector<MapPoint*>& vPrevPoints = instanceFramePoints.at(prevFrameId);

            const std::map<InstanceFrameKey, int>::const_iterator itPrevInstancePose =
                mInstancePoseVertexIds.find(InstanceFrameKey(itInstance->first, prevFrameId));
            if(itPrevInstancePose != mInstancePoseVertexIds.end())
            {
                g2o::VertexSE3Expmap* vPrevInstancePose =
                    static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(itPrevInstancePose->second));
                if(vPrevInstancePose)
                {
                    const g2o::SE3Quat SE3quat = vPrevInstancePose->estimate();
	                    const Sophus::SE3f previousInstancePose(SE3quat.rotation().cast<float>(),
	                                                            SE3quat.translation().cast<float>());
                    const int backendMotionFrameGap =
                        currentOptimizedFrameId > prevFrameId ?
                        static_cast<int>(currentOptimizedFrameId - prevFrameId) : 1;
                    const Sophus::SE3f rawOptimizedMotion =
                            currentInstancePose * previousInstancePose.inverse();
                        Sophus::SE3f optimizedMotion =
                            NormalizeMotionByFrameGap(rawOptimizedMotion, backendMotionFrameGap);
                        if(translationOnlyInstanceMotionWriteback)
                        {
                            optimizedMotion = Sophus::SE3f(
                                Eigen::Matrix3f::Identity(),
                                (currentInstancePose.translation() -
                                 previousInstancePose.translation()) /
                                    static_cast<float>(std::max(1, backendMotionFrameGap)));
                        }
                        const float optimizedTranslationNorm = optimizedMotion.translation().norm();
                        const float optimizedRotationDeg =
                            static_cast<float>(RotationAngleDeg(optimizedMotion.rotationMatrix()));
                        const bool matureMotionRejected =
                            backendMatureMotionGate &&
                            instanceBackendMature &&
                            (optimizedTranslationNorm > backendMatureMaxTranslation ||
                             optimizedRotationDeg > backendMatureMaxRotationDeg);
		                        const bool velocityApplied =
	                            !matureMotionRejected &&
		                            pInstance->RecordBackendMotionObservation(
		                                optimizedMotion,
	                                instanceBackendMature,
	                                static_cast<int>(currentOptimizedFrameId),
	                                backendImmatureMaxTranslation,
		                                backendImmatureMaxRotationDeg,
		                                backendImmatureConfirmFrames);
                        if(velocityApplied)
                        {
                            const Instance::DynamicEntityMotionState semanticMotionState =
                                ClassifyDynamicEntityMotionStateFromVelocity(
                                    optimizedMotion,
                                    instanceBackendMature,
                                    pInstance->GetSemanticLabel());
                            const Sophus::SE3f semanticVelocity =
                                CanonicalVelocityForDynamicEntityState(
                                    optimizedMotion,
                                    semanticMotionState);
                            pInstance->UpdateMotionPrior(pKF, semanticVelocity);
                            pInstance->SetInstanceMotionState(
                                currentOptimizedFrameId,
                                currentInstancePose,
                                semanticVelocity,
                                semanticMotionState,
                                instanceBackendMature ? 1.0 : 0.75,
                                instanceBackendMature);
                        }
					                    if(debugDynamicLBA)
					                    {
				                        std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
		                                  << " instance_motion_update"
		                                  << " instance_id=" << itInstance->first
		                                  << " source=instance_pose_vertices"
		                                          << " backend_mature=" << (instanceBackendMature ? 1 : 0)
	                                              << " velocity_applied=" << (velocityApplied ? 1 : 0)
                                                  << " mature_motion_rejected=" << (matureMotionRejected ? 1 : 0)
	                                              << " backend_motion_evidence=" << pInstance->GetBackendMotionEvidence()
                                                  << " backend_zero_evidence="
                                                  << pInstance->GetBackendZeroMotionEvidence()
                                                  << " backend_moving_evidence="
                                                  << pInstance->GetBackendMovingMotionEvidence()
                                                  << " backend_uncertain_evidence="
                                                  << pInstance->GetBackendUncertainMotionEvidence()
                                                  << " dynamic_entity_state="
                                                  << DynamicEntityMotionStateName(
                                                      ClassifyDynamicEntityMotionStateFromVelocity(
                                                          optimizedMotion,
                                                          instanceBackendMature,
                                                          pInstance->GetSemanticLabel()))
	                                              << " immature_max_translation=" << backendImmatureMaxTranslation
	                                              << " immature_max_rotation_deg=" << backendImmatureMaxRotationDeg
                                                  << " mature_max_translation=" << backendMatureMaxTranslation
                                                  << " mature_max_rotation_deg=" << backendMatureMaxRotationDeg
	                                              << " immature_confirm_frames=" << backendImmatureConfirmFrames
	                                          << " prev_frame_id=" << prevFrameId
                                          << " current_frame_id=" << currentOptimizedFrameId
                                          << " frame_gap=" << backendMotionFrameGap
                                          << " backend_motion_max_frame_gap=" << backendMotionMaxFrameGap
			                                  << " translation_norm=" << optimizedTranslationNorm
			                                  << " rotation_deg=" << optimizedRotationDeg
	                                      << " raw_rotation_deg=" << RotationAngleDeg(rawOptimizedMotion.rotationMatrix())
                                          << " raw_translation_norm=" << rawOptimizedMotion.translation().norm()
                                      << " translation_only=" << translationOnlyInstanceMotionWriteback
	                                  << std::endl;
	                    }
	                }
	            }
	            else if(enableCentroidMotionFallback)
	            {
                    std::vector<Eigen::Vector3f> vPrevCommonPoints;
                    std::vector<Eigen::Vector3f> vCurrCommonPoints;
	                    for(size_t i = 0; i < vPrevPoints.size(); ++i)
                    {
                        MapPoint* pMP = vPrevPoints[i];
                        if(!pMP || sCurrentThingOutliers.count(pMP))
                            continue;

	                        const DynamicVertexKey prevKey(pMP, prevFrameId);
	                        const DynamicVertexKey currKey(pMP, currentOptimizedFrameId);
                        const std::map<DynamicVertexKey, int>::iterator itPrevVertex =
                            mThingVertexIds.find(prevKey);
                        const std::map<DynamicVertexKey, int>::iterator itCurrVertex =
                            mThingVertexIds.find(currKey);
                        if(itPrevVertex == mThingVertexIds.end() ||
                           itCurrVertex == mThingVertexIds.end())
                            continue;

                        g2o::VertexSBAPointXYZ* vPrevPoint =
                            static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(itPrevVertex->second));
                        g2o::VertexSBAPointXYZ* vCurrPoint =
                            static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(itCurrVertex->second));
                        if(!vPrevPoint || !vCurrPoint)
                            continue;

                        vPrevCommonPoints.push_back(vPrevPoint->estimate().cast<float>());
                        vCurrCommonPoints.push_back(vCurrPoint->estimate().cast<float>());
                    }

                    Sophus::SE3f motion;
                    if(static_cast<int>(vPrevCommonPoints.size()) >= kDynamicBackendMinFrameSupport &&
                       EstimateCommonPointTranslation(vPrevCommonPoints, vCurrCommonPoints, motion))
                    {
                        const int backendMotionFrameGap =
                            currentOptimizedFrameId > prevFrameId ?
                            static_cast<int>(currentOptimizedFrameId - prevFrameId) : 1;
                        motion = NormalizeMotionByFrameGap(motion, backendMotionFrameGap);
                        const float optimizedTranslationNorm = motion.translation().norm();
                        const float optimizedRotationDeg = 0.0f;
                        const bool matureMotionRejected =
                            backendMatureMotionGate &&
                            instanceBackendMature &&
                            (optimizedTranslationNorm > backendMatureMaxTranslation ||
                             optimizedRotationDeg > backendMatureMaxRotationDeg);
	                            const bool velocityApplied =
                                !matureMotionRejected &&
	                                pInstance->RecordBackendMotionObservation(
	                                    motion,
	                                    instanceBackendMature,
	                                    static_cast<int>(currentOptimizedFrameId),
	                                    backendImmatureMaxTranslation,
	                                    backendImmatureMaxRotationDeg,
	                                    backendImmatureConfirmFrames);
	                        if(velocityApplied)
                            {
                                const Instance::DynamicEntityMotionState semanticMotionState =
                                    ClassifyDynamicEntityMotionStateFromVelocity(
                                        motion,
                                        instanceBackendMature,
                                        pInstance->GetSemanticLabel());
                                const Sophus::SE3f semanticVelocity =
                                    CanonicalVelocityForDynamicEntityState(
                                        motion,
                                        semanticMotionState);
			                    pInstance->UpdateMotionPrior(pKF, semanticVelocity);
                                pInstance->SetInstanceMotionState(
                                    currentOptimizedFrameId,
                                    currentInstancePose,
                                    semanticVelocity,
                                    semanticMotionState,
                                    instanceBackendMature ? 1.0 : 0.75,
                                    instanceBackendMature);
                            }
			                        if(debugDynamicLBA)
			                        {
		                            std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
		                                      << " instance_motion_update"
		                                      << " instance_id=" << itInstance->first
		                                      << " source=common_points_centroid"
		                                              << " backend_mature=" << (instanceBackendMature ? 1 : 0)
	                                                  << " velocity_applied=" << (velocityApplied ? 1 : 0)
                                                      << " mature_motion_rejected=" << (matureMotionRejected ? 1 : 0)
	                                                  << " backend_motion_evidence=" << pInstance->GetBackendMotionEvidence()
                                                      << " backend_zero_evidence="
                                                      << pInstance->GetBackendZeroMotionEvidence()
                                                      << " backend_moving_evidence="
                                                      << pInstance->GetBackendMovingMotionEvidence()
                                                      << " backend_uncertain_evidence="
                                                      << pInstance->GetBackendUncertainMotionEvidence()
                                                      << " dynamic_entity_state="
                                                      << DynamicEntityMotionStateName(
                                                          ClassifyDynamicEntityMotionStateFromVelocity(
                                                              motion,
                                                              instanceBackendMature,
                                                              pInstance->GetSemanticLabel()))
	                                                  << " immature_max_translation=" << backendImmatureMaxTranslation
	                                                  << " immature_max_rotation_deg=" << backendImmatureMaxRotationDeg
                                                      << " mature_max_translation=" << backendMatureMaxTranslation
                                                      << " mature_max_rotation_deg=" << backendMatureMaxRotationDeg
	                                                  << " immature_confirm_frames=" << backendImmatureConfirmFrames
	                                          << " prev_frame_id=" << prevFrameId
                                          << " current_frame_id=" << currentOptimizedFrameId
                                          << " frame_gap=" << backendMotionFrameGap
                                          << " backend_motion_max_frame_gap=" << backendMotionMaxFrameGap
		                                      << " common_points=" << vPrevCommonPoints.size()
		                                      << " translation_norm=" << optimizedTranslationNorm
		                                      << " rotation_deg=" << optimizedRotationDeg
                                      << std::endl;
                        }
                    }
                    else if(debugDynamicLBA)
                    {
	                        std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
	                                  << " instance_motion_update_skipped"
	                                  << " instance_id=" << itInstance->first
	                                  << " reason=insufficient_common_points"
                                      << " backend_mature=" << (instanceBackendMature ? 1 : 0)
	                                  << " common_points=" << vPrevCommonPoints.size()
	                                  << std::endl;
                    }
	            }
                else if(debugDynamicLBA)
                {
	                    std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
	                              << " instance_motion_update_skipped"
	                              << " instance_id=" << itInstance->first
	                              << " reason=no_instance_pose_vertex"
                                  << " backend_mature=" << (instanceBackendMature ? 1 : 0)
	                              << " centroid_motion_fallback=0"
	                              << std::endl;
	                }
		        }
            else if(debugDynamicLBA)
            {
                std::cerr << "[STSLAM_DYNAMIC_LBA] kf=" << pKF->mnId
                          << " instance_motion_update_skipped"
                          << " instance_id=" << itInstance->first
                          << " reason=no_earlier_window_frame"
                          << " backend_mature=" << (instanceBackendMature ? 1 : 0)
                          << " current_frame_id=" << currentOptimizedFrameId
                          << std::endl;
            }
		    }

    pMap->IncreaseChangeIndex();
}


void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, const bool &bFixScale)
{   
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

    vector<Eigen::Vector3d> vZvectors(nMaxKFid+1); // For debugging
    Eigen::Vector3d z_vec;
    z_vec << 0.0, 0.0, 1.0;

    const int minFeat = 100;

    // Set KeyFrame vertices
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKF->mnId;

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

        if(it!=CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        }
        else
        {
            Sophus::SE3d Tcw = pKF->GetPose().cast<double>();
            g2o::Sim3 Siw(Tcw.unit_quaternion(),Tcw.translation(),1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        if(pKF->mnId==pMap->GetInitKFid())
            VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = bFixScale;

        optimizer.addVertex(VSim3);
        vZvectors[nIDi]=vScw[nIDi].rotation()*z_vec; // For debugging

        vpVertices[nIDi]=VSim3;
    }


    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    // Set Loop edges
    int count_loop = 0;
    for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame*> &spConnections = mit->second;
        const g2o::Sim3 Siw = vScw[nIDi];
        const g2o::Sim3 Swi = Siw.inverse();

        for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;

            optimizer.addEdge(e);
            count_loop++;
            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
        }
    }

    // Set normal edges
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if(iti!=NonCorrectedSim3.end())
            Swi = (iti->second).inverse();
        else
            Swi = vScw[nIDi].inverse();

        KeyFrame* pParentKF = pKF->GetParent();

        // Spanning tree edge
        if(pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if(itj!=NonCorrectedSim3.end())
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];

            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);
            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if(pLKF->mnId<pKF->mnId)
            {
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if(itl!=NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3* el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) /*&& !sLoopEdges.count(pKFn)*/)
            {
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                {
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if(itn!=NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3* en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }

        // Inertial edges if inertial
        if(pKF->bImu && pKF->mPrevKF)
        {
            g2o::Sim3 Spw;
            LoopClosing::KeyFrameAndPose::const_iterator itp = NonCorrectedSim3.find(pKF->mPrevKF);
            if(itp!=NonCorrectedSim3.end())
                Spw = itp->second;
            else
                Spw = vScw[pKF->mPrevKF->mnId];

            g2o::Sim3 Spi = Spw * Swi;
            g2o::EdgeSim3* ep = new g2o::EdgeSim3();
            ep->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mPrevKF->mnId)));
            ep->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            ep->setMeasurement(Spi);
            ep->information() = matLambda;
            optimizer.addEdge(ep);
        }
    }


    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    optimizer.optimize(20);
    optimizer.computeActiveErrors();
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        double s = CorrectedSiw.scale();

        Sophus::SE3f Tiw(CorrectedSiw.rotation().cast<float>(), CorrectedSiw.translation().cast<float>() / s);
        pKFi->SetPose(Tiw);
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        int nIDr;
        if(pMP->mnCorrectedByKF==pCurKF->mnId)
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }


        g2o::Sim3 Srw = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        Eigen::Matrix<double,3,1> eigP3Dw = pMP->GetWorldPos().cast<double>();
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));
        pMP->SetWorldPos(eigCorrectedP3Dw.cast<float>());

        pMP->UpdateNormalAndDepth();
    }

    // TODO Check this changeindex
    pMap->IncreaseChangeIndex();
}

void Optimizer::OptimizeEssentialGraph(KeyFrame* pCurKF, vector<KeyFrame*> &vpFixedKFs, vector<KeyFrame*> &vpFixedCorrectedKFs,
                                       vector<KeyFrame*> &vpNonFixedKFs, vector<MapPoint*> &vpNonCorrectedMPs)
{
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpFixedKFs.size()) + " KFs fixed in the merged map", Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpFixedCorrectedKFs.size()) + " KFs fixed in the old map", Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpNonFixedKFs.size()) + " KFs non-fixed in the merged map", Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpNonCorrectedMPs.size()) + " MPs non-corrected in the merged map", Verbose::VERBOSITY_DEBUG);

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    Map* pMap = pCurKF->GetMap();
    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

    vector<bool> vpGoodPose(nMaxKFid+1);
    vector<bool> vpBadPose(nMaxKFid+1);

    const int minFeat = 100;

    for(KeyFrame* pKFi : vpFixedKFs)
    {
        if(pKFi->isBad())
            continue;

        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKFi->mnId;

        Sophus::SE3d Tcw = pKFi->GetPose().cast<double>();
        g2o::Sim3 Siw(Tcw.unit_quaternion(),Tcw.translation(),1.0);

        vCorrectedSwc[nIDi]=Siw.inverse();
        VSim3->setEstimate(Siw);

        VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = true;

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;

        vpGoodPose[nIDi] = true;
        vpBadPose[nIDi] = false;
    }
    Verbose::PrintMess("Opt_Essential: vpFixedKFs loaded", Verbose::VERBOSITY_DEBUG);

    set<unsigned long> sIdKF;
    for(KeyFrame* pKFi : vpFixedCorrectedKFs)
    {
        if(pKFi->isBad())
            continue;

        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKFi->mnId;

        Sophus::SE3d Tcw = pKFi->GetPose().cast<double>();
        g2o::Sim3 Siw(Tcw.unit_quaternion(),Tcw.translation(),1.0);

        vCorrectedSwc[nIDi]=Siw.inverse();
        VSim3->setEstimate(Siw);

        Sophus::SE3d Tcw_bef = pKFi->mTcwBefMerge.cast<double>();
        vScw[nIDi] = g2o::Sim3(Tcw_bef.unit_quaternion(),Tcw_bef.translation(),1.0);

        VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;

        sIdKF.insert(nIDi);

        vpGoodPose[nIDi] = true;
        vpBadPose[nIDi] = true;
    }

    for(KeyFrame* pKFi : vpNonFixedKFs)
    {
        if(pKFi->isBad())
            continue;

        const int nIDi = pKFi->mnId;

        if(sIdKF.count(nIDi)) // It has already added in the corrected merge KFs
            continue;

        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        Sophus::SE3d Tcw = pKFi->GetPose().cast<double>();
        g2o::Sim3 Siw(Tcw.unit_quaternion(),Tcw.translation(),1.0);

        vScw[nIDi] = Siw;
        VSim3->setEstimate(Siw);

        VSim3->setFixed(false);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;

        sIdKF.insert(nIDi);

        vpGoodPose[nIDi] = false;
        vpBadPose[nIDi] = true;
    }

    vector<KeyFrame*> vpKFs;
    vpKFs.reserve(vpFixedKFs.size() + vpFixedCorrectedKFs.size() + vpNonFixedKFs.size());
    vpKFs.insert(vpKFs.end(),vpFixedKFs.begin(),vpFixedKFs.end());
    vpKFs.insert(vpKFs.end(),vpFixedCorrectedKFs.begin(),vpFixedCorrectedKFs.end());
    vpKFs.insert(vpKFs.end(),vpNonFixedKFs.begin(),vpNonFixedKFs.end());
    set<KeyFrame*> spKFs(vpKFs.begin(), vpKFs.end());

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    for(KeyFrame* pKFi : vpKFs)
    {
        int num_connections = 0;
        const int nIDi = pKFi->mnId;

        g2o::Sim3 correctedSwi;
        g2o::Sim3 Swi;

        if(vpGoodPose[nIDi])
            correctedSwi = vCorrectedSwc[nIDi];
        if(vpBadPose[nIDi])
            Swi = vScw[nIDi].inverse();

        KeyFrame* pParentKFi = pKFi->GetParent();

        // Spanning tree edge
        if(pParentKFi && spKFs.find(pParentKFi) != spKFs.end())
        {
            int nIDj = pParentKFi->mnId;

            g2o::Sim3 Sjw;
            bool bHasRelation = false;

            if(vpGoodPose[nIDi] && vpGoodPose[nIDj])
            {
                Sjw = vCorrectedSwc[nIDj].inverse();
                bHasRelation = true;
            }
            else if(vpBadPose[nIDi] && vpBadPose[nIDj])
            {
                Sjw = vScw[nIDj];
                bHasRelation = true;
            }

            if(bHasRelation)
            {
                g2o::Sim3 Sji = Sjw * Swi;

                g2o::EdgeSim3* e = new g2o::EdgeSim3();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                e->setMeasurement(Sji);

                e->information() = matLambda;
                optimizer.addEdge(e);
                num_connections++;
            }

        }

        // Loop edges
        const set<KeyFrame*> sLoopEdges = pKFi->GetLoopEdges();
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if(spKFs.find(pLKF) != spKFs.end() && pLKF->mnId<pKFi->mnId)
            {
                g2o::Sim3 Slw;
                bool bHasRelation = false;

                if(vpGoodPose[nIDi] && vpGoodPose[pLKF->mnId])
                {
                    Slw = vCorrectedSwc[pLKF->mnId].inverse();
                    bHasRelation = true;
                }
                else if(vpBadPose[nIDi] && vpBadPose[pLKF->mnId])
                {
                    Slw = vScw[pLKF->mnId];
                    bHasRelation = true;
                }


                if(bHasRelation)
                {
                    g2o::Sim3 Sli = Slw * Swi;
                    g2o::EdgeSim3* el = new g2o::EdgeSim3();
                    el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                    el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    el->setMeasurement(Sli);
                    el->information() = matLambda;
                    optimizer.addEdge(el);
                    num_connections++;
                }
            }
        }

        // Covisibility graph edges
        const vector<KeyFrame*> vpConnectedKFs = pKFi->GetCovisiblesByWeight(minFeat);
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKFi && !pKFi->hasChild(pKFn) && !sLoopEdges.count(pKFn) && spKFs.find(pKFn) != spKFs.end())
            {
                if(!pKFn->isBad() && pKFn->mnId<pKFi->mnId)
                {

                    g2o::Sim3 Snw =  vScw[pKFn->mnId];
                    bool bHasRelation = false;

                    if(vpGoodPose[nIDi] && vpGoodPose[pKFn->mnId])
                    {
                        Snw = vCorrectedSwc[pKFn->mnId].inverse();
                        bHasRelation = true;
                    }
                    else if(vpBadPose[nIDi] && vpBadPose[pKFn->mnId])
                    {
                        Snw = vScw[pKFn->mnId];
                        bHasRelation = true;
                    }

                    if(bHasRelation)
                    {
                        g2o::Sim3 Sni = Snw * Swi;

                        g2o::EdgeSim3* en = new g2o::EdgeSim3();
                        en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                        en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                        en->setMeasurement(Sni);
                        en->information() = matLambda;
                        optimizer.addEdge(en);
                        num_connections++;
                    }
                }
            }
        }

        if(num_connections == 0 )
        {
            Verbose::PrintMess("Opt_Essential: KF " + to_string(pKFi->mnId) + " has 0 connections", Verbose::VERBOSITY_DEBUG);
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(KeyFrame* pKFi : vpNonFixedKFs)
    {
        if(pKFi->isBad())
            continue;

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        double s = CorrectedSiw.scale();
        Sophus::SE3d Tiw(CorrectedSiw.rotation(),CorrectedSiw.translation() / s);

        pKFi->mTcwBefMerge = pKFi->GetPose();
        pKFi->mTwcBefMerge = pKFi->GetPoseInverse();
        pKFi->SetPose(Tiw.cast<float>());
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for(MapPoint* pMPi : vpNonCorrectedMPs)
    {
        if(pMPi->isBad())
            continue;

        KeyFrame* pRefKF = pMPi->GetReferenceKeyFrame();
        while(pRefKF->isBad())
        {
            if(!pRefKF)
            {
                Verbose::PrintMess("MP " + to_string(pMPi->mnId) + " without a valid reference KF", Verbose::VERBOSITY_DEBUG);
                break;
            }

            pMPi->EraseObservation(pRefKF);
            pRefKF = pMPi->GetReferenceKeyFrame();
        }

        if(vpBadPose[pRefKF->mnId])
        {
            Sophus::SE3f TNonCorrectedwr = pRefKF->mTwcBefMerge;
            Sophus::SE3f Twr = pRefKF->GetPoseInverse();

            Eigen::Vector3f eigCorrectedP3Dw = Twr * TNonCorrectedwr.inverse() * pMPi->GetWorldPos();
            pMPi->SetWorldPos(eigCorrectedP3Dw);

            pMPi->UpdateNormalAndDepth();
        }
        else
        {
            cout << "ERROR: MapPoint has a reference KF from another map" << endl;
        }

    }
}

int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2,
                            const bool bFixScale, Eigen::Matrix<double,7,7> &mAcumHessian, const bool bAllPoints)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Camera poses
    const Eigen::Matrix3f R1w = pKF1->GetRotation();
    const Eigen::Vector3f t1w = pKF1->GetTranslation();
    const Eigen::Matrix3f R2w = pKF2->GetRotation();
    const Eigen::Vector3f t2w = pKF2->GetTranslation();

    // Set Sim3 vertex
    ORB_SLAM3::VertexSim3Expmap * vSim3 = new ORB_SLAM3::VertexSim3Expmap();
    vSim3->_fix_scale=bFixScale;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->pCamera1 = pKF1->mpCamera;
    vSim3->pCamera2 = pKF2->mpCamera;
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = vpMatches1.size();
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<ORB_SLAM3::EdgeSim3ProjectXYZ*> vpEdges12;
    vector<ORB_SLAM3::EdgeInverseSim3ProjectXYZ*> vpEdges21;
    vector<size_t> vnIndexEdge;
    vector<bool> vbIsInKF2;

    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);
    vbIsInKF2.reserve(2*N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;
    int nBadMPs = 0;
    int nInKF2 = 0;
    int nOutKF2 = 0;
    int nMatchWithoutMP = 0;

    vector<int> vIdsOnlyInKF2;

    for(int i=0; i<N; i++)
    {
        if(!vpMatches1[i])
            continue;

        MapPoint* pMP1 = vpMapPoints1[i];
        MapPoint* pMP2 = vpMatches1[i];

        const int id1 = 2*i+1;
        const int id2 = 2*(i+1);

        const int i2 = get<0>(pMP2->GetIndexInKeyFrame(pKF2));

        Eigen::Vector3f P3D1c;
        Eigen::Vector3f P3D2c;

        if(pMP1 && pMP2)
        {
            if(!pMP1->isBad() && !pMP2->isBad())
            {
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                Eigen::Vector3f P3D1w = pMP1->GetWorldPos();
                P3D1c = R1w*P3D1w + t1w;
                vPoint1->setEstimate(P3D1c.cast<double>());
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                Eigen::Vector3f P3D2w = pMP2->GetWorldPos();
                P3D2c = R2w*P3D2w + t2w;
                vPoint2->setEstimate(P3D2c.cast<double>());
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
            {
                nBadMPs++;
                continue;
            }
        }
        else
        {
            nMatchWithoutMP++;

            //TODO The 3D position in KF1 doesn't exist
            if(!pMP2->isBad())
            {
                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                Eigen::Vector3f P3D2w = pMP2->GetWorldPos();
                P3D2c = R2w*P3D2w + t2w;
                vPoint2->setEstimate(P3D2c.cast<double>());
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);

                vIdsOnlyInKF2.push_back(id2);
            }
            continue;
        }

        if(i2<0 && !bAllPoints)
        {
            Verbose::PrintMess("    Remove point -> i2: " + to_string(i2) + "; bAllPoints: " + to_string(bAllPoints), Verbose::VERBOSITY_DEBUG);
            continue;
        }

        if(P3D2c(2) < 0)
        {
            Verbose::PrintMess("Sim3: Z coordinate is negative", Verbose::VERBOSITY_DEBUG);
            continue;
        }

        nCorrespondences++;

        // Set edge x1 = S12*X2
        Eigen::Matrix<double,2,1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        ORB_SLAM3::EdgeSim3ProjectXYZ* e12 = new ORB_SLAM3::EdgeSim3ProjectXYZ();

        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        Eigen::Matrix<double,2,1> obs2;
        cv::KeyPoint kpUn2;
        bool inKF2;
        if(i2 >= 0)
        {
            kpUn2 = pKF2->mvKeysUn[i2];
            obs2 << kpUn2.pt.x, kpUn2.pt.y;
            inKF2 = true;

            nInKF2++;
        }
        else
        {
            float invz = 1/P3D2c(2);
            float x = P3D2c(0)*invz;
            float y = P3D2c(1)*invz;

            obs2 << x, y;
            kpUn2 = cv::KeyPoint(cv::Point2f(x, y), pMP2->mnTrackScaleLevel);

            inKF2 = false;
            nOutKF2++;
        }

        ORB_SLAM3::EdgeInverseSim3ProjectXYZ* e21 = new ORB_SLAM3::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);

        vbIsInKF2.push_back(inKF2);
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers
    int nBad=0;
    int nBadOutKF2 = 0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        ORB_SLAM3::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        ORB_SLAM3::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i]=static_cast<ORB_SLAM3::EdgeSim3ProjectXYZ*>(NULL);
            vpEdges21[i]=static_cast<ORB_SLAM3::EdgeInverseSim3ProjectXYZ*>(NULL);
            nBad++;

            if(!vbIsInKF2[i])
            {
                nBadOutKF2++;
            }
            continue;
        }

        //Check if remove the robust adjustment improve the result
        e12->setRobustKernel(0);
        e21->setRobustKernel(0);
    }

    int nMoreIterations;
    if(nBad>0)
        nMoreIterations=10;
    else
        nMoreIterations=5;

    if(nCorrespondences-nBad<10)
        return 0;

    // Optimize again only with inliers
    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    mAcumHessian = Eigen::MatrixXd::Zero(7, 7);
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        ORB_SLAM3::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        ORB_SLAM3::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        e12->computeError();
        e21->computeError();

        if(e12->chi2()>th2 || e21->chi2()>th2){
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
        }
        else{
            nIn++;
        }
    }

    // Recover optimized Sim3
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12= vSim3_recov->estimate();

    return nIn;
}

void Optimizer::LocalInertialBA(KeyFrame *pKF, bool *pbStopFlag, Map *pMap, int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges, bool bLarge, bool bRecInit)
{
    Map* pCurrentMap = pKF->GetMap();

    int maxOpt=10;
    int opt_it=10;
    if(bLarge)
    {
        maxOpt=25;
        opt_it=4;
    }
    const int Nd = std::min((int)pCurrentMap->KeyFramesInMap()-2,maxOpt);
    const unsigned long maxKFid = pKF->mnId;

    vector<KeyFrame*> vpOptimizableKFs;
    const vector<KeyFrame*> vpNeighsKFs = pKF->GetVectorCovisibleKeyFrames();
    list<KeyFrame*> lpOptVisKFs;

    vpOptimizableKFs.reserve(Nd);
    vpOptimizableKFs.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    for(int i=1; i<Nd; i++)
    {
        if(vpOptimizableKFs.back()->mPrevKF)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
            vpOptimizableKFs.back()->mnBALocalForKF = pKF->mnId;
        }
        else
            break;
    }

    int N = vpOptimizableKFs.size();

    // Optimizable points seen by temporal optimizable keyframes
    list<MapPoint*> lLocalMapPoints;
    for(int i=0; i<N; i++)
    {
        vector<MapPoint*> vpMPs = vpOptimizableKFs[i]->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
        }
    }

    // Fixed Keyframe: First frame previous KF to optimization window)
    list<KeyFrame*> lFixedKeyFrames;
    if(vpOptimizableKFs.back()->mPrevKF)
    {
        lFixedKeyFrames.push_back(vpOptimizableKFs.back()->mPrevKF);
        vpOptimizableKFs.back()->mPrevKF->mnBAFixedForKF=pKF->mnId;
    }
    else
    {
        vpOptimizableKFs.back()->mnBALocalForKF=0;
        vpOptimizableKFs.back()->mnBAFixedForKF=pKF->mnId;
        lFixedKeyFrames.push_back(vpOptimizableKFs.back());
        vpOptimizableKFs.pop_back();
    }

    // Optimizable visual KFs
    const int maxCovKF = 0;
    for(int i=0, iend=vpNeighsKFs.size(); i<iend; i++)
    {
        if(lpOptVisKFs.size() >= maxCovKF)
            break;

        KeyFrame* pKFi = vpNeighsKFs[i];
        if(pKFi->mnBALocalForKF == pKF->mnId || pKFi->mnBAFixedForKF == pKF->mnId)
            continue;
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
        {
            lpOptVisKFs.push_back(pKFi);

            vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
            for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
            {
                MapPoint* pMP = *vit;
                if(pMP)
                    if(!pMP->isBad())
                        if(pMP->mnBALocalForKF!=pKF->mnId)
                        {
                            lLocalMapPoints.push_back(pMP);
                            pMP->mnBALocalForKF=pKF->mnId;
                        }
            }
        }
    }

    // Fixed KFs which are not covisible optimizable
    const int maxFixKF = 200;

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,tuple<int,int>> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
            {
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                {
                    lFixedKeyFrames.push_back(pKFi);
                    break;
                }
            }
        }
        if(lFixedKeyFrames.size()>=maxFixKF)
            break;
    }

    bool bNonFixed = (lFixedKeyFrames.size() == 0);

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    if(bLarge)
    {
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        solver->setUserLambdaInit(1e-2); // to avoid iterating for finding optimal lambda
        optimizer.setAlgorithm(solver);
    }
    else
    {
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        solver->setUserLambdaInit(1e0);
        optimizer.setAlgorithm(solver);
    }


    // Set Local temporal KeyFrame vertices
    N=vpOptimizableKFs.size();
    for(int i=0; i<N; i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];

        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(false);
        optimizer.addVertex(VP);

        if(pKFi->bImu)
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+3*(pKFi->mnId)+1);
            VV->setFixed(false);
            optimizer.addVertex(VV);
            VertexGyroBias* VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid+3*(pKFi->mnId)+2);
            VG->setFixed(false);
            optimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid+3*(pKFi->mnId)+3);
            VA->setFixed(false);
            optimizer.addVertex(VA);
        }
    }

    // Set Local visual KeyFrame vertices
    for(list<KeyFrame*>::iterator it=lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it!=itEnd; it++)
    {
        KeyFrame* pKFi = *it;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(false);
        optimizer.addVertex(VP);
    }

    // Set Fixed KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        if(pKFi->bImu) // This should be done only for keyframe just before temporal window
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+3*(pKFi->mnId)+1);
            VV->setFixed(true);
            optimizer.addVertex(VV);
            VertexGyroBias* VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid+3*(pKFi->mnId)+2);
            VG->setFixed(true);
            optimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid+3*(pKFi->mnId)+3);
            VA->setFixed(true);
            optimizer.addVertex(VA);
        }
    }

    // Create intertial constraints
    vector<EdgeInertial*> vei(N,(EdgeInertial*)NULL);
    vector<EdgeGyroRW*> vegr(N,(EdgeGyroRW*)NULL);
    vector<EdgeAccRW*> vear(N,(EdgeAccRW*)NULL);

    for(int i=0;i<N;i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];

        if(!pKFi->mPrevKF)
        {
            cout << "NOT INERTIAL LINK TO PREVIOUS FRAME!!!!" << endl;
            continue;
        }
        if(pKFi->bImu && pKFi->mPrevKF->bImu && pKFi->mpImuPreintegrated)
        {
            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+1);
            g2o::HyperGraph::Vertex* VG1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+2);
            g2o::HyperGraph::Vertex* VA1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+3);
            g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+1);
            g2o::HyperGraph::Vertex* VG2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+2);
            g2o::HyperGraph::Vertex* VA2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+3);

            if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
            {
                cerr << "Error " << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;
                continue;
            }

            vei[i] = new EdgeInertial(pKFi->mpImuPreintegrated);

            vei[i]->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            vei[i]->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            vei[i]->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
            vei[i]->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
            vei[i]->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            vei[i]->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

            if(i==N-1 || bRecInit)
            {
                // All inertial residuals are included without robust cost function, but not that one linking the
                // last optimizable keyframe inside of the local window and the first fixed keyframe out. The
                // information matrix for this measurement is also downweighted. This is done to avoid accumulating
                // error due to fixing variables.
                g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
                vei[i]->setRobustKernel(rki);
                if(i==N-1)
                    vei[i]->setInformation(vei[i]->information()*1e-2);
                rki->setDelta(sqrt(16.92));
            }
            optimizer.addEdge(vei[i]);

            vegr[i] = new EdgeGyroRW();
            vegr[i]->setVertex(0,VG1);
            vegr[i]->setVertex(1,VG2);
            Eigen::Matrix3d InfoG = pKFi->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse();
            vegr[i]->setInformation(InfoG);
            optimizer.addEdge(vegr[i]);

            vear[i] = new EdgeAccRW();
            vear[i]->setVertex(0,VA1);
            vear[i]->setVertex(1,VA2);
            Eigen::Matrix3d InfoA = pKFi->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse();
            vear[i]->setInformation(InfoA);           

            optimizer.addEdge(vear[i]);
        }
        else
            cout << "ERROR building inertial edge" << endl;
    }

    // Set MapPoint vertices
    const int nExpectedSize = (N+lFixedKeyFrames.size())*lLocalMapPoints.size();

    // Mono
    vector<EdgeMono*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    // Stereo
    vector<EdgeStereo*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);



    const float thHuberMono = sqrt(5.991);
    const float chi2Mono2 = 5.991;
    const float thHuberStereo = sqrt(7.815);
    const float chi2Stereo2 = 7.815;

    const unsigned long iniMPid = maxKFid*5;

    map<int,int> mVisEdges;
    for(int i=0;i<N;i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];
        mVisEdges[pKFi->mnId] = 0;
    }
    for(list<KeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
    {
        mVisEdges[(*lit)->mnId] = 0;
    }

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());

        unsigned long id = pMP->mnId+iniMPid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

        // Create visual constraints
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
                continue;

            if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            {
                const int leftIndex = get<0>(mit->second);

                cv::KeyPoint kpUn;

                // Monocular left observation
                if(leftIndex != -1 && pKFi->mvuRight[leftIndex]<0)
                {
                    mVisEdges[pKFi->mnId]++;

                    kpUn = pKFi->mvKeysUn[leftIndex];
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMono* e = new EdgeMono(0);

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pKFi->mpCamera->uncertainty2(obs);

                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                // Stereo-observation
                else if(leftIndex != -1)// Stereo observation
                {
                    kpUn = pKFi->mvKeysUn[leftIndex];
                    mVisEdges[pKFi->mnId]++;

                    const float kp_ur = pKFi->mvuRight[leftIndex];
                    Eigen::Matrix<double,3,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    EdgeStereo* e = new EdgeStereo(0);

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pKFi->mpCamera->uncertainty2(obs.head(2));

                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }

                // Monocular right observation
                if(pKFi->mpCamera2){
                    int rightIndex = get<1>(mit->second);

                    if(rightIndex != -1 ){
                        rightIndex -= pKFi->NLeft;
                        mVisEdges[pKFi->mnId]++;

                        Eigen::Matrix<double,2,1> obs;
                        cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
                        obs << kp.pt.x, kp.pt.y;

                        EdgeMono* e = new EdgeMono(1);

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pKFi->mpCamera->uncertainty2(obs);

                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave]/unc2;
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);
                        vpEdgesMono.push_back(e);
                        vpEdgeKFMono.push_back(pKFi);
                        vpMapPointEdgeMono.push_back(pMP);
                    }
                }
            }
        }
    }

    //cout << "Total map points: " << lLocalMapPoints.size() << endl;
    for(map<int,int>::iterator mit=mVisEdges.begin(), mend=mVisEdges.end(); mit!=mend; mit++)
    {
        assert(mit->second>=3);
    }

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    float err = optimizer.activeRobustChi2();
    optimizer.optimize(opt_it); // Originally to 2
    float err_end = optimizer.activeRobustChi2();
    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

    // Check inlier observations
    // Mono
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        EdgeMono* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];
        bool bClose = pMP->mTrackDepth<10.f;

        if(pMP->isBad())
            continue;

        if((e->chi2()>chi2Mono2 && !bClose) || (e->chi2()>1.5f*chi2Mono2 && bClose) || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }


    // Stereo
    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        EdgeStereo* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>chi2Stereo2)
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Get Map Mutex and erase outliers
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);


    // TODO: Some convergence problems have been detected here
    if((2*err < err_end || isnan(err) || isnan(err_end)) && !bLarge) //bGN)
    {
        cout << "FAIL LOCAL-INERTIAL BA!!!!" << endl;
        return;
    }



    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    for(list<KeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
        (*lit)->mnBAFixedForKF = 0;

    // Recover optimized data
    // Local temporal Keyframes
    N=vpOptimizableKFs.size();
    for(int i=0; i<N; i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];

        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        Sophus::SE3f Tcw(VP->estimate().Rcw[0].cast<float>(), VP->estimate().tcw[0].cast<float>());
        pKFi->SetPose(Tcw);
        pKFi->mnBALocalForKF=0;

        if(pKFi->bImu)
        {
            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+1));
            pKFi->SetVelocity(VV->estimate().cast<float>());
            VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+2));
            VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+3));
            Vector6d b;
            b << VG->estimate(), VA->estimate();
            pKFi->SetNewBias(IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]));

        }
    }

    // Local visual KeyFrame
    for(list<KeyFrame*>::iterator it=lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it!=itEnd; it++)
    {
        KeyFrame* pKFi = *it;
        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        Sophus::SE3f Tcw(VP->estimate().Rcw[0].cast<float>(), VP->estimate().tcw[0].cast<float>());
        pKFi->SetPose(Tcw);
        pKFi->mnBALocalForKF=0;
    }

    //Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+iniMPid+1));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
        pMP->UpdateNormalAndDepth();
    }

    pMap->IncreaseChangeIndex();
}

Eigen::MatrixXd Optimizer::Marginalize(const Eigen::MatrixXd &H, const int &start, const int &end)
{
    // Goal
    // a  | ab | ac       a*  | 0 | ac*
    // ba | b  | bc  -->  0   | 0 | 0
    // ca | cb | c        ca* | 0 | c*

    // Size of block before block to marginalize
    const int a = start;
    // Size of block to marginalize
    const int b = end-start+1;
    // Size of block after block to marginalize
    const int c = H.cols() - (end+1);

    // Reorder as follows:
    // a  | ab | ac       a  | ac | ab
    // ba | b  | bc  -->  ca | c  | cb
    // ca | cb | c        ba | bc | b

    Eigen::MatrixXd Hn = Eigen::MatrixXd::Zero(H.rows(),H.cols());
    if(a>0)
    {
        Hn.block(0,0,a,a) = H.block(0,0,a,a);
        Hn.block(0,a+c,a,b) = H.block(0,a,a,b);
        Hn.block(a+c,0,b,a) = H.block(a,0,b,a);
    }
    if(a>0 && c>0)
    {
        Hn.block(0,a,a,c) = H.block(0,a+b,a,c);
        Hn.block(a,0,c,a) = H.block(a+b,0,c,a);
    }
    if(c>0)
    {
        Hn.block(a,a,c,c) = H.block(a+b,a+b,c,c);
        Hn.block(a,a+c,c,b) = H.block(a+b,a,c,b);
        Hn.block(a+c,a,b,c) = H.block(a,a+b,b,c);
    }
    Hn.block(a+c,a+c,b,b) = H.block(a,a,b,b);

    // Perform marginalization (Schur complement)
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Hn.block(a+c,a+c,b,b),Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType singularValues_inv=svd.singularValues();
    for (int i=0; i<b; ++i)
    {
        if (singularValues_inv(i)>1e-6)
            singularValues_inv(i)=1.0/singularValues_inv(i);
        else singularValues_inv(i)=0;
    }
    Eigen::MatrixXd invHb = svd.matrixV()*singularValues_inv.asDiagonal()*svd.matrixU().transpose();
    Hn.block(0,0,a+c,a+c) = Hn.block(0,0,a+c,a+c) - Hn.block(0,a+c,a+c,b)*invHb*Hn.block(a+c,0,b,a+c);
    Hn.block(a+c,a+c,b,b) = Eigen::MatrixXd::Zero(b,b);
    Hn.block(0,a+c,a+c,b) = Eigen::MatrixXd::Zero(a+c,b);
    Hn.block(a+c,0,b,a+c) = Eigen::MatrixXd::Zero(b,a+c);

    // Inverse reorder
    // a*  | ac* | 0       a*  | 0 | ac*
    // ca* | c*  | 0  -->  0   | 0 | 0
    // 0   | 0   | 0       ca* | 0 | c*
    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(H.rows(),H.cols());
    if(a>0)
    {
        res.block(0,0,a,a) = Hn.block(0,0,a,a);
        res.block(0,a,a,b) = Hn.block(0,a+c,a,b);
        res.block(a,0,b,a) = Hn.block(a+c,0,b,a);
    }
    if(a>0 && c>0)
    {
        res.block(0,a+b,a,c) = Hn.block(0,a,a,c);
        res.block(a+b,0,c,a) = Hn.block(a,0,c,a);
    }
    if(c>0)
    {
        res.block(a+b,a+b,c,c) = Hn.block(a,a,c,c);
        res.block(a+b,a,c,b) = Hn.block(a,a+c,c,b);
        res.block(a,a+b,b,c) = Hn.block(a+c,a,b,c);
    }

    res.block(a,a,b,b) = Hn.block(a+c,a+c,b,b);

    return res;
}

void Optimizer::InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale, Eigen::Vector3d &bg, Eigen::Vector3d &ba, bool bMono, Eigen::MatrixXd  &covInertial, bool bFixedVel, bool bGauss, float priorG, float priorA)
{
    Verbose::PrintMess("inertial optimization", Verbose::VERBOSITY_NORMAL);
    int its = 200;
    long unsigned int maxKFid = pMap->GetMaxKFid();
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    if (priorG!=0.f)
        solver->setUserLambdaInit(1e3);

    optimizer.setAlgorithm(solver);

    // Set KeyFrame vertices (fixed poses and optimizable velocities)
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        VertexVelocity* VV = new VertexVelocity(pKFi);
        VV->setId(maxKFid+(pKFi->mnId)+1);
        if (bFixedVel)
            VV->setFixed(true);
        else
            VV->setFixed(false);

        optimizer.addVertex(VV);
    }

    // Biases
    VertexGyroBias* VG = new VertexGyroBias(vpKFs.front());
    VG->setId(maxKFid*2+2);
    if (bFixedVel)
        VG->setFixed(true);
    else
        VG->setFixed(false);
    optimizer.addVertex(VG);
    VertexAccBias* VA = new VertexAccBias(vpKFs.front());
    VA->setId(maxKFid*2+3);
    if (bFixedVel)
        VA->setFixed(true);
    else
        VA->setFixed(false);

    optimizer.addVertex(VA);
    // prior acc bias
    Eigen::Vector3f bprior;
    bprior.setZero();

    EdgePriorAcc* epa = new EdgePriorAcc(bprior);
    epa->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
    double infoPriorA = priorA;
    epa->setInformation(infoPriorA*Eigen::Matrix3d::Identity());
    optimizer.addEdge(epa);
    EdgePriorGyro* epg = new EdgePriorGyro(bprior);
    epg->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
    double infoPriorG = priorG;
    epg->setInformation(infoPriorG*Eigen::Matrix3d::Identity());
    optimizer.addEdge(epg);

    // Gravity and scale
    VertexGDir* VGDir = new VertexGDir(Rwg);
    VGDir->setId(maxKFid*2+4);
    VGDir->setFixed(false);
    optimizer.addVertex(VGDir);
    VertexScale* VS = new VertexScale(scale);
    VS->setId(maxKFid*2+5);
    VS->setFixed(!bMono); // Fixed for stereo case
    optimizer.addVertex(VS);

    // Graph edges
    // IMU links with gravity and scale
    vector<EdgeInertialGS*> vpei;
    vpei.reserve(vpKFs.size());
    vector<pair<KeyFrame*,KeyFrame*> > vppUsedKF;
    vppUsedKF.reserve(vpKFs.size());
    //std::cout << "build optimization graph" << std::endl;

    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        if(pKFi->mPrevKF && pKFi->mnId<=maxKFid)
        {
            if(pKFi->isBad() || pKFi->mPrevKF->mnId>maxKFid)
                continue;
            if(!pKFi->mpImuPreintegrated)
                std::cout << "Not preintegrated measurement" << std::endl;

            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+(pKFi->mPrevKF->mnId)+1);
            g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+(pKFi->mnId)+1);
            g2o::HyperGraph::Vertex* VG = optimizer.vertex(maxKFid*2+2);
            g2o::HyperGraph::Vertex* VA = optimizer.vertex(maxKFid*2+3);
            g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(maxKFid*2+4);
            g2o::HyperGraph::Vertex* VS = optimizer.vertex(maxKFid*2+5);
            if(!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
            {
                cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG << ", "<< VA << ", " << VP2 << ", " << VV2 <<  ", "<< VGDir << ", "<< VS <<endl;

                continue;
            }
            EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
            ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
            ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
            ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            ei->setVertex(6,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
            ei->setVertex(7,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS));

            vpei.push_back(ei);

            vppUsedKF.push_back(make_pair(pKFi->mPrevKF,pKFi));
            optimizer.addEdge(ei);

        }
    }

    // Compute error for different scales
    std::set<g2o::HyperGraph::Edge*> setEdges = optimizer.edges();

    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(its);

    scale = VS->estimate();

    // Recover optimized data
    // Biases
    VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid*2+2));
    VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid*2+3));
    Vector6d vb;
    vb << VG->estimate(), VA->estimate();
    bg << VG->estimate();
    ba << VA->estimate();
    scale = VS->estimate();


    IMU::Bias b (vb[3],vb[4],vb[5],vb[0],vb[1],vb[2]);
    Rwg = VGDir->estimate().Rwg;

    //Keyframes velocities and biases
    const int N = vpKFs.size();
    for(size_t i=0; i<N; i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;

        VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+(pKFi->mnId)+1));
        Eigen::Vector3d Vw = VV->estimate(); // Velocity is scaled after
        pKFi->SetVelocity(Vw.cast<float>());

        if ((pKFi->GetGyroBias() - bg.cast<float>()).norm() > 0.01)
        {
            pKFi->SetNewBias(b);
            if (pKFi->mpImuPreintegrated)
                pKFi->mpImuPreintegrated->Reintegrate();
        }
        else
            pKFi->SetNewBias(b);


    }
}


void Optimizer::InertialOptimization(Map *pMap, Eigen::Vector3d &bg, Eigen::Vector3d &ba, float priorG, float priorA)
{
    int its = 200; // Check number of iterations
    long unsigned int maxKFid = pMap->GetMaxKFid();
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    solver->setUserLambdaInit(1e3);

    optimizer.setAlgorithm(solver);

    // Set KeyFrame vertices (fixed poses and optimizable velocities)
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        VertexVelocity* VV = new VertexVelocity(pKFi);
        VV->setId(maxKFid+(pKFi->mnId)+1);
        VV->setFixed(false);

        optimizer.addVertex(VV);
    }

    // Biases
    VertexGyroBias* VG = new VertexGyroBias(vpKFs.front());
    VG->setId(maxKFid*2+2);
    VG->setFixed(false);
    optimizer.addVertex(VG);

    VertexAccBias* VA = new VertexAccBias(vpKFs.front());
    VA->setId(maxKFid*2+3);
    VA->setFixed(false);

    optimizer.addVertex(VA);
    // prior acc bias
    Eigen::Vector3f bprior;
    bprior.setZero();

    EdgePriorAcc* epa = new EdgePriorAcc(bprior);
    epa->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
    double infoPriorA = priorA;
    epa->setInformation(infoPriorA*Eigen::Matrix3d::Identity());
    optimizer.addEdge(epa);
    EdgePriorGyro* epg = new EdgePriorGyro(bprior);
    epg->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
    double infoPriorG = priorG;
    epg->setInformation(infoPriorG*Eigen::Matrix3d::Identity());
    optimizer.addEdge(epg);

    // Gravity and scale
    VertexGDir* VGDir = new VertexGDir(Eigen::Matrix3d::Identity());
    VGDir->setId(maxKFid*2+4);
    VGDir->setFixed(true);
    optimizer.addVertex(VGDir);
    VertexScale* VS = new VertexScale(1.0);
    VS->setId(maxKFid*2+5);
    VS->setFixed(true); // Fixed since scale is obtained from already well initialized map
    optimizer.addVertex(VS);

    // Graph edges
    // IMU links with gravity and scale
    vector<EdgeInertialGS*> vpei;
    vpei.reserve(vpKFs.size());
    vector<pair<KeyFrame*,KeyFrame*> > vppUsedKF;
    vppUsedKF.reserve(vpKFs.size());

    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        if(pKFi->mPrevKF && pKFi->mnId<=maxKFid)
        {
            if(pKFi->isBad() || pKFi->mPrevKF->mnId>maxKFid)
                continue;

            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+(pKFi->mPrevKF->mnId)+1);
            g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+(pKFi->mnId)+1);
            g2o::HyperGraph::Vertex* VG = optimizer.vertex(maxKFid*2+2);
            g2o::HyperGraph::Vertex* VA = optimizer.vertex(maxKFid*2+3);
            g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(maxKFid*2+4);
            g2o::HyperGraph::Vertex* VS = optimizer.vertex(maxKFid*2+5);
            if(!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
            {
                cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG << ", "<< VA << ", " << VP2 << ", " << VV2 <<  ", "<< VGDir << ", "<< VS <<endl;

                continue;
            }
            EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
            ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
            ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
            ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            ei->setVertex(6,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
            ei->setVertex(7,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS));

            vpei.push_back(ei);

            vppUsedKF.push_back(make_pair(pKFi->mPrevKF,pKFi));
            optimizer.addEdge(ei);

        }
    }

    // Compute error for different scales
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(its);


    // Recover optimized data
    // Biases
    VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid*2+2));
    VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid*2+3));
    Vector6d vb;
    vb << VG->estimate(), VA->estimate();
    bg << VG->estimate();
    ba << VA->estimate();

    IMU::Bias b (vb[3],vb[4],vb[5],vb[0],vb[1],vb[2]);

    //Keyframes velocities and biases
    const int N = vpKFs.size();
    for(size_t i=0; i<N; i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;

        VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+(pKFi->mnId)+1));
        Eigen::Vector3d Vw = VV->estimate();
        pKFi->SetVelocity(Vw.cast<float>());

        if ((pKFi->GetGyroBias() - bg.cast<float>()).norm() > 0.01)
        {
            pKFi->SetNewBias(b);
            if (pKFi->mpImuPreintegrated)
                pKFi->mpImuPreintegrated->Reintegrate();
        }
        else
            pKFi->SetNewBias(b);
    }
}

void Optimizer::InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale)
{
    int its = 10;
    long unsigned int maxKFid = pMap->GetMaxKFid();
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Set KeyFrame vertices (all variables are fixed)
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        VertexVelocity* VV = new VertexVelocity(pKFi);
        VV->setId(maxKFid+1+(pKFi->mnId));
        VV->setFixed(true);
        optimizer.addVertex(VV);

        // Vertex of fixed biases
        VertexGyroBias* VG = new VertexGyroBias(vpKFs.front());
        VG->setId(2*(maxKFid+1)+(pKFi->mnId));
        VG->setFixed(true);
        optimizer.addVertex(VG);
        VertexAccBias* VA = new VertexAccBias(vpKFs.front());
        VA->setId(3*(maxKFid+1)+(pKFi->mnId));
        VA->setFixed(true);
        optimizer.addVertex(VA);
    }

    // Gravity and scale
    VertexGDir* VGDir = new VertexGDir(Rwg);
    VGDir->setId(4*(maxKFid+1));
    VGDir->setFixed(false);
    optimizer.addVertex(VGDir);
    VertexScale* VS = new VertexScale(scale);
    VS->setId(4*(maxKFid+1)+1);
    VS->setFixed(false);
    optimizer.addVertex(VS);

    // Graph edges
    int count_edges = 0;
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        if(pKFi->mPrevKF && pKFi->mnId<=maxKFid)
        {
            if(pKFi->isBad() || pKFi->mPrevKF->mnId>maxKFid)
                continue;
                
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex((maxKFid+1)+pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex((maxKFid+1)+pKFi->mnId);
            g2o::HyperGraph::Vertex* VG = optimizer.vertex(2*(maxKFid+1)+pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VA = optimizer.vertex(3*(maxKFid+1)+pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(4*(maxKFid+1));
            g2o::HyperGraph::Vertex* VS = optimizer.vertex(4*(maxKFid+1)+1);
            if(!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
            {
                Verbose::PrintMess("Error" + to_string(VP1->id()) + ", " + to_string(VV1->id()) + ", " + to_string(VG->id()) + ", " + to_string(VA->id()) + ", " + to_string(VP2->id()) + ", " + to_string(VV2->id()) +  ", " + to_string(VGDir->id()) + ", " + to_string(VS->id()), Verbose::VERBOSITY_NORMAL);

                continue;
            }
            count_edges++;
            EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
            ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
            ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
            ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            ei->setVertex(6,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
            ei->setVertex(7,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS));
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            ei->setRobustKernel(rk);
            rk->setDelta(1.f);
            optimizer.addEdge(ei);
        }
    }

    // Compute error for different scales
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    float err = optimizer.activeRobustChi2();
    optimizer.optimize(its);
    optimizer.computeActiveErrors();
    float err_end = optimizer.activeRobustChi2();
    // Recover optimized data
    scale = VS->estimate();
    Rwg = VGDir->estimate().Rwg;
}

void Optimizer::LocalBundleAdjustment(KeyFrame* pMainKF,vector<KeyFrame*> vpAdjustKF, vector<KeyFrame*> vpFixedKF, bool *pbStopFlag)
{
    bool bShowImages = false;

    vector<MapPoint*> vpMPs;

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    optimizer.setVerbose(false);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;
    set<KeyFrame*> spKeyFrameBA;

    Map* pCurrentMap = pMainKF->GetMap();

    // Set fixed KeyFrame vertices
    int numInsertedPoints = 0;
    for(KeyFrame* pKFi : vpFixedKF)
    {
        if(pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
        {
            Verbose::PrintMess("ERROR LBA: KF is bad or is not in the current map", Verbose::VERBOSITY_NORMAL);
            continue;
        }

        pKFi->mnBALocalForMerge = pMainKF->mnId;

        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;

        set<MapPoint*> spViewMPs = pKFi->GetMapPoints();
        for(MapPoint* pMPi : spViewMPs)
        {
            if(pMPi)
                if(!pMPi->isBad() && pMPi->GetMap() == pCurrentMap)

                    if(pMPi->mnBALocalForMerge!=pMainKF->mnId)
                    {
                        vpMPs.push_back(pMPi);
                        pMPi->mnBALocalForMerge=pMainKF->mnId;
                        numInsertedPoints++;
                    }
        }

        spKeyFrameBA.insert(pKFi);
    }

    // Set non fixed Keyframe vertices
    set<KeyFrame*> spAdjustKF(vpAdjustKF.begin(), vpAdjustKF.end());
    numInsertedPoints = 0;
    for(KeyFrame* pKFi : vpAdjustKF)
    {
        if(pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
            continue;

        pKFi->mnBALocalForMerge = pMainKF->mnId;

        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;

        set<MapPoint*> spViewMPs = pKFi->GetMapPoints();
        for(MapPoint* pMPi : spViewMPs)
        {
            if(pMPi)
            {
                if(!pMPi->isBad() && pMPi->GetMap() == pCurrentMap)
                {
                    if(pMPi->mnBALocalForMerge != pMainKF->mnId)
                    {
                        vpMPs.push_back(pMPi);
                        pMPi->mnBALocalForMerge = pMainKF->mnId;
                        numInsertedPoints++;
                    }
                }
            }
        }

        spKeyFrameBA.insert(pKFi);
    }

    const int nExpectedSize = (vpAdjustKF.size()+vpFixedKF.size())*vpMPs.size();

    vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices
    map<KeyFrame*, int> mpObsKFs;
    map<KeyFrame*, int> mpObsFinalKFs;
    map<MapPoint*, int> mpObsMPs;
    for(unsigned int i=0; i < vpMPs.size(); ++i)
    {
        MapPoint* pMPi = vpMPs[i];
        if(pMPi->isBad())
            continue;

        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMPi->GetWorldPos().cast<double>());
        const int id = pMPi->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);


        const map<KeyFrame*,tuple<int,int>> observations = pMPi->GetObservations();
        int nEdges = 0;
        //SET EDGES
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {
            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid || pKF->mnBALocalForMerge != pMainKF->mnId || !pKF->GetMapPoint(get<0>(mit->second)))
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[get<0>(mit->second)];

            if(pKF->mvuRight[get<0>(mit->second)]<0) //Monocular
            {
                mpObsMPs[pMPi]++;
                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuber2D);

                e->pCamera = pKF->mpCamera;

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vpEdgeKFMono.push_back(pKF);
                vpMapPointEdgeMono.push_back(pMPi);

                mpObsKFs[pKF]++;
            }
            else // RGBD or Stereo
            {
                mpObsMPs[pMPi]+=2;
                Eigen::Matrix<double,3,1> obs;
                const float kp_ur = pKF->mvuRight[get<0>(mit->second)];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuber3D);

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vpEdgeKFStereo.push_back(pKF);
                vpMapPointEdgeStereo.push_back(pMPi);

                mpObsKFs[pKF]++;
            }
        }
    }

    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore= true;

    if(pbStopFlag)
        if(*pbStopFlag)
            bDoMore = false;

    map<unsigned long int, int> mWrongObsKF;
    if(bDoMore)
    {
        // Check inlier observations
        int badMonoMP = 0, badStereoMP = 0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                e->setLevel(1);
                badMonoMP++;
            }
            e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
            MapPoint* pMP = vpMapPointEdgeStereo[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>7.815 || !e->isDepthPositive())
            {
                e->setLevel(1);
                badStereoMP++;
            }

            e->setRobustKernel(0);
        }
        Verbose::PrintMess("[BA]: First optimization(Huber), there are " + to_string(badMonoMP) + " monocular and " + to_string(badStereoMP) + " stereo bad edges", Verbose::VERBOSITY_DEBUG);

    optimizer.initializeOptimization(0);
    optimizer.optimize(10);
    }

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());
    set<MapPoint*> spErasedMPs;
    set<KeyFrame*> spErasedKFs;

    // Check inlier observations
    int badMonoMP = 0, badStereoMP = 0;
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
            mWrongObsKF[pKFi->mnId]++;
            badMonoMP++;

            spErasedMPs.insert(pMP);
            spErasedKFs.insert(pKFi);
        }
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
            mWrongObsKF[pKFi->mnId]++;
            badStereoMP++;

            spErasedMPs.insert(pMP);
            spErasedKFs.insert(pKFi);
        }
    }

    Verbose::PrintMess("[BA]: Second optimization, there are " + to_string(badMonoMP) + " monocular and " + to_string(badStereoMP) + " sterero bad edges", Verbose::VERBOSITY_DEBUG);

    // Get Map Mutex
    unique_lock<mutex> lock(pMainKF->GetMap()->mMutexMapUpdate);

    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }
    for(unsigned int i=0; i < vpMPs.size(); ++i)
    {
        MapPoint* pMPi = vpMPs[i];
        if(pMPi->isBad())
            continue;

        const map<KeyFrame*,tuple<int,int>> observations = pMPi->GetObservations();
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {
            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid || pKF->mnBALocalForKF != pMainKF->mnId || !pKF->GetMapPoint(get<0>(mit->second)))
                continue;

            if(pKF->mvuRight[get<0>(mit->second)]<0) //Monocular
            {
                mpObsFinalKFs[pKF]++;
            }
            else // RGBD or Stereo
            {
                mpObsFinalKFs[pKF]++;
            }
        }
    }

    // Recover optimized data
    // Keyframes
    for(KeyFrame* pKFi : vpAdjustKF)
    {
        if(pKFi->isBad())
            continue;

        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        Sophus::SE3f Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());

        int numMonoBadPoints = 0, numMonoOptPoints = 0;
        int numStereoBadPoints = 0, numStereoOptPoints = 0;
        vector<MapPoint*> vpMonoMPsOpt, vpStereoMPsOpt;
        vector<MapPoint*> vpMonoMPsBad, vpStereoMPsBad;

        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];
            KeyFrame* pKFedge = vpEdgeKFMono[i];

            if(pKFi != pKFedge)
            {
                continue;
            }

            if(pMP->isBad())
                continue;

            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                numMonoBadPoints++;
                vpMonoMPsBad.push_back(pMP);

            }
            else
            {
                numMonoOptPoints++;
                vpMonoMPsOpt.push_back(pMP);
            }

        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
            MapPoint* pMP = vpMapPointEdgeStereo[i];
            KeyFrame* pKFedge = vpEdgeKFMono[i];

            if(pKFi != pKFedge)
            {
                continue;
            }

            if(pMP->isBad())
                continue;

            if(e->chi2()>7.815 || !e->isDepthPositive())
            {
                numStereoBadPoints++;
                vpStereoMPsBad.push_back(pMP);
            }
            else
            {
                numStereoOptPoints++;
                vpStereoMPsOpt.push_back(pMP);
            }
        }

        pKFi->SetPose(Tiw);
    }

    //Points
    for(MapPoint* pMPi : vpMPs)
    {
        if(pMPi->isBad())
            continue;

        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMPi->mnId+maxKFid+1));
        pMPi->SetWorldPos(vPoint->estimate().cast<float>());
        pMPi->UpdateNormalAndDepth();

    }
}


void Optimizer::MergeInertialBA(KeyFrame* pCurrKF, KeyFrame* pMergeKF, bool *pbStopFlag, Map *pMap, LoopClosing::KeyFrameAndPose &corrPoses)
{
    const int Nd = 6;
    const unsigned long maxKFid = pCurrKF->mnId;

    vector<KeyFrame*> vpOptimizableKFs;
    vpOptimizableKFs.reserve(2*Nd);

    // For cov KFS, inertial parameters are not optimized
    const int maxCovKF = 30;
    vector<KeyFrame*> vpOptimizableCovKFs;
    vpOptimizableCovKFs.reserve(maxCovKF);

    // Add sliding window for current KF
    vpOptimizableKFs.push_back(pCurrKF);
    pCurrKF->mnBALocalForKF = pCurrKF->mnId;
    for(int i=1; i<Nd; i++)
    {
        if(vpOptimizableKFs.back()->mPrevKF)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
            vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
        }
        else
            break;
    }

    list<KeyFrame*> lFixedKeyFrames;
    if(vpOptimizableKFs.back()->mPrevKF)
    {
        vpOptimizableCovKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
        vpOptimizableKFs.back()->mPrevKF->mnBALocalForKF=pCurrKF->mnId;
    }
    else
    {
        vpOptimizableCovKFs.push_back(vpOptimizableKFs.back());
        vpOptimizableKFs.pop_back();
    }

    // Add temporal neighbours to merge KF (previous and next KFs)
    vpOptimizableKFs.push_back(pMergeKF);
    pMergeKF->mnBALocalForKF = pCurrKF->mnId;

    // Previous KFs
    for(int i=1; i<(Nd/2); i++)
    {
        if(vpOptimizableKFs.back()->mPrevKF)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
            vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
        }
        else
            break;
    }

    // We fix just once the old map
    if(vpOptimizableKFs.back()->mPrevKF)
    {
        lFixedKeyFrames.push_back(vpOptimizableKFs.back()->mPrevKF);
        vpOptimizableKFs.back()->mPrevKF->mnBAFixedForKF=pCurrKF->mnId;
    }
    else
    {
        vpOptimizableKFs.back()->mnBALocalForKF=0;
        vpOptimizableKFs.back()->mnBAFixedForKF=pCurrKF->mnId;
        lFixedKeyFrames.push_back(vpOptimizableKFs.back());
        vpOptimizableKFs.pop_back();
    }

    // Next KFs
    if(pMergeKF->mNextKF)
    {
        vpOptimizableKFs.push_back(pMergeKF->mNextKF);
        vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
    }

    while(vpOptimizableKFs.size()<(2*Nd))
    {
        if(vpOptimizableKFs.back()->mNextKF)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mNextKF);
            vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
        }
        else
            break;
    }

    int N = vpOptimizableKFs.size();

    // Optimizable points seen by optimizable keyframes
    list<MapPoint*> lLocalMapPoints;
    map<MapPoint*,int> mLocalObs;
    for(int i=0; i<N; i++)
    {
        vector<MapPoint*> vpMPs = vpOptimizableKFs[i]->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            // Using mnBALocalForKF we avoid redundance here, one MP can not be added several times to lLocalMapPoints
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                    if(pMP->mnBALocalForKF!=pCurrKF->mnId)
                    {
                        mLocalObs[pMP]=1;
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pCurrKF->mnId;
                    }
                    else {
                        mLocalObs[pMP]++;
                    }
        }
    }

    std::vector<std::pair<MapPoint*, int>> pairs;
    pairs.reserve(mLocalObs.size());
    for (auto itr = mLocalObs.begin(); itr != mLocalObs.end(); ++itr)
        pairs.push_back(*itr);
    sort(pairs.begin(), pairs.end(),sortByVal);

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    int i=0;
    for(vector<pair<MapPoint*,int>>::iterator lit=pairs.begin(), lend=pairs.end(); lit!=lend; lit++, i++)
    {
        map<KeyFrame*,tuple<int,int>> observations = lit->first->GetObservations();
        if(i>=maxCovKF)
            break;
        for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pCurrKF->mnId && pKFi->mnBAFixedForKF!=pCurrKF->mnId) // If optimizable or already included...
            {
                pKFi->mnBALocalForKF=pCurrKF->mnId;
                if(!pKFi->isBad())
                {
                    vpOptimizableCovKFs.push_back(pKFi);
                    break;
                }
            }
        }
    }

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e3);

    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    // Set Local KeyFrame vertices
    N=vpOptimizableKFs.size();
    for(int i=0; i<N; i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];

        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(false);
        optimizer.addVertex(VP);

        if(pKFi->bImu)
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+3*(pKFi->mnId)+1);
            VV->setFixed(false);
            optimizer.addVertex(VV);
            VertexGyroBias* VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid+3*(pKFi->mnId)+2);
            VG->setFixed(false);
            optimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid+3*(pKFi->mnId)+3);
            VA->setFixed(false);
            optimizer.addVertex(VA);
        }
    }

    // Set Local cov keyframes vertices
    int Ncov=vpOptimizableCovKFs.size();
    for(int i=0; i<Ncov; i++)
    {
        KeyFrame* pKFi = vpOptimizableCovKFs[i];

        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(false);
        optimizer.addVertex(VP);

        if(pKFi->bImu)
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+3*(pKFi->mnId)+1);
            VV->setFixed(false);
            optimizer.addVertex(VV);
            VertexGyroBias* VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid+3*(pKFi->mnId)+2);
            VG->setFixed(false);
            optimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid+3*(pKFi->mnId)+3);
            VA->setFixed(false);
            optimizer.addVertex(VA);
        }
    }

    // Set Fixed KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        if(pKFi->bImu)
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+3*(pKFi->mnId)+1);
            VV->setFixed(true);
            optimizer.addVertex(VV);
            VertexGyroBias* VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid+3*(pKFi->mnId)+2);
            VG->setFixed(true);
            optimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid+3*(pKFi->mnId)+3);
            VA->setFixed(true);
            optimizer.addVertex(VA);
        }
    }

    // Create intertial constraints
    vector<EdgeInertial*> vei(N,(EdgeInertial*)NULL);
    vector<EdgeGyroRW*> vegr(N,(EdgeGyroRW*)NULL);
    vector<EdgeAccRW*> vear(N,(EdgeAccRW*)NULL);
    for(int i=0;i<N;i++)
    {
        //cout << "inserting inertial edge " << i << endl;
        KeyFrame* pKFi = vpOptimizableKFs[i];

        if(!pKFi->mPrevKF)
        {
            Verbose::PrintMess("NOT INERTIAL LINK TO PREVIOUS FRAME!!!!", Verbose::VERBOSITY_NORMAL);
            continue;
        }
        if(pKFi->bImu && pKFi->mPrevKF->bImu && pKFi->mpImuPreintegrated)
        {
            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+1);
            g2o::HyperGraph::Vertex* VG1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+2);
            g2o::HyperGraph::Vertex* VA1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+3);
            g2o::HyperGraph::Vertex* VP2 = optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+1);
            g2o::HyperGraph::Vertex* VG2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+2);
            g2o::HyperGraph::Vertex* VA2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+3);

            if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
            {
                cerr << "Error " << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;
                continue;
            }

            vei[i] = new EdgeInertial(pKFi->mpImuPreintegrated);

            vei[i]->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            vei[i]->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            vei[i]->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
            vei[i]->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
            vei[i]->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            vei[i]->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

            // TODO Uncomment
            g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
            vei[i]->setRobustKernel(rki);
            rki->setDelta(sqrt(16.92));
            optimizer.addEdge(vei[i]);

            vegr[i] = new EdgeGyroRW();
            vegr[i]->setVertex(0,VG1);
            vegr[i]->setVertex(1,VG2);
            Eigen::Matrix3d InfoG = pKFi->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse();
            vegr[i]->setInformation(InfoG);
            optimizer.addEdge(vegr[i]);

            vear[i] = new EdgeAccRW();
            vear[i]->setVertex(0,VA1);
            vear[i]->setVertex(1,VA2);
            Eigen::Matrix3d InfoA = pKFi->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse();
            vear[i]->setInformation(InfoA);
            optimizer.addEdge(vear[i]);
        }
        else
            Verbose::PrintMess("ERROR building inertial edge", Verbose::VERBOSITY_NORMAL);
    }

    Verbose::PrintMess("end inserting inertial edges", Verbose::VERBOSITY_NORMAL);


    // Set MapPoint vertices
    const int nExpectedSize = (N+Ncov+lFixedKeyFrames.size())*lLocalMapPoints.size();

    // Mono
    vector<EdgeMono*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    // Stereo
    vector<EdgeStereo*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float chi2Mono2 = 5.991;
    const float thHuberStereo = sqrt(7.815);
    const float chi2Stereo2 = 7.815;

    const unsigned long iniMPid = maxKFid*5;

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        if (!pMP)
            continue;

        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());

        unsigned long id = pMP->mnId+iniMPid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

        // Create visual constraints
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if (!pKFi)
                continue;

            if ((pKFi->mnBALocalForKF!=pCurrKF->mnId) && (pKFi->mnBAFixedForKF!=pCurrKF->mnId))
                continue;

            if (pKFi->mnId>maxKFid){
                continue;
            }


            if(optimizer.vertex(id)==NULL || optimizer.vertex(pKFi->mnId)==NULL)
                continue;

            if(!pKFi->isBad())
            {
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[get<0>(mit->second)];

                if(pKFi->mvuRight[get<0>(mit->second)]<0) // Monocular observation
                {
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMono* e = new EdgeMono();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);
                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else // stereo observation
                {
                    const float kp_ur = pKFi->mvuRight[get<0>(mit->second)];
                    Eigen::Matrix<double,3,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    EdgeStereo* e = new EdgeStereo();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(8);

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

    // Check inlier observations
    // Mono
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        EdgeMono* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>chi2Mono2)
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Stereo
    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        EdgeStereo* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>chi2Stereo2)
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Get Map Mutex and erase outliers
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);
    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }


    // Recover optimized data
    //Keyframes
    for(int i=0; i<N; i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];

        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        Sophus::SE3f Tcw(VP->estimate().Rcw[0].cast<float>(), VP->estimate().tcw[0].cast<float>());
        pKFi->SetPose(Tcw);

        Sophus::SE3d Tiw = pKFi->GetPose().cast<double>();
        g2o::Sim3 g2oSiw(Tiw.unit_quaternion(),Tiw.translation(),1.0);
        corrPoses[pKFi] = g2oSiw;

        if(pKFi->bImu)
        {
            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+1));
            pKFi->SetVelocity(VV->estimate().cast<float>());
            VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+2));
            VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+3));
            Vector6d b;
            b << VG->estimate(), VA->estimate();
            pKFi->SetNewBias(IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]));
        }
    }

    for(int i=0; i<Ncov; i++)
    {
        KeyFrame* pKFi = vpOptimizableCovKFs[i];

        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        Sophus::SE3f Tcw(VP->estimate().Rcw[0].cast<float>(), VP->estimate().tcw[0].cast<float>());
        pKFi->SetPose(Tcw);

        Sophus::SE3d Tiw = pKFi->GetPose().cast<double>();
        g2o::Sim3 g2oSiw(Tiw.unit_quaternion(),Tiw.translation(),1.0);
        corrPoses[pKFi] = g2oSiw;

        if(pKFi->bImu)
        {
            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+1));
            pKFi->SetVelocity(VV->estimate().cast<float>());
            VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+2));
            VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+3));
            Vector6d b;
            b << VG->estimate(), VA->estimate();
            pKFi->SetNewBias(IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]));
        }
    }

    //Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+iniMPid+1));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
        pMP->UpdateNormalAndDepth();
    }

    pMap->IncreaseChangeIndex();
}

int Optimizer::PoseInertialOptimizationLastKeyFrame(Frame *pFrame, bool bRecInit)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    optimizer.setVerbose(false);
    optimizer.setAlgorithm(solver);

    int nInitialMonoCorrespondences=0;
    int nInitialStereoCorrespondences=0;
    int nInitialCorrespondences=0;

    // Set Frame vertex
    VertexPose* VP = new VertexPose(pFrame);
    VP->setId(0);
    VP->setFixed(false);
    optimizer.addVertex(VP);
    VertexVelocity* VV = new VertexVelocity(pFrame);
    VV->setId(1);
    VV->setFixed(false);
    optimizer.addVertex(VV);
    VertexGyroBias* VG = new VertexGyroBias(pFrame);
    VG->setId(2);
    VG->setFixed(false);
    optimizer.addVertex(VG);
    VertexAccBias* VA = new VertexAccBias(pFrame);
    VA->setId(3);
    VA->setFixed(false);
    optimizer.addVertex(VA);

    // Set MapPoint vertices
    const int N = pFrame->N;
    const int Nleft = pFrame->Nleft;
    const bool bRight = (Nleft!=-1);

    vector<EdgeMonoOnlyPose*> vpEdgesMono;
    vector<EdgeStereoOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeMono;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesMono.reserve(N);
    vpEdgesStereo.reserve(N);
    vnIndexEdgeMono.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);

        for(int i=0; i<N; i++)
        {
            MapPoint* pMP = pFrame->mvpMapPoints[i];
            if(pMP)
            {
                cv::KeyPoint kpUn;

                // Left monocular observation
                if((!bRight && pFrame->mvuRight[i]<0) || i < Nleft)
                {
                    if(i < Nleft) // pair left-right
                        kpUn = pFrame->mvKeys[i];
                    else
                        kpUn = pFrame->mvKeysUn[i];

                    nInitialMonoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(),0);

                    e->setVertex(0,VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                // Stereo observation
                else if(!bRight)
                {
                    nInitialStereoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    kpUn = pFrame->mvKeysUn[i];
                    const float kp_ur = pFrame->mvuRight[i];
                    Eigen::Matrix<double,3,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    EdgeStereoOnlyPose* e = new EdgeStereoOnlyPose(pMP->GetWorldPos());

                    e->setVertex(0, VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs.head(2));

                    const float &invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }

                // Right monocular observation
                if(bRight && i >= Nleft)
                {
                    nInitialMonoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    kpUn = pFrame->mvKeysRight[i - Nleft];
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(),1);

                    e->setVertex(0,VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
            }
        }
    }
    nInitialCorrespondences = nInitialMonoCorrespondences + nInitialStereoCorrespondences;

    KeyFrame* pKF = pFrame->mpLastKeyFrame;
    VertexPose* VPk = new VertexPose(pKF);
    VPk->setId(4);
    VPk->setFixed(true);
    optimizer.addVertex(VPk);
    VertexVelocity* VVk = new VertexVelocity(pKF);
    VVk->setId(5);
    VVk->setFixed(true);
    optimizer.addVertex(VVk);
    VertexGyroBias* VGk = new VertexGyroBias(pKF);
    VGk->setId(6);
    VGk->setFixed(true);
    optimizer.addVertex(VGk);
    VertexAccBias* VAk = new VertexAccBias(pKF);
    VAk->setId(7);
    VAk->setFixed(true);
    optimizer.addVertex(VAk);

    EdgeInertial* ei = new EdgeInertial(pFrame->mpImuPreintegrated);

    ei->setVertex(0, VPk);
    ei->setVertex(1, VVk);
    ei->setVertex(2, VGk);
    ei->setVertex(3, VAk);
    ei->setVertex(4, VP);
    ei->setVertex(5, VV);
    optimizer.addEdge(ei);

    EdgeGyroRW* egr = new EdgeGyroRW();
    egr->setVertex(0,VGk);
    egr->setVertex(1,VG);
    Eigen::Matrix3d InfoG = pFrame->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse();
    egr->setInformation(InfoG);
    optimizer.addEdge(egr);

    EdgeAccRW* ear = new EdgeAccRW();
    ear->setVertex(0,VAk);
    ear->setVertex(1,VA);
    Eigen::Matrix3d InfoA = pFrame->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse();
    ear->setInformation(InfoA);
    optimizer.addEdge(ear);

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    float chi2Mono[4]={12,7.5,5.991,5.991};
    float chi2Stereo[4]={15.6,9.8,7.815,7.815};

    int its[4]={10,10,10,10};

    int nBad = 0;
    int nBadMono = 0;
    int nBadStereo = 0;
    int nInliersMono = 0;
    int nInliersStereo = 0;
    int nInliers = 0;
    for(size_t it=0; it<4; it++)
    {
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad = 0;
        nBadMono = 0;
        nBadStereo = 0;
        nInliers = 0;
        nInliersMono = 0;
        nInliersStereo = 0;
        float chi2close = 1.5*chi2Mono[it];

        // For monocular observations
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            EdgeMonoOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();
            bool bClose = pFrame->mvpMapPoints[idx]->mTrackDepth<10.f;

            if((chi2>chi2Mono[it]&&!bClose)||(bClose && chi2>chi2close)||!e->isDepthPositive())
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBadMono++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
                nInliersMono++;
            }

            if (it==2)
                e->setRobustKernel(0);
        }

        // For stereo observations
        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            EdgeStereoOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1); // not included in next optimization
                nBadStereo++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
                nInliersStereo++;
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        nInliers = nInliersMono + nInliersStereo;
        nBad = nBadMono + nBadStereo;

        if(optimizer.edges().size()<10)
        {
            break;
        }

    }

    // If not too much tracks, recover not too bad points
    if ((nInliers<30) && !bRecInit)
    {
        nBad=0;
        const float chi2MonoOut = 18.f;
        const float chi2StereoOut = 24.f;
        EdgeMonoOnlyPose* e1;
        EdgeStereoOnlyPose* e2;
        for(size_t i=0, iend=vnIndexEdgeMono.size(); i<iend; i++)
        {
            const size_t idx = vnIndexEdgeMono[i];
            e1 = vpEdgesMono[i];
            e1->computeError();
            if (e1->chi2()<chi2MonoOut)
                pFrame->mvbOutlier[idx]=false;
            else
                nBad++;
        }
        for(size_t i=0, iend=vnIndexEdgeStereo.size(); i<iend; i++)
        {
            const size_t idx = vnIndexEdgeStereo[i];
            e2 = vpEdgesStereo[i];
            e2->computeError();
            if (e2->chi2()<chi2StereoOut)
                pFrame->mvbOutlier[idx]=false;
            else
                nBad++;
        }
    }

    // Recover optimized pose, velocity and biases
    pFrame->SetImuPoseVelocity(VP->estimate().Rwb.cast<float>(), VP->estimate().twb.cast<float>(), VV->estimate().cast<float>());
    Vector6d b;
    b << VG->estimate(), VA->estimate();
    pFrame->mImuBias = IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]);

    // Recover Hessian, marginalize keyFframe states and generate new prior for frame
    Eigen::Matrix<double,15,15> H;
    H.setZero();

    H.block<9,9>(0,0)+= ei->GetHessian2();
    H.block<3,3>(9,9) += egr->GetHessian2();
    H.block<3,3>(12,12) += ear->GetHessian2();

    int tot_in = 0, tot_out = 0;
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
    {
        EdgeMonoOnlyPose* e = vpEdgesMono[i];

        const size_t idx = vnIndexEdgeMono[i];

        if(!pFrame->mvbOutlier[idx])
        {
            H.block<6,6>(0,0) += e->GetHessian();
            tot_in++;
        }
        else
            tot_out++;
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
    {
        EdgeStereoOnlyPose* e = vpEdgesStereo[i];

        const size_t idx = vnIndexEdgeStereo[i];

        if(!pFrame->mvbOutlier[idx])
        {
            H.block<6,6>(0,0) += e->GetHessian();
            tot_in++;
        }
        else
            tot_out++;
    }

    pFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb,VP->estimate().twb,VV->estimate(),VG->estimate(),VA->estimate(),H);

    return nInitialCorrespondences-nBad;
}

int Optimizer::PoseInertialOptimizationLastFrame(Frame *pFrame, bool bRecInit)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    int nInitialMonoCorrespondences=0;
    int nInitialStereoCorrespondences=0;
    int nInitialCorrespondences=0;

    // Set Current Frame vertex
    VertexPose* VP = new VertexPose(pFrame);
    VP->setId(0);
    VP->setFixed(false);
    optimizer.addVertex(VP);
    VertexVelocity* VV = new VertexVelocity(pFrame);
    VV->setId(1);
    VV->setFixed(false);
    optimizer.addVertex(VV);
    VertexGyroBias* VG = new VertexGyroBias(pFrame);
    VG->setId(2);
    VG->setFixed(false);
    optimizer.addVertex(VG);
    VertexAccBias* VA = new VertexAccBias(pFrame);
    VA->setId(3);
    VA->setFixed(false);
    optimizer.addVertex(VA);

    // Set MapPoint vertices
    const int N = pFrame->N;
    const int Nleft = pFrame->Nleft;
    const bool bRight = (Nleft!=-1);

    vector<EdgeMonoOnlyPose*> vpEdgesMono;
    vector<EdgeStereoOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeMono;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesMono.reserve(N);
    vpEdgesStereo.reserve(N);
    vnIndexEdgeMono.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);

        for(int i=0; i<N; i++)
        {
            MapPoint* pMP = pFrame->mvpMapPoints[i];
            if(pMP)
            {
                cv::KeyPoint kpUn;
                // Left monocular observation
                if((!bRight && pFrame->mvuRight[i]<0) || i < Nleft)
                {
                    if(i < Nleft) // pair left-right
                        kpUn = pFrame->mvKeys[i];
                    else
                        kpUn = pFrame->mvKeysUn[i];

                    nInitialMonoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(),0);

                    e->setVertex(0,VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                // Stereo observation
                else if(!bRight)
                {
                    nInitialStereoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    kpUn = pFrame->mvKeysUn[i];
                    const float kp_ur = pFrame->mvuRight[i];
                    Eigen::Matrix<double,3,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    EdgeStereoOnlyPose* e = new EdgeStereoOnlyPose(pMP->GetWorldPos());

                    e->setVertex(0, VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs.head(2));

                    const float &invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }

                // Right monocular observation
                if(bRight && i >= Nleft)
                {
                    nInitialMonoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    kpUn = pFrame->mvKeysRight[i - Nleft];
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(),1);

                    e->setVertex(0,VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
            }
        }
    }

    nInitialCorrespondences = nInitialMonoCorrespondences + nInitialStereoCorrespondences;

    // Set Previous Frame Vertex
    Frame* pFp = pFrame->mpPrevFrame;

    VertexPose* VPk = new VertexPose(pFp);
    VPk->setId(4);
    VPk->setFixed(false);
    optimizer.addVertex(VPk);
    VertexVelocity* VVk = new VertexVelocity(pFp);
    VVk->setId(5);
    VVk->setFixed(false);
    optimizer.addVertex(VVk);
    VertexGyroBias* VGk = new VertexGyroBias(pFp);
    VGk->setId(6);
    VGk->setFixed(false);
    optimizer.addVertex(VGk);
    VertexAccBias* VAk = new VertexAccBias(pFp);
    VAk->setId(7);
    VAk->setFixed(false);
    optimizer.addVertex(VAk);

    EdgeInertial* ei = new EdgeInertial(pFrame->mpImuPreintegratedFrame);

    ei->setVertex(0, VPk);
    ei->setVertex(1, VVk);
    ei->setVertex(2, VGk);
    ei->setVertex(3, VAk);
    ei->setVertex(4, VP);
    ei->setVertex(5, VV);
    optimizer.addEdge(ei);

    EdgeGyroRW* egr = new EdgeGyroRW();
    egr->setVertex(0,VGk);
    egr->setVertex(1,VG);
    Eigen::Matrix3d InfoG = pFrame->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse();
    egr->setInformation(InfoG);
    optimizer.addEdge(egr);

    EdgeAccRW* ear = new EdgeAccRW();
    ear->setVertex(0,VAk);
    ear->setVertex(1,VA);
    Eigen::Matrix3d InfoA = pFrame->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse();
    ear->setInformation(InfoA);
    optimizer.addEdge(ear);

    if (!pFp->mpcpi)
        Verbose::PrintMess("pFp->mpcpi does not exist!!!\nPrevious Frame " + to_string(pFp->mnId), Verbose::VERBOSITY_NORMAL);

    EdgePriorPoseImu* ep = new EdgePriorPoseImu(pFp->mpcpi);

    ep->setVertex(0,VPk);
    ep->setVertex(1,VVk);
    ep->setVertex(2,VGk);
    ep->setVertex(3,VAk);
    g2o::RobustKernelHuber* rkp = new g2o::RobustKernelHuber;
    ep->setRobustKernel(rkp);
    rkp->setDelta(5);
    optimizer.addEdge(ep);

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={15.6f,9.8f,7.815f,7.815f};
    const int its[4]={10,10,10,10};

    int nBad=0;
    int nBadMono = 0;
    int nBadStereo = 0;
    int nInliersMono = 0;
    int nInliersStereo = 0;
    int nInliers=0;
    for(size_t it=0; it<4; it++)
    {
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad=0;
        nBadMono = 0;
        nBadStereo = 0;
        nInliers=0;
        nInliersMono=0;
        nInliersStereo=0;
        float chi2close = 1.5*chi2Mono[it];

        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            EdgeMonoOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];
            bool bClose = pFrame->mvpMapPoints[idx]->mTrackDepth<10.f;

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if((chi2>chi2Mono[it]&&!bClose)||(bClose && chi2>chi2close)||!e->isDepthPositive())
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBadMono++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
                nInliersMono++;
            }

            if (it==2)
                e->setRobustKernel(0);

        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            EdgeStereoOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBadStereo++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
                nInliersStereo++;
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        nInliers = nInliersMono + nInliersStereo;
        nBad = nBadMono + nBadStereo;

        if(optimizer.edges().size()<10)
        {
            break;
        }
    }


    if ((nInliers<30) && !bRecInit)
    {
        nBad=0;
        const float chi2MonoOut = 18.f;
        const float chi2StereoOut = 24.f;
        EdgeMonoOnlyPose* e1;
        EdgeStereoOnlyPose* e2;
        for(size_t i=0, iend=vnIndexEdgeMono.size(); i<iend; i++)
        {
            const size_t idx = vnIndexEdgeMono[i];
            e1 = vpEdgesMono[i];
            e1->computeError();
            if (e1->chi2()<chi2MonoOut)
                pFrame->mvbOutlier[idx]=false;
            else
                nBad++;

        }
        for(size_t i=0, iend=vnIndexEdgeStereo.size(); i<iend; i++)
        {
            const size_t idx = vnIndexEdgeStereo[i];
            e2 = vpEdgesStereo[i];
            e2->computeError();
            if (e2->chi2()<chi2StereoOut)
                pFrame->mvbOutlier[idx]=false;
            else
                nBad++;
        }
    }

    nInliers = nInliersMono + nInliersStereo;


    // Recover optimized pose, velocity and biases
    pFrame->SetImuPoseVelocity(VP->estimate().Rwb.cast<float>(), VP->estimate().twb.cast<float>(), VV->estimate().cast<float>());
    Vector6d b;
    b << VG->estimate(), VA->estimate();
    pFrame->mImuBias = IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]);

    // Recover Hessian, marginalize previous frame states and generate new prior for frame
    Eigen::Matrix<double,30,30> H;
    H.setZero();

    H.block<24,24>(0,0)+= ei->GetHessian();

    Eigen::Matrix<double,6,6> Hgr = egr->GetHessian();
    H.block<3,3>(9,9) += Hgr.block<3,3>(0,0);
    H.block<3,3>(9,24) += Hgr.block<3,3>(0,3);
    H.block<3,3>(24,9) += Hgr.block<3,3>(3,0);
    H.block<3,3>(24,24) += Hgr.block<3,3>(3,3);

    Eigen::Matrix<double,6,6> Har = ear->GetHessian();
    H.block<3,3>(12,12) += Har.block<3,3>(0,0);
    H.block<3,3>(12,27) += Har.block<3,3>(0,3);
    H.block<3,3>(27,12) += Har.block<3,3>(3,0);
    H.block<3,3>(27,27) += Har.block<3,3>(3,3);

    H.block<15,15>(0,0) += ep->GetHessian();

    int tot_in = 0, tot_out = 0;
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
    {
        EdgeMonoOnlyPose* e = vpEdgesMono[i];

        const size_t idx = vnIndexEdgeMono[i];

        if(!pFrame->mvbOutlier[idx])
        {
            H.block<6,6>(15,15) += e->GetHessian();
            tot_in++;
        }
        else
            tot_out++;
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
    {
        EdgeStereoOnlyPose* e = vpEdgesStereo[i];

        const size_t idx = vnIndexEdgeStereo[i];

        if(!pFrame->mvbOutlier[idx])
        {
            H.block<6,6>(15,15) += e->GetHessian();
            tot_in++;
        }
        else
            tot_out++;
    }

    H = Marginalize(H,0,14);

    pFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb,VP->estimate().twb,VV->estimate(),VG->estimate(),VA->estimate(),H.block<15,15>(15,15));
    delete pFp->mpcpi;
    pFp->mpcpi = NULL;

    return nInitialCorrespondences-nBad;
}

void Optimizer::OptimizeEssentialGraph4DoF(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections)
{
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<4, 4> > BlockSolver_4_4;

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolverX::LinearSolverType * linearSolver =
            new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    optimizer.setAlgorithm(solver);

    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);

    vector<VertexPose4DoF*> vpVertices(nMaxKFid+1);

    const int minFeat = 100;
    // Set KeyFrame vertices
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;

        VertexPose4DoF* V4DoF;

        const int nIDi = pKF->mnId;

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

        if(it!=CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            const g2o::Sim3 Swc = it->second.inverse();
            Eigen::Matrix3d Rwc = Swc.rotation().toRotationMatrix();
            Eigen::Vector3d twc = Swc.translation();
            V4DoF = new VertexPose4DoF(Rwc, twc, pKF);
        }
        else
        {
            Sophus::SE3d Tcw = pKF->GetPose().cast<double>();
            g2o::Sim3 Siw(Tcw.unit_quaternion(),Tcw.translation(),1.0);

            vScw[nIDi] = Siw;
            V4DoF = new VertexPose4DoF(pKF);
        }

        if(pKF==pLoopKF)
            V4DoF->setFixed(true);

        V4DoF->setId(nIDi);
        V4DoF->setMarginalized(false);

        optimizer.addVertex(V4DoF);
        vpVertices[nIDi]=V4DoF;
    }
    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;

    // Edge used in posegraph has still 6Dof, even if updates of camera poses are just in 4DoF
    Eigen::Matrix<double,6,6> matLambda = Eigen::Matrix<double,6,6>::Identity();
    matLambda(0,0) = 1e3;
    matLambda(1,1) = 1e3;
    matLambda(0,0) = 1e3;

    // Set Loop edges
    Edge4DoF* e_loop;
    for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame*> &spConnections = mit->second;
        const g2o::Sim3 Siw = vScw[nIDi];

        for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sij = Siw * Sjw.inverse();
            Eigen::Matrix4d Tij;
            Tij.block<3,3>(0,0) = Sij.rotation().toRotationMatrix();
            Tij.block<3,1>(0,3) = Sij.translation();
            Tij(3,3) = 1.;

            Edge4DoF* e = new Edge4DoF(Tij);
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));

            e->information() = matLambda;
            e_loop = e;
            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
        }
    }

    // 1. Set normal edges
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Siw;

        // Use noncorrected poses for posegraph edges
        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if(iti!=NonCorrectedSim3.end())
            Siw = iti->second;
        else
            Siw = vScw[nIDi];

        // 1.1.0 Spanning tree edge
        KeyFrame* pParentKF = static_cast<KeyFrame*>(NULL);
        if(pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Swj;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if(itj!=NonCorrectedSim3.end())
                Swj = (itj->second).inverse();
            else
                Swj =  vScw[nIDj].inverse();

            g2o::Sim3 Sij = Siw * Swj;
            Eigen::Matrix4d Tij;
            Tij.block<3,3>(0,0) = Sij.rotation().toRotationMatrix();
            Tij.block<3,1>(0,3) = Sij.translation();
            Tij(3,3)=1.;

            Edge4DoF* e = new Edge4DoF(Tij);
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // 1.1.1 Inertial edges
        KeyFrame* prevKF = pKF->mPrevKF;
        if(prevKF)
        {
            int nIDj = prevKF->mnId;

            g2o::Sim3 Swj;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(prevKF);

            if(itj!=NonCorrectedSim3.end())
                Swj = (itj->second).inverse();
            else
                Swj =  vScw[nIDj].inverse();

            g2o::Sim3 Sij = Siw * Swj;
            Eigen::Matrix4d Tij;
            Tij.block<3,3>(0,0) = Sij.rotation().toRotationMatrix();
            Tij.block<3,1>(0,3) = Sij.translation();
            Tij(3,3)=1.;

            Edge4DoF* e = new Edge4DoF(Tij);
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // 1.2 Loop edges
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if(pLKF->mnId<pKF->mnId)
            {
                g2o::Sim3 Swl;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if(itl!=NonCorrectedSim3.end())
                    Swl = itl->second.inverse();
                else
                    Swl = vScw[pLKF->mnId].inverse();

                g2o::Sim3 Sil = Siw * Swl;
                Eigen::Matrix4d Til;
                Til.block<3,3>(0,0) = Sil.rotation().toRotationMatrix();
                Til.block<3,1>(0,3) = Sil.translation();
                Til(3,3) = 1.;

                Edge4DoF* e = new Edge4DoF(Til);
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                e->information() = matLambda;
                optimizer.addEdge(e);
            }
        }

        // 1.3 Covisibility graph edges
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKF && pKFn!=prevKF && pKFn!=pKF->mNextKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                {
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                        continue;

                    g2o::Sim3 Swn;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if(itn!=NonCorrectedSim3.end())
                        Swn = itn->second.inverse();
                    else
                        Swn = vScw[pKFn->mnId].inverse();

                    g2o::Sim3 Sin = Siw * Swn;
                    Eigen::Matrix4d Tin;
                    Tin.block<3,3>(0,0) = Sin.rotation().toRotationMatrix();
                    Tin.block<3,1>(0,3) = Sin.translation();
                    Tin(3,3) = 1.;
                    Edge4DoF* e = new Edge4DoF(Tin);
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    e->information() = matLambda;
                    optimizer.addEdge(e);
                }
            }
        }
    }

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        VertexPose4DoF* Vi = static_cast<VertexPose4DoF*>(optimizer.vertex(nIDi));
        Eigen::Matrix3d Ri = Vi->estimate().Rcw[0];
        Eigen::Vector3d ti = Vi->estimate().tcw[0];

        g2o::Sim3 CorrectedSiw = g2o::Sim3(Ri,ti,1.);
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();

        Sophus::SE3d Tiw(CorrectedSiw.rotation(),CorrectedSiw.translation());
        pKFi->SetPose(Tiw.cast<float>());
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        int nIDr;

        KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
        nIDr = pRefKF->mnId;

        g2o::Sim3 Srw = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        Eigen::Matrix<double,3,1> eigP3Dw = pMP->GetWorldPos().cast<double>();
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));
        pMP->SetWorldPos(eigCorrectedP3Dw.cast<float>());

        pMP->UpdateNormalAndDepth();
    }
    pMap->IncreaseChangeIndex();
}

} //namespace ORB_SLAM
