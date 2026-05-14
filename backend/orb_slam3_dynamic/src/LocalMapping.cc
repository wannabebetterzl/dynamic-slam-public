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


#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "Converter.h"
#include "GeometricTools.h"
#include "Instance.h"
#include "Map.h"

#include<cstdlib>
#include<cmath>
#include<mutex>
#include<chrono>
#include<sstream>
#include<map>
#include<set>

namespace ORB_SLAM3
{

namespace
{

bool GetEnvFlagOrDefault(const char* name, const bool defaultValue)
{
    const char* envValue = std::getenv(name);
    if(!envValue)
        return defaultValue;
    return std::string(envValue) != "0";
}

double GetEnvDoubleOrDefault(const char* name,
                             const double defaultValue,
                             const double minValue)
{
    const char* envValue = std::getenv(name);
    if(!envValue)
        return defaultValue;
    const double value = std::atof(envValue);
    if(!std::isfinite(value))
        return defaultValue;
    return std::max(minValue, value);
}

bool ForceFilterDetectedDynamicFeatures()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_FORCE_FILTER_DETECTED_DYNAMIC_FEATURES", false);
    return value;
}

bool SplitDetectedDynamicFeaturesFromStaticMapping()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_RGBD_DYNAMIC_FRONTEND_SPLIT", false);
    return value;
}

bool DynamicMapAdmissionVeto()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_VETO", false);
    return value;
}

bool DynamicMapAdmissionVetoCreateNewMapPoints()
{
    static const bool value =
        GetEnvFlagOrDefault(
            "STSLAM_DYNAMIC_MAP_ADMISSION_VETO_LOCAL_MAPPING_CREATE_NEW_MAPPOINTS",
            DynamicMapAdmissionVeto());
    return value;
}

bool DynamicMapAdmissionBoundaryVeto()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_VETO", false);
    return value;
}

bool DynamicMapAdmissionBoundaryVetoCreateNewMapPoints()
{
    static const bool value =
        GetEnvFlagOrDefault(
            "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_VETO_LOCAL_MAPPING_CREATE_NEW_MAPPOINTS",
            DynamicMapAdmissionBoundaryVeto());
    return value;
}

bool DynamicMapAdmissionBoundarySameCountControl()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_SAME_COUNT_CONTROL", false);
    return value;
}

bool DynamicMapAdmissionBoundarySameCountControlCreateNewMapPoints()
{
    static const bool value =
        GetEnvFlagOrDefault(
            "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_SAME_COUNT_CONTROL_LOCAL_MAPPING_CREATE_NEW_MAPPOINTS",
            DynamicMapAdmissionBoundarySameCountControl());
    return value;
}

bool DynamicMapAdmissionBoundaryMatchedControl()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_MATCHED_CONTROL", false);
    return value;
}

bool DynamicMapAdmissionBoundaryMatchedControlCreateNewMapPoints()
{
    static const bool value =
        GetEnvFlagOrDefault(
            "STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_MATCHED_CONTROL_LOCAL_MAPPING_CREATE_NEW_MAPPOINTS",
            DynamicMapAdmissionBoundaryMatchedControl());
    return value;
}

bool EnableNearBoundaryDiagnostics()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_NEAR_BOUNDARY_DIAGNOSTICS", false);
    return value;
}

bool DynamicMapAdmissionDelayedBoundary()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_DELAYED_BOUNDARY", false);
    return value;
}

bool DynamicMapAdmissionDelayedBoundaryCreateNewMapPoints()
{
    static const bool value =
        GetEnvFlagOrDefault(
            "STSLAM_DYNAMIC_MAP_ADMISSION_DELAYED_BOUNDARY_LOCAL_MAPPING_CREATE_NEW_MAPPOINTS",
            DynamicMapAdmissionDelayedBoundary());
    return value;
}

int GetEnvIntOrDefault(const char* name, const int defaultValue, const int minValue);

int DynamicMapAdmissionBoundaryRadiusPx()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_RADIUS_PX", 5, 0);
    return value;
}

int DynamicMapAdmissionMatchedControlGridCols()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_MATCHED_CONTROL_GRID_COLS", 4, 1);
    return value;
}

int DynamicMapAdmissionMatchedControlGridRows()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_MATCHED_CONTROL_GRID_ROWS", 3, 1);
    return value;
}

struct AdmissionMatchedControlBin
{
    int gridX;
    int gridY;
    int depthBin;
    int octave;

    bool operator<(const AdmissionMatchedControlBin& other) const
    {
        if(gridX != other.gridX)
            return gridX < other.gridX;
        if(gridY != other.gridY)
            return gridY < other.gridY;
        if(depthBin != other.depthBin)
            return depthBin < other.depthBin;
        return octave < other.octave;
    }
};

int QuantizeAdmissionMatchedControlDepth(const float depth)
{
    if(depth <= 0.0f || !std::isfinite(depth))
        return -1;
    if(depth < 1.0f)
        return 0;
    if(depth < 2.0f)
        return 1;
    if(depth < 3.0f)
        return 2;
    if(depth < 4.0f)
        return 3;
    if(depth < 6.0f)
        return 4;
    return 5;
}

AdmissionMatchedControlBin MakeAdmissionMatchedControlBin(const KeyFrame* pKF,
                                                          const int idx,
                                                          const float depth,
                                                          const bool includeGrid)
{
    AdmissionMatchedControlBin bin;
    bin.gridX = includeGrid ? 0 : -1;
    bin.gridY = includeGrid ? 0 : -1;
    bin.depthBin = QuantizeAdmissionMatchedControlDepth(depth);
    bin.octave = 0;

    if(!pKF || idx < 0 || idx >= static_cast<int>(pKF->mvKeysUn.size()))
        return bin;

    const cv::KeyPoint& kp = pKF->mvKeysUn[idx];
    bin.octave = kp.octave;
    if(includeGrid)
    {
        const int gridCols = DynamicMapAdmissionMatchedControlGridCols();
        const int gridRows = DynamicMapAdmissionMatchedControlGridRows();
        const float width = std::max(1, pKF->mnMaxX - pKF->mnMinX);
        const float height = std::max(1, pKF->mnMaxY - pKF->mnMinY);
        const float normX = (kp.pt.x - pKF->mnMinX) / width;
        const float normY = (kp.pt.y - pKF->mnMinY) / height;
        bin.gridX =
            std::min(gridCols - 1, std::max(0, static_cast<int>(std::floor(normX * gridCols))));
        bin.gridY =
            std::min(gridRows - 1, std::max(0, static_cast<int>(std::floor(normY * gridRows))));
    }

    return bin;
}

float PairMatchedControlDepth(const KeyFrame* pKF1,
                              const int idx1,
                              const KeyFrame* pKF2,
                              const int idx2)
{
    float sum = 0.0f;
    int count = 0;
    if(pKF1 && idx1 >= 0 && idx1 < static_cast<int>(pKF1->mvDepth.size()) &&
       pKF1->mvDepth[idx1] > 0.0f)
    {
        sum += pKF1->mvDepth[idx1];
        ++count;
    }
    if(pKF2 && idx2 >= 0 && idx2 < static_cast<int>(pKF2->mvDepth.size()) &&
       pKF2->mvDepth[idx2] > 0.0f)
    {
        sum += pKF2->mvDepth[idx2];
        ++count;
    }
    return count > 0 ? sum / static_cast<float>(count) : -1.0f;
}

void AddAdmissionMatchedControlBudget(
    std::map<AdmissionMatchedControlBin, int>& exactBudget,
    std::map<AdmissionMatchedControlBin, int>& fallbackBudget,
    const AdmissionMatchedControlBin& exactBin,
    const AdmissionMatchedControlBin& fallbackBin)
{
    ++exactBudget[exactBin];
    ++fallbackBudget[fallbackBin];
}

bool ConsumeAdmissionMatchedControlBudget(
    std::map<AdmissionMatchedControlBin, int>& exactBudget,
    std::map<AdmissionMatchedControlBin, int>& fallbackBudget,
    const AdmissionMatchedControlBin& exactBin,
    const AdmissionMatchedControlBin& fallbackBin,
    bool& usedExact)
{
    usedExact = false;
    std::map<AdmissionMatchedControlBin, int>::iterator exactIt = exactBudget.find(exactBin);
    if(exactIt != exactBudget.end() && exactIt->second > 0)
    {
        --exactIt->second;
        std::map<AdmissionMatchedControlBin, int>::iterator fallbackIt =
            fallbackBudget.find(fallbackBin);
        if(fallbackIt != fallbackBudget.end() && fallbackIt->second > 0)
            --fallbackIt->second;
        usedExact = true;
        return true;
    }

    std::map<AdmissionMatchedControlBin, int>::iterator fallbackIt =
        fallbackBudget.find(fallbackBin);
    if(fallbackIt != fallbackBudget.end() && fallbackIt->second > 0)
    {
        --fallbackIt->second;
        return true;
    }

    return false;
}

int DynamicMapAdmissionDelayedBoundarySupportRadiusPx()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_DELAYED_BOUNDARY_SUPPORT_RADIUS_PX",
                           18,
                           0);
    return value;
}

int DynamicMapAdmissionDelayedBoundaryMinSupport()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_DELAYED_BOUNDARY_MIN_SUPPORT",
                           2,
                           0);
    return value;
}

int DynamicMapAdmissionDelayedBoundaryMinObservations()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_DELAYED_BOUNDARY_MIN_OBS",
                           2,
                           0);
    return value;
}

bool DynamicMapAdmissionSupportQuality()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_SUPPORT_QUALITY", false);
    return value;
}

int DynamicMapAdmissionSupportQualityMinReliableSupport()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_SUPPORT_QUALITY_MIN_RELIABLE_SUPPORT",
                           2,
                           0);
    return value;
}

int DynamicMapAdmissionSupportQualityMinDepthSupport()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_SUPPORT_QUALITY_MIN_DEPTH_SUPPORT",
                           2,
                           0);
    return value;
}

int DynamicMapAdmissionSupportQualityMinResidualObs()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_SUPPORT_QUALITY_MIN_RESIDUAL_OBS",
                           1,
                           0);
    return value;
}

int DynamicMapAdmissionSupportQualityMinFrameSpan()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_SUPPORT_QUALITY_MIN_FRAME_SPAN",
                           2,
                           0);
    return value;
}

double DynamicMapAdmissionSupportQualityMinInlierRate()
{
    static const double value =
        std::min(1.0,
                 GetEnvDoubleOrDefault(
                     "STSLAM_DYNAMIC_MAP_ADMISSION_SUPPORT_QUALITY_MIN_INLIER_RATE",
                     0.70,
                     0.0));
    return value;
}

double DynamicMapAdmissionSupportQualityMaxMeanChi2()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_SUPPORT_QUALITY_MAX_MEAN_CHI2",
                              3.0,
                              0.0);
    return value;
}

double DynamicMapAdmissionSupportQualityMinFoundRatio()
{
    static const double value =
        std::min(1.0,
                 GetEnvDoubleOrDefault(
                     "STSLAM_DYNAMIC_MAP_ADMISSION_SUPPORT_QUALITY_MIN_FOUND_RATIO",
                     0.35,
                     0.0));
    return value;
}

double DynamicMapAdmissionSupportQualityMaxDepthRelDiff()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_SUPPORT_QUALITY_MAX_DEPTH_REL_DIFF",
                              0.25,
                              0.0);
    return value;
}

bool DynamicMapAdmissionScoreBased()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_SCORE_BASED", false);
    return value;
}

int DynamicMapAdmissionScoreMinRawSupport()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_SCORE_MIN_RAW_SUPPORT",
                           1,
                           0);
    return value;
}

double DynamicMapAdmissionScoreMinSupportScore()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_SCORE_MIN_SUPPORT_SCORE",
                              0.35,
                              0.0);
    return value;
}

double DynamicMapAdmissionScoreMinCandidateScore()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_SCORE_MIN_CANDIDATE_SCORE",
                              0.45,
                              0.0);
    return value;
}

double DynamicMapAdmissionScoreMinTotalScore()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_SCORE_MIN_TOTAL_SCORE",
                              0.95,
                              0.0);
    return value;
}

double DynamicMapAdmissionScoreSupportWeight()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_SCORE_SUPPORT_WEIGHT",
                              1.0,
                              0.0);
    return value;
}

double DynamicMapAdmissionScoreCandidateWeight()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_SCORE_CANDIDATE_WEIGHT",
                              1.0,
                              0.0);
    return value;
}

bool DynamicMapAdmissionV5UsefulnessLog()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_V5_USEFULNESS_LOG",
                            false);
    return value;
}

bool DynamicMapAdmissionV5TraceLog()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_V5_TRACE_LOG",
                            false);
    return value;
}

bool DynamicMapAdmissionStateAware()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_STATE_AWARE",
                            false);
    return value;
}

int DynamicMapAdmissionStateMinTrackingInliers()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_STATE_MIN_TRACKING_INLIERS",
                           160,
                           0);
    return value;
}

int DynamicMapAdmissionStateMaxKeyFrameGap()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_STATE_MAX_KEYFRAME_GAP",
                           2,
                           0);
    return value;
}

double DynamicMapAdmissionStateStepRatio()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_STATE_STEP_RATIO",
                              1.60,
                              1.0);
    return value;
}

double DynamicMapAdmissionStateMinKeyFrameStep()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_STATE_MIN_KEYFRAME_STEP",
                              0.030,
                              0.0);
    return value;
}

double DynamicMapAdmissionStateMinLBAEdgesPerMP()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_STATE_MIN_LBA_EDGES_PER_MP",
                              1.60,
                              0.0);
    return value;
}

double DynamicMapAdmissionStateMinNeedScore()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_STATE_MIN_NEED_SCORE",
                              1.0,
                              0.0);
    return value;
}

double DynamicMapAdmissionStateKfStepEwmaAlpha()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_STATE_KF_STEP_EWMA_ALPHA",
                              0.20,
                              0.0);
    return std::min(1.0, value);
}

bool DynamicMapAdmissionCoverageAwareV7()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_COVERAGE_AWARE_V7",
                            false);
    return value;
}

int DynamicMapAdmissionV7MinUnmatchedBoundaryFeatures()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_V7_MIN_UNMATCHED_BOUNDARY_FEATURES",
                           40,
                           0);
    return value;
}

int DynamicMapAdmissionV7MinUnmatchedBoundaryCells()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_V7_MIN_UNMATCHED_BOUNDARY_CELLS",
                           4,
                           0);
    return value;
}

int DynamicMapAdmissionV7GridCols()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_V7_GRID_COLS",
                           4,
                           1);
    return value;
}

int DynamicMapAdmissionV7GridRows()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_V7_GRID_ROWS",
                           3,
                           1);
    return value;
}

double DynamicMapAdmissionV7MinNeedScore()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_V7_MIN_NEED_SCORE",
                              1.0,
                              0.0);
    return value;
}

int DynamicMapAdmissionV7MaxPromotionsPerKeyFrame()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_V7_MAX_PROMOTIONS_PER_KEYFRAME",
                           24,
                           0);
    return value;
}

int DynamicMapAdmissionV7MaxPromotionsPerNeighbor()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_V7_MAX_PROMOTIONS_PER_NEIGHBOR",
                           4,
                           0);
    return value;
}

bool DynamicMapAdmissionV7Probation()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_V7_PROBATION",
                            DynamicMapAdmissionCoverageAwareV7());
    return value;
}

int DynamicMapAdmissionV7ProbationMinAgeKFs()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_V7_PROBATION_MIN_AGE_KFS",
                           1,
                           0);
    return value;
}

int DynamicMapAdmissionV7ProbationMinPoseUse()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_V7_PROBATION_MIN_POSE_USE",
                           1,
                           0);
    return value;
}

double DynamicMapAdmissionV7ProbationMinInlierRate()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_V7_PROBATION_MIN_INLIER_RATE",
                              0.55,
                              0.0);
    return std::min(1.0, value);
}

double DynamicMapAdmissionV7ProbationMaxMeanChi2()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_V7_PROBATION_MAX_MEAN_CHI2",
                              25.0,
                              0.0);
    return value;
}

int DynamicMapAdmissionV7ProbationLowUseAgeKFs()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_V7_PROBATION_LOW_USE_AGE_KFS",
                           3,
                           0);
    return value;
}

bool DynamicMapAdmissionConstraintRoleLog()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_DYNAMIC_MAP_ADMISSION_CONSTRAINT_ROLE_LOG",
                            false);
    return value;
}

struct AdmissionStateAwarenessContext
{
    bool enabled = false;
    bool allowAdmission = true;
    bool trackingPressure = false;
    bool keyframePressure = false;
    bool scalePressure = false;
    bool localBAPressure = false;
    double needScore = 0.0;
    int trackingInliers = -1;
    int keyframeFrameGap = -1;
    double keyframeStep = -1.0;
    double keyframeStepEwma = -1.0;
    double keyframeStepRatio = 0.0;
    int lastLBAEdges = 0;
    int lastLBAMapPoints = 0;
    double lastLBAEdgesPerMP = 0.0;
};

AdmissionStateAwarenessContext MakeAdmissionStateAwarenessContext(
    Tracking* pTracker,
    KeyFrame* pCurrentKF,
    const bool hasKfStepEwma,
    const double kfStepEwma,
    const int lastLBAEdges,
    const int lastLBAMapPoints)
{
    AdmissionStateAwarenessContext context;
    context.enabled = DynamicMapAdmissionStateAware();
    if(!context.enabled)
        return context;

    context.trackingInliers =
        pTracker ? pTracker->GetMatchesInliers() : -1;
    context.trackingPressure =
        context.trackingInliers >= 0 &&
        context.trackingInliers <= DynamicMapAdmissionStateMinTrackingInliers();

    if(pCurrentKF && pCurrentKF->mPrevKF)
    {
        context.keyframeFrameGap =
            static_cast<int>(pCurrentKF->mnFrameId) -
            static_cast<int>(pCurrentKF->mPrevKF->mnFrameId);
        context.keyframePressure =
            context.keyframeFrameGap >= 0 &&
            context.keyframeFrameGap <= DynamicMapAdmissionStateMaxKeyFrameGap();
        context.keyframeStep =
            static_cast<double>(
                (pCurrentKF->GetCameraCenter() -
                 pCurrentKF->mPrevKF->GetCameraCenter()).norm());
    }

    context.keyframeStepEwma = hasKfStepEwma ? kfStepEwma : -1.0;
    if(hasKfStepEwma && kfStepEwma > 1e-9 && context.keyframeStep >= 0.0)
    {
        context.keyframeStepRatio = context.keyframeStep / kfStepEwma;
        context.scalePressure =
            context.keyframeStep >= DynamicMapAdmissionStateMinKeyFrameStep() &&
            context.keyframeStepRatio >= DynamicMapAdmissionStateStepRatio();
    }

    context.lastLBAEdges = lastLBAEdges;
    context.lastLBAMapPoints = lastLBAMapPoints;
    if(lastLBAEdges > 0 && lastLBAMapPoints > 0)
    {
        context.lastLBAEdgesPerMP =
            static_cast<double>(lastLBAEdges) /
            static_cast<double>(lastLBAMapPoints);
        context.localBAPressure =
            context.lastLBAEdgesPerMP <
            DynamicMapAdmissionStateMinLBAEdgesPerMP();
    }

    context.needScore =
        (context.trackingPressure ? 1.0 : 0.0) +
        (context.keyframePressure ? 1.0 : 0.0) +
        (context.scalePressure ? 1.0 : 0.0) +
        (context.localBAPressure ? 1.0 : 0.0);
    context.allowAdmission =
        context.needScore >= DynamicMapAdmissionStateMinNeedScore();
    return context;
}

struct AdmissionCoverageContext
{
    bool enabled = false;
    bool coveragePressure = false;
    int unmatchedBoundaryFeatures = 0;
    int unmatchedBoundaryCells = 0;
    int totalBoundaryFeatures = 0;
};

AdmissionCoverageContext MakeAdmissionCoverageContext(KeyFrame* pKF)
{
    AdmissionCoverageContext context;
    context.enabled = DynamicMapAdmissionCoverageAwareV7();
    if(!context.enabled || !pKF)
        return context;

    const int gridCols = DynamicMapAdmissionV7GridCols();
    const int gridRows = DynamicMapAdmissionV7GridRows();
    const double width = std::max(1.0, static_cast<double>(pKF->mnMaxX - pKF->mnMinX));
    const double height = std::max(1.0, static_cast<double>(pKF->mnMaxY - pKF->mnMinY));
    std::set<std::pair<int, int> > cells;

    for(int idx = 0; idx < pKF->N; ++idx)
    {
        if(!pKF->IsFeatureStaticNearDynamicMask(idx))
            continue;

        ++context.totalBoundaryFeatures;
        MapPoint* pMP = pKF->GetMapPoint(idx);
        if(pMP && !pMP->isBad())
            continue;

        ++context.unmatchedBoundaryFeatures;
        if(idx >= static_cast<int>(pKF->mvKeysUn.size()))
            continue;

        const cv::Point2f pt = pKF->mvKeysUn[idx].pt;
        int cellX = static_cast<int>(
            std::floor((static_cast<double>(pt.x) - pKF->mnMinX) *
                       static_cast<double>(gridCols) / width));
        int cellY = static_cast<int>(
            std::floor((static_cast<double>(pt.y) - pKF->mnMinY) *
                       static_cast<double>(gridRows) / height));
        cellX = std::max(0, std::min(gridCols - 1, cellX));
        cellY = std::max(0, std::min(gridRows - 1, cellY));
        cells.insert(std::make_pair(cellX, cellY));
    }

    context.unmatchedBoundaryCells = static_cast<int>(cells.size());
    context.coveragePressure =
        context.unmatchedBoundaryFeatures >=
            DynamicMapAdmissionV7MinUnmatchedBoundaryFeatures() ||
        context.unmatchedBoundaryCells >=
            DynamicMapAdmissionV7MinUnmatchedBoundaryCells();
    return context;
}

struct AdmissionV5AggregateStats
{
    long long candidateEvents = 0;
    long long supportCandidates = 0;
    long long supportAccepted = 0;
    long long rejectSupport = 0;
    long long geomEvents = 0;
    long long rejectParallax = 0;
    long long rejectTriangulate = 0;
    long long rejectDepth = 0;
    long long rejectReproj1 = 0;
    long long rejectReproj2 = 0;
    long long rejectScale = 0;
    long long rejectScore = 0;
    long long created = 0;
    double supportScoreSum = 0.0;
    double rawSupportSum = 0.0;
    double reliableSupportSum = 0.0;
    double residualSupportSum = 0.0;
    double depthSupportSum = 0.0;
    double createdSupportScoreSum = 0.0;
    double createdCandidateScoreSum = 0.0;
    double createdTotalScoreSum = 0.0;
    double createdReprojRatio1Sum = 0.0;
    double createdReprojRatio2Sum = 0.0;
    double createdParallaxScoreSum = 0.0;
    double createdScaleScoreSum = 0.0;
    long long lifecycleRows = 0;
    long long lifecycleScoreRecent = 0;
    long long lifecycleScorePreBad = 0;
    long long lifecycleScoreCulledFoundRatio = 0;
    long long lifecycleScoreCulledLowObs = 0;
    long long lifecycleScoreSurvived = 0;
    long long lifecycleScoreMatured = 0;
    long long lifecyclePoseUseEdges = 0;
    long long lifecyclePoseUseInliers = 0;
    double lifecyclePoseUseChi2WeightedSum = 0.0;
};

AdmissionV5AggregateStats& MutableAdmissionV5AggregateStats()
{
    static AdmissionV5AggregateStats stats;
    return stats;
}

double SafeMean(const double sum, const long long count)
{
    return count > 0 ? sum / static_cast<double>(count) : 0.0;
}

void PrintAdmissionV5AggregateSummary()
{
    if(!DynamicMapAdmissionV5UsefulnessLog())
        return;

    static bool printed = false;
    if(printed)
        return;
    printed = true;

    const AdmissionV5AggregateStats& stats = MutableAdmissionV5AggregateStats();
    std::cout << "[STSLAM_DYNAMIC_MAP_ADMISSION_V5_SUMMARY]"
              << " candidate_events=" << stats.candidateEvents
              << " support_candidates=" << stats.supportCandidates
              << " support_accepted=" << stats.supportAccepted
              << " reject_support=" << stats.rejectSupport
              << " geom_events=" << stats.geomEvents
              << " reject_parallax=" << stats.rejectParallax
              << " reject_triangulate=" << stats.rejectTriangulate
              << " reject_depth=" << stats.rejectDepth
              << " reject_reproj1=" << stats.rejectReproj1
              << " reject_reproj2=" << stats.rejectReproj2
              << " reject_scale=" << stats.rejectScale
              << " reject_score=" << stats.rejectScore
              << " created=" << stats.created
              << " support_score_mean="
              << SafeMean(stats.supportScoreSum, stats.supportCandidates)
              << " raw_support_mean="
              << SafeMean(stats.rawSupportSum, stats.supportCandidates)
              << " reliable_support_mean="
              << SafeMean(stats.reliableSupportSum, stats.supportCandidates)
              << " residual_support_mean="
              << SafeMean(stats.residualSupportSum, stats.supportCandidates)
              << " depth_support_mean="
              << SafeMean(stats.depthSupportSum, stats.supportCandidates)
              << " created_support_score_mean="
              << SafeMean(stats.createdSupportScoreSum, stats.created)
              << " created_candidate_score_mean="
              << SafeMean(stats.createdCandidateScoreSum, stats.created)
              << " created_total_score_mean="
              << SafeMean(stats.createdTotalScoreSum, stats.created)
              << " created_reproj_ratio1_mean="
              << SafeMean(stats.createdReprojRatio1Sum, stats.created)
              << " created_reproj_ratio2_mean="
              << SafeMean(stats.createdReprojRatio2Sum, stats.created)
              << " created_parallax_score_mean="
              << SafeMean(stats.createdParallaxScoreSum, stats.created)
              << " created_scale_score_mean="
              << SafeMean(stats.createdScaleScoreSum, stats.created)
              << " lifecycle_rows=" << stats.lifecycleRows
              << " lifecycle_score_recent_sum=" << stats.lifecycleScoreRecent
              << " lifecycle_score_prebad_sum=" << stats.lifecycleScorePreBad
              << " lifecycle_score_culled_found_ratio_sum="
              << stats.lifecycleScoreCulledFoundRatio
              << " lifecycle_score_culled_low_obs_sum="
              << stats.lifecycleScoreCulledLowObs
              << " lifecycle_score_survived_sum="
              << stats.lifecycleScoreSurvived
              << " lifecycle_score_matured_sum=" << stats.lifecycleScoreMatured
              << " lifecycle_pose_use_edges_sum="
              << stats.lifecyclePoseUseEdges
              << " lifecycle_pose_use_inliers_sum="
              << stats.lifecyclePoseUseInliers
              << " lifecycle_pose_use_inlier_rate="
              << (stats.lifecyclePoseUseEdges > 0 ?
                  static_cast<double>(stats.lifecyclePoseUseInliers) /
                      static_cast<double>(stats.lifecyclePoseUseEdges) :
                  0.0)
              << " lifecycle_pose_use_chi2_mean="
              << SafeMean(stats.lifecyclePoseUseChi2WeightedSum,
                          stats.lifecyclePoseUseEdges)
              << std::endl;
}

double Clamp01(const double value)
{
    return std::max(0.0, std::min(1.0, value));
}

double SaturatingCountScore(const int value, const int target)
{
    if(target <= 0)
        return 1.0;
    return Clamp01(static_cast<double>(value) / static_cast<double>(target));
}

int CountCleanStaticMapSupportNearFeature(KeyFrame* pKF,
                                          const int idx,
                                          const int radiusPx,
                                          const int minObservations)
{
    if(!pKF || idx < 0 || idx >= static_cast<int>(pKF->mvKeysUn.size()))
        return 0;

    const cv::Point2f center = pKF->mvKeysUn[idx].pt;
    const std::vector<size_t> vNeighborIndices =
        pKF->GetFeaturesInArea(center.x,
                               center.y,
                               static_cast<float>(std::max(0, radiusPx)));
    int support = 0;
    for(size_t neighbor = 0; neighbor < vNeighborIndices.size(); ++neighbor)
    {
        const size_t candidateIdx = vNeighborIndices[neighbor];
        if(static_cast<int>(candidateIdx) == idx)
            continue;

        MapPoint* pMP = pKF->GetMapPoint(candidateIdx);
        if(!pMP || pMP->isBad() || pMP->IsDynamicInstanceObservationPoint() ||
           pMP->IsInstanceStructurePoint())
            continue;
        if(pMP->Observations() < minObservations)
            continue;
        if(pKF->GetFeatureInstanceId(candidateIdx) > 0 ||
           pKF->IsFeatureStaticNearDynamicMask(candidateIdx))
            continue;

        ++support;
    }

    return support;
}

struct AdmissionSupportQualityResult
{
    int rawSupport = 0;
    int foundStableSupport = 0;
    int frameStableSupport = 0;
    int rawDepthConsistentSupport = 0;
    int reliableSupport = 0;
    int residualReliableSupport = 0;
    int depthConsistentSupport = 0;
    bool pass = false;
};

struct AdmissionScoreCandidateInfo
{
    int riskSides = 0;
    int rawSupport = 0;
    int foundStableSupport = 0;
    int frameStableSupport = 0;
    int rawDepthConsistentSupport = 0;
    int reliableSupport = 0;
    int residualReliableSupport = 0;
    int depthConsistentSupport = 0;
    bool binarySupportPass = false;
    double supportScore = 0.0;
};

void AccumulateAdmissionScoreSupport(AdmissionScoreCandidateInfo& info,
                                     const AdmissionSupportQualityResult& quality)
{
    const bool firstRiskSide = info.riskSides == 0;
    ++info.riskSides;
    info.rawSupport += quality.rawSupport;
    info.foundStableSupport += quality.foundStableSupport;
    info.frameStableSupport += quality.frameStableSupport;
    info.rawDepthConsistentSupport += quality.rawDepthConsistentSupport;
    info.reliableSupport += quality.reliableSupport;
    info.residualReliableSupport += quality.residualReliableSupport;
    info.depthConsistentSupport += quality.depthConsistentSupport;
    info.binarySupportPass =
        firstRiskSide ? quality.pass : (info.binarySupportPass && quality.pass);
}

double ComputeAdmissionSupportScore(const AdmissionScoreCandidateInfo& info)
{
    const double reliableScore =
        SaturatingCountScore(info.reliableSupport,
                             std::max(1, DynamicMapAdmissionSupportQualityMinReliableSupport()));
    const double depthScore =
        SaturatingCountScore(info.depthConsistentSupport,
                             std::max(1, DynamicMapAdmissionSupportQualityMinDepthSupport()));
    const double residualScore =
        SaturatingCountScore(info.residualReliableSupport,
                             std::max(1, DynamicMapAdmissionSupportQualityMinReliableSupport()));
    const double foundScore =
        SaturatingCountScore(info.foundStableSupport,
                             std::max(1, info.rawSupport));
    const double frameScore =
        SaturatingCountScore(info.frameStableSupport,
                             std::max(1, info.rawSupport));

    return Clamp01(0.35 * reliableScore +
                   0.30 * depthScore +
                   0.15 * residualScore +
                   0.10 * foundScore +
                   0.10 * frameScore);
}

bool AdmissionScoreSupportAllowsGeometry(const AdmissionScoreCandidateInfo& info)
{
    if(info.rawSupport < DynamicMapAdmissionScoreMinRawSupport())
        return false;
    if(info.depthConsistentSupport <= 0 && info.reliableSupport <= 0)
        return false;
    return info.supportScore >= DynamicMapAdmissionScoreMinSupportScore();
}

double ComputeAdmissionCandidateScore(const double parallaxScore,
                                      const double reprojRatio1,
                                      const double reprojRatio2,
                                      const double scaleScore)
{
    const double reprojScore = Clamp01(1.0 - std::max(reprojRatio1, reprojRatio2));
    return Clamp01(0.15 * Clamp01(parallaxScore) +
                   0.60 * reprojScore +
                   0.25 * Clamp01(scaleScore));
}

void PrintAdmissionV5CandidateSupportEvent(KeyFrame* pCurrentKF,
                                           KeyFrame* pNeighborKF,
                                           const int idx1,
                                           const int idx2,
                                           const bool isBoundaryRiskCurrent,
                                           const bool isBoundaryRiskNeighbor,
                                           const bool supportOk,
                                           const AdmissionScoreCandidateInfo& info)
{
    if(!pCurrentKF || !pNeighborKF)
        return;

    AdmissionV5AggregateStats& stats = MutableAdmissionV5AggregateStats();
    ++stats.candidateEvents;
    ++stats.supportCandidates;
    stats.supportScoreSum += info.supportScore;
    stats.rawSupportSum += info.rawSupport;
    stats.reliableSupportSum += info.reliableSupport;
    stats.residualSupportSum += info.residualReliableSupport;
    stats.depthSupportSum += info.depthConsistentSupport;
    if(supportOk)
        ++stats.supportAccepted;
    else
        ++stats.rejectSupport;

    if(!DynamicMapAdmissionV5TraceLog())
        return;

    std::cout << "[STSLAM_DYNAMIC_MAP_ADMISSION_V5_CANDIDATE]"
              << " frame=" << pCurrentKF->mnFrameId
              << " stage=support"
              << " decision=" << (supportOk ? "support_accepted" : "reject_support")
              << " current_kf=" << pCurrentKF->mnId
              << " neighbor_kf=" << pNeighborKF->mnId
              << " idx1=" << idx1
              << " idx2=" << idx2
              << " risk_current=" << (isBoundaryRiskCurrent ? 1 : 0)
              << " risk_neighbor=" << (isBoundaryRiskNeighbor ? 1 : 0)
              << " support_candidate=1"
              << " support_accepted=" << (supportOk ? 1 : 0)
              << " reject_support=" << (supportOk ? 0 : 1)
              << " geom_candidate=0"
              << " reject_parallax=0"
              << " reject_triangulate=0"
              << " reject_depth=0"
              << " reject_reproj1=0"
              << " reject_reproj2=0"
              << " reject_scale=0"
              << " reject_score=0"
              << " created=0"
              << " raw_support=" << info.rawSupport
              << " found_support=" << info.foundStableSupport
              << " frame_support=" << info.frameStableSupport
              << " raw_depth_support=" << info.rawDepthConsistentSupport
              << " reliable_support=" << info.reliableSupport
              << " residual_support=" << info.residualReliableSupport
              << " depth_support=" << info.depthConsistentSupport
              << " support_score=" << info.supportScore
              << " binary_support_pass=" << (info.binarySupportPass ? 1 : 0)
              << " parallax_score=0"
              << " reproj_ratio1=1"
              << " reproj_ratio2=1"
              << " scale_score=0"
              << " candidate_score=0"
              << " total_score=0"
              << std::endl;
}

void PrintAdmissionV5CandidateGeometryEvent(KeyFrame* pCurrentKF,
                                            KeyFrame* pNeighborKF,
                                            const int idx1,
                                            const int idx2,
                                            const char* decision,
                                            const AdmissionScoreCandidateInfo& info,
                                            const double parallaxScore,
                                            const double reprojRatio1,
                                            const double reprojRatio2,
                                            const double scaleScore,
                                            const double candidateScore,
                                            const double totalScore)
{
    if(!pCurrentKF || !pNeighborKF)
        return;

    const std::string reason = decision ? decision : "unknown";
    AdmissionV5AggregateStats& stats = MutableAdmissionV5AggregateStats();
    ++stats.candidateEvents;
    ++stats.geomEvents;
    if(reason == "reject_parallax")
        ++stats.rejectParallax;
    else if(reason == "reject_triangulate")
        ++stats.rejectTriangulate;
    else if(reason == "reject_depth")
        ++stats.rejectDepth;
    else if(reason == "reject_reproj1")
        ++stats.rejectReproj1;
    else if(reason == "reject_reproj2")
        ++stats.rejectReproj2;
    else if(reason == "reject_scale")
        ++stats.rejectScale;
    else if(reason == "reject_score")
        ++stats.rejectScore;
    else if(reason == "created")
    {
        ++stats.created;
        stats.createdSupportScoreSum += info.supportScore;
        stats.createdCandidateScoreSum += candidateScore;
        stats.createdTotalScoreSum += totalScore;
        stats.createdReprojRatio1Sum += reprojRatio1;
        stats.createdReprojRatio2Sum += reprojRatio2;
        stats.createdParallaxScoreSum += parallaxScore;
        stats.createdScaleScoreSum += scaleScore;
    }

    if(!DynamicMapAdmissionV5TraceLog())
        return;

    std::cout << "[STSLAM_DYNAMIC_MAP_ADMISSION_V5_CANDIDATE]"
              << " frame=" << pCurrentKF->mnFrameId
              << " stage=geometry"
              << " decision=" << reason
              << " current_kf=" << pCurrentKF->mnId
              << " neighbor_kf=" << pNeighborKF->mnId
              << " idx1=" << idx1
              << " idx2=" << idx2
              << " risk_current=" << (pCurrentKF->IsFeatureStaticNearDynamicMask(idx1) ? 1 : 0)
              << " risk_neighbor=" << (pNeighborKF->IsFeatureStaticNearDynamicMask(idx2) ? 1 : 0)
              << " support_candidate=0"
              << " support_accepted=0"
              << " reject_support=0"
              << " geom_candidate=1"
              << " reject_parallax=" << (reason == "reject_parallax" ? 1 : 0)
              << " reject_triangulate=" << (reason == "reject_triangulate" ? 1 : 0)
              << " reject_depth=" << (reason == "reject_depth" ? 1 : 0)
              << " reject_reproj1=" << (reason == "reject_reproj1" ? 1 : 0)
              << " reject_reproj2=" << (reason == "reject_reproj2" ? 1 : 0)
              << " reject_scale=" << (reason == "reject_scale" ? 1 : 0)
              << " reject_score=" << (reason == "reject_score" ? 1 : 0)
              << " created=" << (reason == "created" ? 1 : 0)
              << " raw_support=" << info.rawSupport
              << " found_support=" << info.foundStableSupport
              << " frame_support=" << info.frameStableSupport
              << " raw_depth_support=" << info.rawDepthConsistentSupport
              << " reliable_support=" << info.reliableSupport
              << " residual_support=" << info.residualReliableSupport
              << " depth_support=" << info.depthConsistentSupport
              << " support_score=" << info.supportScore
              << " binary_support_pass=" << (info.binarySupportPass ? 1 : 0)
              << " parallax_score=" << parallaxScore
              << " reproj_ratio1=" << reprojRatio1
              << " reproj_ratio2=" << reprojRatio2
              << " scale_score=" << scaleScore
              << " candidate_score=" << candidateScore
              << " total_score=" << totalScore
              << std::endl;
}

void RecordAdmissionV5LifecycleEvent(KeyFrame* pCurrentKF,
                                     const int recentPointsAtEntry,
                                     const int remainingRecentPoints,
                                     const int scoreRecent,
                                     const int scorePreBad,
                                     const int scoreCulledFoundRatio,
                                     const int scoreCulledLowObs,
                                     const int scoreSurvived,
                                     const int scoreMatured,
                                     const int scorePoseUseEdges,
                                     const int scorePoseUseInliers,
                                     const double scorePoseUseChi2Mean)
{
    AdmissionV5AggregateStats& stats = MutableAdmissionV5AggregateStats();
    ++stats.lifecycleRows;
    stats.lifecycleScoreRecent += scoreRecent;
    stats.lifecycleScorePreBad += scorePreBad;
    stats.lifecycleScoreCulledFoundRatio += scoreCulledFoundRatio;
    stats.lifecycleScoreCulledLowObs += scoreCulledLowObs;
    stats.lifecycleScoreSurvived += scoreSurvived;
    stats.lifecycleScoreMatured += scoreMatured;
    stats.lifecyclePoseUseEdges += scorePoseUseEdges;
    stats.lifecyclePoseUseInliers += scorePoseUseInliers;
    stats.lifecyclePoseUseChi2WeightedSum +=
        scorePoseUseChi2Mean * static_cast<double>(scorePoseUseEdges);

    if(!DynamicMapAdmissionV5TraceLog() || !pCurrentKF)
        return;

    std::cout << "[STSLAM_DYNAMIC_MAP_ADMISSION_V5_LIFECYCLE]"
              << " frame=" << pCurrentKF->mnFrameId
              << " current_kf=" << pCurrentKF->mnId
              << " recent_points=" << recentPointsAtEntry
              << " remaining_recent_points=" << remainingRecentPoints
              << " score_recent=" << scoreRecent
              << " score_prebad=" << scorePreBad
              << " score_culled_found_ratio=" << scoreCulledFoundRatio
              << " score_culled_low_obs=" << scoreCulledLowObs
              << " score_survived=" << scoreSurvived
              << " score_matured=" << scoreMatured
              << " score_pose_use_edges=" << scorePoseUseEdges
              << " score_pose_use_inliers=" << scorePoseUseInliers
              << " score_pose_use_chi2_mean=" << scorePoseUseChi2Mean
              << std::endl;
}

bool IsDepthConsistentForSupportQuality(const float candidateDepth,
                                        const float supportDepth)
{
    if(candidateDepth <= 0.0f || supportDepth <= 0.0f ||
       !std::isfinite(candidateDepth) || !std::isfinite(supportDepth))
        return false;

    const double denom = std::max(static_cast<double>(candidateDepth),
                                  static_cast<double>(supportDepth));
    if(denom <= 0.0)
        return false;

    const double relDiff =
        std::fabs(static_cast<double>(candidateDepth) -
                  static_cast<double>(supportDepth)) /
        denom;
    return relDiff <= DynamicMapAdmissionSupportQualityMaxDepthRelDiff();
}

bool IsSupportQualityStableAcrossFrames(MapPoint* pMP)
{
    if(!pMP)
        return false;

    const int minFrameSpan = DynamicMapAdmissionSupportQualityMinFrameSpan();
    if(minFrameSpan <= 0)
        return true;

    const long firstFrame = pMP->mnFirstObservationFrame;
    const long lastFrame = pMP->mnLastObservationFrame;
    const long frameSpan =
        (firstFrame >= 0 && lastFrame >= firstFrame) ?
        (lastFrame - firstFrame) :
        0;
    return frameSpan >= minFrameSpan ||
           pMP->mnObservationCount >= minFrameSpan + 1 ||
           pMP->Observations() >= minFrameSpan + 1;
}

bool IsSupportQualityResidualStable(MapPoint* pMP)
{
    if(!pMP)
        return false;

    const int minResidualObs = DynamicMapAdmissionSupportQualityMinResidualObs();
    if(minResidualObs <= 0)
        return true;

    const int poseUseCount = pMP->GetSupportQualityPoseUseCount();
    if(poseUseCount < minResidualObs)
        return false;

    return pMP->GetSupportQualityPoseUseInlierRate() >=
               DynamicMapAdmissionSupportQualityMinInlierRate() &&
           pMP->GetSupportQualityPoseUseMeanChi2() <=
               DynamicMapAdmissionSupportQualityMaxMeanChi2();
}

AdmissionSupportQualityResult EvaluateCleanStaticMapSupportQualityNearFeature(
    KeyFrame* pKF,
    const int idx,
    const int radiusPx,
    const int minObservations,
    const float candidateDepth)
{
    AdmissionSupportQualityResult result;
    if(!pKF || idx < 0 || idx >= static_cast<int>(pKF->mvKeysUn.size()))
        return result;

    const cv::Point2f center = pKF->mvKeysUn[idx].pt;
    const std::vector<size_t> vNeighborIndices =
        pKF->GetFeaturesInArea(center.x,
                               center.y,
                               static_cast<float>(std::max(0, radiusPx)));
    for(size_t neighbor = 0; neighbor < vNeighborIndices.size(); ++neighbor)
    {
        const size_t candidateIdx = vNeighborIndices[neighbor];
        if(static_cast<int>(candidateIdx) == idx)
            continue;

        MapPoint* pMP = pKF->GetMapPoint(candidateIdx);
        if(!pMP || pMP->isBad() || pMP->IsDynamicInstanceObservationPoint() ||
           pMP->IsInstanceStructurePoint())
            continue;
        if(pMP->Observations() < minObservations)
            continue;
        if(pKF->GetFeatureInstanceId(candidateIdx) > 0 ||
           pKF->IsFeatureStaticNearDynamicMask(candidateIdx))
            continue;

        ++result.rawSupport;

        const bool foundStable =
            pMP->GetFoundRatio() >=
            DynamicMapAdmissionSupportQualityMinFoundRatio();
        const bool frameStable = IsSupportQualityStableAcrossFrames(pMP);
        const bool residualStable = IsSupportQualityResidualStable(pMP);
        const float supportDepth =
            candidateIdx < pKF->mvDepth.size() ?
            pKF->mvDepth[candidateIdx] :
            -1.0f;
        const bool depthStable =
            IsDepthConsistentForSupportQuality(candidateDepth, supportDepth);
        if(foundStable)
            ++result.foundStableSupport;
        if(frameStable)
            ++result.frameStableSupport;
        if(residualStable)
            ++result.residualReliableSupport;
        if(depthStable)
            ++result.rawDepthConsistentSupport;

        const bool reliable = foundStable && frameStable && residualStable;
        if(!reliable)
            continue;

        ++result.reliableSupport;
        if(depthStable)
            ++result.depthConsistentSupport;
    }

    result.pass =
        result.reliableSupport >=
            DynamicMapAdmissionSupportQualityMinReliableSupport() &&
        result.depthConsistentSupport >=
            DynamicMapAdmissionSupportQualityMinDepthSupport();
    return result;
}

bool RequireMotionEvidenceForRgbdDynamicSplit()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_RGBD_DYNAMIC_SPLIT_REQUIRE_MOTION_EVIDENCE", true);
    return value;
}

bool SequentialLocalMappingMode()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_SEQUENTIAL_LOCAL_MAPPING", false);
    return value;
}

bool HoldAcceptKeyFramesWhenSequentialQueueHasWork()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_SEQUENTIAL_LOCAL_MAPPING_HOLD_ACCEPT_WHEN_QUEUED", true);
    return value;
}

int GetEnvIntOrDefault(const char* name, const int defaultValue, const int minValue);

int SequentialLocalMappingMaintenancePeriod()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_SEQUENTIAL_LOCAL_MAPPING_MAINTENANCE_PERIOD", 1, 1);
    return value;
}

bool RequireBackendMotionEvidenceForRgbdDynamicSplit()
{
    static const bool value =
        GetEnvFlagOrDefault("STSLAM_RGBD_DYNAMIC_SPLIT_REQUIRE_BACKEND_EVIDENCE", true);
    return value;
}

int GetEnvIntOrDefault(const char* name, const int defaultValue, const int minValue)
{
    const char* envValue = std::getenv(name);
    if(!envValue)
        return defaultValue;
    const int value = std::atoi(envValue);
    return std::max(minValue, value);
}

int GetRgbdDynamicSplitMinBackendMotionEvidence()
{
    static const int value =
        GetEnvIntOrDefault("STSLAM_RGBD_DYNAMIC_SPLIT_MIN_BACKEND_EVIDENCE", 1, 0);
    return value;
}

double GetRgbdDynamicSplitMinTranslation()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_RGBD_DYNAMIC_SPLIT_MIN_TRANSLATION", 0.01, 0.0);
    return value;
}

double GetRgbdDynamicSplitMinRotationDeg()
{
    static const double value =
        GetEnvDoubleOrDefault("STSLAM_RGBD_DYNAMIC_SPLIT_MIN_ROTATION_DEG", 1.0, 0.0);
    return value;
}

double RotationAngleDeg(const Eigen::Matrix3f& rotation)
{
    const Eigen::AngleAxisd angleAxis(rotation.cast<double>());
    return std::abs(angleAxis.angle()) * 180.0 / 3.14159265358979323846;
}

bool IsFiniteSE3(const Sophus::SE3f& pose)
{
    return pose.matrix3x4().allFinite();
}

bool ShouldDetachRgbdInstanceFromStaticPath(Map* pMap, const int instanceId)
{
    if(!SplitDetectedDynamicFeaturesFromStaticMapping())
        return false;
    if(!RequireMotionEvidenceForRgbdDynamicSplit())
        return instanceId > 0;
    if(!pMap || instanceId <= 0)
        return false;

    Instance* pInstance = pMap->GetInstance(instanceId);
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
        const double rotationDeg = RotationAngleDeg(record.velocity.rotationMatrix());
        return translationNorm >= GetRgbdDynamicSplitMinTranslation() ||
               rotationDeg >= GetRgbdDynamicSplitMinRotationDeg();
    }

    return false;
}

bool UseStrictPaperArchitectureDefaults()
{
    const char* envValue = std::getenv("STSLAM_MODULE8_PROFILE");
    if(!envValue || std::string(envValue).empty())
        return true;

    const std::string profile(envValue);
    return profile == "paper_strict" || profile == "paper_eq16";
}

bool StrictInstanceStructureFromDynamicObservationsOnly()
{
    return GetEnvFlagOrDefault("STSLAM_STRICT_INSTANCE_STRUCTURE_DYNAMIC_ONLY",
                               UseStrictPaperArchitectureDefaults());
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

float GetSameInstanceParallaxCosThreshold()
{
    const char* envValue = std::getenv("STSLAM_SAME_INSTANCE_PARALLAX_COS_MAX");
    if(!envValue)
        return 0.99995f;

    const float value = std::atof(envValue);
    return std::min(0.999999f, std::max(0.99f, value));
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

template<typename TKey>
std::string FormatTopCounts(const std::map<TKey, int>& counts, size_t maxItems = 5)
{
    std::vector<std::pair<TKey, int> > items(counts.begin(), counts.end());
    std::sort(items.begin(), items.end(),
              [](const std::pair<TKey, int>& lhs, const std::pair<TKey, int>& rhs)
              {
                  if(lhs.second != rhs.second)
                      return lhs.second > rhs.second;
                  return lhs.first < rhs.first;
              });

    std::ostringstream oss;
    const size_t n = std::min(maxItems, items.size());
    for(size_t i = 0; i < n; ++i)
    {
        if(i > 0)
            oss << ",";
        oss << items[i].first << ":" << items[i].second;
    }
    return oss.str();
}

struct KeyFrameFeatureStats
{
    int totalFeatures = 0;
    int unmatchedFeatures = 0;
    int totalInstanceFeatures = 0;
    int unmatchedInstanceFeatures = 0;
    std::map<int, int> totalPerInstance;
    std::map<int, int> unmatchedPerInstance;
};

KeyFrameFeatureStats CollectKeyFrameFeatureStats(KeyFrame* pKF)
{
    KeyFrameFeatureStats stats;
    if(!pKF)
        return stats;

    const int nFeatures = pKF->N;
    for(int idx = 0; idx < nFeatures; ++idx)
    {
        ++stats.totalFeatures;

        const int instanceId = pKF->GetFeatureInstanceId(idx);
        MapPoint* pMP = pKF->GetMapPoint(idx);
        const bool hasValidMapPoint = pMP && !pMP->isBad();

        if(!hasValidMapPoint)
            ++stats.unmatchedFeatures;

        if(instanceId <= 0)
            continue;

        ++stats.totalInstanceFeatures;
        ++stats.totalPerInstance[instanceId];

        if(!hasValidMapPoint)
        {
            ++stats.unmatchedInstanceFeatures;
            ++stats.unmatchedPerInstance[instanceId];
        }
    }

    return stats;
}

int CountCommonFeatureWords(const DBoW2::FeatureVector& lhs,
                            const DBoW2::FeatureVector& rhs)
{
    int commonWords = 0;
    DBoW2::FeatureVector::const_iterator lit = lhs.begin();
    DBoW2::FeatureVector::const_iterator rit = rhs.begin();

    while(lit != lhs.end() && rit != rhs.end())
    {
        if(lit->first == rit->first)
        {
            ++commonWords;
            ++lit;
            ++rit;
        }
        else if(lit->first < rit->first)
        {
            lit = lhs.lower_bound(rit->first);
        }
        else
        {
            rit = rhs.lower_bound(lit->first);
        }
    }

    return commonWords;
}

std::map<int, int> CollectSharedUnmatchedInstanceSupport(const KeyFrameFeatureStats& lhs,
                                                         const KeyFrameFeatureStats& rhs)
{
    std::map<int, int> sharedCounts;
    for(std::map<int, int>::const_iterator it = lhs.unmatchedPerInstance.begin();
        it != lhs.unmatchedPerInstance.end();
        ++it)
    {
        std::map<int, int>::const_iterator jt = rhs.unmatchedPerInstance.find(it->first);
        if(jt == rhs.unmatchedPerInstance.end())
            continue;

        sharedCounts[it->first] = std::min(it->second, jt->second);
    }

    return sharedCounts;
}

int MaxCountValue(const std::map<int, int>& counts)
{
    int bestValue = 0;
    for(std::map<int, int>::const_iterator it = counts.begin(); it != counts.end(); ++it)
        bestValue = std::max(bestValue, it->second);
    return bestValue;
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

} // namespace

LocalMapping::LocalMapping(System* pSys, Atlas *pAtlas, const float bMonocular, bool bInertial, const string &_strSeqName):
    mpSystem(pSys), mbMonocular(bMonocular), mbInertial(bInertial), mbResetRequested(false), mbResetRequestedActiveMap(false), mbFinishRequested(false), mbFinished(true), mpAtlas(pAtlas), bInitializing(false),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true),
    mIdxInit(0), mScale(1.0), mInitSect(0), mbNotBA1(true), mbNotBA2(true), mIdxIteration(0),
    mbSequentialMaintenancePending(false), mnSequentialKeyFramesSinceMaintenance(0),
    mnAdmissionLastLBAEdges(0), mnAdmissionLastLBAMapPoints(0),
    mnAdmissionLastLBAOptKeyFrames(0), mnAdmissionLastLBAFixedKeyFrames(0),
    mbAdmissionStateHasKfStepEwma(false), mdAdmissionStateKfStepEwma(0.0),
    infoInertial(Eigen::MatrixXd::Zero(9,9))
{
    mnMatchesInliers = 0;

    mbBadImu = false;

    mTinit = 0.f;

    mNumLM = 0;
    mNumKFCulling=0;

#ifdef REGISTER_TIMES
    nLBA_exec = 0;
    nLBA_abort = 0;
#endif

}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::Run()
{
    mbFinished = false;

    while(1)
    {
        if(CheckFinish())
            break;

        RunOneStep();

        usleep(3000);
    }

    PrintAdmissionV5AggregateSummary();
    SetFinish();
}

bool LocalMapping::RunOneStep()
{
    bool processedKeyFrame = false;

    // Tracking will see that Local Mapping is busy
    SetAcceptKeyFrames(false);

    // Check if there are keyframes in the queue
    if(CheckNewKeyFrames() && !mbBadImu)
    {
#ifdef REGISTER_TIMES
        double timeLBA_ms = 0;
        double timeKFCulling_ms = 0;

        std::chrono::steady_clock::time_point time_StartProcessKF = std::chrono::steady_clock::now();
#endif
        processedKeyFrame = true;

        // BoW conversion and insertion in Map
        ProcessNewKeyFrame();
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndProcessKF = std::chrono::steady_clock::now();

        double timeProcessKF = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndProcessKF - time_StartProcessKF).count();
        vdKFInsert_ms.push_back(timeProcessKF);
#endif

        // Check recent MapPoints
        MapPointCulling();
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndMPCulling = std::chrono::steady_clock::now();

        double timeMPCulling = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndMPCulling - time_EndProcessKF).count();
        vdMPCulling_ms.push_back(timeMPCulling);
#endif

        // Triangulate new MapPoints
        CreateNewMapPoints();
        UpdateInstanceStructure();

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndMPCreation = std::chrono::steady_clock::now();

        double timeMPCreation = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndMPCreation - time_EndMPCulling).count();
        vdMPCreation_ms.push_back(timeMPCreation);
#endif

        mbSequentialMaintenancePending = true;
        if(SequentialLocalMappingMode())
            ++mnSequentialKeyFramesSinceMaintenance;

        mbAbortBA = false;

        const bool queueDrained = !CheckNewKeyFrames();
        bool runMaintenanceNow = queueDrained && !stopRequested();
        if(runMaintenanceNow && SequentialLocalMappingMode())
        {
            runMaintenanceNow =
                mnSequentialKeyFramesSinceMaintenance >= SequentialLocalMappingMaintenancePeriod();
        }

        if(runMaintenanceNow)
            RunMaintenanceStep(false);
    }
    else if(Stop() && !mbBadImu)
    {
        // Safe area to stop
        while(isStopped() && !CheckFinish())
        {
            usleep(3000);
        }
    }

    ResetIfRequested();

    // Tracking will see that Local Mapping is idle
    SetAcceptKeyFrames(true);

    return processedKeyFrame;
}

bool LocalMapping::RunMaintenanceStep(const bool force)
{
    if(!mpCurrentKeyFrame || !mbSequentialMaintenancePending)
        return false;
    if(!force && CheckNewKeyFrames())
        return false;
    if(stopRequested())
        return false;

#ifdef REGISTER_TIMES
    double timeKFCulling_ms = 0;
    std::chrono::steady_clock::time_point time_StartMaintenance = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point time_EndMPCreation = time_StartMaintenance;
#endif

    SearchInNeighbors();

#ifdef REGISTER_TIMES
    time_EndMPCreation = std::chrono::steady_clock::now();
#endif

    bool b_doneLBA = false;
    int num_FixedKF_BA = 0;
    int num_OptKF_BA = 0;
    int num_MPs_BA = 0;
    int num_edges_BA = 0;

    if(mpAtlas->KeyFramesInMap()>2)
    {
        if(mbInertial && mpCurrentKeyFrame->GetMap()->isImuInitialized())
        {
            float dist = (mpCurrentKeyFrame->mPrevKF->GetCameraCenter() - mpCurrentKeyFrame->GetCameraCenter()).norm() +
                    (mpCurrentKeyFrame->mPrevKF->mPrevKF->GetCameraCenter() - mpCurrentKeyFrame->mPrevKF->GetCameraCenter()).norm();

            if(dist>0.05)
                mTinit += mpCurrentKeyFrame->mTimeStamp - mpCurrentKeyFrame->mPrevKF->mTimeStamp;
            if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2())
            {
                if((mTinit<10.f) && (dist<0.02))
                {
                    cout << "Not enough motion for initializing. Reseting..." << endl;
                    unique_lock<mutex> lock(mMutexReset);
                    mbResetRequestedActiveMap = true;
                    mpMapToReset = mpCurrentKeyFrame->GetMap();
                    mbBadImu = true;
                }
            }

            bool bLarge = ((mpTracker->GetMatchesInliers()>75)&&mbMonocular)||((mpTracker->GetMatchesInliers()>100)&&!mbMonocular);
            Optimizer::LocalInertialBA(mpCurrentKeyFrame, &mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA, bLarge, !mpCurrentKeyFrame->GetMap()->GetIniertialBA2());
            b_doneLBA = true;
        }
        else
        {
            Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA);
            b_doneLBA = true;
        }
    }

    if(b_doneLBA)
    {
        mnAdmissionLastLBAEdges = num_edges_BA;
        mnAdmissionLastLBAMapPoints = num_MPs_BA;
        mnAdmissionLastLBAOptKeyFrames = num_OptKF_BA;
        mnAdmissionLastLBAFixedKeyFrames = num_FixedKF_BA;
    }

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndLBA = std::chrono::steady_clock::now();

    if(b_doneLBA)
    {
        const double timeLBA_ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLBA - time_EndMPCreation).count();
        vdLBA_ms.push_back(timeLBA_ms);

        nLBA_exec += 1;
        if(mbAbortBA)
            nLBA_abort += 1;
        vnLBA_edges.push_back(num_edges_BA);
        vnLBA_KFopt.push_back(num_OptKF_BA);
        vnLBA_KFfixed.push_back(num_FixedKF_BA);
        vnLBA_MPs.push_back(num_MPs_BA);
    }
#endif

    if(!mpCurrentKeyFrame->GetMap()->isImuInitialized() && mbInertial)
    {
        if (mbMonocular)
            InitializeIMU(1e2, 1e10, true);
        else
            InitializeIMU(1e2, 1e5, true);
    }

    KeyFrameCulling();

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndKFCulling = std::chrono::steady_clock::now();
    timeKFCulling_ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndKFCulling - time_EndLBA).count();
    vdKFCulling_ms.push_back(timeKFCulling_ms);
#endif

    if ((mTinit<50.0f) && mbInertial)
    {
        if(mpCurrentKeyFrame->GetMap()->isImuInitialized() && mpTracker->mState==Tracking::OK)
        {
            if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA1()){
                if (mTinit>5.0f)
                {
                    cout << "start VIBA 1" << endl;
                    mpCurrentKeyFrame->GetMap()->SetIniertialBA1();
                    if (mbMonocular)
                        InitializeIMU(1.f, 1e5, true);
                    else
                        InitializeIMU(1.f, 1e5, true);
                    cout << "end VIBA 1" << endl;
                }
            }
            else if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2()){
                if (mTinit>15.0f){
                    cout << "start VIBA 2" << endl;
                    mpCurrentKeyFrame->GetMap()->SetIniertialBA2();
                    if (mbMonocular)
                        InitializeIMU(0.f, 0.f, true);
                    else
                        InitializeIMU(0.f, 0.f, true);
                    cout << "end VIBA 2" << endl;
                }
            }

            if (((mpAtlas->KeyFramesInMap())<=200) &&
                    ((mTinit>25.0f && mTinit<25.5f)||
                    (mTinit>35.0f && mTinit<35.5f)||
                    (mTinit>45.0f && mTinit<45.5f)||
                    (mTinit>55.0f && mTinit<55.5f)||
                    (mTinit>65.0f && mTinit<65.5f)||
                    (mTinit>75.0f && mTinit<75.5f))){
                if (mbMonocular)
                    ScaleRefinement();
            }
        }
    }

#ifdef REGISTER_TIMES
    vdLBASync_ms.push_back(timeKFCulling_ms);
    vdKFCullingSync_ms.push_back(timeKFCulling_ms);
#endif

    if(mpLoopCloser)
        mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndLocalMap = std::chrono::steady_clock::now();
    const double timeLocalMap = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLocalMap - time_StartMaintenance).count();
    vdLMTotal_ms.push_back(timeLocalMap);
#endif

    mbSequentialMaintenancePending = false;
    mnSequentialKeyFramesSinceMaintenance = 0;
    return true;
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
    if(SequentialLocalMappingMode() && HoldAcceptKeyFramesWhenSequentialQueueHasWork())
        SetAcceptKeyFrames(false);
}


bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    mpAtlas->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::EmptyQueue()
{
    while(CheckNewKeyFrames())
        ProcessNewKeyFrame();
}

void LocalMapping::UpdateInstanceStructure()
{
    if(!mpCurrentKeyFrame)
        return;

    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    const bool strictDynamicOnlyStructure =
        StrictInstanceStructureFromDynamicObservationsOnly();
    int ordinaryInstancePointsNotPromoted = 0;
    int dynamicOrStructurePointsBound = 0;
    for(size_t i = 0; i < vpMapPointMatches.size(); ++i)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(!pMP || pMP->isBad())
            continue;

        const int semanticLabel = mpCurrentKeyFrame->GetFeatureSemanticLabel(i);
        const int instanceId = mpCurrentKeyFrame->GetFeatureInstanceId(i);
        if(instanceId <= 0)
        {
            if(semanticLabel > 0 && pMP->GetSemanticLabel() <= 0)
                pMP->SetSemanticLabel(semanticLabel);
            continue;
        }

        if(strictDynamicOnlyStructure &&
           !pMP->IsInstanceStructurePoint() &&
           !pMP->IsDynamicInstanceObservationPoint())
        {
            if(semanticLabel > 0 && pMP->GetSemanticLabel() <= 0)
                pMP->SetSemanticLabel(semanticLabel);
            ++ordinaryInstancePointsNotPromoted;
            continue;
        }

        Instance* pInstance = BindMapPointToInstance(pMP, instanceId, semanticLabel, false);
        if(!pInstance)
            continue;
        ++dynamicOrStructurePointsBound;

        if(HasMatureInstanceBackendState(pInstance, mpCurrentKeyFrame))
        {
            pInstance->UpdateMotionPrior(mpCurrentKeyFrame, pInstance->GetVelocity());
            pInstance->UpdatePoseProxy(mpCurrentKeyFrame, pInstance->GetLastPoseEstimate());
        }
    }

    if(strictDynamicOnlyStructure &&
       (ordinaryInstancePointsNotPromoted > 0 || dynamicOrStructurePointsBound > 0))
    {
        std::cout << "[STSLAM_INSTANCE_LIFECYCLE] keyframe_id=" << mpCurrentKeyFrame->mnId
                  << " frame=" << mpCurrentKeyFrame->mnFrameId
                  << " ordinary_instance_points_not_promoted="
                  << ordinaryInstancePointsNotPromoted
                  << " dynamic_or_structure_points_bound="
                  << dynamicOrStructurePointsBound
                  << std::endl;
    }
}

void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    const int recentPointsAtEntry = mlpRecentAddedMapPoints.size();
    int borrar = recentPointsAtEntry;
    const bool nearBoundaryDiagnostics = EnableNearBoundaryDiagnostics();
    const bool v5UsefulnessLog = DynamicMapAdmissionV5UsefulnessLog();
    const bool v7Probation = DynamicMapAdmissionV7Probation();
    const bool constraintRoleLog = DynamicMapAdmissionConstraintRoleLog();
    int recentNearBoundaryPoints = 0;
    int recentCleanStaticPoints = 0;
    int recentDirectDynamicPoints = 0;
    int recentScoreAdmissionPoints = 0;
    int culledNearBoundaryPreBad = 0;
    int culledNearBoundaryFoundRatio = 0;
    int culledNearBoundaryLowObs = 0;
    int culledScoreAdmissionPreBad = 0;
    int culledScoreAdmissionFoundRatio = 0;
    int culledScoreAdmissionLowObs = 0;
    int culledScoreAdmissionV7Residual = 0;
    int culledScoreAdmissionV7LowUse = 0;
    int culledCleanFoundRatio = 0;
    int culledCleanLowObs = 0;
    int survivedNearBoundaryPoints = 0;
    int survivedCleanStaticPoints = 0;
    int survivedScoreAdmissionPoints = 0;
    int maturedNearBoundaryPoints = 0;
    int maturedCleanStaticPoints = 0;
    int maturedScoreAdmissionPoints = 0;
    int scoreAdmissionPoseUseEdges = 0;
    int scoreAdmissionPoseUseInliers = 0;
    double scoreAdmissionPoseUseChi2Sum = 0.0;
    int scoreAdmissionLocalBAWindowPoints = 0;
    int scoreAdmissionLocalBAEdgePoints = 0;
    int scoreAdmissionLocalBAEdges = 0;
    int scoreAdmissionLocalBAInliers = 0;
    int scoreAdmissionLocalBAFixedEdges = 0;
    int scoreAdmissionLocalBALocalEdges = 0;
    double scoreAdmissionLocalBAChi2Sum = 0.0;
    int scoreAdmissionObsGe2 = 0;
    int scoreAdmissionObsGe3 = 0;
    int scoreAdmissionObsSum = 0;
    double scoreAdmissionRefDistanceSum = 0.0;
    int scoreAdmissionRefDistanceCount = 0;

    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        const bool admissionNearBoundary =
            nearBoundaryDiagnostics && pMP &&
            pMP->WasCreatedFromStaticNearDynamicBoundary();
        const bool admissionDirectDynamic =
            nearBoundaryDiagnostics && pMP &&
            pMP->WasCreatedFromDirectDynamicAdmission();
        const bool scoreAdmission =
            (v5UsefulnessLog || v7Probation) && pMP &&
            pMP->WasCreatedFromScoreAdmission();
        if(nearBoundaryDiagnostics && pMP)
        {
            if(admissionDirectDynamic)
                ++recentDirectDynamicPoints;
            else if(admissionNearBoundary)
                ++recentNearBoundaryPoints;
            else
                ++recentCleanStaticPoints;
        }
        if(scoreAdmission)
        {
            ++recentScoreAdmissionPoints;
            const int poseUseCount = pMP->GetSupportQualityPoseUseCount();
            scoreAdmissionPoseUseEdges += poseUseCount;
            scoreAdmissionPoseUseInliers += pMP->GetSupportQualityPoseUseInliers();
            scoreAdmissionPoseUseChi2Sum +=
                pMP->GetSupportQualityPoseUseMeanChi2() *
                static_cast<double>(poseUseCount);
            if(constraintRoleLog)
            {
                const int lbaWindows = pMP->GetScoreAdmissionLocalBAWindowCount();
                const int lbaEdges = pMP->GetScoreAdmissionLocalBAEdgeCount();
                if(lbaWindows > 0)
                    ++scoreAdmissionLocalBAWindowPoints;
                if(lbaEdges > 0)
                    ++scoreAdmissionLocalBAEdgePoints;
                scoreAdmissionLocalBAEdges += lbaEdges;
                scoreAdmissionLocalBAInliers +=
                    pMP->GetScoreAdmissionLocalBAInliers();
                scoreAdmissionLocalBAFixedEdges +=
                    pMP->GetScoreAdmissionLocalBAFixedEdges();
                scoreAdmissionLocalBALocalEdges +=
                    pMP->GetScoreAdmissionLocalBALocalEdges();
                scoreAdmissionLocalBAChi2Sum +=
                    pMP->GetScoreAdmissionLocalBAMeanChi2() *
                    static_cast<double>(lbaEdges);

                const int observations = pMP->Observations();
                scoreAdmissionObsSum += observations;
                if(observations >= 2)
                    ++scoreAdmissionObsGe2;
                if(observations >= 3)
                    ++scoreAdmissionObsGe3;

                KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
                if(pRefKF && !pRefKF->isBad())
                {
                    scoreAdmissionRefDistanceSum +=
                        static_cast<double>((pMP->GetWorldPos() -
                                             pRefKF->GetCameraCenter()).norm());
                    ++scoreAdmissionRefDistanceCount;
                }
            }
        }

        bool v7ResidualReject = false;
        bool v7LowUseReject = false;
        if(v7Probation && scoreAdmission && pMP && !pMP->isBad())
        {
            const int ageKFs =
                static_cast<int>(nCurrentKFid) -
                static_cast<int>(pMP->mnFirstKFid);
            const int poseUseCount = pMP->GetSupportQualityPoseUseCount();
            if(ageKFs >= DynamicMapAdmissionV7ProbationMinAgeKFs() &&
               poseUseCount >= DynamicMapAdmissionV7ProbationMinPoseUse())
            {
                v7ResidualReject =
                    pMP->GetSupportQualityPoseUseInlierRate() <
                        DynamicMapAdmissionV7ProbationMinInlierRate() ||
                    pMP->GetSupportQualityPoseUseMeanChi2() >
                        DynamicMapAdmissionV7ProbationMaxMeanChi2();
            }
            else if(ageKFs >= DynamicMapAdmissionV7ProbationLowUseAgeKFs() &&
                    poseUseCount < DynamicMapAdmissionV7ProbationMinPoseUse() &&
                    pMP->Observations() <= cnThObs)
            {
                v7LowUseReject = true;
            }
        }

        if(pMP->isBad())
        {
            if(admissionNearBoundary)
                ++culledNearBoundaryPreBad;
            if(scoreAdmission)
                ++culledScoreAdmissionPreBad;
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(v7ResidualReject)
        {
            ++culledScoreAdmissionV7Residual;
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(v7LowUseReject)
        {
            ++culledScoreAdmissionV7LowUse;
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio()<0.25f)
        {
            if(admissionNearBoundary)
                ++culledNearBoundaryFoundRatio;
            else if(nearBoundaryDiagnostics && !admissionDirectDynamic)
                ++culledCleanFoundRatio;
            if(scoreAdmission)
                ++culledScoreAdmissionFoundRatio;
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            if(admissionNearBoundary)
                ++culledNearBoundaryLowObs;
            else if(nearBoundaryDiagnostics && !admissionDirectDynamic)
                ++culledCleanLowObs;
            if(scoreAdmission)
                ++culledScoreAdmissionLowObs;
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
        {
            if(admissionNearBoundary)
                ++maturedNearBoundaryPoints;
            else if(nearBoundaryDiagnostics && !admissionDirectDynamic)
                ++maturedCleanStaticPoints;
            if(scoreAdmission)
                ++maturedScoreAdmissionPoints;
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else
        {
            if(admissionNearBoundary)
                ++survivedNearBoundaryPoints;
            else if(nearBoundaryDiagnostics && !admissionDirectDynamic)
                ++survivedCleanStaticPoints;
            if(scoreAdmission)
                ++survivedScoreAdmissionPoints;
            lit++;
            borrar--;
        }
    }

    if(nearBoundaryDiagnostics)
    {
        std::cout << "[STSLAM_NEAR_BOUNDARY_CULLING]"
                  << " frame=" << mpCurrentKeyFrame->mnFrameId
                  << " current_kf=" << mpCurrentKeyFrame->mnId
                  << " recent_points=" << recentPointsAtEntry
                  << " remaining_recent_points=" << borrar
                  << " recent_near_boundary=" << recentNearBoundaryPoints
                  << " recent_clean_static=" << recentCleanStaticPoints
                  << " recent_direct_dynamic=" << recentDirectDynamicPoints
                  << " near_prebad=" << culledNearBoundaryPreBad
                  << " near_culled_found_ratio="
                  << culledNearBoundaryFoundRatio
                  << " near_culled_low_obs=" << culledNearBoundaryLowObs
                  << " near_survived=" << survivedNearBoundaryPoints
                  << " near_matured=" << maturedNearBoundaryPoints
                  << " clean_culled_found_ratio=" << culledCleanFoundRatio
                  << " clean_culled_low_obs=" << culledCleanLowObs
                  << " clean_survived=" << survivedCleanStaticPoints
                  << " clean_matured=" << maturedCleanStaticPoints
                  << std::endl;
    }

    if(v5UsefulnessLog && recentScoreAdmissionPoints > 0)
    {
        RecordAdmissionV5LifecycleEvent(
            mpCurrentKeyFrame,
            recentPointsAtEntry,
            borrar,
            recentScoreAdmissionPoints,
            culledScoreAdmissionPreBad,
            culledScoreAdmissionFoundRatio,
            culledScoreAdmissionLowObs,
            survivedScoreAdmissionPoints,
            maturedScoreAdmissionPoints,
            scoreAdmissionPoseUseEdges,
            scoreAdmissionPoseUseInliers,
            scoreAdmissionPoseUseEdges > 0 ?
                scoreAdmissionPoseUseChi2Sum /
                    static_cast<double>(scoreAdmissionPoseUseEdges) :
                0.0);
    }

    if(v7Probation && recentScoreAdmissionPoints > 0)
    {
        std::cout << "[STSLAM_DYNAMIC_MAP_ADMISSION_V7_PROBATION]"
                  << " frame=" << mpCurrentKeyFrame->mnFrameId
                  << " stage=map_point_culling"
                  << " current_kf=" << mpCurrentKeyFrame->mnId
                  << " recent_points=" << recentPointsAtEntry
                  << " remaining_recent_points=" << borrar
                  << " score_recent=" << recentScoreAdmissionPoints
                  << " score_prebad=" << culledScoreAdmissionPreBad
                  << " score_culled_found_ratio="
                  << culledScoreAdmissionFoundRatio
                  << " score_culled_low_obs="
                  << culledScoreAdmissionLowObs
                  << " v7_residual_rejected="
                  << culledScoreAdmissionV7Residual
                  << " v7_low_use_rejected="
                  << culledScoreAdmissionV7LowUse
                  << " score_survived=" << survivedScoreAdmissionPoints
                  << " score_matured=" << maturedScoreAdmissionPoints
                  << " score_pose_use_edges="
                  << scoreAdmissionPoseUseEdges
                  << " score_pose_use_inliers="
                  << scoreAdmissionPoseUseInliers
                  << " score_pose_use_chi2_mean="
                  << (scoreAdmissionPoseUseEdges > 0 ?
                      scoreAdmissionPoseUseChi2Sum /
                          static_cast<double>(scoreAdmissionPoseUseEdges) :
                      0.0)
                  << " score_lba_window_points="
                  << scoreAdmissionLocalBAWindowPoints
                  << " score_lba_edge_points="
                  << scoreAdmissionLocalBAEdgePoints
                  << " score_lba_edges="
                  << scoreAdmissionLocalBAEdges
                  << " score_lba_inliers="
                  << scoreAdmissionLocalBAInliers
                  << " score_lba_fixed_edges="
                  << scoreAdmissionLocalBAFixedEdges
                  << " score_lba_local_edges="
                  << scoreAdmissionLocalBALocalEdges
                  << " score_lba_chi2_mean="
                  << (scoreAdmissionLocalBAEdges > 0 ?
                      scoreAdmissionLocalBAChi2Sum /
                          static_cast<double>(scoreAdmissionLocalBAEdges) :
                      0.0)
                  << " score_obs_ge2=" << scoreAdmissionObsGe2
                  << " score_obs_ge3=" << scoreAdmissionObsGe3
                  << " score_obs_sum=" << scoreAdmissionObsSum
                  << " score_ref_distance_mean="
                  << (scoreAdmissionRefDistanceCount > 0 ?
                      scoreAdmissionRefDistanceSum /
                          static_cast<double>(scoreAdmissionRefDistanceCount) :
                      0.0)
                  << " probation_min_age_kfs="
                  << DynamicMapAdmissionV7ProbationMinAgeKFs()
                  << " probation_min_pose_use="
                  << DynamicMapAdmissionV7ProbationMinPoseUse()
                  << " probation_min_inlier_rate="
                  << DynamicMapAdmissionV7ProbationMinInlierRate()
                  << " probation_max_mean_chi2="
                  << DynamicMapAdmissionV7ProbationMaxMeanChi2()
                  << " probation_low_use_age_kfs="
                  << DynamicMapAdmissionV7ProbationLowUseAgeKFs()
                  << std::endl;
    }
}


void LocalMapping::CreateNewMapPoints()
{
    const bool debugFocusFrame =
        mpCurrentKeyFrame && DebugFocusFrame(mpCurrentKeyFrame->mnFrameId);
    std::map<int, int> createdPointsPerInstance;
    std::map<int, int> mismatchedPairInstanceCounts;
    int totalCandidateMatches = 0;
    int triangulatedPoints = 0;
    int createdMapPoints = 0;
    int sameInstanceMatches = 0;
    int mismatchedInstanceMatches = 0;
    int createdInstanceBoundPoints = 0;
    int createdSemanticOnlyPoints = 0;
    int sameInstancePairsEnteringGeometry = 0;
    int sameInstancePairsAfterParallax = 0;
    int sameInstancePairsAfterTriangulation = 0;
    int sameInstancePairsAfterPositiveDepth = 0;
    int sameInstancePairsAfterReproj1 = 0;
    int sameInstancePairsAfterReproj2 = 0;
    int sameInstancePairsAfterScale = 0;
    const bool nearBoundaryDiagnostics = EnableNearBoundaryDiagnostics();
    int lmCreatedNearBoundaryPoints = 0;
    int lmCreatedCleanStaticPoints = 0;
    int lmCreatedDirectDynamicPoints = 0;
    const KeyFrameFeatureStats currentKFStats =
        debugFocusFrame ? CollectKeyFrameFeatureStats(mpCurrentKeyFrame) : KeyFrameFeatureStats();
    std::map<int, int> skippedShortBaselineNeighbors;
    const AdmissionStateAwarenessContext admissionStateContext =
        MakeAdmissionStateAwarenessContext(
            mpTracker,
            mpCurrentKeyFrame,
            mbAdmissionStateHasKfStepEwma,
            mdAdmissionStateKfStepEwma,
            mnAdmissionLastLBAEdges,
            mnAdmissionLastLBAMapPoints);
    const AdmissionCoverageContext admissionCoverageContext =
        MakeAdmissionCoverageContext(mpCurrentKeyFrame);
    const bool dynamicMapAdmissionCoverageAwareV7 =
        DynamicMapAdmissionCoverageAwareV7();
    int v7PromotedThisKeyFrame = 0;
    if(admissionStateContext.keyframeStep >= 0.0)
    {
        const double alpha = DynamicMapAdmissionStateKfStepEwmaAlpha();
        if(mbAdmissionStateHasKfStepEwma)
        {
            mdAdmissionStateKfStepEwma =
                (1.0 - alpha) * mdAdmissionStateKfStepEwma +
                alpha * admissionStateContext.keyframeStep;
        }
        else
        {
            mdAdmissionStateKfStepEwma = admissionStateContext.keyframeStep;
            mbAdmissionStateHasKfStepEwma = true;
        }
    }

    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    // For stereo inertial case
    if(mbMonocular)
        nn=30;
    vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    if (mbInertial)
    {
        KeyFrame* pKF = mpCurrentKeyFrame;
        int count=0;
        while((vpNeighKFs.size()<=nn)&&(pKF->mPrevKF)&&(count++<nn))
        {
            vector<KeyFrame*>::iterator it = std::find(vpNeighKFs.begin(), vpNeighKFs.end(), pKF->mPrevKF);
            if(it==vpNeighKFs.end())
                vpNeighKFs.push_back(pKF->mPrevKF);
            pKF = pKF->mPrevKF;
        }
    }

    float th = 0.6f;

    ORBmatcher matcher(th,false);

    Sophus::SE3<float> sophTcw1 = mpCurrentKeyFrame->GetPose();
    Eigen::Matrix<float,3,4> eigTcw1 = sophTcw1.matrix3x4();
    Eigen::Matrix<float,3,3> Rcw1 = eigTcw1.block<3,3>(0,0);
    Eigen::Matrix<float,3,3> Rwc1 = Rcw1.transpose();
    Eigen::Vector3f tcw1 = sophTcw1.translation();
    Eigen::Vector3f Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;
    int countStereo = 0;
    int countStereoGoodProj = 0;
    int countStereoAttempt = 0;
    int totalStereoPts = 0;
    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];
        const KeyFrameFeatureStats neighKFStats =
            debugFocusFrame ? CollectKeyFrameFeatureStats(pKF2) : KeyFrameFeatureStats();
        const int commonWords =
            debugFocusFrame ? CountCommonFeatureWords(mpCurrentKeyFrame->mFeatVec, pKF2->mFeatVec) : 0;
        const std::map<int, int> sharedUnmatchedInstances =
            debugFocusFrame ? CollectSharedUnmatchedInstanceSupport(currentKFStats, neighKFStats)
                            : std::map<int, int>();
        const int maxSharedInstanceSupport =
            debugFocusFrame ? MaxCountValue(sharedUnmatchedInstances) : 0;
        const int kfGap = std::abs(static_cast<int>(mpCurrentKeyFrame->mnId) - static_cast<int>(pKF2->mnId));

        GeometricCamera* pCamera1 = mpCurrentKeyFrame->mpCamera, *pCamera2 = pKF2->mpCamera;

        // Check first that baseline is not too short
        Eigen::Vector3f Ow2 = pKF2->GetCameraCenter();
        Eigen::Vector3f vBaseline = Ow2-Ow1;
        const float baseline = vBaseline.norm();

        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)
            {
                if(debugFocusFrame)
                    skippedShortBaselineNeighbors[pKF2->mnId] = -1;
                continue;
            }
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline/medianDepthKF2;
            float minRatioBaselineDepth = 0.01f;
            if(maxSharedInstanceSupport >= 20)
            {
                minRatioBaselineDepth = 0.003f;
                if(kfGap <= 2 && maxSharedInstanceSupport >= 40)
                    minRatioBaselineDepth = 0.001f;
            }

            if(ratioBaselineDepth<minRatioBaselineDepth)
            {
                if(debugFocusFrame)
                {
                    std::cout << "[STSLAM_FOCUS] frame=" << mpCurrentKeyFrame->mnFrameId
                              << " stage=create_new_map_points_skip"
                              << " current_kf=" << mpCurrentKeyFrame->mnId
                              << " neighbor_kf=" << pKF2->mnId
                              << " reason=short_baseline"
                              << " kf_gap=" << kfGap
                              << " baseline=" << baseline
                              << " median_depth=" << medianDepthKF2
                              << " ratio=" << ratioBaselineDepth
                              << " min_ratio=" << minRatioBaselineDepth
                              << " shared_top_instances=" << FormatTopCounts(sharedUnmatchedInstances)
                              << std::endl;
                }
                continue;
            }
        }

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        bool bCoarse = mbInertial && mpTracker->mState==Tracking::RECENTLY_LOST && mpCurrentKeyFrame->GetMap()->GetIniertialBA2();

        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,vMatchedIndices,false,bCoarse);
        const bool dynamicMapAdmissionVetoCreateNewMapPoints =
            DynamicMapAdmissionVetoCreateNewMapPoints();
        const bool dynamicMapAdmissionBoundaryVetoCreateNewMapPoints =
            DynamicMapAdmissionBoundaryVetoCreateNewMapPoints();
        const bool dynamicMapAdmissionBoundarySameCountControlCreateNewMapPoints =
            DynamicMapAdmissionBoundarySameCountControlCreateNewMapPoints();
        const bool dynamicMapAdmissionBoundaryMatchedControlCreateNewMapPoints =
            DynamicMapAdmissionBoundaryMatchedControlCreateNewMapPoints();
        const bool dynamicMapAdmissionDelayedBoundaryCreateNewMapPoints =
            DynamicMapAdmissionDelayedBoundaryCreateNewMapPoints();
        const bool dynamicMapAdmissionSupportQuality =
            DynamicMapAdmissionSupportQuality();
        const bool dynamicMapAdmissionScoreBased =
            DynamicMapAdmissionScoreBased() &&
            dynamicMapAdmissionSupportQuality &&
            dynamicMapAdmissionDelayedBoundaryCreateNewMapPoints;
        const bool dynamicMapAdmissionV5UsefulnessLog =
            DynamicMapAdmissionV5UsefulnessLog() &&
            dynamicMapAdmissionScoreBased;
        const bool dynamicMapAdmissionBoundaryGateCreateNewMapPoints =
            dynamicMapAdmissionBoundaryVetoCreateNewMapPoints ||
            dynamicMapAdmissionDelayedBoundaryCreateNewMapPoints;
        int supportPromotedBoundaryPairs = 0;
        int promotedGeomEnter = 0;
        int promotedGeomAfterParallax = 0;
        int promotedGeomAfterTriangulation = 0;
        int promotedGeomAfterDepth = 0;
        int promotedGeomAfterReproj1 = 0;
        int promotedGeomAfterReproj2 = 0;
        int promotedGeomAfterScale = 0;
        int promotedGeomCreated = 0;
        std::set<std::pair<size_t, size_t> > promotedBoundaryPairsForGeometry;
        std::map<std::pair<size_t, size_t>, AdmissionScoreCandidateInfo> scoreCandidateInfoByPair;
        int scoreSupportCandidateBoundaryPairs = 0;
        int scoreSupportAcceptedBoundaryPairs = 0;
        int scoreSupportRejectedBoundaryPairs = 0;
        int scoreCandidateGeomEvaluatedBoundaryPairs = 0;
        int scorePostGeomRejectedBoundaryPairs = 0;
        int scoreCreatedBoundaryPairs = 0;
        int stateAwareCandidateBoundaryPairs = 0;
        int stateAwareAllowedBoundaryPairs = 0;
        int stateAwareRejectedBoundaryPairs = 0;
        int v7CoverageCandidateBoundaryPairs = 0;
        int v7CoverageAllowedBoundaryPairs = 0;
        int v7CoverageRejectedBoundaryPairs = 0;
        int v7CoverageStateAllowedBoundaryPairs = 0;
        int v7CoverageGapAllowedBoundaryPairs = 0;
        int v7CoverageQuotaRejectedBoundaryPairs = 0;
        int v7PromotedThisNeighbor = 0;
        if((ForceFilterDetectedDynamicFeatures() ||
            dynamicMapAdmissionVetoCreateNewMapPoints ||
            dynamicMapAdmissionBoundaryGateCreateNewMapPoints ||
            dynamicMapAdmissionBoundarySameCountControlCreateNewMapPoints ||
            dynamicMapAdmissionBoundaryMatchedControlCreateNewMapPoints ||
            SplitDetectedDynamicFeaturesFromStaticMapping()) &&
           !vMatchedIndices.empty())
        {
            vector<pair<size_t,size_t> > vStaticMatchedIndices;
            vStaticMatchedIndices.reserve(vMatchedIndices.size());
            int skippedInstancePairs = 0;
            int skippedBoundaryPairs = 0;
            int delayedBoundaryRejectedPairs = 0;
            int boundarySupportSum = 0;
            int qualityRejectedBoundaryPairs = 0;
            int qualityRawSupportSum = 0;
            int qualityFoundSupportSum = 0;
            int qualityFrameSupportSum = 0;
            int qualityRawDepthSupportSum = 0;
            int qualityReliableSupportSum = 0;
            int qualityResidualSupportSum = 0;
            int qualityDepthSupportSum = 0;
            int boundarySameCountBudget = 0;
            int skippedNonBoundaryControlPairs = 0;
            int boundaryMatchedControlBudget = 0;
            int skippedMatchedNonBoundaryControlPairs = 0;
            int exactSkippedMatchedNonBoundaryControlPairs = 0;
            int fallbackSkippedMatchedNonBoundaryControlPairs = 0;
            std::map<AdmissionMatchedControlBin, int> boundaryMatchedExactBudget;
            std::map<AdmissionMatchedControlBin, int> boundaryMatchedFallbackBudget;
            if(dynamicMapAdmissionBoundarySameCountControlCreateNewMapPoints)
            {
                for(size_t pairIdx = 0; pairIdx < vMatchedIndices.size(); ++pairIdx)
                {
                    const int idx1 = static_cast<int>(vMatchedIndices[pairIdx].first);
                    const int idx2 = static_cast<int>(vMatchedIndices[pairIdx].second);
                    const int instanceId1 = mpCurrentKeyFrame->GetFeatureInstanceId(idx1);
                    const int instanceId2 = pKF2->GetFeatureInstanceId(idx2);
                    const bool shouldSkipInstancePair =
                        (ForceFilterDetectedDynamicFeatures() ||
                         dynamicMapAdmissionVetoCreateNewMapPoints) ?
                        (instanceId1 > 0 || instanceId2 > 0) :
                        (ShouldDetachRgbdInstanceFromStaticPath(mpCurrentKeyFrame->GetMap(), instanceId1) ||
                         ShouldDetachRgbdInstanceFromStaticPath(mpCurrentKeyFrame->GetMap(), instanceId2));
                    if(shouldSkipInstancePair)
                        continue;

                    if(mpCurrentKeyFrame->IsFeatureStaticNearDynamicMask(idx1) ||
                       pKF2->IsFeatureStaticNearDynamicMask(idx2))
                        ++boundarySameCountBudget;
                }
            }
            if(dynamicMapAdmissionBoundaryMatchedControlCreateNewMapPoints)
            {
                for(size_t pairIdx = 0; pairIdx < vMatchedIndices.size(); ++pairIdx)
                {
                    const int idx1 = static_cast<int>(vMatchedIndices[pairIdx].first);
                    const int idx2 = static_cast<int>(vMatchedIndices[pairIdx].second);
                    const int instanceId1 = mpCurrentKeyFrame->GetFeatureInstanceId(idx1);
                    const int instanceId2 = pKF2->GetFeatureInstanceId(idx2);
                    const bool shouldSkipInstancePair =
                        (ForceFilterDetectedDynamicFeatures() ||
                         dynamicMapAdmissionVetoCreateNewMapPoints) ?
                        (instanceId1 > 0 || instanceId2 > 0) :
                        (ShouldDetachRgbdInstanceFromStaticPath(mpCurrentKeyFrame->GetMap(), instanceId1) ||
                         ShouldDetachRgbdInstanceFromStaticPath(mpCurrentKeyFrame->GetMap(), instanceId2));
                    if(shouldSkipInstancePair)
                        continue;

                    const bool isBoundaryRiskPair =
                        mpCurrentKeyFrame->IsFeatureStaticNearDynamicMask(idx1) ||
                        pKF2->IsFeatureStaticNearDynamicMask(idx2);
                    if(isBoundaryRiskPair)
                    {
                        ++boundaryMatchedControlBudget;
                        const float pairDepth =
                            PairMatchedControlDepth(mpCurrentKeyFrame, idx1, pKF2, idx2);
                        AddAdmissionMatchedControlBudget(
                            boundaryMatchedExactBudget,
                            boundaryMatchedFallbackBudget,
                            MakeAdmissionMatchedControlBin(
                                mpCurrentKeyFrame, idx1, pairDepth, true),
                            MakeAdmissionMatchedControlBin(
                                mpCurrentKeyFrame, idx1, pairDepth, false));
                    }
                }
            }
            for(size_t pairIdx = 0; pairIdx < vMatchedIndices.size(); ++pairIdx)
            {
                const int idx1 = static_cast<int>(vMatchedIndices[pairIdx].first);
                const int idx2 = static_cast<int>(vMatchedIndices[pairIdx].second);
                const int instanceId1 = mpCurrentKeyFrame->GetFeatureInstanceId(idx1);
                const int instanceId2 = pKF2->GetFeatureInstanceId(idx2);
                const bool shouldSkipInstancePair =
                    (ForceFilterDetectedDynamicFeatures() ||
                     dynamicMapAdmissionVetoCreateNewMapPoints) ?
                    (instanceId1 > 0 || instanceId2 > 0) :
                    (ShouldDetachRgbdInstanceFromStaticPath(mpCurrentKeyFrame->GetMap(), instanceId1) ||
                     ShouldDetachRgbdInstanceFromStaticPath(mpCurrentKeyFrame->GetMap(), instanceId2));
                if(shouldSkipInstancePair)
                {
                    ++skippedInstancePairs;
                    continue;
                }
                const bool isBoundaryRiskCurrent =
                    mpCurrentKeyFrame->IsFeatureStaticNearDynamicMask(idx1);
                const bool isBoundaryRiskNeighbor =
                    pKF2->IsFeatureStaticNearDynamicMask(idx2);
                const bool isBoundaryRiskPair =
                    isBoundaryRiskCurrent || isBoundaryRiskNeighbor;
                if(dynamicMapAdmissionBoundaryGateCreateNewMapPoints &&
                   isBoundaryRiskPair)
                {
                    int support1 = 0;
                    int support2 = 0;
                    bool supportOk = false;
                    AdmissionScoreCandidateInfo scoreInfo;
                    if(dynamicMapAdmissionDelayedBoundaryCreateNewMapPoints)
                    {
                        if(isBoundaryRiskCurrent)
                        {
                            if(dynamicMapAdmissionSupportQuality)
                            {
                                const AdmissionSupportQualityResult quality1 =
                                    EvaluateCleanStaticMapSupportQualityNearFeature(
                                        mpCurrentKeyFrame,
                                        idx1,
                                        DynamicMapAdmissionDelayedBoundarySupportRadiusPx(),
                                        DynamicMapAdmissionDelayedBoundaryMinObservations(),
                                        idx1 < static_cast<int>(mpCurrentKeyFrame->mvDepth.size()) ?
                                        mpCurrentKeyFrame->mvDepth[idx1] :
                                        -1.0f);
                                support1 = quality1.rawSupport;
                                qualityRawSupportSum += quality1.rawSupport;
                                qualityFoundSupportSum +=
                                    quality1.foundStableSupport;
                                qualityFrameSupportSum +=
                                    quality1.frameStableSupport;
                                qualityRawDepthSupportSum +=
                                    quality1.rawDepthConsistentSupport;
                                qualityReliableSupportSum += quality1.reliableSupport;
                                qualityResidualSupportSum +=
                                    quality1.residualReliableSupport;
                                qualityDepthSupportSum +=
                                    quality1.depthConsistentSupport;
                                if(dynamicMapAdmissionScoreBased)
                                    AccumulateAdmissionScoreSupport(scoreInfo, quality1);
                                supportOk = quality1.pass;
                            }
                            else
                            {
                                support1 = CountCleanStaticMapSupportNearFeature(
                                    mpCurrentKeyFrame,
                                    idx1,
                                    DynamicMapAdmissionDelayedBoundarySupportRadiusPx(),
                                    DynamicMapAdmissionDelayedBoundaryMinObservations());
                            }
                        }
                        if(isBoundaryRiskNeighbor)
                        {
                            if(dynamicMapAdmissionSupportQuality)
                            {
                                const AdmissionSupportQualityResult quality2 =
                                    EvaluateCleanStaticMapSupportQualityNearFeature(
                                        pKF2,
                                        idx2,
                                        DynamicMapAdmissionDelayedBoundarySupportRadiusPx(),
                                        DynamicMapAdmissionDelayedBoundaryMinObservations(),
                                        idx2 < static_cast<int>(pKF2->mvDepth.size()) ?
                                        pKF2->mvDepth[idx2] :
                                        -1.0f);
                                support2 = quality2.rawSupport;
                                qualityRawSupportSum += quality2.rawSupport;
                                qualityFoundSupportSum +=
                                    quality2.foundStableSupport;
                                qualityFrameSupportSum +=
                                    quality2.frameStableSupport;
                                qualityRawDepthSupportSum +=
                                    quality2.rawDepthConsistentSupport;
                                qualityReliableSupportSum += quality2.reliableSupport;
                                qualityResidualSupportSum +=
                                    quality2.residualReliableSupport;
                                qualityDepthSupportSum +=
                                    quality2.depthConsistentSupport;
                                if(dynamicMapAdmissionScoreBased)
                                    AccumulateAdmissionScoreSupport(scoreInfo, quality2);
                                supportOk = (!isBoundaryRiskCurrent || supportOk) &&
                                            quality2.pass;
                            }
                            else
                            {
                                support2 = CountCleanStaticMapSupportNearFeature(
                                    pKF2,
                                    idx2,
                                    DynamicMapAdmissionDelayedBoundarySupportRadiusPx(),
                                    DynamicMapAdmissionDelayedBoundaryMinObservations());
                            }
                        }

                        if(!dynamicMapAdmissionSupportQuality)
                        {
                            const bool currentOk =
                                !isBoundaryRiskCurrent ||
                                support1 >= DynamicMapAdmissionDelayedBoundaryMinSupport();
                            const bool neighborOk =
                                !isBoundaryRiskNeighbor ||
                                support2 >= DynamicMapAdmissionDelayedBoundaryMinSupport();
                            supportOk = currentOk && neighborOk;
                        }
                        boundarySupportSum += support1 + support2;
                    }

                    if(dynamicMapAdmissionScoreBased)
                    {
                        ++scoreSupportCandidateBoundaryPairs;
                        scoreInfo.supportScore =
                            ComputeAdmissionSupportScore(scoreInfo);
                        supportOk =
                            AdmissionScoreSupportAllowsGeometry(scoreInfo);
                        if(dynamicMapAdmissionCoverageAwareV7 && supportOk)
                        {
                            ++v7CoverageCandidateBoundaryPairs;
                            const bool stateNeed =
                                admissionStateContext.needScore >=
                                DynamicMapAdmissionV7MinNeedScore();
                            const bool coverageNeed =
                                admissionCoverageContext.coveragePressure;
                            bool quotaOk = true;
                            const int maxPromotionsPerKF =
                                DynamicMapAdmissionV7MaxPromotionsPerKeyFrame();
                            const int maxPromotionsPerNeighbor =
                                DynamicMapAdmissionV7MaxPromotionsPerNeighbor();
                            if(maxPromotionsPerKF > 0 &&
                               v7PromotedThisKeyFrame >= maxPromotionsPerKF)
                                quotaOk = false;
                            if(maxPromotionsPerNeighbor > 0 &&
                               v7PromotedThisNeighbor >= maxPromotionsPerNeighbor)
                                quotaOk = false;

                            if((stateNeed || coverageNeed) && quotaOk)
                            {
                                ++v7CoverageAllowedBoundaryPairs;
                                if(stateNeed)
                                    ++v7CoverageStateAllowedBoundaryPairs;
                                if(coverageNeed)
                                    ++v7CoverageGapAllowedBoundaryPairs;
                                ++v7PromotedThisKeyFrame;
                                ++v7PromotedThisNeighbor;
                            }
                            else
                            {
                                ++v7CoverageRejectedBoundaryPairs;
                                if((stateNeed || coverageNeed) && !quotaOk)
                                    ++v7CoverageQuotaRejectedBoundaryPairs;
                                supportOk = false;
                            }
                        }
                        else if(DynamicMapAdmissionStateAware() && supportOk)
                        {
                            ++stateAwareCandidateBoundaryPairs;
                            if(admissionStateContext.allowAdmission)
                                ++stateAwareAllowedBoundaryPairs;
                            else
                            {
                                ++stateAwareRejectedBoundaryPairs;
                                supportOk = false;
                            }
                        }
                        if(dynamicMapAdmissionV5UsefulnessLog)
                        {
                            PrintAdmissionV5CandidateSupportEvent(
                                mpCurrentKeyFrame,
                                pKF2,
                                idx1,
                                idx2,
                                isBoundaryRiskCurrent,
                                isBoundaryRiskNeighbor,
                                supportOk,
                                scoreInfo);
                        }
                    }

                    if(dynamicMapAdmissionDelayedBoundaryCreateNewMapPoints &&
                       supportOk)
                    {
                        ++supportPromotedBoundaryPairs;
                        promotedBoundaryPairsForGeometry.insert(vMatchedIndices[pairIdx]);
                        if(dynamicMapAdmissionScoreBased)
                        {
                            ++scoreSupportAcceptedBoundaryPairs;
                            scoreCandidateInfoByPair[vMatchedIndices[pairIdx]] = scoreInfo;
                        }
                    }
                    else
                    {
                        ++skippedBoundaryPairs;
                        if(dynamicMapAdmissionScoreBased)
                            ++scoreSupportRejectedBoundaryPairs;
                        if(dynamicMapAdmissionDelayedBoundaryCreateNewMapPoints)
                        {
                            ++delayedBoundaryRejectedPairs;
                            if(dynamicMapAdmissionSupportQuality)
                                ++qualityRejectedBoundaryPairs;
                        }
                        continue;
                    }
                }
                if(dynamicMapAdmissionBoundarySameCountControlCreateNewMapPoints &&
                   !isBoundaryRiskPair &&
                   skippedNonBoundaryControlPairs < boundarySameCountBudget)
                {
                    ++skippedNonBoundaryControlPairs;
                    continue;
                }
                if(dynamicMapAdmissionBoundaryMatchedControlCreateNewMapPoints &&
                   !isBoundaryRiskPair)
                {
                    bool usedExact = false;
                    const float pairDepth =
                        PairMatchedControlDepth(mpCurrentKeyFrame, idx1, pKF2, idx2);
                    if(ConsumeAdmissionMatchedControlBudget(
                           boundaryMatchedExactBudget,
                           boundaryMatchedFallbackBudget,
                           MakeAdmissionMatchedControlBin(
                               mpCurrentKeyFrame, idx1, pairDepth, true),
                           MakeAdmissionMatchedControlBin(
                               mpCurrentKeyFrame, idx1, pairDepth, false),
                           usedExact))
                    {
                        ++skippedMatchedNonBoundaryControlPairs;
                        if(usedExact)
                            ++exactSkippedMatchedNonBoundaryControlPairs;
                        else
                            ++fallbackSkippedMatchedNonBoundaryControlPairs;
                        continue;
                    }
                }
                vStaticMatchedIndices.push_back(vMatchedIndices[pairIdx]);
            }

            if(skippedInstancePairs > 0)
            {
                std::cout << (dynamicMapAdmissionVetoCreateNewMapPoints ?
                              "[STSLAM_DYNAMIC_MAP_ADMISSION_VETO]" :
                              "[STSLAM_STATIC_MAPPING_DYNAMIC_SPLIT]")
                          << " frame=" << mpCurrentKeyFrame->mnFrameId
                          << " stage=create_new_map_points"
                          << " current_kf=" << mpCurrentKeyFrame->mnId
                          << " neighbor_kf=" << pKF2->mnId
                          << " skipped_instance_pairs=" << skippedInstancePairs
                          << " kept_static_pairs=" << vStaticMatchedIndices.size()
                          << std::endl;
            }
            if(skippedBoundaryPairs > 0)
            {
                std::cout << "[STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_VETO]"
                          << " frame=" << mpCurrentKeyFrame->mnFrameId
                          << " stage=create_new_map_points"
                          << " current_kf=" << mpCurrentKeyFrame->mnId
                          << " neighbor_kf=" << pKF2->mnId
                          << " radius_px=" << DynamicMapAdmissionBoundaryRadiusPx()
                          << " skipped_boundary_pairs=" << skippedBoundaryPairs
                          << " kept_pairs=" << vStaticMatchedIndices.size()
                          << std::endl;
            }
            if(dynamicMapAdmissionDelayedBoundaryCreateNewMapPoints)
            {
                std::cout << "[STSLAM_DYNAMIC_MAP_ADMISSION_DELAYED_BOUNDARY]"
                          << " frame=" << mpCurrentKeyFrame->mnFrameId
                          << " stage=create_new_map_points"
                          << " current_kf=" << mpCurrentKeyFrame->mnId
                          << " neighbor_kf=" << pKF2->mnId
                          << " radius_px=" << DynamicMapAdmissionBoundaryRadiusPx()
                          << " support_radius_px="
                          << DynamicMapAdmissionDelayedBoundarySupportRadiusPx()
                          << " min_support="
                          << DynamicMapAdmissionDelayedBoundaryMinSupport()
                          << " min_obs="
                          << DynamicMapAdmissionDelayedBoundaryMinObservations()
                          << " support_quality="
                          << (dynamicMapAdmissionSupportQuality ? 1 : 0)
                          << " score_based="
                          << (dynamicMapAdmissionScoreBased ? 1 : 0)
                          << " quality_min_reliable_support="
                          << DynamicMapAdmissionSupportQualityMinReliableSupport()
                          << " quality_min_depth_support="
                          << DynamicMapAdmissionSupportQualityMinDepthSupport()
                          << " quality_min_residual_obs="
                          << DynamicMapAdmissionSupportQualityMinResidualObs()
                          << " quality_min_inlier_rate="
                          << DynamicMapAdmissionSupportQualityMinInlierRate()
                          << " quality_max_mean_chi2="
                          << DynamicMapAdmissionSupportQualityMaxMeanChi2()
                          << " quality_min_found_ratio="
                          << DynamicMapAdmissionSupportQualityMinFoundRatio()
                          << " quality_min_frame_span="
                          << DynamicMapAdmissionSupportQualityMinFrameSpan()
                          << " quality_max_depth_rel_diff="
                          << DynamicMapAdmissionSupportQualityMaxDepthRelDiff()
                          << " delayed_rejected_boundary_pairs="
                          << delayedBoundaryRejectedPairs
                          << " support_promoted_boundary_pairs="
                          << supportPromotedBoundaryPairs
                          << " support_sum=" << boundarySupportSum
                          << " quality_rejected_boundary_pairs="
                          << qualityRejectedBoundaryPairs
                          << " quality_raw_support_sum="
                          << qualityRawSupportSum
                          << " quality_found_support_sum="
                          << qualityFoundSupportSum
                          << " quality_frame_support_sum="
                          << qualityFrameSupportSum
                          << " quality_raw_depth_support_sum="
                          << qualityRawDepthSupportSum
                          << " quality_reliable_support_sum="
                          << qualityReliableSupportSum
                          << " quality_residual_support_sum="
                          << qualityResidualSupportSum
                          << " quality_depth_support_sum="
                          << qualityDepthSupportSum
                          << " kept_pairs=" << vStaticMatchedIndices.size()
                          << std::endl;
            }
            if(dynamicMapAdmissionBoundarySameCountControlCreateNewMapPoints)
            {
                std::cout << "[STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_CONTROL]"
                          << " frame=" << mpCurrentKeyFrame->mnFrameId
                          << " stage=create_new_map_points"
                          << " current_kf=" << mpCurrentKeyFrame->mnId
                          << " neighbor_kf=" << pKF2->mnId
                          << " radius_px=" << DynamicMapAdmissionBoundaryRadiusPx()
                          << " boundary_budget=" << boundarySameCountBudget
                          << " skipped_nonboundary_pairs="
                          << skippedNonBoundaryControlPairs
                          << " kept_pairs=" << vStaticMatchedIndices.size()
                          << std::endl;
            }
            if(dynamicMapAdmissionBoundaryMatchedControlCreateNewMapPoints)
            {
                std::cout << "[STSLAM_DYNAMIC_MAP_ADMISSION_BOUNDARY_MATCHED_CONTROL]"
                          << " frame=" << mpCurrentKeyFrame->mnFrameId
                          << " stage=create_new_map_points"
                          << " current_kf=" << mpCurrentKeyFrame->mnId
                          << " neighbor_kf=" << pKF2->mnId
                          << " radius_px=" << DynamicMapAdmissionBoundaryRadiusPx()
                          << " boundary_budget=" << boundaryMatchedControlBudget
                          << " skipped_matched_nonboundary_pairs="
                          << skippedMatchedNonBoundaryControlPairs
                          << " exact_skipped_pairs="
                          << exactSkippedMatchedNonBoundaryControlPairs
                          << " fallback_skipped_pairs="
                          << fallbackSkippedMatchedNonBoundaryControlPairs
                          << " kept_pairs=" << vStaticMatchedIndices.size()
                          << std::endl;
            }
            vMatchedIndices.swap(vStaticMatchedIndices);
        }

        int rawSameInstancePairs = 0;
        int rawMismatchedInstancePairs = 0;
        int rawNoInstancePairs = 0;
        std::map<int, int> rawSamePerInstance;
        std::map<int, int> rawMismatchedPerInstance;
        if(debugFocusFrame)
        {
            for(size_t pairIdx = 0; pairIdx < vMatchedIndices.size(); ++pairIdx)
            {
                const int idx1 = static_cast<int>(vMatchedIndices[pairIdx].first);
                const int idx2 = static_cast<int>(vMatchedIndices[pairIdx].second);
                const int instanceId1 = mpCurrentKeyFrame->GetFeatureInstanceId(idx1);
                const int instanceId2 = pKF2->GetFeatureInstanceId(idx2);

                if(instanceId1 > 0 && instanceId1 == instanceId2)
                {
                    ++rawSameInstancePairs;
                    ++rawSamePerInstance[instanceId1];
                }
                else if(instanceId1 > 0 || instanceId2 > 0)
                {
                    ++rawMismatchedInstancePairs;
                    if(instanceId1 > 0)
                        ++rawMismatchedPerInstance[instanceId1];
                    if(instanceId2 > 0)
                        ++rawMismatchedPerInstance[instanceId2];
                }
                else
                {
                    ++rawNoInstancePairs;
                }
            }
        }

        if(debugFocusFrame)
        {
            std::cout << "[STSLAM_FOCUS] frame=" << mpCurrentKeyFrame->mnFrameId
                      << " stage=create_new_map_points_pair"
                      << " current_kf=" << mpCurrentKeyFrame->mnId
                      << " neighbor_kf=" << pKF2->mnId
                      << " common_words=" << commonWords
                      << " current_unmatched=" << currentKFStats.unmatchedFeatures << "/" << currentKFStats.totalFeatures
                      << " current_instance_unmatched=" << currentKFStats.unmatchedInstanceFeatures << "/" << currentKFStats.totalInstanceFeatures
                      << " current_top_unmatched_instances=" << FormatTopCounts(currentKFStats.unmatchedPerInstance)
                      << " neighbor_unmatched=" << neighKFStats.unmatchedFeatures << "/" << neighKFStats.totalFeatures
                      << " neighbor_instance_unmatched=" << neighKFStats.unmatchedInstanceFeatures << "/" << neighKFStats.totalInstanceFeatures
                      << " neighbor_top_unmatched_instances=" << FormatTopCounts(neighKFStats.unmatchedPerInstance)
                      << " shared_top_instances=" << FormatTopCounts(sharedUnmatchedInstances)
                      << " raw_same_instance_pairs=" << rawSameInstancePairs
                      << " raw_same_top_instances=" << FormatTopCounts(rawSamePerInstance)
                      << " raw_mismatched_instance_pairs=" << rawMismatchedInstancePairs
                      << " raw_mismatched_top_instances=" << FormatTopCounts(rawMismatchedPerInstance)
                      << " raw_no_instance_pairs=" << rawNoInstancePairs
                      << " triangulation_candidates=" << vMatchedIndices.size()
                      << std::endl;
        }

        Sophus::SE3<float> sophTcw2 = pKF2->GetPose();
        Eigen::Matrix<float,3,4> eigTcw2 = sophTcw2.matrix3x4();
        Eigen::Matrix<float,3,3> Rcw2 = eigTcw2.block<3,3>(0,0);
        Eigen::Matrix<float,3,3> Rwc2 = Rcw2.transpose();
        Eigen::Vector3f tcw2 = sophTcw2.translation();

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        totalCandidateMatches += nmatches;
        int pairSameInstanceEnter = 0;
        int pairSameInstanceAfterParallax = 0;
        int pairSameInstanceAfterTriangulation = 0;
        int pairSameInstanceAfterDepth = 0;
        int pairSameInstanceAfterReproj1 = 0;
        int pairSameInstanceAfterReproj2 = 0;
        int pairSameInstanceAfterScale = 0;
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const std::pair<size_t, size_t>& matchedPair = vMatchedIndices[ikp];
            const int &idx1 = matchedPair.first;
            const int &idx2 = matchedPair.second;
            const bool promotedBoundaryCandidate =
                promotedBoundaryPairsForGeometry.find(matchedPair) !=
                promotedBoundaryPairsForGeometry.end();
            const std::map<std::pair<size_t, size_t>, AdmissionScoreCandidateInfo>::const_iterator scoreInfoIt =
                scoreCandidateInfoByPair.find(matchedPair);
            const bool scoreBasedBoundaryCandidate =
                dynamicMapAdmissionScoreBased &&
                scoreInfoIt != scoreCandidateInfoByPair.end();
            if(promotedBoundaryCandidate)
                ++promotedGeomEnter;
            double parallaxScore = 0.0;
            double reprojRatio1 = 1.0;
            double reprojRatio2 = 1.0;
            double scaleScore = 0.0;
            double finalCandidateScore = 0.0;
            double finalTotalScore = 0.0;
            const int instanceIdCandidate1 = mpCurrentKeyFrame->GetFeatureInstanceId(idx1);
            const int instanceIdCandidate2 = pKF2->GetFeatureInstanceId(idx2);
            const bool sameInstanceCandidate =
                instanceIdCandidate1 > 0 && instanceIdCandidate1 == instanceIdCandidate2;
            if(sameInstanceCandidate)
            {
                ++sameInstancePairsEnteringGeometry;
                ++pairSameInstanceEnter;
            }

            const cv::KeyPoint &kp1 = (mpCurrentKeyFrame -> NLeft == -1) ? mpCurrentKeyFrame->mvKeysUn[idx1]
                                                                         : (idx1 < mpCurrentKeyFrame -> NLeft) ? mpCurrentKeyFrame -> mvKeys[idx1]
                                                                                                               : mpCurrentKeyFrame -> mvKeysRight[idx1 - mpCurrentKeyFrame -> NLeft];
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = (!mpCurrentKeyFrame->mpCamera2 && kp1_ur>=0);
            const bool bRight1 = (mpCurrentKeyFrame -> NLeft == -1 || idx1 < mpCurrentKeyFrame -> NLeft) ? false
                                                                                                         : true;

            const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                            : (idx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[idx2]
                                                                                     : pKF2 -> mvKeysRight[idx2 - pKF2 -> NLeft];

            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = (!pKF2->mpCamera2 && kp2_ur>=0);
            const bool bRight2 = (pKF2 -> NLeft == -1 || idx2 < pKF2 -> NLeft) ? false
                                                                               : true;

            if(mpCurrentKeyFrame->mpCamera2 && pKF2->mpCamera2){
                if(bRight1 && bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetRightPose();
                    Ow1 = mpCurrentKeyFrame->GetRightCameraCenter();

                    sophTcw2 = pKF2->GetRightPose();
                    Ow2 = pKF2->GetRightCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera2;
                    pCamera2 = pKF2->mpCamera2;
                }
                else if(bRight1 && !bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetRightPose();
                    Ow1 = mpCurrentKeyFrame->GetRightCameraCenter();

                    sophTcw2 = pKF2->GetPose();
                    Ow2 = pKF2->GetCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera2;
                    pCamera2 = pKF2->mpCamera;
                }
                else if(!bRight1 && bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetPose();
                    Ow1 = mpCurrentKeyFrame->GetCameraCenter();

                    sophTcw2 = pKF2->GetRightPose();
                    Ow2 = pKF2->GetRightCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera;
                    pCamera2 = pKF2->mpCamera2;
                }
                else{
                    sophTcw1 = mpCurrentKeyFrame->GetPose();
                    Ow1 = mpCurrentKeyFrame->GetCameraCenter();

                    sophTcw2 = pKF2->GetPose();
                    Ow2 = pKF2->GetCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera;
                    pCamera2 = pKF2->mpCamera;
                }
                eigTcw1 = sophTcw1.matrix3x4();
                Rcw1 = eigTcw1.block<3,3>(0,0);
                Rwc1 = Rcw1.transpose();
                tcw1 = sophTcw1.translation();

                eigTcw2 = sophTcw2.matrix3x4();
                Rcw2 = eigTcw2.block<3,3>(0,0);
                Rwc2 = Rcw2.transpose();
                tcw2 = sophTcw2.translation();
            }

            // Check parallax between rays
            Eigen::Vector3f xn1 = pCamera1->unprojectEig(kp1.pt);
            Eigen::Vector3f xn2 = pCamera2->unprojectEig(kp2.pt);

            Eigen::Vector3f ray1 = Rwc1 * xn1;
            Eigen::Vector3f ray2 = Rwc2 * xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(ray1.norm() * ray2.norm());

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            if (bStereo1 || bStereo2) totalStereoPts++;
            
            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            Eigen::Vector3f x3D;

            bool goodProj = false;
            bool bPointStereo = false;
            const float defaultParallaxCosThreshold = mbInertial ? 0.9996f : 0.9998f;
            const bool allowRelaxedSameInstanceParallax =
                sameInstanceCandidate && mbMonocular && kfGap <= 2 && maxSharedInstanceSupport >= 40;
            const float parallaxCosThreshold =
                allowRelaxedSameInstanceParallax ? GetSameInstanceParallaxCosThreshold()
                                                 : defaultParallaxCosThreshold;

            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 ||
                                                                          (cosParallaxRays<parallaxCosThreshold)))
            {
                if(sameInstanceCandidate)
                {
                    ++sameInstancePairsAfterParallax;
                    ++pairSameInstanceAfterParallax;
                }
                if(promotedBoundaryCandidate)
                    ++promotedGeomAfterParallax;
                parallaxScore = Clamp01(
                    (static_cast<double>(parallaxCosThreshold) -
                     static_cast<double>(cosParallaxRays)) /
                    std::max(1e-9,
                             1.0 - static_cast<double>(parallaxCosThreshold)));
                goodProj = GeometricTools::Triangulate(xn1, xn2, eigTcw1, eigTcw2, x3D);
                if(!goodProj)
                {
                    if(dynamicMapAdmissionV5UsefulnessLog &&
                       scoreBasedBoundaryCandidate)
                    {
                        PrintAdmissionV5CandidateGeometryEvent(
                            mpCurrentKeyFrame,
                            pKF2,
                            idx1,
                            idx2,
                            "reject_triangulate",
                            scoreInfoIt->second,
                            parallaxScore,
                            reprojRatio1,
                            reprojRatio2,
                            scaleScore,
                            finalCandidateScore,
                            finalTotalScore);
                    }
                    continue;
                }
            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                countStereoAttempt++;
                bPointStereo = true;
                if(sameInstanceCandidate)
                {
                    ++sameInstancePairsAfterParallax;
                    ++pairSameInstanceAfterParallax;
                }
                if(promotedBoundaryCandidate)
                    ++promotedGeomAfterParallax;
                parallaxScore = 1.0;
                goodProj = mpCurrentKeyFrame->UnprojectStereo(idx1, x3D);
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                countStereoAttempt++;
                bPointStereo = true;
                if(sameInstanceCandidate)
                {
                    ++sameInstancePairsAfterParallax;
                    ++pairSameInstanceAfterParallax;
                }
                if(promotedBoundaryCandidate)
                    ++promotedGeomAfterParallax;
                parallaxScore = 1.0;
                goodProj = pKF2->UnprojectStereo(idx2, x3D);
            }
            else
            {
                if(dynamicMapAdmissionV5UsefulnessLog &&
                   scoreBasedBoundaryCandidate)
                {
                    PrintAdmissionV5CandidateGeometryEvent(
                        mpCurrentKeyFrame,
                        pKF2,
                        idx1,
                        idx2,
                        "reject_parallax",
                        scoreInfoIt->second,
                        parallaxScore,
                        reprojRatio1,
                        reprojRatio2,
                        scaleScore,
                        finalCandidateScore,
                        finalTotalScore);
                }
                continue; //No stereo and very low parallax
            }

            if(goodProj && bPointStereo)
                countStereoGoodProj++;

            if(!goodProj)
            {
                if(dynamicMapAdmissionV5UsefulnessLog &&
                   scoreBasedBoundaryCandidate)
                {
                    PrintAdmissionV5CandidateGeometryEvent(
                        mpCurrentKeyFrame,
                        pKF2,
                        idx1,
                        idx2,
                        "reject_triangulate",
                        scoreInfoIt->second,
                        parallaxScore,
                        reprojRatio1,
                        reprojRatio2,
                        scaleScore,
                        finalCandidateScore,
                        finalTotalScore);
                }
                continue;
            }
            if(sameInstanceCandidate)
            {
                ++sameInstancePairsAfterTriangulation;
                ++pairSameInstanceAfterTriangulation;
            }
            if(promotedBoundaryCandidate)
                ++promotedGeomAfterTriangulation;

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3D) + tcw1(2);
            if(z1<=0)
            {
                if(dynamicMapAdmissionV5UsefulnessLog &&
                   scoreBasedBoundaryCandidate)
                {
                    PrintAdmissionV5CandidateGeometryEvent(
                        mpCurrentKeyFrame,
                        pKF2,
                        idx1,
                        idx2,
                        "reject_depth",
                        scoreInfoIt->second,
                        parallaxScore,
                        reprojRatio1,
                        reprojRatio2,
                        scaleScore,
                        finalCandidateScore,
                        finalTotalScore);
                }
                continue;
            }

            float z2 = Rcw2.row(2).dot(x3D) + tcw2(2);
            if(z2<=0)
            {
                if(dynamicMapAdmissionV5UsefulnessLog &&
                   scoreBasedBoundaryCandidate)
                {
                    PrintAdmissionV5CandidateGeometryEvent(
                        mpCurrentKeyFrame,
                        pKF2,
                        idx1,
                        idx2,
                        "reject_depth",
                        scoreInfoIt->second,
                        parallaxScore,
                        reprojRatio1,
                        reprojRatio2,
                        scaleScore,
                        finalCandidateScore,
                        finalTotalScore);
                }
                continue;
            }
            if(sameInstanceCandidate)
            {
                ++sameInstancePairsAfterPositiveDepth;
                ++pairSameInstanceAfterDepth;
            }
            if(promotedBoundaryCandidate)
                ++promotedGeomAfterDepth;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3D)+tcw1(0);
            const float y1 = Rcw1.row(1).dot(x3D)+tcw1(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                cv::Point2f uv1 = pCamera1->project(cv::Point3f(x1,y1,z1));
                float errX1 = uv1.x - kp1.pt.x;
                float errY1 = uv1.y - kp1.pt.y;

                const double reprojError1 =
                    static_cast<double>(errX1 * errX1 + errY1 * errY1);
                const double reprojLimit1 =
                    static_cast<double>(5.991 * sigmaSquare1);
                reprojRatio1 =
                    reprojLimit1 > 0.0 ? reprojError1 / reprojLimit1 : 1.0;
                if(reprojRatio1 > 1.0)
                {
                    if(dynamicMapAdmissionV5UsefulnessLog &&
                       scoreBasedBoundaryCandidate)
                    {
                        PrintAdmissionV5CandidateGeometryEvent(
                            mpCurrentKeyFrame,
                            pKF2,
                            idx1,
                            idx2,
                            "reject_reproj1",
                            scoreInfoIt->second,
                            parallaxScore,
                            reprojRatio1,
                            reprojRatio2,
                            scaleScore,
                            finalCandidateScore,
                            finalTotalScore);
                    }
                    continue;
                }

            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                const double reprojError1 =
                    static_cast<double>(errX1 * errX1 + errY1 * errY1 +
                                        errX1_r * errX1_r);
                const double reprojLimit1 =
                    static_cast<double>(7.8 * sigmaSquare1);
                reprojRatio1 =
                    reprojLimit1 > 0.0 ? reprojError1 / reprojLimit1 : 1.0;
                if(reprojRatio1 > 1.0)
                {
                    if(dynamicMapAdmissionV5UsefulnessLog &&
                       scoreBasedBoundaryCandidate)
                    {
                        PrintAdmissionV5CandidateGeometryEvent(
                            mpCurrentKeyFrame,
                            pKF2,
                            idx1,
                            idx2,
                            "reject_reproj1",
                            scoreInfoIt->second,
                            parallaxScore,
                            reprojRatio1,
                            reprojRatio2,
                            scaleScore,
                            finalCandidateScore,
                            finalTotalScore);
                    }
                    continue;
                }
            }
            if(sameInstanceCandidate)
            {
                ++sameInstancePairsAfterReproj1;
                ++pairSameInstanceAfterReproj1;
            }
            if(promotedBoundaryCandidate)
                ++promotedGeomAfterReproj1;

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3D)+tcw2(0);
            const float y2 = Rcw2.row(1).dot(x3D)+tcw2(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                cv::Point2f uv2 = pCamera2->project(cv::Point3f(x2,y2,z2));
                float errX2 = uv2.x - kp2.pt.x;
                float errY2 = uv2.y - kp2.pt.y;
                const double reprojError2 =
                    static_cast<double>(errX2 * errX2 + errY2 * errY2);
                const double reprojLimit2 =
                    static_cast<double>(5.991 * sigmaSquare2);
                reprojRatio2 =
                    reprojLimit2 > 0.0 ? reprojError2 / reprojLimit2 : 1.0;
                if(reprojRatio2 > 1.0)
                {
                    if(dynamicMapAdmissionV5UsefulnessLog &&
                       scoreBasedBoundaryCandidate)
                    {
                        PrintAdmissionV5CandidateGeometryEvent(
                            mpCurrentKeyFrame,
                            pKF2,
                            idx1,
                            idx2,
                            "reject_reproj2",
                            scoreInfoIt->second,
                            parallaxScore,
                            reprojRatio1,
                            reprojRatio2,
                            scaleScore,
                            finalCandidateScore,
                            finalTotalScore);
                    }
                    continue;
                }
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                const double reprojError2 =
                    static_cast<double>(errX2 * errX2 + errY2 * errY2 +
                                        errX2_r * errX2_r);
                const double reprojLimit2 =
                    static_cast<double>(7.8 * sigmaSquare2);
                reprojRatio2 =
                    reprojLimit2 > 0.0 ? reprojError2 / reprojLimit2 : 1.0;
                if(reprojRatio2 > 1.0)
                {
                    if(dynamicMapAdmissionV5UsefulnessLog &&
                       scoreBasedBoundaryCandidate)
                    {
                        PrintAdmissionV5CandidateGeometryEvent(
                            mpCurrentKeyFrame,
                            pKF2,
                            idx1,
                            idx2,
                            "reject_reproj2",
                            scoreInfoIt->second,
                            parallaxScore,
                            reprojRatio1,
                            reprojRatio2,
                            scaleScore,
                            finalCandidateScore,
                            finalTotalScore);
                    }
                    continue;
                }
            }
            if(sameInstanceCandidate)
            {
                ++sameInstancePairsAfterReproj2;
                ++pairSameInstanceAfterReproj2;
            }
            if(promotedBoundaryCandidate)
                ++promotedGeomAfterReproj2;

            //Check scale consistency
            Eigen::Vector3f normal1 = x3D - Ow1;
            float dist1 = normal1.norm();

            Eigen::Vector3f normal2 = x3D - Ow2;
            float dist2 = normal2.norm();

            if(dist1==0 || dist2==0)
            {
                if(dynamicMapAdmissionV5UsefulnessLog &&
                   scoreBasedBoundaryCandidate)
                {
                    PrintAdmissionV5CandidateGeometryEvent(
                        mpCurrentKeyFrame,
                        pKF2,
                        idx1,
                        idx2,
                        "reject_scale",
                        scoreInfoIt->second,
                        parallaxScore,
                        reprojRatio1,
                        reprojRatio2,
                        scaleScore,
                        finalCandidateScore,
                        finalTotalScore);
                }
                continue;
            }

            if(mbFarPoints && (dist1>=mThFarPoints||dist2>=mThFarPoints)) // MODIFICATION
            {
                if(dynamicMapAdmissionV5UsefulnessLog &&
                   scoreBasedBoundaryCandidate)
                {
                    PrintAdmissionV5CandidateGeometryEvent(
                        mpCurrentKeyFrame,
                        pKF2,
                        idx1,
                        idx2,
                        "reject_scale",
                        scoreInfoIt->second,
                        parallaxScore,
                        reprojRatio1,
                        reprojRatio2,
                        scaleScore,
                        finalCandidateScore,
                        finalTotalScore);
                }
                continue;
            }

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
            {
                if(dynamicMapAdmissionV5UsefulnessLog &&
                   scoreBasedBoundaryCandidate)
                {
                    PrintAdmissionV5CandidateGeometryEvent(
                        mpCurrentKeyFrame,
                        pKF2,
                        idx1,
                        idx2,
                        "reject_scale",
                        scoreInfoIt->second,
                        parallaxScore,
                        reprojRatio1,
                        reprojRatio2,
                        scaleScore,
                        finalCandidateScore,
                        finalTotalScore);
                }
                continue;
            }
            const double ratioScale =
                std::max(1e-9,
                         static_cast<double>(ratioDist) /
                         std::max(1e-9, static_cast<double>(ratioOctave)));
            scaleScore =
                Clamp01(1.0 -
                        std::fabs(std::log(ratioScale)) /
                        std::max(1e-9, std::log(static_cast<double>(ratioFactor))));
            if(sameInstanceCandidate)
            {
                ++sameInstancePairsAfterScale;
                ++pairSameInstanceAfterScale;
            }
            if(promotedBoundaryCandidate)
                ++promotedGeomAfterScale;

            if(scoreBasedBoundaryCandidate)
            {
                ++scoreCandidateGeomEvaluatedBoundaryPairs;
                finalCandidateScore =
                    ComputeAdmissionCandidateScore(parallaxScore,
                                                   reprojRatio1,
                                                   reprojRatio2,
                                                   scaleScore);
                finalTotalScore =
                    DynamicMapAdmissionScoreSupportWeight() *
                        scoreInfoIt->second.supportScore +
                    DynamicMapAdmissionScoreCandidateWeight() *
                        finalCandidateScore;
                if(finalCandidateScore <
                       DynamicMapAdmissionScoreMinCandidateScore() ||
                   finalTotalScore < DynamicMapAdmissionScoreMinTotalScore())
                {
                    ++scorePostGeomRejectedBoundaryPairs;
                    if(dynamicMapAdmissionV5UsefulnessLog)
                    {
                        PrintAdmissionV5CandidateGeometryEvent(
                            mpCurrentKeyFrame,
                            pKF2,
                            idx1,
                            idx2,
                            "reject_score",
                            scoreInfoIt->second,
                            parallaxScore,
                            reprojRatio1,
                            reprojRatio2,
                            scaleScore,
                            finalCandidateScore,
                            finalTotalScore);
                    }
                    continue;
                }
                ++scoreCreatedBoundaryPairs;
                if(dynamicMapAdmissionV5UsefulnessLog)
                {
                    PrintAdmissionV5CandidateGeometryEvent(
                        mpCurrentKeyFrame,
                        pKF2,
                        idx1,
                        idx2,
                        "created",
                        scoreInfoIt->second,
                        parallaxScore,
                        reprojRatio1,
                        reprojRatio2,
                        scaleScore,
                        finalCandidateScore,
                        finalTotalScore);
                }
            }

            // Triangulation is succesfull
            ++triangulatedPoints;
            MapPoint* pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpAtlas->GetCurrentMap());
            ++createdMapPoints;
            if(promotedBoundaryCandidate)
                ++promotedGeomCreated;
            const bool createdDirectDynamic =
                mpCurrentKeyFrame->GetFeatureInstanceId(idx1) > 0 ||
                pKF2->GetFeatureInstanceId(idx2) > 0;
            const bool createdStaticNearDynamicBoundary =
                !createdDirectDynamic &&
                (mpCurrentKeyFrame->IsFeatureStaticNearDynamicMask(idx1) ||
                 pKF2->IsFeatureStaticNearDynamicMask(idx2));
            if(nearBoundaryDiagnostics)
            {
                pMP->SetAdmissionDiagnostics(
                    createdDirectDynamic,
                    createdStaticNearDynamicBoundary,
                    static_cast<long>(mpCurrentKeyFrame->mnFrameId),
                    static_cast<long>(mpCurrentKeyFrame->mnId),
                    idx1,
                    DynamicMapAdmissionBoundaryRadiusPx());
            }
            if((dynamicMapAdmissionV5UsefulnessLog ||
                dynamicMapAdmissionCoverageAwareV7) &&
               scoreBasedBoundaryCandidate)
            {
                pMP->SetScoreAdmissionDiagnostics(
                    scoreInfoIt->second.supportScore,
                    finalCandidateScore,
                    finalTotalScore,
                    scoreInfoIt->second.rawSupport,
                    scoreInfoIt->second.reliableSupport,
                    scoreInfoIt->second.residualReliableSupport,
                    scoreInfoIt->second.depthConsistentSupport);
            }
            if(nearBoundaryDiagnostics)
            {
                if(createdDirectDynamic)
                    ++lmCreatedDirectDynamicPoints;
                else if(createdStaticNearDynamicBoundary)
                    ++lmCreatedNearBoundaryPoints;
                else
                    ++lmCreatedCleanStaticPoints;
            }
            if (bPointStereo)
                countStereo++;
            
            pMP->AddObservation(mpCurrentKeyFrame,idx1);
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            const int instanceId1 = mpCurrentKeyFrame->GetFeatureInstanceId(idx1);
            const int instanceId2 = pKF2->GetFeatureInstanceId(idx2);
            const int semanticLabel1 = mpCurrentKeyFrame->GetFeatureSemanticLabel(idx1);
            const int semanticLabel2 = pKF2->GetFeatureSemanticLabel(idx2);

            if(instanceId1 > 0 && instanceId1 == instanceId2)
            {
                ++sameInstanceMatches;
                const int semanticLabel = (semanticLabel1 > 0) ? semanticLabel1 : semanticLabel2;
                Instance* pInstance = static_cast<Instance*>(NULL);
                if(StrictInstanceStructureFromDynamicObservationsOnly())
                {
                    pMP->SetInstanceId(instanceId1);
                    if(semanticLabel > 0)
                        pMP->SetSemanticLabel(semanticLabel);
                    pInstance = EnsureInstanceInMap(pMP->GetMap(), instanceId1, semanticLabel);
                }
                else
                {
                    pInstance = BindMapPointToInstance(pMP, instanceId1, semanticLabel, true);
                }
                if(pInstance)
                {
                    ++createdInstanceBoundPoints;
                    createdPointsPerInstance[instanceId1]++;
                    if(HasMatureInstanceBackendState(pInstance, mpCurrentKeyFrame))
                    {
                        pInstance->UpdateMotionPrior(mpCurrentKeyFrame, pInstance->GetVelocity());
                        pInstance->UpdatePoseProxy(mpCurrentKeyFrame, pInstance->GetLastPoseEstimate());
                    }
                }
            }
            else
            {
                if(instanceId1 > 0 || instanceId2 > 0)
                {
                    ++mismatchedInstanceMatches;
                    if(instanceId1 > 0)
                        mismatchedPairInstanceCounts[instanceId1]++;
                    if(instanceId2 > 0)
                        mismatchedPairInstanceCounts[instanceId2]++;
                }
                const int semanticLabel = (semanticLabel1 > 0) ? semanticLabel1 : semanticLabel2;
                if(semanticLabel > 0)
                {
                    pMP->SetSemanticLabel(semanticLabel);
                    ++createdSemanticOnlyPoints;
                }
            }

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpAtlas->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);
        }

        if(dynamicMapAdmissionDelayedBoundaryCreateNewMapPoints &&
           supportPromotedBoundaryPairs > 0)
        {
            std::cout << "[STSLAM_DYNAMIC_MAP_ADMISSION_PROMOTED_GEOM]"
                      << " frame=" << mpCurrentKeyFrame->mnFrameId
                      << " stage=create_new_map_points"
                      << " current_kf=" << mpCurrentKeyFrame->mnId
                      << " neighbor_kf=" << pKF2->mnId
                      << " support_promoted_boundary_pairs="
                      << supportPromotedBoundaryPairs
                      << " promoted_geom_enter=" << promotedGeomEnter
                      << " promoted_geom_parallax="
                      << promotedGeomAfterParallax
                      << " promoted_geom_triangulated="
                      << promotedGeomAfterTriangulation
                      << " promoted_geom_depth="
                      << promotedGeomAfterDepth
                      << " promoted_geom_reproj1="
                      << promotedGeomAfterReproj1
                      << " promoted_geom_reproj2="
                      << promotedGeomAfterReproj2
                      << " promoted_geom_scale="
                      << promotedGeomAfterScale
                      << " promoted_geom_created="
                      << promotedGeomCreated
                      << std::endl;
        }

        if(dynamicMapAdmissionScoreBased &&
           scoreSupportCandidateBoundaryPairs > 0)
        {
            std::cout << "[STSLAM_DYNAMIC_MAP_ADMISSION_SCORE_BASED]"
                      << " frame=" << mpCurrentKeyFrame->mnFrameId
                      << " stage=create_new_map_points"
                      << " current_kf=" << mpCurrentKeyFrame->mnId
                      << " neighbor_kf=" << pKF2->mnId
                      << " min_support_score="
                      << DynamicMapAdmissionScoreMinSupportScore()
                      << " min_candidate_score="
                      << DynamicMapAdmissionScoreMinCandidateScore()
                      << " min_total_score="
                      << DynamicMapAdmissionScoreMinTotalScore()
                      << " support_candidates="
                      << scoreSupportCandidateBoundaryPairs
                      << " support_accepted="
                      << scoreSupportAcceptedBoundaryPairs
                      << " support_rejected="
                      << scoreSupportRejectedBoundaryPairs
                      << " geom_evaluated="
                      << scoreCandidateGeomEvaluatedBoundaryPairs
                      << " post_geom_rejected="
                      << scorePostGeomRejectedBoundaryPairs
                      << " score_created="
                      << scoreCreatedBoundaryPairs
                      << std::endl;
        }

        if(dynamicMapAdmissionCoverageAwareV7 &&
           v7CoverageCandidateBoundaryPairs > 0)
        {
            std::cout << "[STSLAM_DYNAMIC_MAP_ADMISSION_V7_COVERAGE]"
                      << " frame=" << mpCurrentKeyFrame->mnFrameId
                      << " stage=create_new_map_points"
                      << " current_kf=" << mpCurrentKeyFrame->mnId
                      << " neighbor_kf=" << pKF2->mnId
                      << " v7_candidates="
                      << v7CoverageCandidateBoundaryPairs
                      << " v7_allowed="
                      << v7CoverageAllowedBoundaryPairs
                      << " v7_rejected="
                      << v7CoverageRejectedBoundaryPairs
                      << " v7_state_allowed="
                      << v7CoverageStateAllowedBoundaryPairs
                      << " v7_coverage_allowed="
                      << v7CoverageGapAllowedBoundaryPairs
                      << " v7_quota_rejected="
                      << v7CoverageQuotaRejectedBoundaryPairs
                      << " v7_promoted_keyframe_so_far="
                      << v7PromotedThisKeyFrame
                      << " v7_promoted_neighbor="
                      << v7PromotedThisNeighbor
                      << " v7_max_promotions_per_kf="
                      << DynamicMapAdmissionV7MaxPromotionsPerKeyFrame()
                      << " v7_max_promotions_per_neighbor="
                      << DynamicMapAdmissionV7MaxPromotionsPerNeighbor()
                      << " unmatched_boundary_features="
                      << admissionCoverageContext.unmatchedBoundaryFeatures
                      << " unmatched_boundary_cells="
                      << admissionCoverageContext.unmatchedBoundaryCells
                      << " total_boundary_features="
                      << admissionCoverageContext.totalBoundaryFeatures
                      << " coverage_pressure="
                      << (admissionCoverageContext.coveragePressure ? 1 : 0)
                      << " min_unmatched_boundary_features="
                      << DynamicMapAdmissionV7MinUnmatchedBoundaryFeatures()
                      << " min_unmatched_boundary_cells="
                      << DynamicMapAdmissionV7MinUnmatchedBoundaryCells()
                      << " state_need_score="
                      << admissionStateContext.needScore
                      << " v7_min_need_score="
                      << DynamicMapAdmissionV7MinNeedScore()
                      << " tracking_pressure="
                      << (admissionStateContext.trackingPressure ? 1 : 0)
                      << " keyframe_pressure="
                      << (admissionStateContext.keyframePressure ? 1 : 0)
                      << " scale_pressure="
                      << (admissionStateContext.scalePressure ? 1 : 0)
                      << " lba_pressure="
                      << (admissionStateContext.localBAPressure ? 1 : 0)
                      << std::endl;
        }

        if(DynamicMapAdmissionStateAware() &&
           stateAwareCandidateBoundaryPairs > 0)
        {
            std::cout << "[STSLAM_DYNAMIC_MAP_ADMISSION_STATE_AWARE]"
                      << " frame=" << mpCurrentKeyFrame->mnFrameId
                      << " stage=create_new_map_points"
                      << " current_kf=" << mpCurrentKeyFrame->mnId
                      << " neighbor_kf=" << pKF2->mnId
                      << " state_candidates="
                      << stateAwareCandidateBoundaryPairs
                      << " state_allowed="
                      << stateAwareAllowedBoundaryPairs
                      << " state_rejected="
                      << stateAwareRejectedBoundaryPairs
                      << " need_score="
                      << admissionStateContext.needScore
                      << " min_need_score="
                      << DynamicMapAdmissionStateMinNeedScore()
                      << " tracking_inliers="
                      << admissionStateContext.trackingInliers
                      << " tracking_pressure="
                      << (admissionStateContext.trackingPressure ? 1 : 0)
                      << " keyframe_gap="
                      << admissionStateContext.keyframeFrameGap
                      << " keyframe_pressure="
                      << (admissionStateContext.keyframePressure ? 1 : 0)
                      << " kf_step="
                      << admissionStateContext.keyframeStep
                      << " kf_step_ewma="
                      << admissionStateContext.keyframeStepEwma
                      << " kf_step_ratio="
                      << admissionStateContext.keyframeStepRatio
                      << " scale_pressure="
                      << (admissionStateContext.scalePressure ? 1 : 0)
                      << " last_lba_edges="
                      << admissionStateContext.lastLBAEdges
                      << " last_lba_mps="
                      << admissionStateContext.lastLBAMapPoints
                      << " last_lba_edges_per_mp="
                      << admissionStateContext.lastLBAEdgesPerMP
                      << " lba_pressure="
                      << (admissionStateContext.localBAPressure ? 1 : 0)
                      << std::endl;
        }

        if(debugFocusFrame)
        {
            std::cout << "[STSLAM_FOCUS] frame=" << mpCurrentKeyFrame->mnFrameId
                      << " stage=create_new_map_points_pair_geom"
                      << " current_kf=" << mpCurrentKeyFrame->mnId
                      << " neighbor_kf=" << pKF2->mnId
                      << " same_geom={enter:" << pairSameInstanceEnter
                      << ",parallax:" << pairSameInstanceAfterParallax
                      << ",triangulated:" << pairSameInstanceAfterTriangulation
                      << ",depth:" << pairSameInstanceAfterDepth
                      << ",reproj1:" << pairSameInstanceAfterReproj1
                      << ",reproj2:" << pairSameInstanceAfterReproj2
                      << ",scale:" << pairSameInstanceAfterScale
                      << "}"
                      << std::endl;
        }
    }

    if(debugFocusFrame)
    {
        std::cout << "[STSLAM_FOCUS] frame=" << mpCurrentKeyFrame->mnFrameId
                  << " stage=create_new_map_points"
                  << " current_kf=" << mpCurrentKeyFrame->mnId
                  << " neighbor_kfs=" << vpNeighKFs.size()
                  << " candidate_matches=" << totalCandidateMatches
                  << " triangulated_points=" << triangulatedPoints
                  << " created_map_points=" << createdMapPoints
                  << " same_instance_matches=" << sameInstanceMatches
                  << " mismatched_instance_matches=" << mismatchedInstanceMatches
                  << " created_instance_bound_points=" << createdInstanceBoundPoints
                  << " created_semantic_only_points=" << createdSemanticOnlyPoints
                  << " same_instance_geom={enter:" << sameInstancePairsEnteringGeometry
                  << ",parallax:" << sameInstancePairsAfterParallax
                  << ",triangulated:" << sameInstancePairsAfterTriangulation
                  << ",depth:" << sameInstancePairsAfterPositiveDepth
                  << ",reproj1:" << sameInstancePairsAfterReproj1
                  << ",reproj2:" << sameInstancePairsAfterReproj2
                  << ",scale:" << sameInstancePairsAfterScale
                  << "}"
                  << " created_per_instance=" << FormatTopCounts(createdPointsPerInstance)
                  << " mismatched_instances=" << FormatTopCounts(mismatchedPairInstanceCounts)
                  << std::endl;
    }

    if(nearBoundaryDiagnostics)
    {
        std::cout << "[STSLAM_NEAR_BOUNDARY_ADMISSION]"
                  << " stage=create_new_map_points"
                  << " frame=" << mpCurrentKeyFrame->mnFrameId
                  << " current_kf=" << mpCurrentKeyFrame->mnId
                  << " radius_px=" << DynamicMapAdmissionBoundaryRadiusPx()
                  << " created_near_boundary_new_points="
                  << lmCreatedNearBoundaryPoints
                  << " created_clean_static_new_points="
                  << lmCreatedCleanStaticPoints
                  << " created_direct_dynamic_new_points="
                  << lmCreatedDirectDynamicPoints
                  << " triangulated_points=" << triangulatedPoints
                  << " created_map_points=" << createdMapPoints
                  << std::endl;
    }
}

void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if(mbMonocular)
        nn=30;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;
    }

    // Add some covisible of covisible
    // Extend to some second neighbors if abort is not requested
    for(int i=0, imax=vpTargetKFs.size(); i<imax; i++)
    {
        const vector<KeyFrame*> vpSecondNeighKFs = vpTargetKFs[i]->GetBestCovisibilityKeyFrames(20);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
            pKFi2->mnFuseTargetForKF=mpCurrentKeyFrame->mnId;
        }
        if (mbAbortBA)
            break;
    }

    // Extend to temporal neighbors
    if(mbInertial)
    {
        KeyFrame* pKFi = mpCurrentKeyFrame->mPrevKF;
        while(vpTargetKFs.size()<20 && pKFi)
        {
            if(pKFi->isBad() || pKFi->mnFuseTargetForKF==mpCurrentKeyFrame->mnId)
            {
                pKFi = pKFi->mPrevKF;
                continue;
            }
            vpTargetKFs.push_back(pKFi);
            pKFi->mnFuseTargetForKF=mpCurrentKeyFrame->mnId;
            pKFi = pKFi->mPrevKF;
        }
    }

    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;

        matcher.Fuse(pKFi,vpMapPointMatches);
        if(pKFi->NLeft != -1) matcher.Fuse(pKFi,vpMapPointMatches,true);
    }


    if (mbAbortBA)
        return;

    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);
    if(mpCurrentKeyFrame->NLeft != -1) matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates,true);


    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    const int Nd = 21;
    mpCurrentKeyFrame->UpdateBestCovisibles();
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    float redundant_th;
    if(!mbInertial)
        redundant_th = 0.9;
    else if (mbMonocular)
        redundant_th = 0.9;
    else
        redundant_th = 0.5;

    const bool bInitImu = mpAtlas->isImuInitialized();
    int count=0;

    // Compoute last KF from optimizable window:
    unsigned int last_ID;
    if (mbInertial)
    {
        int count = 0;
        KeyFrame* aux_KF = mpCurrentKeyFrame;
        while(count<Nd && aux_KF->mPrevKF)
        {
            aux_KF = aux_KF->mPrevKF;
            count++;
        }
        last_ID = aux_KF->mnId;
    }



    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        count++;
        KeyFrame* pKF = *vit;

        if((pKF->mnId==pKF->GetMap()->GetInitKFid()) || pKF->isBad())
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = (pKF -> NLeft == -1) ? pKF->mvKeysUn[i].octave
                                                                     : (i < pKF -> NLeft) ? pKF -> mvKeys[i].octave
                                                                                          : pKF -> mvKeysRight[i].octave;
                        const map<KeyFrame*, tuple<int,int>> observations = pMP->GetObservations();
                        int nObs=0;
                        for(map<KeyFrame*, tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            tuple<int,int> indexes = mit->second;
                            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
                            int scaleLeveli = -1;
                            if(pKFi -> NLeft == -1)
                                scaleLeveli = pKFi->mvKeysUn[leftIndex].octave;
                            else {
                                if (leftIndex != -1) {
                                    scaleLeveli = pKFi->mvKeys[leftIndex].octave;
                                }
                                if (rightIndex != -1) {
                                    int rightLevel = pKFi->mvKeysRight[rightIndex - pKFi->NLeft].octave;
                                    scaleLeveli = (scaleLeveli == -1 || scaleLeveli > rightLevel) ? rightLevel
                                                                                                  : scaleLeveli;
                                }
                            }

                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>thObs)
                                    break;
                            }
                        }
                        if(nObs>thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }

        if(nRedundantObservations>redundant_th*nMPs)
        {
            if (mbInertial)
            {
                if (mpAtlas->KeyFramesInMap()<=Nd)
                    continue;

                if(pKF->mnId>(mpCurrentKeyFrame->mnId-2))
                    continue;

                if(pKF->mPrevKF && pKF->mNextKF)
                {
                    const float t = pKF->mNextKF->mTimeStamp-pKF->mPrevKF->mTimeStamp;

                    if((bInitImu && (pKF->mnId<last_ID) && t<3.) || (t<0.5))
                    {
                        pKF->mNextKF->mpImuPreintegrated->MergePrevious(pKF->mpImuPreintegrated);
                        pKF->mNextKF->mPrevKF = pKF->mPrevKF;
                        pKF->mPrevKF->mNextKF = pKF->mNextKF;
                        pKF->mNextKF = NULL;
                        pKF->mPrevKF = NULL;
                        pKF->SetBadFlag();
                    }
                    else if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2() && ((pKF->GetImuPosition()-pKF->mPrevKF->GetImuPosition()).norm()<0.02) && (t<3))
                    {
                        pKF->mNextKF->mpImuPreintegrated->MergePrevious(pKF->mpImuPreintegrated);
                        pKF->mNextKF->mPrevKF = pKF->mPrevKF;
                        pKF->mPrevKF->mNextKF = pKF->mNextKF;
                        pKF->mNextKF = NULL;
                        pKF->mPrevKF = NULL;
                        pKF->SetBadFlag();
                    }
                }
            }
            else
            {
                pKF->SetBadFlag();
            }
        }
        if((count > 20 && mbAbortBA) || count>100)
        {
            break;
        }
    }
}

void LocalMapping::RequestReset()
{
    QueueReset();
    cout << "LM: Map reset, waiting..." << endl;

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
    cout << "LM: Map reset, Done!!!" << endl;
}

void LocalMapping::QueueReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        cout << "LM: Map reset recieved" << endl;
        mbResetRequested = true;
    }
}

void LocalMapping::RequestResetActiveMap(Map* pMap)
{
    QueueResetActiveMap(pMap);
    cout << "LM: Active map reset, waiting..." << endl;

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequestedActiveMap)
                break;
        }
        usleep(3000);
    }
    cout << "LM: Active map reset, Done!!!" << endl;
}

void LocalMapping::QueueResetActiveMap(Map* pMap)
{
    {
        unique_lock<mutex> lock(mMutexReset);
        cout << "LM: Active map reset recieved" << endl;
        mbResetRequestedActiveMap = true;
        mpMapToReset = pMap;
    }
}

void LocalMapping::ServicePendingResetRequests()
{
    ResetIfRequested();
    SetAcceptKeyFrames(true);
}

void LocalMapping::ResetIfRequested()
{
    bool executed_reset = false;
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbResetRequested)
        {
            executed_reset = true;

            cout << "LM: Reseting Atlas in Local Mapping..." << endl;
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();
            mbResetRequested = false;
            mbResetRequestedActiveMap = false;

            // Inertial parameters
            mTinit = 0.f;
            mbNotBA2 = true;
            mbNotBA1 = true;
            mbBadImu=false;

            mIdxInit=0;

            cout << "LM: End reseting Local Mapping..." << endl;
        }

        if(mbResetRequestedActiveMap) {
            executed_reset = true;
            cout << "LM: Reseting current map in Local Mapping..." << endl;
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();

            // Inertial parameters
            mTinit = 0.f;
            mbNotBA2 = true;
            mbNotBA1 = true;
            mbBadImu=false;

            mbResetRequested = false;
            mbResetRequestedActiveMap = false;
            cout << "LM: End reseting Local Mapping..." << endl;
        }
    }
    if(executed_reset)
        cout << "LM: Reset free the mutex" << endl;

}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    PrintAdmissionV5AggregateSummary();
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void LocalMapping::InitializeIMU(float priorG, float priorA, bool bFIBA)
{
    if (mbResetRequested)
        return;

    float minTime;
    int nMinKF;
    if (mbMonocular)
    {
        minTime = 2.0;
        nMinKF = 10;
    }
    else
    {
        minTime = 1.0;
        nMinKF = 10;
    }


    if(mpAtlas->KeyFramesInMap()<nMinKF)
        return;

    // Retrieve all keyframe in temporal order
    list<KeyFrame*> lpKF;
    KeyFrame* pKF = mpCurrentKeyFrame;
    while(pKF->mPrevKF)
    {
        lpKF.push_front(pKF);
        pKF = pKF->mPrevKF;
    }
    lpKF.push_front(pKF);
    vector<KeyFrame*> vpKF(lpKF.begin(),lpKF.end());

    if(vpKF.size()<nMinKF)
        return;

    mFirstTs=vpKF.front()->mTimeStamp;
    if(mpCurrentKeyFrame->mTimeStamp-mFirstTs<minTime)
        return;

    bInitializing = true;

    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    const int N = vpKF.size();
    IMU::Bias b(0,0,0,0,0,0);

    // Compute and KF velocities mRwg estimation
    if (!mpCurrentKeyFrame->GetMap()->isImuInitialized())
    {
        Eigen::Matrix3f Rwg;
        Eigen::Vector3f dirG;
        dirG.setZero();
        for(vector<KeyFrame*>::iterator itKF = vpKF.begin(); itKF!=vpKF.end(); itKF++)
        {
            if (!(*itKF)->mpImuPreintegrated)
                continue;
            if (!(*itKF)->mPrevKF)
                continue;

            dirG -= (*itKF)->mPrevKF->GetImuRotation() * (*itKF)->mpImuPreintegrated->GetUpdatedDeltaVelocity();
            Eigen::Vector3f _vel = ((*itKF)->GetImuPosition() - (*itKF)->mPrevKF->GetImuPosition())/(*itKF)->mpImuPreintegrated->dT;
            (*itKF)->SetVelocity(_vel);
            (*itKF)->mPrevKF->SetVelocity(_vel);
        }

        dirG = dirG/dirG.norm();
        Eigen::Vector3f gI(0.0f, 0.0f, -1.0f);
        Eigen::Vector3f v = gI.cross(dirG);
        const float nv = v.norm();
        const float cosg = gI.dot(dirG);
        const float ang = acos(cosg);
        Eigen::Vector3f vzg = v*ang/nv;
        Rwg = Sophus::SO3f::exp(vzg).matrix();
        mRwg = Rwg.cast<double>();
        mTinit = mpCurrentKeyFrame->mTimeStamp-mFirstTs;
    }
    else
    {
        mRwg = Eigen::Matrix3d::Identity();
        mbg = mpCurrentKeyFrame->GetGyroBias().cast<double>();
        mba = mpCurrentKeyFrame->GetAccBias().cast<double>();
    }

    mScale=1.0;

    mInitTime = mpTracker->mLastFrame.mTimeStamp-vpKF.front()->mTimeStamp;

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale, mbg, mba, mbMonocular, infoInertial, false, false, priorG, priorA);

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    if (mScale<1e-1)
    {
        cout << "scale too small" << endl;
        bInitializing=false;
        return;
    }

    // Before this line we are not changing the map
    {
        unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
        if ((fabs(mScale - 1.f) > 0.00001) || !mbMonocular) {
            Sophus::SE3f Twg(mRwg.cast<float>().transpose(), Eigen::Vector3f::Zero());
            mpAtlas->GetCurrentMap()->ApplyScaledRotation(Twg, mScale, true);
            mpTracker->UpdateFrameIMU(mScale, vpKF[0]->GetImuBias(), mpCurrentKeyFrame);
        }

        // Check if initialization OK
        if (!mpAtlas->isImuInitialized())
            for (int i = 0; i < N; i++) {
                KeyFrame *pKF2 = vpKF[i];
                pKF2->bImu = true;
            }
    }

    mpTracker->UpdateFrameIMU(1.0,vpKF[0]->GetImuBias(),mpCurrentKeyFrame);
    if (!mpAtlas->isImuInitialized())
    {
        mpAtlas->SetImuInitialized();
        mpTracker->t0IMU = mpTracker->mCurrentFrame.mTimeStamp;
        mpCurrentKeyFrame->bImu = true;
    }

    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    if (bFIBA)
    {
        if (priorA!=0.f)
            Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 100, false, mpCurrentKeyFrame->mnId, NULL, true, priorG, priorA);
        else
            Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 100, false, mpCurrentKeyFrame->mnId, NULL, false);
    }

    std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();

    Verbose::PrintMess("Global Bundle Adjustment finished\nUpdating map ...", Verbose::VERBOSITY_NORMAL);

    // Get Map Mutex
    unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);

    unsigned long GBAid = mpCurrentKeyFrame->mnId;

    // Process keyframes in the queue
    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    // Correct keyframes starting at map first keyframe
    list<KeyFrame*> lpKFtoCheck(mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.begin(),mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.end());

    while(!lpKFtoCheck.empty())
    {
        KeyFrame* pKF = lpKFtoCheck.front();
        const set<KeyFrame*> sChilds = pKF->GetChilds();
        Sophus::SE3f Twc = pKF->GetPoseInverse();
        for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
        {
            KeyFrame* pChild = *sit;
            if(!pChild || pChild->isBad())
                continue;

            if(pChild->mnBAGlobalForKF!=GBAid)
            {
                Sophus::SE3f Tchildc = pChild->GetPose() * Twc;
                pChild->mTcwGBA = Tchildc * pKF->mTcwGBA;

                Sophus::SO3f Rcor = pChild->mTcwGBA.so3().inverse() * pChild->GetPose().so3();
                if(pChild->isVelocitySet()){
                    pChild->mVwbGBA = Rcor * pChild->GetVelocity();
                }
                else {
                    Verbose::PrintMess("Child velocity empty!! ", Verbose::VERBOSITY_NORMAL);
                }

                pChild->mBiasGBA = pChild->GetImuBias();
                pChild->mnBAGlobalForKF = GBAid;

            }
            lpKFtoCheck.push_back(pChild);
        }

        pKF->mTcwBefGBA = pKF->GetPose();
        pKF->SetPose(pKF->mTcwGBA);

        if(pKF->bImu)
        {
            pKF->mVwbBefGBA = pKF->GetVelocity();
            pKF->SetVelocity(pKF->mVwbGBA);
            pKF->SetNewBias(pKF->mBiasGBA);
        } else {
            cout << "KF " << pKF->mnId << " not set to inertial!! \n";
        }

        lpKFtoCheck.pop_front();
    }

    // Correct MapPoints
    const vector<MapPoint*> vpMPs = mpAtlas->GetCurrentMap()->GetAllMapPoints();

    for(size_t i=0; i<vpMPs.size(); i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        if(pMP->mnBAGlobalForKF==GBAid)
        {
            // If optimized by Global BA, just update
            pMP->SetWorldPos(pMP->mPosGBA);
        }
        else
        {
            // Update according to the correction of its reference keyframe
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

            if(pRefKF->mnBAGlobalForKF!=GBAid)
                continue;

            // Map to non-corrected camera
            Eigen::Vector3f Xc = pRefKF->mTcwBefGBA * pMP->GetWorldPos();

            // Backproject using corrected camera
            pMP->SetWorldPos(pRefKF->GetPoseInverse() * Xc);
        }
    }

    Verbose::PrintMess("Map updated!", Verbose::VERBOSITY_NORMAL);

    mnKFs=vpKF.size();
    mIdxInit++;

    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
    {
        (*lit)->SetBadFlag();
        delete *lit;
    }
    mlNewKeyFrames.clear();

    mpTracker->mState=Tracking::OK;
    bInitializing = false;

    mpCurrentKeyFrame->GetMap()->IncreaseChangeIndex();

    return;
}

void LocalMapping::ScaleRefinement()
{
    // Minimum number of keyframes to compute a solution
    // Minimum time (seconds) between first and last keyframe to compute a solution. Make the difference between monocular and stereo
    // unique_lock<mutex> lock0(mMutexImuInit);
    if (mbResetRequested)
        return;

    // Retrieve all keyframes in temporal order
    list<KeyFrame*> lpKF;
    KeyFrame* pKF = mpCurrentKeyFrame;
    while(pKF->mPrevKF)
    {
        lpKF.push_front(pKF);
        pKF = pKF->mPrevKF;
    }
    lpKF.push_front(pKF);
    vector<KeyFrame*> vpKF(lpKF.begin(),lpKF.end());

    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    const int N = vpKF.size();

    mRwg = Eigen::Matrix3d::Identity();
    mScale=1.0;

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    if (mScale<1e-1) // 1e-1
    {
        cout << "scale too small" << endl;
        bInitializing=false;
        return;
    }
    
    Sophus::SO3d so3wg(mRwg);
    // Before this line we are not changing the map
    unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    if ((fabs(mScale-1.f)>0.002)||!mbMonocular)
    {
        Sophus::SE3f Tgw(mRwg.cast<float>().transpose(),Eigen::Vector3f::Zero());
        mpAtlas->GetCurrentMap()->ApplyScaledRotation(Tgw,mScale,true);
        mpTracker->UpdateFrameIMU(mScale,mpCurrentKeyFrame->GetImuBias(),mpCurrentKeyFrame);
    }
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
    {
        (*lit)->SetBadFlag();
        delete *lit;
    }
    mlNewKeyFrames.clear();

    double t_inertial_only = std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t0).count();

    // To perform pose-inertial opt w.r.t. last keyframe
    mpCurrentKeyFrame->GetMap()->IncreaseChangeIndex();

    return;
}



bool LocalMapping::IsInitializing()
{
    return bInitializing;
}


double LocalMapping::GetCurrKFTime()
{

    if (mpCurrentKeyFrame)
    {
        return mpCurrentKeyFrame->mTimeStamp;
    }
    else
        return 0.0;
}

KeyFrame* LocalMapping::GetCurrKF()
{
    return mpCurrentKeyFrame;
}

} //namespace ORB_SLAM
