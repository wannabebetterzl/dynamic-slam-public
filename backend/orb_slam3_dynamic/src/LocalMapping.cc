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
    mIdxInit(0), mScale(1.0), mInitSect(0), mbNotBA1(true), mbNotBA2(true), mIdxIteration(0), infoInertial(Eigen::MatrixXd::Zero(9,9))
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

            mbAbortBA = false;

            if(!CheckNewKeyFrames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                SearchInNeighbors();
            }

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndMPCreation = std::chrono::steady_clock::now();

            double timeMPCreation = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndMPCreation - time_EndMPCulling).count();
            vdMPCreation_ms.push_back(timeMPCreation);
#endif

            bool b_doneLBA = false;
            int num_FixedKF_BA = 0;
            int num_OptKF_BA = 0;
            int num_MPs_BA = 0;
            int num_edges_BA = 0;

            if(!CheckNewKeyFrames() && !stopRequested())
            {
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
#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_EndLBA = std::chrono::steady_clock::now();

                if(b_doneLBA)
                {
                    timeLBA_ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLBA - time_EndMPCreation).count();
                    vdLBA_ms.push_back(timeLBA_ms);

                    nLBA_exec += 1;
                    if(mbAbortBA)
                    {
                        nLBA_abort += 1;
                    }
                    vnLBA_edges.push_back(num_edges_BA);
                    vnLBA_KFopt.push_back(num_OptKF_BA);
                    vnLBA_KFfixed.push_back(num_FixedKF_BA);
                    vnLBA_MPs.push_back(num_MPs_BA);
                }

#endif

                // Initialize IMU here
                if(!mpCurrentKeyFrame->GetMap()->isImuInitialized() && mbInertial)
                {
                    if (mbMonocular)
                        InitializeIMU(1e2, 1e10, true);
                    else
                        InitializeIMU(1e2, 1e5, true);
                }


                // Check redundant local Keyframes
                KeyFrameCulling();

#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_EndKFCulling = std::chrono::steady_clock::now();

                timeKFCulling_ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndKFCulling - time_EndLBA).count();
                vdKFCulling_ms.push_back(timeKFCulling_ms);
#endif

                if ((mTinit<50.0f) && mbInertial)
                {
                    if(mpCurrentKeyFrame->GetMap()->isImuInitialized() && mpTracker->mState==Tracking::OK) // Enter here everytime local-mapping is called
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

                        // scale refinement
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
            }

#ifdef REGISTER_TIMES
            vdLBASync_ms.push_back(timeKFCulling_ms);
            vdKFCullingSync_ms.push_back(timeKFCulling_ms);
#endif

            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndLocalMap = std::chrono::steady_clock::now();

            double timeLocalMap = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLocalMap - time_StartProcessKF).count();
            vdLMTotal_ms.push_back(timeLocalMap);
#endif
        }
        else if(Stop() && !mbBadImu)
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if(CheckFinish())
                break;
        }

        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        usleep(3000);
    }

    SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
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

    int borrar = mlpRecentAddedMapPoints.size();

    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;

        if(pMP->isBad())
            lit = mlpRecentAddedMapPoints.erase(lit);
        else if(pMP->GetFoundRatio()<0.25f)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
        {
            lit++;
            borrar--;
        }
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
    const KeyFrameFeatureStats currentKFStats =
        debugFocusFrame ? CollectKeyFrameFeatureStats(mpCurrentKeyFrame) : KeyFrameFeatureStats();
    std::map<int, int> skippedShortBaselineNeighbors;

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
        if((ForceFilterDetectedDynamicFeatures() ||
            SplitDetectedDynamicFeaturesFromStaticMapping()) &&
           !vMatchedIndices.empty())
        {
            vector<pair<size_t,size_t> > vStaticMatchedIndices;
            vStaticMatchedIndices.reserve(vMatchedIndices.size());
            int skippedInstancePairs = 0;
            for(size_t pairIdx = 0; pairIdx < vMatchedIndices.size(); ++pairIdx)
            {
                const int idx1 = static_cast<int>(vMatchedIndices[pairIdx].first);
                const int idx2 = static_cast<int>(vMatchedIndices[pairIdx].second);
                const int instanceId1 = mpCurrentKeyFrame->GetFeatureInstanceId(idx1);
                const int instanceId2 = pKF2->GetFeatureInstanceId(idx2);
                const bool shouldSkipInstancePair =
                    ForceFilterDetectedDynamicFeatures() ?
                    (instanceId1 > 0 || instanceId2 > 0) :
                    (ShouldDetachRgbdInstanceFromStaticPath(mpCurrentKeyFrame->GetMap(), instanceId1) ||
                     ShouldDetachRgbdInstanceFromStaticPath(mpCurrentKeyFrame->GetMap(), instanceId2));
                if(shouldSkipInstancePair)
                {
                    ++skippedInstancePairs;
                    continue;
                }
                vStaticMatchedIndices.push_back(vMatchedIndices[pairIdx]);
            }

            if(skippedInstancePairs > 0)
            {
                std::cout << "[STSLAM_STATIC_MAPPING_DYNAMIC_SPLIT]"
                          << " frame=" << mpCurrentKeyFrame->mnFrameId
                          << " stage=create_new_map_points"
                          << " current_kf=" << mpCurrentKeyFrame->mnId
                          << " neighbor_kf=" << pKF2->mnId
                          << " skipped_instance_pairs=" << skippedInstancePairs
                          << " kept_static_pairs=" << vStaticMatchedIndices.size()
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
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;
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
                goodProj = GeometricTools::Triangulate(xn1, xn2, eigTcw1, eigTcw2, x3D);
                if(!goodProj)
                    continue;
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
                goodProj = pKF2->UnprojectStereo(idx2, x3D);
            }
            else
            {
                continue; //No stereo and very low parallax
            }

            if(goodProj && bPointStereo)
                countStereoGoodProj++;

            if(!goodProj)
                continue;
            if(sameInstanceCandidate)
            {
                ++sameInstancePairsAfterTriangulation;
                ++pairSameInstanceAfterTriangulation;
            }

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3D) + tcw1(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3D) + tcw2(2);
            if(z2<=0)
                continue;
            if(sameInstanceCandidate)
            {
                ++sameInstancePairsAfterPositiveDepth;
                ++pairSameInstanceAfterDepth;
            }

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

                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;

            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }
            if(sameInstanceCandidate)
            {
                ++sameInstancePairsAfterReproj1;
                ++pairSameInstanceAfterReproj1;
            }

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
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }
            if(sameInstanceCandidate)
            {
                ++sameInstancePairsAfterReproj2;
                ++pairSameInstanceAfterReproj2;
            }

            //Check scale consistency
            Eigen::Vector3f normal1 = x3D - Ow1;
            float dist1 = normal1.norm();

            Eigen::Vector3f normal2 = x3D - Ow2;
            float dist2 = normal2.norm();

            if(dist1==0 || dist2==0)
                continue;

            if(mbFarPoints && (dist1>=mThFarPoints||dist2>=mThFarPoints)) // MODIFICATION
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;
            if(sameInstanceCandidate)
            {
                ++sameInstancePairsAfterScale;
                ++pairSameInstanceAfterScale;
            }

            // Triangulation is succesfull
            ++triangulatedPoints;
            MapPoint* pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpAtlas->GetCurrentMap());
            ++createdMapPoints;
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
    {
        unique_lock<mutex> lock(mMutexReset);
        cout << "LM: Map reset recieved" << endl;
        mbResetRequested = true;
    }
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

void LocalMapping::RequestResetActiveMap(Map* pMap)
{
    {
        unique_lock<mutex> lock(mMutexReset);
        cout << "LM: Active map reset recieved" << endl;
        mbResetRequestedActiveMap = true;
        mpMapToReset = pMap;
    }
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
