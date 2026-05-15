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

#include "MapPoint.h"
#include "Instance.h"
#include "ORBmatcher.h"

#include<mutex>
#include<map>
#include<cmath>
#include<fstream>
#include<iomanip>
#include<vector>

namespace ORB_SLAM3
{

namespace
{

struct AdmissionDiagnosticsRecord
{
    bool directDynamic = false;
    bool staticNearDynamicBoundary = false;
    bool scoreAdmission = false;
    long frameId = -1;
    long keyFrameId = -1;
    int featureIdx = -1;
    int boundaryRadiusPx = 0;
    int scoreRawSupport = 0;
    int scoreReliableSupport = 0;
    int scoreResidualSupport = 0;
    int scoreDepthSupport = 0;
    double supportScore = 0.0;
    double candidateScore = 0.0;
    double totalScore = 0.0;
    long neighborKeyFrameId = -1;
    int neighborFeatureIdx = -1;
    double baseline = 0.0;
    double cosParallaxRays = 0.0;
    double parallaxScore = 0.0;
    double reprojRatio1 = 0.0;
    double reprojRatio2 = 0.0;
    double scaleScore = 0.0;
    double finalCandidateScore = 0.0;
    double finalTotalScore = 0.0;
    double dist1 = 0.0;
    double dist2 = 0.0;
    double ratioDist = 0.0;
    double ratioOctave = 0.0;
    bool stereoPoint = false;
    bool boundaryCurrent = false;
    bool boundaryNeighbor = false;
    int lifecyclePrebad = 0;
    int lifecycleFoundRatioCull = 0;
    int lifecycleLowObsCull = 0;
    int lifecycleV7ResidualCull = 0;
    int lifecycleV7LowUseCull = 0;
    int lifecycleMatured = 0;
    int lifecycleSurvived = 0;
    long firstMaturedFrameId = -1;
    long lastLifecycleFrameId = -1;
    long lastLifecycleKeyFrameId = -1;
};

struct SupportQualityPoseUseRecord
{
    int observations = 0;
    int inliers = 0;
    double chi2Sum = 0.0;
};

struct ScoreAdmissionConstraintRoleRecord
{
    int localBAWindows = 0;
    int localBAEdges = 0;
    int localBAInliers = 0;
    int localBAFixedEdges = 0;
    int localBALocalEdges = 0;
    double localBAChi2Sum = 0.0;
};

std::mutex gAdmissionDiagnosticsMutex;
std::map<const MapPoint*, AdmissionDiagnosticsRecord> gAdmissionDiagnostics;

std::mutex gSupportQualityPoseUseMutex;
std::map<const MapPoint*, SupportQualityPoseUseRecord> gSupportQualityPoseUse;

std::mutex gScoreAdmissionConstraintRoleMutex;
std::map<const MapPoint*, ScoreAdmissionConstraintRoleRecord> gScoreAdmissionConstraintRole;

void CleanupInstanceMembership(Map* pMap,
                               const int instanceId,
                               const int semanticLabel,
                               MapPoint* pOldPoint,
                               MapPoint* pReplacementPoint,
                               const bool transferStructureMembership)
{
    if(!pMap || !pOldPoint || instanceId <= 0)
        return;

    Instance* pInstance = pMap->GetInstance(instanceId);
    if(!pInstance)
        return;

    pInstance->RemoveMapPoint(pOldPoint);

    if(transferStructureMembership &&
       pReplacementPoint &&
       !pReplacementPoint->IsDynamicInstanceObservationPoint())
    {
        const int replacementInstanceId = pReplacementPoint->GetInstanceId();
        if(replacementInstanceId <= 0)
        {
            pReplacementPoint->SetInstanceId(instanceId);
            if(semanticLabel > 0 && pReplacementPoint->GetSemanticLabel() <= 0)
                pReplacementPoint->SetSemanticLabel(semanticLabel);
            pInstance->AddMapPoint(pReplacementPoint);
        }
        else if(replacementInstanceId == instanceId)
        {
            pInstance->AddMapPoint(pReplacementPoint);
        }
    }

    if(pInstance->NumMapPoints() == 0)
        pMap->EraseInstance(instanceId);
}

} // namespace

long unsigned int MapPoint::nNextId=0;
mutex MapPoint::mGlobalMutex;

    MapPoint::MapPoint():
        mnFirstKFid(0), mnFirstFrame(0), nObs(0), mnTrackReferenceForFrame(0),
        mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
        mnCorrectedReference(0), mnBAGlobalForKF(0), mnVisible(1), mnFound(1), mbBad(false),
        mpReplaced(static_cast<MapPoint*>(NULL)), mnInstanceId(-1), mnSemanticLabel(0),
    mnFirstObservationFrame(-1), mnLastObservationFrame(-1), mnObservationCount(0),
    mnLifecycleType(kStaticWorldMapPoint)
{
    mpReplaced = static_cast<MapPoint*>(NULL);
}

MapPoint::MapPoint(const Eigen::Vector3f &Pos, KeyFrame *pRefKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap),
    mnOriginMapId(pMap->GetId()), mnInstanceId(-1), mnSemanticLabel(0),
    mnFirstObservationFrame(-1), mnLastObservationFrame(-1), mnObservationCount(0),
    mnLifecycleType(kStaticWorldMapPoint)
{
    SetWorldPos(Pos);

    mNormalVector.setZero();

    mbTrackInViewR = false;
    mbTrackInView = false;

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

MapPoint::MapPoint(const double invDepth, cv::Point2f uv_init, KeyFrame* pRefKF, KeyFrame* pHostKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap),
    mnOriginMapId(pMap->GetId()), mnInstanceId(-1), mnSemanticLabel(0),
    mnFirstObservationFrame(-1), mnLastObservationFrame(-1), mnObservationCount(0),
    mnLifecycleType(kStaticWorldMapPoint)
{
    mInvDepth=invDepth;
    mInitU=(double)uv_init.x;
    mInitV=(double)uv_init.y;
    mpHostKF = pHostKF;

    mNormalVector.setZero();

    // Worldpos is not set
    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

MapPoint::MapPoint(const Eigen::Vector3f &Pos, Map* pMap, Frame* pFrame, const int &idxF):
    mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
    mnBALocalForKF(0), mnFuseCandidateForKF(0),mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
    mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap), mnOriginMapId(pMap->GetId()),
    mnInstanceId(-1), mnSemanticLabel(0), mnFirstObservationFrame(-1),
    mnLastObservationFrame(-1), mnObservationCount(0), mnLifecycleType(kStaticWorldMapPoint)
{
    SetWorldPos(Pos);

    Eigen::Vector3f Ow;
    if(pFrame -> Nleft == -1 || idxF < pFrame -> Nleft){
        Ow = pFrame->GetCameraCenter();
    }
    else{
        Eigen::Matrix3f Rwl = pFrame->GetRwc();
        Eigen::Vector3f tlr = pFrame->GetRelativePoseTlr().translation();
        Eigen::Vector3f twl = pFrame->GetOw();

        Ow = Rwl * tlr + twl;
    }
    mNormalVector = mWorldPos - Ow;
    mNormalVector = mNormalVector / mNormalVector.norm();

    Eigen::Vector3f PC = mWorldPos - Ow;
    const float dist = PC.norm();
    const int level = (pFrame -> Nleft == -1) ? pFrame->mvKeysUn[idxF].octave
                                              : (idxF < pFrame -> Nleft) ? pFrame->mvKeys[idxF].octave
                                                                         : pFrame -> mvKeysRight[idxF].octave;
    const float levelScaleFactor =  pFrame->mvScaleFactors[level];
    const int nLevels = pFrame->mnScaleLevels;

    mfMaxDistance = dist*levelScaleFactor;
    mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nLevels-1];

    pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

void MapPoint::SetWorldPos(const Eigen::Vector3f &Pos) {
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    mWorldPos = Pos;
}

Eigen::Vector3f MapPoint::GetWorldPos() {
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos;
}

Eigen::Vector3f MapPoint::GetNormal() {
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector;
}

void MapPoint::SetInstanceId(int id)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnInstanceId = id;
}

int MapPoint::GetInstanceId()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mnInstanceId;
}

void MapPoint::SetSemanticLabel(int label)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnSemanticLabel = label;
}

int MapPoint::GetSemanticLabel()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mnSemanticLabel;
}

void MapPoint::SetLifecycleType(LifecycleType type)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnLifecycleType = static_cast<int>(type);
}

MapPoint::LifecycleType MapPoint::GetLifecycleType()
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mnLifecycleType == static_cast<int>(kDynamicInstanceObservationPoint))
        return kDynamicInstanceObservationPoint;
    if(mnLifecycleType == static_cast<int>(kInstanceStructurePoint))
        return kInstanceStructurePoint;
    return kStaticWorldMapPoint;
}

bool MapPoint::IsDynamicInstanceObservationPoint()
{
    return GetLifecycleType() == kDynamicInstanceObservationPoint;
}

bool MapPoint::IsInstanceStructurePoint()
{
    return GetLifecycleType() == kInstanceStructurePoint;
}

void MapPoint::UpdateObservationStats(int frameId)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mnFirstObservationFrame < 0)
        mnFirstObservationFrame = frameId;
    mnLastObservationFrame = frameId;
    ++mnObservationCount;
}

void MapPoint::SetAdmissionDiagnostics(bool directDynamic,
                                       bool staticNearDynamicBoundary,
                                       long frameId,
                                       long keyFrameId,
                                       int featureIdx,
                                       int boundaryRadiusPx)
{
    unique_lock<mutex> lock(gAdmissionDiagnosticsMutex);
    AdmissionDiagnosticsRecord& record = gAdmissionDiagnostics[this];
    record.directDynamic = directDynamic;
    record.staticNearDynamicBoundary = staticNearDynamicBoundary;
    record.frameId = frameId;
    record.keyFrameId = keyFrameId;
    record.featureIdx = featureIdx;
    record.boundaryRadiusPx = boundaryRadiusPx;
}

bool MapPoint::WasCreatedFromDirectDynamicAdmission()
{
    unique_lock<mutex> lock(gAdmissionDiagnosticsMutex);
    std::map<const MapPoint*, AdmissionDiagnosticsRecord>::const_iterator it =
        gAdmissionDiagnostics.find(this);
    return it != gAdmissionDiagnostics.end() && it->second.directDynamic;
}

bool MapPoint::WasCreatedFromStaticNearDynamicBoundary()
{
    unique_lock<mutex> lock(gAdmissionDiagnosticsMutex);
    std::map<const MapPoint*, AdmissionDiagnosticsRecord>::const_iterator it =
        gAdmissionDiagnostics.find(this);
    return it != gAdmissionDiagnostics.end() && it->second.staticNearDynamicBoundary;
}

long MapPoint::GetAdmissionFrameId()
{
    unique_lock<mutex> lock(gAdmissionDiagnosticsMutex);
    std::map<const MapPoint*, AdmissionDiagnosticsRecord>::const_iterator it =
        gAdmissionDiagnostics.find(this);
    return it != gAdmissionDiagnostics.end() ? it->second.frameId : -1;
}

long MapPoint::GetAdmissionKeyFrameId()
{
    unique_lock<mutex> lock(gAdmissionDiagnosticsMutex);
    std::map<const MapPoint*, AdmissionDiagnosticsRecord>::const_iterator it =
        gAdmissionDiagnostics.find(this);
    return it != gAdmissionDiagnostics.end() ? it->second.keyFrameId : -1;
}

int MapPoint::GetAdmissionFeatureIdx()
{
    unique_lock<mutex> lock(gAdmissionDiagnosticsMutex);
    std::map<const MapPoint*, AdmissionDiagnosticsRecord>::const_iterator it =
        gAdmissionDiagnostics.find(this);
    return it != gAdmissionDiagnostics.end() ? it->second.featureIdx : -1;
}

int MapPoint::GetAdmissionBoundaryRadiusPx()
{
    unique_lock<mutex> lock(gAdmissionDiagnosticsMutex);
    std::map<const MapPoint*, AdmissionDiagnosticsRecord>::const_iterator it =
        gAdmissionDiagnostics.find(this);
    return it != gAdmissionDiagnostics.end() ? it->second.boundaryRadiusPx : 0;
}

void MapPoint::SetScoreAdmissionDiagnostics(double supportScore,
                                            double candidateScore,
                                            double totalScore,
                                            int rawSupport,
                                            int reliableSupport,
                                            int residualSupport,
                                            int depthSupport)
{
    unique_lock<mutex> lock(gAdmissionDiagnosticsMutex);
    AdmissionDiagnosticsRecord& record = gAdmissionDiagnostics[this];
    record.scoreAdmission = true;
    record.supportScore = supportScore;
    record.candidateScore = candidateScore;
    record.totalScore = totalScore;
    record.scoreRawSupport = rawSupport;
    record.scoreReliableSupport = reliableSupport;
    record.scoreResidualSupport = residualSupport;
    record.scoreDepthSupport = depthSupport;
}

void MapPoint::SetScoreAdmissionGeometryDiagnostics(long neighborKeyFrameId,
                                                    int neighborFeatureIdx,
                                                    double baseline,
                                                    double cosParallaxRays,
                                                    double parallaxScore,
                                                    double reprojRatio1,
                                                    double reprojRatio2,
                                                    double scaleScore,
                                                    double finalCandidateScore,
                                                    double finalTotalScore,
                                                    double dist1,
                                                    double dist2,
                                                    double ratioDist,
                                                    double ratioOctave,
                                                    bool stereoPoint,
                                                    bool boundaryCurrent,
                                                    bool boundaryNeighbor)
{
    unique_lock<mutex> lock(gAdmissionDiagnosticsMutex);
    AdmissionDiagnosticsRecord& record = gAdmissionDiagnostics[this];
    record.scoreAdmission = true;
    record.neighborKeyFrameId = neighborKeyFrameId;
    record.neighborFeatureIdx = neighborFeatureIdx;
    record.baseline = baseline;
    record.cosParallaxRays = cosParallaxRays;
    record.parallaxScore = parallaxScore;
    record.reprojRatio1 = reprojRatio1;
    record.reprojRatio2 = reprojRatio2;
    record.scaleScore = scaleScore;
    record.finalCandidateScore = finalCandidateScore;
    record.finalTotalScore = finalTotalScore;
    record.dist1 = dist1;
    record.dist2 = dist2;
    record.ratioDist = ratioDist;
    record.ratioOctave = ratioOctave;
    record.stereoPoint = stereoPoint;
    record.boundaryCurrent = boundaryCurrent;
    record.boundaryNeighbor = boundaryNeighbor;
}

void MapPoint::MarkScoreAdmissionLifecycleEvent(int eventType,
                                                long frameId,
                                                long keyFrameId)
{
    unique_lock<mutex> lock(gAdmissionDiagnosticsMutex);
    AdmissionDiagnosticsRecord& record = gAdmissionDiagnostics[this];
    if(!record.scoreAdmission)
        return;

    switch(eventType)
    {
        case kScoreAdmissionLifecyclePrebad:
            ++record.lifecyclePrebad;
            break;
        case kScoreAdmissionLifecycleFoundRatioCull:
            ++record.lifecycleFoundRatioCull;
            break;
        case kScoreAdmissionLifecycleLowObsCull:
            ++record.lifecycleLowObsCull;
            break;
        case kScoreAdmissionLifecycleV7ResidualCull:
            ++record.lifecycleV7ResidualCull;
            break;
        case kScoreAdmissionLifecycleV7LowUseCull:
            ++record.lifecycleV7LowUseCull;
            break;
        case kScoreAdmissionLifecycleMatured:
            ++record.lifecycleMatured;
            if(record.firstMaturedFrameId < 0)
                record.firstMaturedFrameId = frameId;
            break;
        case kScoreAdmissionLifecycleSurvived:
            ++record.lifecycleSurvived;
            break;
        default:
            return;
    }

    record.lastLifecycleFrameId = frameId;
    record.lastLifecycleKeyFrameId = keyFrameId;
}

bool MapPoint::WasCreatedFromScoreAdmission()
{
    unique_lock<mutex> lock(gAdmissionDiagnosticsMutex);
    std::map<const MapPoint*, AdmissionDiagnosticsRecord>::const_iterator it =
        gAdmissionDiagnostics.find(this);
    return it != gAdmissionDiagnostics.end() && it->second.scoreAdmission;
}

double MapPoint::GetScoreAdmissionSupportScore()
{
    unique_lock<mutex> lock(gAdmissionDiagnosticsMutex);
    std::map<const MapPoint*, AdmissionDiagnosticsRecord>::const_iterator it =
        gAdmissionDiagnostics.find(this);
    return it != gAdmissionDiagnostics.end() ? it->second.supportScore : 0.0;
}

double MapPoint::GetScoreAdmissionCandidateScore()
{
    unique_lock<mutex> lock(gAdmissionDiagnosticsMutex);
    std::map<const MapPoint*, AdmissionDiagnosticsRecord>::const_iterator it =
        gAdmissionDiagnostics.find(this);
    return it != gAdmissionDiagnostics.end() ? it->second.candidateScore : 0.0;
}

double MapPoint::GetScoreAdmissionTotalScore()
{
    unique_lock<mutex> lock(gAdmissionDiagnosticsMutex);
    std::map<const MapPoint*, AdmissionDiagnosticsRecord>::const_iterator it =
        gAdmissionDiagnostics.find(this);
    return it != gAdmissionDiagnostics.end() ? it->second.totalScore : 0.0;
}

void MapPoint::UpdateSupportQualityPoseUse(double chi2, bool inlier)
{
    if(!std::isfinite(chi2))
        return;

    unique_lock<mutex> lock(gSupportQualityPoseUseMutex);
    SupportQualityPoseUseRecord& record = gSupportQualityPoseUse[this];
    ++record.observations;
    if(inlier)
        ++record.inliers;
    record.chi2Sum += chi2;
}

int MapPoint::GetSupportQualityPoseUseCount()
{
    unique_lock<mutex> lock(gSupportQualityPoseUseMutex);
    std::map<const MapPoint*, SupportQualityPoseUseRecord>::const_iterator it =
        gSupportQualityPoseUse.find(this);
    return it != gSupportQualityPoseUse.end() ? it->second.observations : 0;
}

int MapPoint::GetSupportQualityPoseUseInliers()
{
    unique_lock<mutex> lock(gSupportQualityPoseUseMutex);
    std::map<const MapPoint*, SupportQualityPoseUseRecord>::const_iterator it =
        gSupportQualityPoseUse.find(this);
    return it != gSupportQualityPoseUse.end() ? it->second.inliers : 0;
}

double MapPoint::GetSupportQualityPoseUseInlierRate()
{
    unique_lock<mutex> lock(gSupportQualityPoseUseMutex);
    std::map<const MapPoint*, SupportQualityPoseUseRecord>::const_iterator it =
        gSupportQualityPoseUse.find(this);
    if(it == gSupportQualityPoseUse.end() || it->second.observations <= 0)
        return 0.0;
    return static_cast<double>(it->second.inliers) /
           static_cast<double>(it->second.observations);
}

double MapPoint::GetSupportQualityPoseUseMeanChi2()
{
    unique_lock<mutex> lock(gSupportQualityPoseUseMutex);
    std::map<const MapPoint*, SupportQualityPoseUseRecord>::const_iterator it =
        gSupportQualityPoseUse.find(this);
    if(it == gSupportQualityPoseUse.end() || it->second.observations <= 0)
        return 0.0;
    return it->second.chi2Sum / static_cast<double>(it->second.observations);
}

void MapPoint::MarkScoreAdmissionLocalBAWindow()
{
    unique_lock<mutex> lock(gScoreAdmissionConstraintRoleMutex);
    ++gScoreAdmissionConstraintRole[this].localBAWindows;
}

void MapPoint::UpdateScoreAdmissionLocalBAUse(double chi2, bool inlier, bool fixedCamera)
{
    if(!std::isfinite(chi2))
        return;

    unique_lock<mutex> lock(gScoreAdmissionConstraintRoleMutex);
    ScoreAdmissionConstraintRoleRecord& record = gScoreAdmissionConstraintRole[this];
    ++record.localBAEdges;
    if(inlier)
        ++record.localBAInliers;
    if(fixedCamera)
        ++record.localBAFixedEdges;
    else
        ++record.localBALocalEdges;
    record.localBAChi2Sum += chi2;
}

int MapPoint::GetScoreAdmissionLocalBAWindowCount()
{
    unique_lock<mutex> lock(gScoreAdmissionConstraintRoleMutex);
    std::map<const MapPoint*, ScoreAdmissionConstraintRoleRecord>::const_iterator it =
        gScoreAdmissionConstraintRole.find(this);
    return it != gScoreAdmissionConstraintRole.end() ? it->second.localBAWindows : 0;
}

int MapPoint::GetScoreAdmissionLocalBAEdgeCount()
{
    unique_lock<mutex> lock(gScoreAdmissionConstraintRoleMutex);
    std::map<const MapPoint*, ScoreAdmissionConstraintRoleRecord>::const_iterator it =
        gScoreAdmissionConstraintRole.find(this);
    return it != gScoreAdmissionConstraintRole.end() ? it->second.localBAEdges : 0;
}

int MapPoint::GetScoreAdmissionLocalBAInliers()
{
    unique_lock<mutex> lock(gScoreAdmissionConstraintRoleMutex);
    std::map<const MapPoint*, ScoreAdmissionConstraintRoleRecord>::const_iterator it =
        gScoreAdmissionConstraintRole.find(this);
    return it != gScoreAdmissionConstraintRole.end() ? it->second.localBAInliers : 0;
}

int MapPoint::GetScoreAdmissionLocalBAFixedEdges()
{
    unique_lock<mutex> lock(gScoreAdmissionConstraintRoleMutex);
    std::map<const MapPoint*, ScoreAdmissionConstraintRoleRecord>::const_iterator it =
        gScoreAdmissionConstraintRole.find(this);
    return it != gScoreAdmissionConstraintRole.end() ? it->second.localBAFixedEdges : 0;
}

int MapPoint::GetScoreAdmissionLocalBALocalEdges()
{
    unique_lock<mutex> lock(gScoreAdmissionConstraintRoleMutex);
    std::map<const MapPoint*, ScoreAdmissionConstraintRoleRecord>::const_iterator it =
        gScoreAdmissionConstraintRole.find(this);
    return it != gScoreAdmissionConstraintRole.end() ? it->second.localBALocalEdges : 0;
}

double MapPoint::GetScoreAdmissionLocalBAMeanChi2()
{
    unique_lock<mutex> lock(gScoreAdmissionConstraintRoleMutex);
    std::map<const MapPoint*, ScoreAdmissionConstraintRoleRecord>::const_iterator it =
        gScoreAdmissionConstraintRole.find(this);
    if(it == gScoreAdmissionConstraintRole.end() || it->second.localBAEdges <= 0)
        return 0.0;
    return it->second.localBAChi2Sum /
           static_cast<double>(it->second.localBAEdges);
}

void MapPoint::DumpScoreAdmissionLifecycleCsv(const string& filename)
{
    std::vector<std::pair<const MapPoint*, AdmissionDiagnosticsRecord> > records;
    {
        unique_lock<mutex> lock(gAdmissionDiagnosticsMutex);
        records.reserve(gAdmissionDiagnostics.size());
        for(std::map<const MapPoint*, AdmissionDiagnosticsRecord>::const_iterator it =
                gAdmissionDiagnostics.begin();
            it != gAdmissionDiagnostics.end();
            ++it)
        {
            if(it->second.scoreAdmission)
                records.push_back(*it);
        }
    }

    std::map<const MapPoint*, SupportQualityPoseUseRecord> poseUseRecords;
    {
        unique_lock<mutex> lock(gSupportQualityPoseUseMutex);
        poseUseRecords = gSupportQualityPoseUse;
    }

    std::map<const MapPoint*, ScoreAdmissionConstraintRoleRecord> roleRecords;
    {
        unique_lock<mutex> lock(gScoreAdmissionConstraintRoleMutex);
        roleRecords = gScoreAdmissionConstraintRole;
    }

    std::ofstream f(filename.c_str());
    if(!f.is_open())
        return;

    f << std::fixed << std::setprecision(6);
    f << "mp_id,is_bad,first_kf_id,first_frame,admission_frame,admission_kf,"
      << "admission_feature,neighbor_kf,neighbor_feature,boundary_radius_px,"
      << "support_score,candidate_score,total_score,raw_support,reliable_support,"
      << "residual_support,depth_support,geom_baseline,geom_cos_parallax,"
      << "geom_parallax_score,geom_reproj_ratio1,geom_reproj_ratio2,"
      << "geom_scale_score,geom_final_candidate_score,geom_final_total_score,"
      << "geom_dist1,geom_dist2,geom_ratio_dist,geom_ratio_octave,geom_stereo,"
      << "geom_boundary_current,geom_boundary_neighbor,observations,found,visible,"
      << "found_ratio,pose_use_edges,pose_use_inliers,pose_use_inlier_rate,"
      << "pose_use_chi2_mean,lba_windows,lba_edges,lba_inliers,lba_inlier_rate,"
      << "lba_local_edges,lba_fixed_edges,lba_local_edge_rate,lba_fixed_edge_rate,"
      << "lba_chi2_mean,lifecycle_prebad,lifecycle_found_ratio_cull,"
      << "lifecycle_low_obs_cull,lifecycle_v7_residual_cull,"
      << "lifecycle_v7_low_use_cull,lifecycle_matured,lifecycle_survived,"
      << "first_matured_frame,last_lifecycle_frame,last_lifecycle_kf,"
      << "ref_distance\n";

    for(size_t i = 0; i < records.size(); ++i)
    {
        const MapPoint* pMP = records[i].first;
        const AdmissionDiagnosticsRecord& record = records[i].second;
        if(!pMP)
            continue;
        MapPoint* pMutableMP = const_cast<MapPoint*>(pMP);

        bool isBad = false;
        int found = 0;
        int visible = 0;
        {
            unique_lock<mutex> lock(pMutableMP->mMutexFeatures);
            isBad = pMutableMP->mbBad;
            found = pMutableMP->mnFound;
            visible = pMutableMP->mnVisible;
        }

        const int observations = pMutableMP->Observations();
        const double foundRatio =
            visible > 0 ? static_cast<double>(found) / static_cast<double>(visible) : 0.0;

        SupportQualityPoseUseRecord poseUse;
        std::map<const MapPoint*, SupportQualityPoseUseRecord>::const_iterator poseIt =
            poseUseRecords.find(pMP);
        if(poseIt != poseUseRecords.end())
            poseUse = poseIt->second;
        const double poseUseInlierRate =
            poseUse.observations > 0 ?
                static_cast<double>(poseUse.inliers) /
                    static_cast<double>(poseUse.observations) :
                0.0;
        const double poseUseMeanChi2 =
            poseUse.observations > 0 ?
                poseUse.chi2Sum / static_cast<double>(poseUse.observations) :
                0.0;

        ScoreAdmissionConstraintRoleRecord role;
        std::map<const MapPoint*, ScoreAdmissionConstraintRoleRecord>::const_iterator roleIt =
            roleRecords.find(pMP);
        if(roleIt != roleRecords.end())
            role = roleIt->second;
        const double lbaInlierRate =
            role.localBAEdges > 0 ?
                static_cast<double>(role.localBAInliers) /
                    static_cast<double>(role.localBAEdges) :
                0.0;
        const double lbaLocalEdgeRate =
            role.localBAEdges > 0 ?
                static_cast<double>(role.localBALocalEdges) /
                    static_cast<double>(role.localBAEdges) :
                0.0;
        const double lbaFixedEdgeRate =
            role.localBAEdges > 0 ?
                static_cast<double>(role.localBAFixedEdges) /
                    static_cast<double>(role.localBAEdges) :
                0.0;
        const double lbaMeanChi2 =
            role.localBAEdges > 0 ?
                role.localBAChi2Sum / static_cast<double>(role.localBAEdges) :
                0.0;

        double refDistance = 0.0;
        KeyFrame* pRefKF = pMutableMP->GetReferenceKeyFrame();
        if(pRefKF && !pRefKF->isBad())
        {
            refDistance =
                static_cast<double>((pMutableMP->GetWorldPos() -
                                     pRefKF->GetCameraCenter()).norm());
        }

        f << pMutableMP->mnId << ','
          << (isBad ? 1 : 0) << ','
          << pMutableMP->mnFirstKFid << ','
          << pMutableMP->mnFirstFrame << ','
          << record.frameId << ','
          << record.keyFrameId << ','
          << record.featureIdx << ','
          << record.neighborKeyFrameId << ','
          << record.neighborFeatureIdx << ','
          << record.boundaryRadiusPx << ','
          << record.supportScore << ','
          << record.candidateScore << ','
          << record.totalScore << ','
          << record.scoreRawSupport << ','
          << record.scoreReliableSupport << ','
          << record.scoreResidualSupport << ','
          << record.scoreDepthSupport << ','
          << record.baseline << ','
          << record.cosParallaxRays << ','
          << record.parallaxScore << ','
          << record.reprojRatio1 << ','
          << record.reprojRatio2 << ','
          << record.scaleScore << ','
          << record.finalCandidateScore << ','
          << record.finalTotalScore << ','
          << record.dist1 << ','
          << record.dist2 << ','
          << record.ratioDist << ','
          << record.ratioOctave << ','
          << (record.stereoPoint ? 1 : 0) << ','
          << (record.boundaryCurrent ? 1 : 0) << ','
          << (record.boundaryNeighbor ? 1 : 0) << ','
          << observations << ','
          << found << ','
          << visible << ','
          << foundRatio << ','
          << poseUse.observations << ','
          << poseUse.inliers << ','
          << poseUseInlierRate << ','
          << poseUseMeanChi2 << ','
          << role.localBAWindows << ','
          << role.localBAEdges << ','
          << role.localBAInliers << ','
          << lbaInlierRate << ','
          << role.localBALocalEdges << ','
          << role.localBAFixedEdges << ','
          << lbaLocalEdgeRate << ','
          << lbaFixedEdgeRate << ','
          << lbaMeanChi2 << ','
          << record.lifecyclePrebad << ','
          << record.lifecycleFoundRatioCull << ','
          << record.lifecycleLowObsCull << ','
          << record.lifecycleV7ResidualCull << ','
          << record.lifecycleV7LowUseCull << ','
          << record.lifecycleMatured << ','
          << record.lifecycleSurvived << ','
          << record.firstMaturedFrameId << ','
          << record.lastLifecycleFrameId << ','
          << record.lastLifecycleKeyFrameId << ','
          << refDistance << '\n';
    }
}


KeyFrame* MapPoint::GetReferenceKeyFrame()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
}

void MapPoint::AddObservation(KeyFrame* pKF, int idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    tuple<int,int> indexes;

    if(mObservations.count(pKF)){
        indexes = mObservations[pKF];
    }
    else{
        indexes = tuple<int,int>(-1,-1);
    }

    if(pKF -> NLeft != -1 && idx >= pKF -> NLeft){
        get<1>(indexes) = idx;
    }
    else{
        get<0>(indexes) = idx;
    }

    mObservations[pKF]=indexes;

    if(!pKF->mpCamera2 && pKF->mvuRight[idx]>=0)
        nObs+=2;
    else
        nObs++;
}

void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad=false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
        {
            tuple<int,int> indexes = mObservations[pKF];
            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

            if(leftIndex != -1){
                if(!pKF->mpCamera2 && pKF->mvuRight[leftIndex]>=0)
                    nObs-=2;
                else
                    nObs--;
            }
            if(rightIndex != -1){
                nObs--;
            }

            mObservations.erase(pKF);

            if(mpRefKF==pKF)
                mpRefKF=mObservations.begin()->first;

            // If only 2 observations or less, discard point
            if(nObs<=2)
                bBad=true;
        }
    }

    if(bBad)
        SetBadFlag();
}


std::map<KeyFrame*, std::tuple<int,int>>  MapPoint::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}

int MapPoint::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

void MapPoint::SetBadFlag()
{
    map<KeyFrame*, tuple<int,int>> obs;
    Map* pMap = NULL;
    int instanceId = -1;
    int semanticLabel = 0;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad=true;
        obs = mObservations;
        pMap = mpMap;
        instanceId = mnInstanceId;
        semanticLabel = mnSemanticLabel;
        mObservations.clear();
    }
    for(map<KeyFrame*, tuple<int,int>>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        int leftIndex = get<0>(mit -> second), rightIndex = get<1>(mit -> second);
        if(leftIndex != -1){
            pKF->EraseMapPointMatch(leftIndex);
        }
        if(rightIndex != -1){
            pKF->EraseMapPointMatch(rightIndex);
        }
    }

    CleanupInstanceMembership(pMap,
                              instanceId,
                              semanticLabel,
                              this,
                              static_cast<MapPoint*>(NULL),
                              false);

    if(pMap)
        pMap->EraseMapPoint(this);
}

MapPoint* MapPoint::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}

void MapPoint::Replace(MapPoint* pMP)
{
    if(pMP->mnId==this->mnId)
        return;

    int nvisible, nfound;
    map<KeyFrame*,tuple<int,int>> obs;
    Map* pMap = NULL;
    int instanceId = -1;
    int semanticLabel = 0;
    bool transferStructureMembership = false;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        obs=mObservations;
        mObservations.clear();
        mbBad=true;
        nvisible = mnVisible;
        nfound = mnFound;
        pMap = mpMap;
        instanceId = mnInstanceId;
        semanticLabel = mnSemanticLabel;
        transferStructureMembership =
            mnLifecycleType == static_cast<int>(MapPoint::kInstanceStructurePoint);
        mpReplaced = pMP;
    }

    for(map<KeyFrame*,tuple<int,int>>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        // Replace measurement in keyframe
        KeyFrame* pKF = mit->first;

        tuple<int,int> indexes = mit -> second;
        int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

        if(!pMP->IsInKeyFrame(pKF))
        {
            if(leftIndex != -1){
                pKF->ReplaceMapPointMatch(leftIndex, pMP);
                pMP->AddObservation(pKF,leftIndex);
            }
            if(rightIndex != -1){
                pKF->ReplaceMapPointMatch(rightIndex, pMP);
                pMP->AddObservation(pKF,rightIndex);
            }
        }
        else
        {
            if(leftIndex != -1){
                pKF->EraseMapPointMatch(leftIndex);
            }
            if(rightIndex != -1){
                pKF->EraseMapPointMatch(rightIndex);
            }
        }
    }
    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
    pMP->ComputeDistinctiveDescriptors();

    CleanupInstanceMembership(pMap,
                              instanceId,
                              semanticLabel,
                              this,
                              pMP,
                              transferStructureMembership);

    if(pMap)
        pMap->EraseMapPoint(this);
}

bool MapPoint::isBad()
{
    unique_lock<mutex> lock1(mMutexFeatures,std::defer_lock);
    unique_lock<mutex> lock2(mMutexPos,std::defer_lock);
    lock(lock1, lock2);

    return mbBad;
}

void MapPoint::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible+=n;
}

void MapPoint::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound+=n;
}

float MapPoint::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound)/mnVisible;
}

void MapPoint::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;

    map<KeyFrame*,tuple<int,int>> observations;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if(mbBad)
            return;
        observations=mObservations;
    }

    if(observations.empty())
        return;

    vDescriptors.reserve(observations.size());

    for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        if(!pKF->isBad()){
            tuple<int,int> indexes = mit -> second;
            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

            if(leftIndex != -1){
                vDescriptors.push_back(pKF->mDescriptors.row(leftIndex));
            }
            if(rightIndex != -1){
                vDescriptors.push_back(pKF->mDescriptors.row(rightIndex));
            }
        }
    }

    if(vDescriptors.empty())
        return;

    // Compute distances between them
    const size_t N = vDescriptors.size();

    float Distances[N][N];
    for(size_t i=0;i<N;i++)
    {
        Distances[i][i]=0;
        for(size_t j=i+1;j<N;j++)
        {
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for(size_t i=0;i<N;i++)
    {
        vector<int> vDists(Distances[i],Distances[i]+N);
        sort(vDists.begin(),vDists.end());
        int median = vDists[0.5*(N-1)];

        if(median<BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();
    }
}

cv::Mat MapPoint::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

tuple<int,int> MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF];
    else
        return tuple<int,int>(-1,-1);
}

bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}

void MapPoint::UpdateNormalAndDepth()
{
    map<KeyFrame*,tuple<int,int>> observations;
    KeyFrame* pRefKF;
    Eigen::Vector3f Pos;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if(mbBad)
            return;
        observations = mObservations;
        pRefKF = mpRefKF;
        Pos = mWorldPos;
    }

    if(observations.empty())
        return;

    Eigen::Vector3f normal;
    normal.setZero();
    int n=0;
    for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        tuple<int,int> indexes = mit -> second;
        int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

        if(leftIndex != -1){
            Eigen::Vector3f Owi = pKF->GetCameraCenter();
            Eigen::Vector3f normali = Pos - Owi;
            normal = normal + normali / normali.norm();
            n++;
        }
        if(rightIndex != -1){
            Eigen::Vector3f Owi = pKF->GetRightCameraCenter();
            Eigen::Vector3f normali = Pos - Owi;
            normal = normal + normali / normali.norm();
            n++;
        }
    }

    Eigen::Vector3f PC = Pos - pRefKF->GetCameraCenter();
    const float dist = PC.norm();

    tuple<int ,int> indexes = observations[pRefKF];
    int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
    int level;
    if(pRefKF -> NLeft == -1){
        level = pRefKF->mvKeysUn[leftIndex].octave;
    }
    else if(leftIndex != -1){
        level = pRefKF -> mvKeys[leftIndex].octave;
    }
    else{
        level = pRefKF -> mvKeysRight[rightIndex - pRefKF -> NLeft].octave;
    }

    //const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
    const float levelScaleFactor =  pRefKF->mvScaleFactors[level];
    const int nLevels = pRefKF->mnScaleLevels;

    {
        unique_lock<mutex> lock3(mMutexPos);
        mfMaxDistance = dist*levelScaleFactor;
        mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1];
        mNormalVector = normal/n;
    }
}

void MapPoint::SetNormalVector(const Eigen::Vector3f& normal)
{
    unique_lock<mutex> lock3(mMutexPos);
    mNormalVector = normal;
}

float MapPoint::GetMinDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f * mfMinDistance;
}

float MapPoint::GetMaxDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f * mfMaxDistance;
}

int MapPoint::PredictScale(const float &currentDist, KeyFrame* pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}

int MapPoint::PredictScale(const float &currentDist, Frame* pF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pF->mnScaleLevels)
        nScale = pF->mnScaleLevels-1;

    return nScale;
}

void MapPoint::PrintObservations()
{
    cout << "MP_OBS: MP " << mnId << endl;
    for(map<KeyFrame*,tuple<int,int>>::iterator mit=mObservations.begin(), mend=mObservations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKFi = mit->first;
        tuple<int,int> indexes = mit->second;
        int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
        cout << "--OBS in KF " << pKFi->mnId << " in map " << pKFi->GetMap()->GetId() << endl;
    }
}

Map* MapPoint::GetMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mpMap;
}

void MapPoint::UpdateMap(Map* pMap)
{
    unique_lock<mutex> lock(mMutexMap);
    mpMap = pMap;
}

void MapPoint::PreSave(set<KeyFrame*>& spKF,set<MapPoint*>& spMP)
{
    mBackupReplacedId = -1;
    if(mpReplaced && spMP.find(mpReplaced) != spMP.end())
        mBackupReplacedId = mpReplaced->mnId;

    mBackupObservationsId1.clear();
    mBackupObservationsId2.clear();
    // Save the id and position in each KF who view it
    for(std::map<KeyFrame*,std::tuple<int,int> >::const_iterator it = mObservations.begin(), end = mObservations.end(); it != end; ++it)
    {
        KeyFrame* pKFi = it->first;
        if(spKF.find(pKFi) != spKF.end())
        {
            mBackupObservationsId1[it->first->mnId] = get<0>(it->second);
            mBackupObservationsId2[it->first->mnId] = get<1>(it->second);
        }
        else
        {
            EraseObservation(pKFi);
        }
    }

    // Save the id of the reference KF
    if(spKF.find(mpRefKF) != spKF.end())
    {
        mBackupRefKFId = mpRefKF->mnId;
    }
}

void MapPoint::PostLoad(map<long unsigned int, KeyFrame*>& mpKFid, map<long unsigned int, MapPoint*>& mpMPid)
{
    mpRefKF = mpKFid[mBackupRefKFId];
    if(!mpRefKF)
    {
        cout << "ERROR: MP without KF reference " << mBackupRefKFId << "; Num obs: " << nObs << endl;
    }
    mpReplaced = static_cast<MapPoint*>(NULL);
    if(mBackupReplacedId>=0)
    {
        map<long unsigned int, MapPoint*>::iterator it = mpMPid.find(mBackupReplacedId);
        if (it != mpMPid.end())
            mpReplaced = it->second;
    }

    mObservations.clear();

    for(map<long unsigned int, int>::const_iterator it = mBackupObservationsId1.begin(), end = mBackupObservationsId1.end(); it != end; ++it)
    {
        KeyFrame* pKFi = mpKFid[it->first];
        map<long unsigned int, int>::const_iterator it2 = mBackupObservationsId2.find(it->first);
        std::tuple<int, int> indexes = tuple<int,int>(it->second,it2->second);
        if(pKFi)
        {
           mObservations[pKFi] = indexes;
        }
    }

    mBackupObservationsId1.clear();
    mBackupObservationsId2.clear();
}

} //namespace ORB_SLAM
