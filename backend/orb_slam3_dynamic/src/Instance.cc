#include "Instance.h"

#include "MapPoint.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <vector>

namespace ORB_SLAM3
{

namespace
{

bool IsFiniteSE3(const Sophus::SE3f& pose)
{
    return pose.matrix3x4().allFinite();
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

double MotionRotationDeg(const Sophus::SE3f& motion)
{
    const Eigen::AngleAxisd angleAxis(motion.rotationMatrix().cast<double>());
    return std::abs(angleAxis.angle()) * 180.0 / 3.14159265358979323846;
}

bool IsSemanticZeroVelocity(const Sophus::SE3f& motion)
{
    if(!IsFiniteSE3(motion))
        return false;

    const double maxTranslation =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_ENTITY_ZERO_TRANSLATION", 0.03, 0.0);
    const double maxRotationDeg =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_ENTITY_ZERO_ROTATION_DEG", 2.0, 0.0);
    return motion.translation().cast<double>().norm() <= maxTranslation &&
           MotionRotationDeg(motion) <= maxRotationDeg;
}

bool IsArticulatedSemanticLabel(const int semanticLabel)
{
    const int articulatedSemanticLabel = static_cast<int>(
        GetEnvDoubleOrDefault("STSLAM_ARTICULATED_SEMANTIC_LABEL", 11.0, 0.0));
    return articulatedSemanticLabel > 0 && semanticLabel == articulatedSemanticLabel;
}

Sophus::SE3f CanonicalVelocityForState(const Sophus::SE3f& velocity,
                                       Instance::DynamicEntityMotionState state)
{
    if(state == Instance::kZeroVelocityDynamicEntity)
        return Sophus::SE3f();
    return IsFiniteSE3(velocity) ? velocity : Sophus::SE3f();
}

Instance::DynamicEntityMotionState ClassifySemanticMotionState(const Sophus::SE3f& velocity,
                                                              bool reliable,
                                                              int semanticLabel)
{
    if(!IsFiniteSE3(velocity))
        return Instance::kUncertainDynamicEntity;

    const double maxTranslation =
        GetEnvDoubleOrDefault("STSLAM_DYNAMIC_ENTITY_ZERO_TRANSLATION", 0.03, 0.0);
    if(IsArticulatedSemanticLabel(semanticLabel) &&
       velocity.translation().cast<double>().norm() <= maxTranslation)
    {
        return Instance::kZeroVelocityDynamicEntity;
    }

    if(IsSemanticZeroVelocity(velocity))
        return Instance::kZeroVelocityDynamicEntity;
    return reliable ? Instance::kMovingDynamicEntity : Instance::kUncertainDynamicEntity;
}

int GetBackendZeroConfirmFrames()
{
    return static_cast<int>(
        GetEnvDoubleOrDefault("STSLAM_BACKEND_ZERO_CONFIRM_FRAMES", 1.0, 1.0));
}

int GetBackendMovingConfirmFrames(int fallbackConfirmFrames)
{
    return static_cast<int>(
        GetEnvDoubleOrDefault("STSLAM_BACKEND_MOVING_CONFIRM_FRAMES",
                              static_cast<double>(std::max(1, fallbackConfirmFrames)),
                              1.0));
}

int GetBackendUncertainConfirmFrames()
{
    return static_cast<int>(
        GetEnvDoubleOrDefault("STSLAM_BACKEND_UNCERTAIN_CONFIRM_FRAMES", 2.0, 1.0));
}

Sophus::SE3f DecayMotionTowardIdentity(const Sophus::SE3f& motion, float keepRatio)
{
    if(!IsFiniteSE3(motion))
        return Sophus::SE3f();

    keepRatio = std::max(0.0f, std::min(1.0f, keepRatio));
    if(keepRatio <= 0.0f)
        return Sophus::SE3f();
    if(keepRatio >= 1.0f)
        return motion;

    Sophus::SE3f::Tangent tangent = motion.log();
    if(!tangent.allFinite())
        return Sophus::SE3f();
    tangent *= keepRatio;

    Sophus::SE3f decayed = Sophus::SE3f::exp(tangent);
    if(!IsFiniteSE3(decayed))
        return Sophus::SE3f();

    const Eigen::AngleAxisf angleAxis(decayed.rotationMatrix());
    if(decayed.translation().norm() < 1e-4f && std::abs(angleAxis.angle()) < 1e-4f)
        return Sophus::SE3f();

    return decayed;
}

} // namespace

Instance::Instance()
    : mnInstanceId(-1), mnSemanticLabel(0), mLastPose(Sophus::SE3f()),
      mVelocity(Sophus::SE3f()), mbInitialized(false), mbInitializationMotionReliable(true),
      mnInitFrameCount(0),
      mnInitializedFrame(-1), mnLastSeenFrame(-1), mnLastPoseProxyKFId(-1),
      mnStaticMotionEvidence(0), mnDynamicMotionEvidence(0), mnUncertainMotionEvidence(0),
      mnLastMotionGateFrame(-1), mnBackendMotionEvidence(0),
      mnBackendZeroMotionEvidence(0), mnBackendMovingMotionEvidence(0),
      mnBackendUncertainMotionEvidence(0), mnLastBackendMotionFrame(-1),
      mnDynamicSupplyGateFailureEvidence(0), mnLastDynamicSupplyGateFrame(-1),
      mCurrentDynamicEntityMotionState(kDynamicEntityUnknown),
      mLastBackendObservedVelocity(Sophus::SE3f())
{
}

Instance::Instance(int instanceId, int semanticLabel)
    : mnInstanceId(instanceId), mnSemanticLabel(semanticLabel),
      mLastPose(Sophus::SE3f()), mVelocity(Sophus::SE3f()),
      mbInitialized(false), mbInitializationMotionReliable(true),
      mnInitFrameCount(0), mnInitializedFrame(-1), mnLastSeenFrame(-1),
      mnLastPoseProxyKFId(-1), mnStaticMotionEvidence(0), mnDynamicMotionEvidence(0),
      mnUncertainMotionEvidence(0), mnLastMotionGateFrame(-1), mnBackendMotionEvidence(0),
      mnBackendZeroMotionEvidence(0), mnBackendMovingMotionEvidence(0),
      mnBackendUncertainMotionEvidence(0), mnLastBackendMotionFrame(-1),
      mnDynamicSupplyGateFailureEvidence(0),
      mnLastDynamicSupplyGateFrame(-1),
      mCurrentDynamicEntityMotionState(kDynamicEntityUnknown),
      mLastBackendObservedVelocity(Sophus::SE3f())
{
}

int Instance::GetId() const
{
    return mnInstanceId;
}

int Instance::GetSemanticLabel() const
{
    return mnSemanticLabel;
}

void Instance::SetSemanticLabel(int semanticLabel)
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    mnSemanticLabel = semanticLabel;
}

void Instance::AddMapPoint(MapPoint* pMP)
{
    if(!pMP || pMP->IsDynamicInstanceObservationPoint())
        return;

    pMP->SetLifecycleType(MapPoint::kInstanceStructurePoint);

    std::unique_lock<std::mutex> lock(mMutexInstance);
    mspMapPoints.insert(pMP);
}

void Instance::RemoveMapPoint(MapPoint* pMP)
{
    if(!pMP)
        return;

    std::unique_lock<std::mutex> lock(mMutexInstance);
    mspMapPoints.erase(pMP);
}

std::set<MapPoint*> Instance::GetMapPoints() const
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    return mspMapPoints;
}

size_t Instance::NumMapPoints() const
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    return mspMapPoints.size();
}

void Instance::UpdateMotionPrior(KeyFrame* pKF, const Sophus::SE3f& motion)
{
    if(!pKF || !IsFiniteSE3(motion))
        return;

    std::unique_lock<std::mutex> lock(mMutexInstance);
    mmKeyframeMotionPriors[pKF] = motion;
}

Sophus::SE3f Instance::GetMotionPriorForKeyFrame(KeyFrame* pKF) const
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    const auto it = mmKeyframeMotionPriors.find(pKF);
    if(it == mmKeyframeMotionPriors.end())
        return Sophus::SE3f();
    return it->second;
}

void Instance::UpdatePoseProxy(KeyFrame* pKF, const Sophus::SE3f& poseProxy)
{
    if(!pKF || !IsFiniteSE3(poseProxy))
        return;

    std::unique_lock<std::mutex> lock(mMutexInstance);
    mmKeyframePoseProxy[pKF] = poseProxy;
    if(pKF->mnId >= mnLastPoseProxyKFId)
    {
        mLastPose = poseProxy;
        mnLastPoseProxyKFId = static_cast<int>(pKF->mnId);
    }
    if(pKF->mnFrameId >= 0)
    {
        InstanceMotionStateRecord& record =
            mmMotionStatesByFrame[static_cast<unsigned long>(pKF->mnFrameId)];
        record.frameId = static_cast<unsigned long>(pKF->mnFrameId);
        record.pose = poseProxy;
        record.velocity = mVelocity;
        record.state = mCurrentDynamicEntityMotionState;
        record.confidence = std::max(record.confidence, 0.5);
        record.reliable = record.reliable || mbInitializationMotionReliable;
    }
}

Sophus::SE3f Instance::GetPoseProxyForKeyFrame(KeyFrame* pKF) const
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    const auto it = mmKeyframePoseProxy.find(pKF);
    if(it == mmKeyframePoseProxy.end())
        return mLastPose;
    return it->second;
}

void Instance::PredictPose(const Sophus::SE3f& velocity)
{
    if(!IsFiniteSE3(velocity))
        return;

    std::unique_lock<std::mutex> lock(mMutexInstance);
    if(!IsFiniteSE3(mLastPose))
        return;

    const Sophus::SE3f predictedPose = velocity * mLastPose;
    if(!IsFiniteSE3(predictedPose))
        return;

    mVelocity = velocity;
    mLastPose = predictedPose;
}

int Instance::MarkSeenInFrame(int frameId)
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    const int previousSeenFrame = mnLastSeenFrame;
    mnLastSeenFrame = frameId;
    return previousSeenFrame;
}

void Instance::ResetInitializationCounter()
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    mnInitFrameCount = 0;
}

int Instance::AdvanceInitializationCounter(bool consecutive)
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    if(consecutive)
        ++mnInitFrameCount;
    else
        mnInitFrameCount = 1;
    return mnInitFrameCount;
}

bool Instance::IsInitialized() const
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    return mbInitialized;
}

int Instance::GetInitializedFrame() const
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    return mnInitializedFrame;
}

Sophus::SE3f Instance::GetVelocity() const
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    return mVelocity;
}

Sophus::SE3f Instance::GetLastPoseEstimate() const
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    return mLastPose;
}

bool Instance::HasReliableInitializationMotion() const
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    return mbInitializationMotionReliable;
}

void Instance::UpdateVelocityEstimate(const Sophus::SE3f& velocity)
{
    if(!IsFiniteSE3(velocity))
        return;

    std::unique_lock<std::mutex> lock(mMutexInstance);
    mVelocity = velocity;
    if(!mmMotionStatesByFrame.empty())
    {
        InstanceMotionStateRecord& record = mmMotionStatesByFrame.rbegin()->second;
        record.velocity = velocity;
        if(IsFiniteSE3(record.pose))
            mLastPose = record.pose;
    }
}

void Instance::SetInstanceMotionState(unsigned long frameId,
                                      const Sophus::SE3f& pose,
                                      const Sophus::SE3f& velocity,
                                      DynamicEntityMotionState state,
                                      double confidence,
                                      bool reliable)
{
    if(!IsFiniteSE3(pose) || !IsFiniteSE3(velocity) || !std::isfinite(confidence))
        return;

    confidence = std::max(0.0, std::min(1.0, confidence));
    const Sophus::SE3f canonicalVelocity =
        CanonicalVelocityForState(velocity, state);
    std::unique_lock<std::mutex> lock(mMutexInstance);

    InstanceMotionStateRecord& record = mmMotionStatesByFrame[frameId];
    record.frameId = frameId;
    record.pose = pose;
    record.velocity = canonicalVelocity;
    record.state = state;
    record.confidence = confidence;
    record.reliable = reliable;

    if(mmMotionStatesByFrame.rbegin()->first == frameId)
    {
        mLastPose = pose;
        mVelocity = canonicalVelocity;
        mCurrentDynamicEntityMotionState = state;
        mbInitializationMotionReliable = mbInitializationMotionReliable || reliable;
    }
}

bool Instance::GetInstanceMotionState(unsigned long frameId,
                                      InstanceMotionStateRecord& record) const
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    const std::map<unsigned long, InstanceMotionStateRecord>::const_iterator it =
        mmMotionStatesByFrame.find(frameId);
    if(it == mmMotionStatesByFrame.end())
        return false;

    record = it->second;
    return true;
}

bool Instance::GetLatestInstanceMotionState(InstanceMotionStateRecord& record) const
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    if(mmMotionStatesByFrame.empty())
        return false;

    record = mmMotionStatesByFrame.rbegin()->second;
    return true;
}

Instance::DynamicEntityMotionState Instance::GetCurrentDynamicEntityMotionState() const
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    return mCurrentDynamicEntityMotionState;
}

bool Instance::RecordBackendMotionObservation(const Sophus::SE3f& observedVelocity,
                                              bool backendMature,
                                              int frameId,
                                              float maxImmatureTranslation,
                                              float maxImmatureRotationDeg,
                                              int immatureConfirmFrames)
{
    if(!IsFiniteSE3(observedVelocity))
        return false;

    const float translationNorm = observedVelocity.translation().norm();
    const Eigen::AngleAxisf angleAxis(observedVelocity.rotationMatrix());
    const float rotationDeg =
        std::abs(angleAxis.angle()) * 180.0f / 3.14159265358979323846f;
    maxImmatureTranslation = std::max(0.0f, maxImmatureTranslation);
    maxImmatureRotationDeg = std::max(0.0f, maxImmatureRotationDeg);
    immatureConfirmFrames = std::max(1, immatureConfirmFrames);

    std::unique_lock<std::mutex> lock(mMutexInstance);

    DynamicEntityMotionState candidateState =
        ClassifySemanticMotionState(observedVelocity, backendMature, mnSemanticLabel);
    const bool withinMagnitudeGate =
        translationNorm <= maxImmatureTranslation &&
        rotationDeg <= maxImmatureRotationDeg;
    if(!backendMature && !withinMagnitudeGate)
        candidateState = kUncertainDynamicEntity;

    if(mnLastBackendMotionFrame >= 0 && frameId > mnLastBackendMotionFrame + 1)
    {
        mnBackendZeroMotionEvidence = 0;
        mnBackendMovingMotionEvidence = 0;
        mnBackendUncertainMotionEvidence = 0;
    }

    if(frameId != mnLastBackendMotionFrame)
    {
        if(candidateState == kZeroVelocityDynamicEntity)
        {
            ++mnBackendZeroMotionEvidence;
            mnBackendMovingMotionEvidence = std::max(0, mnBackendMovingMotionEvidence - 1);
            mnBackendUncertainMotionEvidence = 0;
        }
        else if(candidateState == kMovingDynamicEntity)
        {
            ++mnBackendMovingMotionEvidence;
            mnBackendZeroMotionEvidence = std::max(0, mnBackendZeroMotionEvidence - 1);
            mnBackendUncertainMotionEvidence = 0;
        }
        else
        {
            ++mnBackendUncertainMotionEvidence;
            mnBackendZeroMotionEvidence = std::max(0, mnBackendZeroMotionEvidence - 1);
            mnBackendMovingMotionEvidence = std::max(0, mnBackendMovingMotionEvidence - 1);
        }
    }

    mnLastBackendMotionFrame = frameId;
    mLastBackendObservedVelocity = observedVelocity;
    mnBackendMotionEvidence = std::max(mnBackendZeroMotionEvidence,
                              std::max(mnBackendMovingMotionEvidence,
                                       mnBackendUncertainMotionEvidence));

    DynamicEntityMotionState committedState = kDynamicEntityUnknown;
    if(mnBackendZeroMotionEvidence >= GetBackendZeroConfirmFrames())
        committedState = kZeroVelocityDynamicEntity;
    else if(mnBackendMovingMotionEvidence >= GetBackendMovingConfirmFrames(immatureConfirmFrames))
        committedState = kMovingDynamicEntity;
    else if(mnBackendUncertainMotionEvidence >= GetBackendUncertainConfirmFrames())
        committedState = kUncertainDynamicEntity;

    if(committedState != kDynamicEntityUnknown)
    {
        const Sophus::SE3f canonicalVelocity =
            CanonicalVelocityForState(observedVelocity, committedState);
        mVelocity = canonicalVelocity;
        mCurrentDynamicEntityMotionState = committedState;
        if(frameId >= 0)
        {
            InstanceMotionStateRecord& record =
                mmMotionStatesByFrame[static_cast<unsigned long>(frameId)];
            record.frameId = static_cast<unsigned long>(frameId);
            record.pose = mLastPose;
            record.velocity = canonicalVelocity;
            record.state = mCurrentDynamicEntityMotionState;
            record.confidence = backendMature ? 1.0 :
                (committedState == kZeroVelocityDynamicEntity ? 0.7 : 0.75);
            record.reliable = backendMature ||
                              committedState == kZeroVelocityDynamicEntity;
        }
        return committedState != kUncertainDynamicEntity;
    }

    if(frameId >= 0 &&
       (candidateState == kMovingDynamicEntity ||
        candidateState == kUncertainDynamicEntity))
    {
        const Sophus::SE3f tentativeVelocity =
            IsFiniteSE3(observedVelocity) ? observedVelocity : Sophus::SE3f();
        mVelocity = tentativeVelocity;
        mCurrentDynamicEntityMotionState = kUncertainDynamicEntity;

        InstanceMotionStateRecord& record =
            mmMotionStatesByFrame[static_cast<unsigned long>(frameId)];
        record.frameId = static_cast<unsigned long>(frameId);
        record.pose = mLastPose;
        record.velocity = tentativeVelocity;
        record.state = kUncertainDynamicEntity;
        record.confidence = candidateState == kMovingDynamicEntity ? 0.45 : 0.35;
        record.reliable = false;
    }

    return false;
}

void Instance::RecordMotionGateState(int motionState,
                                     const Sophus::SE3f& observedVelocity,
                                     int frameId,
                                     float staticVelocityDecay)
{
    std::unique_lock<std::mutex> lock(mMutexInstance);

    if(frameId >= 0 && frameId == mnLastMotionGateFrame)
        return;

    if(mnLastMotionGateFrame >= 0 && frameId > mnLastMotionGateFrame + 1)
    {
        mnStaticMotionEvidence = 0;
        mnDynamicMotionEvidence = 0;
        mnUncertainMotionEvidence = 0;
    }
    mnLastMotionGateFrame = frameId;

    if(motionState == 2)
    {
        ++mnDynamicMotionEvidence;
        mnStaticMotionEvidence = std::max(0, mnStaticMotionEvidence - 1);
        mnUncertainMotionEvidence = 0;
        if(IsFiniteSE3(observedVelocity))
            mVelocity = observedVelocity;
        mCurrentDynamicEntityMotionState = kMovingDynamicEntity;
        if(frameId >= 0)
        {
            InstanceMotionStateRecord& record =
                mmMotionStatesByFrame[static_cast<unsigned long>(frameId)];
            record.frameId = static_cast<unsigned long>(frameId);
            record.pose = mLastPose;
            record.velocity = mVelocity;
            record.state = kMovingDynamicEntity;
            record.confidence = std::max(record.confidence, 0.7);
            record.reliable = record.reliable || mbInitializationMotionReliable;
        }
        return;
    }

    if(motionState == 1)
    {
        ++mnStaticMotionEvidence;
        mnDynamicMotionEvidence = std::max(0, mnDynamicMotionEvidence - 1);
        mnUncertainMotionEvidence = 0;
        mVelocity = DecayMotionTowardIdentity(mVelocity, staticVelocityDecay);
        mCurrentDynamicEntityMotionState = kZeroVelocityDynamicEntity;
        if(frameId >= 0)
        {
            InstanceMotionStateRecord& record =
                mmMotionStatesByFrame[static_cast<unsigned long>(frameId)];
            record.frameId = static_cast<unsigned long>(frameId);
            record.pose = mLastPose;
            record.velocity = Sophus::SE3f();
            record.state = kZeroVelocityDynamicEntity;
            record.confidence = std::max(record.confidence, 0.7);
            record.reliable = true;
        }
        return;
    }

    ++mnUncertainMotionEvidence;
    mnStaticMotionEvidence = std::max(0, mnStaticMotionEvidence - 1);
    mnDynamicMotionEvidence = std::max(0, mnDynamicMotionEvidence - 1);
    mCurrentDynamicEntityMotionState = kUncertainDynamicEntity;
    if(frameId >= 0)
    {
        InstanceMotionStateRecord& record =
            mmMotionStatesByFrame[static_cast<unsigned long>(frameId)];
        record.frameId = static_cast<unsigned long>(frameId);
        record.pose = mLastPose;
        record.velocity = IsFiniteSE3(observedVelocity) ? observedVelocity : mVelocity;
        record.state = kUncertainDynamicEntity;
        record.confidence = std::max(record.confidence, 0.35);
        record.reliable = false;
    }
}

int Instance::GetStaticMotionEvidence() const
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    return mnStaticMotionEvidence;
}

int Instance::GetDynamicMotionEvidence() const
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    return mnDynamicMotionEvidence;
}

int Instance::GetUncertainMotionEvidence() const
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    return mnUncertainMotionEvidence;
}

int Instance::GetBackendMotionEvidence() const
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    return mnBackendMotionEvidence;
}

int Instance::GetBackendZeroMotionEvidence() const
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    return mnBackendZeroMotionEvidence;
}

int Instance::GetBackendMovingMotionEvidence() const
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    return mnBackendMovingMotionEvidence;
}

int Instance::GetBackendUncertainMotionEvidence() const
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    return mnBackendUncertainMotionEvidence;
}

bool Instance::HasStaticMotionEvidence(int minCount) const
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    return mnStaticMotionEvidence >= std::max(1, minCount);
}

bool Instance::HasDynamicMotionEvidence(int minCount) const
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    return mnDynamicMotionEvidence >= std::max(1, minCount);
}

void Instance::RecordDynamicSupplyGateResult(bool passed, bool hasEvidence, int frameId)
{
    if(!hasEvidence || frameId < 0)
        return;

    std::unique_lock<std::mutex> lock(mMutexInstance);
    if(frameId == mnLastDynamicSupplyGateFrame)
        return;

    if(mnLastDynamicSupplyGateFrame >= 0 && frameId > mnLastDynamicSupplyGateFrame + 1)
        mnDynamicSupplyGateFailureEvidence = 0;

    mnLastDynamicSupplyGateFrame = frameId;
    if(passed)
        mnDynamicSupplyGateFailureEvidence = 0;
    else
        ++mnDynamicSupplyGateFailureEvidence;
}

int Instance::GetDynamicSupplyGateFailureEvidence() const
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    return mnDynamicSupplyGateFailureEvidence;
}

void Instance::SetShapeTemplate(const std::vector<Eigen::Vector3f>& shapeTemplate)
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    mvShapeTemplate = shapeTemplate;
}

std::vector<Eigen::Vector3f> Instance::GetShapeTemplate() const
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    return mvShapeTemplate;
}

void Instance::MarkBackendOutlier(unsigned long frameId, MapPoint* pMP)
{
    if(!pMP)
        return;

    std::unique_lock<std::mutex> lock(mMutexInstance);
    ++mmBackendOutlierCounts[std::make_pair(frameId, pMP)];
}

void Instance::ClearBackendOutlier(unsigned long frameId, MapPoint* pMP)
{
    if(!pMP)
        return;

    std::unique_lock<std::mutex> lock(mMutexInstance);
    mmBackendOutlierCounts.erase(std::make_pair(frameId, pMP));
}

double Instance::GetBackendOutlierWeight(unsigned long frameId, MapPoint* pMP) const
{
    if(!pMP)
        return 1.0;

    std::unique_lock<std::mutex> lock(mMutexInstance);
    double qualityWeight = 1.0;
    const auto itQuality = mmObservationQualityWeights.find(std::make_pair(frameId, pMP));
    if(itQuality != mmObservationQualityWeights.end())
        qualityWeight = std::max(0.05, std::min(1.0, itQuality->second));

    const auto it = mmBackendOutlierCounts.find(std::make_pair(frameId, pMP));
    if(it == mmBackendOutlierCounts.end())
        return qualityWeight;

    const double backendWeight =
        std::max(0.05, 1.0 / static_cast<double>(1 + it->second));
    return std::max(0.05, qualityWeight * backendWeight);
}

void Instance::SetObservationQualityWeight(unsigned long frameId, MapPoint* pMP, double weight)
{
    if(!pMP || !std::isfinite(weight))
        return;

    std::unique_lock<std::mutex> lock(mMutexInstance);
    mmObservationQualityWeights[std::make_pair(frameId, pMP)] =
        std::max(0.05, std::min(1.0, weight));
}

bool Instance::HasObservationQualityWeight(unsigned long frameId, MapPoint* pMP) const
{
    if(!pMP)
        return false;

    std::unique_lock<std::mutex> lock(mMutexInstance);
    return mmObservationQualityWeights.count(std::make_pair(frameId, pMP)) > 0;
}

double Instance::GetObservationQualityWeight(unsigned long frameId, MapPoint* pMP) const
{
    if(!pMP)
        return 1.0;

    std::unique_lock<std::mutex> lock(mMutexInstance);
    const auto it = mmObservationQualityWeights.find(std::make_pair(frameId, pMP));
    if(it == mmObservationQualityWeights.end())
        return 1.0;
    return std::max(0.05, std::min(1.0, it->second));
}

void Instance::SetStructureLocalPoint(MapPoint* pMP, const Eigen::Vector3f& localPoint)
{
    if(!pMP || !localPoint.allFinite())
        return;

    std::unique_lock<std::mutex> lock(mMutexInstance);
    mmStructureLocalPoints[pMP] = localPoint;
}

bool Instance::GetStructureLocalPoint(MapPoint* pMP, Eigen::Vector3f& localPoint) const
{
    if(!pMP)
        return false;

    std::unique_lock<std::mutex> lock(mMutexInstance);
    const std::map<MapPoint*, Eigen::Vector3f>::const_iterator it =
        mmStructureLocalPoints.find(pMP);
    if(it == mmStructureLocalPoints.end() || !it->second.allFinite())
        return false;

    localPoint = it->second;
    return true;
}

void Instance::AddDynamicObservation(unsigned long frameId,
                                     int featureIdx,
                                     MapPoint* pMP,
                                     const Eigen::Vector3f& pointWorld,
                                     double qualityWeight)
{
    if(!pMP || featureIdx < 0 || !pointWorld.allFinite() || !std::isfinite(qualityWeight))
        return;

    DynamicObservationRecord record;
    record.frameId = frameId;
    record.featureIdx = featureIdx;
    record.pBackendPoint = pMP;
    record.pointWorld = pointWorld;
    record.qualityWeight = std::max(0.05, std::min(1.0, qualityWeight));

    std::unique_lock<std::mutex> lock(mMutexInstance);
    std::vector<DynamicObservationRecord>& vRecords = mmDynamicObservationsByFrame[frameId];
    for(size_t i = 0; i < vRecords.size(); ++i)
    {
        if(vRecords[i].pBackendPoint == pMP)
        {
            vRecords[i] = record;
            mmObservationQualityWeights[std::make_pair(frameId, pMP)] = record.qualityWeight;
            return;
        }
    }

    vRecords.push_back(record);
    mmObservationQualityWeights[std::make_pair(frameId, pMP)] = record.qualityWeight;
}

std::vector<Instance::DynamicObservationRecord> Instance::GetDynamicObservationsForFrame(unsigned long frameId) const
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    const std::map<unsigned long, std::vector<DynamicObservationRecord> >::const_iterator it =
        mmDynamicObservationsByFrame.find(frameId);
    if(it == mmDynamicObservationsByFrame.end())
        return std::vector<DynamicObservationRecord>();

    return it->second;
}

std::vector<Instance::DynamicObservationRecord> Instance::GetAllDynamicObservations() const
{
    std::unique_lock<std::mutex> lock(mMutexInstance);
    std::vector<DynamicObservationRecord> vRecords;
    for(std::map<unsigned long, std::vector<DynamicObservationRecord> >::const_iterator it =
            mmDynamicObservationsByFrame.begin();
        it != mmDynamicObservationsByFrame.end(); ++it)
    {
        vRecords.insert(vRecords.end(), it->second.begin(), it->second.end());
    }
    return vRecords;
}

bool Instance::SetInitializedMotionState(const Sophus::SE3f& velocity,
                                         const Sophus::SE3f& initialPose,
                                         const std::vector<Eigen::Vector3f>& shapeTemplate,
                                         int initializedFrameId,
                                         bool initializationMotionReliable,
                                         DynamicEntityMotionState initialMotionState,
                                         double initialMotionConfidence)
{
    if(!IsFiniteSE3(velocity) || !IsFiniteSE3(initialPose) ||
       !std::isfinite(initialMotionConfidence))
        return false;

    initialMotionConfidence = std::max(0.0, std::min(1.0, initialMotionConfidence));
    if(initialMotionState == kDynamicEntityUnknown)
    {
        initialMotionState = ClassifySemanticMotionState(
            velocity, initializationMotionReliable, mnSemanticLabel);
    }
    const Sophus::SE3f canonicalVelocity =
        CanonicalVelocityForState(velocity, initialMotionState);

    std::unique_lock<std::mutex> lock(mMutexInstance);
    mVelocity = canonicalVelocity;
    mbInitializationMotionReliable = initializationMotionReliable;
    // initialPose is the instance pose at initializedFrameId.  Keep the
    // velocity as a separate motion prior; PredictPose() advances it later.
    mLastPose = initialPose;
    mvShapeTemplate = shapeTemplate;
    mbInitialized = true;
    mnInitFrameCount = 0;
    mnInitializedFrame = initializedFrameId;
    mCurrentDynamicEntityMotionState = initialMotionState;
    mnLastPoseProxyKFId = -1;
    mnStaticMotionEvidence = 0;
    mnDynamicMotionEvidence = 0;
    mnUncertainMotionEvidence = 0;
    mnLastMotionGateFrame = -1;
    mnBackendMotionEvidence = 0;
    mnBackendZeroMotionEvidence = 0;
    mnBackendMovingMotionEvidence = 0;
    mnBackendUncertainMotionEvidence = 0;
    mnLastBackendMotionFrame = -1;
    if(initializedFrameId >= 0)
    {
        InstanceMotionStateRecord& record =
            mmMotionStatesByFrame[static_cast<unsigned long>(initializedFrameId)];
        record.frameId = static_cast<unsigned long>(initializedFrameId);
        record.pose = initialPose;
        record.velocity = canonicalVelocity;
        record.state = initialMotionState;
        record.confidence = initialMotionConfidence;
        record.reliable = initializationMotionReliable;
    }
    return true;
}

Eigen::MatrixXf Instance::GetDistanceMatrix() const
{
    std::vector<Eigen::Vector3f> points;
    {
        std::unique_lock<std::mutex> lock(mMutexInstance);
        points.reserve(mspMapPoints.size());
        for(MapPoint* pMP : mspMapPoints)
        {
            if(!pMP || pMP->isBad())
                continue;
            points.push_back(pMP->GetWorldPos());
        }
    }

    const int n = static_cast<int>(points.size());
    if(n == 0)
        return Eigen::MatrixXf();

    Eigen::MatrixXf distances = Eigen::MatrixXf::Zero(n, n);
    for(int i = 0; i < n; ++i)
    {
        for(int j = i + 1; j < n; ++j)
        {
            const float distance = (points[i] - points[j]).norm();
            distances(i, j) = distance;
            distances(j, i) = distance;
        }
    }
    return distances;
}

} // namespace ORB_SLAM3
