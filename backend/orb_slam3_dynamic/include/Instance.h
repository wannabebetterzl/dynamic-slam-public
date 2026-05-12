#ifndef INSTANCE_H
#define INSTANCE_H

#include <map>
#include <mutex>
#include <set>
#include <vector>

#include "Thirdparty/Sophus/sophus/se3.hpp"

#include "Eigen/Core"

namespace ORB_SLAM3
{

class KeyFrame;
class MapPoint;

class Instance
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    enum DynamicEntityMotionState
    {
        kDynamicEntityUnknown = 0,
        kZeroVelocityDynamicEntity = 1,
        kMovingDynamicEntity = 2,
        kUncertainDynamicEntity = 3
    };

    struct InstanceMotionStateRecord
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        InstanceMotionStateRecord()
            : frameId(0),
              state(kDynamicEntityUnknown),
              confidence(0.0),
              reliable(false)
        {
        }

        unsigned long frameId;
        Sophus::SE3f pose;
        Sophus::SE3f velocity;
        DynamicEntityMotionState state;
        double confidence;
        bool reliable;
    };

    struct DynamicObservationRecord
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        DynamicObservationRecord()
            : frameId(0),
              featureIdx(-1),
              qualityWeight(1.0),
              pBackendPoint(NULL)
        {
        }

        unsigned long frameId;
        int featureIdx;
        double qualityWeight;
        Eigen::Vector3f pointWorld;
        MapPoint* pBackendPoint;
    };

    Instance();
    Instance(int instanceId, int semanticLabel);

    int GetId() const;
    int GetSemanticLabel() const;
    void SetSemanticLabel(int semanticLabel);

    void AddMapPoint(MapPoint* pMP);
    void RemoveMapPoint(MapPoint* pMP);
    std::set<MapPoint*> GetMapPoints() const;
    size_t NumMapPoints() const;

    void UpdateMotionPrior(KeyFrame* pKF, const Sophus::SE3f& motion);
    Sophus::SE3f GetMotionPriorForKeyFrame(KeyFrame* pKF) const;

    void UpdatePoseProxy(KeyFrame* pKF, const Sophus::SE3f& poseProxy);
    Sophus::SE3f GetPoseProxyForKeyFrame(KeyFrame* pKF) const;

    void PredictPose(const Sophus::SE3f& velocity);
    Eigen::MatrixXf GetDistanceMatrix() const;
    int MarkSeenInFrame(int frameId);
    void ResetInitializationCounter();
    int AdvanceInitializationCounter(bool consecutive);
    bool IsInitialized() const;
    int GetInitializedFrame() const;
    Sophus::SE3f GetVelocity() const;
    Sophus::SE3f GetLastPoseEstimate() const;
    bool HasReliableInitializationMotion() const;
    void UpdateVelocityEstimate(const Sophus::SE3f& velocity);
    void SetInstanceMotionState(unsigned long frameId,
                                const Sophus::SE3f& pose,
                                const Sophus::SE3f& velocity,
                                DynamicEntityMotionState state,
                                double confidence,
                                bool reliable);
    bool GetInstanceMotionState(unsigned long frameId,
                                InstanceMotionStateRecord& record) const;
    bool GetLatestInstanceMotionState(InstanceMotionStateRecord& record) const;
    DynamicEntityMotionState GetCurrentDynamicEntityMotionState() const;
    bool RecordBackendMotionObservation(const Sophus::SE3f& observedVelocity,
                                        bool backendMature,
                                        int frameId,
                                        float maxImmatureTranslation,
                                        float maxImmatureRotationDeg,
                                        int immatureConfirmFrames);
    void RecordMotionGateState(int motionState,
                               const Sophus::SE3f& observedVelocity,
                               int frameId,
                               float staticVelocityDecay);
    int GetStaticMotionEvidence() const;
    int GetDynamicMotionEvidence() const;
    int GetUncertainMotionEvidence() const;
    int GetBackendMotionEvidence() const;
    int GetBackendZeroMotionEvidence() const;
    int GetBackendMovingMotionEvidence() const;
    int GetBackendUncertainMotionEvidence() const;
    bool HasStaticMotionEvidence(int minCount) const;
    bool HasDynamicMotionEvidence(int minCount) const;
    void RecordDynamicSupplyGateResult(bool passed, bool hasEvidence, int frameId);
    int GetDynamicSupplyGateFailureEvidence() const;
    void SetShapeTemplate(const std::vector<Eigen::Vector3f>& shapeTemplate);
    std::vector<Eigen::Vector3f> GetShapeTemplate() const;
    void MarkBackendOutlier(unsigned long frameId, MapPoint* pMP);
    void ClearBackendOutlier(unsigned long frameId, MapPoint* pMP);
    void SetObservationQualityWeight(unsigned long frameId, MapPoint* pMP, double weight);
    bool HasObservationQualityWeight(unsigned long frameId, MapPoint* pMP) const;
    double GetObservationQualityWeight(unsigned long frameId, MapPoint* pMP) const;
    void SetStructureLocalPoint(MapPoint* pMP, const Eigen::Vector3f& localPoint);
    bool GetStructureLocalPoint(MapPoint* pMP, Eigen::Vector3f& localPoint) const;
    void AddDynamicObservation(unsigned long frameId,
                               int featureIdx,
                               MapPoint* pMP,
                               const Eigen::Vector3f& pointWorld,
                               double qualityWeight);
    std::vector<DynamicObservationRecord> GetDynamicObservationsForFrame(unsigned long frameId) const;
    std::vector<DynamicObservationRecord> GetAllDynamicObservations() const;
    double GetBackendOutlierWeight(unsigned long frameId, MapPoint* pMP) const;
    bool SetInitializedMotionState(const Sophus::SE3f& velocity,
                                   const Sophus::SE3f& initialPose,
                                   const std::vector<Eigen::Vector3f>& shapeTemplate,
                                   int initializedFrameId,
                                   bool initializationMotionReliable = true,
                                   DynamicEntityMotionState initialMotionState =
                                       kDynamicEntityUnknown,
                                   double initialMotionConfidence = 1.0);

private:
    int mnInstanceId;
    int mnSemanticLabel;

    std::map<KeyFrame*, Sophus::SE3f> mmKeyframeMotionPriors;
    std::map<KeyFrame*, Sophus::SE3f> mmKeyframePoseProxy;
    std::set<MapPoint*> mspMapPoints;

    Sophus::SE3f mLastPose;
    Sophus::SE3f mVelocity;

    bool mbInitialized;
    bool mbInitializationMotionReliable;
    int mnInitFrameCount;
    int mnInitializedFrame;
    int mnLastSeenFrame;
    int mnLastPoseProxyKFId;
    int mnStaticMotionEvidence;
    int mnDynamicMotionEvidence;
    int mnUncertainMotionEvidence;
    int mnLastMotionGateFrame;
    int mnBackendMotionEvidence;
    int mnBackendZeroMotionEvidence;
    int mnBackendMovingMotionEvidence;
    int mnBackendUncertainMotionEvidence;
    int mnLastBackendMotionFrame;
    int mnDynamicSupplyGateFailureEvidence;
    int mnLastDynamicSupplyGateFrame;
    DynamicEntityMotionState mCurrentDynamicEntityMotionState;

    std::vector<Eigen::Vector3f> mvShapeTemplate;
    Sophus::SE3f mLastBackendObservedVelocity;
    std::map<std::pair<unsigned long, MapPoint*>, int> mmBackendOutlierCounts;
    std::map<std::pair<unsigned long, MapPoint*>, double> mmObservationQualityWeights;
    std::map<unsigned long, std::vector<DynamicObservationRecord> > mmDynamicObservationsByFrame;
    std::map<MapPoint*, Eigen::Vector3f> mmStructureLocalPoints;
    std::map<unsigned long, InstanceMotionStateRecord> mmMotionStatesByFrame;

private:
    mutable std::mutex mMutexInstance;
};

} // namespace ORB_SLAM3

#endif // INSTANCE_H
