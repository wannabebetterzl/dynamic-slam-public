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


#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Viewer.h"
#include "FrameDrawer.h"
#include "Atlas.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Frame.h"
#include "ORBVocabulary.h"
#include "KeyFrameDatabase.h"
#include "ORBextractor.h"
#include "MapDrawer.h"
#include "System.h"
#include "ImuTypes.h"
#include "Instance.h"
#include "PanopticTypes.h"
#include "Settings.h"

#include "GeometricCamera.h"

#include <array>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <unordered_set>

namespace ORB_SLAM3
{

class Viewer;
class FrameDrawer;
class Atlas;
class LocalMapping;
class LoopClosing;
class System;
class Settings;

class Tracking
{  

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Atlas* pAtlas,
             KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, Settings* settings, const string &_nameSeq=std::string());

    ~Tracking();

    // Parse the config file
    bool ParseCamParamFile(cv::FileStorage &fSettings);
    bool ParseORBParamFile(cv::FileStorage &fSettings);
    bool ParseIMUParamFile(cv::FileStorage &fSettings);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    Sophus::SE3f GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp, string filename);
    Sophus::SE3f GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, string filename);
    Sophus::SE3f GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, const PanopticFrameObservation& panopticObservation, string filename);
    Sophus::SE3f GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, const PanopticFrameObservation& panopticObservation, const cv::Mat &dynamicDepth, string filename);
    Sophus::SE3f GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, const cv::Mat &panopticMask, string filename);
    Sophus::SE3f GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, const cv::Mat &panopticMask, const cv::Mat &dynamicDepth, string filename);
    Sophus::SE3f GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename);
    Sophus::SE3f GrabImageMonocular(const cv::Mat &im, const double &timestamp, const PanopticFrameObservation& panopticObservation, string filename);
    Sophus::SE3f GrabImageMonocular(const cv::Mat &im, const double &timestamp, const cv::Mat &panopticMask, string filename);

    void GrabImuData(const IMU::Point &imuMeasurement);

    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);
    void SetStepByStep(bool bSet);
    bool GetStepByStep();

    // Load new settings
    // The focal lenght should be similar or scale prediction will fail when projecting points
    void ChangeCalibration(const string &strSettingPath);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag);

    void UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame);
    KeyFrame* GetLastKeyFrame()
    {
        return mpLastKeyFrame;
    }

    void CreateMapInAtlas();
    //std::mutex mMutexTracks;

    //--
    void NewDataset();
    int GetNumberDataset();
    int GetMatchesInliers();

    //DEBUG
    void SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, string strFolder="");
    void SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, Map* pMap);

    float GetImageScale();

#ifdef REGISTER_LOOP
    void RequestStop();
    bool isStopped();
    void Release();
    bool stopRequested();
#endif

public:

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        RECENTLY_LOST=3,
        LOST=4,
        OK_KLT=5
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Input sensor
    int mSensor;

    // Current Frame
    Frame mCurrentFrame;
    Frame mLastFrame;
    std::deque<Frame> mRecentPanopticFrames;
    struct PanopticCanonicalAssociation
    {
        int rawInstanceId = -1;
        int canonicalInstanceId = -1;
        int semanticId = -1;
        int featureCount = 0;
        int maskArea = 0;
        cv::Rect bbox;
        int bestMatches = 0;
        int secondBestMatches = 0;
        int correctionStreak = 0;
        bool matchedToPrevious = false;
        bool frameCorrection = false;
        bool permanentCorrection = false;
    };
    mutable std::map<int, int> mmPermanentPanopticIdCorrections;
    mutable std::map<std::pair<int, int>, int> mmPanopticIdCorrectionStreaks;
    mutable std::map<unsigned long, std::map<int, int>> mmFramePanopticRawToCanonicalIds;
    mutable std::map<unsigned long, std::map<int, PanopticCanonicalAssociation>> mmFramePanopticCanonicalAssociations;
    mutable unsigned long mnLastPanopticRefinementFrameId = static_cast<unsigned long>(-1);
    std::vector<WindowFrameSnapshot> mvFramesSinceLastKeyFrame;
    bool mbCurrentFrameCreatedKeyFrame = false;

    cv::Mat mImGray;
    cv::Mat mPreviousImGrayForSparseFlow;

    // Initialization Variables (Monocular)
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<Sophus::SE3f> mlRelativeFramePoses;
    list<KeyFrame*> mlpReferences;
    list<double> mlFrameTimes;
    list<bool> mlbLost;

    // frames with estimated pose
    int mTrackedFr;
    bool mbStep;

    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking;

    void Reset(bool bLocMap = false);
    void ResetActiveMap(bool bLocMap = false);

    float mMeanTrack;
    bool mbInitWith3KFs;
    double t0; // time-stamp of first read frame
    double t0vis; // time-stamp of first inserted keyframe
    double t0IMU; // time-stamp of IMU initialization
    bool mFastInit = false;


    vector<MapPoint*> GetLocalMapMPS();

    bool mbWriteStats;

#ifdef REGISTER_TIMES
    void LocalMapStats2File();
    void TrackStats2File();
    void PrintTimeStats();

    vector<double> vdRectStereo_ms;
    vector<double> vdResizeImage_ms;
    vector<double> vdORBExtract_ms;
    vector<double> vdStereoMatch_ms;
    vector<double> vdIMUInteg_ms;
    vector<double> vdPosePred_ms;
    vector<double> vdLMTrack_ms;
    vector<double> vdNewKF_ms;
    vector<double> vdTrackTotal_ms;
#endif

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();

    // Map initialization for stereo and RGB-D
    void StereoInitialization();

    // Map initialization for monocular
    void MonocularInitialization();
    //void CreateNewMapPoints();
    void CreateInitialMapMonocular();

    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();
    bool TrackWithMotionModel();
    bool PredictStateIMU();

    bool Relocalization();

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();
    cv::Mat RefinePanopticWithORBMatches(const Frame& previousFrame,
                                         const Frame& currentFrame,
                                         const cv::Mat& currentRawPanopticMask) const;
    void ExtractInstanceRegionORB(const cv::Mat& imGray,
                                  const cv::Mat& panopticMask,
                                  Frame& frame) const;
    void UpdateSparseFlowGeometryEvidence(Frame& frame,
                                          const cv::Mat& currentGray);
    void ProcessInstances();
    int SplitRgbdDynamicFeatureMatches(const std::string& stage,
                                       bool appendDynamicObservations);
    int SupplyRgbdDepthBackedDynamicObservations(const std::string& stage);
    int SupplyDynamicObservationsForInstance(Instance* pInstance,
                                             const InstanceObservation& observation,
                                             const std::vector<int>& featureIndices,
                                             bool usePredictedMotion,
                                             const Sophus::SE3f& predictedMotion);
    bool InitializeInstance(Instance* pInstance, const std::vector<int>& vCurrIndices);
    bool OptimizePoseWithPanoptic();
    void PushPanopticHistory(const Frame& frame);
    void AppendWindowFrameSnapshot(const Frame& frame);
    std::vector<int> CollectFeatureIndicesForInstance(const Frame& frame, int instanceId) const;
    struct InstanceTrackletStabilityDiagnostics
    {
        bool valid = false;
        bool stable = true;
        bool hardReject = false;
        std::string rejectReason;
        std::string hardRejectReason;
        int semanticTau2 = -1;
        int semanticTau1 = -1;
        int semanticTau = -1;
        int featuresTau2 = 0;
        int featuresTau1 = 0;
        int featuresTau = 0;
        int tracklets = 0;
        double trackletRetention = 0.0;
        double bboxIoUTau2Tau1 = 0.0;
        double bboxIoUTau1Tau = 0.0;
        double bboxCenterShiftTau2Tau1 = 0.0;
        double bboxCenterShiftTau1Tau = 0.0;
        double bboxAreaRatio = 1.0;
        double maskAreaRatio = 1.0;
        double featureDensityRatio = 1.0;
        double currentTrackletCoverage = 0.0;
        double currentTrackletSpreadX = 0.0;
        double currentTrackletSpreadY = 0.0;
        double medianFlowTau2Tau1Px = 0.0;
        double medianFlowTau1TauPx = 0.0;
        double medianFlowAccelerationPx = 0.0;
        double medianFlowAccelerationNormalized = 0.0;
        double medianFlowDirectionCosine = 1.0;
        double triangulationSuccessRatioTau2Tau1 = 0.0;
        double triangulationSuccessRatioTau1Tau = 0.0;
        double triangulationNonpositiveRatio = 0.0;
        double triangulationHighReprojectionRatio = 0.0;
        double triangulationInvalidGeometryRatio = 0.0;
        int triangulationSamples = 0;
    };
    struct CanonicalAssociationDiagnostics
    {
        bool valid = false;
        bool stable = true;
        std::string rejectReason;
        int semanticTau2 = -1;
        int semanticTau1 = -1;
        int semanticTau = -1;
        int rawTau2 = -1;
        int rawTau1 = -1;
        int rawTau = -1;
        int matchedFrames = 0;
        int frameCorrections = 0;
        int permanentCorrections = 0;
        int transientCorrections = 0;
        int ambiguousFrames = 0;
        int missingFrames = 0;
        int minCorrectionStreak = 0;
    };
    InstanceTrackletStabilityDiagnostics EvaluateInstanceTrackletStability(
        const Frame& frameTau2,
        const Frame& frameTau1,
        const Frame& frameTau,
        int instanceId,
        const std::vector<int>& featuresTau2,
        const std::vector<int>& featuresTau1,
        const std::vector<int>& featuresTau,
        const std::vector<std::array<int, 3>>& tracklets) const;
    CanonicalAssociationDiagnostics EvaluateCanonicalAssociationStability(
        const Frame& frameTau2,
        const Frame& frameTau1,
        const Frame& frameTau,
        int canonicalInstanceId) const;
    std::vector<std::array<int, 3>> CollectInstanceTracklets(const Frame& frameTau2,
                                                             const Frame& frameTau1,
                                                             const Frame& frameTau,
                                                             int instanceId) const;
    struct TriangulationQuality
    {
        bool success = false;
        std::string rejectReason;
        double weight = 1.0;
        double minDepth = 0.0;
        double maxReprojectionError = 0.0;
        double parallaxDeg = 0.0;
        bool lowDepth = false;
        bool highReprojectionError = false;
        bool lowParallax = false;
        bool highParallax = false;
        bool highDisparity = false;
    };

    struct TriFrameConsistencyQuality
    {
        bool valid = false;
        bool negativeDepth = false;
        double weight = 1.0;
        double maxReprojectionError = 0.0;
        double dynamicReprojectionError = 0.0;
        double pairTau2Tau1ReprojectionError = 0.0;
        double pairTau1TauReprojectionError = 0.0;
        double motion3dResidual = 0.0;
    };

    struct TriFrameTrackletCandidateQuality
    {
        bool valid = false;
        bool structureEligible = true;
        bool lowDepth = false;
        bool highReprojectionError = false;
        bool highFlowAcceleration = false;
        double weight = 1.0;
        double pairMaxReprojectionError = 0.0;
        double flowAccelerationPx = 0.0;
        double flowAccelerationNormalized = 0.0;
    };

    bool TriangulateMatchedFeatures(const Frame& firstFrame,
                                    int firstIdx,
                                    const Frame& secondFrame,
                                    int secondIdx,
                                    Eigen::Vector3f& pointWorld,
                                    TriangulationQuality* pQuality = static_cast<TriangulationQuality*>(NULL)) const;
    TriFrameTrackletCandidateQuality EvaluateTriFrameTrackletCandidate(
        const Frame& frameTau2,
        int idxTau2,
        const Frame& frameTau1,
        int idxTau1,
        const Frame& frameTau,
        int idxTau,
        const TriangulationQuality& qualityTau2Tau1,
        const TriangulationQuality& qualityTau1Tau) const;
    TriFrameConsistencyQuality EvaluateTriFrameConsistency(
        const Frame& frameTau2,
        int idxTau2,
        const Frame& frameTau1,
        int idxTau1,
        const Frame& frameTau,
        int idxTau,
        const Sophus::SE3f& velocity,
        const Eigen::Vector3f& pointTau2,
        const Eigen::Vector3f& pointTau1) const;
    Sophus::SE3f SolveRigidTransformSVD(const std::vector<Eigen::Vector3f>& src,
                                        const std::vector<Eigen::Vector3f>& dst) const;
    Sophus::SE3f SolveWeightedRigidTransformSVD(const std::vector<Eigen::Vector3f>& src,
                                                const std::vector<Eigen::Vector3f>& dst,
                                                const std::vector<double>& weights) const;

    bool TrackLocalMap();
    void SearchLocalPoints();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    // Perform preintegration from last frame
    void PreintegrateIMU();

    // Reset IMU biases and compute frame velocity
    void ResetFrameIMU();

    bool mbMapUpdated;

    // Imu preintegration from last frame
    IMU::Preintegrated *mpImuPreintegratedFromLastKF;

    // Queue of IMU measurements between frames
    std::list<IMU::Point> mlQueueImuData;

    // Vector of IMU measurements from previous to current frame (to be filled by PreintegrateIMU)
    std::vector<IMU::Point> mvImuFromLastFrame;
    std::mutex mMutexImuQueue;

    // Imu calibration parameters
    IMU::Calib *mpImuCalib;

    // Last Bias Estimation (at keyframe creation)
    IMU::Bias mLastBias;

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO;

    //Other Thread Pointers
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;

    //ORB
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;

    //BoW
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization (only for monocular)
    bool mbReadyToInitializate;
    bool mbSetInit;

    //Local Map
    KeyFrame* mpReferenceKF;
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    std::vector<MapPoint*> mvpLocalMapPoints;
    
    // System
    System* mpSystem;
    
    //Drawers
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;
    bool bStepByStep;

    //Atlas
    Atlas* mpAtlas;

    //Calibration matrix
    cv::Mat mK;
    Eigen::Matrix3f mK_;
    cv::Mat mDistCoef;
    float mbf;
    float mImageScale;

    float mImuFreq;
    double mImuPer;
    bool mInsertKFsLost;

    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    int mnFirstImuFrameId;
    int mnFramesToResetIMU;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;
    cv::Mat mRgbdBackendDynamicDepth;

    //Current matches in frame
    int mnMatchesInliers;

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;
    double mTimeStampLost;
    double time_recently_lost;

    unsigned int mnFirstFrameId;
    unsigned int mnInitialFrameId;
    unsigned int mnLastInitFrameId;

    bool mbCreatedMap;

    //Motion Model
    bool mbVelocity{false};
    Sophus::SE3f mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;

    list<MapPoint*> mlpTemporalPoints;

    //int nMapChangeIndex;

    int mnNumDataset;

    ofstream f_track_stats;
    ofstream f_observability_stats;

    ofstream f_track_times;
    double mTime_PreIntIMU;
    double mTime_PosePred;
    double mTime_LocalMapTrack;
    double mTime_NewKF_Dec;

    GeometricCamera* mpCamera, *mpCamera2;

    int initID, lastID;

    Sophus::SE3f mTlr;
    int mnPanopticRefinementMinMatches = 8;
    float mfPanopticRefinementSecondBestRatio = 1.5f;
    int mnPanopticPermanentCorrectionMinStreak = 3;
    int mnInstanceInitializationMinFeatures = 15;
    int mnInstanceInitializationMinTracklets = 8;
    size_t mnPanopticHistoryLength = 2;

    void newParameterLoader(Settings* settings);

#ifdef REGISTER_LOOP
    bool Stop();

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;
#endif

public:
    cv::Mat mImRight;
};

} //namespace ORB_SLAM

#endif // TRACKING_H
