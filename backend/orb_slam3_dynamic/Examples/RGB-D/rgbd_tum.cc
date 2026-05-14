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

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<cstdlib>
#include<sstream>
#include<iomanip>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);
bool LoadPanopticMask(const string& panopticRoot, const string& imageRelativePath, cv::Mat& panopticMask);
bool DisableFrameSleep();
bool UseViewer();
bool SyncLocalMapping();
bool SyncLocalMappingVerbose();
int SyncLocalMappingMaxWaitMs();
int SequentialLocalMappingMaxSteps();
int SequentialLocalMappingDrainPeriodFrames();
int SequentialLocalMappingMaxQueueBeforeDrain();
void ProcessSequentialLocalMappingIfRequested(ORB_SLAM3::System& slam, int frameIdx, double timestamp);
void WaitForLocalMappingIfRequested(ORB_SLAM3::System& slam, int frameIdx, double timestamp);

int main(int argc, char **argv)
{
    if(argc < 5 || argc > 7)
    {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association [path_to_panoptic_root] [path_to_backend_dynamic_depth_sequence]" << endl;
        return 1;
    }
    const string panopticRoot = (argc >= 6) ? argv[5] : string();
    const string backendDynamicDepthRoot = (argc == 7) ? argv[6] : string();
    int missingPanopticMasks = 0;
    int missingBackendDynamicDepth = 0;

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    string strAssociationFilename = string(argv[4]);
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    const bool bUseViewer = UseViewer();
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::RGBD,bUseViewer);
    float imageScale = SLAM.GetImageScale();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;
    cout << "Viewer enabled: " << (bUseViewer ? "true" : "false") << endl;
    cout << "Sync local mapping: " << (SyncLocalMapping() ? "true" : "false") << endl;
    cout << "Sequential local mapping: " << (SLAM.SequentialLocalMappingEnabled() ? "true" : "false") << endl;

    // Main loop
    cv::Mat imRGB, imD, imDynamicD;
    const bool bDisableFrameSleep = DisableFrameSleep();
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image and depthmap from file
        imRGB = cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        imD = cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        imDynamicD.release();
        if(!backendDynamicDepthRoot.empty())
        {
            imDynamicD = cv::imread(backendDynamicDepthRoot+"/"+vstrImageFilenamesD[ni],cv::IMREAD_UNCHANGED);
            if(imDynamicD.empty())
                ++missingBackendDynamicDepth;
        }
        double tframe = vTimestamps[ni];
        cv::Mat panopticMask;

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

        if(!panopticRoot.empty() && !LoadPanopticMask(panopticRoot, vstrImageFilenamesRGB[ni], panopticMask))
            ++missingPanopticMasks;

        if(imageScale != 1.f)
        {
            int width = imRGB.cols * imageScale;
            int height = imRGB.rows * imageScale;
            cv::resize(imRGB, imRGB, cv::Size(width, height));
            cv::resize(imD, imD, cv::Size(width, height));
            if(!imDynamicD.empty())
                cv::resize(imDynamicD, imDynamicD, cv::Size(width, height));
            if(!panopticMask.empty())
                cv::resize(panopticMask, panopticMask, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        if(panopticRoot.empty())
            SLAM.TrackRGBD(imRGB,imD,tframe);
        else if(!backendDynamicDepthRoot.empty())
            SLAM.TrackRGBD(imRGB,imD,tframe,panopticMask,imDynamicD,vector<ORB_SLAM3::IMU::Point>(),vstrImageFilenamesRGB[ni]);
        else
            SLAM.TrackRGBD(imRGB,imD,tframe,panopticMask,vector<ORB_SLAM3::IMU::Point>(),vstrImageFilenamesRGB[ni]);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        ProcessSequentialLocalMappingIfRequested(SLAM, ni, tframe);
        WaitForLocalMappingIfRequested(SLAM, ni, tframe);

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(!bDisableFrameSleep && ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;
    if(!panopticRoot.empty())
        cout << "missing panoptic masks: " << missingPanopticMasks << endl;
    if(!backendDynamicDepthRoot.empty())
        cout << "missing backend dynamic depth maps: " << missingBackendDynamicDepth << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");   
    SLAM.SaveKeyFrameTimeline("KeyFrameTimeline.csv");

    return 0;
}

bool EnvFlagOrDefault(const char* name, const bool defaultValue)
{
    const char* envValue = getenv(name);
    if(!envValue)
        return defaultValue;

    const string value(envValue);
    return value != "0" && value != "false" && value != "FALSE" &&
           value != "off" && value != "OFF" && value != "no" &&
           value != "NO";
}

int EnvIntOrDefault(const char* name, const int defaultValue, const int minValue)
{
    const char* envValue = getenv(name);
    if(!envValue)
        return defaultValue;

    char* endPtr = NULL;
    const long value = strtol(envValue, &endPtr, 10);
    if(endPtr == envValue)
        return defaultValue;
    return static_cast<int>(std::max<long>(minValue, value));
}

bool DisableFrameSleep()
{
    return EnvFlagOrDefault("STSLAM_DISABLE_FRAME_SLEEP", false);
}

bool UseViewer()
{
    return EnvFlagOrDefault("STSLAM_USE_VIEWER", false);
}

bool SyncLocalMapping()
{
    return EnvFlagOrDefault("STSLAM_SYNC_LOCAL_MAPPING", false);
}

bool SyncLocalMappingVerbose()
{
    return EnvFlagOrDefault("STSLAM_SYNC_LOCAL_MAPPING_VERBOSE", false);
}

int SyncLocalMappingMaxWaitMs()
{
    return EnvIntOrDefault("STSLAM_SYNC_LOCAL_MAPPING_MAX_WAIT_MS", 5000, 0);
}

int SequentialLocalMappingMaxSteps()
{
    return EnvIntOrDefault("STSLAM_SEQUENTIAL_LOCAL_MAPPING_MAX_STEPS", 1000, 1);
}

int SequentialLocalMappingDrainPeriodFrames()
{
    return EnvIntOrDefault("STSLAM_SEQUENTIAL_LOCAL_MAPPING_DRAIN_PERIOD_FRAMES", 1, 0);
}

int SequentialLocalMappingMaxQueueBeforeDrain()
{
    return EnvIntOrDefault("STSLAM_SEQUENTIAL_LOCAL_MAPPING_MAX_QUEUE_BEFORE_DRAIN", 3, 1);
}

void ProcessSequentialLocalMappingIfRequested(ORB_SLAM3::System& slam, const int frameIdx, const double timestamp)
{
    if(!slam.SequentialLocalMappingEnabled())
        return;

    const int queueBefore = slam.LocalMappingKeyframesInQueue();
    if(queueBefore == 0)
        return;

    const int drainPeriod = SequentialLocalMappingDrainPeriodFrames();
    const int maxQueue = SequentialLocalMappingMaxQueueBeforeDrain();
    const bool drainByPeriod = drainPeriod > 0 && (frameIdx % drainPeriod) == 0;
    const bool drainByQueue = queueBefore >= maxQueue;
    if(!drainByPeriod && !drainByQueue)
    {
        if(SyncLocalMappingVerbose())
        {
            cout << "[STSLAM_SEQUENTIAL_LOCAL_MAPPING]"
                 << " frame_index=" << frameIdx
                 << " timestamp=" << fixed << setprecision(6) << timestamp
                 << " processed_keyframes=0"
                 << " queue=" << queueBefore
                 << " drain_reason=deferred"
                 << endl;
        }
        return;
    }

    const int processed = slam.ProcessSequentialLocalMappingQueue(SequentialLocalMappingMaxSteps());
    if(SyncLocalMappingVerbose() || processed > 0)
    {
        cout << "[STSLAM_SEQUENTIAL_LOCAL_MAPPING]"
             << " frame_index=" << frameIdx
             << " timestamp=" << fixed << setprecision(6) << timestamp
             << " processed_keyframes=" << processed
             << " queue=" << slam.LocalMappingKeyframesInQueue()
             << " drain_reason=" << (drainByQueue ? "queue" : "period")
             << endl;
    }
}

void WaitForLocalMappingIfRequested(ORB_SLAM3::System& slam, const int frameIdx, const double timestamp)
{
    if(!SyncLocalMapping())
        return;
    if(slam.SequentialLocalMappingEnabled())
        return;

    const int maxWaitMs = SyncLocalMappingMaxWaitMs();
    const int pollUs = 1000;
    int waitedUs = 0;
    bool timedOut = false;

    while(true)
    {
        const int queueSize = slam.LocalMappingKeyframesInQueue();
        const bool acceptsKeyframes = slam.LocalMappingAcceptKeyFrames();
        if(queueSize == 0 && acceptsKeyframes)
            break;

        if(maxWaitMs > 0 && waitedUs >= maxWaitMs * 1000)
        {
            timedOut = true;
            break;
        }

        usleep(pollUs);
        waitedUs += pollUs;
    }

    if(SyncLocalMappingVerbose() || timedOut)
    {
        cout << "[STSLAM_SYNC_LOCAL_MAPPING]"
             << " frame_index=" << frameIdx
             << " timestamp=" << fixed << setprecision(6) << timestamp
             << " waited_ms=" << setprecision(3)
             << static_cast<double>(waitedUs) / 1000.0
             << " queue=" << slam.LocalMappingKeyframesInQueue()
             << " accept_keyframes=" << (slam.LocalMappingAcceptKeyFrames() ? 1 : 0)
             << " timed_out=" << (timedOut ? 1 : 0)
             << endl;
    }
}

namespace
{

string BaseName(const string& path)
{
    const size_t pos = path.find_last_of("/\\");
    if(pos == string::npos)
        return path;
    return path.substr(pos + 1);
}

string StripExtension(const string& path)
{
    const size_t pos = path.find_last_of('.');
    if(pos == string::npos)
        return path;
    return path.substr(0, pos);
}

bool FileExists(const string& path)
{
    ifstream handle(path.c_str());
    return handle.good();
}

} // namespace

bool LoadPanopticMask(const string& panopticRoot, const string& imageRelativePath, cv::Mat& panopticMask)
{
    vector<string> candidates;
    candidates.push_back(panopticRoot + "/" + StripExtension(imageRelativePath) + ".png");
    candidates.push_back(panopticRoot + "/" + StripExtension(BaseName(imageRelativePath)) + ".png");

    for(const string& candidate : candidates)
    {
        if(!FileExists(candidate))
            continue;

        panopticMask = cv::imread(candidate, cv::IMREAD_UNCHANGED);
        if(panopticMask.empty() || panopticMask.channels() != 1)
            return false;

        if(panopticMask.type() != CV_16UC1 && panopticMask.type() != CV_32SC1)
            panopticMask.convertTo(panopticMask, CV_16UC1);

        return true;
    }

    return false;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    if(!fAssociation.is_open())
    {
        cerr << "Failed to open association file at: " << strAssociationFilename << endl;
        return;
    }

    string s;
    while(getline(fAssociation, s))
    {
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}
