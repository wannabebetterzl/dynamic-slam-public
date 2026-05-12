echo "Configuring and building Thirdparty/DBoW2 ..."

EXTRA_CMAKE_ARGS=()
if [[ "${STSLAM_DISABLE_NATIVE_AVX:-0}" != "0" ]]; then
    EXTRA_CMAKE_ARGS+=(-DSTSLAM_DISABLE_NATIVE_AVX=ON)
    echo "STSLAM_DISABLE_NATIVE_AVX is enabled for this build."
fi

cd Thirdparty/DBoW2
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release "${EXTRA_CMAKE_ARGS[@]}"
make -j

cd ../../g2o

echo "Configuring and building Thirdparty/g2o ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release "${EXTRA_CMAKE_ARGS[@]}"
make -j

cd ../../Sophus

echo "Configuring and building Thirdparty/Sophus ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release "${EXTRA_CMAKE_ARGS[@]}"
make -j

cd ../../../

echo "Uncompress vocabulary ..."

cd Vocabulary
tar -xf ORBvoc.txt.tar.gz
cd ..

echo "Configuring and building ORB_SLAM3 ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release "${EXTRA_CMAKE_ARGS[@]}"
make -j4
