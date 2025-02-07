git clone https://github.com/microsoft/SuperScaler.git
cd SuperScaler
git switch EuroSys24AE
git checkout 53d003681662a817e23c4b1aa47308cf17768d33
git submodule update --init --recursive

cd external/Megatron-LM
cp ../aceso_ae_megatron.patch ./
git apply --whitespace=nowarn aceso_ae_megatron.patch
cd ../..

cp ../super_scaler.patch ./
git apply --whitespace=nowarn super_scaler.patch
cd ..

docker build -t aceso-image:latest -f Dockerfile.cu121 .

docker run -it -d --name=aceso --gpus=all --privileged --net=host --ipc=host --shm-size=1g --ulimit memlock=-1 -v $(pwd):$(pwd) aceso-image bash
