# how to install faiss-gpu from source in a800
> Hope it can help you, welcome to star~ 
- python3.8
- a800

## prepare
- download pcre2-10.37.tar.gz: https://sourceforge.net/projects/pcre/files/pcre2/10.37/pcre2-10.37.tar.gz
- download swig-4.1.1.tar.gz: https://sourceforge.net/projects/swig/files/swig/
- download faiss-1.7.1.tar.gz: https://github.com/facebookresearch/faiss/releases/tag/v1.7.1
- download cmake: https://github.com/Kitware/CMake/releases/download/v3.29.2/cmake-3.29.2.tar.gz

## start building

optional: update cmake
```
tar -xzvf cmake-3.29.2.tar.gz
cd cmake-3.29.2
chmod 777 ./configure
./configure
make
make install
update-alternatives --install /usr/bin/cmake cmake /usr/local/bin/cmake 1 --force
cmake --version
```

```
tar -zxvf swig-4.1.1.tar.gz
cp pcre2-10.37.tar.gz swig-4.1.1/

cd swig-4.1.1/
./Tools/pcre-build.sh
./configure
make
make install
swig -version
```

```
apt-get update -y
apt-get install libopenblas-dev -y
apt-get install libomp-dev -y
apt-get install libssl-dev -y
apt-get install tmux -y
```

```
tar -zxvf faiss-1.7.1.tar.gz
cd faiss-1.7.1
cmake -B build . -DCUDAToolkit_ROOT=/usr/local/cuda/ -DFAISS_ENABLE_GPU=ON -DPython_EXECUTABLE=$(which python) -DBUILD_TESTING=OFF
make -C build -j faiss
make -C build -j swigfaiss
cd build/faiss/python && python setup.py install
CUDA_VISIBLE_DEVICES=0 python test_faiss_gpu.py
```

## test
```
import faiss
import numpy as np
import time

d = 64
nb = 100000
nq = 10000
np.random.seed(0)

if faiss.get_num_gpus() > 0:
    print("Faiss supports GPU")
else:
    print("Faiss does not support support GPU")

xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')


start_time = time.time()
index = faiss.IndexFlatL2(d)
if faiss.get_num_gpus() > 0:
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)

index.add(xb)
D, I = index.search(xq, 10)
end_time = time.time()

print(f"time: {end_time - start_time:.4f} s")

```

- if you meet `ImportError: cannot import name '_swigfaiss'`, please unzip your `faiss-1.7.1-py3.8.egg`
  ```
  cd /opt/conda/lib/python3.8/site-packages/
  unzip faiss-1.7.1-py3.8.egg
  ```
- [index_cpu_to_gpu_multiple slow on first run, fast on subsequent](https://github.com/facebookresearch/faiss/issues/2710)

## ref
- [official tutorial](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
- [ref1](https://blog.csdn.net/qq_41368074/article/details/130714550)
