Emergence detection extractor

code base on https://github.com/longcw/faster_rcnn_pytorch

This is the initial version of Emergence detection extractor, to deploy it on Nebular, further contribution is needed.

  Install the requirements
  conda install pip pyyaml sympy h5py cython numpy scipy
  conda install -c menpo opencv3
  pip install easydict
  build the Cython modules for nms and the roi_pooling layer
  cd faster_rcnn_pytorch/faster_rcnn
  ./make.sh
  Convert GPU process to CPU`
