# Usage: 
#   cd model_vis
#   bazel run //:build_wheel -- $PWD/wheels 

PKG_DIR="$(pwd)"
DST_DIR="$1"
mkdir $DST_DIR
echo "Build wheel at destination: $DST_DIR"
python setup.py bdist_wheel -d $DST_DIR
python setup.py sdist -d $DST_DIR