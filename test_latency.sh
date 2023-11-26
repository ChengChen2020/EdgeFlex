pp=$1
ne=$2
np=$3
ff="$pp$ne$np.tar.gz"
dd="ckpt_0.0001_${pp}_100_5_${ne}_${np}_1.0_False_AdaptE"
echo "$ff"
echo "$dd"
tar -xzvf "$ff"
mv "$dd" checkpoint
python3 test_latency.py --pp "$pp" --n_embed "$ne" --n_parts "$np"
cd checkpoint || exit
rm -r "$dd"
cd ..