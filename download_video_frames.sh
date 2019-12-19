wget $1
mkdir $3
ffmpeg -i $2 -f image2 -r 1 $3/frame%09d.png
