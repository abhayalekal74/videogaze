wget $1
mkdir $3
ffmpeg -i $2 -f image2 -r $4 $3/frame%09d.png
