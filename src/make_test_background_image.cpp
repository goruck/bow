/*
 *  make_test_background_image.cpp
 *
 *  Created by Roy Shilkrot on 8/19/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 *  Adapted and updated for OpenCV 3.2 by Lindo St. Angel 2017.
 *
 */

/* compile with
gcc -Wall make_test_background_image.cpp -I/usr/local/include/ \
    -L/usr/lib/ -lstdc++ \
    -L/usr/local/lib -lopencv_highgui -lopencv_core -lopencv_imgcodecs \
    -lopencv_imgproc -lopencv_videoio -lopencv_video -lopencv_videostab \
    -lopencv_flann -lopencv_ml -lm -lopencv_xfeatures2d -lopencv_features2d \
    -o make_test_background_image -std=c++11
*/

#include <stdio.h>
#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>

#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	string dir, filepath;
	DIR *dp;
	struct dirent *dirp;
	struct stat filestat;
	
	//get images
        dir = "/home/lindo/dev/opencv-test/bow/TEST";
	dp = opendir( dir.c_str() );
	int count = 0;
	Mat accum;
	while ((dirp = readdir( dp )))
        {
		count++;
		
		filepath = dir + "/" + dirp->d_name;
		
		// If the file is a directory (or is in some way invalid) we'll skip it 
		if (stat( filepath.c_str(), &filestat )) continue;
		if (S_ISDIR( filestat.st_mode ))         continue;
		if (dirp->d_name[0] == '.')		 continue; //hidden file!
		
		cout << "eval file " << filepath << endl;

		Mat img = imread(filepath),img64;
		img.convertTo(img64, CV_64FC3);
		
		if (!accum.data) {
			accum.create(img.size(), CV_64FC3);
		}
		if (img64.size() == accum.size()) {
			accum += img64;
		}
	}
	
	accum /= count;
	Mat accum_8UC3; accum.convertTo(accum_8UC3, CV_8UC3);
	
	imwrite("background.png", accum_8UC3);
	
	imshow("accum", accum_8UC3);
	waitKey(0);
}
