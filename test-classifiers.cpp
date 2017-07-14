/*
 *  test classifiers created with bow-classification.cpp
 *
 *  Created by Roy Shilkrot on 8/19/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 *  Adpated and updated for OpenCV 3.2 by Lindo St. Angel 2017.
 *
 */

/* compile with
gcc -Wall test-classifiers.cpp -I/usr/local/include/ \
    -L/usr/lib/ -lstdc++ \
    -L/usr/local/lib -lopencv_highgui -lopencv_core -lopencv_imgcodecs \
    -lopencv_imgproc -lopencv_videoio -lopencv_video -lopencv_videostab \
    -lopencv_flann -lopencv_ml -lm -lopencv_xfeatures2d -lopencv_features2d \
    -o test-classifiers.out -std=c++11
*/

#include "opencv2/opencv_modules.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/ml.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <functional>
#include <string>
// Includes for std::fixed and std::setprecision
#include <iomanip>

#include <algorithm> // for std::find

// Includes for directory operations
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ml;

int main (int argc, char * const argv[]) {

  cout << "------- test ---------\n";

  // Create SURF feature detector
  double hessianThreshold = 150;
  Ptr<SURF> detector = SURF::create(hessianThreshold);
  if( !detector ) {
    cerr << "feature detector was not created" << endl;
    exit(EXIT_FAILURE);
  }

  // Setup extractor to convert images to presence vectors
  Ptr< cv::DescriptorMatcher > descMatcher;
  descMatcher = DescriptorMatcher::create( "BruteForce" );
  Ptr< SURF > descExtractor = SURF::create();
  Ptr< cv::BOWImgDescriptorExtractor > bowExtractor;
  bowExtractor = new BOWImgDescriptorExtractor( descExtractor, descMatcher );

  // Read vocabulary from disk
  FileStorage fs("vocabulary.xml", FileStorage::READ);
  Mat vocabulary;
  fs["vocabulary"] >> vocabulary;
  fs.release();

  bowExtractor->BOWImgDescriptorExtractor::setVocabulary( vocabulary );

  vector<KeyPoint> keypoints;
  Mat response_hist; // histogram of the presence (or absence) of each word in vocabulary

  // evaluate
  int count = 0;
  int good = 0, bad = 0;

  // Open test data file that points to test images
  // Format is "FILENAME X,Y,W,H CLASS"
  ifstream testDataFile; // object for handling file input
  //testDataFile.open( "/home/lindo/dev/opencv-test/bow/test_one_class_rect.txt" );
  //testDataFile.open( "/home/lindo/dev/opencv-test/bow/train.txt" );
  testDataFile.open( "/home/lindo/dev/opencv-test/bow/test.txt" );
  if (!testDataFile.is_open()) {
    cout << "Could not open the test data file." << endl;
    cout << "Program terminating." << endl;;
    exit(EXIT_FAILURE);
  }

  map <string, map<string,int> > confusionMatrix; // [actual][predicted]
  confusionMatrix.clear();

  vector<string> testClasses;
  testClasses.clear();
  string line;
  // Scan test database for the actual test classes and record them
  while ( getline(testDataFile, line) ) {
    //if (count++ > 15) break;

    istringstream iss(line);
    iss.ignore( 256, ' ' ); // skip filename
    iss.ignore( 256, ',' ); // skip x coord
    iss.ignore( 256, ',' ); // skip y coord
    iss.ignore( 256, ',' ); // skip width
    iss.ignore( 256, ' ' ); // skip height
    string actClass;
    iss >> actClass; // read actual class of image

    if ( actClass == "" ) continue; // skip images w/o class information

    if ( find(testClasses.begin(), testClasses.end(), actClass) != testClasses.end() )
       continue; // class already present in vector, skip
    
    testClasses.push_back(actClass); // record class
  }

  cout << "Read " << testClasses.size() << " actual test classes." << endl;

  count = 0;
  testDataFile.clear();    // clear eof errors (if any)
  testDataFile.seekg( 0 ); // rewind the stream to the beginning
  while ( getline(testDataFile, line) ) {
    //if (count > 15) break;
    count++;

    istringstream iss(line);
    string filepath;
    iss >> filepath; // path to image name
    cout << "eval file: " << filepath << endl;
    Rect r; char delim;
    iss >> r.x >> delim; // x coord of region of interest
    iss >> r.y >> delim; // y coord of region of interest
    iss >> r.width >> delim; // width of region of interest
    iss >> r.height; // height of region of interest
    string actClass;
    iss >> actClass; // actual (true) class of image

    if (actClass == "") {
      cout << endl << "Image without class id, skipping: " << filepath << endl;
      continue;
    }
			
    Mat img = imread(filepath);
    if( img.empty() ) { // Check for invalid input
      cout << endl << "Could not open or find image, skipping: " << filepath << endl;
      continue;
    }

    /*if( img.size() != Size(640,480) ) {
        cout << "image not 640x480, skipping" << endl;
        continue;
    }*/

    //img = img(clipping_rect);
    //img_fg = img - bg_;

    //rectangle(img, r, Scalar(0,255,0), 2, 8, 0); // draw region of interest on image
    //imshow("before crop", img);

    r &= Rect(0, 0, img.cols, img.rows); // region of interest in terms of img rows and cols
    if(r.width != 0) {
      img = img(r); // crop to interesting region
    }

    //putText(img, actClass, Point(20,20), CV_FONT_HERSHEY_PLAIN, 2.0, Scalar(255), 2); // display img w/class #
    //imshow("after crop", img);
    //waitKey(0);

    detector->detectAndCompute( img, noArray(), keypoints, noArray() );
    if ( keypoints.empty() ) {
      continue;
    }

    vector< vector<int> > pointIdxsOfClusters;

    bowExtractor->BOWImgDescriptorExtractor::compute( img, keypoints, response_hist, &pointIdxsOfClusters );
    if ( response_hist.empty() ) {
      continue;
    }

    // look at actual matching...
    /*{
      Mat out;
      drawKeypoints( img, keypoints, out, Scalar(0,0,255) );
      for ( unsigned int i = 0; i < pointIdxsOfClusters.size(); i++ ) {
	 if( pointIdxsOfClusters[i].size() > 0 ) {
	    Scalar clr( i/1000.0*255.0, 0, 0 );
	    for ( unsigned int j = 0; j < pointIdxsOfClusters[i].size(); j++ ) {
	      circle( out, keypoints[pointIdxsOfClusters[i][j]].pt, 1, clr, 2 );
	    }
	  }
	}
        imshow( "matches", out );
        waitKey(0);
      }*/
          
		
    // test vs. SVMs
    float minf = FLT_MAX;
    //float maxf = FLT_MIN;
    string minClass, maxClass, testClass;
    for ( vector<string>::iterator it = testClasses.begin(); it != testClasses.end(); ++it ) {
      testClass = *it;

      // Read classifier from disk
      string svmFilename = "/home/lindo/dev/opencv-test/bow/CLASS/class_" + testClass + ".xml";
      Ptr<ml::SVM> svm = Algorithm::load<ml::SVM>( svmFilename );
      if( !svm ) {
        cerr << "svm could not be loaded: " << svmFilename << endl;
        exit(EXIT_FAILURE);
      }

      int flag = StatModel::RAW_OUTPUT; // = 0
      // If flag = 0 then predict returns the class label
      // If flag = StatModel::RAW_OUTPUT then predict returns the decision function value
      // The min decision function value is the highest confidence prediction of the "1" (pos) class
      float res = svm->predict( response_hist, noArray(), flag );
      cout << "class: " << testClass << ", response: " << res << endl;

      if ( res < minf ) { // find most neg dec funct value = most likely prediction
        minf = res;
        minClass = testClass;
      }

    }
          
    if ( minClass == actClass ) {
      cout << "+++good prediction: " << minClass
           << ", distance from decision boundry: " << minf << endl;
      good++;
    } else {
      cout << "---bad prediction: " << minClass
           << ", actual: " << actClass
           << ", distance from decision boundry: " << minf << endl;
      bad++;
    }

    confusionMatrix[actClass][minClass]++;
        
  }

  testDataFile.close();

  // Print out some stats.
  cout << "total test samples: " << count << ", ";
  cout << "total classified good: " << fixed << setprecision(2) << 100 * ((float) good / count) << "%, ";
  cout << "total classified bad: " << fixed << setprecision(2) << 100 * ((float) bad / count) << "%, ";
  cout << endl;

  // Print out confusion matrix columns which are the predicted class vlaues
  int colWid = 13; // cout column width for std::setw()
  cout << "confusion matrix: " << endl;
  cout << setw(colWid) << "predicted->";
  for ( vector<string>::iterator it = testClasses.begin(); it != testClasses.end(); ++it ) {
    cout << setw(colWid) << *it;
  }
  cout << endl;

  // Print out confusion matrix, rows are actual classes, columns are predicted classes
  for ( vector<string>::iterator it = testClasses.begin(); it != testClasses.end(); ++it ) {
    string actClass = *it; // row
    cout << setw(colWid) << actClass;
    for ( vector<string>::iterator it1 = testClasses.begin(); it1 != testClasses.end(); ++it1 ) {
      string predClass = *it1; // column
      cout << setw(colWid) << confusionMatrix[actClass][predClass];
    }
    cout << endl;
  }

  return (0);
}
