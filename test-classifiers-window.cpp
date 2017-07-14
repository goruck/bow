// test classifiers created with bow-classification.cpp
// with windowing

/* compile with
gcc -Wall test-classifiers-window.cpp -I/usr/local/include/ \
    -L/usr/lib/ -lstdc++ \
    -L/usr/local/lib -lopencv_highgui -lopencv_core -lopencv_imgcodecs \
    -lopencv_imgproc -lopencv_videoio -lopencv_video -lopencv_videostab \
    -lopencv_flann -lopencv_ml -lm -lopencv_xfeatures2d -lopencv_features2d \
    -o test-classifiers-window.out -std=c++11
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
  double hessianThreshold = 400;
  //double hessianThreshold = 150;
  int nOctaves = 4, nOctaveLayers = 3;
  bool extended = false, upright = false;
  Ptr<SURF> detector = SURF::create( hessianThreshold, nOctaves, nOctaveLayers, extended, upright );
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

  // evaluate
  int count = 0;
  int good = 0, bad = 0;

  // Open test data file that points to test images
  // Format is "FILENAME X,Y,W,H CLASS1 CLASS2..."
  ifstream testDataFile; // object for handling file input
  //testDataFile.open( "/home/lindo/dev/opencv-test/bow/test_one_class_rect.txt" );
  //testDataFile.open( "/home/lindo/dev/opencv-test/bow/train.txt" );
  //testDataFile.open( "/home/lindo/dev/opencv-test/bow/test-no-rects-subset.txt" );
  testDataFile.open( "/home/lindo/dev/opencv-test/bow/test-no-rects.txt" );
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
    //iss.ignore( 256, ',' ); // skip x coord
    //iss.ignore( 256, ',' ); // skip y coord
    //iss.ignore( 256, ',' ); // skip width
    //iss.ignore( 256, ' ' ); // skip height

    while ( !iss.eof() ) {
      string actClass;
      iss >> actClass; // read actual class(es) of image

      if ( actClass == "" ) continue; // skip images w/o class information

      if ( find(testClasses.begin(), testClasses.end(), actClass) != testClasses.end() )
        continue; // class already present in vector, skip
    
      testClasses.push_back(actClass); // record class
    }

  }

  cout << "Read " << testClasses.size() << " actual test classes." << endl;

  count = 0;
  int numObj = 0;
  testDataFile.clear();    // clear eof errors (if any)
  testDataFile.seekg( 0 ); // rewind the stream to the beginning
  vector<string> actClasses; // actual class(es) in a test image
  Mat background = imread("background.png");
  while ( getline(testDataFile, line) ) {
    //if (count++ > 15) break;

    istringstream iss(line);
    string filepath;
    iss >> filepath; // path to image name
    cout << "eval file: " << filepath << endl;
    //Rect r; char delim;
    //iss >> r.x >> delim; // x coord of region of interest
    //iss >> r.y >> delim; // y coord of region of interest
    //iss >> r.width >> delim; // width of region of interest
    //iss >> r.height; // height of region of interest

    cout << "actual class(es): ";
    actClasses.clear();
    while ( !iss.eof() ) {
      string actClass;
      iss >> actClass; // read actual class(es) of test image
      cout << actClass << " ";

      if ( actClass == "" ) continue;
    
      actClasses.push_back(actClass); // record actual class(es)

      numObj++; // number of total objects to be classified
    }
    cout << endl;

    if ( actClasses.empty() ) continue;
			
    Mat img = imread(filepath);
    if( img.empty() ) { // Check for invalid input
      cout << endl << "Could not open or find image, skipping: " << filepath << endl;
      continue;
    }

    if( img.size() != Size(640,480) ) {
      cout << "image not 640x480, skipping" << endl;
      continue;
    }

    Mat diff = ( img - background ), diff_8UC1;
	
    cvtColor(diff, diff_8UC1, CV_BGR2GRAY);
    //imshow("img no back", diff_8UC1);
    Mat fg_mask = (diff_8UC1 > 5);
    GaussianBlur(fg_mask, fg_mask, Size(11,11), 5.0);
    fg_mask = (fg_mask > 50);
	
    /*{
      Mat out; img.copyTo(out, fg_mask);
      imshow("foreground", out);
      imshow("to scan",img);
      waitKey(0);
    }*/

    Rect crop_rect(0,100,640,480-100); //crop off top section
    img = img(crop_rect);
    fg_mask = fg_mask(crop_rect);

    Mat copy; cvtColor( img, copy, CV_BGR2HSV );

    vector<Point> check_points;
    //Sliding window
    int winsize = 200;
    map<string,pair<int,float> > found_classes;
    for ( int x = 0; x < img.cols; x += winsize/4 ) {
      for ( int y = 0; y < img.rows; y += winsize/4 ) {
        if ( fg_mask.at<uchar>(y,x) == 0 ) {
          continue;
        }
        check_points.push_back(Point(x,y));
      }
    }
	
    cout << "to check: " << check_points.size() << " points"<<endl;

    map<string, pair<int, float>> foundClasses;
    foundClasses.clear();
    for ( unsigned int i = 0; i < check_points.size(); i++ ) {
      int x = check_points[i].x;
      int y = check_points[i].y;
      Mat wimg,response_hist;
      img(Rect( x-winsize/2, y-winsize/2, winsize, winsize ) & Rect( 0, 0, img.cols, img.rows) ).copyTo( wimg );

      
      /*{
        imshow("windowed image", wimg);
        waitKey(0);
      }*/
		
      vector<KeyPoint> keypoints;

      detector->detectAndCompute( wimg, noArray(), keypoints, noArray() );
      if ( keypoints.empty() ) {
        continue;
      }

      //vector< vector<int> > pointIdxsOfClusters;

      bowExtractor->BOWImgDescriptorExtractor::compute( wimg, keypoints, response_hist/*, &pointIdxsOfClusters*/ );
      if ( response_hist.empty() ) {
        continue;
      }

      // look at actual matching...
      /*{
       Mat out;
        drawKeypoints( wimg, keypoints, out, Scalar(0,0,255) );
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
      string minClass;
      for ( vector<string>::iterator it = testClasses.begin(); it != testClasses.end(); ++it ) {
        string testClass = *it;

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
        //cout << "class: " << testClass << ", response: " << res << endl;

        if ( (testClass == "misc") && (res > 0.9) ) continue;

        if ( res > 1.0 ) continue;

        if ( res < minf ) { // find most neg dec funct value = most likely prediction
          minf = res;
          minClass = testClass;
        }

      }

      foundClasses[minClass].first++;
      foundClasses[minClass].second += minf;

    }

    float max_class_f = FLT_MIN, max_class_f1 = FLT_MIN; string max_class, max_class1;
    vector<float> scores;
    for (map<string,pair<int,float> >::iterator it=foundClasses.begin(); it != foundClasses.end(); ++it) {
      // (*it).first = class, (*it).second.first = #num occurances of class, (*it).second.second = cum distance metric
      float score = sqrtf((float)((*it).second.first*(*it).second.first + (*it).second.second*(*it).second.second));

      if ( score > 1e+10 ) continue; // impossible score, skip

      scores.push_back(score);

      cout << (*it).first << "(" << score << "," << (*it).second.first << "," << (*it).second.second << ")";
      cout << endl;
      if(score > max_class_f) { //1st place thrown off
        max_class_f1 = max_class_f;
        max_class1 = max_class;
			
        max_class_f = score;
        max_class = (*it).first;
      } else if (score >  max_class_f1) { //2nd place thrown off
        max_class_f1 = score;
        max_class1 = (*it).first;
      }

    }

    cout << "max_class: " << max_class << ", " << "max_class1: " << max_class1 << endl;

    // Check for matches between actual class(es) in test image and up to the 2 highest scored predicitons.
    for ( vector<string>::iterator it = actClasses.begin(); it != actClasses.end(); ++it ) {
      if ( (*it) == max_class ) { // found a match with highest scored prediction
        good++;
        confusionMatrix[(*it)][max_class]++;
      } else if ( /*((max_class_f - max_class_f1) < 10) &&*/ ((*it) == max_class1) ) { // found a match w/2nd highest score pred
        good++;
        confusionMatrix[(*it)][max_class1]++;
      } else { // no match
        bad++;
        confusionMatrix[(*it)][max_class]++; // choose max_class as the (incorrect) prediction
      }
    }

  }

  testDataFile.close();

  // Print out some stats.
  cout << "total test samples: " << numObj << ", ";
  cout << "total classified good: " << fixed << setprecision(2) << 100 * ((float) good / numObj ) << "%, ";
  cout << "total classified bad: " << fixed << setprecision(2) << 100 * ((float) bad / numObj ) << "%, ";
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
