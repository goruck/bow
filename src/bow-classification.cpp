/*
 *  bow-classification.cpp
 *
 *  Created by Roy Shilkrot on 8/19/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 *  Adpated and updated for OpenCV 3.2 by Lindo St. Angel 2017.
 *
 */

/* compile with
gcc -Wall bow-classification.cpp -I/usr/local/include/ \
    -L/usr/lib/ -lstdc++ \
    -L/usr/local/lib -lopencv_highgui -lopencv_core -lopencv_imgcodecs \
    -lopencv_imgproc -lopencv_videoio -lopencv_video -lopencv_videostab \
    -lopencv_flann -lopencv_ml -lm -lopencv_xfeatures2d -lopencv_features2d \
    -o bow-classification.out -std=c++11
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
        
  string dir = "/home/lindo/dev/opencv-test/bow/TRAIN", filepath;
  DIR *dp = NULL;
  struct dirent *dirp;
  struct stat filestat;

  // Open directory containing training images
  dp = opendir( dir.c_str() );
  if (dp == NULL) {
    cerr << "could not open directory of training images" << endl;
    exit(EXIT_FAILURE);
  }

  // Step 1: Extract features of choice from training set that contains all classes
  // (features are the descriptors of the training images' keypoints)
	
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
	
  cout << "------- extract descriptors -------" << endl;
  int count = 0;
  vector<KeyPoint> keypoints;
  Mat descriptors;
  Mat training_descriptors(1, detector->descriptorSize(), detector->descriptorType());
  Rect clipping_rect = Rect(0,120,640,480-120);
  Mat bg_ = imread("background.png"), img_fg, img;
  Mat bgc = bg_(clipping_rect);
  while ( (dirp = readdir( dp )) ) {
    //if (count++ > 15) break;

    filepath = dir + "/" + dirp->d_name;
		
    // If the file is a directory (or is in some way invalid) we'll skip it 
    if (stat( filepath.c_str(), &filestat )) continue;
    if (S_ISDIR( filestat.st_mode ))         continue;
		
    img = imread(filepath);
    if( img.empty() ) { // Check for invalid input
      cout << endl << "Could not open or find image, skipping " << filepath << endl;
      continue;
    }

    img = img(clipping_rect);
    img_fg = img - bgc;

    // Compute keypoints amd thier associated descriptors
		  
    detector->detectAndCompute( img_fg, noArray(), keypoints, noArray(), false );
    if ( keypoints.empty() ) {
      continue;
    }

    /*{
        Mat out; //img_fg.copyTo(out);
        drawKeypoints(img_fg, keypoints, out, Scalar(255));
        imshow("fg",img_fg);
        imshow("keypoints", out);
        waitKey(0);
      }*/

    detector->detectAndCompute( img, noArray(), keypoints, descriptors, false ); // compute descriptors
    //detector->detectAndCompute( img_fg, noArray(), keypoints, descriptors, false ); // compute descriptors
    if ( descriptors.empty() ) {
      continue;
    }
		
    training_descriptors.push_back(descriptors);
    cout << ".";
  }

  cout << endl;
  closedir( dp );

  cout << "Total training descriptors: " << training_descriptors.rows << endl;
  // Save to disk
  FileStorage fs("descriptors.xml", FileStorage::WRITE);
  fs << "descriptors" << descriptors;
  fs.release();

  // Step 2: Create a vocabulary of features by clustering the features
  // (use BOW trainer to figure out which keypoints seem to form meaningful clusters)

  cout << "------- create vocabulary ---------" << endl;
    
  int clusterCount = 1000; //num clusters = size of vocabulary
  int attempts = 3;
  //int flags = cv::KMEANS_PP_CENTERS; // can cause segmentation fault
  int flags = cv::KMEANS_RANDOM_CENTERS;
  TermCriteria terminate_criterion( TermCriteria::EPS | TermCriteria::COUNT, 10, 1.0 );
  BOWKMeansTrainer bowTrainer( clusterCount, terminate_criterion, attempts, flags );
  bowTrainer.add(training_descriptors);
  cout << "cluster BOW features" << endl;
  Mat vocabulary = bowTrainer.cluster();
  
  // Save vocabulary to disk
  FileStorage fs1("vocabulary.xml", FileStorage::WRITE);
  fs1 << "vocabulary" << vocabulary;
  fs1.release();

  // Step 3: Train classifier

  // Setup extractor to convert images to presence vectors
  Ptr< cv::DescriptorMatcher > descMatcher;
  descMatcher = DescriptorMatcher::create( "BruteForce" );

  Ptr<SURF> descExtractor = SURF::create( hessianThreshold, nOctaves, nOctaveLayers, extended, upright );

  Ptr< cv::BOWImgDescriptorExtractor > bowExtractor;
  bowExtractor = new BOWImgDescriptorExtractor( descExtractor, descMatcher );
  bowExtractor->BOWImgDescriptorExtractor::setVocabulary( vocabulary );

  // Setup training data for classifiers
  map<string, Mat> classes_training_data;
  classes_training_data.clear();

  cout << "------- create presence vectors ---------" << endl;

  Mat response_hist; // histogram of the presence (or absence) of each word in vocabulary
  count = 0;
  char buf[255];
  //ifstream ifs("/home/lindo/dev/opencv-test/bow/training.txt");
  //ifstream ifs("/home/lindo/dev/opencv-test/bow/train.txt"); // with food names, manually classified
  ifstream ifs("/home/lindo/dev/opencv-test/bow/training-names.txt"); // with food names, converted from codes
  int total_samples = 0;

  do {
    ifs.getline(buf, 255);
    string line(buf);
    istringstream iss(line);
    //cout << line << endl;
    iss >> filepath; // path to image name
    Rect r; char delim;
    iss >> r.x >> delim; // x coord of region of interest
    iss >> r.y >> delim; // y coord of region of interest
    iss >> r.width >> delim; // width of region of interest
    iss >> r.height; // height of region of interest
    //cout << r.x << "," << r.y << endl;
    string class_;
    iss >> class_; // class number
    //cout << "class_ " << class_ << endl;

    if (class_ == "") {
      cout << endl << "Image without class id, skipping" << endl;
      continue;
    }

    img = imread(filepath);
    if( img.empty() ) { // Check for invalid input
      cout << endl << "Could not open or find image, skipping" << filepath << endl;
      continue;
    }

    /*{
      rectangle(img, r, Scalar(0,255,0), 2, 8, 0); // draw region of interest on image
      imshow("before crop", img);
    }*/

    r &= Rect(0, 0, img.cols, img.rows); // region of interest in terms of img rows and cols
    if(r.width != 0) {
      img = img(r); // crop to interesting region
    }

    /*{
      putText(img, class_, Point(20,20), CV_FONT_HERSHEY_PLAIN, 2.0, Scalar(255), 2); // display img w/class #
      imshow("after crop", img);
    }*/

    detector->detectAndCompute( img, noArray(), keypoints, noArray() );
    if ( keypoints.empty() ) {
      continue;
    }
    //cout << endl << "number of keypoints in training image: " << keypoints.size() << endl;

    // create presence vector which will be normalized to # of keypoints in image
    bowExtractor->BOWImgDescriptorExtractor::compute( img, keypoints, response_hist );
    if ( response_hist.empty() ) {
      continue;
    }
    //cout << endl << "train response histogram: " << response_hist << endl;
		
    if(classes_training_data.count(class_) == 0) { // class not yet created...
      classes_training_data[class_].create( 0, response_hist.cols, response_hist.type() );
    }
	  
    classes_training_data[class_].push_back( response_hist );

    total_samples++;

    cout << ".";

    //waitKey(0);

  } while (!ifs.eof());

  cout << endl;
  cout << "Number of presence vectors: " << total_samples << endl;
	
  // Train 1-vs-all SVMs
  Ptr<SVM> svm = SVM::create();
  svm->setType(SVM::C_SVC);
  svm->setKernel(SVM::RBF);
  //svm->setKernel(SVM::LINEAR); // RBF vs LINEAR seems not to matter much
  svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, FLT_EPSILON));

  map<string, Mat>::iterator it;
  for (it = classes_training_data.begin(); it != classes_training_data.end(); ++it) {
    string class_ = it->first;
		
    Mat samples(0, response_hist.cols, response_hist.type());
    Mat labels(0, 1, CV_32SC1);
		
    // Copy class sample and label
    cout << "adding " << classes_training_data[class_].rows << " positive" << endl;
    samples.push_back(classes_training_data[class_]);
    Mat class_label = Mat::ones(classes_training_data[class_].rows, 1, CV_32SC1);
    labels.push_back(class_label);

    // Copy rest of samples and labels
    map<string, Mat>::iterator it1;
    for (it1 = classes_training_data.begin(); it1 != classes_training_data.end(); ++it1) {
      string not_class_ = it1->first;
      if ( not_class_ == class_ ) continue;
      samples.push_back(classes_training_data[not_class_]);
      class_label = Mat::zeros(classes_training_data[not_class_].rows, 1, CV_32SC1);
      labels.push_back(class_label);
    }
		
    Mat samples_32f;
    samples.convertTo(samples_32f, CV_32F);
    if ( samples.rows == 0 ) {
      cout << "No rows in samples, skipping" << endl;
      continue;
    }

    //cout << "class: " << class_ << endl;
    //cout << "samples: " << samples_32f << endl;
    //cout << "labels: " << labels << endl;

    // Construct training data from samples read from file above
    Ptr<TrainData> td = TrainData::create(
                                             samples_32f, // Array of samples
                                             ROW_SAMPLE,  // Data in rows
                                             labels,      // Array of responses
                                             noArray(),   // Use all features
                                             noArray(),   // Use all data points
                                             noArray(),   // Do not use samples weights
                                             noArray()    // Do not specify inp and out types
			                   );

    cout << "svm training for class: " << class_ << " starting" << endl;

    // Auto train using k-fold cross-validation
    svm->trainAuto(
                      td,
	              10,
                      SVM::getDefaultGrid(SVM::C),
                      SVM::getDefaultGrid(SVM::GAMMA),
                      SVM::getDefaultGrid(SVM::P),
                      SVM::getDefaultGrid(SVM::NU),
                      SVM::getDefaultGrid(SVM::COEF),
                      SVM::getDefaultGrid(SVM::DEGREE),
                      false
                    );

    cout << "svm training for class: " << class_ << " completed" << endl;

    string svmFilename = "/home/lindo/dev/opencv-test/bow/CLASS/class_" + class_ + ".xml";
    svm->save( svmFilename );

    cout << "saved svm classifier to file" << endl;

    // Prepare for the next run. 
    svm->clear();
    samples.release();
    labels.release();
    class_label.release();
  }
	
  cout << "done" << endl;

  return 0;

}
