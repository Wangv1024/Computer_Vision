#include "opencv2/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include "dirent.h"

using namespace std;
using namespace cv;

const double NORM_WIDTH = 500;
const double FACE_WIDTH = 250.0f;
/*
   The cascade classifiers that come with opencv are kept in the
   following folder: bulid/etc/haarscascades
   Set OPENCV_ROOT to the location of opencv in your system
*/
string FACES_CASCADE_NAME = "/usr/local/Cellar/opencv/2.4.13.2/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";

/*  The mouth cascade is assumed to be in the local folder */
string MOUTH_CASCADE_NAME = "Mouth.xml";
string HAND_CASCADE_NAME = "Hand.xml";

Rect transferToOriginalPosition( Rect face_rect, double scale_factor){
  Rect result = Rect(face_rect.x * (1.0f / scale_factor),
                      face_rect.y * (1.0f / scale_factor),
                      face_rect.width * (1.0f / scale_factor), 
                      face_rect.height * (1.0f / scale_factor) );
  return result;
}

void drawEllipse(Mat frame, const Rect rect, int r, int g, int b) {
  int width2 = rect.width/2;
  int height2 = rect.height/2;
  Point center(rect.x + width2, rect.y + height2);
  ellipse(frame, center, Size(width2, height2), 0, 0, 360,
          Scalar(r, g, b), 2, 8, 0 );
}


bool detectSilence(Mat frame, Point location, double scale_factor, Mat ROI, CascadeClassifier cascade)
{
  // frame,location are used only for drawing the detected mouths
  vector<Rect> mouths;
  cascade.detectMultiScale(ROI, mouths, 1.1, 5, 0, Size(50, 50));

  int nmouths = (int)mouths.size();
  for( int i = 0; i < nmouths ; i++ ) {
    Rect mouth_i = Rect( mouths[i].x * (1.0f / scale_factor),
                         mouths[i].y * (1.0f / scale_factor),
                         mouths[i].width * (1.0f / scale_factor),
                         mouths[i].height * (1.0f / scale_factor)
                                  );

    ///  blue circle indicates detected mouth
    drawEllipse(frame, mouth_i + location, 255, 255, 0);
  }
  return(nmouths == 0);
}

bool detectSilence(Mat frame, Point location, Mat ROI, CascadeClassifier cascade)
{
  // frame,location are used only for drawing the detected mouths
  vector<Rect> mouths;
  cascade.detectMultiScale(ROI, mouths, 1.1, 5, 0, Size(10, 10));

  int nmouths = (int)mouths.size();
  for( int i = 0; i < nmouths ; i++ ) {
    Rect mouth_i = mouths[i];
    drawEllipse(frame, mouth_i + location, 255, 255, 0);
  }
  return(nmouths == 0);
}


// you need to rewrite this function
int detect(Mat frame,
           CascadeClassifier cascade_face, CascadeClassifier cascade_mouth) {

//// Normalize the image 
  Size original_size = frame.size();
  double const scale_factor = NORM_WIDTH / original_size.width;
  double const NORM_HEIGHT = original_size.height * scale_factor;

  Mat resized_image(Size(NORM_WIDTH, NORM_HEIGHT), frame.type());
  resize(frame, resized_image, resized_image.size(), 0, 0, cv::INTER_LINEAR);

  Mat resized_gray_image;
  cvtColor(resized_image, resized_gray_image, CV_BGR2GRAY);

  // imshow("Test_Resize", resized_image);
  // imshow("Test_Resize_gray", resized_gray_image);


  vector<Rect> faces;
  cascade_face.detectMultiScale(resized_gray_image, faces,
                                1.1, 3, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50));

  int detected = 0;
  int nfaces = (int)faces.size();
  for( int i = 0; i < nfaces ; i++ ) {
    Rect lower_face = Rect( faces[i].x,
                            faces[i].y + ( faces[i].height * 2 ) / 3,
                            faces[i].width,
                            ( faces[i].height * 1.0 ) / 3 );

    //  Draw purple circle to indicates detected lower face
    Rect original_face_position = transferToOriginalPosition(lower_face, scale_factor);
    drawEllipse(frame, original_face_position, 255, 0, 255); // purple
    
    //// Extract lower part of face from resized image and resize it.
    Mat lower_part_face_from_resize_image(resized_gray_image, lower_face);

    double face_struct_factor = FACE_WIDTH / lower_face.width ;
    double FACE_HEIGHT = FACE_WIDTH / lower_face.width * lower_face.height;
    Mat lower_part_face_resized(Size( FACE_WIDTH, FACE_HEIGHT), lower_part_face_from_resize_image.type());
    resize(lower_part_face_from_resize_image, lower_part_face_resized, lower_part_face_resized.size(),
            0, 0, cv::INTER_CUBIC);

    imshow("Extracted resized lower_face", lower_part_face_resized);

    // if we detect mouth on the face, we draw it out and output number. 
    if(detectSilence(frame,
                     Point(lower_face.x / scale_factor, lower_face.y / scale_factor),
                     scale_factor * face_struct_factor,
                     lower_part_face_resized, cascade_mouth)){
      drawEllipse(frame, original_face_position, 0, 255, 0); // purple
      detected ++;
    }

  }

  return(detected);


  //equalizeHist(frame_gray, frame_gray); // input, outuput
  //medianBlur(frame_gray, frame_gray, 5); // input, output, neighborhood_size
  //blur(frame_gray, frame_gray, Size(5,5), Point(-1,-1));
/*  input,output,neighborood_size,center_location (neg means - true center) */


  /* frame_gray - the input image
     faces - the output detections.
     1.1 - scale factor for increasing/decreasing image or pattern resolution
     3 - minNeighbors.
         larger (4) would be more selective in determining detection
	 smaller (2,1) less selective in determining detection
	 0 - return all detections.
     0|CV_HAAR_SCALE_IMAGE - flags. This flag means scale image to match pattern
     Size(30, 30)) - size in pixels of smallest allowed detection
  */

//   int detected = 0;

//   int nfaces = (int)faces.size();
//   for( int i = 0; i < nfaces ; i++ ) {
//     Rect face = faces[i];
// //    drawEllipse(frame, face, 255, 0, 255);  // purpose circle one on face area
//     int x1 = face.x;
//     int y1 = face.y + ( face.height * 2 ) / 3;
//     Rect lower_face =  Rect(x1, y1, face.width, face.height / 3);
//     drawEllipse(frame, lower_face, 100, 0, 255);  // Red circle on lower face
// //    continue;

//     Mat lower_faceROI = frame_gray(lower_face);
//     if(detectSilence(frame, Point(x1, y1), lower_faceROI, cascade_mouth)) {
//       drawEllipse(frame, face, 0, 255, 0);  // Green circle indicates detected slience sign
//       detected++;
//     }
//   }
//   return(detected);
}


int runonFolder(const CascadeClassifier cascade1,
                const CascadeClassifier cascade2,
                string folder) {
  if(folder.at(folder.length()-1) != '/') folder += '/';
  DIR *dir = opendir(folder.c_str());
  if(dir == NULL) {
    cerr << "Can't open folder " << folder << endl;
    exit(1);
  }
  bool finish = false;
  string windowName;
  struct dirent *entry;
  int detections = 0;
  while (!finish && (entry = readdir(dir)) != NULL) {
    char *name = entry->d_name;
    string dname = folder + name;
    Mat img = imread(dname.c_str(), CV_LOAD_IMAGE_UNCHANGED);
    if(!img.empty()) {
      int d = detect(img, cascade1, cascade2);
      cerr << d << " detections" << endl;
      detections += d;
      if(!windowName.empty()) destroyWindow(windowName);
      windowName = name;
      namedWindow(windowName.c_str(),CV_WINDOW_AUTOSIZE);
      imshow(windowName.c_str(), img);
      int key = cvWaitKey(0); // Wait for a keystroke
      switch(key) {
        case 27 : // <Esc>
          finish = true; break;
        default :
          break;
      }
    } // if image is available
  }
  closedir(dir);
  return(detections);
}

void runonVideo(const CascadeClassifier cascade1,
                const CascadeClassifier cascade2) {
  VideoCapture videocapture(0);
  if(!videocapture.isOpened()) {
    cerr <<  "Can't open default video camera" << endl ;
    exit(1);
  }
  string windowName = "Live Video";
  namedWindow("video", CV_WINDOW_AUTOSIZE);
  Mat frame;
  bool finish = false;
  while(!finish) {
    if(!videocapture.read(frame)) {
      cout <<  "Can't capture frame" << endl ;
      break;
    }
    detect(frame, cascade1, cascade2);
    imshow("video", frame);
    if(cvWaitKey(30) >= 0) finish = true;
  }
}

int main(int argc, char** argv) {
  if(argc != 1 && argc != 2) {
    cerr << argv[0] << ": "
         << "got " << argc-1
         << " arguments. Expecting 0 or 1 : [image-folder]"
         << endl;
    return(-1);
  }

  string foldername = (argc == 1) ? "" : argv[1];
  CascadeClassifier faces_cascade, mouth_cascade;

///   Changes to use hand to detect 
  // if(
  //   !faces_cascade.load(FACES_CASCADE_NAME)
  //   || !mouth_cascade.load(MOUTH_CASCADE_NAME)) {
  //   cerr << FACES_CASCADE_NAME << " or " << MOUTH_CASCADE_NAME
  //        << " are not in a proper cascade format" << endl;
  //   return(-1);
  // }

  //   Changes to use hand to detect 
  if(
    !faces_cascade.load(FACES_CASCADE_NAME)
    || !mouth_cascade.load(MOUTH_CASCADE_NAME)) {
    cerr << FACES_CASCADE_NAME << " or " << MOUTH_CASCADE_NAME
         << " are not in a proper cascade format" << endl;
    return(-1);
  }

  int detections = 0;
  if(argc == 2) {
    detections = runonFolder(faces_cascade, mouth_cascade, foldername);
    cout << "Total of " << detections << " detections" << endl;
  }
  else runonVideo(faces_cascade, mouth_cascade);

  return(0);
}
