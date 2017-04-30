#include "opencv2/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include "dirent.h"

using namespace std;
using namespace cv;


/*
   The cascade classifiers that come with opencv are kept in the
   following folder: bulid/etc/haarscascades
   Set OPENCV_ROOT to the location of opencv in your system
*/
double const NORM_WIDTH = 500.0f;
double const FACE_NORM_WIDTH = 215;


string cascades = "/usr/local/Cellar/opencv/2.4.13.2/share/OpenCV/haarcascades/";
string FACES_CASCADE_NAME = cascades + "haarcascade_frontalface_alt.xml";
string EYES_CASCADE_NAME = cascades + "haarcascade_eye.xml";
string LEFT_EYES_CASCADE_NAME = cascades + "haarcascade_mcs_lefteye.xml";
string RIGHT_EYES_CASCADE_NAME = cascades + "haarcascade_mcs_righteye.xml";

void drawEllipse(Mat frame, const Rect rect, int r, int g, int b) {
  int width2 = rect.width/2;
  int height2 = rect.height/2;
  Point center(rect.x + width2, rect.y + height2);
  ellipse(frame, center, Size(width2, height2), 0, 0, 360,
          Scalar(r, g, b), 2, 8, 0 );
}

Rect coordinate_Transfer_toOrigin_Image(Rect face_resized, double resize_factor){
  Rect origin = Rect( face_resized.x * 1.0 / resize_factor,
                      face_resized.y * 1.0 / resize_factor,
                      face_resized.width * 1.0 / resize_factor,
                      face_resized.height * 1.0 / resize_factor
                      );
  return origin;
}


bool detectWink(Mat frame, Point location, Mat detect_image, CascadeClassifier cascade, double face_norm_Scale) {
  // frame,ctr are only used for drawing the detected eyes
  vector<Rect> eyes;
//  cascade.detectMultiScale(detect_image, eyes, 1.1, 5, 0, Size(FACE_NORM_WIDTH / 8, FACE_NORM_WIDTH / 16));
//  cascade.detectMultiScale(detect_image, eyes, 1.1, 7, 0, Size(30, 30));
  cascade.detectMultiScale(detect_image, eyes, 1.1, 6, 0, Size(40, 40));
//  cascade.detectMultiScale(detect_image, eyes, 1.09, 7, 0, Size(FACE_NORM_WIDTH * 8.0 / 65.0, FACE_NORM_WIDTH * 9.0 / 65.0));

  int neyes = (int)eyes.size();
  for( int i = 0; i < neyes ; i++ ) {
    Rect eyes_i = eyes[i];

    Rect origin_eye_postion = Rect( eyes_i.x / face_norm_Scale,
                                    eyes_i.y / face_norm_Scale,
                                    eyes_i.width / face_norm_Scale,
                                    eyes_i.height / face_norm_Scale
                                    );
    // draw detected eyes with blue circle
    drawEllipse(frame, origin_eye_postion + location, 255, 255, 0);
  }
  return(neyes == 1);
}

// you need to rewrite this function
int detect(Mat frame,
           CascadeClassifier cascade_face, CascadeClassifier cascade_eyes) {




//  equalizeHist(frame_gray, frame_gray); // input, outuput
//  medianBlur(frame_gray, frame_gray, 5); // input, output, neighborhood_size
//  blur(frame_gray, frame_gray, Size(5,5), Point(-1,-1));
/*  input,output,neighborood_size,center_location (neg means - true center) */
/*double scaling = 1.05;
  Size s = frame.size();
  if (s.width > 500 || s.height > 500) {
    scaling = 1.50;
  }  */
  
  Size s = frame.size();
  double const RESIZE_SCALE = NORM_WIDTH / s.width;
  double const NORM_HEIGHT = RESIZE_SCALE * s.height;

  // Normalize all image to size of Norm_width  Norm_height
  cout << s.width << " " << RESIZE_SCALE << " " << s.height << endl;
  Mat frame_normalized(Size(NORM_WIDTH, NORM_HEIGHT), frame.type());
  resize(frame, frame_normalized, frame_normalized.size(), 0, 0, cv::INTER_LINEAR);
//  imshow("resized", frame_normalized);


  // Transfer image to gray image and detect all the faces
  Mat frame_normalized_gray;
  vector<Rect> faces;  

  cvtColor(frame_normalized, frame_normalized_gray, CV_BGR2GRAY);  
  // cascade_face.detectMultiScale(frame_normalized_gray, faces,
  //                               1.02, 7, 0|CV_HAAR_SCALE_IMAGE, Size(80, 80)); //
  cascade_face.detectMultiScale(frame_normalized_gray, faces,
                                1.03, 4, 0|CV_HAAR_SCALE_IMAGE, Size(70, 70)); //

  
  /* frame_gray - the input image
     faces - the output detections.
     1.1 - scale factor for increasing/decreasing image or pattern resolution
     3 - minNeighbors.
         larger (4) would be more selective in determining detection
	 smaller (2,1) less selective in determining detection
	 0 - return all detections.
     0|CV_HAAR_SCALE_IMAGE - flags. This flag means scale image to match pattern
     Size(30, 30)) - size in pixels of smallest allowed detection  */

  int detected = 0;

  int nfaces = (int)faces.size();
  for( int i = 0; i < nfaces ; i++ ) {
    // Rect face = Rect( faces[i].x,
    //                   faces[i].y,
    //                   faces[i].width,
    //                   faces[i].height * 0.75
    //   );
    Rect face = Rect( faces[i].x,
                      faces[i].y,
                      faces[i].width,
                      faces[i].height * 0.6
      );


    /// transfer x y position to original not normalized image and draw a circle
    Rect origin_face_position = coordinate_Transfer_toOrigin_Image(face, RESIZE_SCALE);


    /// extract face from original image and normalize it to specific size 
//    Mat faceROI(frame, origin_face_position);

    Mat faceROI(frame_normalized_gray, face); // extract from normalized one
    Mat Gray_Extracted_Face = faceROI;
//    cvtColor(faceROI, Gray_Extracted_Face, CV_BGR2GRAY);  
    //--- normalize face for future detection
    double const FACE_NORM_SCALE = FACE_NORM_WIDTH / Gray_Extracted_Face.size().width;
    double const FACE_NORM_HEIGHT = FACE_NORM_SCALE * Gray_Extracted_Face.size().height;
    Mat Normlized_GrayFace_ForDetect(Size(FACE_NORM_WIDTH, FACE_NORM_HEIGHT), faceROI.type());

    resize(Gray_Extracted_Face, Normlized_GrayFace_ForDetect, Normlized_GrayFace_ForDetect.size(),
            0, 0, cv::INTER_CUBIC);

//    resize(faceROI, dst, dst.size(), 0, 0, cv::INTER_CUBIC);
    imshow("face_Normalized" + to_string(i), Normlized_GrayFace_ForDetect);
    cout << "face " << i << " :" << face.width << " , " << face.height << endl;

    //  from normalize the gray face image, we detect eyes on it.
    if(detectWink(frame, Point(origin_face_position.x, origin_face_position.y),
                   Normlized_GrayFace_ForDetect, cascade_eyes, FACE_NORM_SCALE * RESIZE_SCALE)) {
      drawEllipse(frame, origin_face_position, 0, 255, 0);
      detected++;
    }
    else
        drawEllipse(frame, origin_face_position, 255, 0, 255);  // contains no open eyes, or more than 1 eyes, draw purple
  }

  // int detected = 0;

  // int nfaces = (int)faces.size();
  // for( int i = 0; i < nfaces ; i++ ) {
  //   Rect face = faces[i];
  //   drawEllipse(frame, face, 255, 0, 255);
  //   Mat faceROI = frame_gray(face);
  //   Mat dst(Size(150, 150), faceROI.type());
  //   resize(faceROI, dst, dst.size(), 0, 0, cv::INTER_CUBIC);
  //   imshow("faceROI" + to_string(i), dst);
  //   cout << "face " << i << " :" << face.width << " , " << face.height << endl;
  //   if(detectWink(frame, Point(face.x, face.y), dst, cascade_eyes)) {
  //     drawEllipse(frame, face, 0, 255, 0);
  //     detected++;
  //   }
  // }
  return(detected);
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
  int number = 1;
  while (!finish && (entry = readdir(dir)) != NULL) {
    char *name = entry->d_name;
    string dname = folder + name;
    Mat img = imread(dname.c_str(), CV_LOAD_IMAGE_UNCHANGED);
    if(!img.empty()) {
      int d = detect(img, cascade1, cascade2);
      cerr << "Picture: " << number++ << "  " << d << " detections" << endl;
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
  CascadeClassifier faces_cascade, eyes_cascade;

  if(
    !faces_cascade.load(FACES_CASCADE_NAME)
    || !eyes_cascade.load(EYES_CASCADE_NAME)) {
    cerr << FACES_CASCADE_NAME << " or " << EYES_CASCADE_NAME
         << " are not in a proper cascade format" << endl;
    return(-1);
  }

  int detections = 0;
  if(argc == 2) {
    detections = runonFolder(faces_cascade, eyes_cascade, foldername);
    cout << "Total of " << detections << " detections" << endl;
  }
  else runonVideo(faces_cascade, eyes_cascade);

  return(0);
}
