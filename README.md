# ⏰ Divide Equipment 안전 장비 분류

22년도 2학기 IT정보제어공학과 스마트비전 기말 프로젝트  
안전 장비를 분류하여 위험도에 따라 색을 분류하고 촬영하는 프로그램
---
  
* #### :computer: Environment
  * Language: C++
  * IDE: Visual Studio
  * Yolov4 / Darknet / OpenCV / YoloLabel
  * Developer: 김주하
  
  
* #### 💡 Data Set
  * Class Num: 5
  * Class Name: Person, Helmet, Vest, Boots, Glove
  * Batch / Subdivision: 64 / 32
  * Max_Batches / Step: 15000 / 12000, 13500
  * Filters: 30
  * Width & Height: 320
  * Number of dataset: About 1000 pictures

  
---


<!-------------------------------------------------------------Part 1------------------------------------------------------------------------------------------>

 ## 1. 블록도 설명

![블록도](./MainIMG/블록도.JPG)  
  

 1. __데이터 학습 및 추론__  
    * YoloLabel로 라벨링된 데이터를 다크넷으로 학습시킴   
    * 예측 완료된 객체 정보를 person 안의 안전 장비를 분류하는 함수로 보냄

    
 2. __안전 장비 분류__    
    * 내부 안전 장비에 따라서 색상을 분류함
    * 분류 결과를 저장하여 촬영 함수에서 사용할 수 있도록 함
    
    
 3. __화면 출력__    
    * 분류된 결과에 따라 바운딩 박스가 그려진 화면을 출력함
    
    
---


<!-------------------------------------------------------------Part 2------------------------------------------------------------------------------------------>

 ## 2. 코드 설명
 ### 메인 외 선언</br>
  * 여러 함수들에서 사용해야 하는 것들과 포함해야 되는 것들을 외부에 선언해 줌
  
       ```c++
          #define _CRT_SECURE_NO_WARNINGS
          #include <iostream>
          #include <opencv2/opencv.hpp>
          #include <ctime>
          using namespace std;
          using namespace cv;
          using namespace cv::dnn;
          const float CONFIDENCE_THRESHOLD = 0.5;
          const float NMS_THRESHOLD = 0.5;
          const int NUM_CLASSES = 5;
          const Scalar colors[] = {
            {0, 0, 255},{0, 94, 255}, {166,97,243},{22, 219, 29}};//바운딩 박스를 그릴 빨강, 주황, 핑크, 초록 색상
          Mat frame, blob;
          vector<Mat> detections;
          vector<int> indices[NUM_CLASSES];
          vector<Rect> boxes[NUM_CLASSES];
          vector<float> scores[NUM_CLASSES];
          bool wear;

          void trainandpredict();
          void findClass();
          void divide();
          void print();
          void cheese();
 

       ```
### 메인 함수</br>
  * 카메라 입력을 받아오는 것과 블롭화, 딜레이 기능을 제외한 기능을 함수로 선언해 줌
  
       ```c++ 
          int main()
          {
             VideoCapture cap(0);
             if (!cap.isOpened()) {
               cerr << "Camera open failed!" << endl;
               return -1;
             }
             TickMeter tm;
             while (true) {
               static double ms = 0;
               static bool first = false;
               tm.start();
               
               cap >> frame;
               if(frame.empty()) break;
               blobFromImage(frame, blob, 1 / 255.f, Size(320, 320), Scalar(), true, false, CV_32F);
               trainandpredict();//학습 및 추론
               findClass();//객체 검출
               divide();//색상 분류
               print();//출력
               if(!wear) {
                 if(!first) { cheese(); ms = 0;	first = true; }
                 else {
                   if (ms > 60) { cheese(); ms = 0; }
                 }
                }//사진이 찍히면 타이머 0으로 초기화
               tm.stop();
               ms += tm.getTimeSec();//사진이 찍힌 후 시간 계산
               tm.reset();
               if (waitKey(1) == 27) break;
              }
              
            return 0;
          }
       ```
### 학습 및 추론 함수</br>
  * 앞서 학습된 데이터로 추론하고 TickMeter를 이용하여 추론 시간을 측정한 후 출력함 
  
       ```c++ 
          void trainandpredict()
          {
            TickMeter tm;
            auto net = readNetFromDarknet("yolov4-safetyequipment.cfg", "yolov4-safetyequipment_final.weights");
            net.setPreferableBackend(DNN_BACKEND_OPENCV);
            net.setPreferableTarget(DNN_TARGET_CPU);
            auto output_names = net.getUnconnectedOutLayersNames();

            tm.start();
            net.setInput(blob);
            tm.stop();
            double ms = tm.getTimeMilli();
            cout << "연산 속도: " << ms << endl;
            net.forward(detections, output_names);
          }
       ```
       
### 객체 탐지 함수</br>
  * 예측된 객체의 정보(바운딩 박스의 중심축, 크기, 클래스 신뢰도)를 vector에 저장함 
  
       ```c++ 
          void findClass()
          {
            for (auto& output : detections)
            {
              const auto num_boxes = output.rows;
              for (int i = 0; i < num_boxes; i++)
              {
                auto x = output.at<float>(i, 0) * frame.cols;
                auto y = output.at<float>(i, 1) * frame.rows;
                auto width = output.at<float>(i, 2) * frame.cols;
                auto height = output.at<float>(i, 3) * frame.rows;
                Rect rect(x - width / 2, y - height / 2, width, height);
                for (int c = 0; c < NUM_CLASSES; c++)
                {
                  auto confidence = *output.ptr<float>(i, 5 + c);
                  if (confidence >= CONFIDENCE_THRESHOLD)
                  {
                    boxes[c].push_back(rect);
                    scores[c].push_back(confidence);
                  }
                }
              }
            }
          }
       ```
