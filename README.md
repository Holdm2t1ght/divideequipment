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
### 안전 장비 분류 함수</br>
  * Person 클래스가 검출되면 클래스의 바운딩 박스 안에 있는 안전 장비를 파악하여 Person 바운딩 박스의 색을 변경함 
  
       ```c++ 
          void divide()
          {
            vector<string> class_names = { "person","helmet", "vest", "boots", "glove" };
            for (int c = 0; c < NUM_CLASSES; c++)
              NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);
            wear = 1;

            for (int c = 0; c < NUM_CLASSES; c++)
            {
              for (int i = 0; i < indices[c].size(); ++i)
              {
                auto idx = indices[c][i];
                const auto& rect = boxes[c][idx];
                if (c == 0)
                {
                  int n = 0;//기본 빨강색
                  bool equipment[4] = { 0,0,0,0 };
                  bool important = 0;
                  for (int d = 1; d < NUM_CLASSES; d++)
                  {
                    for (int j = 0; j < indices[d].size(); ++j)
                    {
                      auto idx_equip = indices[d][j];
                      const auto& rect_equip = boxes[d][idx_equip];
                      int x = rect_equip.x + (rect_equip.width / 2);
                      int y = rect_equip.y + (rect_equip.height / 2);
                      if (x >= rect.x && x <= rect.x + rect.width && y >= rect.y && y <= rect.y + rect.height)
                        equipment[d - 1] = true;//각각의 장비가 있으면 true로 표시
                    }
                  }
                  if (equipment[0] && equipment[1]) {//안전모, 안전조끼 둘 다 있으면 important[1] = true;
                    important = true;
                  }
                  else wear = 0;

                  if (important)//안전모와 조끼가 있을 때
                  {
                    if (equipment[2] && equipment[3]) n = 3;//장갑, 장화 둘 다 있으면 초록
                    else if (equipment[2] || equipment[3]) n = 2;//장갑, 장화 둘 중 하나만 있으면 핑크
                    else n = 1;//장갑, 장화 둘 다 없으면 주황
                  }//안전모와 조끼 중 하나라도 없으면 빨강

                  rectangle(frame, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height),colors[n], 2);
                  string label_str = class_names[c] + ": " + format("%.02lf", scores[c][idx]);
                  int baseline;
                  auto label_bg_sz = getTextSize(label_str, FONT_HERSHEY_PLAIN, 1, 2, &baseline);
                  rectangle(frame, Point(rect.x, rect.y - label_bg_sz.height - baseline), Point(rect.x + label_bg_sz.width, rect.y), colors[n], FILLED);
                  putText(frame, label_str, Point(rect.x, rect.y - baseline), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 0));
                }
              }
            }
          }
       ```
### 화면 출력 함수</br>
  * time_h 객체를 선언한 후 출력되는 화면에 현재 시간을 함께  
  
       ```c++ 
          void print()//화면 출력 함수
          {
            time_t timer;
            struct tm* t;
            timer = time(NULL); // 1970년 1월 1일 0시 0분 0초부터 시작하여 현재까지의 초
            t = localtime(&timer);
            string cur_time = to_string(t->tm_year + 1900) + "." + to_string(t->tm_mon + 1) + "."
              + to_string(t->tm_mday) + " " + to_string(t->tm_hour) + ":" + to_string(t->tm_min) + ":" + to_string(t->tm_sec);
            int baseline;
            auto time_sz = getTextSize(cur_time, FONT_HERSHEY_PLAIN, 1, 1, &baseline);
            rectangle(frame, Point(0, 0), Point(0 + time_sz.width, time_sz.height + 5), Scalar(140,140,140), FILLED);
            putText(frame, cur_time, Point(0, time_sz.height +2), FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 255));
            imshow("output", frame);
          }
       ```
### 촬영 함수</br>
  * 현재 출력된 화면을 폴더에 저장함  
  
       ```c++ 
          void cheese()
          {
            static int cnt = 0;
            vector<int> params;
            params.push_back(IMWRITE_JPEG_QUALITY);
            params.push_back(95);
            imwrite(to_string(cnt++)+".JPG", frame, params);
          }
       ```
        
        

 <!-------------------------------------------------------------Part 3------------------------------------------------------------------------------------------>
## 3. 구현

   * __객체 검출__    
     > 사람, 안전모, 안전조끼, 안전화, 장갑을 분류함</br>
     > 일부 장비를 검출하지 못하는 경우가 있음</br>

![검출구현1](./MainIMG/검출구현1.JPG)
![검출구현2](./MainIMG/검출구현2.JPG)</br>
![검출구현3](./MainIMG/검출구현1.JPG)
![검출구현4](./MainIMG/검출구현2.JPG)
   * __장비 분류__    
     > 사람 클래스 안에 안전 장비의 중심 좌표를 파악함</br>
     > 넷 다 있으면 초록, 안전모, 조끼가 없으면 빨강</br>
     > 둘 다 있는 경우에 안전화, 장갑 둘 다 없으면 주황, 둘 중 하나만 없으면 핑크</br>

![분류구현1](./MainIMG/분류구현1.JPG)
![분류구현2](./MainIMG/분류구현2.JPG)</br>
![분류구현3](./MainIMG/분류구현1.JPG)
![분류구현4](./MainIMG/분류구현2.JPG)
   * __촬영__     
     > 빨간색 바운딩박스가 존재하면 첫 번째는 바로 화면을 폴더에 저장함</br>
     > 촬영 이후 60초를 측정해 1분이 지났을 때 다시 저장되게 딜레이 설정</br>

![촬영구현1](./MainIMG/촬영구현1.JPG)
![촬영구현2](./MainIMG/촬영구현2.JPG) 


              

 <!-------------------------------------------------------------Part 4------------------------------------------------------------------------------------------>
## 4. 문제점
   * __학습 데이터 부족 문제__     
     > darknet 학습시에 한 클래스당 수천 개의 데이터를 넣어야 학습이 되는데 수백 개만 넣음</br>
     > 초반 person 데이터 없이 진행했을 때 사람 머리를 안전모로 인식 -> person 데이터 추가  

   * __안전화 정의 안 됨__    
     > 공사장이나 특정 기관에서 안전화를 특정짓지 않음(운동화, 부츠, 워커 등 다양)  
     > 장화, 워커 등 발목 위로 올라가는 신발 기준으로 데이터 학습  

   * __안 썼을 때도 인식__  
     > 안전모를 들고 있을 때도 person 클래스 안에 있기 때문에 인식됨  
     > no helmet, helmet 등으로 학습시켜야 했음   
