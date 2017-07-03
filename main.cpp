#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>

#define qrt(x) x * x

using namespace std;
using namespace cv;

// TexStudio, MixTex (latex)

// гонсалес вудс цифровая обработка сигналов matlab
// Embedded Vison Alliance

int main() {

//    VideoCapture cap("/home/green/DesktopFolder/Programming/C++/VideoRecord/2.mp4");
    VideoCapture cap(0);
//
//    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1024);
//    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 768);

    if (!cap.isOpened())
    {
        cout << "Can't open camera" << endl;
        return -1;
    }

    Mat cur, prev;

    cap >> cur;

    while(1)
    {
//        prev = cur.clone();

        cap >> cur;

        Mat gray;

//        fastNlMeansDenoising(cur, cur);

        GaussianBlur(cur, cur, Size(3,3), 0, 0, BORDER_DEFAULT);
        cvtColor(cur, gray, CV_BGR2GRAY);

        Mat grad_x, grad_y;


        Mat abs_grad_x, abs_grad_y;

        /// Gradient X
        //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
        Sobel(gray, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
        convertScaleAbs( grad_x, abs_grad_x );

        /// Gradient Y
        //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
        Sobel(gray, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT );
        convertScaleAbs( grad_y, abs_grad_y );

        /// Total Gradient (approximate)
        addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, gray );
//        for(int i = 0; i < cur.rows; ++i)
//        {
//            for(int j = 0; j < cur.cols; ++j)
//            {
//                auto c = cur.at<Vec3b>(i, j);
//                int sum = c[0] + c[1] + c[2];
//                sum /= 3;
//                cur.at<Vec3b>(i, j) = Vec3b(sum, sum, sum);
//            }
//        }

        if (cur.empty())
        {
            break;
        }
//
//        for(int i = 1; i < gray.rows - 1; ++i)
//        {
//            for(int j = 1; j < gray.cols - 1; ++j)
//            {
//                int sum = 0;
//
//                int sumX = 0, sumY = 0;
//
//                for(int ii = -1; ii <= 1; ++ii)
//                {
//                    for(int jj = -1; jj <= 1; ++jj)
//                    {
//                        sumX += GX[ii + 1][jj + 1] * gray.at<Vec3b>(i + ii, j + jj)[1];
//                        sumY += GY[ii + 1][jj + 1] * gray.at<Vec3b>(i + ii, j + jj)[1];
//                    }
//                }
//
//                sum = abs(sumX) + abs(sumY);
////                cout << sum << endl;
//
//
//                sum = max(0, sum);
//                sum = min(255, sum);
//
//                //sum = 255 - sum;
//
//                cur.at<int>(i, j) = sum;
//            }
//        }

//        Mat show = cur.clone();

//        for(int i = 0; i < cur.rows; ++i)
//        {
//            for(int j = 0; j < cur.cols; ++j)
//            {
//                auto p = prev.at<Vec3b>(i, j);
//                auto c = cur.at<Vec3b>(i, j);
//                if (abs((p[0] + p[1] + p[2]) - (c[0] + c[1] + c[2])) >= 1)
//                {
//                    show.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
//                }
//                else
//                {
//                    show.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
//                }
//            }
//        }

        imshow("Demo", gray);
        if (waitKey(1) == 27)
        {
            break;
        }
    }

    return 0;
}