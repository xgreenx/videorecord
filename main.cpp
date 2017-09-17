#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>

#define qrt(x) x * x

using namespace std;
using namespace cv;

// TexStudio, MixTex (latex)

// гонсалес вудс цифровая обработка сигналов matlab
// Embedded Vison Alliance

void countoursSobel(Mat& gray)
{
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
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, gray);
}

Mat mergeMat(const vector<Mat>& v, const int& column = 1)
{
    int width = 0;
    int height = 0;

    double coff = ceil((double)v.size() / column);

    for(auto a : v)
    {
        width = max(width, (int)(a.size().width / column));
        height = max(height, (int)(a.size().height / coff));
    }

    Mat final(height * coff, width * column, CV_8UC1);

    int x = 0, y = 0, i = 0;
    for(auto a : v)
    {
        resize(a, a, Size(), 1. / column, 1. / coff, INTER_CUBIC);
        Mat temp(final, Rect(x, y, a.size().width, a.size().height));
        x += a.size().width;
        a.copyTo(temp);

        ++i;
        if (i % column == 0)
        {
            x = 0;
            y += height;
        }
    }

    return final;
}

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

    deque<Mat> deq;

    Mat temp;
    int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    VideoWriter video("out.mp4", CV_FOURCC('M','J','P','G'), 10, Size(frame_width, frame_height), false);

    float prefetch = 2;

    while(1)
    {
//        prev = cur.clone();

        cap >> cur;

        if (cur.empty())
        {
            break;
        }


        deq.push_back(cur.clone());

//        if (deq.size() == 1)
//        {
//            temp = cur.clone();
//        }
//        else
//        {
//            temp += deq.back();
//        }
//
//        if (deq.size() <= prefetch)
//        {
//            continue;
//        }
//        else
//        {
//            deq.pop_front();
//        }
//
//        cur = deq.front();
//        cur /= prefetch + 1;

        Mat gray, lab;

//        fastNlMeansDenoising(cur, cur);

        int gaussianKernel = 3;

        GaussianBlur(cur, cur, Size(gaussianKernel, gaussianKernel), 0, 0, BORDER_DEFAULT);
        cvtColor(cur, gray, CV_BGR2GRAY);
        cvtColor(cur, lab, CV_BGR2Lab);

        Mat originalGray = gray.clone();

        countoursSobel(gray);

        vector<Mat> channels;

        split(lab, channels);
        int erosion_type;
        int erosion_size = 5;
//        erosion_type = MORPH_RECT;
//        erosion_type = MORPH_CROSS;
        erosion_type = MORPH_ELLIPSE;

        Mat element = getStructuringElement(
                erosion_type,
                Size(2 * erosion_size + 1, 2 * erosion_size+1),
                Point(erosion_size, erosion_size)
        );

        Mat rounded(channels[1] > 128);

        erode(rounded, rounded, element);
        dilate(rounded, rounded, element);

        imshow(
            "VideoRecord",
            mergeMat(
                {
                    gray,
                    channels[1] > 128,
                    originalGray,
                    rounded
                },
                2
            )
        );
//        imshow("VideoRecord", ((channels[1] > 128) & originalGray) | gray);
//        imshow("VideoRecord", (((~channels[1]) | (channels[1]) > 128)));

//        Mat frame = rounded & originalGray;

//        cout << frame.rows << ' ' << frame.cols << endl;
//        cout << frame_height << ' ' << frame_width << endl;
//        imshow("VideoRecord", frame);
//        video << frame;
        if (waitKey(3) == 27)
        {
            break;
        }
    }

    return 0;
}