#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include "Component.h"

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

Mat mergeMat(const vector<Mat>& v, const int& column = 1, double resizeCoff = 1.)
{
    int width = 0;
    int height = 0;

    double coff = ceil((double)v.size() / column);

    for(auto a : v)
    {
        width = max(width, (int)(a.size().width / column));
        height = max(height, (int)(a.size().height / coff));
    }

    width *= resizeCoff;
    height *= resizeCoff;

    Mat final(height * coff, width * column, CV_8UC3);

    int x = 0, y = 0, i = 0;
    for(auto a : v)
    {
        if (a.type() != CV_8UC3)
        {
            cvtColor(a, a, CV_GRAY2RGB);
        }

        resize(a, a, Size(), resizeCoff / column, resizeCoff / coff, INTER_CUBIC);
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

void componentOfConnectivity(
        int i,
        int j,
        vector<vector<int> >& was,
        const Mat& m,
        Component& comp,
        const uchar& color)
{
    deque<pair<int, int> > v;

    v.push_back({i, j});
    was[i][j] = true;

    while(!v.empty())
    {
        pair<int, int>& p = v.front();
        v.pop_front();

        i = p.first;
        j = p.second;

        ++comp.count;

        if (i < comp.Top.first)
        {
            comp.Top = {i, j};
        }

        if (i > comp.Down.first)
        {
            comp.Down = {i, j};
        }

        if (j > comp.Right.second)
        {
            comp.Right = {i, j};
        }

        if (j < comp.Left.second)
        {
            comp.Left = {i, j};
        }

        if (i + 1 < was.size() &&
            !was[i + 1][j] &&
            m.at<uchar>(i + 1, j) == color)
        {
            v.push_back({i + 1, j});
            was[i + 1][j] = true;
        }

        if (j + 1 < was[i].size() &&
            !was[i][j + 1] &&
            m.at<uchar>(i, j + 1) == color)
        {
            v.push_back({i, j + 1});
            was[i][j + 1] = true;
        }

        if (i - 1 >= 0 &&
            !was[i - 1][j] &&
            m.at<uchar>(i - 1, j) == color)
        {
            v.push_back({i - 1, j});
            was[i - 1][j] = true;
        }

        if (j - 1 >= 0 &&
            !was[i][j - 1] &&
            m.at<uchar>(i, j - 1) == color)
        {
            v.push_back({i, j - 1});
            was[i][j - 1] = true;
        }
    }
};
//
//void changeColor(int i, int j, const Vec3b& oldColor, const Vec3b& newColor, const vector<Vec3b>& colors, vector<vector<int> >& was, Mat& m, Mat& d)
//{
//    vector<int> v;
//    was[i][j] = true;
//
//    v.push_back(i);
//    v.push_back(j);
//    int ii = 0;
//
//    while(ii < v.size())
//    {
//        i = v[ii++];
//        j = v[ii++];
//
//        d.at<uchar>(i, j) = newColor;
//
//        if (i + 1 < was.size() &&
//            !was[i + 1][j] &&
//            m.at<uchar>(i + 1, j) == oldColor)
//        {
//            v.push_back(i + 1);
//            v.push_back(j);
//            was[i + 1][j] = true;
//        }
//
//        if (j + 1 < was[i].size() &&
//            !was[i][j + 1] &&
//            m.at<uchar>(i, j + 1) == oldColor)
//        {
//            v.push_back(i);
//            v.push_back(j + 1);
//            was[i][j + 1] = true;
//        }
//
//        if (i - 1 >= 0 &&
//            !was[i - 1][j] &&
//            m.at<uchar>(i - 1, j) == oldColor)
//        {
//            v.push_back(i - 1);
//            v.push_back(j);
//            was[i - 1][j] = true;
//        }
//
//        if (j - 1 >= 0 &&
//            !was[i][j - 1] &&
//            m.at<uchar>(i, j - 1) == oldColor)
//        {
//            v.push_back(i);
//            v.push_back(j - 1);
//            was[i][j - 1] = true;
//        }
//    }
//}

void changeColor(const Rect& comp, vector<Vec3b>& colors, vector<vector<int> >& was, const Mat& s, int radiusX = 30, int radiusY = 200)
{
    int midX = comp.x + (comp.width >> 1);
    int midY = comp.y + (comp.height >> 1);
    int my = min(midY + radiusY, (int)was.size());
    int mx = min(midX + radiusX, (int)was[0].size());
    int ddd = max(0, midY);
    int dd = max(0, midX - radiusX);

    for (int i = ddd; i < my; ++i)
    {
        for (int j = dd; j < mx; ++j)
        {
            int index = s.at<int>(i,j);
            if (index > 0 && index < colors.size())
                colors[index - 1] = Vec3b(255, 255, 255);
        }
    }
}

int edgeThresh = 1;
int lowThreshold = 10;
int const max_lowThreshold = 100;
char* window_name = "Edge Map";
int ratio = 3;
int kernel_size = 3;

int main() {

//    VideoCapture cap("/home/green/DesktopFolder/Programming/C++/VideoRecord/2.mp4");
    VideoCapture cap(0);

    if (!cap.isOpened())
    {
        cout << "Can't open camera" << endl;
        return -1;
    }

    Mat cur;
    cap >> cur;
    resize(cur, cur, Size(), 0.5, 0.5, INTER_CUBIC);
    int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    VideoWriter video("out.avi", CV_FOURCC('M','J','P','G'), 25, Size(frame_width, frame_height), true);

    vector<vector<int> > was(cur.rows, vector<int>(cur.cols, 0));

    while(1)
    {
        for (auto a = was.begin(); a != was.end(); ++a)
        {
            fill(a->begin(), a->end(), 0);
        }

        cap >> cur;


        if (cur.empty())
        {
            break;
        }
        resize(cur, cur, Size(), 0.5, 0.5, INTER_CUBIC);

        Mat gray, lab;

        int blurKernel = 3;

        cvtColor(cur, gray, CV_BGR2GRAY);
//        cvtColor(cur, lab, CV_BGR2Lab);
//        GaussianBlur(gray, gray, Size(gaussianKernel, gaussianKernel), 0, 0, BORDER_DEFAULT);

//        Mat1b originalGray = gray.clone();
        /// Create a window
        namedWindow(window_name, CV_WINDOW_AUTOSIZE);

        /// Create a Trackbar for user to enter threshold
        createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold);

        Mat dst, detected_edges;
        GaussianBlur(gray, detected_edges, Size(blurKernel, blurKernel), 0);
        int erosion_type;
        int erosion_size = 0;
//
////        erosion_type = MORPH_RECT;
//        erosion_type = MORPH_CROSS;
        erosion_type = MORPH_ELLIPSE;
//
        Mat element;
        element = getStructuringElement(
                erosion_type,
                Size(2 * erosion_size + 1, 2 * erosion_size+1),
                Point(erosion_size, erosion_size)
        );

//        Mat1b rounded;
//        threshold(channels[1], rounded, 128, 255, cv::THRESH_BINARY);
//
//        erode(detected_edges, detected_edges, element);
//        dilate(detected_edges, detected_edges, element);

        /// Canny detector
        Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

//
        /// Using Canny's output as a mask, we display our result
        dst = Scalar::all(0);

        cur.copyTo(dst, detected_edges);

//        countoursSobel(gray);

//        vector<Mat> channels;
//
//        split(lab, channels);
//        vector<Component> result;
//
//        for (int i = 0; i < rounded.rows; ++i)
//        {
//            for (int j = 0; j < rounded.cols; ++j)
//            {
//                if (!was[i][j] && rounded.at<uchar>(i, j) > 0)
//                {
//                    Component comp(i, j);
//
//                    componentOfConnectivity(i, j, was, rounded, comp, rounded.at<uchar>(i, j));
//                    result.push_back(comp);
//                }
//            }
//        }

        // Container of faces
        vector<Rect> faces;
        CascadeClassifier face_cascade;
        face_cascade.load( "/home/green/Desktop/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_alt2.xml" );
        face_cascade.detectMultiScale(gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50) );
        for( int i = 0; i < faces.size(); i++ )
        {
            rectangle(gray, faces[i], Scalar( 255, 0, 255 ), 4, 8, 0 );
        }

//        sort(result.begin(), result.end(), [&](const Component& l, const Component& r){
//            return l.count > r.count;
//        });

//        cout << was.size() << '|' << rounded.rows << ' ' << rounded.cols << '|' << was[0].size() << endl;

//        Mat1b updatedRounded = cur.clone();
//
//        for (auto a = was.begin(); a != was.end(); ++a)
//        {
//            fill(a->begin(), a->end(), 0);
//        }
//
////        cout << result.size() << endl;
//
//        for (auto a = result.begin(); a != result.end();)
//        {
//            if (a->count < 10000)
//            {
//                changeColor(a->Top.first, a->Top.second, 255, 0, was, updatedRounded, updatedRounded);
//                a = result.erase(a);
//            }
//            else
//            {
//                ++a;
//            }
//        }
//        dilate(updatedRounded, updatedRounded, element);

        Mat1b colorComponents = Mat::zeros(gray.rows, gray.cols, gray.type());

//        erosion_size = 3;
//        element = getStructuringElement(
//                erosion_type,
//                Size(2 * erosion_size + 1, 2 * erosion_size+1),
//                Point(erosion_size, erosion_size)
//        );
//
//        erode(gray, gray, element);
//        dilate(gray, gray, element);

        Mat1b gg;

        cvtColor(dst, gg, CV_BGR2GRAY);

//        imshow(window_name, gg);
        gg = gg > 0;;
        Mat1b d = gg.clone();
        Mat markers = Mat::zeros(gg.size(), CV_32SC1);

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(gg, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
        for (unsigned int i=0; i<contours.size(); i++)
        {
            if (hierarchy[i][3] >= 0)
            {
                drawContours(gg, contours, i, Scalar(255, 0, 0), 2, 16, hierarchy);
                drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);
            }
        }

        erosion_size = 2;
//
        erosion_type = MORPH_CROSS;
//        erosion_type = MORPH_OPEN;
////        erosion_type = MORPH_ELLIPSE;
////
        element = getStructuringElement(
                erosion_type,
                Size(2 * erosion_size + 1, 2 * erosion_size+1),
                Point(erosion_size, erosion_size)
        );
//
//        erode(gg, gg, element);
//        dilate(gg, gg, element);
//        morphologyEx(gg, gg, erosion_type, element);

//        erosion_size = 7;
//        erosion_type = MORPH_CLOSE;
////        erosion_type = MORPH_ELLIPSE;
////
//        element = getStructuringElement(
//                MORPH_CROSS,
//                Size(2 * erosion_size + 1, 2 * erosion_size+1),
//                Point(erosion_size, erosion_size)
//        );
////        morphologyEx(gg, gg, erosion_type, element);
//
////        erosion_size = 7;
////        erosion_type = MORPH_OPEN;
////
////        element = getStructuringElement(
////                erosion_type,
////                Size(2 * erosion_size + 1, 2 * erosion_size+1),
////                Point(erosion_size, erosion_size)
////        );
////
////        dilate(gg, gg, element);
//
////        for (auto a = was.begin(); a != was.end(); ++a)
////        {
////            fill(a->begin(), a->end(), 0);
////        }

        circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);
        watershed(cur, markers);
        Mat mark = Mat::zeros(markers.size(), CV_8UC1);
        markers.convertTo(mark, CV_8UC1);
        bitwise_not(mark, mark);
//    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
        // image looks like at that point
        // Generate random colors
        vector<Vec3b> colors;
        for (size_t i = 0; i < contours.size(); i++)
        {
            int b = theRNG().uniform(0, 255);
            int g = theRNG().uniform(0, 255);
            int r = theRNG().uniform(0, 255);
            colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
        }
        // Create the result image
        dst = Mat::zeros(markers.size(), CV_8UC3);
        // Fill labeled objects with random colors

        Mat dst2 = dst.clone();

        for (int i = 0; i < markers.rows; i++)
        {
            for (int j = 0; j < markers.cols; j++)
            {
                int index = markers.at<int>(i,j);
                if (index > 0 && index <= static_cast<int>(contours.size()))
                    dst.at<Vec3b>(i,j) = colors[index-1];
                else
                    dst.at<Vec3b>(i,j) = Vec3b(0,0,0);
            }
        }

        for (auto a : faces)
        {
            changeColor(a, colors, was, markers, 10);
        }

        for (int i = 0; i < colors.size(); ++i)
        {
            if (colors[i] != Vec3b(255, 255, 255))
            {
                colors[i] = Vec3b(0, 0, 0);
            }
        }

        for (int i = 0; i < markers.rows; i++)
        {
            for (int j = 0; j < markers.cols; j++)
            {
                int index = markers.at<int>(i,j);
                if (index > 0 && index <= static_cast<int>(contours.size()) && colors[index - 1] == Vec3b(255, 255, 255))
                    dst2.at<Vec3b>(i,j) = cur.at<Vec3b>(i, j);
                else
                    dst2.at<Vec3b>(i,j) = Vec3b(0,0,0);
            }
        }


//        for (auto a = faces.begin(); a != faces.end(); ++a)
//        {
//            changeColor(*a, was, gg, colorComponents, 50);
//        }
//        Методы детектування и трекинга объектив в системах компьютерного зору

//        imshow(window_name, dst);

        Mat frame = mergeMat(
                {
                        gray,
                        gg,
                        dst2,
                        dst
                },
                2,
                2
        );
        imshow(
            window_name,
            frame
        );

//        imshow("VideoRecord", originalGray);
//        imshow("VideoRecord", ((channels[1] > 128) & originalGray) | gray);
//        imshow("VideoRecord", (((~channels[1]) | (channels[1]) > 128)));

        video << frame;

        if (waitKey(1) == 27)
        {
            break;
        }
    }

    video.release();

    return 0;
}



//    int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
//    int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
//    VideoWriter video("out.mp4", CV_FOURCC('M','J','P','G'), 10, Size(frame_width, frame_height), false);
//        Mat frame = rounded & originalGray;
//        video << frame;