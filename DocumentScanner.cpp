#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

///// Document Scanner like CAM SCANNER!

///////////////// Preprocess  //////////////////////
Mat Preprocess(Mat img)
{
    Mat imgGray, imgBlur, imgEdge, imgDil;

    cvtColor(img, imgGray, COLOR_BGR2GRAY);
    GaussianBlur(img, imgBlur, Size(9, 9), 3, 0);
    Canny(imgBlur, imgEdge, 25, 75);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(imgEdge, imgDil, kernel);

    return imgDil;

}

/////////////////  Contour Detection  //////////////////////
vector<Point> getContours(Mat imgDil)
{
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    int maxarea = 0;

    findContours(imgDil, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //drawContours(imgDil, contours, -1, Scalar(255, 0, 255), 2);

    vector<vector<Point>> conPoly(contours.size());
    vector<Rect> boundRect(contours.size());

    vector<Point> contour;

    for (int i = 0; i < contours.size(); i++)
    {
        int area = contourArea(contours[i]);
        if (area > 200)
        {
            float peri = arcLength(contours[i], true);
            approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);


            boundRect[i] = boundingRect(conPoly[i]);
            if (area > maxarea && conPoly[i].size() == 4)
            {
                contour = { conPoly[i][0], conPoly[i][1], conPoly[i][2], conPoly[i][3]};
                maxarea = area;
                cout << contour;
                
            }
            //rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 5);
        }

    }
    return contour;
}

/////////////////  Draw the Contour  //////////////////////
void drawContour(vector<Point> contour, Scalar color, Mat img)
{
    for (int i = 0; i < contour.size(); i++)
    {
        circle(img, contour[i], 5, color, FILLED);
        putText(img, to_string(i), contour[i], FONT_HERSHEY_COMPLEX, 1, color, 1);
    }
    vector<vector<Point>> data(1);
    data[0].push_back(contour[0]);
    data[0].push_back(contour[1]);
    data[0].push_back(contour[3]);
    data[0].push_back(contour[2]);
    drawContours(img, data, -1, Scalar(0, 255, 0), 5, LINE_8);
    //drawContours(img, contour, -1, Scalar(255, 0, 255), 2);
}

/////////////////  Reorder the points of contour  //////////////////////
vector<Point> reorder(vector<Point> points)
{
    vector<Point> new_points(4);
    vector<int> sumofPoints(4, 0), subofPoints(4, 0);

    // find max and min x+y and x-y 's
    for (int i = 0; i < 4; i++)
    {
        sumofPoints[i] = points[i].x + points[i].y;
        subofPoints[i] = points[i].x - points[i].y;
    }
    new_points[0] = points[min_element(sumofPoints.begin(), sumofPoints.end()) - sumofPoints.begin()];
    new_points[3] = points[max_element(sumofPoints.begin(), sumofPoints.end()) - sumofPoints.begin()];

    new_points[2] = points[min_element(subofPoints.begin(), subofPoints.end()) - subofPoints.begin()];
    new_points[1] = points[max_element(subofPoints.begin(), subofPoints.end()) - subofPoints.begin()];

    // TO APPEND WE CAN USE THIS: new_points.push_back();

    return new_points;
}

/////////////////  Warp and transform the img //////////////////////
Mat Warp_img(Mat img, vector<Point> points, int thresh)
{
    Mat imgWarp, matrix, imgCropped;
    float w = 520, h = 596;

    Point2f src[4] = { points[0],points[1] ,points[2] ,points[3] };
    Point2f dst[4] = { {0.0,0.0},{w,0.0},{0.0,h},{w,h} };

    matrix = getPerspectiveTransform(src, dst);
    warpPerspective(img, imgWarp, matrix, Point(w, h));

    Rect roi(thresh, thresh, w - 2*thresh, h - 2* thresh);
    imgCropped = imgWarp(roi);
    return imgCropped;
}

void main() {

    string path = "Resources/page.jpg";
    Mat imgOrg = imread(path), imgEdge, imgWarp;
    vector<Point> imgContour, points;
    resize(imgOrg, imgOrg, Size(), 0.5, 0.5);

    // Preprocess
    imgEdge = Preprocess(imgOrg);
    // Get Biggest Contour
    imgContour = getContours(imgEdge);
    points = reorder(imgContour);
    // Warp
    imgWarp = Warp_img(imgOrg, points, 10);
    // plot contour
    drawContour(points, Scalar(0, 255, 0), imgOrg);

    // plot 
    imshow("Image", imgOrg);
    imshow("Image Edges", imgEdge);
    imshow("Image Cropped", imgWarp);

    waitKey(0);

}