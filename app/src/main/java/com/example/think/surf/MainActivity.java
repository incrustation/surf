package com.example.think.surf;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static String TAG = "MainActivity";
    JavaCameraView javaCameraView;

    Mat mRgba, imgSurf, imgGray, tag, tagGray, H, train_tag_corner;
    Mat scene_descriptors, tag_descriptors;
    FeatureDetector surf;
    DescriptorExtractor freak;
    DescriptorMatcher flann;
    MatOfDMatch matches;
    List<DMatch> matched, good_match;
    List<KeyPoint> tag_list, scene_list, final_list;
    List<Point> objpt_list, scenept_list, obj_corner;
    MatOfPoint2f obj, scene;

    MatOfKeyPoint scene_points, tag_points, final_keypoints;
    double max_dist = 0;
    double min_dist = 100000;

    // Used to load the 'native-lib' library on application startup.

    BaseLoaderCallback mLoaderCallBack = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch(status){
                case BaseLoaderCallback.SUCCESS:{
                    javaCameraView.enableView();
                    break;
                }
                default:{
                    super.onManagerConnected(status);
                    break;
                }
            }
            super.onManagerConnected(status);
        }
    };

    static {
        if (OpenCVLoader.initDebug()){
            Log.i(TAG, "OpenCV Loaded Successfully");
        } else {
            Log.i(TAG, "OpenCV Not Loaded");
        }
    }
    static {
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[] {Manifest.permission.CAMERA}, 1);
            }
        }
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        javaCameraView = (JavaCameraView)findViewById(R.id.java_camera_view);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);

        //create feature detector, descriptor extractor, and descriptor matcher
        surf = FeatureDetector.create(FeatureDetector.SURF);
        freak = DescriptorExtractor.create(DescriptorExtractor.FREAK); // FREAK won't work for whatever reason
        flann = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
    }

    @Override
    protected void onPause(){
        super.onPause();
        if(javaCameraView != null){
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onDestroy(){
        super.onDestroy();
        if(javaCameraView != null){
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onResume(){
        super.onResume();
        if (OpenCVLoader.initDebug()){
            Log.i(TAG, "OpenCV Loaded Successfully");
            mLoaderCallBack.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        } else {
            Log.i(TAG, "OpenCV Not Loaded");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallBack);
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        //tag stands for the cannon image
        //scene is what the camera is seeing
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        imgSurf = new Mat(height, width, CvType.CV_8UC4);
        imgGray = new Mat(height, width, CvType.CV_8UC4);
        tag = new Mat(height, width, CvType.CV_8UC4);
        tagGray = new Mat(height, width, CvType.CV_8UC4);

        scene_points = new MatOfKeyPoint();
        tag_points = new MatOfKeyPoint();
        //these are keypoints on the scene image that are deemed good matches
        final_keypoints = new MatOfKeyPoint();

        scene_descriptors = new Mat(height, width, CvType.CV_8UC4);
        tag_descriptors = new Mat(height, width, CvType.CV_8UC4);

        matches = new MatOfDMatch();
        //this is the homography transform matrix
        H = new Mat(height, width, CvType.CV_8UC4);

        //good match points from tag and scene, currently not used
        obj = new MatOfPoint2f();
        scene = new MatOfPoint2f();

        //trying to obtain the tag image from Drawable
        Bitmap tag_dummy = BitmapFactory.decodeResource(getResources(), R.mipmap.bar_code);
        Utils.bitmapToMat(tag_dummy, tag);
        Imgproc.cvtColor(tag, tagGray, Imgproc.COLOR_RGB2BGR);
        //extract descriptors from tag
        surf.detect(tagGray, tag_points);
        tag_list = tag_points.toList();
        freak.compute(tagGray, tag_points, tag_descriptors);
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        imgSurf.release();
        imgGray.release();
        tag.release();
        tagGray.release();
        tag_points.release();
        scene_points.release();
        tag_descriptors.release();
        scene_descriptors.release();
        matches.release();
        obj.release();
        scene.release();
        H.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        good_match = new ArrayList<>();
        objpt_list = new ArrayList<>();
        scene_list = new ArrayList<>();
        scenept_list = new ArrayList<>();
        obj_corner = new ArrayList<>();
        final_list = new ArrayList<>();

        //get descriptors from current frame
        mRgba = inputFrame.rgba();
        Imgproc.cvtColor(mRgba, imgGray, Imgproc.COLOR_RGB2BGR);
        surf.detect(imgGray, scene_points);
        scene_list = scene_points.toList();
        freak.compute(imgGray, scene_points, scene_descriptors);

        //converting descriptors to the correct type
        if(tag_descriptors.type()!=CvType.CV_32F) {
            tag_descriptors.convertTo(tag_descriptors, CvType.CV_32F);
        }
        if(scene_descriptors.type()!=CvType.CV_32F) {
            scene_descriptors.convertTo(scene_descriptors, CvType.CV_32F);
        }

        //match the points
        flann.match(scene_descriptors, tag_descriptors, matches);
        matched = matches.toList();


        for( int i = 0; i < matched.size(); i++ ) {
            double dist = matched.get(i).distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        for( int i = 0; i < matched.size(); i++ )
        {
            double good_dist = (max_dist - min_dist)/3;
            if( matched.get(i).distance < min_dist+good_dist ) {
                good_match.add( matched.get(i));
            }
        }

        for( int i = 0; i < good_match.size(); i++ ) {
            KeyPoint from_tag = tag_list.get(good_match.get(i).trainIdx);
            KeyPoint from_scene = scene_list.get(good_match.get(i).queryIdx);
            objpt_list.add( from_tag.pt);
            scenept_list.add( from_scene.pt);
            final_list.add(from_scene);
        }

        good_match.clear();

        obj.fromList(objpt_list);
        scene.fromList(scenept_list);


        objpt_list.clear();
        scenept_list.clear();

        obj_corner.add(0, new Point(0,0));
        obj_corner.add(1, new Point(tag.cols(), 0));
        obj_corner.add(2, new Point(tag.cols(), tag.rows()));
        obj_corner.add(3, new Point(0, tag.rows()));
        train_tag_corner = Converters.vector_Point2f_to_Mat(obj_corner);

        //if need to specify RANSAC need to come up with a threshold
        if (final_list.size()!=0) {
            final_keypoints.fromList(final_list);
            Features2d.drawKeypoints(imgGray, final_keypoints, imgSurf, new Scalar(0,255,255),0);
            return imgSurf;
//            H = findHomography(obj, scene);
        }

        Features2d.drawKeypoints(imgGray, scene_points, imgSurf, new Scalar(255, 255, 255),0);

        obj_corner.clear();

        return imgSurf;
    }
}
