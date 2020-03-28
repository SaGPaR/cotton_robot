# cotton picking bot
Cotton Picking Robot


This is a repository for Cotton Picking Robot
Activities:

1. Image acquisation and processing (Cotton Boll classification)
2. Robot localization
3. Auto steering and Auto Navigation
4. Picking controller 

For ensemble TiftTrack YOLO + Color Segmentation + Image Transformation for Tracking bolls
boll_track.py [<video_source>]


Before.

Make sure you install darknet and Darkflow.
https://github.com/thtrieu/darkflow

If Darkflow is running then copy this repo inside the darkflow folder and run the cotton boll tracking and counting boll_track.py [<video_source>]


Please visit github.com/kadefue as majority of this is inspired from his code. I have also forked his code on my repo. i used transfer learning to update the weights of last three layers to make the system detect cotton which occupies huge pixel area. 
