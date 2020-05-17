**Object Detection in Grocery Images**

Hello, I am Parimi Rama Shiva Sagar.

An assignment has been given to me to detect objects in  a Grocery data set, Compiled by Gulvarol and hosted on Github for Public access. I was given a part of with ready made test and train splits in two folders. 

**ASSUMPTIONS MADE**

  1. I assumed that the Entire dataset that is hosted on github is not necessary. But included a script to take them into consideration as well. Recognising brands etc 
  
  2. I assumed that I only had to predict the boxes. 
  
  3. I felt that the products are really similar (All are cigarette packs).so  Didn’t train the product on multiple anchor boxes.  
  
  4. I assumed that I needn’t predict the brands and compare them based on the deliverables asked.
  
  5. I assumed that Data Augmentation is not necessary for a mean Average Precision of 0.7 
  
  8. I assumed all the data given is Accurate and precise.

**PLAN EMPLOYED :**

	
  I immediately felt that the YOLO algorithm would do the purpose really well. Yolo would take data in a specific format. The format and Data preparation are discussed below. I wanted the model to converge really fast.So I obtained Pretrained Darknet Weights instead of initializing them randomly. Also I wrote the script considering all the classes under one object.I also wanted to test the longer one. So I immediately began preparing for a 11 class prediction. This means predicting 11 different Objects with just 200+  images. Although it seemed very difficult, I felt it to be more cool. Now, training requires a GPU .Google colab seemed to be a perfect choice. Hence I started Coding in google colab. 

**DATA PREPARATION :** 

	
  
 The Data set Given contains images in Two folders. Their bounding boxes are in a file Annotations.txt on Github. The format of the given data includes a text file with each line corresponding to an image either in Test or Train folders. The format of the given data was  

‘Image name (String) , number of detections(int) , [X( centre), Y(centre), Width , Height, class ] ...... Number of detection times. 

 Whereas, YOLO was trained to accept data in a specific format. Every Image should contain a txt file exactly with the same name as the image.  IMAGEAME.JPG , IMAGENAME.TXT

 Each text file should contain the objects in a New line. And each line should be ,

 Class(int), X(float between zero and one ) , Y( float between zero and one) , Width( float between zero and one) , Height ( Float between zero and one) 

 To obtain float X and Width should be divided by width of the image whereas. Y and height should be divided by height of the image . Width and height of the image are not the same as width and height of detection. 

I followed the following steps to prepare data for yolo in required format.

1. Read the annotations.txt file. 
2. Splitted it into lines 
3. Then, I converted the object into a list. So each element in a list contains a Line in annotations.txt as a string. Lets name it List1
4. Now I created an Empty dict. Itereated through the entire List1, Converted each string into a list of multiple strings using split (). Now the name of the Image is the first part of the variable returned by split which is P in our Code. P[0] contains the name of the image. 
5. Now I added the name of the image as the key and the rest of the list as an array to the empty dict created above.
6. I have listed the images in train and test directories to obtain Trainlist and testlist as a List of strings with each string containing  an Image name. 
7. Now I created Two more lists by using the above train and test lists. But in every string, I removed the last three characters( JPG) and replaced them with TXT. Thus we got two lists of names of Text files which would be handy ( But not much necessary) 
8. Iterating over the list (TRAINLIST) which contains the names of images, I retrieved the List of Objects which were stored in Dictionary data , And reshaped it into a Numpy array of (N,5) . N is the first string in the list , when converted to int , says the number of predictions in the corresponding Image. 
9. Numpy array is converted to float. And is Reassigned to a Dictionary. Now the entire dictionary contains Names of the Images as the keys and Corresponding Numpy arrays.
10.  Now the first column of numpy array is converted into integer type  and written into the .txt 
11. Steps 8,9,10 are iterated over Trainlist and test list. 
12.  Now we have files exactly as needed. We should Write two more text files.’ Train.txt’ and ‘Text.txt’ containing the list of images in train and test sets. 
13. The folders should now be merged and the folder name is renamed to ‘obj’ . Moving all the content in colab is difficult. We can iterate the process and move it, but we have to rename them again. So, I instead zipped them as a folder. Moved the zipped file into my google drive, Unzipped it and Copied content into one folder. Later copied the folder from drive and pasted it into Cloud VM for processing. 



**HYPER PARAMETER TUNING:**



1. Hyper parameter tuning is very important for any Deep learning model to perform well. 
2. So, I personally observed the images and found out that all are small boxes and mostly CIgarette packs of equal size. Which means we don’t need Many anchor boxes . One or Two would suffice. Also It seemed more likely that the pre established anchor boxes in yolo algorithm would suffice this assignment.
3. I felt that the Anchor boxes are in two different ratios, 2:5 and 1.5-1.8:5 in a very few cases after a rough manual division. 
4. Based on the images, I understood that there is only one class. And Around 4000-5000 objects which are all very similar. I guessed that  training 2000 iterations would suffice. 
5. I gave the batch size as 64 and subdivisions as 16 , Classes as 1 in one model and Classes as 11 in the other model. 
6. I trained till 1000 iterations and the model with 11 classes stopped converging. As the data is insufficient. Should have done Augmentation. I continued with the model with only one class. 
7. As Andrew NG said, Build your first model quickly and then tune everything else to perfection, I started doing the same and the environment was quite unsupportive.
8. I had to tune other parameters such as Filters based on the number of classes,batch size (set to 6000 for 1 class and 22000 for 11 classes) anchor boxes etc.
9. I also downloaded a pre trained weights file. For easier and smoother convergence.

**TRAINING THE MODEL :** 



1. The data should be in Darknet/data/obj/ folder. Both training and testing images with their corresponding test files are pasted in it.
2. Two text files namely train.txt and test.txt which contains names of train and test images are pasted in the Darknet/data folder.
3. Now Two files were written Namely obj.data Containing lines about what to do during testing and what to do during  validation and where to backup. And obj.names containing the names for classes.
4. Copy the [folder](https://drive.google.com/open?id=1xtLAYfkVWLDh0aU2O5g_XaJrboRsAU7X) and paste it in your google drive before you start running the Colab notebook
5. Run the command below to start training. 

#./darknet detector train  -- weights(your weights file) -- cfg(your cfg file)  -- obj.data


**TESTING AND RESULTS :**



1. I was monitoring the Convergence of loss throughout the process. Stopped the Training at 1900 iterations Due to connectivity issues and found the loss to be very low. 
2. Calculated the mean average precision using darknet map command and found it to be 89.25 . I assumed  that it would suffice. Backed the weights up to my [google drive folder](https://drive.google.com/open?id=1mGWPj2QL7Q7JcsCPf-O66tuwX1IZNmyH) after every 100 iterations. 
3. I also calculated the precision , recall map and saved them in the [folder](https://drive.google.com/open?id=1mch4slym-COZvIAlnuJEMW39V4JH4vwh) in my drive. 


**INSTRUCTIONS TO TRAINING THE MODEL ON YOUR OWN**
1) Find the ipynb Notebook attached
2) Choose between, Data preperation, Training or testing. 
3) Run the cells under that Heading in Google colab. 





