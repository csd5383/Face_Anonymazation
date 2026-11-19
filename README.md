# Face_Anonymazation
project in account of HY472 and Antonis Argyros , anonymozation is the proccess of completely nullifying the person's identity. In this attempt i will use it on faces 

I used the https://docs.opencv.org/4.13.0/d0/dd4/tutorial_dnn_face.html face detector with the 2026may yunet model for face detection. The im,ages will be from the FDDB dataset  

Setup: use the set up you need a jpg image and a onnx model file. The jpg image will be the input for the face anonymization process, and the onnx model file will be  used to detect the faces of the people in the images . For my setup i run the following command in the terminal :

Command : 
python xx.py --input_dir  --output_dir --boxes_dir  --face_detection_model  --score_threshold 

Testing : i found that for many faces and a blurred image the face detection model is not able to detect the majority of the faces so i lowered the threshold to 0.8 and the results are more optimal 

The face detector itself make a box around every "shape" that has a score above the one mentioned before and some spots that mark some features on the faces such as the eyes 

With that i extracted the boxes of the face and applied a gausian blur on the eclipse of the box coordinates on the faces creating a very naive result , this can surelly be surpassed. An idea that i had 