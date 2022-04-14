# YOLO PC Part Detector

## About
Last year, I had the opportunity to take a coding class in my school. We learned Python and created simple games with the Turtle module. I always wanted to learn how to program but never really got the chance to. I took this as the perfect opportunity. Besides for what we did in class, I went on my own working on personal projects, such as a space invaders game, and watching tutorials about learning Python. It brought me to where I was before I started this project: able to write lengthy and complicated scripts on my own.
I did not really know much about computer vision and AI. I knew what the concepts were but I never actually had any experience.
As I was brainstorming ideas, I looked to my right and saw my PC. I thought to myself about how people new to PC building might confuse parts with each other or just do not know what they are. So I decided my project would detect PC parts.

## Gathering Data
In order to get the data, I used a Bing API (an image search API). It was much faster than downloading hundreds of images manually and also gave me some insight into cloud-based computing.

I used the YoloLabel GUI in order to label my data. It’s worth noting that for every GPU, I also labeled the fans, thus increasing the accuracy for fans. Also, when possible, I labeled each individual RAM stick as opposed to all of them in one label.

I created a script that took about a random 30% of my data for a test set and the remaining 70% for training.

I used the YoloV4 tiny model. It runs faster on a cloud deployment than a heavier, bigger model.

I used Streamlit and Darknet in my project. Streamlit is the tool that runs the webapp and Darknet is the neural network that I completed the training with. I used a cloud GPU to train. I also used a Virtual Computer to test the network once trained. To evaluate images, I used cv2, while Darknet was just for training.

## Analysis
I used a total of 590 images

- CPU: 84

- GPU: 107

- Motherboard: 117

- PSU: 99

- Fans: 88, the number of labels for this class is much larger due to labeling fans on GPU’s

- RAM: 95, when possible, individual RAM sticks were labeled as opposed to all of them as a whole

Training took…

mAP and IoU scores - 0.066080 avg loss - When you see that average loss 0.xxxxxx avg no longer decreases at many iterations then you should stop training. The final average loss can be from 0.05 (for a small model and easy dataset) to 3.0 (for a big model and a difficult dataset).

more mAP and IoU

When tested with uploaded images and images from a webcam, promising results were produced. The first test was with a fan, and the fan was predicted correctly with 99% confidence first try. However, sometimes it would predict a fan or CPU when there was none there. This pattern was the same for uploaded images.
 
## Next Steps/Conclusion
The Normalized Confusion Matrix shows the weak points of the network. Fans and Motherboards were almost perfect, while the rest lagged behind. A major issue with the network, as shown by the confusion matrix, is that when nothing is there, Fans and RAM are predicted quite often.

If done again, I would increase the dataset size and label things differently. For example, there wasn’t a consistent labeling method with RAM: if individual labels were possible, it was done, if not, the entire set of RAM was labeled. I would also add more real world images. The white background on some images may have caused some false detections due to the training.

Next steps for the web app would be to add real-time camera/video detection. I also would increase the dataset by at least of factor of two. If continued to be worked on, this project could possibly be used in factories, quality control, or even stores like Micro Center to help people who are new to the PC world not be overwhelmed by the vast amount of parts.

My favorite thing I learned while creating my project was the GPU training. I thought it was really cool how a computer was able to learn to recognize things just like humans can. It was also cool to see the accuracy of the network improving as training went on.

