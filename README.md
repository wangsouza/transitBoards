# transitBoards

run:\
```python predict_video.py --model weights/frozen_inference_graph.pb --labels classes.pbtxt --input input/videos/transitBoard.mp4 --output output/output.avi --num-classes 3```

**It is necessary to create a symbolic link from the object_detection folder in the tensorflow library**

[link to tensorflow](https://cutt.ly/Zy40fW1)

**Place the weight file in the weights folder**

[link to weight](https://drive.google.com/file/d/1VeJNXVn-R1QfMOxZPQOQluM0ilHeeVra/view?usp=sharing)


Multiple detection:
![Alt text](readme/transitBoard1.png?raw=true "Title")

Detecting far objects:
![Alt text](readme/transitBoard2.png?raw=true "Title")

Pedestrian Crossing board:
![Alt text](readme/transitBoard3.png?raw=true "Title")
