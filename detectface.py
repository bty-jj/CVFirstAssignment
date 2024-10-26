import cv2
import numpy as np
def draw_detections(img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = (255,0,0)

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f'{class_id}: {score:.2f}'

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv2.FILLED)

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

class FaceDet:
    def __init__(self,model='./yolov8n.onnx',confidence_thres=0.3,iou_thres=0.4):
        self.model=cv2.dnn.readNetFromONNX(model)
        self.confidence_thres=confidence_thres
        self.iou_thres=iou_thres
    def detect(self,imgarr):
        h,w,_=imgarr.shape
        img=cv2.resize(imgarr,(640,640))[:,:,::-1]
        img=cv2.dnn.blobFromImage(img)/255.0
        self.model.setInput(img)
        pred=self.model.forward()
        resimg,hasface=self.postprocess(imgarr,pred,w,h)
        return resimg,hasface 
    def postprocess(self,input_image, output,img_width,img_height):
        confidence_thres=self.confidence_thres
        iou_thres=self.iou_thres
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))
        #print(outputs)

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []
        
        # Calculate the scaling factors for the bounding box coordinates
        x_factor = img_width / 640
        y_factor = img_height / 640

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)

        # Iterate over the selected indices after non-maximum suppression
        hasface=False
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            if class_id==0:
                draw_detections(input_image,box,score,class_id)
                hasface=True 
        return input_image, hasface

if __name__=='__main__':
    #fd=FaceDet() 
    # 打开摄像头，参数可以是摄像头的索引，通常0是默认摄像头  
    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():  
        print("无法打开摄像头")  
        exit()  

    while True:  
        # 读取摄像头的一帧  
        ret, frame = cap.read()  
        
        if not ret:  
            print("无法读取摄像头数据")  
            break  
        #farme,hasface=fd.detect(frame)
        #print(hasface)
        # 显示图像  
        cv2.imshow('Camera Feed', frame)  
        
        # 检测按键，如果按下 'q' 则退出  
        if cv2.waitKey(1000) & 0xFF == ord('q'):  
            break  

    # 释放摄像头资源  
    cap.release()  
    cv2.destroyAllWindows()