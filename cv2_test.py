import numpy as np
import cv2
from keras.models import load_model
from skimage.transform import resize, pyramid_reduce
import PIL
from PIL import Image
from keras.preprocessing import image

model = load_model('/home/puneeth/Desktop/only_names.h5')
uniq_labels = ['A', 'B', 'E', 'H', 'N', 'P', 'T', 'U', 'nothing']
def prediction(pred):
    return(chr(pred+ 65))

def crop_image(image, x, y, width, height):
    return image[y:y + height, x:x + width]

def keras_predict(model, image):
    data = np.asarray( image, dtype="int32" )
    
    pred_probab = model.predict(data)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

def main():

	cap = cv2.VideoCapture(0)

	while(True):

	    ret, frame = cap.read()
	    im2 = crop_image(frame, 300,300,300,300)
		# load the model we save
		img = img.resize((64,64,3)
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		# print(uniq_labels)
		images = np.vstack([x])
		classes = model.predict_classes(images, batch_size=10)
	    cv2.putText(frame, curr, (700, 300), cv2.FONT_HERSHEY_COMPLEX, 4.0, (255, 255, 255), lineType=cv2.LINE_AA)
	    cv2.rectangle(frame, (300, 300), (600, 600), (255, 255, 00), 3)
	    cv2.imshow("frame",frame)
	    print(uniq_labels[classes[0]])
	    # cv2.imshow('frame',blurred)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break
if __name__ == '__main__':
    main()

cap.release()
cv2.destroyAllWindows()