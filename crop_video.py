import face_detection
import cv2
from PIL import Image
import numpy as np
import os
import os.path
from tqdm import tqdm

def face_detect(image_size, full_frames, na):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cuda')

    batch_size = 10
    ori_images = []
    ori_images = full_frames
    H,W = full_frames[0].shape[0],full_frames[0].shape[1]
    images = ori_images

    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = [0,0,0,0]
    x11,x21,x31,x41 = 1000,1000,-1,-1
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
            exit("error")
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        a,b,c,d = -50,-50,50,50
        x1 = max(x1 + a, 0)
        y1 = max(y1 + b, 0)
        x2 = min(x2 + c, image.shape[1])
        y2 = min(y2 + d, image.shape[0])
        results.append([x1, y1, x2, y2])
        break

    p = results[0]
    results = []
    for i in range(len(images)):
        results.append(p)
    boxes = np.array(results)
    results = [image[y1: y2, x1:x2] for image, (x1, y1, x2, y2) in zip(images, boxes)]
    pp = os.path.join('./data/crop_img/',na.split('.')[0])
    if not os.path.exists(pp):
        os.makedirs(pp)

    for i in range(len(results)):
        im = results[i]
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)
        im = im.resize((image_size, image_size),Image.ANTIALIAS)
        im.save(os.path.join(pp,str(i)+'.jpg'))
    res = []
    quad = []

    if not os.path.exists('./data/crop_video'):
        os.makedirs('./data/crop_video')
    crop_vi = os.path.join('./data/crop_video', na)
    out_edit_crop = cv2.VideoWriter(crop_vi, cv2.VideoWriter_fourcc(*'mp4v'), 25, (image_size, image_size))
    for i in range(0,len(results)):
        im = results[i]
        im = cv2.resize(im,(image_size,image_size),cv2.INTER_AREA)
        out_edit_crop.write(im)
    out_edit_crop.release()

    del detector
    pa = os.path.join('./data/crop_'+na.split('.')[0]+ '.txt')
    with open(pa, 'w') as f:
        line = str(H) + ' ' + str(W)
        f.write(line)
        f.write("\n")
        for (x1, y1, x2, y2) in boxes:
            line = str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2)
            f.write(line)
            f.write("\n")

if __name__ == '__main__':
    pa = './data/full_video/'
    na = os.listdir(pa)
    for i in na:
        path_video = os.path.join(pa,i)
        video_stream = cv2.VideoCapture(path_video)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        print('Reading video frames...')
        resize_factor = 1
        full_frames = []
        img_path = os.path.join('./data/full_img/',i.split('.')[0])
        os.makedirs(img_path,exist_ok=True)
        c = 0
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            full_frames.append(frame)
            cv2.imwrite(os.path.join(img_path,str(c)+'.jpg'),frame)
            c += 1
        print ("Number of frames available for inference: "+str(len(full_frames)))

        face_detect(256, full_frames, i)
