from pathlib import Path
import cv2
import dlib
import numpy as np
from contextlib import contextmanager
from .wide_resnet import WideResNet
import os

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

def yield_images_from_dir(image_dir):
    image_dir = Path(image_dir)
    for image_path in image_dir.glob("*.*"):
        img = cv2.imread(str(image_path), 1)
        if img is not None:
            h, w, _ = img.shape
            r = 640 / max(w, h)
            yield cv2.resize(img, (int(w * r), int(h * r)))

@contextmanager
def image_processing(image_path, dstimg_path, weight_file, shape_predictor):
    depth = 16
    width = 8
    k = width
    margin = 0.4
    img_size = 64
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    model = WideResNet(img_size, depth=depth, k=width)()
    model.load_weights(weight_file)
    img = cv2.imread(str(image_path), 1)
    if img is not None:
        h, w, _ = img.shape
        r = 640 / max(w, h)
        cv2.resize(img, (int(w * r), int(h * r)))
    else:
        raise ValueError('the image is corrupted or does not exist')
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = np.shape(input_img)
    detected = detector(input_img, 1)  # 调用权重文件
    faces = np.empty((len(detected), img_size, img_size, 3))
    if len(detected) > 0:
        for i, d in enumerate(detected):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - margin * w), 0)
            yw1 = max(int(y1 - margin * h), 0)
            xw2 = min(int(x2 + margin * w), img_w - 1)
            yw2 = min(int(y2 + margin * h), img_h - 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
            faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

        # predict ages and genders of the detected faces
        results = model.predict(faces)
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()

        # draw results
        for i, d in enumerate(detected):
            label = "{}, {}".format(int(predicted_ages[i]),
                                    "M" if predicted_genders[i][0] < 0.5 else "F")
            draw_label(img, (d.left(), d.top()), label)

    cv2.imwrite(dstimg_path, img)


@contextmanager
def main(rawimg_path,dstimg_path):
    img_file = os.listdir(raw_path)
    depth = 16
    width = 8
    k = width
    weight_file = './pretrained_models/age_gender_train_model.hdf5'
    margin = 0.4
    image_dir = rawimg_path

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r'./pretrained_models/face_detector_train_model.dat')

    img_size = 64
    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights(weight_file)

    image_generator = yield_images_from_dir(image_dir)

    import pdb
    pdb.set_trace()

    count = 0
    for img in image_generator:
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        detected = detector(input_img, 1) #调用权重文件
        faces = np.empty((len(detected), img_size, img_size, 3))

        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

            # predict ages and genders of the detected faces
            results = model.predict(faces)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()

            # draw results
            for i, d in enumerate(detected):
                label = "{}, {}".format(int(predicted_ages[i]),
                                        "M" if predicted_genders[i][0] < 0.5 else "F")
                draw_label(img, (d.left(), d.top()), label)

        cv2.imwrite(dstimg_path+'/'+img_file[count], img)
        count+=1
        #cv2.imshow("result", img)
        #key = cv2.waitKey(-1) if image_dir else cv2.waitKey(30)

        #if key == 27:  # ESC
        #    break
