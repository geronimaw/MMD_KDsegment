from dataloader import DataLoader
from model import JointDetModelFull
import cv2
import numpy as np
import time
from keras.models import *

op = {
    'dataDir': 'D:/sinc/BabyPose/',
    'dataset': 'xlsx/test_joint_head_tuttiNo28.xls',
    'modelPath': 'D:/sinc/iciap2022/models/DetNet.json',
    'weightsPath': 'D:/sinc/iciap2022/models/DetNet.hdf5',
    'testPath': 'D:/sinc/distillation/segmentation/prediction_preSigmoid/',
    'saveDir': '',

    'type': 'toolPartDetFull',
    'v': '256*320_ftblr_head',

    'jointRadius': 6,  # 15 diviso valore scala #CONTROLLARE
    'modelOutputScale': 5,  # 10

    'inputWidth': 480,
    'inputHeight': 640,

    'ngpus': 1,
    'testBatchSize': 1,
    'batchSize': 2,
    'toolJointNames': ['right_hand', 'right_elbow', 'right_shoulder', 'left_hand', 'left_elbow', 'left_shoulder',
                       'right_hip', 'right_knee', 'right_foot', 'left_hip', 'left_knee', 'left_foot'],
    'toolCompoNames': [['right_hand', 'right_elbow'], ['right_elbow', 'right_shoulder'],
                       ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_hand'], ['right_foot', 'right_knee'],
                       ['right_knee', 'right_hip'], ['left_hip', 'left_knee'], ['left_knee', 'left_foot']]

}

def prediction(get_fm, output, doskolka, path_colab):
    dataloader = DataLoader(op, doskolka, path_colab)
    # doskolka = 0 per prendere tutti i file della cartella
    frame_batch_CPU, frame_batch_map_CPU, vector_mean = dataloader.load()

    ### Loading the model
    print("load model...")
    # json_file = open(op['modelPath'], 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # model = model_from_json(loaded_model_json)
    model = JointDetModelFull()
    model.load_weights(op['weightsPath'])


    ### Predicting the maps
    # t = time.time()
    predictions = model.predict(frame_batch_CPU, verbose=0)[output]
    # predictions_Sigm = model.predict(frame_batch_CPU, verbose=0)[1]
    # fm_encoder = model.predict(frame_batch_CPU, verbose=0)[2]
    # elapsed = time.time() - t
    # print(f"elapsed time for " + str(frame_batch_CPU.size[0]) + " frames: " + str(elapsed) + " s")

    if get_fm is None:
        ############ Test the network
        ## Metrics
        dice_joint = -1 * np.ones((predictions.shape[0], len(op['toolJointNames'])), dtype=np.float16)
        dice_connection = -1 * np.ones((predictions.shape[0], len(op['toolCompoNames'])), dtype=np.float16)
        rec_joint = -1 * np.ones((predictions.shape[0], len(op['toolJointNames'])), dtype=np.float16)
        rec_connection = -1 * np.ones((predictions.shape[0], len(op['toolCompoNames'])), dtype=np.float16)
        prec_joint = -1 * np.ones((predictions.shape[0], len(op['toolJointNames'])), dtype=np.float16)
        prec_connection = -1 * np.ones((predictions.shape[0], len(op['toolCompoNames'])), dtype=np.float16)


        for im_num in range(predictions.shape[0]):
            # for visualization purposes, adding the mean removed during preprocessing
            drawing_joint = cv2.cvtColor(np.uint8((frame_batch_CPU[im_num] + vector_mean[im_num]) * 255), cv2.COLOR_GRAY2BGR)
            drawing_connection = cv2.cvtColor(np.uint8((frame_batch_CPU[im_num] + vector_mean[im_num]) * 255), cv2.COLOR_GRAY2BGR)
            # cv2.imwrite(op['testPath'] + "/superimposed_joint_" + str(im_num) + "_.png", drawing_joint)
            # cv2.imwrite(op['testPath'] + "/superimposed_connection_" + str(im_num) + "_.png", drawing_connection)

            for joint in range(len(op['toolJointNames']) + len(op['toolCompoNames'])):

                if joint < len(op['toolJointNames']):

                    ret, thresh = cv2.threshold(np.uint8(predictions[im_num, :, :, joint] * 255), 230, 255,
                                                cv2.THRESH_BINARY)
                    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
                    prova2 = frame_batch_map_CPU[im_num, :, :, joint]
                    contours_gt, hierarchy = cv2.findContours(np.uint8(frame_batch_map_CPU[im_num, :, :, joint] * 255),
                                                                   cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

                    cv2.drawContours(drawing_joint, contours, -1, (0, 255, 0), 1)  # green
                    cv2.drawContours(drawing_joint, contours_gt, -1, (255, 0, 0), 1)  # blue

                    seg = thresh
                    gt = np.uint8(frame_batch_map_CPU[im_num, :, :, joint] * 255)

                    # Metrics
                    if np.sum(gt) == 0:
                        print("joint is not present")

                    else:
                        dice = np.sum(seg[gt == 255]) * 2.0 / (np.sum(seg) + np.sum(gt))
                        dice_joint[im_num, joint] = dice

                        TP = np.sum(np.logical_and(seg == 255, gt == 255))
                        TN = np.sum(np.logical_and(seg == 0, gt == 0))
                        FP = np.sum(np.logical_and(seg == 255, gt == 0))
                        FN = np.sum(np.logical_and(seg == 0, gt == 255))

                        rec_joint[im_num, joint] = TP / (TP + FN)
                        prec_joint[im_num, joint] = TP / (TP + FP)

                    # Save predictions
                    meanpixel = np.ndarray.min(predictions[im_num, :, :, joint])
                    pred = predictions[im_num, :, :, joint] - meanpixel
                    maxpixel = np.ndarray.max(pred)
                    pred = (predictions[im_num, :, :, joint] / maxpixel)
                    cv2.imwrite(op['testPath'] + "f" + str(im_num) + "_m" + str(joint) + ".png", pred * 255)
                    cv2.imwrite(op['testPath'] + "f" + str(im_num) + "_m" + str(joint) + "_sigm.png",
                                predictions_Sigm[im_num, :, :, joint])


                else:

                    ret, thresh = cv2.threshold(np.uint8(predictions[im_num, :, :, joint] * 255), 200, 255, 0)
                    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:] #ГЛАВНОЕ ИЗМЕНЕНИЕ
                    contours_gt, hierarchy = cv2.findContours(np.uint8(frame_batch_map_CPU[im_num, :, :, joint] * 255),
                                                                   cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:] #ГЛАВНОЕ ИЗМЕНЕНИЕ

                    cv2.drawContours(drawing_connection, contours, -1, (0, 0, 255), 1)  # green
                    cv2.drawContours(drawing_connection, contours_gt, -1, (255, 0, 0), 1)  # blue

                    seg = thresh
                    gt = np.uint8(frame_batch_map_CPU[im_num, :, :, joint] * 255)

                    ## Metrics
                    if np.sum(gt) == 0:
                        print("joint is not present")
                    else:
                        dice = np.sum(seg[gt == 255]) * 2.0 / (np.sum(seg) + np.sum(gt))
                        dice_connection[im_num, joint - len(op['toolJointNames'])] = dice

                        TP = np.sum(np.logical_and(seg == 255, gt == 255))
                        TN = np.sum(np.logical_and(seg == 0, gt == 0))
                        FP = np.sum(np.logical_and(seg == 255, gt == 0))
                        FN = np.sum(np.logical_and(seg == 0, gt == 255))

                        rec_connection[im_num, joint - len(op['toolJointNames'])] = TP / (TP + FN)
                        prec_connection[im_num, joint - len(op['toolJointNames'])] = TP / (TP + FP)

                    # Save predictions
                    # Save predictions
                    meanpixel = np.ndarray.min(predictions[im_num, :, :, joint])
                    pred = predictions[im_num, :, :, joint] - meanpixel
                    maxpixel = np.ndarray.max(pred)
                    pred = (predictions[im_num, :, :, joint] / maxpixel)
                    cv2.imwrite(op['testPath'] + "f" + str(im_num) + "_m" + str(joint) + ".png", pred * 255)
                    cv2.imwrite(op['testPath'] + "f" + str(im_num) + "_m" + str(joint) + "_sigm.png",
                                predictions_Sigm[im_num, :, :, joint])



        np.savetxt(op['testPath'] + "/detection_dice_joint.csv", dice_joint, delimiter=",")
        np.savetxt(op['testPath'] + "/detection_dice_connection.csv", dice_connection, delimiter=",")

        np.savetxt(op['testPath'] + "/detection_rec_joint.csv", rec_joint, delimiter=",")
        np.savetxt(op['testPath'] + "/detection_rec_connection.csv", rec_connection, delimiter=",")

        np.savetxt(op['testPath'] + "/detection_prec_joint.csv", prec_joint, delimiter=",")
        np.savetxt(op['testPath'] + "/detection_prec_connection.csv", prec_connection, delimiter=",")

        dice_joint[dice_joint == -1] = np.nan
        print("median dsc joint: ", np.nanmedian(dice_joint, axis=0))
        # print("iqr dsc joint: ", np.nanpercentile(dice_joint, [25,50,75], axis=0))
        print("IQR dsc joint: ", np.nanpercentile(dice_joint, 75, axis=0) - np.nanpercentile(dice_joint, 25, axis=0))
        #
        dice_connection[dice_connection == -1] = np.nan
        print("median dsc connection: ", np.nanmedian(dice_connection, axis=0))
        # print("iqr dsc joint: ", np.nanpercentile(dice_connection, [25,50,75], axis=0))
        print("IQR dsc connection: ",
              np.nanpercentile(dice_connection, 75, axis=0) - np.nanpercentile(dice_connection, 25, axis=0))

        rec_joint[rec_joint == -1] = np.nan
        print("median se joint: ", np.nanmedian(rec_joint, axis=0))
        print("IQR se joint: ", np.nanpercentile(rec_joint, 75, axis=0) - np.nanpercentile(rec_joint, 25, axis=0))
        #
        rec_connection[rec_connection == -1] = np.nan
        print("median se connection: ", np.nanmedian(rec_connection, axis=0))
        print("IQR se connection: ",
              np.nanpercentile(rec_connection, 75, axis=0) - np.nanpercentile(rec_connection, 25, axis=0))

    else:
        return predictions
