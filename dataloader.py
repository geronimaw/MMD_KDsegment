import os
import numpy as np
import pandas as pd
import cv2
modality = 'test'


############  MASKS GENERATION
def genSepJointMap(annotations, jointNames, radius, frame, scale, index_image, name_image):
    print("[Generating joint masks  for: " + name_image + " ...]")

    if scale is None or scale == 0:
        scale = 1

    # radius = radius / scale
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    jm_height = int(np.floor(frame_height / scale))
    jm_width = int(np.floor(frame_width / scale))
    joint_num = len(jointNames)

    # print("altezza", jm_height)
    # print("larghezza", jm_width)

    jointmap = np.zeros((jm_height, jm_width, joint_num), dtype=np.uint16)

    # for joint_idx in range(0,  joint_num):
    # joint_idx=joint_num

    joint_anno = annotations  # ho una stringa con dentro i valori del dataframe

    # joint_anno = annotations[annotations['class'] == jointNames[joint_idx]].reset_index(drop=True)

    if joint_anno is not None:

        # for tool_idx in range(0, len(joint_anno)):
        for tool_idx in range(1, len(
                jointNames) + 1):  # parto da uno perchÃ¨ la prima riga contiene le annotazioni del dataframe

            # print("tool_idx:", tool_idx-1)

            line = joint_anno.split('\n')[tool_idx]
            line = joint_anno.splitlines()[tool_idx]

            # print(joint_anno.loc[tool_idx][:])

            joint_x = " ".join(line.split()[3])
            joint_x = joint_x.replace(" ", "")
            joint_x = float(joint_x)
            joint_x = jm_width * joint_x

            joint_y = " ".join(line.split()[4])
            joint_y = joint_y.replace(" ", "")
            joint_y = float(joint_y)
            joint_y = jm_height * joint_y

            # print("(" +  str(joint_x) +"," + str(joint_y)+ ")")

            # flip joint_x and joint_y
            temp = joint_y
            joint_y = joint_x
            joint_x = temp
            joint_x = int(round(joint_x))
            joint_y = int(round(joint_y))

            to_save = np.zeros((jm_height, jm_width), dtype=np.uint8)

            if (joint_x >= 0) and (joint_y > 0):
                cv2.circle(to_save, (joint_x, joint_y), radius, 255, -1)
                jointmap[:, :, tool_idx - 1] = to_save / 255

            # cv2.imshow("...", to_save)
            # cv2.waitKey()

            orig = cv2.imread(name_image, 0)
            orig = cv2.resize(orig, (round(jm_width), round(jm_height)))
            orig = orig * (to_save + 1)
            # cv2.imshow("...", orig)
            # cv2.waitKey()

            # print(ntpath.basename(name_image)[:4])
            # print("SAVED IN: content/joint_masks/jointmap_" + modality + "_" + ntpath.basename(name_image)[:4] + "_" + str(tool_idx-1) + '.png')
            ##cv2.imwrite("content/joint_masks/jointmap_" + modality + "_" + ntpath.basename(name_image)[:4] + "_" + str(tool_idx-1) + '.png', to_save)
            # cv2.imwrite("content/joint_masks/jointmap_" + modality + "_" + ntpath.basename(name_image)[:4] + "_" + str(tool_idx-1) + '.png', orig)

    print("----")

    return jointmap


def genSepPAFMapDet_2(annotations, toolCompoNames, side_thickness, frame, scale, index_image, name_image):
    if scale is None or scale == 0:
        scale = 1

    # side_thickness = side_thickness / scale
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    pm_width = int(np.floor(frame_width / scale))
    pm_height = int(np.floor(frame_height / scale))
    compo_num = len(toolCompoNames)

    paf_map = np.zeros((pm_height, pm_width, compo_num), dtype=np.uint16)

    # print("number of combination ", compo_num)

    nlines = len(annotations.splitlines())
    # print("no. joints", nlines-1)  #first line is header

    for compo_idx in range(0, compo_num):
        compo = toolCompoNames[compo_idx]

        # print("compo", compo)
        # print("compo_idx", compo_idx)

        for l in range(0, nlines):  # every "l" is a joint

            line = annotations.split('\n')[l]
            line = annotations.splitlines()[l]

            class_name = " ".join(line.split()[1])
            class_name = class_name.replace(" ", "")

            if str(class_name) == compo[0]:
                # print(class_name)

                joint1_x = " ".join(line.split()[3])
                joint1_x = joint1_x.replace(" ", "")
                joint1_x = float(joint1_x)
                joint1_x = pm_width * joint1_x

                if (joint1_x > 0):

                    joint1_y = " ".join(line.split()[4])
                    joint1_y = joint1_y.replace(" ", "")
                    joint1_y = float(joint1_y)
                    joint1_y = pm_height * joint1_y

                    line2 = annotations.split('\n')[l + 1]
                    line2 = annotations.splitlines()[l + 1]

                    joint2_x = " ".join(line2.split()[3])
                    joint2_x = joint2_x.replace(" ", "")
                    joint2_x = float(joint2_x)
                    joint2_x = pm_width * joint2_x

                    if (joint2_x > 0):
                        joint2_y = " ".join(line2.split()[4])
                        joint2_y = joint2_y.replace(" ", "")
                        joint2_y = float(joint2_y)
                        joint2_y = pm_height * joint2_y

                        temp = joint1_y
                        joint1_y = joint1_x
                        joint1_x = int(round(temp))
                        # joint1_y = int(round(frame_height - joint1_y)) #because I use Point later on
                        joint1_y = int(round(joint1_y))

                        temp = joint2_y
                        joint2_y = joint2_x
                        joint2_x = int(round(temp))
                        # joint2_y = int(round(frame_height - joint2_y)) #because I use Point later on
                        joint2_y = int(round(joint2_y))

                        # print(joint1_x)
                        # print(joint1_y)
                        # print(joint2_x)
                        # print(joint2_y)

                        img = np.zeros((pm_height, pm_width), np.uint8)

                        cv2.line(img, (joint1_x, joint1_y), (joint2_x, joint2_y), 255, side_thickness)
                        # cv2.imshow("",img)
                        # cv2.waitKey()

                        # orig = cv2.imread(name_image, 0)
                        # orig = cv2.resize(orig, (round(pm_width), round(pm_height)))
                        # orig = orig * (img + 1)
                        # print("SAVED IN: content/conn_masks/joint_conn_" + modality + "_" + ntpath.basename(name_image)[
                        #                                                                    :4] + "_" + str(
                        #    compo_idx) + '.png')
                        # cv2.imwrite("content/conn_masks/joint_conn_" + modality + "_" + ntpath.basename(name_image)[:4] + "_" + str(compo_idx) + '.png', img)  #salverÃ  l'ultimo
                        # cv2.imwrite("content/conn_masks/joint_conn_" + modality + "_" + ntpath.basename(name_image)[:4] + "_" + str(compo_idx) + '.png', orig)  # salverÃ  l'ultimo

                        img = img / 255
                        paf_map[:, :, compo_idx] = img

    return paf_map

vector_mean = []

def preProcess(imgsRGB, inputWidth, inputHeight):  # check normalization

    if isinstance(imgsRGB, pd.Series):

        imgs_scaled = np.zeros([len(imgsRGB), inputWidth, inputHeight, 1], dtype=np.float16)
        # print("imgs_scaled: ", imgs_scaled.shape)
        # print("minmaxloc: ", cv2.minMaxLoc(np.uint8(imgs_scaled[0])))

        for i in range(0, len(imgsRGB)):

            #normalization
            #print("here")
            x = cv2.resize(imgsRGB[i], (round(inputHeight), round(inputWidth)))
            x = x.astype('float16')
            mean = np.mean(x)
            #std = np.std(x)
            x -= mean

            vector_mean.append(mean)
            #x /= std

            #print(mean)
            #print(std)
            #print(x.shape)


            imgs_scaled[i, :, :, :] = np.expand_dims(x, axis=2)
            #imgs_scaled[i, :, :, :] = np.expand_dims(cv2.resize(imgsRGB[i], (round(inputHeight), round(inputWidth))),
            #                                         axis=2)

            # print("minmaxloc: ", cv2.minMaxLoc(np.uint8(imgs_scaled[i, :, :, :])))

            # imgs_scaled[i, :, :, :] = imgsRGB[i]
            # cv2.imshow(imgs_scaled[i, :, :, :])
            # cv2.waitKey()'''

    else:
        imgs_scaled = cv2.resize(imgsRGB, (round(inputHeight), round(inputWidth)))
        # imgs_scaled = imgsRGB

    imgs_scaled = np.divide(imgs_scaled, 255)
    # print("imgs_scaled in preproc", imgs_scaled.shape)

    return imgs_scaled


def string2dataframe(annot_string):
    name_col = ['class', 'id', 'x', 'y']

    classes = []
    idens = []
    xs = []
    ys = []
    D = []

    nlines = len(annot_string.splitlines())
    # print("no. joints", nlines-1)  #first line is header

    for l in range(1, nlines):  # every "l" is a joint

        # line = annot_string.split('\n')[l]
        line = annot_string.splitlines()[l]

        class_name = " ".join(line.split()[1])
        class_name = class_name.replace(" ", "")
        classes.append(class_name)

        iden = " ".join(line.split()[2])
        iden = iden.replace(" ", "")
        idens.append(iden)

        x = " ".join(line.split()[3])
        x = x.replace(" ", "")
        xs.append(x)

        y = " ".join(line.split()[4])
        y = y.replace(" ", "")
        ys.append(y)

    D = np.vstack((np.asarray(classes).T, np.asarray(idens).T, np.asarray(xs).T, np.asarray(ys).T))
    D = D.T
    # print(D)

    anno_df = pd.DataFrame(data=D, columns=name_col)
    # print(anno_df)

    # print(classes)
    # print(idens)
    # print(xs)
    # print(ys)

    return anno_df


# Code for the generations of the input and of the labels for the training of the detection network.
class DataLoader:
    def __init__(self, opt, doskolka, path_colab=None):
        test_data_tab = pd.DataFrame([])
        self.opt = opt
        self.testBatchSize = opt['testBatchSize'] or opt['batchSize'] or 1
        self.path_colab = path_colab

        test_data_file = os.path.join(opt['dataDir'], opt['dataset'])

        if path_colab:
            test_data_file = path_colab
        # test_data_file = os.path.join(opt['dataDir'], 'test_joint_head_.xls')

        if os.path.exists(test_data_file):
            test_data_tab = pd.read_excel(test_data_file)
            test_data_tab = test_data_tab.reset_index()
            test_data_tab = test_data_tab.drop(['index'], axis=1)
            test_data_tab = test_data_tab.drop(['Unnamed: 0'], axis=1)
            if doskolka > 0:
                test_data_tab = test_data_tab.loc[:doskolka]
                test_data_tab = test_data_tab.reset_index()
                test_data_tab = test_data_tab.drop(['index'], axis=1)
            # print(test_data_tab)
        else:
            pass

        self.testDataTab = test_data_tab

        self.testBatches = np.floor(len(self.testDataTab) / self.testBatchSize)
        self.testSamples = self.testBatches * self.testBatchSize

        print('Test Sample number: ' + str(self.testSamples))
        print('==================================================================')

        self.inputWidth = opt['inputWidth'] or 640
        self.inputHeight = opt['inputHeight'] or 480
        self.toolJointNames = opt['toolJointNames']
        self.toolCompoNames = opt['toolCompoNames']
        self.jointRadius = opt['jointRadius'] or 20
        self.modelOutputScale = opt['modelOutputScale'] or 4

    def load(self):

        # testing
        batch_size = self.testBatchSize
        data_tab = self.testDataTab
        nSamples = int(self.testSamples)
        aug_param_tab = None

        # perm = np.random.permutation(len(data_tab))

        input_width = self.inputWidth
        input_height = self.inputHeight
        jointNum = len(self.toolJointNames)
        compoNum = len(self.toolCompoNames)
        jointNames = self.toolJointNames
        compoNames = self.toolCompoNames
        j_radius = self.jointRadius
        model_output_scale = self.modelOutputScale

        # indices = perm[0: nSamples]
        indices = np.linspace(0, nSamples - 1, nSamples)

        # Here the first function starts
        frame_tab = pd.Series([])
        joint_batch_anno = pd.DataFrame([])
        frame_batch_map = np.zeros([nSamples, int(np.floor(input_width / model_output_scale)),
                                    int(np.floor(input_height / model_output_scale)),
                                    jointNum + compoNum], dtype=np.float16)

        print("METHOD: load")
        path = self.opt['dataDir']
        if self.path_colab is not None:
            path = '/content/drive/MyDrive/KD/babypose_test/'

        for i in range(0, nSamples):
        # for i in range(0, 10):
            frame_data = data_tab.loc[indices[i]]
            frame = cv2.imread(path + frame_data['filename'][7:], 0)
            frame = cv2.resize(frame, (input_height, input_width))

            # images
            aug_frame = frame

            aug_frame_resized = cv2.resize(aug_frame, (
            round(input_height / model_output_scale), round(input_width / model_output_scale)))

            frame_tab = frame_tab.append([pd.Series([aug_frame_resized])], ignore_index=True)

            # info annotation
            '''df_anno = string2dataframe(aug_annos)
            joint_batch_anno = joint_batch_anno.append(df_anno, ignore_index=True)   
            print("size joint_batch_anno: ", joint_batch_anno.shape)'''
            if self.path_colab is None:
                aug_annos = frame_data['annotations']

                # stack all the annotation maps (both joint and connection maps)
                jointmap = genSepJointMap(aug_annos, jointNames, j_radius, aug_frame, model_output_scale, i,
                                          path + frame_data['filename'][7:])
                frame_batch_map[i, :, :, 0:jointNum] = jointmap
                # print("size frame_batch_map: ", frame_batch_map.shape)
                #
                compmap = genSepPAFMapDet_2(aug_annos, compoNames, j_radius, aug_frame, model_output_scale, i,
                                            path + frame_data['filename'][7:])
                frame_batch_map[i, :, :, jointNum: jointNum + compoNum] = compmap
                # print("size frame_batch_map: ", frame_batch_map.shape)

        # frame_batch = preProcess(frame_tab, input_width, input_height)
        frame_batch = preProcess(frame_tab, round(input_width / model_output_scale),
                                 round(input_height / model_output_scale))

        frame_batch_CPU = frame_batch

        if self.path_colab is not None:
            return frame_batch_CPU, vector_mean

        else:
            frame_batch_map_CPU = frame_batch_map
            return frame_batch_CPU, frame_batch_map_CPU, vector_mean
            # frame_batch_anno_CPU = joint_batch_anno

        # return frame_batch_CPU, frame_batch_map_CPU, frame_batch_anno_CPU
