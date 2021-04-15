import os
import numpy as np
import h5py
import cv2

def getAllFile(folder):
    allFile = os.listdir(folder)
    fileNames = []
    for file in allFile:
        fullPath = os.path.join(folder, file)
        fileNames.append(fullPath)
    return fileNames

def cutToLowPatchAndSaveToH5File_YCrCb_MUL(dataFolderOrigin, patchNum, savePath, size,scales):
    fileNamesOrigin = getAllFile(dataFolderOrigin)
    data = []
    label = []
    for i in fileNamesOrigin:
        fileOr_image = cv2.imread(i)
        fileOr_image=cv2.cvtColor(fileOr_image,cv2.COLOR_BGR2YCrCb)

        height, width, channel = fileOr_image.shape
        for scale in scales:
            fileLw_image = cv2.resize(fileOr_image,(int(width/scale),int(height/scale)),interpolation=cv2.INTER_AREA)
            fileLw_image = cv2.resize(fileLw_image,(width,height),interpolation=cv2.INTER_CUBIC)
            for j in range(patchNum):
                n = int(np.random.rand() * (width - size))
                m = int(np.random.rand() * (height - size))
                patchI = fileLw_image[m:m + size, n:n + size,0]
                patchO = fileOr_image[m:m + size, n:n + size,0]

                patchI = np.reshape(patchI, (size, size, 1))
                patchO = np.reshape(patchO, (size, size, 1))

                data.append(patchI)
                label.append(patchO)

        print(os.path.basename(i), "end")
    data = np.array(data)
    label = np.array(label)
    print('total patch:', len(data))
    with h5py.File(savePath, 'w') as file:
        file.create_dataset('train_data', data=data)
        file.create_dataset('train_label', data=label)
        print('DataSet saved.')

def bicubicInterpolationTofileSave(dataFolder,scale, savePath):
    fileNames = getAllFile(dataFolder)
    for fileName in fileNames:
        data_image = cv2.imread(fileName)
        height,width, channel = data_image.shape
        new_image=cv2.resize(data_image,(int(width*scale),int(height*scale)),interpolation=cv2.INTER_CUBIC)
        name = os.path.basename(fileName)
        newFilePath = os.path.join(savePath, name)
        cv2.imwrite(newFilePath, new_image)
        print(newFilePath)

def reduce_size(srcFolder, destFolder, magnification):
    fileNames = getAllFile(srcFolder)
    for fileName in fileNames:
        image = cv2.imread(fileName)

        height, width, channel = image.shape
        if int(height / magnification)*magnification!=height or int(width /magnification)*magnification!=width:
            print("Inseparable size :",fileName)
            continue
        # 向下取整缩小
        output = cv2.resize(image, (int(width /magnification), int(height / magnification)),
                            interpolation=cv2.INTER_AREA)
        fileName = os.path.basename(fileName)
        newFilePath = os.path.join(destFolder, fileName)
        cv2.imwrite(newFilePath, output)
        print("Shrinking success :",fileName)

def psnr(img1, img2):
    diff = np.abs(img1 - img2)
    mse = np.square(diff).mean()
    psnr = 20 * np.log10(255 / np.sqrt(mse))
    return psnr,mse


def SaveTestDataToH5File(dataFolderLR,dataFolderHR, savePath):
    data = []
    label = []
    LRfileNames = getAllFile(dataFolderLR)
    HRfileNames = getAllFile(dataFolderHR)
    for i in range(len(LRfileNames)):
        imageLR = cv2.imread(LRfileNames[i])
        imageHR = cv2.imread(HRfileNames[i])
        data.append(imageLR)
        label.append(imageHR)
        print(i, "end")

    print('total patch:', len(data))
    with h5py.File(savePath, 'w') as file:
        file.create_dataset('len', data=len(data))
        for i in range(len(data)):
            file.create_dataset('test_data_'+str(i), data=np.array(data[i]))
            file.create_dataset('test_label_'+str(i), data=np.array(label[i]))
        print('testDataSet saved.')


if __name__ == '__main__':
    # bicubicInterpolationTofileSave("C:\\Users\\xiong\\Desktop\\programme\\PycharmProjects\\srcnn-test\\DIV2K_train_LR_mild",4,
    #                                "C:\\Users\\xiong\\Desktop\\programme\\PycharmProjects\\srcnn-test\\DIV2K_train_LR_mild_Interpolation")
    # cutToPatchAndSaveToH5File("C:\\Users\\xiong\\Desktop\\programme\\PycharmProjects\\srcnn-test\\DIV2K_train_LR_mild_Interpolation",
    #                           "C:\\Users\\xiong\\Desktop\\programme\\PycharmProjects\\srcnn-test\\DIV2K_train_HR",
    #                         20,'./dataset.h5',128)
    # cutToLowPatchAndSaveToH5File_YCrCb_MUL('./DIV2K_train_HR', 30, './dataset.h5', 64, [2, 3, 4])
    SaveTestDataToH5File('./SelfExSR-data/Set5/image_SRF_4_LR', './SelfExSR-data/Set5/image_SRF_4_HR', './testdataset.h5')
    # reduce_size('./DIV2K_train_HR','./DIV2K_train_LR_X4',4)
    # bicubicInterpolationTofileSave('./DIV2K_train_LR_X4',4,'./DIV2K_train_LR_X4_Interpolation')
    # bicubicInterpolationTofileSave("./SelfExSR-data/Set5/image_SRF_4_LR",4, "./output")