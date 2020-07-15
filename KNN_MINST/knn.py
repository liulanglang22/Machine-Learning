import  operator
import  struct
import  numpy as np

# 训练集
train_images_file = 'train-images.idx3-ubyte'
# 训练集标签
train_labels_file = 'train-labels.idx1-ubyte'
# 测试集
test_images_file = 't10k-images.idx3-ubyte'
# 测试集标签
test_labels_file = 't10k-labels.idx1-ubyte'


def decode_file(idx3_ubyte):
    bin_data = open(idx3_ubyte, 'rb').read()
    # 解析文件头信息，依次是魔数、图片数量、每张图片高和宽
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))
    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已经解析 %d 张' % (i + 1))
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_labels(idx1_ubyte):
    bin_data = open(idx1_ubyte, 'rb').read()
    # 解析文件头信息，依次是魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))
    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d 张' % (i + 1))
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)

    return labels


def classify(inX, dataset, labels, k):
    datasetsize = dataset.shape[0]

    # 距离计算公式
    diffMat = np.tile(inX, (datasetsize, 1)) - dataset
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5

    # 距离从大到小排序，返回距离的序号
    sortDistance = distances.argsort()
    # 字典
    classCount = {}

    # 前k个距离最小的
    for i in range(k):
        # sortDistance[0]返回的是距离最小的数据样本的序号
        # labels[sortDistance[0]]距离最小的数据样本的标签
        voteIlabel = labels[sortDistance[i]]
        # 以标签为key，支持该标签+1
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 排序
    sortClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)
    return sortClassCount[0][0]


if __name__ == '__main__':
    train_images = decode_file(train_images_file)
    train_labels = decode_labels(train_labels_file)
    test_images = decode_file(test_images_file)
    test_labels = decode_labels(test_labels_file)

    m = 60000
    trainMat = np.zeros((m, 784))
    # 文件名下划线左边的数字是标签
    for i in range(m):
        for j in range(28):
            for k in range(28):
                trainMat[i, 28 * j + k] = train_images[i][j][k]
    errorCount = 0.0
    mTest = 10000
    for i in range(mTest):
        classNumStr = test_labels[i]
        vectorUnderTest = np.zeros(784)
        for j in range(28):
            for k in range(28):
                vectorUnderTest[28 * j + k] = test_images[i][j][k]  # 第i幅测试图

        Result = classify(vectorUnderTest, trainMat, train_labels, 3)
        if Result != classNumStr:
            errorCount += 1.0
            print("识别错误")
            print("识别结果：%d 正确答案：%d" % (Result, classNumStr))
        if (i + 1) % 100 == 0:
            print('已经测试了 %d 张' % (i + 1))
    print("\n错误数： %d" % errorCount)
    print("\n错误率： %f" % (errorCount / float(mTest)))
    print('正确率：%f' % (1 - (errorCount / float(mTest))))



