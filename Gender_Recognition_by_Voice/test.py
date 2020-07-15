import numpy as np
import csv


N = 29


def load_data_set(file_name):
    """
    :param file_name: 文件名字
    :return

    train_mat：离散化的训练数据集
    train_classes： 训练数据集所属的分类
    test_mat：离散化的测试数据集
    test_classes：测试数据集所述的分类
    label_name：特征的名称
    """

    data_mat = []
    with open(file_name) as file_obj:
        voice_reader = csv.DictReader(file_obj)
        list_class = []
        # 文件头
        label_name = list(voice_reader.fieldnames)
        num = len(label_name) - 1

        for line in voice_reader.reader:
            data_mat.append(line[:num])
            gender = 1 if line[-1] == 'male' else 0
            list_class.append(gender)

        # 求每一个特征的平均值
        data_mat = np.array(data_mat).astype(float)
        count_vector = np.count_nonzero(data_mat, axis=0)
        sum_vector = np.sum(data_mat, axis=0)
        mean_vector = sum_vector / count_vector

        # 数据缺失的地方 用 平均值填充
        for row in range(len(data_mat)):
            for col in range(num):
                if data_mat[row][col] == 0.0:
                    data_mat[row][col] = mean_vector[col]

        # 将数据连续型的特征值离散化处理
        min_vector = data_mat.min(axis=0)
        max_vector = data_mat.max(axis=0)
        diff_vector = max_vector - min_vector
        diff_vector /= N

        new_data_set = []
        for i in range(len(data_mat)):
            line = np.array((data_mat[i] - min_vector) / diff_vector).astype(int)
            new_data_set.append(line)

        # 随机划分数据集为训练集和测试集
        test_set = list(range(len(new_data_set)))
        train_set = []
        for i in range(2168):
            random_index = int(np.random.uniform(0, len(test_set)))
            train_set.append(test_set[random_index])
            del test_set[random_index]

        # 训练数据集
        train_mat = []
        train_classes = []
        for index in train_set:
            train_mat.append(new_data_set[index])
            train_classes.append(list_class[index])

        # 测试数据集
        test_mat = []
        test_classes = []
        for index in test_set:
            test_mat.append(new_data_set[index])
            test_classes.append(list_class[index])

    return train_mat, train_classes, test_mat, test_classes, label_name


def native_bayes(train_matrix, list_classes):
    """
    :param train_matrix: 训练样本矩阵
    :param list_classes: 训练样本分类向量
    :return:p_1_class 任一样本分类为1的概率  p_feature,p_1_feature 分别为给定类别的情况下所以特征所有取值的概率
    """

    # 训练样本个数
    num_train_data = len(train_matrix)
    num_feature = len(train_matrix[0])
    # 分类为1的样本占比
    p_1_class = sum(list_classes) / float(num_train_data)

    n = N + 1
    list_classes_1 = []
    train_data_1 = []

    for i in list(range(num_train_data)):
        if list_classes[i] == 1:
            list_classes_1.append(i)
            train_data_1.append(train_matrix[i])

    # 分类为1 情况下的各特征的概率
    train_data_1 = np.matrix(train_data_1)
    p_1_feature = {}
    for i in list(range(num_feature)):
        feature_values = np.array(train_data_1[:, i]).flatten()
        # 避免某些特征值概率为0 影响总体概率，每个特征值最少个数为1
        feature_values = feature_values.tolist() + list(range(n))
        p = {}
        count = len(feature_values)
        for value in set(feature_values):
            p[value] = np.log(feature_values.count(value) / float(count))
        p_1_feature[i] = p

    # 所有分类下的各特征的概率
    p_feature = {}
    train_matrix = np.matrix(train_matrix)
    for i in list(range(num_feature)):
        feature_values = np.array(train_matrix[:, i]).flatten()
        feature_values = feature_values.tolist() + list(range(n))
        p = {}
        count = len(feature_values)
        for value in set(feature_values):
            p[value] = np.log(feature_values.count(value) / float(count))
        p_feature[i] = p

    return p_feature, p_1_feature, p_1_class


def classify_bayes(test_vector, p_feature, p_1_feature, p_1_class):
    """
    :param test_vector: 要分类的测试向量
    :param p_feature: 所有分类的情况下特征所有取值的概率
    :param p_1_feature: 类别为1的情况下所有特征所有取值的概率
    :param p_1_class: 任一样本分类为1的概率
    :return: 1 表示男性 0 表示女性
    """
    # 计算每个分类的概率(概率相乘取对数 = 概率各自对数相加)
    sum = 0.0
    for i in list(range(len(test_vector))):
        sum += p_1_feature[i][test_vector[i]]
        sum -= p_feature[i][test_vector[i]]
    p1 = sum + np.log(p_1_class)
    p0 = 1 - p1
    if p1 > p0:
        return 1
    else:
        return 0


def test_bayes():
    file_name = 'voice.csv'
    train_mat, train_classes, test_mat, test_classes, label_name = load_data_set(file_name)

    p_feature, p_1_feature, p_1_class = native_bayes(train_mat, train_classes)

    count = 0.0
    correct_count = 0.0
    man_correct = 0.0
    man_count = 0.0
    for i in list(range(len(test_mat))):
        test_vector = test_mat[i]
        result = classify_bayes(test_vector, p_feature, p_1_feature, p_1_class)
        if test_classes[i] == 1 :
            man_count += 1
        if result == test_classes[i]:
            correct_count += 1
            if result == 1:
                man_correct += 1
        count += 1
    correct_rate = 0.0
    man_correct_rate = 0.0
    woman_correct_rate = 0.0
    correct_rate = correct_count / count
    man_correct_rate = man_correct/man_count
    woman_correct_rate=(correct_count-man_correct)/(count-man_count)
    print(correct_rate)
    print('man:',man_correct_rate)
    print('woman:', woman_correct_rate)
    rate = [man_correct_rate,woman_correct_rate,correct_rate]
    with open('data.csv', 'a', newline='') as file:
        row = rate
        csv.writer(file).writerow(row)


if __name__ == '__main__':
    for i in range(30):
        test_bayes()
    empty = []
    with open('data.csv', 'a', newline='') as file:
        row = empty
        csv.writer(file).writerow(row)
