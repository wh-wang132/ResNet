import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from prettytable import PrettyTable


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.0
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.0
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.0
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self, folder_path):
        matrix = self.matrix
        print(matrix)

        # 增加图形尺寸3倍，提高分辨率
        plt.figure(figsize=(20, 16))
        plt.imshow(matrix, cmap=cm.get_cmap("Blues"))

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45, fontsize=28)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels, fontsize=28)
        # 显示colorbar
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=24)
        plt.xlabel("True Labels", fontsize=32)
        plt.ylabel("Predicted Labels", fontsize=32)
        # plt.title(folder_path + ' Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(
                    x,
                    y,
                    str(info),
                    verticalalignment="center",
                    horizontalalignment="center",
                    color="white" if info > thresh else "black",
                    fontsize=24,
                )
        plt.tight_layout()
        # DPI设为原来的2倍（默认约为100，设为200）
        plt.savefig(
            folder_path + "/" + " Confusion_matrix.png", dpi=100, bbox_inches="tight"
        )
        plt.close()
