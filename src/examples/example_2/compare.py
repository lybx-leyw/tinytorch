import torch
import torch.nn as nn
import torch.optim as optim
import time

class ConvTicTacToeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0)
        self.fc = nn.Linear(4, 4)
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    time0 = time.process_time()
    model = ConvTicTacToeModel()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        try:
            with open("tictactoe.txt", "r") as fp:
                for cnt, line in enumerate(fp, 1):
                    line = line.strip()
                    if not line:
                        continue

                    # 解析棋盘数据 - 转换为3x3矩阵
                    data = []
                    for i in range(9):
                        if line[i] == 'X':
                            data.append(1.0)
                        elif line[i] == 'O':
                            data.append(2.0)
                        else:
                            data.append(0.0)

                    # 解析标签
                    label_map = {'P': 0, 'N': 1, 'M': 2, 'Q': 3}
                    label = label_map.get(line[10])
                    if label is None:
                        print("label error")

                    # 创建输入张量 [1, 3, 3, 1]
                    input_tensor = torch.tensor(data, dtype=torch.float32).view(1, 3, 3, 1)
                    label_tensor = torch.tensor([label])

                    output = model(input_tensor)
                    loss = loss_fn(output, label_tensor)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # 记录loss
                    if cnt % 10 == 0:
                        with open("loss_conv_py.txt", "a") as tp:
                            tp.write(f"loss:{100000 * loss.item():10.6f}\n")
        except FileNotFoundError:
            print("file open error")

    time1 = time.process_time()
    print(f"Training time: {time1-time0:.0f}s")
