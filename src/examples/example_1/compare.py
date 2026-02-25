import torch
import torch.nn as nn
import torch.optim as optim
import time

class TicTacToeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 16)
        self.fc2 = nn.Linear(16, 4)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    time0 = time.process_time() 
    model = TicTacToeModel()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(10):
        try:
            with open("tictactoe.txt", "r") as fp:
                for cnt, line in enumerate(fp, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 解析棋盘数据
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
                    
                    input_tensor = torch.tensor(data, dtype=torch.float32).view(1, -1)
                    label_tensor = torch.tensor([label])
                    
                    output = model(input_tensor)
                    loss = loss_fn(output, label_tensor)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # 记录loss
                    if cnt % 10 == 0:
                        with open("loss_py.txt", "a") as tp:
                            tp.write(f"loss:{100000 * loss.item():10.6f}\n")
        except FileNotFoundError:
            print("file open error")
    
    time1 = time.process_time() 
    print(f"{time1-time0:.0f}s")