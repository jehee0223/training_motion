import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from IPython.display import clear_output
import gc
from torch.cuda import memory_allocated, empty_cache
import numpy as np

if torch.cuda.is_available() == True:
    device = 'cuda:0'
    print('GPU 사용 가능')
else:
    device = 'cpu'
    print('GPU 사용 불가능')

length = 60


class skeleton_LSTM(nn.Module):
    def __init__(self):
        super(skeleton_LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=24, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=256, hidden_size=512, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        # self.batch_norm1 = nn.BatchNorm1d(256)
        self.lstm4 = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.lstm5 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm6 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)
        # self.batch_norm2 = nn.BatchNorm1d(64)
        self.lstm7 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = nn.Linear(32, 4)
        # self.lstm_1=nn.LSTM(input_size=32,hidden_size=64, num_layers=1, batch_first=True)
        # self.dropout_1 = nn.Dropout(0.3)
        # self.batch_norm1 = nn.BatchNorm1d(64)

        # self.lstm_2 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        # self.dropout_2 = nn.Dropout(0.5)
        # self.batch_norm2 = nn.BatchNorm1d(16)
        # self.lstm_3 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        # self.lstm_4 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        # self.fc_1 = nn.Linear(32, 2)


    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = self.dropout1(x)
        # x = self.batch_norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x, _ = self.lstm4(x)
        x, _ = self.lstm5(x)
        x, _ = self.lstm6(x)
        x = self.dropout2(x)
        # x = self.batch_norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x, _ = self.lstm7(x)
        x = self.fc(x[:, -1, :])
        # x, _ = self.lstm_1(x)
        # x = self.dropout_1(x)
        # x = self.batch_norm1(x.permute(0, 2, 1)).permute(0, 2, 1)

        # x, _ = self.lstm_2(x)
        # x = self.dropout_2(x)
        # x = self.batch_norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        # x = self.dropout_2(x)
        # x, _ = self.lstm_3(x)
        # x, _ = self.lstm_4(x)
        # x = self.fc_1(x[:, -1, :])
        return x


def init_model():
    plt.rc('font', size=10)
    global net, loss_fn, optim
    net = skeleton_LSTM().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = Adam(net.parameters(), lr=0.0001)


def init_epoch():
    global epoch_cnt
    epoch_cnt = 0


def init_log():
    plt.rc('font', size=10)
    global log_stack, iter_log, tloss_log, tacc_log, vloss_log, vacc_log, time_log
    iter_log, tloss_log, tacc_log, vloss_log, vacc_log, time_log, log_stack = [], [], [], [], [], [], []


def record_train_log(t_loss, t_acc, t_time):
    time_log.append(t_time)
    tloss_log.append(t_loss)
    tacc_log.append(t_acc)
    iter_log.append(epoch_cnt)


def record_valid_log(v_loss, v_acc):
    vloss_log.append(v_loss)
    vacc_log.append(v_acc)


def last(log_list):
    if len(log_list) > 0:
        return log_list[len(log_list) - 1]
    else:
        return -1


def print_log():
    # 학습 추이 출력

    # 소숫점 3자리 수까지 조절
    train_loss = round(float(last(tloss_log)), 10)
    train_acc = round(float(last(tacc_log)), 10)
    val_loss = round(float(last(vloss_log)), 10)
    val_acc = round(float(last(vacc_log)), 10)
    time_spent = round(float(last(time_log)), 10)

    log_str = 'Epoch: {:3} | T_Loss {:5} | T_acc {:10} | V_Loss {:5} | V_acc. {:5} | \
🕒 {:5}'.format(last(iter_log), train_loss, train_acc, val_loss, val_acc, time_spent)

    log_stack.append(log_str) # 프린트 준비

    # 학습 추이 그래프 출력
    hist_fig, loss_axis = plt.subplots(figsize=(10, 3), dpi=99) # 그래프 사이즈 설정
    hist_fig.patch.set_facecolor('white') # 그래프 배경색 설정

    # Loss Line 구성
    loss_t_line = plt.plot(iter_log, tloss_log, label='Train Loss', color='red', marker='o')
    loss_v_line = plt.plot(iter_log, vloss_log, label='Valid Loss', color='blue', marker='s')
    loss_axis.set_xlabel('epoch')
    loss_axis.set_ylabel('loss')

    # Acc. Line 구성
    acc_axis = loss_axis.twinx()
    acc_t_line = acc_axis.plot(iter_log, tacc_log, label='Train Acc.', color='red', marker='+')
    acc_v_line = acc_axis.plot(iter_log, vacc_log, label='Valid Acc.', color='blue', marker='x')
    acc_axis.set_ylabel('accuracy')

    # 그래프 출력
    hist_lines = loss_t_line + loss_v_line + acc_t_line + acc_v_line # 위에서 선언한 plt정보들 통합
    loss_axis.legend(hist_lines, [l.get_label() for l in hist_lines]) # 순서대로 그려주기
    loss_axis.grid() # 격자 설정
    plt.title('Learning history until epoch {}'.format(last(iter_log)))
    plt.draw()

    # 텍스트 로그 출력
    clear_output(wait=True)
    plt.show()
    for idx in reversed(range(len(log_stack))): # 반대로 sort 시켜서 출력
        print(log_stack[idx])


def clear_memory():
    if device != 'cpu':
        empty_cache()
    gc.collect()


def epoch(data_loader, mode = 'train'):
    global epoch_cnt

    # 사용되는 변수 초기화
    iter_loss, iter_acc, last_grad_performed = [], [], False

    # 1 iteration 학습 알고리즘(for문을 나오면 1 epoch 완료)
    for _data, _label in data_loader:
        data, label = _data.to(device), _label.type(torch.LongTensor).to(device)

        # 1. Feed-forward
        if mode == 'train':
            net.train()
        else:
            # 학습때만 쓰이는 Dropout, Batch Mormalization을 미사용
            net.eval()

        result = net(data) # 1 Batch에 대한 결과가 모든 Class에 대한 확률값으로
        _, out = torch.max(result, 1) # result에서 최대 확률값을 기준으로 예측 class 도출

        # 2. Loss 계산
        loss = loss_fn(result, label) # GT 와 Label 비교하여 Loss 산정
        iter_loss.append(loss.item()) # 학습 추이를 위하여 Loss를 기록

        # 3. 역전파 학습 후 Gradient Descent
        if mode == 'train':
            optim.zero_grad() # 미분을 통해 얻은 기울기르 초기화 for 다음 epoch
            loss.backward() # 역전파 학습
            optim.step() # Gradient Descent 수행
            last_grad_performed = True # for문 나가면 epoch 카운터 += 1

        # 4. 정확도 계산
        acc_partial = (out == label).float().sum() # GT == Label 인 개수
        acc_partial = acc_partial / len(label) # ( TP / (TP + TN)) 해서 정확도 산출
        iter_acc.append(acc_partial.item()) # 학습 추이를 위하여 Acc. 기록

    # 역전파 학습 후 Epoch 카운터 += 1
    if last_grad_performed:
        epoch_cnt += 1

    clear_memory()

    # loss와 acc의 평균값 for 학습추이 그래프, 모든 GT와 Label값 for 컨퓨전 매트릭스
    return np.average(iter_loss), np.average(iter_acc)

def epoch_not_finished():
    # return epoch_cnt < training.maximum_epoch
    return epoch_cnt < 50
