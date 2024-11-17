import torch
import time
import LSTM as model
from tensor_split import train_loader, val_loader
from tqdm import tqdm


model.init_model()
model.init_epoch()
model.init_log()
maximum_epoch = 50 # LSTM.py 마지막에 상수로 지정(변경시에 직접 가서 변경필요)

while model.epoch_not_finished():

    start_time = time.time()
    tloss, tacc = model.epoch(train_loader, mode='train')
    # with tqdm(train_loader, desc="Training", leave=False) as train_bar:
    #     for batch_data in train_bar:
    #         batch_loss, batch_acc = model.epoch(train_loader, mode='train')
    #         tloss += batch_loss
    #         tacc += batch_acc
    #         # 현재 배치의 손실과 정확도를 tqdm에 업데이트
    #         train_bar.set_postfix(loss=batch_loss, acc=batch_acc)
    end_time = time.time()
    time_taken = end_time - start_time
    model.record_train_log(tloss, tacc, time_taken)
    with torch.no_grad():
        # vloss, vacc = model.epoch(val_loader, mode='val')
        # with tqdm(val_loader, desc="Validation", leave=False) as val_bar:
        #     for batch_data in val_bar:
        #         batch_loss, batch_acc = model.epoch(val_loader, mode='val')
        #         vloss += batch_loss
        #         vacc += batch_acc
        #         # 현재 배치의 손실과 정확도를 tqdm에 업데이트
        #         val_bar.set_postfix(loss=batch_loss, acc=batch_acc)
        vloss, vacc= model.epoch(val_loader, mode='val')
        model.record_valid_log(vloss, vacc)

model.print_log()
net = model.skeleton_LSTM()
torch.save(net.state_dict(), 'test.pth')
print('\n Training completed')

# # 정확도 검증
# with torch.no_grad():
#     test_loss,test_acc=model.epoch(test_loader,mode='test')
#     model.test_acc=round(test_acc,4)
#     model.test_loss=round(test_loss,4)
#     print('Test Acc: {}'.format(test_acc))
#     print('Test Loss: {}'. format(test_loss))
