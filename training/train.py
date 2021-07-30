from tqdm import tqdm
import torch

def train(train_loader, model, criterion, optimizer, epoch, verbose=False, scale=4, n_c=128):
    epoch_loss = 0
    h_eye = None
    w_eye = None

    model.train()
    data_enumerator = enumerate(tqdm(train_loader)) if verbose else enumerate(train_loader)
    for iteration, data in data_enumerator:
        x_input, target = data[0], data[1] # input and target are both tensor, input:[N,C,T,H,W] , target:[N,C,H,W]

        x_input = x_input.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        B, _, T, _ ,_ = x_input.shape
        out = []
        init = True
        for i in range(T-1):
            if init:
                init_temp = torch.zeros_like(x_input[:,0:1,0,:,:])  # (N, 1, H, W)
                init_o = init_temp.repeat(1, scale*scale*3,1,1)  # (N, C=3scale^2, H, W)
                init_h = init_temp.repeat(1, n_c, 1,1)  # (N, C=n_c, H, W)
                h, prediction = model(x_input[:, :, i:i + 2, :, :], init_h, init_o, h_eye, w_eye, init) # 2 frames
                out.append(prediction)
                init = False
            else:
                h, prediction = model(x_input[:, :, i:i + 2, :, :], h, prediction, h_eye, w_eye, init)
                out.append(prediction)
        prediction = torch.stack(out, dim=2) # over T dimension
        loss = criterion(prediction, target)/(B*T)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(train_loader), loss.item(), (t1 - t0)))
    print("===> Epoch[{}]: Loss: {:.4f}".format(epoch, epoch_loss/len(train_loader)))


