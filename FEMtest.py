import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader

from model import FEMtest
import config
from lib.dataset.FEMdata import load_data_from_directory, split_data, repeat_tensor_elements, CustomDataset

def executeEpoch(model, loader, loss_function, opt, sched, e, sw, mode='train', lE=None):
    assert mode == 'train' or mode =='val', 'mode should be train or val'
    lE = 0.0
    if mode == 'train':
        model.train()
    else:
        model.eval()

    for b, (input, coord, output) in enumerate(loader):
        input = input.cuda()
        coord = coord.cuda()
        output = output.cuda()
        if mode == 'train':
            output_xyz, _ = model(input, coord)
            source = output_xyz
            #print(source.shape)
            #print(output-output_xyz)
            loss1 = loss_function(source, output)
            loss1.backward()
            opt.step()
            opt.zero_grad()
            print(f'Epoch {e}/B {b}. Loss {loss1.item()}')

        lE+=loss1.item()

    lE /= len(loader)
    sw.add_scalar(f'{mode}/loss', lE, e)
    if mode == 'train':
        sched.step()


def train(args):

    input_data = load_data_from_directory(config.INPUT_PATH,needsorted=True)
    coord_data = load_data_from_directory(config.COORD_PATH,needsorted=False)
    output_data = load_data_from_directory(config.OUTPUT_PATH,needsorted=True)
    new_coord_data = [repeat_tensor_elements(tensor, 150) for tensor in coord_data][0]
    num = 0
    for i in range (len(output_data)):
        for j in range(len(output_data[i])):
            for k in range(len(output_data[i][j])):
                if output_data[i][j][k] > 1e5 or output_data[i][j][k] < -1e5:
                    output_data[i][j][k] = 0
                    num +=1

    print('num =', num)


    input_data = torch.Tensor(input_data)
    new_coord_data = torch.Tensor(new_coord_data)
    output_data = torch.Tensor(output_data)
    #print(output_data)

    assert all(isinstance(t, torch.Tensor) for t in input_data)
    assert all(isinstance(t, torch.Tensor) for t in new_coord_data)
    assert all(isinstance(t, torch.Tensor) for t in output_data)

    dataset = CustomDataset(input_data, new_coord_data, output_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, pin_memory=True, drop_last=True, num_workers=config.batch_size, shuffle=True)


    model = FEMtest().cuda()
    loss_fn = torch.nn.L1Loss(reduction='mean')
    opt = torch.optim.Adam(model.parameters(), lr=1e-1, eps=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

    print(model)

    expPath = 'runs/'
    writer = SummaryWriter(expPath)

    for e in range(config.epochs):
        executeEpoch(model, dataloader, loss_fn, opt, sched, e, writer, mode='train')

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains FastReg registration model')
    args = parser.parse_args()
    train(args)