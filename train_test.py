import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from lib.models.fastreg import FastReg
import config
from lib.dataset.FEMdata import load_data_from_directory, split_data, repeat_tensor_elements

def executeEpoch(model, loader, opt, sched, e, sw, mode='train'):
    assert mode == 'train' or mode =='val', 'mode should be train or val'

    if mode == 'train':
        model.train()
    else:
        model.eval()

    lE, lpE, lnE = 0, 0, 0
    rotE, transE = 0, 0
    maxInliersE, actualInliersE = 0, 0

    for coord, output in loader:
        coord = coord.cuda()
        output = output.cuda()

        if mode == 'train':


def train(args):
    input_data = load_data_from_directory(config.INPUT_PATH,needsorted=True)
    coord_data = load_data_from_directory(config.COORD_PATH,needsorted=False)
    output_data = load_data_from_directory(config.OUTPUT_PATH,needsorted=True)
    new_coord_data = [repeat_tensor_elements(tensor, 100) for tensor in coord_data][0]


    input_data = torch.Tensor(input_data)
    new_coord_data = torch.Tensor(new_coord_data)
    output_data = torch.Tensor(output_data)

    #input_data = torch.tensor([s.stripe() for s in input_data])
    #new_coord_data = torch.tensor([s.stripe() for s in new_coord_data])
    #output_data = torch.tensor([s.stripe() for s in output_data])

    print(input_data)
    print(new_coord_data)

    assert all(isinstance(t, torch.Tensor) for t in input_data)
    assert all(isinstance(t, torch.Tensor) for t in new_coord_data)
    assert all(isinstance(t, torch.Tensor) for t in output_data)

    combined_data = [torch.cat(tensors, dim=1) for tensors in zip(new_coord_data, output_data)]
    trainDataset, valDataset = split_data(combined_data)

    print(trainDataset.dataset)
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=config.batch_size, pin_memory=True, drop_last=True, num_workers=config.batch_size, shuffle=True)
    valLoader= torch.utils.data.DataLoader(valDataset, batch_size=config.batch_size, pin_memory=True, drop_last=True, num_workers=config.batch_size)

    model = FastReg(config.T).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-1, eps=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

    expPath = 'runs/'
    writer = SummaryWriter(expPath)

    for e in range(config.epochs):
        executeEpoch(model, trainLoader, opt, sched, e, writer, mode='train')

        if (e + 1) % config.val_period == 0:
            # run validation
            executeEpoch(model, valLoader, opt, sched, e, writer, mode='val')

            # saves model
            torch.save(model.state_dict(), f'{expPath}/model{e}.pth')

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains FastReg registration model')
    args = parser.parse_args()
    train(args)