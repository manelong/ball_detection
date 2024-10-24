import os.path

import torch


def print_train_loss(model_path_tmp):
    for i in range(1, 21):
        model_path = model_path_tmp % i

        # read the model
        checkpoint = torch.load(model_path)

        print(i, checkpoint['train_loss'])

        # save the checkpoint['train_loss'] to a txt file, keeping six decimal places
        with open('%s/train_loss.txt' % os.path.dirname(model_path_tmp), 'a') as f:
            f.write(str(i) + ' ' + str(round(checkpoint['train_loss'], 6)) + '\n')


if __name__ == '__main__':
    print_train_loss('../checkpoint/2024-05-29_16-31-41/checkpoint_ep%d.pth.tar')