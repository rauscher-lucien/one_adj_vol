import os
import torch
import matplotlib.pyplot as plt
import pickle
import time
import logging

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


from utils import *
from transforms import *
from dataset import *

from model_v_1 import *
from model import *


class Trainer:

    def __init__(self, data_dict):

        self.train_data_path = data_dict['train_data_path']

        self.project_dir = data_dict['project_dir']
        self.project_name = data_dict['project_name']

        self.results_dir, self.checkpoints_dir = create_result_dir(self.project_dir, self.project_name)
        self.train_results_dir, self.val_results_dir = create_train_val_dir(self.results_dir)

        self.num_epoch = data_dict['num_epoch']
        self.batch_size = data_dict['batch_size']
        self.lr = data_dict['lr']

        self.num_freq_disp = data_dict['num_freq_disp']
        self.train_continue = data_dict['train_continue']

        self.log_scaling = data_dict['log_scaling']
        self.model_depth = data_dict['model_depth']
        self.num_volumes = data_dict['num_volumes']

        self.device = get_device()

        self.writer = SummaryWriter(self.results_dir + '/tensorboard_logs')


    def save(self, checkpoints_dir, model, optimizer, epoch):
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch}, 
                os.path.join(checkpoints_dir, 'best_model.pth'))

        

    def load(self, checkpoints_dir, model, device, epoch=[], optimizer=[]):

        dict_net = torch.load('%s/best_model.pth' % (checkpoints_dir), map_location=device)

        model.load_state_dict(dict_net['model'])
        optimizer.load_state_dict(dict_net['optimizer'])
        epoch = dict_net['epoch']

        print('Loaded %dth network' % epoch)

        return model, optimizer, epoch
    

    def train(self):

        start_time = time.time()
        mean, std = compute_global_mean_and_std(self.train_data_path, self.checkpoints_dir, num_volumes_to_use=self.num_volumes)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")


        transform_train = transforms.Compose([
            Normalize(mean, std),
            RandomCrop(output_size=(64,64)),
            RandomHorizontalFlip(),
            ToTensor()
        ])


        transform_inv_train = transforms.Compose([
            ToNumpy(),
            Denormalize(mean, std)
        ])


        ### make dataset and loader ###

        dataset_train = N2NDataset(self.train_data_path, num_volumes=self.num_volumes, transform=transform_train)

        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0)

        num_train = len(dataset_train)
        num_batch_train = int((num_train / self.batch_size) + ((num_train % self.batch_size) != 0))


        ### initialize network ###

        model = UNet(1, 1, depth=self.model_depth).to(self.device)

        criterion = nn.MSELoss().to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), self.lr)

        st_epoch = 0
        best_train_loss = float('inf')

        if self.train_continue == 'on':
            print(self.checkpoints_dir)
            model, optimizer, st_epoch = self.load(self.checkpoints_dir, model, self.device, st_epoch, optimizer)
            model = model.to(self.device)


        for epoch in range(st_epoch + 1, self.num_epoch + 1):
            model.train()  # Ensure model is in training mode
            train_loss = 0.0

            for batch, data in enumerate(loader_train, 1):

                input_img, target_img = [x.squeeze(0).to(self.device) for x in data]
                denoised_input = model(input_img)

                #plot_intensity_line_distribution(input_img, 'input')
                #plot_intensity_line_distribution(denoised_input, 'output')
                #plot_intensity_line_distribution(target_img, 'target')

                loss = criterion(denoised_input, target_img)


                train_loss += loss.item() 
                loss.backward()
                optimizer.step()

                
            if epoch % self.num_freq_disp == 0:
                # Assuming transform_inv_train can handle the entire stack
                input_img = transform_inv_train(input_img)[..., 0]
                target_img = transform_inv_train(target_img)[..., 0]
                denoised_input = transform_inv_train(denoised_input)[..., 0]

                #plot_intensity_line_distribution(input_img, 'input')
                #plot_intensity_line_distribution(denoised_input, 'output')

                for j in range(target_img.shape[0]):
                    
                    plt.imsave(os.path.join(self.train_results_dir, f"{j}_input.png"), input_img[j, :, :], cmap='gray')
                    plt.imsave(os.path.join(self.train_results_dir, f"{j}_target.png"), target_img[j, :, :], cmap='gray')
                    plt.imsave(os.path.join(self.train_results_dir, f"{j}_output.png"), denoised_input[j, :, :], cmap='gray')

            avg_train_loss = train_loss / len(loader_train)
            self.writer.add_scalar('Loss/train', avg_train_loss, epoch)

            print(f'Epoch [{epoch}/{self.num_epoch}], Train Loss: {avg_train_loss:.4f}')

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                self.save(self.checkpoints_dir, model, optimizer, epoch)
                print(f"Saved best model at epoch {epoch} with loss {best_train_loss:.4f}.")
 
        self.writer.close()
