import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torchattacks
from torch import unsqueeze

# Pulled from this awesome blog post:
# https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
class LeNet5(nn.Module):
    
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return logits#, probs

# Helper function for training a model
def train_model(model, 
                num_epochs, 
                learning_rate, 
                device, 
                criterion,
                optimizer, 
                train_loader, 
                adv = False, 
                atk = None
                ):
    # For updating learning rate
    def update_lr(optimizer, lr):    
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    # Send model to gpu if available
    model.to(device)
    # Train the model
    total_step = len(train_loader)
    curr_lr = learning_rate
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            if adv:
                if atk is None:
                    print('ERROR: adv=True called with no attack passed')
                    break
                atk_inds = np.random.choice(range(images.shape[0]), size=images.shape[0]//2, replace=False)
                orig_inds = [ind for ind in range(images.shape[0]) if ind not in atk_inds]
                adv_images = atk(images[atk_inds], labels[atk_inds])
                orig_images = images[orig_inds]
                mixed_set = torch.cat((adv_images, orig_images))
                mixed_labels = torch.cat((labels[atk_inds], labels[orig_inds]))
                outputs = model(mixed_set)
                loss = criterion(outputs, mixed_labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        # Decay learning rate
        if (epoch+1) % 4 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)
    
    return model

# Helper function for testing a model
def test_model(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            test_loss += criterion(outputs, labels)
            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# Helper function for getting losses and accuracies
def get_accs(model, test_loader, criterion, device, eps, alpha, steps, adv=True):
    # Create attacker
    atk = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps, random_start=True)
    # Instantiate result dictionary -> we will return this dict with wrong guesses, total guesses, % wrong and avg loss
    result_dict = {
        0: [0, 0, 0, 0],
        1: [0, 0, 0, 0],
        2: [0, 0, 0, 0],
        3: [0, 0, 0, 0],
        4: [0, 0, 0, 0],
        5: [0, 0, 0, 0],
        6: [0, 0, 0, 0],
        7: [0, 0, 0, 0],
        8: [0, 0, 0, 0],
        9: [0, 0, 0, 0]
    }

    for i, (raw_images, labels) in enumerate(test_loader):
        print("Working on round {}".format(i+1))
        # It might take too long to enumerate through the entire test set
        # We will test on 10 rounds of 100 examples (1/10th test set total)
        # This should yield ~100 examples of each class
        if i>=10:
            break
        # Get images from batch one at a time
        for j in range(raw_images.shape[0]):
            raw_image = raw_images[j, ::]
            label = labels[j]
            
            # Create adv image
            if adv:
                image = atk(unsqueeze(raw_image, 0).to(device), unsqueeze(label, 0).to(device))
            else: 
                image = unsqueeze(raw_image, 0).to(device)
            # Get guess on adversary
            outputs = model(image)
            # print(outputs.shape)
            # print(unsqueeze(label, 0).shape)
            loss = criterion(outputs, unsqueeze(label, 0).to(device))

            guess = outputs.max(1, keepdim=True)[1].item()
            # If wrong, increment num wrong for class
            if label.item()  != guess:
                result_dict[label.item()][0] += 1
            # Increment num tested for class
            result_dict[label.item()][1] += 1
            # Record Loss
            result_dict[label.item()][3] += loss.item()

    # Calculate % correct for each class
    for key in result_dict.keys():
        result_dict[key][2] = result_dict[key][0]/result_dict[key][1]
        result_dict[key][3] = result_dict[key][3]/result_dict[key][1]
    return result_dict

# Using the above class, plot the loss/avg accuracy per class
def show_both_losses_by_class(pgd_loss, regular_loss, acc=False):
    labels = np.arange(10)
    pgd_losses = []
    regular_losses = []
    for key in pgd_loss:
        if acc:
            pgd_losses.append(pgd_loss[key][2])
            regular_losses.append(regular_loss[key][2])
        else:
            pgd_losses.append(pgd_loss[key][3])
            regular_losses.append(regular_loss[key][3])
    if acc:
        lim = 1.1
    else:
        lim = max(max(pgd_losses), max(regular_losses))
    plt.figure()
    plt.plot(labels, pgd_losses, color='r')
    plt.plot(labels, regular_losses, color='b')
    plt.xlabel('Class')
    plt.ylim(0, lim)
    plt.ylabel('Loss')
    plt.legend(['PGD Err', 'STD Err'])
    plt.show()
    
# Helper function for updating phis for FRL Reweight Algorithm
def update_phis(model, 
                atk, 
                phi,
                a1, 
                a2,
                t1,
                t2,
                val_loader,
                device, 
                early_stop=20
                ):
    #instantiate label_wise natural and boundary error
    r_nat_i = torch.zeros(10).to(device)
    r_nat_label_cnt = torch.zeros(10).to(device)
    r_bnd_i = torch.zeros(10).to(device)
    r_bnd_label_cnt = torch.zeros(10).to(device)
    r_nat = 0
    r_bnd = 0

    # add count variable to be able to average phi values
    count = 0
    for i, (images, labels) in enumerate(val_loader):
        # Increment count for averaging error values later
        count +=1
        # If early stopping (for efficiency,) break out of loop
        if i>early_stop:
            break

        # Send images & labels to device
        images = images.to(device)
        labels = labels.to(device)

        ### Calculate Natural Error
        outputs = model(images)
        pred = outputs.max(1, keepdim=True)[1]
        # total natural error is sum(pred != label)/(size)
        r_nat += pred.ne(labels.view_as(pred)).sum().item()/len(labels)
        # Calculate label-wise natural error
        for j in range(len(phi[0])):
            inds = torch.where(labels == j)
            ind_preds = pred[inds]
            # Track how many were incorrect, and how many were observed
            r_nat_i[j] += ind_preds.ne(labels[inds].view_as(ind_preds)).sum().item()
            r_nat_label_cnt[j] += len(labels[inds])

        ### Calculate Boundary Error
        # Basically the same as above, but error is sum(pred != adv_pred)/(size)
        adv_images = atk(images, labels)
        adv_outputs = model(adv_images)
        adv_pred = adv_outputs.max(1, keepdim=True)[1]
        r_bnd += pred.ne(adv_pred.view_as(pred)).sum().item()/len(adv_pred)
        for j in range(len(phi[1])):
            inds = torch.where(labels == j)
            ind_preds = pred[inds]
            ind_adv = adv_pred[inds]
            r_bnd_i[j] += ind_preds.ne(ind_adv.view_as(ind_preds)).sum().item()
            r_bnd_label_cnt[j] += len(labels[inds])
    
    # Calculate average natural/boundary error for each class
    for i in range(len(phi[0])):
        r_nat_i[i] /= r_nat_label_cnt[i]
        r_bnd_i[i] /= r_bnd_label_cnt[i]
    
    # Calculate average natural/boundary error for all classes
    r_nat /= count
    r_bnd /= count

    # Update Phi Values
    phi[0] += a1*(r_nat_i-r_nat-t1)
    phi[1] += a2*(r_bnd_i-r_bnd-t2)
    return phi
    
# FRL Reweight Algorithm
def reweight_for_robust_fairness(model,
                                 atk,
                                 num_epochs, 
                                 learning_rate, 
                                 device, 
                                 criterion,
                                 optimizer, 
                                 train_loader,
                                 val_loader,
                                 phi,
                                 a1, 
                                 a2,
                                 t1,
                                 t2
                                 ):
    # For updating learning rate
    def update_lr(optimizer, lr):    
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Send model to 
    if torch.cuda.is_available():
        model.cuda()
    
    # Instantiate softmax object for transforming phi values into weights for loss 
    sm = torch.nn.Softmax()
    
    # Instantiate optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # FRL Rewight Process
    total_step = len(train_loader)
    curr_lr = learning_rate
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            # Generate Adversarial Images
            adv_images = atk(images, labels)
            # Create natural outputs
            outputs = model(images)
            # Create predictions from natural model outputs
            preds = outputs.max(1, keepdim=True)[1][:,0]
            # Create adversarial model outputs
            adv_outputs =  model(adv_images)
            # Generate phi values based on current model
            phi = update_phis(model = model, 
                              atk = atk,
                              phi = torch.stack([torch.ones(10),torch.ones(10)]).cuda(),
                              a1 = a1,
                              a2 = a2,
                              t1 = t2,
                              t2 = t2,
                              val_loader = val_loader, 
                              device = device)
            # Softmax phi values to turn them into weights for loss
            nat_phi = sm(phi[0])
            bnd_phi = sm(phi[1])
            # Generate loss 1 (loss of natural model outputs w.r.t. natural labels)
            criterion1 = nn.CrossEntropyLoss(weight=nat_phi)
            loss1 = criterion1(outputs, labels)
            # Generate loss 2 (loss of adversarial model outputs w.r.t. natural model predictions)
            criterion2 = nn.CrossEntropyLoss(weight=bnd_phi)
            loss2 = criterion2(adv_outputs, preds)
            # Combine losses for optimization
            loss = sum([loss1, loss2])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        # Decay learning rate
        if (epoch+1) % 4 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)
    
    return model

# Helper function for updating epsilon for FRL Remargin Algorithm 
# This process is much the same as update_phis, but it also calculates the epsilon values at the end 
# after calculating the natural and boundary errors
def update_epislon(model, 
                atk,
                phi, 
                eps,
                a1,
                a2, 
                t1,
                t2,
                val_loader,
                device, 
                early_stop=20m
                ):
    #instantiate label_wise natural and boundary error
    r_nat_i = torch.zeros(10).to(device)
    r_nat_label_cnt = torch.zeros(10).to(device)
    r_bnd_i = torch.zeros(10).to(device)
    r_bnd_label_cnt = torch.zeros(10).to(device)
    r_nat = 0
    r_bnd = 0

    # add count variable to be able to average phi values
    count = 0
    for i, (images, labels) in enumerate(val_loader):
        count +=1
        if i>early_stop:
            break

        # Send images & labels to device
        images = images.to(device)
        labels = labels.to(device)

        # Calculate Natural Error
        outputs = model(images)
        pred = outputs.max(1, keepdim=True)[1]
        r_nat += pred.ne(labels.view_as(pred)).sum().item()/len(labels)
        for j in range(len(phi[0])):
            inds = torch.where(labels == j)
            ind_preds = pred[inds]
            r_nat_i[j] += ind_preds.ne(labels[inds].view_as(ind_preds)).sum().item()
            r_nat_label_cnt[j] += len(labels[inds])

        # Calculate Boundary Error
        adv_images = atk(images, labels)
        adv_outputs = model(adv_images)
        adv_pred = adv_outputs.max(1, keepdim=True)[1]
        r_bnd += pred.ne(adv_pred.view_as(pred)).sum().item()/len(adv_pred)
        for j in range(len(phi[1])):
            inds = torch.where(labels == j)
            ind_preds = pred[inds]
            ind_adv = adv_pred[inds]
            r_bnd_i[j] += ind_preds.ne(ind_adv.view_as(ind_preds)).sum().item()
            r_bnd_label_cnt[j] += len(labels[inds])

    for i in range(len(phi[0])):
        r_nat_i[i] /= r_nat_label_cnt[i]
        r_bnd_i[i] /= r_bnd_label_cnt[i]
    
    r_nat /= count
    r_bnd /= count
    eps *= torch.exp(a2*(r_bnd_i-t2))

    phi += a1*(r_nat_i-r_nat-t1)
    return eps, phi
    
# FRL Remargin Algorithm
# Again, this is much the same as the reweight_for_robust_fairness algorithm, 
# but it creates class-wise adversarial images based on the epsilon balls calculated in the 
# update_epsilon process
def remargin_for_robust_fairness(model,
                                 atk,
                                 num_epochs, 
                                 learning_rate, 
                                 device, 
                                 criterion,
                                 optimizer, 
                                 train_loader,
                                 val_loader,
                                 phi,
                                 eps,
                                 a1, 
                                 a2,
                                 a3, 
                                 a4,
                                 t1,
                                 t2
                                 ):
    # For updating learning rate
    def update_lr(optimizer, lr):    
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    if torch.cuda.is_available():
        model.cuda()
    sm = torch.nn.Softmax()

    # Train the model
    total_step = len(train_loader)
    curr_lr = learning_rate
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # if i > 3:
            #     return model

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            adv_images = atk(images, labels)
            outputs = model(images)
            preds = outputs.max(1, keepdim=True)[1][:,0]
            adv_outputs =  model(adv_images)
            
            eps, phi = update_epislon(model = model, 
                                    atk = atk,
                                    phi = torch.stack([torch.ones(10),torch.ones(10)]).cuda(),
                                    eps = torch.zeros(10).to(device) + 8/255,
                                    a1 = a1,
                                    a2 = a2,
                                    t1 = t2,
                                    t2 = t2,
                                    val_loader = val_loader, 
                                    device = device, 
                                    early_stop=20)
            
            ## Create adv images with new epsilons
            eps_atks = []
            for j in range(len(eps)):
                inds = torch.where(labels == j)
                atk_j = torchattacks.PGD(model, eps=eps[j], alpha=1/255, steps=20, random_start=True)
                eps_atks.append(atk_j(images[inds], labels[inds]))
            adv_eps_images = torch.cat(eps_atks)
            indices = torch.randperm(len(adv_eps_images))
            adv_eps_images = adv_eps_images[indices]
            outputs_eps = model(adv_eps_images)

            nat_phi = sm(phi[0])


            criterion1 = nn.CrossEntropyLoss(weight=nat_phi)
            loss1 = criterion1(outputs, labels)
            criterion2 = nn.CrossEntropyLoss()
            loss2 = criterion2(outputs_eps, labels)
            
            loss = sum([a3*loss1, a4*loss2])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        # Decay learning rate
        if (epoch+1) % 4 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)
    
    return model

