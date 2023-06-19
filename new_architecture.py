from util import *
from voc import *
from model import *
from train import *
import torchvision.transforms.functional as TF

if __name__ == "__main__":
    device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'epochs'    : 20,
        'bz'        : 8,
        'lr'        : 1e-3,
        'device'    : device,
        'early_stop': 5,
        'remark'    : 'NewArchitecture'
    }

    ''' Transformations '''
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])
    target_transform = MaskToTensor()
    TF_transform = lambda x: [x, TF.hflip(x), TF.rotate(x.unsqueeze(0), angle = 5, fill = 0).squeeze(0), TF.rotate(x.unsqueeze(0), angle = 5, fill = 0).squeeze(0), *TF.five_crop(x, (224, 224))]

    ''' Data loaders '''
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        input_transform, target_transform, TF_transform, config['device'], config['bz'], collate_fn=collate_fn)

    ''' Prepare model '''
    new_model = New_arch(n_class=21)
    #new_model.apply(init_weights)
    new_model = new_model.to(device)

    optimizer = torch.optim.Adam(new_model.parameters(), config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'], eta_min = 0.0001)

    class_weight = getClassWeights(input_transform, target_transform, TF_transform)
    criterion =  torch.nn.CrossEntropyLoss(weight = class_weight)

    ''' Train model '''
    best_iou_score, best_accuracy, min_validation_loss, \
    training_loss_history, validation_loss_history, early_stop_epoch \
        = train(new_model, train_loader, val_loader, criterion, scheduler, optimizer, config)

    ''' Test model '''
    test_loss, iou, acc = modelTest(new_model, test_loader, criterion, early_stop_epoch, config['remark'])
    print(f'Test IOU: {round(iou, 3)}. Test acc: {round(acc, 3)}. Test loss: {round(test_loss, 3)}')

#     ''' Visualize some test sample '''
#     imgs, masks_gt = next(iter(train_loader))
#     imgs, masks_gt = imgs[:8], masks_gt[:8]
#     masks_gt = F.one_hot(masks_gt.to(torch.int64), num_classes=21).permute(0, 3, 1, 2).to(torch.float64)
#     masks_pred = new_model(imgs)
#     masks_pred = torch.argmax(masks_pred, dim=1).cpu().numpy()
#     masks_gt = torch.argmax(masks_gt, dim=1).cpu().numpy()
#     plot_images(np.concatenate([masks_gt,masks_pred]), voc_pallet(), cols=8)

#     imgs, masks_gt = next(iter(test_loader))
#     imgs, masks_gt = imgs[:8], masks_gt[:8]
#     masks_gt = F.one_hot(masks_gt.to(torch.int64), num_classes=21).permute(0, 3, 1, 2).to(torch.float64)
#     masks_pred = new_model(imgs)
#     masks_pred = torch.argmax(masks_pred, dim=1).cpu().numpy()
#     masks_gt = torch.argmax(masks_gt, dim=1).cpu().numpy()
#     plot_images(np.concatenate([masks_gt,masks_pred]), voc_pallet(), cols=8)
