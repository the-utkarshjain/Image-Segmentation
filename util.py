import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

''' Evaluation Metrics '''

def compute_iou(pred, target):
    '''
    Input:
        - pred, target: np array (H, W) predicted mask and tg mask
    Output:
        - IoU, average over mask for each class (including background)
    ref: https://stackoverflow.com/a/48383182
    '''
    ious = []
    for cls_id in torch.unique(target):
        pred_mask = (pred == cls_id)
        target_mask = (target == cls_id)
        intersection = target_mask[pred_mask].sum()
        union = pred_mask.sum() + target_mask.sum() - intersection
        ious.append(float(intersection) / float(union)) if union !=0 else ious.append(float('nan'))

    return torch.mean(torch.tensor(ious))

def compute_pixel_acc(pred, target):
    matched = torch.eq(pred, target)
    return torch.sum(matched) / torch.numel(matched)


''' Image and Visualization '''

def plot_rgb_img(img_arr, path='foo.png'):
    plt.imshow(img_arr)

def color_palette(num_objects=21):
    ''' Return 82-way colored palette for visualization
        https://matplotlib.org/stable/tutorials/colors/colormaps.html
    Output:
        - COLOR_PALETTE: np.array (num_objects, 3)
    '''
    from matplotlib.cm import get_cmap
    cmap = get_cmap('rainbow', num_objects)
    COLOR_PALETTE = np.array([cmap(i)[:3] for i in range(num_objects + 3)])  # (82, 3)
    COLOR_PALETTE = np.array(COLOR_PALETTE * 255, dtype=np.uint8)

    np.random.seed(10)
    np.random.shuffle(COLOR_PALETTE)
    COLOR_PALETTE[-1] = [255, 255, 225]  

    return COLOR_PALETTE

def voc_pallet():
    pallet = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]  
    return np.array(pallet).reshape(-1,3)

def plot_images(imgs, pallet=None, annotations=None, cols=5, path='./imgs/foo.png', title=None):
    '''
    Input:
        - imgs: numpy array of shape (n, H,W,3), representing n images to plot
        - cols: number of columns for subplots
        - annotations: A list of annotations containing titles for each img subplot
    '''
    if annotations is None:
        annotations = [None] * imgs.shape[0]

    rows = imgs.shape[0] // cols if imgs.shape[0] % cols == 0 else imgs.shape[0] // cols+1
    fig, axs = plt.subplots(rows,cols)
    if title:
        fig.suptitle(title, fontsize=16)
    
    for idx, (img, ann) in enumerate(zip(imgs, annotations)):
        j, i = idx % cols, idx // cols
        ax = axs[j] if rows == 1 else axs[i,j]
        
        if imgs.shape[-1] ==3: 
            ax.imshow(img)
        else:
            assert pallet is not None, "Need color pallet to visualize segmentation mask"
            ax.imshow(pallet[np.squeeze(img)])
        
        ax.set_title(ann)
        ax.axis('off')
        
    fig.set_figheight(3*rows)
    fig.set_figwidth(3*cols)

    plt.subplots_adjust(left=0.05, right=0.95,
                      bottom=0.1, top=0.9,
                      wspace=0.4, hspace=0.3)

    if path is not None: plt.savefig(path)



''' Training statistics '''

def plot_stats(stats, path=None):
    plt.plot(stats); plt.show()
    if path:
        plt.savefig(path)

def plot_loss_acc(loss_trn, acc_trn, loss_val, acc_val, label_axes=False, path=None):
    fig, ax = plt.subplots(1,2, figsize=(12,4))

    ax[0].set_title("Avg loss per sample")
    ax[1].set_title("Avg accuracy per sample")

    if label_axes:
        ax[0].set_ylabel('Loss (Cross Entropy)')
        ax[0].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].set_xlabel('Epoch')

    ax[0].plot(loss_trn, label="training")
    ax[0].plot(loss_val, label="validation")
    ax[1].plot(acc_trn, label="training")
    ax[1].plot(acc_val, label="validation")
    ax[0].legend()
    ax[1].legend()

    fig.set_figheight(3)
    fig.set_figwidth(8)
    plt.show()
    plt.legend(loc="upper left")
    if path is not None: plt.savefig(path)
    plt.clf()



''' Time stamp and model saving '''

def get_time():
    return datetime.now().strftime("%m-%d_%H:%M")

def save_model(model, remark=""):
    torch.save(model.state_dict(), f'./trained_model/{remark}.pt')
    #torch.save(model.state_dict(), f'./trained_model/{remark}_{get_time()}.pt')

def load_model(model, path=""):
    model.load_state_dict(torch.load(path))
    return model
        