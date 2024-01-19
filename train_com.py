import time

from ADUnet import *

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from SSIM import *
from datasets import *

parser = argparse.ArgumentParser(description="Common_train")
parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--save_path", type=str, default="latest/", help='path to save models and log files')
parser.add_argument("--save_freq", type=int, default=1, help='save intermediate model')
parser.add_argument("--data_path", type=str, default='./dataset/leftImg8bit_trainval_rain/leftImg8bit_rain/',
                    help='path to training data')
parser.add_argument("--gt_path", type=str, default='./dataset/leftImg8bit_trainvaltest/leftImg8bit/',
                    help='path to groundtruth data')
parser.add_argument("--depth_path", type=str, default='./dataset/leftImg8bit_trainval_rain/depth_rain/train',
                    help='path to groundtruth data')
parser.add_argument("--pre_train", type=str,
                    default='./ADUNet/best.pth',
                    help='Path of pth')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="1", help='GPU id')
parser.add_argument("--display_iter", type=int, default=10, help='number of recursive stages')
opt = parser.parse_args()


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return PSNR / Img.shape[0]



device = torch.device("cuda:" + str(opt.gpu_id) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
train_dataset = cityDataset(data_path=opt.data_path,
                            train='train', gt_path=opt.gt_path
                            )

train_loader = DataLoader(dataset=train_dataset,
                          num_workers=8,
                          batch_size=opt.batch_size,
                          shuffle=True)

test_dataset = cityDataset(data_path=opt.data_path,
                           train='test',
                           gt_path=opt.gt_path)

test_loader = DataLoader(dataset=test_dataset,
                         num_workers=8,
                         batch_size=opt.batch_size,
                         shuffle=True)
model = ADUNet(3, 3).cuda()
criterion = SSIM().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, verbose=True)

iter = len(train_loader)
best_ps = 0
best_ep = 0
fepoch = 0

# fepoch = torch.load(opt.pre_train)['epoch']
best_ps = torch.load(opt.pre_train)['best_avg']
optimizer.load_state_dict(torch.load(opt.pre_train)['optimizer'])
model.load_state_dict(torch.load(opt.pre_train)['net'])

if not os.path.isdir(opt.save_path):
    os.mkdir(opt.save_path)

if fepoch == 0:
    f = open(opt.save_path + 'log.txt','a+')
    f.write(f"{model}")
    f.close()

for epoch in range(fepoch, opt.epochs):
    # if epoch-best_ep>5:
    #     break
    model.train()
    sum_ps = 0
    sum_loss = 0
    start_time = time.time()
    for iteration, (img_in, img_gt) in enumerate(train_loader):
        img_in = img_in.cuda()
        img_gt = img_gt.cuda()

        optimizer.zero_grad()
        # img_out = model(img_in)
        _, _, img_out = model(img_in)
        loss_ = criterion(img_out, img_gt)
        loss = -loss_
        psnr = batch_PSNR(img_out, img_gt, 1.)
        sum_ps += psnr
        sum_loss += loss.item()
        loss.backward()
        optimizer.step()

        if ((iteration + 1) % opt.display_iter) == 0:
            print(f"[epoch {epoch}] [iteration {iteration + 1}/{iter}] SSIM:{loss.item()},PSNR:{psnr}")

    end_time = time.time()
    print(f"Epoch {epoch}: Time: {end_time-start_time}s\n")
    avg_ps = sum_ps/iter
    avg_loss = sum_loss/iter
    scheduler.step(avg_ps)

    if epoch % opt.save_freq == 0:
        model.eval()
        with torch.no_grad():
            sum_ps = 0
            sum_loss = 0
            for iter_val, (img_in, img_gt) in enumerate(test_loader):
                img_in = img_in.cuda()
                img_gt = img_gt.cuda()
                # img_out = model(img_in)
                _, _, img_out = model(img_in)
                loss = criterion(img_out, img_gt)
                psnr = batch_PSNR(img_out, img_gt, 1.)
                sum_ps += psnr
                sum_loss += loss.item()
                print(f"[epoch {epoch}] [Test {iter_val + 1}] SSIM:{loss.item()}, PSNR:{psnr}")

            avg_ps_test = sum_ps/len(test_loader)
            avg_loss_test = sum_loss/len(test_loader)
            if avg_ps_test > best_ps:
                best_ps = avg_ps_test
                best_ep = epoch
                checkpoint = {
                    "net": model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "epoch": epoch,
                    "best_avg": best_ps
                }
                if not os.path.isdir(opt.save_path):
                    os.mkdir(opt.save_path)

                torch.save(checkpoint, opt.save_path + f"best.pth")
                # torch.save(model.state_dict(), opt.save_path + f"best.pth")


            f = open(opt.save_path + 'log.txt', 'a+')
            f.write(f"[epoch {epoch}] Train SSIM:{avg_loss} PSNR:{avg_ps} Test SSIM:{avg_loss_test} PSNR:{avg_ps_test}\n")
            f.close()
