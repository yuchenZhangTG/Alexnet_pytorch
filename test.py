import torch
from torch import nn
from torchvision import transforms
import os
import argparse
from train import TinyImageNetDataset, myalexnet,ensure_dir
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='PyTorch Tiny ImageNet testing')
    parser.add_argument('-d','--data', default='./', type=str, metavar='S',
                        help='directory of input data (default:./)')
    parser.add_argument('-m', '--model', default="./model", type=str, metavar='S',
                        help='directory of the saved model and weights (default:./model)')
    parser.add_argument('-o','--output', default="./output", type=str, metavar='S',
                        help='directory of output images (default:./output)')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of cup and data loading workers (default: 0)')
    parser.add_argument('-bs','--batch-size', type=int, default=4, metavar='N',
                        help='batch size (default: 4)')
    args = parser.parse_args()
    ensure_dir(args.data)
    ensure_dir(args.model)
    ensure_dir(args.output)
    torch.set_num_threads(args.workers)
    
    state_dir=os.path.join(os.path.expanduser(args.model),'state.bth')
    transform= transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    
    testset=TinyImageNetDataset(root=args.data,mode='test',transform=transform)
    testloader=torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.workers)
    class_names=testset.words
    
    alexnet= myalexnet()
    alexnet.classifier=nn.Sequential(
                nn.Dropout(),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 200),
            )        
    alexnet.load_state_dict(torch.load(state_dir))   
    alexnet.eval()
    
    for i,inputs in enumerate(testloader):
        with torch.set_grad_enabled(False):
            outputs = alexnet(inputs)
            preds = torch.argmax(outputs, 1)
            
            images_so_far=0;
            num_images=4;
            img=inputs.cpu()
            for j in range(img.shape[0]):        
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                inp=img[j,:,:,:].view(3,64,64).numpy().transpose((1,2,0))
                inp=inp*0.5+0.5
                plt.imshow(inp)
                plt.savefig(os.path.join(args.output,'train_image{:03d}.png'.format(i)))
                
if __name__== "__main__":
    main()
            