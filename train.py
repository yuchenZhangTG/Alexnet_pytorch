from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models.alexnet import AlexNet

from PIL import Image
import os,sys,errno
import argparse
import time
import torch.utils.model_zoo as model_zoo
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root='./', mode='train', transform=None,download=False):
        self.root=os.path.expanduser(root)
        self.transform = transform
        self.mode=mode
        self.url='http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        self.dtiny=os.path.join(self.root,'tiny-imagenet-200')
        if download:
            self.download()
        
        ids=[]
        self.words=[]
        self.imagedirs=[]
        with open(os.path.join(self.dtiny,'wnids.txt'), 'r') as fnids:
            for line in fnids:
                linsp=line.split()
                ids.append(linsp[0])
            
        with open(os.path.join(self.dtiny,'words.txt'), 'r') as fwords:
            for line in fwords: 
                linsp=line.split()
                id=linsp[0]
                word=linsp[1]
                if id in ids:
                    self.words.append(word)
        
        self.folder=os.path.join(self.dtiny,self.mode)
        tids=os.listdir(self.folder)
        if self.mode=='train':
            for fid in tids:
                imfs=os.listdir(os.path.join(self.folder,fid,'images'))
                for imf in imfs:
                    imdir=os.path.join(fid,'images',imf)
                    with Image.open(os.path.join(self.folder,imdir)) as img:
                        if img.mode=='RGB':
                            self.imagedirs.append((imdir,ids.index(fid)))
        elif self.mode=='val':
             with open(os.path.join(self.folder,'val_annotations.txt')) as anno:
                 for line in anno:
                     linsp=line.split()
                     image=linsp[0]
                     id=linsp[1]
                     imdir=os.path.join('images',image)
                     with Image.open(os.path.join(self.folder,imdir)) as img:
                         if img.mode=='RGB':
                             self.imagedirs.append((imdir,ids.index(id)))
        elif self.mode=='test':
            imfs=os.listdir(os.path.join(self.folder,'images'))
            for imf in imfs:
                imdir=os.path.join('images',imf)
                with Image.open(os.path.join(self.folder,imdir)) as img:
                    if img.mode=='RGB':
                        self.imagedirs.append(imdir)
            
                          
    def __getitem__(self, index):
        if self.mode=='test':
            imdir=self.imagedirs[index]
            target=None
        else:
            imdir, target =self.imagedirs[index]
            target=torch.tensor(target)
        
        img = Image.open(os.path.join(self.folder,imdir))
        if self.transform is not None:
            img = self.transform(img)
        
        if self.mode=='test':
            return img
        else:
            return (img, target)
    
    def __len__(self):
        return len(self.imagedirs)
    
    def download(self):
        from six.moves import urllib
        import gzip
        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        print('Downloading ' + self.url)
        data = urllib.request.urlopen(self.url)
        filename = self.url.rpartition('/')[2]
        file_path = os.path.join(self.root, filename)
        with open(file_path, 'wb') as f:
            f.write(data.read())
        with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                gzip.GzipFile(file_path) as zip_f:
            out_f.write(zip_f.read())
        os.unlink(file_path)
        
def ensure_dir(d):
    directory=os.path.abspath(d)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
class myAlexNet(AlexNet):
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def myalexnet(pretrained=False, **kwargs):
    model = myAlexNet(**kwargs)
    model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    }
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model


def train_model(args,model, device,dataloader):
    alexnet=model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(alexnet.classifier[6].parameters(), lr=args.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.05)
    state_dir=os.path.join(os.path.expanduser(args.save),'state.bth')
    
    start = time.time()
    torch.save(model.state_dict(), state_dir)
    #best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_plot=range(args.epochs)
    loss_plot={'train':[], 'val':[]}
    acc_plot={'train':[], 'val':[]}
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 5)
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            running_loss=0
            running_corrects=0
            total=0
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)                
                labels = labels.to(device)
                total+=labels.data.shape[0]
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / total
            epoch_acc = running_corrects.double() / total
            elapsed = time.time() - start
            loss_plot[phase].append(epoch_loss)
            acc_plot[phase].append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f} time: {:.1f}'
                  .format(phase, epoch_loss, epoch_acc, elapsed))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                #best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), state_dir)
        print()
    
    print('Training complete in {:.0f}m {:.0f}s'.format(
        elapsed // 60, elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    #model.load_state_dict(best_model_wts)
    model.load_state_dict(torch.load(state_dir))
    
    elapsed = time.time() - start
    plt.figure(1)
    plt.rcParams.update({'font.size': 18})
    plt.hold(True)
    for phase in ['train', 'val']:
        plt.plot(epochs_plot,loss_plot[phase],label=phase) 
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png',bbox_inches='tight')
    plt.figure(2)
    plt.hold(True)
    for phase in ['train', 'val']:
        plt.plot(epochs_plot,[k*100 for k in acc_plot[phase]],label=phase) 
    plt.xlabel('Epochs')
    plt.ylabel('Accuracty')
    plt.legend()
    plt.savefig('acc.png',bbox_inches='tight')
    
    
    return model

def visualize_model(args, model,device,dataloader,class_names,num_images=4):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(3)
    state_dir=os.path.join(os.path.expanduser(args.save),'state.bth')
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            img=inputs.cpu()
            for j in range(img.shape[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                inp=img[j,:,:,:].view(3,64,64).numpy().transpose((1,2,0))
                inp=inp*0.5+0.5
                plt.imshow(inp)
                plt.savefig(os.path.join(args.save,'train_image.png'),bbox_inches='tight')

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def main():    
    

    parser = argparse.ArgumentParser(description='PyTorch Tiny ImageNet Training')
    
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of cup and data loading workers (default: 0)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run (default:10)')
    parser.add_argument('-d','--data', default='./', type=str, metavar='S',
                        help='directory of input data (default:./)')
    parser.add_argument('-s', '--save', default='./model', type=str, metavar='S',
                        help='directory of the saved model and weights (default:./model)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='M',
                        help='learning rate (default: 0.01)')
    parser.add_argument('-bs','--batch-size', type=int, default=4, metavar='N',
                        help='batch size (default: 4)')
    args = parser.parse_args()
    ensure_dir(args.data)
    ensure_dir(args.save)
    
    torch.set_num_threads(args.workers)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    transform= transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
           
    dataset={x: TinyImageNetDataset(root=args.data,mode=x,transform=transform)
        for x in ['train', 'val']}
    class_names=dataset['train'].words
    dataloader={x: torch.utils.data.DataLoader(dataset[x],batch_size=args.batch_size,shuffle=True,num_workers=args.workers)
        for x in ['train', 'val']}    
    
    
    alexnet= myalexnet(pretrained=True)            
    for param in alexnet.parameters():
        param.requires_grad = False
    
    alexnet.classifier=nn.Sequential(
                nn.Dropout(),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 200),
            )         
    alexnet = train_model(args,alexnet,device,dataloader)
    
    visualize_model(args, alexnet,device,dataloader,class_names)
    
    
if __name__== "__main__":
    main()
