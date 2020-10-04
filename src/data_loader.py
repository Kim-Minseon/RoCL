import torch
from data.cifar import CIFAR10, CIFAR100
from torchvision import transforms

def get_dataset(args):
 
    ### color augmentation ###    
    color_jitter = transforms.ColorJitter(0.8*args.color_jitter_strength, 0.8*args.color_jitter_strength, 0.8*args.color_jitter_strength, 0.2*args.color_jitter_strength)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
   
    learning_type = args.train_type
    
    if args.dataset == 'cifar-10':

        if learning_type =='contrastive':
            transform_train = transforms.Compose([
                rnd_color_jitter,
                rnd_gray,
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(32),
                transforms.ToTensor(),
            ])

            transform_test = transform_train

        elif learning_type=='linear_eval':
            transform_train = transforms.Compose([
                rnd_color_jitter,
                rnd_gray,
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(32),
                transforms.ToTensor(),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])

        elif learning_type=='test':
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(32),
                transforms.ToTensor(),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            assert('wrong learning type')
        
        train_dst   = CIFAR10(root='./Data', train=True, download=True,
                                        transform=transform_train,contrastive_learning=learning_type)
        val_dst     = CIFAR10(root='./Data', train=False, download=True,
                                       transform=transform_test,contrastive_learning=learning_type)

        if learning_type=='contrastive':
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dst,
                num_replicas=args.ngpu,
                rank=args.local_rank,
                )
            train_loader = torch.utils.data.DataLoader(train_dst,batch_size=args.batch_size,num_workers=4,
                    pin_memory=False,
                    shuffle=(train_sampler is None),
                    sampler=train_sampler,
                )

            val_loader = torch.utils.data.DataLoader(val_dst,batch_size=100,
                    num_workers=4,
                    pin_memory=False,
                    shuffle=False,
                )
            
            return train_loader, train_dst, val_loader, val_dst, train_sampler
        else:
            train_loader = torch.utils.data.DataLoader(train_dst,
                                                  batch_size=args.batch_size,
                                                  shuffle=True, num_workers=4)
            val_batch = 100
            val_loader = torch.utils.data.DataLoader(val_dst, batch_size=val_batch,
                                                 shuffle=False, num_workers=4)

            return train_loader, train_dst, val_loader, val_dst

    if args.dataset == 'cifar-100':

        if learning_type=='contrastive':
            transform_train = transforms.Compose([
                rnd_color_jitter,
                rnd_gray,
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(32),
                transforms.ToTensor()
            ])

            transform_test = transform_train

        elif learning_type=='linear_eval':
            transform_train = transforms.Compose([
                rnd_color_jitter,
                rnd_gray,
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(32),
                transforms.ToTensor()
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor()
            ])

        elif learning_type=='test':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            assert('wrong learning type')
    
        train_dst   = CIFAR100(root='./Data', train=True, download=True,
                                        transform=transform_train,contrastive_learning=learning_type)
        val_dst     = CIFAR100(root='./Data', train=False, download=True,
                                       transform=transform_test,contrastive_learning=learning_type)

        if learning_type=='contrastive':
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dst,
                num_replicas=args.ngpu,
                rank=args.local_rank,
                )
            train_loader = torch.utils.data.DataLoader(train_dst,batch_size=args.batch_size,num_workers=4,
                    pin_memory=True,
                    shuffle=(train_sampler is None),
                    sampler=train_sampler,
                )

            val_loader = torch.utils.data.DataLoader(val_dst,batch_size=100,num_workers=4,
                    pin_memory=True,
                )
            return train_loader, train_dst, val_loader, val_dst, train_sampler

        else:
            train_loader = torch.utils.data.DataLoader(train_dst,
                                                  batch_size=args.batch_size,
                                                  shuffle=True, num_workers=4)

            val_loader = torch.utils.data.DataLoader(val_dst, batch_size=100,
                                                 shuffle=False, num_workers=4)

            return train_loader, train_dst, val_loader, val_dst

