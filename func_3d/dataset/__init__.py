from .btcv import BTCV
from .amos import AMOS
from .leaderboard import Leaderboard
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



def get_dataloader(args, deploy_mode=False):
    # transform_train = transforms.Compose([
    #     transforms.Resize((args.image_size,args.image_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_train_seg = transforms.Compose([
    #     transforms.Resize((args.out_size,args.out_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.Resize((args.image_size, args.image_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_test_seg = transforms.Compose([
    #     transforms.Resize((args.out_size,args.out_size)),
    #     transforms.ToTensor(),
    # ])

    if deploy_mode:
        assert args.dataset == 'leaderboard'
        leaderboard_deploy_dataset = Leaderboard(args, args.data_path, transform = None, transform_msk= None, mode = 'Deploy', prompt=args.prompt)
        nice_deploy_loader = DataLoader(leaderboard_deploy_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        return nice_deploy_loader
    
    if args.dataset == 'btcv':
        '''btcv data'''
        btcv_train_dataset = BTCV(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
        btcv_test_dataset = BTCV(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt)

        nice_train_loader = DataLoader(btcv_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(btcv_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif args.dataset == 'amos':
        '''amos data'''
        amos_train_dataset = AMOS(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
        amos_test_dataset = AMOS(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt)

        nice_train_loader = DataLoader(amos_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(amos_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif args.dataset == 'leaderboard':
        leaderboard_train_dataset = Leaderboard(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt, variation=0.2)
        leaderboard_test_dataset = Leaderboard(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt, variation=0.2)

        nice_train_loader = DataLoader(leaderboard_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(leaderboard_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    else:
        print("the dataset is not supported now!!!")
        
    return nice_train_loader, nice_test_loader