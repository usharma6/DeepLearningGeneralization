import os
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms

def AugmentedImageNet(train_transforms=[], val_transforms=[], test_transforms=[], batch_size=100):
  
    data_transforms = {
        'train': transforms.Compose(
            train_transforms
        ),
        'val': transforms.Compose(
            val_transforms
        ),
        'test': transforms.Compose(
            test_transforms
        )
    }

    data_dir = 'tiny-imagenet-200/'
    num_workers = {
        'train' : 100,
        'val'   : 0,
        'test'  : 0
    }
    image_datasets = {x: dataset.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val','test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=100,
                                                 shuffle=True, pin_memory =True)
                  for x in ['train', 'val', 'test']}

    return dataloaders

DATA_DICT = {
    "baseline" : AugmentedImageNet(train_transforms=[   
      transforms.ToTensor(),
      ], 
    val_transforms=[
      transforms.ToTensor(),
    ],
    test_transforms=[]),

    "Evaluation_Blur" : AugmentedImageNet(train_transforms=[   
      transforms.ToTensor(),
      transforms.GaussianBlur(3, sigma=1.0)
      ], 
    val_transforms=[
      transforms.ToTensor(),
      transforms.GaussianBlur(3, sigma=1.0)
    ],
    test_transforms=[]),

    "Evaluation_Rotation" : AugmentedImageNet(train_transforms=[   
      transforms.ToTensor(),
      transforms.RandomRotation(10),
      ],  
    val_transforms=[
      transforms.ToTensor(),
      transforms.RandomRotation(10),
    ],
    test_transforms=[]),

    "Evaluation_Jitter" : AugmentedImageNet(train_transforms=[   
      transforms.ColorJitter(),   
      transforms.ToTensor(),
      ],  
    val_transforms=[
      transforms.ColorJitter(),   
      transforms.ToTensor(),
    ],
    test_transforms=[]), 

    "Evaluation_Grayscale" : AugmentedImageNet(train_transforms=[   
      transforms.ToTensor(),
      transforms.RandomGrayscale(1),
      ],  
    val_transforms=[
      transforms.ToTensor(),
      transforms.RandomGrayscale(1), 
    ],
    test_transforms=[]),

    "Evaluation_Erasing" : AugmentedImageNet(train_transforms=[   
      transforms.ToTensor(),
      transforms.RandomErasing(p=1), 
      ], 
    val_transforms=[
      transforms.ToTensor(),
      transforms.RandomErasing(p=1), 
    ],
    test_transforms=[]), 
  
    "Blur" : AugmentedImageNet(train_transforms=[  
      transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.1),   
      transforms.ToTensor(),                                                                             
      transforms.RandomGrayscale(0.1),
      transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation(10)]), p=0.1),
      transforms.RandomErasing(p=0.1)], 
    val_transforms=[
      transforms.ToTensor(),
      transforms.GaussianBlur(3, sigma=1.0)
    ],
    test_transforms=[]),

    "Rotation" : AugmentedImageNet(train_transforms=[  
      transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.1),   
      transforms.ToTensor(),                                                                             
      transforms.RandomGrayscale(0.1),
      transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(3, sigma=1.0)]), p=0.1),
      transforms.RandomErasing(p=0.1)], 
    val_transforms=[
      transforms.ToTensor(),
      transforms.RandomRotation(10),
    ],
    test_transforms=[]),

    "Jitter" : AugmentedImageNet(train_transforms=[  
      transforms.ToTensor(),
      transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation(10)]), p=0.1),                                                                             
      transforms.RandomGrayscale(0.1),
      transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(3, sigma=1.0)]), p=0.1),
      transforms.RandomErasing(p=0.1)], 
    val_transforms=[
      transforms.ColorJitter(),   
      transforms.ToTensor(),
    ],
    test_transforms=[]), 

    "Grayscale" : AugmentedImageNet(train_transforms=[  
      transforms.ToTensor(),
      transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation(10)]), p=0.1),                                                                             
      transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.1),
      transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(3, sigma=1.0)]), p=0.1),
      transforms.RandomErasing(p=0.1)], 
    val_transforms=[
      transforms.ToTensor(),
      transforms.RandomGrayscale(1), 
    ],
    test_transforms=[]),

    "Erasing" : AugmentedImageNet(train_transforms=[  
      transforms.ToTensor(),
      transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation(10)]), p=0.1),                                                                             
      transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.1),
      transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(3, sigma=1.0)]), p=0.1),
      transforms.RandomGrayscale(0.1)],
    val_transforms=[
      transforms.ToTensor(),
      transforms.RandomErasing(p=1), 
    ],
    test_transforms=[]), 
}