import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()

        # Load the pretrained VGG-16 and use its features
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())

        # Encoder
        self.features_block1 = nn.Sequential(*features[:5])  # First pooling
        self.features_block2 = nn.Sequential(*features[5:10])  # Second pooling
        self.features_block3 = nn.Sequential(*features[10:17])  # Third pooling
        self.features_block4 = nn.Sequential(*features[17:24])  # Fourth pooling
        self.features_block5 = nn.Sequential(*features[24:])  # Fifth pooling

        # Modify the classifier part of VGG-16
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, kernel_size=1)
        )

        # Decoder
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_final = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)

        # Skip connections
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        _, _, iH, iW = x.size()
        # print(f'Input size: {x.size()}')  # Print input dimensions
    
        # Encoder
        x = self.features_block1(x)
        # print(f'After features_block1: {x.size()}')  # After first block
        
        x = self.features_block2(x)
        # print(f'After features_block2: {x.size()}')  # After second block
        
        x = self.features_block3(x)
        pool3_output = x  # Save for skip connection
        # print(f'After features_block3 (pool3_output): {x.size()}')  # After third block
        
        x = self.features_block4(x)
        pool4_output = x  # Save for skip connection
        # print(f'After features_block4 (pool4_output): {x.size()}')  # After fourth block
        
        x = self.features_block5(x)
        # print(f'After features_block5: {x.size()}')  # After fifth block

        # Apply classifier to the last feature map
        x = self.classifier(x)
        # print(f'After classifier: {x.size()}')  # After classifier

        # Decoder
        x = self.upscore2(x)  # First upsampling
        # print(f'After upscore2: {x.size()}')  # After first upsampling
        
        # Skip connection from pool4
        pool4_output = self.score_pool4(pool4_output)
        # print(f'After score_pool4: {pool4_output.size()}')
        _, _, H, W = pool4_output.size()
        x = x[:, :, :H, :W]
        x = x + pool4_output  # Element-wise addition
        # print(f'After skip connection from pool4: {x.size()}')
        
        x = self.upscore_pool4(x)  # Upsample again
        # print(f'After upscore_pool4: {x.size()}')  # After second upsampling

        # Skip connection from pool3
        pool3_output = self.score_pool3(pool3_output)
        _, _, H, W = pool3_output.size()
        x = x[:, :, :H, :W]
        x = x + pool3_output  # Element-wise addition
        # print(f'After skip connection from pool3: {x.size()}')
        
        # Final upsampling
        x = self.upscore_final(x)
        x = x[:, :, :iH, :iW]
        # print(f'After upscore_final: {x.size()}')  # Final size

        return x
        raise NotImplementedError("Implement the forward method")
