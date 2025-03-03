import torch
import torch.nn as nn

class SimpleConvNet(nn.Module):
    '''
    Simple convolutional neural network for image classification.
    Compatible with LfF pipeline.
    '''
    def __init__(self, num_classes=10, kernel_size=7, padding=3, feature_pos='logits', use_pattern_norm=False):
        '''
        Model initializer.
        ------------------------------------------------------------------------
        num_classes: Number of output classes (default = 10, for digits)
        kernel_size: Size of convolution filters (default = 7)
        padding: Padding step (default = 3)
        feature_pos: Indicates where to extract features ('pre', 'post', 'logits')
        use_pattern_norm: Indicates whether to use pattern normalization (default False)
        ------------------------------------------------------------------------
        '''
        super(SimpleConvNet, self).__init__()

        layers = [
            nn.Conv2d(3, 16, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            nn.Tanh(),
        ]
        self.extracter = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if use_pattern_norm:
            self.avgpool = nn.Sequential(self.avgpool, self.pattern_norm())
        self.fc = nn.Linear(128, num_classes)
        self.feature_pos = feature_pos

        if feature_pos not in ['pre', 'post', 'logits']:
            raise ValueError(f"Invalid feature_pos: {feature_pos}")

    def pattern_norm(self):
        ''' Dummy pattern normalization function placeholder for LfF compatibility '''
        return nn.Identity()

    def forward(self, x, logits_only=True):
        pre_gap_feats = self.extracter(x)
        post_gap_feats = self.avgpool(pre_gap_feats)
        post_gap_feats = torch.flatten(post_gap_feats, 1)
        logits = self.fc(post_gap_feats)

        if logits_only:
            return logits

        if self.feature_pos == 'pre':
            return logits, pre_gap_feats
        elif self.feature_pos == 'post':
            return logits, post_gap_feats
        return logits

    def get_feature_dim(self):
        ''' Returns feature dimension for LfF compatibility '''
        return 128


def simple_convnet(num_classes=10, use_pattern_norm=False):
    '''
    Create a SimpleConvNet instance with default parameters and optional pattern normalization.
    Compatible with LfF pipeline.
    '''
    model = SimpleConvNet(num_classes=num_classes, use_pattern_norm=use_pattern_norm)
    if use_pattern_norm:
        model.avgpool = nn.Sequential(
            model.avgpool,
            model.pattern_norm()
        )
    return model