from MixFormer import *

def get_model(name, num_class):
    if name == 'mit_b0':
        model = MixFormerReID(mit_b0(drop_rate=0.2), 256, num_class, 'mit_b0.pth')
        dim_feature = 256
    elif name == 'mit_b1':
        model = MixFormerReID(mit_b1(drop_rate=0.2), 512, num_class, 'mit_b1.pth')
        dim_feature = 512
    elif name == 'mit_b2':
        model = MixFormerReID(mit_b2(drop_rate=0.2), 512, num_class, 'mit_b2.pth')
        dim_feature = 512
    elif name == 'mit_b3':
        model = MixFormerReID(mit_b3(drop_rate=0.2), 512, num_class, 'mit_b3.pth')
        dim_feature = 512
    elif name == 'mit_b4':
        model = MixFormerReID(mit_b4(drop_rate=0.2), 512, num_class, 'mit_b4.pth')
        dim_feature = 512
    elif name == 'mit_b5':
        model = MixFormerReID(mit_b5(drop_rate=0.2), 512, num_class, 'mit_b5.pth')
        dim_feature = 512

    ##You can implment your own model here.

    return model, dim_feature
