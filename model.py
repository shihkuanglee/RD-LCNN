import torch
import torch.nn.functional as F


def init_weights(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")


class Max_Feature_Map(torch.nn.Module):

    def __init__(self):
        super(Max_Feature_Map, self).__init__()

    def forward(self, input, dim=1):
        inputs = torch.split(input, input.size(dim) // 2, dim=dim)
        return torch.max(inputs[0], inputs[1])


class T45_LCNN(torch.nn.Module):

    def __init__(self, data_shape=[863, 600], LDO_p1=0.75, LDO_p2=0.00):
        super(T45_LCNN, self).__init__()

        data_shape_final = [
            data_shape[0] // (2 ** 4), 
            data_shape[1] // (2 ** 4),
            32
        ]

        data_points = torch.tensor(data_shape_final).prod()

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, (5, 5), stride=(1, 1), padding=(2, 2)),
            Max_Feature_Map(),


            torch.nn.MaxPool2d((2, 2), stride=(2, 2)),


            torch.nn.Conv2d(32, 64, (1, 1), stride=(1, 1), padding=(0, 0)),
            Max_Feature_Map(),
            torch.nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.Conv2d(32,  96, (3, 3), stride=(1, 1), padding=(1, 1)),
            Max_Feature_Map(),


            torch.nn.MaxPool2d((2, 2), stride=(2, 2)),


            torch.nn.BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.Conv2d(48, 96, (1, 1), stride=(1, 1), padding=(0, 0)),
            Max_Feature_Map(),
            torch.nn.BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.Conv2d(48, 128, (3, 3), stride=(1, 1), padding=(1, 1)),
            Max_Feature_Map(),


            torch.nn.MaxPool2d((2, 2), stride=(2, 2)),


            torch.nn.Conv2d(64, 128, (1, 1), stride=(1, 1), padding=(0, 0)),
            Max_Feature_Map(),
            torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.Conv2d(64,  64, (3, 3), stride=(1, 1), padding=(1, 1)),
            Max_Feature_Map(),
            torch.nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.Conv2d(32,  64, (1, 1), stride=(1, 1), padding=(0, 0)),
            Max_Feature_Map(),
            torch.nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.Conv2d(32,  64, (3, 3), stride=(1, 1), padding=(1, 1)),
            Max_Feature_Map(),


            torch.nn.MaxPool2d((2, 2), stride=(2, 2)),


            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Dropout(p=LDO_p1),
            torch.nn.Linear(data_points, 160),
            Max_Feature_Map(),

            
            torch.nn.BatchNorm1d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.Dropout(p=LDO_p2),
            torch.nn.Linear(80, 2)
            )

        self.apply(init_weights)

    def forward(self, input):
        return self.net(input)

