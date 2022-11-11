import glob
from torch.utils.data import Dataset


class WSIData(Dataset):
    def __init__(self, data_root=None):
        self.data_root = data_root
        self.data_list = []
        types = ('*.svs', '*.tif')
        for type_ in types:
            self.data_list.extend(glob.glob(self.data_root + '/**/'+type_, recursive=True))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        wsi_path = self.data_list[index]
        return wsi_path

class HovernetData(Dataset):
    def __init__(self, data_root=None):
        self.data_root = data_root
        self.data_list = glob.glob(self.data_root + "/*.json")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        hovernet_path = self.data_list[index]
        return hovernet_path
