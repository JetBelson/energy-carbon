import torch.utils.data


class BaseDataLoader():
    def __init__(self):
        pass
    
    def initialize(self, opt):
        self.opt = opt
        pass

    def load_data():
        return None


class ECDataLoader(BaseDataLoader):
    def name(self):
        return "ECDataLoader"
    
    def initialize(self, opt, data_type):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt=opt, data_type=data_type)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=True if data_type=="train" else False
        )

    def load_data(self):
        return self.dataloader
    
    def __len__(self):
        return len(self.dataset)


def CreateDataset(opt, data_type):
    """
    Return dataset according to opt and data_type
    """
    dataset = None
    from ec_dataset import ECDataset
    dataset = ECDataset()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt=opt, data_type=data_type)
    return dataset


def CreateDataLoader(opt, data_type):
    data_loader = ECDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt=opt, data_type=data_type)
    return data_loader


if __name__ == "__main__":
    from opt import *
    base_opt = BaseOptions()
    base_opt = base_opt.parse()
    ec_dataloader = ECDataLoader()
    ec_dataloader.initialize(opt=base_opt, data_type="train")
    train_loader = ec_dataloader.load_data()
    for i in train_loader:
        inputs = i["x"]
        # inputs.squeeze()
        print(inputs)
        outputs = i["y"]
        print(outputs)
