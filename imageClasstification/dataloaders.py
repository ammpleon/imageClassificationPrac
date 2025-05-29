from torch.utils.data import DataLoader

def get_train_loader(train_set, batch_size, num_workers):
    return DataLoader(train_set, batch_size=batch_size, shuffle = True, num_workers=num_workers)

def get_test_loader(test_set, batch_size, num_workers):
    return DataLoader(test_set, batch_size=batch_size, shuffle = True, num_workers=num_workers)