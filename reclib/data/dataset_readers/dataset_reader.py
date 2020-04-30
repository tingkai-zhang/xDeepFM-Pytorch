import torch.utils.data


class DatasetReader(torch.utils.data.Dataset):
    def __init__(self, dataset_path=None, cache_path='.criteo', rebuild_cache=False, min_threshold=10):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def __build_cache(self, path, cache_path):
        pass

    def __get_feat_mapper(self, path):
        pass

    def __yield_buffer(self, path, feat_mapper, defaults, buffer_size=int(1e5)):
        pass
