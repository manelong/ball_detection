from .heatmaps import BinaryFixedSizeMapGenerator, PrototypeBasedBinaryMapGenerator, BinaryFixedSizeMultiMapGenerator

__hm_generator_factory = {
    'binary_fixed_size': BinaryFixedSizeMapGenerator,
    'binary_prototype': PrototypeBasedBinaryMapGenerator,
    'binary_fixed_size_multi': BinaryFixedSizeMultiMapGenerator
        }

def select_heatmap_generator(cfg):
    hm_generator_name = cfg['name']
    if hm_generator_name not in __hm_generator_factory.keys():
        raise KeyError('unknown hm_generator: {}'.format(hm_generator_name))
    return __hm_generator_factory[hm_generator_name](cfg)

