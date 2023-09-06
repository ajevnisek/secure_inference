"""This module controls ResNets integration with induced ReLU."""
import os, pickle
import torch

FLATTENED_RESNET18_DECLARATION_PATH = 'flattened_resnet18.py'
INDUCER_TO_INDUCED = {'relu0':['ResLayer0_BasicBlockV20_relu_1',
                               'ResLayer0_BasicBlockV20_relu_2',
                               'ResLayer0_BasicBlockV21_relu_1',
                               'ResLayer0_BasicBlockV21_relu_2',
                               'ResLayer1_BasicBlockV22_relu_1',
                               'ResLayer1_BasicBlockV22_relu_2',
                               'ResLayer1_BasicBlockV23_relu_1',
                               'ResLayer1_BasicBlockV23_relu_2',
                               ],
                    'ResLayer2_BasicBlockV24_relu_1':['ResLayer2_BasicBlockV24_relu_2',
                                                      'ResLayer2_BasicBlockV25_relu_1',
                                                      'ResLayer2_BasicBlockV25_relu_2',
                                                      'ResLayer3_BasicBlockV26_relu_1',
                                                      'ResLayer3_BasicBlockV26_relu_2',
                                                      'ResLayer3_BasicBlockV27_relu_1',
                                                      'ResLayer3_BasicBlockV27_relu_2',
                               ]}


relu_names_to_dims = {}
def getActivation(name):
    # the hook signat`ure
    def hook(model, input, output):
        relu_names_to_dims[name] = input[0].shape
    return hook


def mul(iterable):
    result = 1
    for item in iterable:
        result *= item
    return result


def find_line_index_which_starts_with(full_text, starter):
    for idx, line in enumerate(full_text.splitlines()):
        if line.replace(' ', '').startswith(starter.replace(' ', '')):
            return idx
    raise ValueError(f"could not find line index for starter {starter}")


def set_hooks_resnet18(model):
    model.relu0.register_forward_hook(getActivation('relu0'))
    model.ResLayer0_BasicBlockV20_relu_1.register_forward_hook(getActivation('ResLayer0_BasicBlockV20_relu_1'))
    model.ResLayer0_BasicBlockV20_relu_2.register_forward_hook(getActivation('ResLayer0_BasicBlockV20_relu_2'))

    model.ResLayer0_BasicBlockV21_relu_1.register_forward_hook(getActivation('ResLayer0_BasicBlockV21_relu_1'))
    model.ResLayer0_BasicBlockV21_relu_2.register_forward_hook(getActivation('ResLayer0_BasicBlockV21_relu_2'))

    model.ResLayer1_BasicBlockV22_relu_1.register_forward_hook(getActivation('ResLayer1_BasicBlockV22_relu_1'))
    model.ResLayer1_BasicBlockV22_relu_2.register_forward_hook(getActivation('ResLayer1_BasicBlockV22_relu_2'))

    model.ResLayer1_BasicBlockV23_relu_1.register_forward_hook(getActivation('ResLayer1_BasicBlockV23_relu_1'))
    model.ResLayer1_BasicBlockV23_relu_2.register_forward_hook(getActivation('ResLayer1_BasicBlockV23_relu_2'))

    model.ResLayer2_BasicBlockV24_relu_1.register_forward_hook(getActivation('ResLayer2_BasicBlockV24_relu_1'))
    model.ResLayer2_BasicBlockV24_relu_2.register_forward_hook(getActivation('ResLayer2_BasicBlockV24_relu_2'))

    model.ResLayer2_BasicBlockV25_relu_1.register_forward_hook(getActivation('ResLayer2_BasicBlockV25_relu_1'))
    model.ResLayer2_BasicBlockV25_relu_2.register_forward_hook(getActivation('ResLayer2_BasicBlockV25_relu_2'))

    model.ResLayer3_BasicBlockV26_relu_1.register_forward_hook(getActivation('ResLayer3_BasicBlockV26_relu_1'))
    model.ResLayer3_BasicBlockV26_relu_2.register_forward_hook(getActivation('ResLayer3_BasicBlockV26_relu_2'))

    model.ResLayer3_BasicBlockV27_relu_1.register_forward_hook(getActivation('ResLayer3_BasicBlockV27_relu_1'))
    model.ResLayer3_BasicBlockV27_relu_2.register_forward_hook(getActivation('ResLayer3_BasicBlockV27_relu_2'))
    return model


class ResNetController:
    def __init__(self,
                 new_name,
                 new_path,
                 old_model,
                 set_hooks_func=set_hooks_resnet18,
                 image_dims: tuple = (3, 32, 32),
                 path_to_flattened_resnet_declaration: str = FLATTENED_RESNET18_DECLARATION_PATH,
                 inducer_to_induced: dict = INDUCER_TO_INDUCED,
                 set_as_relu=None, set_as_identity=None, relu_name_to_dim=None):
        self.new_name = new_name
        self.new_path = new_path
        self.model = old_model
        self.set_hooks = set_hooks_func
        self.image_dims = image_dims
        self.path_to_flattened_resnet_declaration = path_to_flattened_resnet_declaration
        self.inducer_to_induced = inducer_to_induced
        if relu_name_to_dim is None:
            self.relu_name_to_dim = self.infer_relu_dimensions()
        else:
            self.relu_name_to_dim = relu_name_to_dim
        self.set_as_relu = set_as_relu if set_as_relu else []
        self.set_as_identity = set_as_identity if set_as_identity else []


    def infer_relu_dimensions(self):
        model = self.set_hooks(self.model)
        model(torch.rand((16, *self.image_dims)))
        return {k: v[1:] for k, v in relu_names_to_dims.items()}


    def create_new_class_declaration(self):
        with open(self.path_to_flattened_resnet_declaration, 'r') as f:
            original_flattened_resnet18 = f.read()

        resnet_with_induced_relu = original_flattened_resnet18
        idx = find_line_index_which_starts_with(resnet_with_induced_relu, '@BACKBONES')
        resnet_with_induced_relu = '\n'.join(resnet_with_induced_relu.splitlines()[:idx] +
                                             ['from mmcls.models.backbones.induced_relu import InducedReLU',
                                             '', '',
                                              f'class {self.new_name}(torch.nn.Module): ',
                                              ' '* 4 + 'def __init__(self, choosing_matrices: dict):',
                                              ' ' * 8 + f'super({self.new_name}, self).__init__()',
                                              ' ' * 8 + 'self.choosing_matrices = choosing_matrices'] +
                                             resnet_with_induced_relu.splitlines()[idx + 4:])

        for k in self.inducer_to_induced:
            if len(self.inducer_to_induced[k]) > 0:
                # replace the relu declaration to be without inplace:
                idx_to_replace = find_line_index_which_starts_with(resnet_with_induced_relu, f'self.{k}')
                resnet_with_induced_relu = '\n'.join(
                        resnet_with_induced_relu.splitlines()[:idx_to_replace] +
                        [' ' * 8  + f'self.{k} = ReLU(inplace=False)'] +
                        resnet_with_induced_relu.splitlines()[idx_to_replace + 1:])
                # replace the relu evaluation to also record the drelu of this relu
                idx_to_replace = find_line_index_which_starts_with(resnet_with_induced_relu, f'out = self.{k}(out)')
                resnet_with_induced_relu = '\n'.join(
                    resnet_with_induced_relu.splitlines()[:idx_to_replace] +
                    [' ' * 8 + f'{k}_drelu = (out >= 0).float()',
                     ' ' * 8 + f'out = self.{k}(out)'] +
                    resnet_with_induced_relu.splitlines()[idx_to_replace + 1:])
                # replace the relus of induced layers to be induced relus:
                for induced_layer in self.inducer_to_induced[k]:
                    idx_to_replace = find_line_index_which_starts_with(resnet_with_induced_relu, f'self.{induced_layer}')
                    resnet_with_induced_relu = '\n'.join(
                        resnet_with_induced_relu.splitlines()[:idx_to_replace] +
                        [' ' * 8 + f'self.{induced_layer} = InducedReLU(choosing_matrix=choosing_matrices["{k}->{induced_layer}"], this_layer_shape={self.relu_name_to_dim[induced_layer]})'] +
                        resnet_with_induced_relu.splitlines()[idx_to_replace + 1:])

                    idx_to_replace = find_line_index_which_starts_with(resnet_with_induced_relu, f'out = self.{induced_layer}(out)')
                    resnet_with_induced_relu = '\n'.join(
                        resnet_with_induced_relu.splitlines()[:idx_to_replace ] +
                        [' ' * 8 + f'out = self.{induced_layer}(out, {k}_drelu)'] +
                        resnet_with_induced_relu.splitlines()[idx_to_replace + 1:])

        # replace the activations with ReLUs:
        for k in self.set_as_relu:
            idx_to_replace = find_line_index_which_starts_with(resnet_with_induced_relu, f'self.{k}')
            resnet_with_induced_relu = '\n'.join(
                resnet_with_induced_relu.splitlines()[:idx_to_replace] +
                [' ' * 8 + f'self.{k} = ReLU(inplace=False)'] +
                resnet_with_induced_relu.splitlines()[idx_to_replace + 1:])
        # replace the activations with Identities:
        for k in self.set_as_identity:
            idx_to_replace = find_line_index_which_starts_with(resnet_with_induced_relu, f'self.{k}')
            resnet_with_induced_relu = '\n'.join(
                resnet_with_induced_relu.splitlines()[:idx_to_replace] +
                [' ' * 8 + f'self.{k} = Identity()'] +
                resnet_with_induced_relu.splitlines()[idx_to_replace + 1:])
        return resnet_with_induced_relu

    def create_declaration(self):
        new_decl = self.create_new_class_declaration()
        with open(self.new_path, 'w') as f:
            f.write(new_decl)
    def add_declaration_to_mmcls_backbones(self, path_to_mmcls_backbones_dir='mmpretrain/mmcls/models/backbones'):
        new_decl = self.create_new_class_declaration()
        idx = find_line_index_which_starts_with(new_decl, 'class')
        classname = new_decl.splitlines()[idx].split('class ')[1].split('(')[0]
        
        new_decl = '\n'.join(
            new_decl.splitlines()[:idx] +
            ['@BACKBONES.register_module()'] +
            new_decl.splitlines()[idx:])
        
        python_filename = os.path.basename(self.new_path)

        new_path = os.path.join(path_to_mmcls_backbones_dir, os.path.basename(self.new_path))
        with open(new_path, 'w') as f:
            f.write(new_decl)

        init_path = os.path.join(path_to_mmcls_backbones_dir, '__init__.py')
        with open(init_path, 'r') as f:
            init_data = f.read()
        idx = find_line_index_which_starts_with(init_data, '__all__')
        init_data = '\n'.join(
            init_data.splitlines()[:idx] +
            [f'from .{python_filename[:-3]} import {classname}', '', '',
             ] +
            init_data.splitlines()[idx:])
        idx = find_line_index_which_starts_with(init_data, ']')
        init_data = '\n'.join(
            init_data.splitlines()[:idx] +
            [' ' * 4 + f'"{classname}", ', '',] +
            init_data.splitlines()[idx:])
        with open(init_path, 'w') as f:
            f.write(init_data)


def create_dummy_choosing_matrices(controller):
    #from flattened_resnet18_halfway_induced import HalfWayResNet18
    choosing_matrices = {}
    for k in controller.inducer_to_induced:
        if len(controller.inducer_to_induced[k]):
            for induced in controller.inducer_to_induced[k]:
                choosing_matrices[f"{k}->{induced}"] = torch.sparse_coo_tensor(size=(
                    mul(controller.relu_name_to_dim[k]), mul(controller.relu_name_to_dim[induced])))
    torch.save(choosing_matrices, 'all_zeros_choosing_matrices_for_.HalfWayResNet18.pth')


def main():
    #from flattened_resnet18 import FlattenedResNet18
    #flattened_model = FlattenedResNet18()
    #controller = ResNetController(new_name='HalfWayResNet18', new_path='flattened_resnet18_halfway_induced.py',
    #                              old_model=flattened_model, )
    #controller.create_declaration()
    path = 'layer_name_to_matrix.pkl'
    old_layer_name_to_choosing_matrix = pickle.load(open(path, 'rb'))

    old_layer_name_to_new_name = {
        'layer1[0].relu_1': 'ResLayer0_BasicBlockV20_relu_1',
        'layer1[0].relu_2': 'ResLayer0_BasicBlockV20_relu_2',
        'layer1[1].relu_1': 'ResLayer0_BasicBlockV21_relu_1',
        'layer1[1].relu_2': 'ResLayer0_BasicBlockV21_relu_2',
        'layer2[0].relu_1': 'ResLayer1_BasicBlockV22_relu_1',
        'layer2[0].relu_2': 'ResLayer1_BasicBlockV22_relu_2',
        'layer2[1].relu_1': 'ResLayer1_BasicBlockV23_relu_1',
        'layer2[1].relu_2': 'ResLayer1_BasicBlockV23_relu_2',
        'layer3[0].relu_1': 'ResLayer2_BasicBlockV24_relu_1',
        'layer3[0].relu_2': 'ResLayer2_BasicBlockV24_relu_2',
        'layer3[1].relu_1': 'ResLayer2_BasicBlockV25_relu_1',
        'layer3[1].relu_2': 'ResLayer2_BasicBlockV25_relu_2',
        'layer4[0].relu_1': 'ResLayer3_BasicBlockV26_relu_1',
        'layer4[0].relu_2': 'ResLayer3_BasicBlockV26_relu_2',
        'layer4[1].relu_1': 'ResLayer3_BasicBlockV27_relu_1',
        'layer4[1].relu_2': 'ResLayer3_BasicBlockV27_relu_2'
    }
    choosing_matrices = {}
    for k in old_layer_name_to_choosing_matrix:
        new_layer_name = old_layer_name_to_new_name[k]
        if new_layer_name in INDUCER_TO_INDUCED['relu0']:
            choosing_matrices[f'relu0->{new_layer_name}'] = old_layer_name_to_choosing_matrix[k]
        else:
            choosing_matrices[f'ResLayer2_BasicBlockV24_relu_1->{new_layer_name}'] = old_layer_name_to_choosing_matrix[k]
    with open("layer_name_to_matrix_new.pkl", "wb") as f:
        pickle.dump(choosing_matrices, f)

if __name__ == '__main__':
    main()
