import argparse
import json
import pickle

from flattened_resnets_controller import ResNetController
from flattened_resnets_controller import set_hooks_resnet18

from flattened_resnet18 import FlattenedResNet18

"""
For supporting additional architectures, add the appropriate object to the model-to-object
mapping. Also, add a hook function that records the ReLU response at each ReLU layer (which is
to be defined in flattened_resnets_controller).
"""

MODEL_TO_OBJECT_MAPPING = {
	"ResNet18" : FlattenedResNet18(),
}

MODEL_TO_HOOK_FUNC = {
	"ResNet18" : set_hooks_resnet18
}

def parse_args():
	parser = argparse.ArgumentParser(description="create induced model given config file")

	parser.add_argument("network_type", type=str, help="the type of network", choices=list(MODEL_TO_OBJECT_MAPPING.keys()))
	parser.add_argument("network_config", type=str, help="path to MM config file of the given network")
	parser.add_argument("inducer_to_induced_json", type=str, help="path to a json that corresponds to inducer -> induced mapping")
	parser.add_argument("induced_model_name", type=str, help="the name to use for the new induced model")
	parser.add_argument("model_file_name", type=str, help="the file name of the saved induced model")
	parser.add_argument("-b", "--backbones_dir", type=str, default="mmpretrain/mmcls/models/backbones", help="mmcls backbones dir")

	return parser.parse_args()

def add_to_mmcls(args):

	with open(args.inducer_to_induced_json, "rb") as f:
		layer_induce_mapping = json.load(f)

	base_model = MODEL_TO_OBJECT_MAPPING[args.network_type]

	# TODO: currently the default input shape which is (3, 32, 32) is supported. Need to accept this as a parameter.
	controller = ResNetController(new_name=args.induced_model_name,
									new_path=args.model_file_name,
									old_model=base_model,
									set_hooks_func=MODEL_TO_HOOK_FUNC[args.network_type],
									path_to_flattened_resnet_declaration=args.network_config,
									inducer_to_induced=layer_induce_mapping)



	controller.add_declaration_to_mmcls_backbones(path_to_mmcls_backbones_dir=args.backbones_dir)


def main():
	args = parse_args()

	add_to_mmcls(args)

if __name__ == "__main__":
	main()