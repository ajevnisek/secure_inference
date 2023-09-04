import argparse
from flatten_networks import convert_old_format_checkpoint_to_new_format_checkpoint

def parse_args():

	parser = argparse.ArgumentParser(description="convert from structured checkpoints to flattened form")

	parser.add_argument("structured_checkpoint", type=str, help="path to checkpoint of the structured model")
	parser.add_argument("flattened_checkpoint", type=str, help="path to converted checkpoint which is in flattened form")

	return parser.parse_args()

def main():
	args = parse_args()

	convert_old_format_checkpoint_to_new_format_checkpoint(args.structured_checkpoint, args.flattened_checkpoint)

if __name__ == "__main__":
	main()
	