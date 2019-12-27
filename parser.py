import argparse
import json

class ConfigParser(argparse.ArgumentParser):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		# Get namspace
		self.args = argparse.Namespace()
		self.conf_dict = vars(self.args)
		return

	def parse_args(self):
		self.args = super().parse_args()
		self.conf_dict = vars(self.args)
		return self.args

	def read_config(self, in_config_path):
		print("Reading from config...")
		with open(in_config_path, 'r') as f:
			temp_dict = json.load(f)
		for k in temp_dict:
			self.conf_dict[k] = temp_dict[k]
		self.args.__dict__.update(self.conf_dict)
		return self.args


	def write_config(self, out_config_path):
		with open(out_config_path, 'w') as f:
			json.dump(self.conf_dict, f, indent=4)
		return

	def print_config(self):
		print("\n#########################")
		print("Config:\n")
		for k in self.conf_dict:
			print(k, ':', self.conf_dict[k])
		print("#########################\n")