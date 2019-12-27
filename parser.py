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
		ret = super().parse_args()
		self.conf_dict = vars(ret)
		return ret

	def read_config(self, in_config_path):
		with open(in_config_path, 'r') as f:
			self.conf_dict = json.load(f)
		return

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