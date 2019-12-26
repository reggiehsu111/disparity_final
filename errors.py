with open("log_error.txt", "r") as f:
	data = f.read().splitlines()
data = [float(x) for x in data]
sum = 0
for x in data:
	sum += x
print("###########################\n")
print("Average Error:", sum/10,"\n")
print("###########################")