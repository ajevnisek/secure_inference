import json
json.load(open('/home/uriel/research/secure_inference/results/all_induced_relus_except_from_one_layer_name_to_acc_added_1.txt', 'r'))
d = json.load(open('/home/uriel/research/secure_inference/results/all_induced_relus_except_from_one_layer_name_to_acc_added_1.txt', 'r'))
removed = []
for i in range(2, 16):
    new_d = json.load(open(f'/home/uriel/research/secure_inference/results/all_induced_relus_except_from_one_layer_name_to_acc_added_{i}.txt', 'r'))
    not_in_curr = set(d.keys()) ^ set(new_d.keys())
    removed += [x for x in not_in_curr if x not in removed]
print(removed)
