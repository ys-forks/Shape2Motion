import numpy as np

# instance_data = {
#     'motion_dof_type': np.zeros(7)
# }
array = np.zeros(7)
indices = np.array([0,1])
array[[0,1]][[0,1]] = 1
# instance_data[anchor_indices][instance_data[anchor_indices] != 1] += 1

print(array)