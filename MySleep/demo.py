import os
import tensorboard
from torch.utils.tensorboard import SummaryWriter

# write = SummaryWriter("./out_shhs/train/0/log")

# for i in range(100):
#     loss = i
#     write.add_scalar("loss",loss,i)
# write.close()
# files_list = os.listdir('./')
# filter_files_list=[fn for fn in files_list if fn.endswith("pt")]
# for i in range(len(filter_files_list)):
#     print(filter_files_list[i])

# a=100
# for i in range(50):
#     a=a*0.8
#     print(a)

# for i in range(10):
#     if i>5:
#         print(i*5)
#     print(i)

# a=[1,2,3]
# print(len(a))

my_path = './'
for file_name in os.listdir(my_path):
    if file_name.endswith('.pt'):
        os.remove(my_path + file_name)