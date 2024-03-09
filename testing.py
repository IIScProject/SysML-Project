import torch
# x = torch.tensor([1, 2, 3])
# y = torch.tensor([4, 5, 6])
# z = torch.tensor([7, 8, 9])
# tensors = [x, y, z]
# print(tensors)
#
# stacked_tensors = torch.stack(tensors, dim=1)
#
# # Print the stacked tensors
# print(stacked_tensors)

embedding = torch.nn.Embedding(10, 3)
input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
print("Input : ", input.shape)
print(embedding(input).shape)
