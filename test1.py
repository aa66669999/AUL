import torch
a=torch.rand(1)
b=torch.rand(10,5)
e=b.detach().numpy()
sorted_data, indices = torch.sort(b[:, 1], dim=0)
sorted_data = b[indices]
index=indices[:3]
print(b)
print(sorted_data)
print(index)

