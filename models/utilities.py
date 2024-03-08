import torch

'''
L2 Normalization : Model and lambda_l2 as ratio
'''
def l2_loss(model, lambda_l2):
    # L2 regularization
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param) ** 2
    loss = lambda_l2 * l2_reg
    return loss