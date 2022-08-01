import torch
import torch.nn as nn
import torch.nn.functional as F


def knowledge_distill(stu_logits, tea_logits, label, alpha, T):
    xe_loss = F.cross_entropy(stu_logits, label)

    tea_prob = F.softmax(tea_logits/T, dim=-1)
    kl_loss = - tea_prob * F.log_softmax(stu_logits/T, -1) * T * T
    kl_loss = kl_loss.sum(1).mean()

    loss = (1 - alpha) * xe_loss + alpha * kl_loss

    return loss

# def xERM(stu_logits, tea_logits, label, alpha=1, gamma=1, T=1):
#     XE_s = F.cross_entropy(stu_logits, label, reduction='none')
#     XE_t = F.cross_entropy(tea_logits, label, reduction='none')

#     tea_prob = F.softmax(tea_logits/T, dim=-1)
#     kl_loss = - tea_prob * F.log_softmax(stu_logits/T, -1) * T * T
#     kl_loss = kl_loss.sum(1)

#     w_t = torch.pow(XE_s, gamma)/(torch.pow(XE_s, gamma) + torch.pow(XE_t, gamma) + 1e-5)
#     w_t = w_t.detach()
#     w_s = 1 - w_t

#     loss = (w_s * XE_s).mean() + alpha*(w_t * kl_loss).mean()

#     return loss

# def xERM(stu_logits, tea_logits, label, alpha=1, gamma=1, T=1):
#     XE_s = F.cross_entropy(stu_logits, label, reduction='none')
#     XE_t = F.cross_entropy(tea_logits, label, reduction='none')

#     tea_prob = F.softmax(tea_logits/T, dim=-1)
#     kl_loss = - tea_prob * F.log_softmax(stu_logits/T, -1) * T * T
#     kl_loss = kl_loss.sum(1)

#     p_s = torch.exp(-XE_s)
#     p_t = torch.exp(-XE_t)
#     w_t = p_t/(p_s + p_t + 1e-5)
#     w_t = w_t.detach()
#     w_s = 1 - w_t

#     loss = (w_s * XE_s).mean() + alpha*(w_t * kl_loss).mean()

#     return loss


def xERM(stu_logits, tea_logits, label, alpha=1, gamma=1, T=1):
    XE_s = F.cross_entropy(stu_logits, label, reduction='none')
    XE_t = F.cross_entropy(tea_logits, label, reduction='none')

    tea_prob = F.softmax(tea_logits/T, dim=-1)
    kl_loss = - tea_prob * F.log_softmax(stu_logits/T, -1) * T * T
    kl_loss = kl_loss.sum(1)

    p_s = torch.exp(-XE_s)
    p_t = torch.exp(-10*XE_t)
    w_t = p_t/(p_s + p_t + 1e-5)
    w_t = w_t.detach()
    w_s = 1 - w_t

    loss = (w_s * XE_s).mean() + alpha*(w_t * kl_loss).mean()

    return loss