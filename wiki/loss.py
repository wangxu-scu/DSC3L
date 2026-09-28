import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

def cdist(x, y):
    '''
    x: n * dx
    y: m * dy
    '''

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist = torch.pow(x - y, 2).sum(2)
    return dist


def mcml_loss_compute(image_embeddings, text_embeddings, labels, args, f_divergence_type='KL'):
    Z = torch.cat((image_embeddings, text_embeddings), 0)
    # d = cdist(Z, Z)
    dist = Z.unsqueeze(1) - Z.unsqueeze(0)
    dist = (dist ** 2).sum(-1)

    labels = torch.cat((labels, labels), 0)
    # label mask
    mylabels = labels.float()

    # labelD = cdist(mylabels, mylabels)
    labelD = mylabels.unsqueeze(1) - mylabels.unsqueeze(0)
    labelD = (labelD ** 2).sum(-1)
    label_mask = (labelD < 0.5).float()  # 1-similar   0-dissimilar

    neg_dist = -dist
    P = torch.exp(neg_dist)
    if args.mask_out_digonal:
        # mask out diagonal elements
        mask = torch.eye(neg_dist.shape[0]).to(neg_dist.device)
        P = P * (1. - mask)
        label_mask = label_mask * (1. - mask)
    if args.distribution_symmetry:
        P_sum = P.sum() + 1e-8
        P = P / P_sum
        label_mask = label_mask / label_mask.sum() * label_mask.shape[0]
    else:
        P_sum = P.sum(dim=1, keepdim=True) + 1e-8
        P = P / P_sum
        label_mask = label_mask / torch.sum(label_mask, dim=1, keepdim=True)

    # KL Divergence
    q = (1e-8 + label_mask)
    p = (P + 1e-8)
    t = q / p
    if f_divergence_type == 'KL':
        # f_t = t * torch.log(t)
        # mcml_loss = torch.mean(torch.sum(f_t * p, 1))
        mcml_loss = torch.mean(torch.sum(q * torch.log(q / p), 1))
        # mcml_loss = torch.mean(torch.sum(label_mask * torch.log(1e-8 + label_mask / (P + 1e-8)), 1))
    elif f_divergence_type == 'reverse_KL':
        f_t = -torch.log(t)
        mcml_loss = torch.mean(torch.sum(f_t * p, 1))
    elif f_divergence_type == 'X2_divergence':
        f_t = (t - 1) ** 2
        mcml_loss = torch.mean(torch.sum(f_t * p, 1))
    # elif f_divergence_type == 'JS_divergence':
    #     m = (p + q) / 2.0
    #     t1 = p / m
    #     los_s_1 = torch.mean(torch.sum(t1 * torch.log(t1) * m, 1))
    #     t2 = q / m
    #     los_s_2 = torch.mean(torch.sum(t2 * torch.log(t2) * m, 1))
    #     mcml_loss = los_s_1 + los_s_2

    # mcml_loss = torch.mean(torch.sum(P * torch.log(1e-8 + P / (label_mask + 1e-8)), 1)) ## KL(P|Q)
    # mcml_loss = torch.mean(torch.sum(label_mask * torch.log((1e-8 + label_mask) / (P + 1e-8)), 1))  ## KL(Q|P)
    # mcml_loss = mcml_kl_loss
    return mcml_loss

def pairwise_KL(image_embeddings, text_embeddings):
    Z = torch.cat((image_embeddings, text_embeddings), 0)
    dist = Z.unsqueeze(1) - Z.unsqueeze(0)
    dist = (dist ** 2).sum(-1)
    P = F.softmax(-dist, dim=1)

    # Q = to_var(torch.eye(image_embeddings.shape[0], text_embeddings.shape[0]))
    Q = torch.eye(image_embeddings.shape[0], text_embeddings.shape[0]).to(P.device)
    Q = Q.repeat(2,2)
    p = P + 1e-10
    q = Q + 1e-10
    # return (p * (p / q).log()).sum(1).mean()
    # return (q * torch.log(q / p)).sum(1).mean()
    return (q * (q / p).log()).sum(1).mean()


def pairwise_KL_hupeng(image_embeddings, text_embeddings):
    dist = image_embeddings.unsqueeze(1) - text_embeddings.unsqueeze(0)
    dist = (dist ** 2).sum(-1)
    # P = (-dist).exp() + 1e-10
    # P = P / P.sum(1, keepdim=True)
    P = F.softmax(-dist, dim=1)

    # Q = to_var(torch.eye(image_embeddings.shape[0], text_embeddings.shape[0]))
    Q = torch.eye(image_embeddings.shape[0], text_embeddings.shape[0]).to(P.device)

    p = P + 1e-10
    q = Q + 1e-10
    # return (p * (p / q).log()).sum(1).mean()
    # return (q * torch.log(q / p)).sum(1).mean()
    return (q * (q / p).log()).sum(1).mean()


def cmpm_loss_compute(image_embeddings, text_embeddings, labels):
    """ Cross-Modal Projection Matching Loss (CMPM)
    Args:
        text_embeddings: text joint embeddings
        image_embeddings: image joint embeddings
        labels: class labels
    Returns:
        i2t_matching_loss: cmpm loss for image projected to text
        t2i_matching_loss: cmpm loss for text projected to image
    """
    # label mask
    batch_size = image_embeddings.size(0)
    mylabels = labels.float()
    labelD = cdist(mylabels, mylabels)
    label_mask = (labelD < 0.5).float()  # 1-match   0-unmatch
    # cross-modal scalar projection
    image_embeddings_norm = image_embeddings / torch.norm(image_embeddings, 2, -1, True)
    text_embeddings_norm = text_embeddings / torch.norm(text_embeddings, 2, -1, True)

    image_proj_text = image_embeddings @ text_embeddings_norm.t()
    text_proj_image = text_embeddings @ image_embeddings_norm.t()

    # softmax, higher scalar projection gives higher probability
    i2t_pred = F.softmax(image_proj_text)
    t2i_pred = F.softmax(text_proj_image)

    # normalize the true matching distribution
    label_mask = label_mask / torch.sum(label_mask, dim=1, keepdim=True)
    # KL Divergence
    i2t_matching_loss = torch.mean(torch.sum(i2t_pred * torch.log(1e-8 + i2t_pred / (label_mask + 1e-8)), 1))
    t2i_matching_loss = torch.mean(torch.sum(t2i_pred * torch.log(1e-8 + t2i_pred / (label_mask + 1e-8)), 1))

    # averaged cosine distance of positive and negative pairs for observation
    # cosdist = 1.0 - tf.matmul(text_embeddings_norm, tf.transpose(image_embeddings_norm))
    # cosdist = 1.0 - text_embeddings_norm @ image_embeddings_norm.t()
    #
    # pos_avg_dist = tf.reduce_mean(tf.boolean_mask(cosdist, tf.less(labelD, 0.5)))
    # neg_avg_dist = tf.reduce_mean(tf.boolean_mask(cosdist, tf.greater(labelD, 0.5)))
    cmpm_loss = i2t_matching_loss + t2i_matching_loss
    return cmpm_loss


def cmpm_loss_compute_2(image_embeddings, text_embeddings, labels):
    """ Cross-Modal Projection Matching Loss (CMPM)
    Args:
        text_embeddings: text joint embeddings
        image_embeddings: image joint embeddings
        labels: class labels
    Returns:
        i2t_matching_loss: cmpm loss for image projected to text
        t2i_matching_loss: cmpm loss for text projected to image
    """
    # label mask
    batch_size = image_embeddings.size(0)
    mylabels = labels.float()
    labelD = cdist(mylabels, mylabels)
    label_mask = (labelD < 0.5).float()  # 1-match   0-unmatch
    # cross-modal scalar projection
    image_embeddings_norm = image_embeddings / torch.norm(image_embeddings, 2, -1, True)
    text_embeddings_norm = text_embeddings / torch.norm(text_embeddings, 2, -1, True)

    image_proj_text = image_embeddings @ text_embeddings_norm.t()
    text_proj_image = text_embeddings @ image_embeddings_norm.t()

    image_proj_image = image_embeddings @ image_embeddings_norm.t()
    text_proj_text = text_embeddings @ text_embeddings_norm.t()

    # softmax, higher scalar projection gives higher probability
    i2t_pred = F.softmax(image_proj_text)
    t2i_pred = F.softmax(text_proj_image)
    i2i_pred = F.softmax(image_proj_image)
    t2t_pred = F.softmax(text_proj_text)

    # normalize the true matching distribution
    label_mask = label_mask / torch.sum(label_mask, dim=1, keepdim=True)
    # KL Divergence
    i2t_matching_loss = torch.mean(torch.sum(i2t_pred * torch.log(1e-8 + i2t_pred / (label_mask + 1e-8)), 1))
    t2i_matching_loss = torch.mean(torch.sum(t2i_pred * torch.log(1e-8 + t2i_pred / (label_mask + 1e-8)), 1))
    i2i_matching_loss = torch.mean(torch.sum(i2i_pred * torch.log(1e-8 + i2i_pred / (label_mask + 1e-8)), 1))
    t2t_matching_loss = torch.mean(torch.sum(t2t_pred * torch.log(1e-8 + t2t_pred / (label_mask + 1e-8)), 1))

    # averaged cosine distance of positive and negative pairs for observation
    # cosdist = 1.0 - tf.matmul(text_embeddings_norm, tf.transpose(image_embeddings_norm))
    # cosdist = 1.0 - text_embeddings_norm @ image_embeddings_norm.t()
    #
    # pos_avg_dist = tf.reduce_mean(tf.boolean_mask(cosdist, tf.less(labelD, 0.5)))
    # neg_avg_dist = tf.reduce_mean(tf.boolean_mask(cosdist, tf.greater(labelD, 0.5)))
    cmpm_loss = i2t_matching_loss + t2i_matching_loss + i2i_matching_loss + t2t_matching_loss
    return cmpm_loss

def xentropy_loss_compute(image_embeddings, text_embeddings, labels):
    feas = torch.cat((image_embeddings, text_embeddings), 0)
    labels = torch.cat((labels, labels), 0)
    loss = nn.CrossEntropyLoss()(feas, labels.squeeze(1))
    return loss

# def to_var(x):
#     """Converts numpy to variable."""
#     if torch.cuda.is_available():
#         x = x.cuda()
#     return Variable(x)