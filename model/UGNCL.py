import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from model.SGRAF import SGRAF


def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)

    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)

    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def mse_loss(label, alpha, c, kxi1):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    alp = E * (1 - label) + 1
    C = kxi1 * KL(alp, c)
    return (A + B) + C


class DECL(nn.Module):
    def __init__(self, opt):
        super(DECL, self).__init__()
        self.opt = opt
        self.grad_clip = opt.grad_clip
        self.kxi1 = opt.kxi1
        self.kxi2 = opt.kxi2
        self.similarity_model = SGRAF(opt)
        self.theta = opt.theta
        self.params = list(self.similarity_model.params)
        self.optimizer = torch.optim.Adam(self.params, lr=opt.learning_rate)
        self.step = 0

    def state_dict(self):
        return self.similarity_model.state_dict()

    def load_state_dict(self, state_dict):
        self.similarity_model.load_state_dict(state_dict)

    def train_start(self):
        """switch to train mode"""
        self.similarity_model.train_start()

    def val_start(self):
        """switch to valuate mode"""
        self.similarity_model.val_start()

    def get_alpha(self, images, captions, lengths, evi_mode="soft"):
        img_embs1, img_embs2, cap_embs, cap_lens = self.similarity_model.forward_emb(images, captions, lengths)
        sims1, evidences1, sims_tanh1 = self.similarity_model.forward_sim(img_embs1, cap_embs, cap_lens, 'not sims', evi_mode)
        sims2, evidences2, sims_tanh2 = self.similarity_model.forward_sim(img_embs2, cap_embs, cap_lens, 'not sims', evi_mode)
        
        sum_e1 = evidences1 + evidences1.t()
        sum_e2 = evidences2 + evidences2.t()
        
        norm_e1 = sum_e1 / torch.sum(sum_e1, dim=1, keepdim=True)
        norm_e2 = sum_e2 / torch.sum(sum_e2, dim=1, keepdim=True)

        alpha1_i2t = evidences1 + 1
        alpha1_t2i = evidences1.t() + 1

        alpha2_i2t = evidences2 + 1
        alpha2_t2i = evidences2.t() + 1

        return alpha1_i2t, alpha1_t2i, alpha2_i2t, alpha2_t2i, norm_e1, norm_e2, sims1, sims2, sims_tanh1, sims_tanh2

    def TRL_loss(self, scores, neg=None, c_idx=None, h_idx=None, hc_idx=None, hn_idx=None, labels=None, mode="warmup", uncertainty=None):

        if neg is None:
            neg = self.theta

        margin = self.opt.margin
        if mode == "train":
            # the calibrated margins by the estimated soft correspondence labels
            s = (torch.pow(10, labels) - 1) / 9
            soft_margins = margin * s
            soft_margins[c_idx] = margin

            # obtain the trusted soft margins through our derived uncertainty
            trusted_margins = soft_margins
            hc = 1.0 / (1 + torch.pow(-self.opt.triangle, (1 / (uncertainty[hc_idx]) - 1)))
            hn = 1.0 / (1 + torch.pow(-self.opt.triangle, (uncertainty[hn_idx] / (1 - uncertainty[hn_idx]))))
            trusted_margins[hc_idx] = soft_margins * hc
            trusted_margins[hn_idx] = soft_margins * hn

            margin = trusted_margins
            

        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        mask = torch.eye(scores.size(0)) > .5
        mask = mask.cuda()

        cost_s = (margin + scores - d1).clamp(min=0)
        cost_im = (margin + scores - d2).clamp(min=0)
        cost_s, cost_im = cost_s.masked_fill_(mask, 0), cost_im.masked_fill_(mask, 0)

        if mode == "train":
            if len(c_idx) == 0:
                cost_clean = torch.tensor(0).cuda()
            else:
                cost_s_clean, cost_im_clean = cost_s[c_idx].max(1)[0], cost_im[c_idx].max(0)[0]

                cost_clean = cost_s_clean.sum() + cost_im_clean.sum()

            if neg > len(h_idx):
                neg = len(h_idx)

            top_neg_s_hard = (torch.topk(cost_s[h_idx], k=neg, dim=1).values.sum(dim=1)) / neg
            top_neg_im_hard = (torch.topk(cost_im[h_idx], k=neg, dim=1).values.sum(dim=1)) / neg

            cost_hard = top_neg_s_hard.sum() + top_neg_im_hard.sum()

            cost = cost_clean + cost_hard

            return cost
        else:
            top_neg_row = torch.topk(cost_s, k=neg, dim=1).values
            top_neg_column = torch.topk(cost_im.t(), k=neg, dim=1).values
            return (top_neg_row.sum(dim=1) + top_neg_column.sum(dim=1)) / neg  # (K,1)

    def warmup_batch(self, images, captions, lengths):
        self.step += 1
        batch_length = images.size(0)
        neg = max(int(self.opt.batch_size - self.opt.delta * self.step), self.theta)

        alpha1_i2t, alpha1_t2i, alpha2_i2t, alpha2_t2i, norm_e1, norm_e2, sims1, sims2, sims_tanh1, sims_tanh2 = self.get_alpha(images, captions, lengths, evi_mode="warmup")
        self.optimizer.zero_grad()
        batch_labels = torch.eye(batch_length).cuda().long()
        
        loss_edl1_i2t = mse_loss(batch_labels, alpha1_i2t, batch_length, self.kxi1)
        loss_edl1_t2i = mse_loss(batch_labels, alpha1_t2i, batch_length, self.kxi1)
        loss_edl1 = torch.mean(loss_edl1_i2t + loss_edl1_t2i)

        loss_edl2_i2t = mse_loss(batch_labels, alpha2_i2t, batch_length, self.kxi1)
        loss_edl2_t2i = mse_loss(batch_labels, alpha2_t2i, batch_length, self.kxi1)
        loss_edl2 = torch.mean(loss_edl2_i2t + loss_edl2_t2i)

        # sum
        loss_edl = loss_edl1 + loss_edl2

        loss_trl1 = self.TRL_loss(sims_tanh1, neg=neg)
        loss_trl1 = loss_trl1.sum() * self.kxi2

        loss_trl2 = self.TRL_loss(sims_tanh2, neg=neg)
        loss_trl2 = loss_trl2.sum() * self.kxi2

        loss_trl = loss_trl1 + loss_trl2
        
        loss = loss_edl + loss_trl
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
        self.logger.update('Step', self.step)
        self.logger.update('H_n', neg)  # The hardness
        self.logger.update('Lr', self.optimizer.param_groups[0]['lr'])
        self.logger.update('Loss_edl', loss_edl.item(), batch_length)
        self.logger.update('Loss_trl', loss_trl.item(), batch_length)
        self.logger.update('Loss', loss.item(), batch_length)

    def train_batch(self, images, captions, lengths, preds, u_preds, uncertainty, probabilities):
        self.step += 1
        k = images.size(0)
        neg = max(int(self.opt.batch_size - self.opt.delta * self.step), self.theta)
        batch_length = images.size(0)
    
        alpha1_i2t, alpha1_t2i, alpha2_i2t, alpha2_t2i, norm_e1, norm_e2, sims1, sims2, sims_tanh1, sims_tanh2 = self.get_alpha(images, captions, lengths)

        alpha1 = (alpha1_i2t + alpha1_t2i) / 2
        alpha2 = (alpha2_i2t + alpha2_t2i) / 2

        # alpha, b, u = DS_Combin_two(alpha1, alpha2, k)

        self.optimizer.zero_grad()
        preds = preds.cuda()
        u_preds = u_preds.cuda()
        self.optimizer.zero_grad()
        batch_labels = torch.eye(batch_length)
        n_idx = (1 - preds).nonzero().view(1, -1)[0].tolist()
        c_idx = preds.nonzero().view(1, -1)[0].tolist()

        for i in n_idx:
            batch_labels[i][i] = 0

        batch_labels = batch_labels.cuda().long()
        loss_edl1_i2t = mse_loss(batch_labels, alpha1_i2t, batch_length, self.kxi1)
        loss_edl1_t2i = mse_loss(batch_labels, alpha1_t2i, batch_length, self.kxi1)
        loss_edl1 = torch.mean(loss_edl1_i2t + loss_edl1_t2i)

        loss_edl2_i2t = mse_loss(batch_labels, alpha2_i2t, batch_length, self.kxi1)
        loss_edl2_t2i = mse_loss(batch_labels, alpha2_t2i, batch_length, self.kxi1)
        loss_edl2 = torch.mean(loss_edl2_i2t + loss_edl2_t2i)

        loss_edl = (loss_edl1 + loss_edl2) / 2

        # divide by uncertainty
        u_c_idx = torch.nonzero(u_preds > 0).view(1, -1)[0].tolist()
        u_h_idx = torch.nonzero(u_preds < 0).view(1, -1)[0].tolist()
        u_n_idx = torch.nonzero(u_preds == 0).view(1, -1)[0].tolist()

        labels = probabilities.diag().view(probabilities.size(0), 1)

        # divide by probabilities
        u_hc_idx = []
        u_hn_idx = []
        for index in range(u_h_idx):
            if labels[index] > self.opt.eta:
                u_hc_idx.append(u_h_idx[index])
            else:
                u_hn_idx.append(u_h_idx[index])

        loss_trl1 = self.TRL_loss(sims_tanh1, neg=neg, c_idx=u_c_idx, h_idx=u_h_idx, hc_idx=u_hc_idx, hn_idx=u_hn_idx, labels=labels, mode="train", uncertainty=uncertainty)
        loss_trl2 = self.TRL_loss(sims_tanh2, neg=neg, c_idx=u_c_idx, h_idx=u_h_idx, hc_idx=u_hc_idx, hn_idx=u_hn_idx, labels=labels, mode="train", uncertainty=uncertainty)

        loss_trl = (loss_trl1 + loss_trl2) * self.kxi2

        loss = loss_edl + loss_trl

        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
        self.logger.update('Step', self.step)
        self.logger.update('H_n', neg)  # The hardness
        self.logger.update('Lr', self.optimizer.param_groups[0]['lr'])
        self.logger.update('Loss_edl', loss_edl.item(), batch_length)
        self.logger.update('Loss_trl', loss_trl.item(), batch_length)
        self.logger.update('Loss', loss.item(), batch_length)


def DS_Combin_two(alpha1, alpha2, K):
    """
    :param alpha1: Dirichlet distribution parameters of view 1
    :param alpha2: Dirichlet distribution parameters of view 2
    :return: Combined Dirichlet distribution parameters
    """
    alpha = dict()
    alpha[0], alpha[1] = alpha1, alpha2
    b, S, E, u = dict(), dict(), dict(), dict()
    for v in range(2):
        S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
        E[v] = alpha[v] - 1
        b[v] = E[v] / (S[v].expand(E[v].shape))
        u[v] = K / S[v]

    # b^0 @ b^(0+1)
    bb = torch.bmm(b[0].view(-1, K, 1), b[1].view(-1, 1, K))
    # b^0 * u^1
    uv1_expand = u[1].expand(b[0].shape)
    bu = torch.mul(b[0], uv1_expand)
    # b^1 * u^0
    uv_expand = u[0].expand(b[0].shape)
    ub = torch.mul(b[1], uv_expand)
    # calculate C
    bb_sum = torch.sum(bb, dim=(1, 2), out=None)
    bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
    C = bb_sum - bb_diag

    # calculate b^a
    b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
    # calculate u^a
    u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))

    # calculate new S
    S_a = K / u_a
    # calculate new e_k
    e_a = torch.mul(b_a, S_a.expand(b_a.shape))
    alpha_a = e_a + 1
    return alpha_a, b_a, u_a