import torch
import torch.nn as nn
import random
import torch.nn.functional as F

class PSO:
    def __init__(self, model, loss, images, labels, eps):
        # 参数设置
        self.c1 = 0.5
        self.c2 = 0.5
        self.w = 1
        self.pN = 20  # Particle number

        self.images = images.detach()
        self.labels = labels
        self.device = images.device
        self.model = model
        self.loss = loss
        self.eps = eps

        self.resize_rate = 0.9
        self.diversity_prob = 0.5

        [s, b, m, n] = self.images.size()

        self.X = torch.zeros(self.pN, s, b, m, n).to(self.device)  # perturbation
        self.G = torch.zeros(self.pN, s, b, m, n).to(self.device)   # grad
        self.pbest = torch.zeros(self.pN, s, b, m, n).to(self.device)  # Individual optimization
        self.gbest = torch.zeros_like(self.images).to(self.device)  # Global optimization
        self.xbest = torch.zeros_like(self.images).to(self.device)  # best perturbation
        self.p_fit = torch.zeros(self.pN).to(self.device)  # Optimal individual
        self.fit = torch.tensor(0).to(self.device)  # Optimal global

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        if self.resize_rate > 1:
            padded = F.interpolate(padded, size=[img_size, img_size], mode='bilinear', align_corners=False)

        return padded if torch.rand(1) < self.diversity_prob else x

    def Rot(self, x):
        random_number = torch.rand(1)
        rotation_angle = random_number * 20 - 10
        theta = torch.tensor([[torch.cos(torch.deg2rad(rotation_angle)), -torch.sin(torch.deg2rad(rotation_angle)), 0],
                              [torch.sin(torch.deg2rad(rotation_angle)), torch.cos(torch.deg2rad(rotation_angle)), 0]])
        grid = F.affine_grid(theta.unsqueeze(0).expand(x.size(0), -1, -1).float().to(x.device),
                             x.size())
        rotated_tensor = F.grid_sample(x, grid)

        return rotated_tensor

    def input_mix_resize_uni(self, x):
        C, H, W, D = x.shape

        patch_size = (H, 10, 10)
        start_h = random.randint(0, H - patch_size[0])
        start_w = random.randint(0, W - patch_size[1])
        start_d = random.randint(0, D - patch_size[2])

        scaled_tensor = F.interpolate(x.unsqueeze(0), size=patch_size, mode='trilinear',
                                      align_corners=False)
        scaled_tensor = scaled_tensor.squeeze(0)


        filled_tensor = x.clone()
        filled_tensor[:, start_h:start_h + patch_size[0], start_w:start_w + patch_size[1], start_d:start_d + patch_size[2]] = scaled_tensor
        return filled_tensor

    def function(self, i):
        img = self.X[i, :, :, :, :] + self.images
        img.requires_grad = True
        outs_adv = self.model(img)
        # num = self.loss(outs_adv, self.labels)
        out_C = torch.argmax((outs_adv), 1)
        num = len(self.labels)-torch.sum(out_C == self.labels, 0)

        return num


def psodim_attack(model, device, images, labels, eps):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    loss = nn.CrossEntropyLoss()

    attack_steps = 10
    alpha = 2 / 255
    decay = 1.0

    # initialize
    pso_x = PSO(model, loss, images, labels, eps)
    for i in range(pso_x.pN):
        pso_x.X[i, :, :, :, :] = torch.randn_like(pso_x.images).uniform_(-pso_x.eps, pso_x.eps)

    for step in range(attack_steps):
        for i in range(pso_x.pN):
            # update
            img = pso_x.X[i, :, :, :, :] + pso_x.images
            img.requires_grad = True

            a = torch.randint(3, (1,))
            if a == 0:
                outputs = pso_x.model(pso_x.input_diversity(img))
            if a == 1:
                outputs = pso_x.model(pso_x.Rot(img))
            if a == 2:
                outputs = pso_x.model(pso_x.input_mix_resize_uni(img))

            # Calculate loss
            cost = pso_x.loss(outputs, pso_x.labels)

            grad = torch.autograd.grad(cost, img,
                                       retain_graph=False, create_graph=False)[0]

            pso_x.pbest[i, :, :, :, :] = (pso_x.pbest[i, :, :, :, :] * step + grad) / (step + 1)
            pso_x.gbest = (pso_x.gbest * i + grad) / (i + 1)

            r1 = torch.rand(1).to(pso_x.device)
            r2 = torch.rand(1).to(pso_x.device)

            grad_n = pso_x.w * grad + pso_x.c1 * r1 * (pso_x.pbest[i, :, :, :, :] - grad) + pso_x.c2 * r2 * (
                    pso_x.gbest - grad)

            grad_n = grad_n / torch.norm(grad_n, p=1)
            grad_n = grad_n + pso_x.G[i, :, :, :, :] * decay

            # update cumulative gradient
            pso_x.G[i, :, :, :, :] = grad_n

            adv_images = img.detach() + alpha * grad_n.sign()
            delta = torch.clamp(adv_images - pso_x.images, min=-pso_x.eps, max=pso_x.eps)
            adv_images = torch.clamp(pso_x.images + delta, min=0, max=1).detach()
            pso_x.X[i, :, :, :, :] = adv_images - pso_x.images

            num = pso_x.function(i)

            # update individual optimum
            if num > pso_x.p_fit[i]:
                pso_x.p_fit[i] = num
            # Update global optimum
            if pso_x.p_fit[i] > pso_x.fit:
                pso_x.fit = pso_x.p_fit[i]

                pso_x.xbest = pso_x.X[i, :, :, :, :]

    adv_images = pso_x.xbest + pso_x.images

    return adv_images