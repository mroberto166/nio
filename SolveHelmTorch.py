import time

import matplotlib.pyplot as plt
import numpy as np
import torch

pi = np.pi
w = 2 * pi


def g(x_, y_):
    return 0.


def exact_u(x_, y_):
    return torch.sin(w * x_) * torch.sin(w * y_)


def bc(x_, y_, c):
    boundary_data = torch.cos(w * (x_ * torch.cos(c) + y_ * torch.sin(c)))
    return boundary_data
    # return 0.


def solve_helm(kappa, L, plot=False):
    start = time.time()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    nx = ny = 70
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig2, axes = plt.subplots(3, 3, figsize=(10, 10))
    list_neumann_map = list()
    coeff_vec = torch.linspace(0, 2 * pi, L)
    for index, coeff_bc in enumerate(coeff_vec):

        A = torch.zeros((nx * ny, nx * ny))
        b = torch.zeros((nx * ny,))
        tmp = torch.zeros((nx * ny,))
        u_ex = torch.zeros((nx * ny,))
        speed_coeff = torch.zeros((nx * ny,))
        mask = torch.zeros((nx * ny,))

        inputs = torch.zeros((nx * ny, 2))
        h = dx

        i, j = torch.arange(1, nx - 1), torch.arange(1, ny - 1)
        ii, jj = torch.meshgrid(i, j)
        x_, y_ = torch.meshgrid(x[1:-1], y[1:-1])
        l = ii + jj * nx
        m = ii + 1 + jj * nx
        n = ii - 1 + jj * nx
        o = ii + (jj + 1) * nx
        p = ii + (jj - 1) * nx
        A[l, l] = -4 / h ** 2 + w ** 2 * kappa(x_, y_) ** 2
        A[m, l] = 1 / h ** 2
        A[n, l] = 1 / h ** 2
        A[o, l] = 1 / h ** 2
        A[p, l] = 1 / h ** 2

        inputs[l, 0] = x_
        inputs[l, 1] = y_
        b[l] = g(x_, y_)
        u_ex[l] = exact_u(x_, y_)
        speed_coeff[l] = kappa(x_, y_)

        i, j = torch.arange(1, nx - 1), torch.zeros((1,), dtype=i.dtype)
        ii, jj = torch.meshgrid(i, j)
        x_, y_ = torch.meshgrid(x[i], y[j])
        l = ii + jj * nx
        m = ii + 1 + jj * nx
        n = ii - 1 + jj * nx
        o = ii + (jj + 1) * nx
        o2 = ii + (jj + 2) * nx

        A[l, l] = -(1 / dx ** 2 +
                    1 / dx ** 2 -
                    1 / dy ** 2) + w ** 2 * kappa(x_, y_) ** 2

        A[m, l] = 1 / dx ** 2
        A[n, l] = 1 / dx ** 2

        A[o, l] = -1 / dy ** 2 - 1 / dy ** 2
        A[o2, l] = 1 / dy ** 2

        inputs[l, 0] = x_
        inputs[l, 1] = y_

        b[l] = g(x_, y_)
        u_ex[l] = exact_u(x_, y_)
        speed_coeff[l] = kappa(x_, y_)
        tmp[l] = bc(x_, y_, coeff_bc)
        mask[l] = 1

        j, i = torch.arange(1, ny - 1), torch.zeros((1,), dtype=j.dtype)
        ii, jj = torch.meshgrid(i, j)
        x_, y_ = torch.meshgrid(x[i], y[j])

        l = ii + jj * nx
        m = ii + 1 + jj * nx
        m2 = ii + 2 + jj * nx
        o = ii + (jj + 1) * nx
        p = ii + (jj - 1) * nx

        A[l, l] = -(1 / dy ** 2 +
                    1 / dy ** 2 -
                    1 / dx ** 2) + w ** 2 * kappa(x_, y_) ** 2

        A[m, l] = -1 / dx ** 2 - 1 / dx ** 2
        A[m2, l] = 1 / dx ** 2

        A[o, l] = 1 / dy ** 2
        A[p, l] = 1 / dy ** 2

        inputs[l, 0] = x_
        inputs[l, 1] = y_

        b[l] = g(x_, y_)
        u_ex[l] = exact_u(x_, y_)
        speed_coeff[l] = kappa(x_, y_)
        tmp[l] = bc(x_, y_, coeff_bc)
        mask[l] = 1

        i, j = torch.arange(1, nx - 1), torch.full((1,), fill_value=ny - 1, dtype=i.dtype)
        ii, jj = torch.meshgrid(i, j)
        x_, y_ = torch.meshgrid(x[i], y[j])

        l = ii + jj * nx
        m = ii + 1 + jj * nx
        n = ii - 1 + jj * nx
        p = ii + (jj - 1) * nx
        p2 = ii + (jj - 2) * nx

        A[l, l] = -(1 / dx ** 2 +
                    1 / dx ** 2 -
                    1 / dy ** 2) + w ** 2 * kappa(x_, y_) ** 2
        A[m, l] = 1 / dx ** 2
        A[n, l] = 1 / dx ** 2

        A[p, l] = -1 / dy ** 2 - 1 / dy ** 2
        A[p2, l] = 1 / dy ** 2

        inputs[l, 0] = x_
        inputs[l, 1] = y_

        b[l] = g(x_, y_)
        u_ex[l] = exact_u(x_, y_)
        speed_coeff[l] = kappa(x_, y_)
        tmp[l] = bc(x_, y_, coeff_bc)
        mask[l] = 1

        j, i = torch.arange(1, ny - 1), torch.full((1,), fill_value=nx - 1, dtype=i.dtype)
        ii, jj = torch.meshgrid(i, j)
        x_, y_ = torch.meshgrid(x[i], y[j])

        l = ii + jj * nx
        n = ii - 1 + jj * nx
        n2 = ii - 2 + jj * nx
        o = ii + jj * nx
        p = ii + (jj - 1) * nx

        A[l, l] = -(1 / dy ** 2 +
                    1 / dy ** 2 -
                    1 / dx ** 2) + w ** 2 * kappa(x_, y_) ** 2

        A[n2, l] = 1 / dx ** 2
        A[n, l] = - 1 / dx ** 2 - 1 / dx ** 2

        A[o, l] = 1 / dy ** 2
        A[p, l] = 1 / dy ** 2

        inputs[l, 0] = x_
        inputs[l, 1] = y_

        b[l] = g(x_, y_)
        u_ex[l] = exact_u(x_, y_)
        speed_coeff[l] = kappa(x_, y_)
        tmp[l] = bc(x_, y_, coeff_bc)
        mask[l] = 1

        # print("matrix assembled")

        i, j = torch.tensor([0, nx - 1]), torch.tensor([0, ny - 1])
        ii, jj = torch.meshgrid(i, j)
        x_, y_ = torch.meshgrid(x[i], y[j])

        l = ii + jj * nx

        inputs[l, 0] = x_
        inputs[l, 1] = y_

        A[l, l] = (-1 / dx ** 2 - 1 / dy ** 2) + w ** 2 * kappa(x_, y_) ** 2

        b[l] = g(x_, y_)
        u_ex[l] = exact_u(x_, y_)
        speed_coeff[l] = kappa(x_, y_)
        tmp[l] = bc(x_, y_, coeff_bc)
        mask[l] = 1

        mask = mask > 0.5

        A = A.to(device)
        b = b.to(device)
        mask = mask.to(device)
        tmp = tmp.to(device)

        A = A.T
        mul = torch.matmul(A, tmp)
        f = b - mul

        A_masked_1 = A[~mask, :]
        A_masked = A_masked_1[:, ~mask]

        f_masked = f[~mask]
        # print("System solving")
        # print(A.shape)
        u_free = torch.linalg.solve(A_masked, f_masked.reshape(-1, 1))

        u_free = u_free.reshape(nx - 2, ny - 2)
        u_free = torch.nn.functional.pad(u_free, (1, 1, 1, 1), value=0).reshape(-1, )

        u = torch.where(~mask, u_free, tmp)

        gr = inputs.reshape(nx, ny, 2)

        if plot:
            err = (torch.mean(abs(u - u_ex) ** 2) / torch.mean(u_ex ** 2)) ** 0.5 * 100
            q = index // 3
            mod = index % 3
            axes[q, mod].contourf(gr[:, :, 0], gr[:, :, 1], u.reshape(nx, ny).detach(), aspect="auto", cmap="jet", levels=200)
            print("L2 relative error norm: ", err)

        i, j = torch.arange(1, nx - 1), torch.zeros((1,), dtype=i.dtype)
        ii, jj = torch.meshgrid(i, j)
        l = ii + jj * nx
        o = ii + (jj + 1) * nx

        u_der_1 = -(u[o] - u[l]) / dy
        # u_b.append(u[l])

        j, i = torch.arange(1, ny - 1), torch.full((1,), fill_value=nx - 1, dtype=i.dtype)
        ii, jj = torch.meshgrid(i, j)
        l = ii + jj * nx
        n = ii - 1 + jj * nx

        u_der_2 = ((u[l] - u[n]) / dx).T
        # u_b.append(u[l].T)

        # for i in range(0, nx):
        i, j = torch.arange(nx - 2, 0, -1), torch.full((1,), fill_value=ny - 1, dtype=i.dtype)
        i, j = torch.meshgrid(i, j)

        l = i + j * nx
        p = i + (j - 1) * nx

        u_der_3 = (u[l] - u[p]) / dx
        # u_b.append(u[l])

        j, i = torch.arange(ny - 2, 0, -1), torch.full((1,), fill_value=0, dtype=i.dtype)
        i, j = torch.meshgrid(i, j)

        l = i + j * nx
        m = i + 1 + j * nx

        u_der_4 = (-(u[m] - u[l]) / dy).T

        u_der = torch.cat((u_der_1, u_der_2, u_der_3, u_der_4))

        list_neumann_map.append(u_der.reshape(-1, 1))

        if plot:
            ax1.scatter(torch.arange(0, u_der.shape[0]), u_der.detach(), label="Derivative", s=8)
            plt.legend()

    if plot:
        plt.figure()
        plt.contourf(gr[:, :, 0], gr[:, :, 1], speed_coeff.reshape(nx, ny).detach(), levels=200)

    end = time.time()

    elapsed_time = end - start
    print("elapsed time: ", elapsed_time)

    neumann_map = torch.cat(list_neumann_map, 1)
    if plot:
        plt.show()

    return neumann_map, speed_coeff.reshape(nx, ny)
