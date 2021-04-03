#!/usr/bin/env python3
import numpy as np
import math


def partial_overlap(rr, r1, dy):
    r12 = r1**2
    rr2 = rr**2
    theta = 2*np.arccos((r12 + dy**2 - rr2)/(2*r1*dy))
    gamma = 2*np.arccos((rr2 + dy**2 - r12)/(2*rr*dy))
    return r12/4*(theta - np.sin(theta)) + rr2/4*(gamma - np.sin(gamma))


def power_vel_suzlon(ui):
    p = np.zeros_like(ui)
    t1 = np.logical_and(ui > 4, ui <= 10)
    t2 = np.logical_and(ui > 10, ui <= 12)
    t3 = 12 < ui
    p[t1] = -5.5348*ui[t1]**3 + 131.12*ui[t1]**2 - 776.86*ui[t1] + 1408.5
    p[t2] = 4.923*ui[t2]**3 -21.36*ui[t2]**2 + 3091.2*ui[t2] - 12987
    p[t3] = 1500
    return np.sum(p)


def area_overlap(dx, dy, alpha, rr):
    r1 = rr + alpha * dx
    totally_inside = np.logical_and((dy <= r1 - rr), dx > 0)
    partial_inside = np.logical_and(np.logical_and((dy < r1 + rr), (dy > r1 - rr)), dx > 0)
    a_ovp = np.zeros_like(dx)
    a_ovp[totally_inside] = np.pi * rr**2
    a_ovp[partial_inside] = partial_overlap(rr, r1[partial_inside], dy[partial_inside])

    return a_ovp


def aep(layout, w, alpha, rr, boundary_limits=[[0, 2000], [0, 2000]], rho=1.225, Cp=0.4, a=1/3):
    """[summary]

    Returns:
        [float]: [energy in kWhr]
    """
    wx = w[0]
    wy = w[1]
    pk = 1
    x = layout[::2]
    y = layout[1::2]
    X = np.vstack([x, y])

    n = len(x)
    theta = np.arctan2(wy, wx)
    u0 = np.sqrt(wx ** 2 + wy ** 2)

    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    X = np.dot(rot, X)

    udefAo2 = np.zeros_like(x)

    dx_full = np.zeros((n, n))
    dy_full = np.zeros((n, n))

    for i in range(n):
        dx = X[0, :] - X[0, i]
        dy = np.abs(X[1, :] - X[1, i])
        dx_full[i] = dx
        dy_full[i] = dy
        Ao = area_overlap(dx, dy, alpha, rr)
        udef = 2 * a / ((1 + alpha * dx/rr) ** 2)
        udefAo2 = udefAo2 + (udef * Ao) ** 2

    penalty = penalty_function(x, y, dx_full, dy_full, 5*rr, boundary_limits)

    udefAoA = np.sqrt(udefAo2) / (np.pi * rr * rr)
    u = u0 * (1 - udefAoA)
    u3 = u ** 3
    u3s = np.sum(u3)
    Ar = np.pi * rr * rr
    t = 3600 * 24 * 365
    # Aep = 0.5 * u3s * pk * rho * Ar * Cp * t / 3.6e6
    Aep = power_vel_suzlon(u)*pk * 365 * 24
    return Aep, penalty
    # return Aep


def penalty_function(x, y, dx_full, dy_full, min_dist, boundary_limits, rho=1):
    # print(min_dist)
    dist = np.sqrt(dx_full**2 + dy_full**2)
    # print("dist", dist)
    n = len(dist)
    dist[range(n), range(n)] = 1e10
    cond1 = dist < min_dist
    # print(min_dist - dist[cond1])

    beta1 = np.sum(min_dist - dist[cond1])/2
    # print("distance penalty", beta1)

    xlimits = boundary_limits[0]
    ylimits = boundary_limits[1]

    # beta2 calculation
    idx1 = x < xlimits[0]
    idx2 = y < ylimits[0]

    beta2 = abs(x[idx1]).sum() + abs(y[idx2]).sum()

    # beta3 calculation

    idx1 = x > xlimits[1]
    idx2 = y > ylimits[1]

    beta3 = abs(x[idx1] - xlimits[1]).sum() + abs(y[idx2] - ylimits[1]).sum()

    # print(beta1, "1")
    # print(beta2, "2")
    # print(beta3, "3")


    beta = (rho* 5 * beta1) ** 2 + (rho * (beta2 + beta3) ** 2)
    return beta