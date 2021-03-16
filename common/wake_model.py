#!/usr/bin/env python3

import numpy as np
import math


def partial_overlap(rr, r1, dy):
    r12 = r1**2
    rr2 = rr**2
    theta = 2*np.arccos((r12 + dy**2 - rr2)/(2*r1*dy))
    gamma = 2*np.arccos((rr2 + dy**2 - r12)/(2*rr*dy))
    return r12/4*(theta - np.sin(theta)) + rr2/4*(gamma - np.sin(gamma))


def area_overlap(dx, dy, alpha, rr):
    r1 = rr + alpha * dx
    totally_inside = np.logical_and((dy <= r1 - rr), dx > 0)
    partial_inside = np.logical_and(np.logical_and((dy < r1 + rr), (dy > r1 - rr)), dx > 0)
    a_ovp = np.zeros_like(dx)
    a_ovp[totally_inside] = np.pi * rr**2
    a_ovp[partial_inside] = partial_overlap(rr, r1[partial_inside], dy[partial_inside])

    return a_ovp


def aep(layout, w, alpha, rr, rho=1.225, Cp=0.4, a=1/3):
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

    for i in range(n):
        dx = X[0, :] - X[0, i]
        dy = np.abs(X[1, :] - X[1, i])
        Ao = area_overlap(dx, dy, alpha, rr)
        udef = 2 * a / ((1 + alpha * dx/rr) ** 2)
        udefAo2 = udefAo2 + (udef * Ao) ** 2
    udefAoA = np.sqrt(udefAo2) / (np.pi * rr * rr)
    u = u0 * (1 - udefAoA)
    u3 = u ** 3
    u3s = np.sum(u3)
    Ar = np.pi * rr * rr
    t = 3600 * 24 * 365
    Aep = 0.5 * u3s * pk * rho * Ar * Cp * t
    return Aep
