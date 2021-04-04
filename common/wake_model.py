import numpy as np
from .windrose import read_windrose

def partial_overlap(r, r1, dy):
    r12 = r1**2
    r2 = r**2
    theta = 2*np.arccos((r12 + dy**2 - r2)/(2*r1*dy))
    gamma = 2*np.arccos((r2 + dy**2 - r12)/(2*r*dy))
    return r12/4*(theta - np.sin(theta)) + r2/4*(gamma - np.sin(gamma))


def power_vel_suzlon(ui):
    p = np.zeros_like(ui)
    t1 = np.logical_and(ui > 4, ui <= 10)
    t2 = np.logical_and(ui > 10, ui <= 12)
    t3 = 12 < ui
    p[t1] = -5.5348*ui[t1]**3 + 131.12*ui[t1]**2 - 776.86*ui[t1] + 1408.5
    p[t2] = 4.923*ui[t2]**3 -21.36*ui[t2]**2 + 3091.2*ui[t2] - 12987
    p[t3] = 1500
    return np.sum(p)


def area_overlap(dx, dy, alpha, r, rr):
    r1 = rr + alpha * dx
    totally_inside = np.logical_and((dy <= r1 - r), dx > 0)
    partial_inside = np.logical_and(np.logical_and((dy < r1 + r), (dy > r1 - r)), dx > 0)
    a_ovp = np.zeros_like(dx)
    a_ovp[totally_inside] = np.pi * r**2
    a_ovp[partial_inside] = partial_overlap(r, r1[partial_inside], dy[partial_inside])

    return a_ovp

def aep(layout, windspeed_array, theta_array, wind_prob, alpha, r, boundary_limits, rho=1.225, Cp=0.4, a=1/3):
# def aep(layout, alpha, r, boundary_limits, rho=1.225, Cp=0.4, a=1/3):
    """[summary]

    Returns:
        [float]: [energy in kW year]
    """


    rr = r*np.sqrt(2)
    Ar = np.pi * r * r
    
    x = layout[::2]
    y = layout[1::2]
    n = len(x)
    
    dxf = np.vstack([x]*n)
    dyf = np.vstack([y]*n)

    # dimension -> n x n
    dx_full = dxf - dxf.T
    dy_full = dyf - dyf.T

    dxdy_full = np.zeros((n, n, 2))
    dxdy_full[:, :, 0] = dx_full
    dxdy_full[:, :, 1] = dy_full

    # dimension -> 2 x 2 x len(theta)
    rotation_array = np.array([[np.cos(theta_array), np.sin(theta_array)], [-np.sin(theta_array), np.cos(theta_array)]])

    # dimension -> n x n x 2 x len(theta)
    dxdy_full_rotate = dxdy_full.dot(rotation_array)
    dxdy_full_rotate[:, :, 1, :] = np.abs(dxdy_full_rotate[:, :, 1, :])

    # dimension -> n x n x len(theta)
    area_over = area_overlap(dxdy_full_rotate[:, :, 0, :], dxdy_full_rotate[:, :, 1, :], alpha, r, rr)
    u_def = 2*a / ((1 + alpha * dxdy_full_rotate[:, :, 0, :]/rr)**2)
    u_def_ao2 = (u_def*area_over)**2

    # dimension -> n x len(theta)
    u_def_aoa = 1 - np.sqrt(np.sum(u_def_ao2, axis=0)) / Ar
    
    # dimension -> n x len(theta) x len(wind)
    u_effective = np.tensordot(u_def_aoa, windspeed_array, 0)
    
    # dimension -> len(theta) x len(wind) same as wind_prob
    u_effective_sum = np.sum(u_effective**3, 0)
    
    power = 0.5*u_effective_sum * wind_prob * rho * Ar * Cp / 1000
    power_total = np.sum(power)
    penalty = penalty_function(x, y, dx_full, dy_full, 10*r, boundary_limits, rho=1)

    return power_total, penalty

    # wx = w[0]
    # wy = w[1]
    # pk = 1
    # X = np.vstack([x, y])
    # theta = np.arctan2(wy, wx)
    # u0 = np.sqrt(wx ** 2 + wy ** 2)

    # rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    # X = np.dot(rot, X)

    # udefAo2 = np.zeros_like(x)




    # for i in range(n):
    #     dx = X[0, :] - X[0, i]
    #     dy = np.abs(X[1, :] - X[1, i])
    #     dx_full[i] = dx
    #     dy_full[i] = dy
    #     Ao = area_overlap(dx, dy, alpha, r)
    #     udef = 2 * a / ((1 + alpha * dx/rr) ** 2)
    #     udefAo2 = udefAo2 + (udef * Ao) ** 2
    
    # penalty = penalty_function(x, y, dx_full, dy_full, 10*r, boundary_limits)
    
    # udefAoA = np.sqrt(udefAo2) / Ar
    # u = u0 * (1 - udefAoA)
    # u3 = u ** 3
    # u3s = np.sum(u3)
    # Aep = 0.5 * u3s * pk * rho * Ar * Cp / 1000
    # return Aep, penalty


def penalty_function(x, y, dx_full, dy_full, min_dist, boundary_limits, rho=1):
    dist = np.sqrt(dx_full**2 + dy_full**2)
    n = len(dist)
    dist[range(n), range(n)] = 1000
    cond1 = dist < min_dist

    beta1 = np.sum(min_dist - dist[cond1])/2

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

    beta = (rho * beta1) ** 2 + (rho * (beta2 + beta3) ** 2)
    return beta



if __name__ == "__main__":
    # windspeed_array = np.array([2, 6, 12])
    windspeed_array = np.array([12])
    # theta_array = np.array([0, np.pi/2, np.pi, -np.pi/2])
    theta_array = np.array([0])
    # wind_prob = np.ones((4, 3))/12
    wind_prob = np.array([1])
    y = 1000*np.ones(5)
    x = 200*np.linspace(0, 4, 5)
    boundary_limits = np.array([[0, 2000], [0, 2000]])
    layout = np.zeros(10)
    layout[::2] = x
    layout[1::2] = y
    Z_H = 60
    Z_0 = 0.3
    alpha = 0.5 / (np.log(Z_H / Z_0))
    r = 20

    power, penalty = aep(layout, windspeed_array, theta_array, wind_prob, alpha, r, boundary_limits)
    print(power, penalty)
