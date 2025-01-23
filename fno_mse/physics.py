import torch

def get_kernels(device, direction='central'):
    # # Shifting kernel
    # k_s_x, k_s_y, k_s_z = torch.zeros((1, 1, 3, 3, 3)), torch.zeros((1, 1, 3, 3, 3)), torch.zeros((1, 1, 3, 3, 3))

    # k_s_x[0, 0, 1, 1, 2] = 1
    # k_s_y[0, 0, 1, 2, 1] = 1
    # k_s_z[0, 0, 2, 1, 1] = 1

    # Finite diff kernel
    k_f_x, k_f_y, k_f_z = torch.zeros((1, 1, 3, 3, 3)), torch.zeros((1, 1, 3, 3, 3)), torch.zeros((1, 1, 3, 3, 3))

    if direction == 'central':
        k_f_x[0, 0, 1, 1, 2] = 1
        k_f_x[0, 0, 1, 1, 0] = -1

        k_f_y[0, 0, 1, 2, 1] = 1
        k_f_y[0, 0, 1, 0, 1] = -1

        k_f_z[0, 0, 2, 1, 1] = 1
        k_f_z[0, 0, 0, 1, 1] = -1
    
    elif direction == 'forward':
        k_f_x[0, 0, 1, 1, 2] = 1
        k_f_x[0, 0, 1, 1, 1] = -1

        k_f_y[0, 0, 1, 2, 1] = 1
        k_f_y[0, 0, 1, 1, 1] = -1

        k_f_z[0, 0, 2, 1, 1] = 1
        k_f_z[0, 0, 1, 1, 1] = -1
    
    elif direction == 'backward':
        k_f_x[0, 0, 1, 1, 1] = 1
        k_f_x[0, 0, 1, 1, 0] = -1

        k_f_y[0, 0, 1, 1, 1] = 1
        k_f_y[0, 0, 1, 0, 1] = -1

        k_f_z[0, 0, 1, 1, 1] = 1
        k_f_z[0, 0, 0, 1, 1] = -1
    
    else:
        raise Exception('Direction should be "central", "forward", or "backward"')

    return k_f_x.to(device), k_f_y.to(device), k_f_z.to(device)


def divergence(x, kernels, direction='central'):
    k_f_x, k_f_y, k_f_z = kernels
    f_x, f_y, f_z = x.squeeze()
    f_x, f_y, f_z = [f.reshape((1, 1, f.shape[0], f.shape[1], f.shape[2])) for f in [f_x, f_y, f_z]]
    
    fx_p = torch.nn.functional.pad(f_x, (1, 1, 1, 1, 1, 1), mode='reflect')
    # fx_p = torch.nn.functional.pad(fx_p, (1, 1, 1, 1, 0, 0), mode='constant', value=0)
    fy_p = torch.nn.functional.pad(f_y, (1, 1, 1, 1, 1, 1), mode='reflect')
    # fy_p = torch.nn.functional.pad(fy_p, (0, 0, 1, 1, 1, 1), mode='constant', value=0)
    fz_p = torch.nn.functional.pad(f_z, (1, 1, 1, 1, 1, 1), mode='reflect')
    # fz_p = torch.nn.functional.pad(fz_p, (0, 0, 1, 1, 1, 1), mode='constant', value=0)

    dfx_dx = torch.nn.functional.conv3d(input=fx_p, weight=k_f_x)
    dfy_dy = torch.nn.functional.conv3d(input=fy_p, weight=k_f_y)
    dfz_dz = torch.nn.functional.conv3d(input=fz_p, weight=k_f_z)

    return dfx_dx + dfy_dy + dfz_dz

def curl(x, kernels, direction='central'):
    k_f_x, k_f_y, k_f_z = kernels
    f_x, f_y, f_z = x.squeeze()
    f_x, f_y, f_z = [f.reshape((1, 1, f.shape[0], f.shape[1], f.shape[2])) for f in [f_x, f_y, f_z]]

    fx_p = torch.nn.functional.pad(f_x, (1, 1, 1, 1, 1, 1), mode='reflect')
    fy_p = torch.nn.functional.pad(f_y, (1, 1, 1, 1, 1, 1), mode='reflect')
    fz_p = torch.nn.functional.pad(f_z, (1, 1, 1, 1, 1, 1), mode='reflect')

    # i-term
    dfz_dy = torch.nn.functional.conv3d(input=fz_p, weight=k_f_y)
    dfy_dz = torch.nn.functional.conv3d(input=fy_p, weight=k_f_z)
    curl_i = dfz_dy - dfy_dz
    
    # j-term
    dfz_dx = torch.nn.functional.conv3d(input=fz_p, weight=k_f_x)
    dfx_dz = torch.nn.functional.conv3d(input=fx_p, weight=k_f_z)
    curl_j = -(dfz_dx - dfx_dz)

    # k-term
    dfy_dx = torch.nn.functional.conv3d(input=fy_p, weight=k_f_x)
    dfx_dy = torch.nn.functional.conv3d(input=fx_p, weight=k_f_y)
    curl_k = dfy_dx - dfx_dy

    return torch.cat([curl_i, curl_j, curl_k], dim=1)

def conductivity(x):
    _, _, f_z = x.squeeze()
    f_z = f_z.squeeze()

    slicewise_flux = torch.sum(f_z, dim=(1, 2))
    slicewise_flux[-1] = slicewise_flux[-2]
    
    mean_flux = torch.mean(slicewise_flux).float()
    std_flux = torch.std(slicewise_flux).float()
    err_pred = std_flux / mean_flux
    
    FF_pred = f_z.size(dim=0) / mean_flux
    FF_err_pred = FF_pred * err_pred

    ff_dict = {'Slicewise Flux': slicewise_flux,
               'Mean Flux': mean_flux,
               'Std Flux': std_flux,
               'Flux Err': err_pred,
               'FF': FF_pred,
               'FF err': FF_err_pred}
    return ff_dict


if __name__ == "__main__":
    x = torch.randn(1, 3, 100, 100, 100)  # Replace with your vector field data
    curl_f = curl(x)
    # print(curl_f)
    div_curl_f = conductivity(curl_f)
    print(div_curl_f['Mean Flux'])
    # total_divergence = torch.sum(div_curl_f)
    # print(torch.max(torch.abs(div_curl_f)))
