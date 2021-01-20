import torch


def make_batch(datas, category=None):
    if category is None:
        data_batch = datas[0].unsqueeze(0)
        i = 1
        while i < len(datas):
            temp = datas[i].unsqueeze(0)
            data_batch = torch.cat([data_batch, temp], dim=0)
            i += 1
    else:
        data_batch = datas[0][category][0].unsqueeze(0)
        i = 1
        while i < len(datas):
            temp = datas[i][category][0].unsqueeze(0)
            data_batch = torch.cat([data_batch, temp], dim=0)
            i += 1

    data_batch = data_batch.squeeze(dim=0)

    return data_batch





def time_calculator(sec):
    if sec < 60:
        return 0, 0, sec
    if sec < 3600:
        M = sec // 60
        S = sec % (M * 60)
        return 0, int(M), S
    H = sec // 3600
    sec = sec % 3600
    M = sec // 60
    S = sec % 60
    return int(H), int(M), S