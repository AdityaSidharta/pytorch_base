import torch.nn.functional as F


def loss_fn(model, criterion, data):
    img, target = data
    prediction = model(img)
    loss = criterion(prediction, target)
    return loss


def metric_fn(model, data):
    img, target = data
    prediction = model(img)
    metric = F.mse(prediction, target)
    return metric


def pred_fn(model, data):
    img = data
    prediction = model(img)
    prediction_array = prediction.data.cpu().numpy() * 1024.
    return prediction_array.tolist()
