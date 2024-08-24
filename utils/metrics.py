def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.argmax(1, keepdim = True)).sum()
    acc = correct.float() / y.shape[0]
    return acc