from torch.utils.tensorboard import SummaryWriter

def PrintModel(model, data):
    writer = SummaryWriter("./logs")
    writer.add_graph(model, data)

    writer.close()
