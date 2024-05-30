def printData(writer, title, value, epoch):

    writer.add_scalar(title, value, global_step=epoch+1)

    writer.close()