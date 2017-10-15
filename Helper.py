import matplotlib.pyplot as plt

def readParamList(path):
    reader = open(read_path, 'r')
    value = reader.read().split('\n')[:-1]
    reader.close()
    return value

def drawLossGraph(model_path, epoch):
    train_loss_strings = readParamList('/params/train/losses_0')
    train_losses = []
    infer_loss_strings = readParamList('/params/train/infer_losses_0')
    infer_losses = []
    for i in range(len(train_loss_strings)):
        trainlosses.append(float(train_loss_strings[i]))
        inferlosses.append(float(infer_loss_strings[i]))
    plt.plot(range(1,len(train_losses) + 1), train_losses, color='blue')
    plt.plot(range(1,len(infer_losses) + 1), infer_losses, color='red')
    plt.show()

def main(args):
    code.interact(local=dict(globals(), **locals()))

if __name__ == '__main__':
    main(sys.argv[1:])