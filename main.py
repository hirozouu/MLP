import model
import learn
import matplotlib.pyplot as plt


def main():
    param = model.Setting()

    num_layers = [1, 2, 3, 4, 5, 6, 7, 8]

    losses = []
    evals = []
    accs = []

    for layer in num_layers:
        param.setSetting(32, layer)
        loss, eval, acc = learn.learn(param)
        losses.append(loss)
        evals.append(eval)
        accs.append(acc)

    plt.figure()
    for loss, layer in zip(losses, num_layers):
        plt.plot(range(1, param.num_epochs+1), loss, label=layer)
    plt.title('history')
    plt.xlabel('epoch')
    plt.legend()
    #plt.show()
    plt.savefig('figure/total_loss.png')
    plt.close()

    plt.figure()
    for eval, layer in zip(evals, num_layers):
        plt.plot(range(1, param.num_epochs+1), eval, label=layer)
    plt.title('history')
    plt.xlabel('epoch')
    plt.legend()
    #plt.show()
    plt.savefig('figure/eval_loss.png')
    plt.close()

    plt.figure()
    for acc, layer in zip(accs, num_layers):
        plt.plot(range(1, param.num_epochs+1), acc, label=layer)
    plt.title('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    #plt.show()
    plt.savefig('figure/acc.png')
    plt.close()

if __name__ == '__main__':
    main()