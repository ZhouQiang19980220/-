import matplotlib.pyplot as plt

def draw(loss_record, hint):
    plt.plot(loss_record, label=hint)
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.title("accuracy curve")