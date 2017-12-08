import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle

#filename = 'models/4096_random_forest'
#y_pred, y_true = pickle.load(open(filename + '.sav', 'rb'))

# filename = 'late_fusion_predictions'				 # saved only predictions for late fusion so only
# y_pred = pickle.load(open(filename + '.sav', 'rb')) # uncomment this two lines and keep the above lines uncommented

#labels = list(set(sorted(y_true)))

#m = np.zeros((25, 25))
#classes_total = {}
#classes_total_notright = {}
#predicted_right = 0
#for i in range(len(y_pred)):
#    if y_pred[i] == y_true[i]:
#        predicted_right += 1
#        if y_true[i] not in classes_total:
#            classes_total[y_true[i]] = 1
#        else:
#            classes_total[y_true[i]] += 1
#        m[labels.index(y_pred[i]) - 1, labels.index(y_pred[i]) - 1] += 1
#    else:
#        m[labels.index(y_pred[i]) - 1, labels.index(y_true[i]) - 1] += 1

#    if y_true[i] not in classes_total_notright:
#        classes_total_notright[y_true[i]] = 1
#    else:
#        classes_total_notright[y_true[i]] += 1

#acc_total = 0
#for class_ in classes_total_notright:
#    if class_ in classes_total:
#        num = classes_total[class_]
#    else:
#        num = 0
#    acc_total += num / (1. * classes_total_notright[class_])

#print(filename)
#print('AA:', acc_total / 25.)
#print('OA:', predicted_right / 1021.)

#sum_rowsm = np.sum(m, axis=1)
#sum_rowsm[sum_rowsm == 0] = 1
#m = m / sum_rowsm[:, np.newaxis]

# filename = 'input_example.csv'
#data = np.loadtxt(filename, delimiter=';')
def cria_map(matrix, filename):
    data = matrix

    desc = range(1, 26)  # ['Earth', 'Moon', 'Jupiter', 'Mars', 'Pluto', 'Saturn']
    # 
    strategy_path = []
    # 
    # # Computing strategies
    # for s in stat:
    for d in desc:
        strategy_path.append(d)
        # strategy_path.append(s + ': ' + d)

    ##############################################################################

    #  Finishing Touches
    fig, ax = plt.subplots()
    # using the ax subplot object, we use the same
    # syntax as above, but it allows us a little
    # bit more advanced control
    # ax.pcolor(data,cmap=plt.cm.Reds,edgecolors='k')
    # ax.pcolor(data,cmap=plt.cm.Greens)
    # ax.pcolor(data,cmap=plt.cm.gnuplot)
    ax.set_xticks(np.arange(0, len(strategy_path)))
    ax.set_yticks(np.arange(0, len(strategy_path)))

    # cmap = plt.get_cmap('BlueRed2')
    plt.imshow(data, cmap=plt.cm.gnuplot, interpolation='nearest')
    # plt.imshow(data, cmap=plt.cm.gnuplot)
    # plt.clim(-0.05,0.25)
    plt.colorbar()

    # Here we put the x-axis tick labels
    # on the top of the plot.  The y-axis
    # command is redundant, but inocuous.
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    # similar syntax as previous examples
    ax.set_xticklabels(strategy_path, minor=False, fontsize=12, rotation=90)
    ax.set_yticklabels(strategy_path, minor=False, fontsize=12)

    # Here we use a text command instead of the title
    # to avoid collision between the x-axis tick labels
    # and the normal title position
    # plt.text(0.5,1.08,'Main Plot Title',
    #         fontsize=25,
    #         horizontalalignment='center',
    #         transform=ax.transAxes
    #         ) 

    # standard axis elements
    # plt.ylabel('Y Axis Label',fontsize=10)
    # plt.xlabel('X Axis Label',fontsize=10)

    plt.savefig(filename + '.png', bbox_inches='tight')
    plt.savefig(filename + '.pdf', bbox_inches='tight')
    #plt.show()
