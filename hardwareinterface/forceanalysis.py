import os
from sys import meta_path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm

from pycito import utilities as utils

from pydrake.all import PiecewisePolynomial

SOURCE = os.path.join('hardwareinterface','data','hardware_test_07_21.pkl')
EXT = '.pdf'

SAVEDIR = os.path.join('hardwareinterface','figures')

PLOTSTART = 20
PLOTSTOP = 23

FOOTLABELS = ['FR','FL','RR','RL']

PERM = np.eye(4)
#np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1],[0,0,1,0]])

if not os.path.exists(SAVEDIR):
    os.makedirs(SAVEDIR)

def estimate_contact_sequence(forces, time, cutoff=4, factor=2):
    index = np.argmax(time > cutoff)
    thresholds = np.mean(forces[:, :index]) / factor
    incontact = np.zeros_like(forces)
    for k in range(forces.shape[0]):
        incontact[k,:] = forces[k,:] > thresholds
    return incontact

data = utils.load(SOURCE)
py_force = np.column_stack(data['pycito_command']['footforce']) 
py_time = np.array(data['pycito_command']['lcm_timestamp'])

measured_force = PERM.dot(np.column_stack(data['leg_control_data']['footforce']))
measured_time = np.array(data['leg_control_data']['lcm_timestamp']) 

# Account for the time delay - reindex time to be the 'input time' instead of the 'output time'
py_time = np.concatenate([np.array([measured_time[0]]), py_time[:-1]], axis=0)

fig, axs = plt.subplots(2,1)
for k, label in enumerate(FOOTLABELS):
    axs[0].plot(py_time, py_force[k,:], linewidth=1.5, label=label)
    axs[1].plot(measured_time, measured_force[k,:], linewidth=1.5, label=label)
axs[0].set_ylabel('Estimated Force')
axs[1].set_ylabel('Measured Force')
axs[1].set_xlabel('Time (s)')
axs[0].legend(frameon=False, ncol=4)

axs[0].set_xlim([PLOTSTART, PLOTSTOP])
axs[1].set_xlim([PLOTSTART, PLOTSTOP])

fig.savefig(os.path.join(SAVEDIR, 'forces' + EXT), dpi=fig.dpi, bbox_inches='tight')
# Get the contact sequences
py_sequence = estimate_contact_sequence(py_force, py_time)
me_sequence = estimate_contact_sequence(measured_force, measured_time)
# Resample the measured sequences
zoh = PiecewisePolynomial.ZeroOrderHold(np.squeeze(measured_time), me_sequence)
me_resampled_sequence = zoh.vector_values(np.squeeze(py_time))

idxstart = np.argmax(np.squeeze(py_time) > PLOTSTART)
idxstop = np.argmax(np.squeeze(py_time) > PLOTSTOP)

fig2, axs2 = plt.subplots(2,1)
for k, data in enumerate([py_sequence, me_resampled_sequence]):
    axs2[k].imshow(data[:, idxstart:idxstop+1], cmap=cm.binary, interpolation=None)
    #yl = axs2[k].get_ylim()
    axs2[k].set_xticks(np.linspace(0, idxstop-idxstart, 5))
    axs2[k].set_xticklabels(np.linspace(PLOTSTART, PLOTSTOP, 5))
    axs2[k].set_yticks([0, 1, 2, 3])
    axs2[k].set_yticklabels(FOOTLABELS)

axs2[0].set_title('Estimated Contact State')
axs2[1].set_title('Measured Contact State')
axs2[1].set_xlabel('Time (s)')

fig2.savefig(os.path.join(SAVEDIR, 'contactsequence' + EXT), dpi=fig2.dpi, bbox_inches='tight')

both = me_resampled_sequence == py_sequence
accuracy = np.sum(both)/both.size

# Use the confusion matrix from scikit-learn
CM = np.zeros((2,2))
for trueval, estval in zip(me_resampled_sequence.ravel(), py_sequence.ravel()):
    CM[int(trueval), int(estval)] += 1

print(CM)

print(f"Estimator has a true positive rate (sensitivity) of {100 * CM[1,1]/(CM[1,1] + CM[1,0]):0.2f}%")
print(f"Estimator has a true negative rate (specificity) of {100 * CM[0,0]/(CM[0,0] + CM[0,1]):0.2f}%")
print(f"Estimator has a total accuracy of {100 * np.sum(np.diag(CM))/np.sum(CM):0.2f}%")
print(f"Confusion matrix:\n{CM}")

plt.show()