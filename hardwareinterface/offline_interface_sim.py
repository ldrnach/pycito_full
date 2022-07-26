import os
import numpy as np
import matplotlib.pyplot as plt
from a1estimatorinterface import A1ContactEstimationInterface
import pycito.utilities as utils

class Message():
    def __init__(self, q, qd, p, vWorld, rpy, omegaBody, tau_est, time):
        self.q = q
        self.qd = qd
        self.p  =p
        self.vWorld = vWorld
        self.rpy = rpy
        self.omegaBody = omegaBody
        self.tau_est = tau_est
        self.time = time

class Messages():
    def __init__(self, lcm_data):
        self.data = lcm_data
        self.num_msgs = min(len(self.data['state_estimator']['lcm_timestamp']), len(self.data['leg_control_data']['lcm_timestamp']))

    def __iter__(self):
        return MessageIterator(self)

    def get_time(self):
        return self.data['state_estimator']['lcm_timestamp'][:self.num_msgs]


class MessageIterator():
    def __init__(self, messages):
        self.data = messages.data
        self.num_msgs = min(len(messages.data['state_estimator']['lcm_timestamp']), len(messages.data['leg_control_data']['lcm_timestamp']))
        self.current_msg = 0

    def __iter__(self):
        return iter

    def __next__(self):
        if self.current_msg < self.num_msgs:            
            msg = Message(
                q = self.data['leg_control_data']['q'][self.current_msg],
                qd = self.data['leg_control_data']['qd'][self.current_msg],
                p = self.data['state_estimator']['p'][self.current_msg],
                vWorld = self.data['state_estimator']['vWorld'][self.current_msg],
                rpy = self.data['state_estimator']['rpy'][self.current_msg],
                omegaBody = self.data['state_estimator']['omegaBody'][self.current_msg],
                tau_est = self.data['leg_control_data']['tau_est'][self.current_msg],
                time = self.data['state_estimator']['lcm_timestamp'][self.current_msg]
            )
            self.current_msg += 1
            return msg
        else:
            raise StopIteration


# Get the hardware data
SOURCE = os.path.join('hardwareinterface','data','hardware_test_07_21.pkl')
TARGET = os.path.join('hardwareinterface','offline')
TARGETNAME = 'offline_data_07_21.pkl'
if not os.path.exists(TARGET):
    os.makedirs(TARGET)

data = utils.load(SOURCE)
msgs = Messages(data)
t = np.array(msgs.get_time())
interface = A1ContactEstimationInterface()
rpy = []
force = []
for msg in msgs:
    est_rpy, est_force = interface.estimate(msg, msg.time)
    rpy.append(est_rpy)
    force.append(est_force)

# Plot the data
rpy = np.column_stack(rpy)
force = np.column_stack(force)

# Save the data first
output = {'rpy': rpy, 'force': force, 'time': t}
utils.save(os.path.join(TARGET, TARGETNAME), output)

fig, axs = plt.subplots(2,1)
for k, label in enumerate(['Roll','Pitch','Yaw']):
    axs[0].plot(t, rpy[k,:], linewidth=1.5, label=label)
axs[0].set_ylabel('Angle (rad)')
axs[0].legend(frameon=False)

for k, label in enumerate(['FR','FL','RL','RR']):
    axs[1].plot(t, force[k,:], linewidth=1.5, label=label)
axs[1].set_ylabel('Normal Force (N)')
axs[1].set_xlabel('Time (s)')
axs[1].legend(frameon=False)

plt.show()
fig.savefig(os.path.join(TARGET, 'slopefigure_offline_07_21_2022.png'), dpi=fig.dpi)