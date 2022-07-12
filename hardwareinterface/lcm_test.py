import os, sys
from lcm import LCM
from interface_test import EstimationInterfaceTest
from lcmscripts import state_estimator_lcmt, pycito_cmd_lcmt


class my_handler:
    def __init__(self):
        self._lcm = LCM()
        self._lcm.subscribe("state_estimator", self._handle_msg)

    def _run_handler(self):
        while True:
            self._lcm.handle()

    def _handle_msg(self, channel, data):
        msg = state_estimator_lcmt.state_estimator_lcmt.decode(data)
        # print("Received message on channel \"%s\"" % channel)
        # print("   position   = %s" % str(msg.p))
        # print("")
        # print("Checked here")

        command = pycito_cmd_lcmt.pycito_cmd_lcmt()
        command.pitch = msg.p[0]
        self._lcm.publish("pycito_command", command.encode())

if __name__ == '__main__':
    h = my_handler()
    try:
        h._run_handler()
    except KeyboardInterrupt:
        pass