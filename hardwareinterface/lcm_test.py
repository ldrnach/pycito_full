import os, sys
from lcm import LCM
from a1estimatorinterface import A1ContactEstimationInterface
from lcmscripts import full_observer_data_lcmt, pycito_cmd_lcmt


class my_handler:
    def __init__(self):
        self._lcm = LCM()
        self._lcm.subscribe("full_observer_data", self._handle_msg)
        self.estimator = A1ContactEstimationInterface()

    def _run_handler(self):
        while True:
            self._lcm.handle()
            command = pycito_cmd_lcmt.pycito_cmd_lcmt()
            command.pitch = self.pitch
            print(self.pitch)
            self._lcm.publish("pycito_command", command.encode())

    def _handle_msg(self, channel, data):
        msg = full_observer_data_lcmt.full_observer_data_lcmt.decode(data)
        # print("Received message on channel \"%s\"" % channel)
        # print("   position   = %s" % str(msg.p))
        # print("")
        # print("Checked here")
        self.pitch = self.estimator.estimate(msg)


if __name__ == '__main__':
    h = my_handler()
    try:
        h._run_handler()
    except KeyboardInterrupt:
        pass