import concurrent.futures
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
            # self.pitch = self.estimator.estimate(self.msg)
            # command = pycito_cmd_lcmt.pycito_cmd_lcmt()
            # command.pitch = self.pitch
            # print(self.pitch)
            # self._lcm.publish("pycito_command", self.command.encode())

    def _handle_msg(self, channel, data):
        self.msg = full_observer_data_lcmt.full_observer_data_lcmt.decode(data)
        # print("Received message on channel \"%s\"" % channel)
        # print("position = %s" % str(self.msg.p[0]))
        # print("")
        # print("Checked here")

    def _run_estimator(self):
        while True:
            # self._lcm_pitch.handle()
            self.pitch = self.estimator.estimate(self.msg)
            self.command = pycito_cmd_lcmt.pycito_cmd_lcmt()
            self.command.pitch = self.pitch
            print("pitch = %s" % self.pitch)
            self._lcm.publish("pycito_command", self.command.encode())


if __name__ == '__main__':
    h = my_handler()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(h._run_handler)
            executor.submit(h._run_estimator)


    except KeyboardInterrupt:
        pass