"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

class leg_control_command_lcmt(object):
    __slots__ = ["tau_ff", "f_ff", "q_des", "qd_des", "p_des", "v_des", "kp_cartesian", "kd_cartesian", "kp_joint", "kd_joint"]

    __typenames__ = ["float", "float", "float", "float", "float", "float", "float", "float", "float", "float"]

    __dimensions__ = [[12], [12], [12], [12], [12], [12], [12], [12], [12], [12]]

    def __init__(self):
        self.tau_ff = [ 0.0 for dim0 in range(12) ]
        self.f_ff = [ 0.0 for dim0 in range(12) ]
        self.q_des = [ 0.0 for dim0 in range(12) ]
        self.qd_des = [ 0.0 for dim0 in range(12) ]
        self.p_des = [ 0.0 for dim0 in range(12) ]
        self.v_des = [ 0.0 for dim0 in range(12) ]
        self.kp_cartesian = [ 0.0 for dim0 in range(12) ]
        self.kd_cartesian = [ 0.0 for dim0 in range(12) ]
        self.kp_joint = [ 0.0 for dim0 in range(12) ]
        self.kd_joint = [ 0.0 for dim0 in range(12) ]

    def encode(self):
        buf = BytesIO()
        buf.write(leg_control_command_lcmt._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack('>12f', *self.tau_ff[:12]))
        buf.write(struct.pack('>12f', *self.f_ff[:12]))
        buf.write(struct.pack('>12f', *self.q_des[:12]))
        buf.write(struct.pack('>12f', *self.qd_des[:12]))
        buf.write(struct.pack('>12f', *self.p_des[:12]))
        buf.write(struct.pack('>12f', *self.v_des[:12]))
        buf.write(struct.pack('>12f', *self.kp_cartesian[:12]))
        buf.write(struct.pack('>12f', *self.kd_cartesian[:12]))
        buf.write(struct.pack('>12f', *self.kp_joint[:12]))
        buf.write(struct.pack('>12f', *self.kd_joint[:12]))

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != leg_control_command_lcmt._get_packed_fingerprint():
            raise ValueError("Decode error")
        return leg_control_command_lcmt._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = leg_control_command_lcmt()
        self.tau_ff = struct.unpack('>12f', buf.read(48))
        self.f_ff = struct.unpack('>12f', buf.read(48))
        self.q_des = struct.unpack('>12f', buf.read(48))
        self.qd_des = struct.unpack('>12f', buf.read(48))
        self.p_des = struct.unpack('>12f', buf.read(48))
        self.v_des = struct.unpack('>12f', buf.read(48))
        self.kp_cartesian = struct.unpack('>12f', buf.read(48))
        self.kd_cartesian = struct.unpack('>12f', buf.read(48))
        self.kp_joint = struct.unpack('>12f', buf.read(48))
        self.kd_joint = struct.unpack('>12f', buf.read(48))
        return self
    _decode_one = staticmethod(_decode_one)

    def _get_hash_recursive(parents):
        if leg_control_command_lcmt in parents: return 0
        tmphash = (0x93bfbc95a989bb67) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff) + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if leg_control_command_lcmt._packed_fingerprint is None:
            leg_control_command_lcmt._packed_fingerprint = struct.pack(">Q", leg_control_command_lcmt._get_hash_recursive([]))
        return leg_control_command_lcmt._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

    def get_hash(self):
        """Get the LCM hash of the struct"""
        return struct.unpack(">Q", leg_control_command_lcmt._get_packed_fingerprint())[0]

