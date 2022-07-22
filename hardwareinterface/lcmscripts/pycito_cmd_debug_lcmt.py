"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

class pycito_cmd_debug_lcmt(object):
    __slots__ = ["pitch_pycito", "pitch_mit", "pitch_filtered_mit", "pitch_filtered_pycito", "roll_pycito", "roll_mit"]

    __typenames__ = ["float", "float", "float", "float", "float", "float"]

    __dimensions__ = [None, None, None, None, None, None]

    def __init__(self):
        self.pitch_pycito = 0.0
        self.pitch_mit = 0.0
        self.pitch_filtered_mit = 0.0
        self.pitch_filtered_pycito = 0.0
        self.roll_pycito = 0.0
        self.roll_mit = 0.0

    def encode(self):
        buf = BytesIO()
        buf.write(pycito_cmd_debug_lcmt._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack(">ffffff", self.pitch_pycito, self.pitch_mit, self.pitch_filtered_mit, self.pitch_filtered_pycito, self.roll_pycito, self.roll_mit))

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != pycito_cmd_debug_lcmt._get_packed_fingerprint():
            raise ValueError("Decode error")
        return pycito_cmd_debug_lcmt._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = pycito_cmd_debug_lcmt()
        self.pitch_pycito, self.pitch_mit, self.pitch_filtered_mit, self.pitch_filtered_pycito, self.roll_pycito, self.roll_mit = struct.unpack(">ffffff", buf.read(24))
        return self
    _decode_one = staticmethod(_decode_one)

    def _get_hash_recursive(parents):
        if pycito_cmd_debug_lcmt in parents: return 0
        tmphash = (0x74972ddc3b13104a) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff) + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if pycito_cmd_debug_lcmt._packed_fingerprint is None:
            pycito_cmd_debug_lcmt._packed_fingerprint = struct.pack(">Q", pycito_cmd_debug_lcmt._get_hash_recursive([]))
        return pycito_cmd_debug_lcmt._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

    def get_hash(self):
        """Get the LCM hash of the struct"""
        return struct.unpack(">Q", pycito_cmd_debug_lcmt._get_packed_fingerprint())[0]

