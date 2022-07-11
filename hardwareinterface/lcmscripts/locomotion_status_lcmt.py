"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

class locomotion_status_lcmt(object):
    __slots__ = ["operating_mode", "current_fsm"]

    __typenames__ = ["string", "string"]

    __dimensions__ = [None, None]

    def __init__(self):
        self.operating_mode = ""
        self.current_fsm = ""

    def encode(self):
        buf = BytesIO()
        buf.write(locomotion_status_lcmt._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        __operating_mode_encoded = self.operating_mode.encode('utf-8')
        buf.write(struct.pack('>I', len(__operating_mode_encoded)+1))
        buf.write(__operating_mode_encoded)
        buf.write(b"\0")
        __current_fsm_encoded = self.current_fsm.encode('utf-8')
        buf.write(struct.pack('>I', len(__current_fsm_encoded)+1))
        buf.write(__current_fsm_encoded)
        buf.write(b"\0")

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != locomotion_status_lcmt._get_packed_fingerprint():
            raise ValueError("Decode error")
        return locomotion_status_lcmt._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = locomotion_status_lcmt()
        __operating_mode_len = struct.unpack('>I', buf.read(4))[0]
        self.operating_mode = buf.read(__operating_mode_len)[:-1].decode('utf-8', 'replace')
        __current_fsm_len = struct.unpack('>I', buf.read(4))[0]
        self.current_fsm = buf.read(__current_fsm_len)[:-1].decode('utf-8', 'replace')
        return self
    _decode_one = staticmethod(_decode_one)

    def _get_hash_recursive(parents):
        if locomotion_status_lcmt in parents: return 0
        tmphash = (0x756874a1308a6b0d) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff) + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if locomotion_status_lcmt._packed_fingerprint is None:
            locomotion_status_lcmt._packed_fingerprint = struct.pack(">Q", locomotion_status_lcmt._get_hash_recursive([]))
        return locomotion_status_lcmt._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

    def get_hash(self):
        """Get the LCM hash of the struct"""
        return struct.unpack(">Q", locomotion_status_lcmt._get_packed_fingerprint())[0]

