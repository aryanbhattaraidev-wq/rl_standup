"""UDP client for the C++ simulation (protocol described in sim/Protocol.h)."""
import socket
import struct

import torch

from .config import IP, PY_PORT, SIM_PORT, num_envs, state_dim, packet_dim


class Data:
    """
    Lockstep client: one action packet in, one state packet out.
    n_envs = 25 with handshake -100 for training, n_envs = 1 with handshake -101 for solo testing.
    """
    def __init__(self, ip=IP, recv_port=PY_PORT, send_port=SIM_PORT, n_envs=num_envs,
                 handshake=-100.0, timeout=10.0):
        self.n_envs = n_envs
        self.handshake = handshake
        self.timeout = timeout
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.bind((ip, recv_port))
        # Fail loudly instead of hanging forever if the simulation is not running
        self.recv_sock.settimeout(timeout)

        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_addr = (ip, send_port)

    def receive_state(self):
        expected = self.n_envs * packet_dim
        try:
            data, _ = self.recv_sock.recvfrom(65536)
        except socket.timeout:
            raise RuntimeError("Timed out waiting for the simulation. Is rl_standup.exe running?")
        num_floats = len(data) // 4
        if num_floats != expected:
            raise RuntimeError(
                f"\n[Mode Mismatch] Received {num_floats} floats but expected {expected} "
                f"({self.n_envs} bodies).\n"
                "--> Restart rl_standup.exe (it keeps the mode of whichever script connected first)."
            )
        raw = torch.frombuffer(bytearray(data), dtype=torch.float32)
        return raw.view(self.n_envs, packet_dim).clone()

    def send(self, floats):
        self.send_sock.sendto(struct.pack(f"{len(floats)}f", *floats), self.send_addr)

    def drain(self):
        """Drop stale packets left over from a previous run so states and actions stay in lockstep."""
        self.recv_sock.settimeout(0.0)
        try:
            while True:
                self.recv_sock.recvfrom(65536)
        except (BlockingIOError, socket.timeout):
            pass
        finally:
            self.recv_sock.settimeout(self.timeout)

    def get_state(self, extra=(), attempts=5):
        # Handshake: [h, h, extra...] resets all bodies and requests their initial state.
        # UDP has no retry, so if the sim was still starting up and missed the packet, send it again.
        for attempt in range(attempts):
            self.drain()
            self.send([self.handshake, self.handshake, *extra])
            self.recv_sock.settimeout(2.0)
            try:
                return self._split(self.receive_state())
            except RuntimeError as e:
                if "Timed out" not in str(e) or attempt == attempts - 1:
                    raise
            finally:
                self.recv_sock.settimeout(self.timeout)

    def step(self, actions_tensor, extra=()):
        # actions_tensor: shape (n_envs, action_dim); `extra` floats are appended to the packet
        flat = actions_tensor.contiguous().view(-1).tolist()
        self.send(flat + list(extra))
        return self._split(self.receive_state())

    @staticmethod
    def _split(raw):
        return raw[:, :state_dim], raw[:, state_dim]
