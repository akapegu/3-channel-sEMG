"""Preserve every LIS3DH frame on its independent host-arrival timeline."""
import asyncio
import csv
import json
import struct
import time

IMU = '5a153fa9-7be0-400c-8ef8-d84502b31c4d'


def decode_accelerometer(payload):
    if len(payload) % 7:
        raise ValueError(f'Expected 7-byte accelerometer frames; got {len(payload)} bytes')
    # Empty notifications can occur when the firmware FIFO is empty.
    return list(struct.iter_unpack('>Bhhh', payload))


class AccelerometerCapture:
    def __init__(self):
        self.rows = []
        self.packets = []
        self.first = asyncio.Event()
        self.previous = None
        self.gaps = self.duplicates = self.malformed = self.empty = 0

    def notify(self, _, payload):
        mono, unix = time.perf_counter_ns(), time.time_ns()
        try:
            frames = decode_accelerometer(payload)
        except ValueError:
            self.malformed += 1
            return
        if not frames:
            self.empty += 1
            return
        packet = len(self.packets)
        self.packets.append((mono, len(frames)))
        for within, (counter, ax, ay, az) in enumerate(frames):
            step = None if self.previous is None else (counter-self.previous) % 256
            if step == 0:
                self.duplicates += 1
            elif step is not None and step != 1:
                self.gaps += step-1
            self.previous = counter
            self.rows.append((len(self.rows), packet, within, mono, unix, counter, ax, ay, az))
        self.first.set()

    def save(self, output, device):
        path = output.with_suffix('.imu.csv')
        with path.open('x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_index', 'packet_index', 'index_in_packet',
                             'host_monotonic_ns', 'host_unix_ns', 'counter',
                             'accel_x_counts', 'accel_y_counts', 'accel_z_counts'])
            writer.writerows(self.rows)
        span = (self.packets[-1][0]-self.packets[0][0])/1e9 if len(self.packets)>1 else 0
        stats = {'device': device, 'samples_per_axis': len(self.rows),
                 'packets': len(self.packets), 'host_arrival_span_seconds': span,
                 'estimated_received_sps_per_axis': (len(self.rows)-self.packets[0][1])/span if span else None,
                 'counter_gap_missing_modulo_256': self.gaps,
                 'duplicate_counter_transitions': self.duplicates,
                 'malformed_packets': self.malformed, 'empty_notifications': self.empty,
                 'axis_ranges': [[min(r[i] for r in self.rows), max(r[i] for r in self.rows)]
                                 for i in (6,7,8)] if self.rows else None,
                 'note': 'All frames retained, signed big-endian int16 counts. No conversion to g, '
                         'filtering or resampling. Host arrival times are not acquisition times. '
                         'Independent accelerometer counters; modulo-256 ambiguity applies. '
                         'Initial FIFO backlog may affect throughput estimate.'}
        with output.with_suffix('.imu.json').open('x') as f:
            json.dump(stats, f, indent=2)
        print('Accelerometer:', json.dumps(stats, indent=2))
