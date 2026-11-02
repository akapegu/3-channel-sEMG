# Verbatim supplier filters; GPL-3.0; see LICENSE and UPSTREAM_REVISION.txt.
class NotchFilter:
    """50/60Hz mains notch filter. One instance PER CHANNEL (has internal state)."""

    def __init__(self):
        self.z1_1 = 0.0
        self.z2_1 = 0.0
        self.z1_2 = 0.0
        self.z2_2 = 0.0
        self.x_1 = 0.0
        self.x_2 = 0.0
        self.sampling_rate = 0

    def set_sampling_rate(self, sampling_rate: int):
        self.sampling_rate = sampling_rate

    def process(self, input_val: float, notch_type: int) -> float:
        if not notch_type:
            return input_val

        output = input_val

        if self.sampling_rate == 500:
            if notch_type == 1:  # 48-52 Hz
                self.x_1 = output - (-1.56858163 * self.z1_1) - (0.96424138 * self.z2_1)
                output = 0.96508099 * self.x_1 + -1.56202714 * self.z1_1 + 0.96508099 * self.z2_1
                self.z2_1 = self.z1_1
                self.z1_1 = self.x_1

                self.x_2 = output - (-1.61100358 * self.z1_2) - (0.96592171 * self.z2_2)
                output = 1.0 * self.x_2 + -1.61854514 * self.z1_2 + 1.0 * self.z2_2
                self.z2_2 = self.z1_2
                self.z1_2 = self.x_2

            elif notch_type == 2:  # 58-62 Hz
                self.x_1 = output - (-1.40810535 * self.z1_1) - (0.96443153 * self.z2_1)
                output = 0.96508099 * self.x_1 + (-1.40747202 * self.z1_1) + (0.96508099 * self.z2_1)
                self.z2_1 = self.z1_1
                self.z1_1 = self.x_1

                self.x_2 = output - (-1.45687509 * self.z1_2) - (0.96573127 * self.z2_2)
                output = 1.00000000 * self.x_2 + (-1.45839783 * self.z1_2) + (1.00000000 * self.z2_2)
                self.z2_2 = self.z1_2
                self.z1_2 = self.x_2

        elif self.sampling_rate == 250:
            if notch_type == 1:  # 48-52 Hz
                self.x_1 = output - (-0.53127491 * self.z1_1) - (0.93061518 * self.z2_1)
                output = 0.93137886 * self.x_1 + (-0.57635175 * self.z1_1) + 0.93137886 * self.z2_1
                self.z2_1 = self.z1_1
                self.z1_1 = self.x_1

                self.x_2 = output - (-0.66243374 * self.z1_2) - (0.93214913 * self.z2_2)
                output = 1.00000000 * self.x_2 + (-0.61881558 * self.z1_2) + 1.00000000 * self.z2_2
                self.z2_2 = self.z1_2
                self.z1_2 = self.x_2

            elif notch_type == 2:  # 58-62 Hz
                self.x_1 = output - (-0.05269865 * self.z1_1) - (0.93123336 * self.z2_1)
                output = 0.93137886 * self.x_1 + (-0.11711144 * self.z1_1) + 0.93137886 * self.z2_1
                self.z2_1 = self.z1_1
                self.z1_1 = self.x_1

                self.x_2 = output - (-0.18985625 * self.z1_2) - (0.93153034 * self.z2_2)
                output = 1.00000000 * self.x_2 + (-0.12573985 * self.z1_2) + 1.00000000 * self.z2_2
                self.z2_2 = self.z1_2
                self.z1_2 = self.x_2

        return output


class EXGFilter:
    """ECG/EOG/EEG/EMG bandpass filter bank. One instance PER CHANNEL (has internal state)."""

    def __init__(self):
        self.z1 = 0.0
        self.z2 = 0.0
        self.x1 = 0.0
        self.x2 = 0.0
        self.x3 = 0.0
        self.x4 = 0.0
        self.bits = None
        self.bits_points = 0
        self.sampling_rate = 0

    def set_bits(self, bits: str, sampling_rate: int):
        self.sampling_rate = sampling_rate
        self.bits = bits
        self.bits_points = 2 ** int(bits)

    def process(self, input_val: float, exg_type: int) -> float:
        if not exg_type:
            return input_val

        output = input_val
        ch_data = 0.0

        if self.sampling_rate == 500:
            if exg_type == 4:  # EMG, 70Hz
                self.x4 = output - (-0.82523238 * self.z1) - (0.29463653 * self.z2)
                output = 0.52996723 * self.x4 + -1.05993445 * self.z1 + 0.52996723 * self.z2
                self.z2 = self.z1
                self.z1 = self.x4
                ch_data = output
            # (ECG/EOG/EEG cases omitted here- add back from filters.ts if you need them)

        elif self.sampling_rate == 250:
            if exg_type == 4:  # EMG, 70Hz
                self.x4 = output - 0.22115344 * self.z1 - 0.18023207 * self.z2
                output = 0.23976966 * self.x4 + -0.47953932 * self.z1 + 0.23976966 * self.z2
                self.z2 = self.z1
                self.z1 = self.x4
                ch_data = output

        return ch_data


