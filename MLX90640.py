from collections import namedtuple
import numpy as np
import serial
import sys
import time

REFRESH_RATE_Hz = 2
CHESS_PATTERN = True

BAUD_RATE = 115200
ROWS = 24
COLS = 32
V_REF = 3.3
T_REF = 25
ZERO = -273.15
EMISSIVITY = 0.95

CHESS_1 = np.indices((ROWS, COLS)).sum(axis=0) % 2
CHESS_0 = 1 - CHESS_1
CHESS = [CHESS_0, CHESS_1]
INTERLEAVED_0 = np.tile([1, 0], (32, 12)).T
INTERLEAVED_1 = 1 - INTERLEAVED_0
INTERLEAVED = [INTERLEAVED_0, INTERLEAVED_1]
CONVERSION = np.tile([[0, -1, 0, 1], [0, 1, 0, -1]], (12, 8))

EEPROM = namedtuple('EEPROM', [
    'occ_scale_rem', 'occ_scale_col', 'occ_scale_row', 'k_ptat',
    'offset_average', 'occ_row', 'occ_col',
    'acc_scale_rem', 'acc_scale_col', 'acc_scale_row', 'alpha_scale',
    'alpha_ref', 'acc_row', 'acc_col',
    'gain', 'ptat_25', 'k_t_ptat', 'k_v_ptat', 'v_dd_25',
    'k_v_dd', 'k_v_re_ce', 'k_v_ro_ce', 'k_v_re_co', 'k_v_ro_co',
    'il_chess_c1', 'il_chess_c2', 'il_chess_c3',
    'k_ta_re_co', 'k_ta_ro_co', 'k_ta_re_ce', 'k_ta_ro_ce',
    'k_ta_scale_2', 'k_ta_scale_1', 'k_v_scale', 'resolution_cal',
    'alpha_cp_sp_0', 'alpha_cp_sp_1', 'offset_cp_sp_0', 'offset_cp_sp_1',
    'k_ta_cp', 'k_v_cp', 'tgc',
    'k_s_ta', 'k_s_to_scale', 'k_s_to1', 'k_s_to2', 'k_s_to3', 'k_s_to4',
    'ct_step', 'ct3', 'ct4',
    'pixel_outlier', 'pixel_k_ta', 'pixel_alpha', 'pixel_offset'
])

RAM = namedtuple('RAM', [
    'pixels',
    'v_be', 'cp_sp0', 'gain',
    'v_ptat', 'cp_sp1', 'v_dd',
])

Registers = namedtuple('Registers', [
    'last_page', 'new_page_available', 'overwrite_enabled',
    'pages_enabled', 'page_hold_enabled',
    'select_page_enabled', 'selected_page',
    'refresh_rate', 'adc_resolution', 'chess_pattern_enabled',
    'fast_mode_disabled', 'i2c_threshold_mode', 'sda_current_limit_disabled',
])

class MLX90640:
    def __init__(self, port, baudrate=BAUD_RATE):
        self.port = serial.Serial(port, baudrate)
        time.sleep(2) # wait for arduino to restart
        self.mem = {} # addr => value
        self.read_eeprom()
        self.read_ram()
        self.read_registers()

    def read(self, addr):
        self.port.write(b'\x00' + addr.to_bytes(2, 'big'))
        return int.from_bytes(self.port.read(2), 'big')

    def write(self, addr, value, size=16, offset=0):
        if size != 16 or offset != 0:
            prev = self.read(addr)
            mask = ((1 << size) - 1) << offset
            value = (prev & ~mask) | (value << offset)
        self.port.write(b'\x01' + addr.to_bytes(2, 'big') + value.to_bytes(2, 'big'))

    def read_many(self, addr, words):
        self.port.write(b'\x02' + addr.to_bytes(2, 'big') + words.to_bytes(2, 'big'))
        return [int.from_bytes(self.port.read(2), 'big') for i in range(words)]

    def read_page(self, chess, page):
        cmd = 3 if chess else 4
        self.port.write(cmd.to_bytes(1, 'big') + page.to_bytes(1, 'big'))
        return [int.from_bytes(self.port.read(2), 'big') for i in range(ROWS * COLS // 2)]

    def read_into_mem(self, addr, words):
        data = self.read_many(addr, words)
        for i, v in enumerate(data):
            self.mem[addr + i] = v

    def read_ram(self):
        self.read_into_mem(0x0400, 0x0340)
        self.ram = self.decode_ram()

    def read_eeprom(self):
        self.read_into_mem(0x2400, 0x0340)
        self.ee = self.decode_eeprom()

    def read_registers(self):
        for addr in [0x8000, 0x800d, 0x800f]:
            self.mem[addr] = self.read(addr)
        self.reg = self.decode_registers()

    def read_last_page(self):
        chess = self.reg.chess_pattern_enabled
        page = self.reg.last_page
        data = self.read_page(chess, page)
        self.read_into_mem(0x0700, 0x0030)
        for y in range(ROWS):
            for x in range(COLS):
                p = (x ^ y) & 1 if chess else y & 1
                if p == page:
                    addr = 0x0400 + y * COLS + x
                    self.mem[addr] = data.pop(0)
        self.ram = self.decode_ram()

    def set_refresh_rate_Hz(self, rr_Hz):
        rr = np.log(rr_Hz) / np.log(2) + 1
        rr = int(round(np.clip(rr, 0, 7)))
        self.write(0x800d, rr, 3, 7)

    def clear_new_page_available(self):
        self.write(0x8000, 0, 1, 3)

    def enable_chess_pattern(self, enable):
        value = int(bool(enable))
        self.write(0x800d, value, 1, 12)

    def get_next_frame(self):
        while True:
            self.read_registers()
            if self.reg.new_page_available:
                break
            time.sleep(0.01)
        self.clear_new_page_available()
        self.read_last_page()
        return self.get_t_o()

    def unsigned(self, addr, bits=16, offset=0):
        return (self.mem[addr] >> offset) & ((1 << bits) - 1)

    def signed(self, addr, bits=16, offset=0):
        value = self.unsigned(addr, bits, offset)
        if value > (1 << (bits - 1)) - 1:
            value -= 1 << bits
        return value

    def decode_registers(self):
        unsigned = self.unsigned
        return Registers(
            last_page = unsigned(0x8000, 3, 0),
            new_page_available = unsigned(0x8000, 1, 3),
            overwrite_enabled = unsigned(0x8000, 1, 4),
            pages_enabled = unsigned(0x800d, 1, 0),
            page_hold_enabled = unsigned(0x800d, 1, 2),
            select_page_enabled = unsigned(0x800d, 1, 3),
            selected_page = unsigned(0x800d, 3, 4),
            refresh_rate = 2 ** (unsigned(0x800d, 3, 7) - 1),
            adc_resolution = unsigned(0x800d, 2, 10) + 16,
            chess_pattern_enabled = unsigned(0x800d, 1, 12),
            fast_mode_disabled = unsigned(0x800f, 1, 0),
            i2c_threshold_mode = unsigned(0x800f, 1, 1),
            sda_current_limit_disabled = unsigned(0x800f, 1, 2))

    def decode_ram(self):
        signed = self.signed
        return RAM(
            pixels = [signed(i) for i in range(0x0400, 0x0700)],
            v_be = signed(0x0700),
            cp_sp0 = signed(0x0708),
            gain = signed(0x070a),
            v_ptat = signed(0x0720),
            cp_sp1 = signed(0x0728),
            v_dd = signed(0x072a))

    def decode_eeprom(self):
        signed = self.signed
        unsigned = self.unsigned

        occ_scale_rem = unsigned(0x2410, 4, 0)
        occ_scale_col = unsigned(0x2410, 4, 4)
        occ_scale_row = unsigned(0x2410, 4, 8)
        k_ptat = unsigned(0x2410, 4, 12) / 4 + 8
        offset_average = signed(0x2411)
        occ_row = tuple(signed(0x2412 + i // 4, 4, i % 4 * 4) for i in range(ROWS))
        occ_col = tuple(signed(0x2418 + i // 4, 4, i % 4 * 4) for i in range(COLS))

        acc_scale_rem = unsigned(0x2420, 4, 0)
        acc_scale_col = unsigned(0x2420, 4, 4)
        acc_scale_row = unsigned(0x2420, 4, 8)
        alpha_scale = unsigned(0x2420, 4, 12) + 30
        alpha_ref = unsigned(0x2421)
        acc_row = tuple(signed(0x2422 + i // 4, 4, i % 4 * 4) for i in range(ROWS))
        acc_col = tuple(signed(0x2428 + i // 4, 4, i % 4 * 4) for i in range(COLS))

        gain = signed(0x2430)
        ptat_25 = signed(0x2431)
        k_t_ptat = signed(0x2432, 10, 0) / 8
        k_v_ptat = signed(0x2432, 6, 10) / 4096
        v_dd_25 = (unsigned(0x2433, 8, 0) - 256) * 32 - 8192
        k_v_dd = signed(0x2433, 8, 8) * 32

        k_v_re_ce = signed(0x2434, 4, 0)
        k_v_ro_ce = signed(0x2434, 4, 4)
        k_v_re_co = signed(0x2434, 4, 8)
        k_v_ro_co = signed(0x2434, 4, 12)

        il_chess_c1 = signed(0x2435, 6, 0) / 16
        il_chess_c2 = signed(0x2435, 5, 6) / 2
        il_chess_c3 = signed(0x2435, 5, 11) / 8

        k_ta_re_co = signed(0x2436, 8, 0)
        k_ta_ro_co = signed(0x2436, 8, 8)
        k_ta_re_ce = signed(0x2437, 8, 0)
        k_ta_ro_ce = signed(0x2437, 8, 8)

        k_ta_scale_2 = unsigned(0x2438, 4, 0)
        k_ta_scale_1 = unsigned(0x2438, 4, 4) + 8
        k_v_scale = unsigned(0x2438, 4, 8)
        resolution_cal = unsigned(0x2438, 2, 12)

        alpha_cp_scale = unsigned(0x2420, 4, 12) + 27
        alpha_cp_sp_0 = signed(0x2439, 10, 0) / (1 << alpha_cp_scale)
        alpha_cp_sp_1 = alpha_cp_sp_0 * (1 + signed(0x2439, 6, 10) / 128)

        offset_cp_sp_0 = signed(0x243a, 10, 0)
        offset_cp_sp_1 = offset_cp_sp_0 + signed(0x243a, 6, 10)

        k_ta_cp = signed(0x243b, 8, 0) / (1 << k_ta_scale_1)
        k_v_cp = signed(0x243b, 8, 8) / (1 << k_v_scale)

        tgc = signed(0x243c, 8, 0) / 32
        k_s_ta = signed(0x243c, 8, 8) / 8192

        k_s_to_scale = unsigned(0x243f, 4, 0) + 8
        k_s_to1 = signed(0x243d, 8, 0) / (1 << k_s_to_scale)
        k_s_to2 = signed(0x243d, 8, 8) / (1 << k_s_to_scale)
        k_s_to3 = signed(0x243e, 8, 0) / (1 << k_s_to_scale)
        k_s_to4 = signed(0x243e, 8, 8) / (1 << k_s_to_scale)
        ct_step = unsigned(0x243f, 2, 12) * 10
        ct3 = unsigned(0x243f, 4, 4) * ct_step
        ct4 = unsigned(0x243f, 4, 8) * ct_step + ct3

        pixel_outlier = tuple(unsigned(0x2440 + i, 1, 0) for i in range(768))
        pixel_k_ta = tuple(signed(0x2440 + i, 3, 1) for i in range(768))
        pixel_alpha = tuple(signed(0x2440 + i, 6, 4) for i in range(768))
        pixel_offset = tuple(signed(0x2440 + i, 6, 10) for i in range(768))

        return EEPROM(
            occ_scale_rem, occ_scale_col, occ_scale_row, k_ptat,
            offset_average, occ_row, occ_col,
            acc_scale_rem, acc_scale_col, acc_scale_row, alpha_scale,
            alpha_ref, acc_row, acc_col,
            gain, ptat_25, k_t_ptat, k_v_ptat, v_dd_25,
            k_v_dd, k_v_re_ce, k_v_ro_ce, k_v_re_co, k_v_ro_co,
            il_chess_c1, il_chess_c2, il_chess_c3,
            k_ta_re_co, k_ta_ro_co, k_ta_re_ce, k_ta_ro_ce,
            k_ta_scale_2, k_ta_scale_1, k_v_scale, resolution_cal,
            alpha_cp_sp_0, alpha_cp_sp_1, offset_cp_sp_0, offset_cp_sp_1,
            k_ta_cp, k_v_cp, tgc,
            k_s_ta, k_s_to_scale, k_s_to1, k_s_to2, k_s_to3, k_s_to4,
            ct_step, ct3, ct4,
            pixel_outlier, pixel_k_ta, pixel_alpha, pixel_offset)

    def get_page(self):
        return self.reg.last_page

    def get_v_dd(self):
        ee, ram, reg = self.ee, self.ram, self.reg
        resolution_corr = (1 << ee.resolution_cal) / (1 << (reg.adc_resolution - 16))
        return (resolution_corr * ram.v_dd - ee.v_dd_25) / ee.k_v_dd + V_REF

    def get_t_a(self):
        ee, ram = self.ee, self.ram
        d_v = (ram.v_dd - ee.v_dd_25) / ee.k_v_dd
        v_ptat_art = (ram.v_ptat / (ram.v_ptat * ee.k_ptat + ram.v_be)) * (1 << 18)
        return (v_ptat_art / (1 + ee.k_v_ptat * d_v) - ee.ptat_25) / ee.k_t_ptat + T_REF

    def get_t_o(self):
        ee, ram, reg = self.ee, self.ram, self.reg

        occ_row = np.repeat([ee.occ_row], COLS, axis=0).T
        occ_col = np.repeat([ee.occ_col], ROWS, axis=0)
        acc_row = np.repeat([ee.acc_row], COLS, axis=0).T
        acc_col = np.repeat([ee.acc_col], ROWS, axis=0)

        k_ta_rc = np.zeros((ROWS, COLS))
        k_ta_rc[0::2,0::2] = ee.k_ta_ro_co
        k_ta_rc[1::2,0::2] = ee.k_ta_re_co
        k_ta_rc[0::2,1::2] = ee.k_ta_ro_ce
        k_ta_rc[1::2,1::2] = ee.k_ta_re_ce

        k_v_rc = np.zeros((ROWS, COLS))
        k_v_rc[0::2,0::2] = ee.k_v_ro_co
        k_v_rc[1::2,0::2] = ee.k_v_re_co
        k_v_rc[0::2,1::2] = ee.k_v_ro_ce
        k_v_rc[1::2,1::2] = ee.k_v_re_ce

        pixel_k_ta = np.array(ee.pixel_k_ta).reshape((ROWS, COLS))
        pixel_alpha = np.array(ee.pixel_alpha).reshape((ROWS, COLS))
        pixel_offset = np.array(ee.pixel_offset).reshape((ROWS, COLS))
        pixels = np.array(ram.pixels).reshape((ROWS, COLS))

        # device voltage and ambient temperature
        v_dd = self.get_v_dd()
        t_a = self.get_t_a()

        # gain
        k_gain = ee.gain / ram.gain
        pixels = pixels * k_gain

        # offset
        conversion = ee.il_chess_c3 * INTERLEAVED[reg.last_page] - ee.il_chess_c2 * CONVERSION
        if reg.chess_pattern_enabled:
            conversion = 0

        pixel_offset_ref = ee.offset_average + occ_row * (1 << ee.occ_scale_row) + occ_col * (1 << ee.occ_scale_col) + pixel_offset * (1 << ee.occ_scale_rem)
        k_ta = (k_ta_rc + pixel_k_ta * (1 << ee.k_ta_scale_2)) / (1 << ee.k_ta_scale_1)
        k_v = k_v_rc / (1 << ee.k_v_scale)
        k = (1 + k_ta * (t_a - T_REF)) * (1 + k_v * (v_dd - V_REF))
        pixels = pixels + conversion - pixel_offset_ref * k

        # emissivity
        pixels = pixels / EMISSIVITY

        # compensation pixels
        il_offset = ee.il_chess_c1
        if reg.chess_pattern_enabled:
            il_offset = 0
        k = (1 + ee.k_ta_cp * (t_a - T_REF)) * (1 + ee.k_v_cp * (v_dd - V_REF))
        pix_os_cp_sp0 = (ram.cp_sp0 * k_gain) - ee.offset_cp_sp_0 * k
        pix_os_cp_sp1 = (ram.cp_sp1 * k_gain) - (ee.offset_cp_sp_1 + il_offset) * k
        pixels = pixels - ee.tgc * (CHESS_0 * pix_os_cp_sp0 + CHESS_1 * pix_os_cp_sp1)

        # sensitivity
        alpha = (ee.alpha_ref + acc_row * (1 << ee.acc_scale_row) + acc_col * (1 << ee.acc_scale_col) + pixel_alpha * (1 << ee.acc_scale_rem)) / (1 << ee.alpha_scale)
        alpha = (alpha - ee.tgc * (CHESS_0 * ee.alpha_cp_sp_0 + CHESS_1 * ee.alpha_cp_sp_1)) * (1 + ee.k_s_ta * (t_a - T_REF))

        # temperature
        t_r = t_a - 8 # NOTE: this is a fudged number meant to represent room temperature, which is probably lower than the device's measured ambient temperature
        t_a_k4 = (t_a - ZERO) ** 4
        t_r_k4 = (t_r - ZERO) ** 4
        t_ar = t_r_k4 - (t_r_k4 - t_a_k4) / EMISSIVITY
        s_x = ee.k_s_to2 * (alpha ** 3 * pixels + alpha ** 4 * t_ar) ** (1 / 4)
        t_o = (pixels / (alpha * (1 + ee.k_s_to2 * ZERO) + s_x) + t_ar) ** (1 / 4) + ZERO

        # extended temperature ranges
        ct = np.array([-40, 0, ee.ct3, ee.ct4])
        k_s_to = np.array([ee.k_s_to1, ee.k_s_to2, ee.k_s_to3, ee.k_s_to4])
        corr = np.array([
            1 / (1 + ee.k_s_to1 * 40),
            1,
            1 + ee.k_s_to2 * ee.ct3,
            (1 + ee.k_s_to2 * ee.ct3) * (1 + ee.k_s_to3 * (ee.ct4 - ee.ct3)),
        ])

        a_idx = np.full((ROWS, COLS), 3, dtype=int)
        a_idx[t_o < ct[1]] = 0
        a_idx[t_o < ct[2]] = 1
        a_idx[t_o < ct[3]] = 2

        a_ct = ct[a_idx]
        a_k_s_to = k_s_to[a_idx]
        a_corr = corr[a_idx]

        t_o = (pixels / (alpha * a_corr * (1 + a_k_s_to * (t_o - a_ct))) + t_ar) ** (1 / 4) + ZERO

        return t_o

def main():
    import matplotlib.pyplot as plt

    args = sys.argv[1:]
    if len(args) != 1:
        print('Usage: python thermal.py port')
        return

    mlx = MLX90640(args[0])
    mlx.enable_chess_pattern(CHESS_PATTERN)
    mlx.set_refresh_rate_Hz(REFRESH_RATE_Hz)

    fig, ax = plt.subplots()
    im = ax.imshow(np.zeros((ROWS, COLS)), cmap='viridis', interpolation='bicubic')
    plt.colorbar(im, label='Temperature (°C)')
    plt.show(block=False)

    while True:
        a = mlx.get_next_frame()
        a = np.fliplr(a)

        page = mlx.get_page()
        v_dd = mlx.get_v_dd()
        t_a = mlx.get_t_a()
        t_o = np.mean(a)

        print('P=%d, V=%.3f, T_a=%.3f °C, T_o=%.3f °C' % (page, v_dd, t_a, t_o))

        im.set_data(a)
        im.set_clim(a.min(), a.max())

        fig.canvas.draw()
        plt.pause(0.001)

if __name__ == '__main__':
    main()
