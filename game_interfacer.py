from PIL import Image
from PIL import ImageFilter
import win32process
import numpy as np
import math
import win32api
import win32gui
import win32ui
import win32con
from ctypes import windll
import time
import ctypes
from ctypes import *
from ctypes.wintypes import *
from functools import partial
dlluser32 = ctypes.cdll.LoadLibrary('user32.dll')
import socket
import sys
import array
import time
import random
import struct
"""
Simple unicode keyboard automation for windows
Based off of http://stackoverflow.com/questions/11906925/python-simulate-keydown
"""

import ctypes
import sys

import copy

gdi = ctypes.WinDLL("C:\\Windows\\System32\\gdi32.dll")

LONG = ctypes.c_long
DWORD = ctypes.c_ulong
ULONG_PTR = ctypes.POINTER(DWORD)
WORD = ctypes.c_ushort

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_UNICODE = 0x0004

class MOUSEINPUT(ctypes.Structure):
    _fields_ = (('dx', LONG),
                ('dy', LONG),
                ('mouseData', DWORD),
                ('dwFlags', DWORD),
                ('time', DWORD),
                ('dwExtraInfo', ULONG_PTR))


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (('wVk', WORD),
                ('wScan', WORD),
                ('dwFlags', DWORD),
                ('time', DWORD),
                ('dwExtraInfo', ULONG_PTR))


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (('uMsg', DWORD),
                ('wParamL', WORD),
                ('wParamH', WORD))


class _INPUTunion(ctypes.Union):
    _fields_ = (('mi', MOUSEINPUT),
                ('ki', KEYBDINPUT),
                ('hi', HARDWAREINPUT))


class INPUT(ctypes.Structure):
    _fields_ = (('type', DWORD),
                ('union', _INPUTunion))

                
class BITMAPINFOHEADER(Structure):
    _fields_ = [
        ('biSize', DWORD),
        ('biWidth', LONG),
        ('biHeight', LONG),
        ('biPlanes', WORD),
        ('biBitCount', WORD),
        ('biCompression', DWORD),
        ('biSizeImage', DWORD),
        ('biXPelsPerMeter', LONG),
        ('biYPelsPerMeter', LONG),
        ('biClrUsed', DWORD),
        ('biClrImportant', DWORD),
        ]

    def __init__(self, w, h):
        self.biSize = sizeof(self)
        self.biWidth = w
        self.biHeight = h
        self.biPlanes = 1
        self.biBitCount = 24
        self.biSizeImage = w * h * 3


class Controller:
    def __init__(self):
        self.down_scancodes = set()
        self.key_enum = {
            0: [0, 0],      #stationary
            1: [0, 1],      #up
            2: [1, 1],      #up right
            3: [1, 0],      #right
            4: [1, -1],     #down right
            5: [0, -1],     #down
            6: [-1, -1],    #down left
            7: [-1, 0],     #left
            8: [-1, 1]      #up left
        }
        
        self.scancode_symbols = {
            0x4B: "<",
            0x4D: ">",
            0x50: "v",
            0x48: "^",
            0x2A: "focus",
            0x2C: "shift",
            0x2D: "bomb"
        }
        
    def debug_keys(self, message, scancodes):
        keys_msg = "".join(list(map(lambda x: self.scancode_symbols[x], scancodes)))
        print(message + " " + keys_msg)
            
    def send_keyup_events(self, new_keys):
        diff = self.down_scancodes.difference(new_keys)
        keyboard_inputs = list(map(lambda x: INPUT(INPUT_KEYBOARD, _INPUTunion(ki = KEYBDINPUT(0, x, KEYEVENTF_KEYUP | KEYEVENTF_SCANCODE, 0, None))), diff))
        self.send_inputs(keyboard_inputs)
        
    def send_keydown_events(self, new_keys):
        diff = new_keys.difference(self.down_scancodes)
        keyboard_inputs = list(map(lambda x: INPUT(INPUT_KEYBOARD, _INPUTunion(ki = KEYBDINPUT(0, x, KEYEVENTF_SCANCODE, 0, None))), diff))
        self.send_inputs(keyboard_inputs)
        
    def send_keys(self, new_scancodes):
        self.send_keyup_events(new_scancodes)
        self.send_keydown_events(new_scancodes)
        self.down_scancodes = new_scancodes

    def all_keys_up(self):
        self.down_scancodes = set()
        all_keys = [0x2A, 0x2C, 0x2D, 0x4B, 0x4D, 0x50, 0x48]
        keyboard_inputs = list(map(lambda x: INPUT(INPUT_KEYBOARD, _INPUTunion(ki = KEYBDINPUT(0, x, KEYEVENTF_KEYUP | KEYEVENTF_SCANCODE, 0, None))), all_keys))
        self.send_inputs(keyboard_inputs)
        
    def get_new_scancodes(self, focus, shoot, bomb, horizontal, vertical):
        scancodes = set()
        if focus > 0:
            scancodes.add(0x2A)
        if shoot > 0:
            scancodes.add(0x2C)
        if bomb > 0:
            scancodes.add(0x2D)
        if horizontal < 0:
            scancodes.add(0x4B) #left
        elif horizontal > 0:
            scancodes.add(0x4D) #right
        if vertical < 0:
            scancodes.add(0x50) #down
        elif vertical > 0:
            scancodes.add(0x48) #up
        return scancodes
        
    def update_key_state(self, focus, shoot, bomb, horizontal, vertical):
        scancodes = self.get_new_scancodes(focus, shoot, bomb, horizontal, vertical)
        self.send_keys(scancodes)
        
    def update_key_state_by_rolled_input(self, rolled_input):
        focus, shoot, bomb, horizontal, vertical = rolled_input
        scancodes = self.get_new_scancodes(focus, shoot, bomb, horizontal, vertical)
        self.send_keys(scancodes)
        
    def update_key_state_by_enum(self, enum):
        rolled_input = self.key_enum[enum]
        horizontal, vertical = rolled_input[0], rolled_input[1]
        scancodes = self.get_new_scancodes(0, 0, 0, horizontal, vertical)
        self.send_keys(scancodes)
        
    def send_inputs(self, inputs):
        nInputs = len(inputs)
        LPINPUT = INPUT * nInputs
        pInputs = LPINPUT(*inputs)
        cbSize = ctypes.c_int(ctypes.sizeof(INPUT))
        return ctypes.windll.user32.SendInput(nInputs, pInputs, cbSize)
        
    def maybe_explore(self, input, exploration_chance):
        roll = random.randint(1, 100)
        if exploration_chance >= roll:
            return random.randint(0, 8)
        else:
            return input

class GameWindow:
    def __init__(self, capture_width, capture_height):
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.get_game_window_handle()
        self.get_window_dimensions()
        self.get_device_contexts()
        self.pid = self.get_pid(self.th_hwnd)
        
        if capture_width % 4 != 0:
            raise Exception("Capture width isn't multiple of 4, this breaks the .bmp")
       
    def get_pid(self, hwnd):
        thread_id, pid = win32process.GetWindowThreadProcessId(hwnd)
        return pid
        
    def get_game_window_handle(self):
        th_hwnd = win32gui.FindWindow(None, "Touhou Eiyashou - Imperishable Night")
        if th_hwnd == 0:
            raise Exception('Unable to find the Touhou Window')
        self.th_hwnd = th_hwnd

    def get_window_dimensions(self):
        self.game_width_start = 35
        self.game_width_end = 418
        self.game_height_start = 41
        self.game_height_end = 489
        self.game_width = self.game_width_end - self.game_width_start
        self.game_height = self.game_height_end - self.game_height_start
        
    def get_device_contexts(self):
        self.th_dc_h = win32gui.GetWindowDC(self.th_hwnd)
        self.th_dc = win32ui.CreateDCFromHandle(self.th_dc_h)
        self.mem_dc_h = gdi.CreateCompatibleDC(self.th_dc_h)
        self.mem_dc = win32ui.CreateDCFromHandle(self.mem_dc_h)
        
    def get_game_window(self):
        mem_dc_h = self.mem_dc_h
        th_dc = self.th_dc
        th_dc_h = self.th_dc_h
        mem_dc = self.mem_dc
        
        with_stride = 512
        pixel_data = POINTER(c_byte * (448 * 384 * 3))()
        im_size = (self.game_width, self.game_height)

        dib_section = gdi.CreateDIBSection(
            th_dc_h,
            byref(BITMAPINFOHEADER(self.game_width, self.game_height)),
            0,
            byref(pixel_data),
            None,
            0
        )
        
        gdi.SelectObject(mem_dc_h, dib_section)
        
        CAPTUREBLT = 0x40000000
        
        bitblt_success = gdi.BitBlt(
            mem_dc_h,
            0,
            0,
            self.game_width,
            self.game_height, 
            th_dc_h,
            self.game_width_start,
            self.game_height_start,
            win32con.SRCCOPY | CAPTUREBLT
        )

        gdi.DeleteObject(dib_section)
        
        return pixel_data.contents
        
    def get_player_centered_game_window(self, player_width, player_height):
        mem_dc_h = self.mem_dc_h
        th_dc = self.th_dc
        th_dc_h = self.th_dc_h
        mem_dc = self.mem_dc
        
        capture_width_start = int(player_width - (self.capture_width / 2))
        capture_height_start = int(player_height - (self.capture_height / 2))
        
        capture_width = self.capture_width
        capture_height = self.capture_height
        
        blt_width_offset = 0
        blt_height_offset = 0
        
        if capture_width_start < self.game_width_start:
            blt_width_offset = self.game_width_start - capture_width_start
            capture_width -= blt_width_offset
            capture_width_start = self.game_width_start
            
        if capture_height_start < self.game_height_start:
            blt_height_offset = self.game_height_start - capture_height_start
            capture_height -= blt_height_offset
            capture_height_start = self.game_height_start
            
        if capture_width_start + self.capture_width > self.game_width_end:
            capture_width = self.game_width_end - capture_width_start
            
        if capture_height_start + self.capture_height > self.game_height_end:
            capture_height = self.game_height_end - capture_height_start

        stuffed_capture_width = capture_width + (capture_width % 4)
        
        pixel_data = POINTER(c_byte * (self.capture_width * self.capture_height * 3))()

        dib_section = gdi.CreateDIBSection(
            th_dc_h,
            byref(BITMAPINFOHEADER(self.capture_width, self.capture_height)),
            0,
            byref(pixel_data),
            None,
            0
        )
        gdi.SelectObject(mem_dc_h, dib_section)
        
        CAPTUREBLT = 0x40000000
        
        bitblt_success = gdi.BitBlt(
            mem_dc_h,
            blt_width_offset,
            blt_height_offset,
            capture_width,
            capture_height, 
            th_dc_h,
            capture_width_start,
            capture_height_start,
            win32con.SRCCOPY | CAPTUREBLT
        )

        gdi.DeleteObject(dib_section)
        
        return pixel_data.contents

    def focus_window(self):
        win32gui.SetForegroundWindow(self.th_hwnd)
        
    def cleanup(self):
        self.dcObj.DeleteDC()
        screen_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, wDC)
        win32gui.DeleteObject(screen_bitmap.GetHandle())
        
class ServerCommunicator():
    
    def __init__(self):
        self.TCP_IP = '192.168.0.16'
        self.TCP_IP6 = '2603:3004:620:5c00:2e56:dcff:fe38:dfec'
        self.TCP_PORT = 8008
        self.BUFFER_SIZE = 1024
        self.socket = self.get_new_socket()
    
    def get_new_socket(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.TCP_IP, self.TCP_PORT))
        return s
        
    def send_data(self, arr):
        out_view = memoryview(arr).cast('B')
        while len(out_view):
            nsent = self.socket.send(out_view)
            out_view = out_view[nsent:]
        
    def recieve_data(self, length):
        ret = np.zeros(shape = length, dtype=np.float32)
        ret_view = memoryview(ret).cast('B')
        self.socket.recv_into(ret_view)
        return ret
        
    def get_recommended_move(self, screen, state):
        self.send_states(screen, state)
        return self.recieve_data(9)
        
    def send_states(self, screen, state):
        self.send_data(screen)
        self.send_data(state)

    def close(self):
        self.socket.close()

class AddressSpy():
    def __init__(self, pid):
        self.pid = pid
        self.retry_menu_address = 0x004D6DD4 
        self.score_address = 0x004E4B58
        self.is_dead_address = 0x004EA49C
        self.player_y_address = 0x017D6114
        self.player_x_address = 0x017D61AC

    def read_process_memory(self, address, size, allow_partial=False):
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        ERROR_PARTIAL_COPY = 0x012B
        PROCESS_VM_READ = 0x0010
        
        buf = (ctypes.c_char * size)()
        nread = ctypes.c_size_t()
        hProcess = kernel32.OpenProcess(PROCESS_VM_READ, False, self.pid)
        try:
            kernel32.ReadProcessMemory(hProcess, address, buf, size,
                ctypes.byref(nread))
        except WindowsError as e:
            if not allow_partial or e.winerror != ERROR_PARTIAL_COPY:
                raise
        finally:
            kernel32.CloseHandle(hProcess)
        return buf[:nread.value]
            
    def get_player_x(self):
        data = self.read_process_memory(self.player_x_address, 4)
        return int(round(struct.unpack("<f", data)[0]))
        
    def get_player_y(self):
        data = self.read_process_memory(self.player_y_address, 4)
        return int(round(struct.unpack("<f", data)[0]))
            
    def get_score(self):
        return int.from_bytes(self.read_process_memory(self.score_address, 4), byteorder='little', signed=False)
            
    def get_menu_state(self):
        return int.from_bytes(self.read_process_memory(self.retry_menu_address, 4), byteorder='little', signed=False)
        
    def get_death_state(self):
        return int.from_bytes(self.read_process_memory(self.is_dead_address, 4), byteorder='little', signed=False)

    def get_combined_state(self):
        return np.array([self.get_score(), self.get_menu_state(), self.get_death_state()], np.uint32)

class Hyperparameters:
    def __init__(self):
        self.epoch = 0
        
        self.epochs_per_stage =      [750,  750,    750,    -1]
        #Exploration
        self.gammas =                [80,   70,     50,     80]       
        self.gamma_decays =          [0.01, 0.01,   0.02,   0.02]
        
        
        #fps resolution
        self.capture_frequencies =   [20,   15,     12,     5]
        
        self.setup_new_stage()
        
    def setup_new_stage(self):
        self.gamma = self.gammas.pop(0)
        self.gamma_decay = self.gamma_decays.pop(0)
        self.epochs_til_next_stage = self.epochs_per_stage.pop(0)
        self.capture_frequency = self.capture_frequencies.pop(0)
        
    def skip_epcoh(self, epochs):
        while epochs > 0:
            self.update()
            epochs -= 1
        
    def update(self):
        self.epoch += 1
        self.epochs_til_next_stage -= 1
        
        self.gamma -= self.gamma_decay
        if len(self.epochs_per_stage) >= 0 and self.epochs_til_next_stage == 0:
            self.setup_new_stage()
        
    def get_gamma_fps(self):
        return (self.gamma, self.capture_frequency)
        
        
if __name__ == '__main__':
    gw = GameWindow(256, 256)
    controller = Controller()
    spy = AddressSpy(gw.pid)
    
    initial_score = 250
    hyperparameters = Hyperparameters()
    
    gw.focus_window()

    comm = ServerCommunicator()
    
    up = [0, 0, 0, 0, 1]
    accept = [0, 1, 0, 0, 0]
    selected_move = np.zeros(shape=1)
    
    run = True
    
    is_dead = False
    prev_is_dead = False
    #menu state = 0, try again = 1, don't try again = 2
    
    last_score = 0
    
    while run:
        gamma, frames_per_capture = hyperparameters.get_gamma_fps()
        gamma_decay = 0.02
        game_state = spy.get_combined_state()
        current_score = game_state[0]
        menu_state = game_state[1]
        is_dead_state = game_state[2]
        prev_is_dead = is_dead
        is_dead = is_dead_state > 0 or is_dead
        
        if not is_dead and current_score % frames_per_capture == 0 and current_score > initial_score and current_score > last_score:
            game_state[0] = frames_per_capture
            last_score = current_score
            px = spy.get_player_x() + 35 #offset because the sprite coordinate
            py = spy.get_player_y() + 25 #is not the center of the player          
            frame = gw.get_player_centered_game_window(px, py)
            #frame = gw.get_game_window()
            
            cur_time = time.time()
            rec = comm.get_recommended_move(frame, game_state)
            
            sys.stdout.flush()
            time_taken = time.time() - cur_time
            if time_taken > 1.0 / 60.0:
                if time_taken > 1.0 / frames_per_capture:
                    print("WARNING: PROCESSING TIME TAKES LONGER THAN A CAPTURED FRAME")
                else:
                    print("WARNING: PROCESSING TIME IS LONGER THAN 1/60 OF A SECOND")
                sys.stdout.flush()
            
            best_move = np.argmax(rec)
            move = controller.maybe_explore(best_move, gamma)
            controller.update_key_state_by_enum(move)
            selected_move = np.array([move], np.uint8)
            comm.send_data(selected_move)
            
            
        elif not prev_is_dead and is_dead: #pos-edge
            incremental_score = current_score - last_score
            game_state[0] = incremental_score
            frame = gw.get_player_centered_game_window(px, py)
            
            comm.send_states(frame, game_state)            
            while menu_state == 0:
                controller.all_keys_up()
                time.sleep(0.5)
                menu_state = spy.get_combined_state()[1]
            while menu_state != 1:
                time.sleep(0.2)
                controller.update_key_state_by_enum(1)
                time.sleep(0.2)
                controller.all_keys_up()
                time.sleep(0.2)
                menu_state = spy.get_combined_state()[1]
            return_data = comm.recieve_data(9)
            should_update_gamma = return_data[0] > 0.5
            if should_update_gamma:
                gamma -= gamma_decay
                gamma = max(5, gamma)
            controller.update_key_state_by_rolled_input(accept)
            time.sleep(0.1)
            controller.all_keys_up()
            is_dead = False
            last_score = 0
            
            hyperparameters.update()
            print("current epoch is: " + str(hyperparameters.epoch))
            print("epoch til next stage: " + str(hyperparameters.epochs_til_next_stage))
            sys.stdout.flush()
        else:
            if current_score + 1 % frames_per_capture == 0:
                time.sleep(0.001)
            else:
                time.sleep(0.005)
            
