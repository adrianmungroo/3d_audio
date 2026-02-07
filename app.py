import numpy as np
import pygame
import sounddevice as sd
from dataclasses import dataclass
from scipy.io import wavfile
from scipy import signal
import tkinter as tk
from tkinter import filedialog
import os

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

@dataclass
class SpatialParams:
    max_itd: float = 0.00065         # seconds (~0.65 ms)
    room_half_size_m: float = 2.0
    min_dist: float = 0.20
    gain: float = 0.25               # master volume
    head_radius_m: float = 0.0875    # ~17.5cm head diameter / 2
    speed_of_sound: float = 343.0    # m/s

class AudioSource:
    """Represents a single audio source with position and audio data."""
    def __init__(self, position, velocity, prev_position, fs=48000,
                 phase=0.0, base_freq=220.0, enable_tone=True,
                 audio_file_data=None, audio_file_pos=0, use_audio_file=False,
                 audio_file_name=None, gain=1.0, paused=False):
        # Position and velocity
        if isinstance(position, list):
            self.position = np.array(position, dtype=np.float32)
        else:
            self.position = position
        if isinstance(velocity, list):
            self.velocity = np.array(velocity, dtype=np.float32)
        else:
            self.velocity = velocity
        if isinstance(prev_position, list):
            self.prev_position = np.array(prev_position, dtype=np.float32)
        else:
            self.prev_position = prev_position
        
        # Audio generation
        self.phase = phase
        self.base_freq = base_freq
        self.enable_tone = enable_tone
        
        # Audio file playback
        self.audio_file_data = audio_file_data
        self.audio_file_pos = audio_file_pos
        self.use_audio_file = use_audio_file
        self.audio_file_name = audio_file_name
        
        # Per-source gain
        self.gain = gain
        
        # Pause state
        self.paused = paused
        
        # Per-source audio processing objects (critical for multiple sources!)
        max_delay_samp = int(fs * 0.003)
        self.delayL = FractionalDelay(max_delay_samp)
        self.delayR = FractionalDelay(max_delay_samp)
        self.lpf_direct = OnePoleLPF(fs)
        self.lpf_wet = OnePoleLPF(fs)
        self.revL = SimpleReverb(fs)
        self.revR = SimpleReverb(fs)

class FractionalDelay:
    """Fractional delay using linear interpolation."""
    def __init__(self, max_delay_samples: int):
        self.buf = np.zeros(max_delay_samples + 2, dtype=np.float32)
        self.w = 0
        self.N = len(self.buf)

    def process(self, x: np.ndarray, delay_samples: float) -> np.ndarray:
        out = np.zeros_like(x, dtype=np.float32)
        d = float(clamp(delay_samples, 0.0, self.N - 2.0))

        for i, xi in enumerate(x.astype(np.float32)):
            self.buf[self.w] = xi

            r = self.w - d
            if r < 0:
                r += self.N

            r0 = int(r) % self.N
            frac = r - int(r)
            r1 = (r0 + 1) % self.N

            out[i] = (1.0 - frac) * self.buf[r0] + frac * self.buf[r1]
            self.w = (self.w + 1) % self.N

        return out

class OnePoleLPF:
    """Cheap one-pole low-pass (vectorized for performance)."""
    def __init__(self, fs: int):
        self.fs = fs
        self.z = 0.0

    def process(self, x: np.ndarray, cutoff_hz: float) -> np.ndarray:
        cutoff_hz = clamp(cutoff_hz, 80.0, self.fs * 0.45)
        a = np.exp(-2.0 * np.pi * cutoff_hz / self.fs)
        x = x.astype(np.float32)
        # Vectorized filter: y[n] = a*y[n-1] + (1-a)*x[n]
        # Using scipy.signal.lfilter would be ideal, but this is faster for one-pole
        y = np.empty_like(x, dtype=np.float32)
        y[0] = a * self.z + (1.0 - a) * x[0]
        for i in range(1, len(x)):
            y[i] = a * y[i-1] + (1.0 - a) * x[i]
        self.z = y[-1]
        return y

class SimpleReverb:
    """
    Lightweight “roominess”:
    - one feedback delay line
    - plus a couple of early reflection taps
    Not a studio reverb, but good enough for the front/back illusion.
    """
    def __init__(self, fs: int, max_delay_s: float = 0.25):
        self.fs = fs
        self.N = int(fs * max_delay_s) + 1
        self.buf = np.zeros(self.N, dtype=np.float32)
        self.w = 0

    def process(self, x: np.ndarray, delay_s: float, feedback: float, mix: float) -> np.ndarray:
        delay_s = clamp(delay_s, 0.01, 0.25)
        feedback = clamp(feedback, 0.0, 0.95)
        mix = clamp(mix, 0.0, 1.0)

        d = int(self.fs * delay_s)
        out = np.zeros_like(x, dtype=np.float32)

        # early reflection taps relative to delay
        tap1 = max(1, int(d * 0.35))
        tap2 = max(1, int(d * 0.62))

        for i, xi in enumerate(x.astype(np.float32)):
            r = (self.w - d) % self.N
            r1 = (self.w - tap1) % self.N
            r2 = (self.w - tap2) % self.N

            delayed = self.buf[r]
            early = 0.6 * self.buf[r1] + 0.4 * self.buf[r2]

            # write with feedback
            self.buf[self.w] = xi + delayed * feedback
            self.w = (self.w + 1) % self.N

            wet = 0.7 * delayed + 0.3 * early
            out[i] = (1.0 - mix) * xi + mix * wet

        return out

class Demo:
    def __init__(self, fs=48000, block=256):
        self.fs = fs
        self.block = block
        self.params = SpatialParams()

        # 2D positions (meters)
        self.listener = np.array([0.0, 0.0], dtype=np.float32)
        
        # Velocities for Doppler effect (m/s)
        self.listener_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_listener = self.listener.copy()

        # Listener facing angle (radians). Starts at North (+y direction = π/2).
        self.theta = np.pi / 2.0

        # Multiple audio sources (each has its own processing objects)
        self.sources = [
            AudioSource(
                position=np.array([0.9, 0.3], dtype=np.float32),
                velocity=np.array([0.0, 0.0], dtype=np.float32),
                prev_position=np.array([0.9, 0.3], dtype=np.float32),
                fs=self.fs,
                base_freq=220.0,
                enable_tone=True
            )
        ]
        self.selected_source_idx = 0  # Currently selected source for movement

        self.running = True

    def load_audio_file(self, filepath: str, source_idx: int = None):
        """Load an audio file (WAV) and assign it to a source."""
        if source_idx is None:
            source_idx = self.selected_source_idx
        if source_idx < 0 or source_idx >= len(self.sources):
            return False
            
        try:
            # Read WAV file
            sample_rate, audio_data = wavfile.read(filepath)
            
            # Convert to float32 and normalize
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype == np.uint8:
                audio_data = (audio_data.astype(np.float32) - 128.0) / 128.0
            elif audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                audio_data = audio_data.astype(np.float32)
            else:
                audio_data = audio_data.astype(np.float32)
                # Normalize to [-1, 1] range
                max_val = np.abs(audio_data).max()
                if max_val > 0:
                    audio_data = audio_data / max_val
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample if sample rate doesn't match
            if sample_rate != self.fs:
                num_samples = int(len(audio_data) * self.fs / sample_rate)
                audio_data = signal.resample(audio_data, num_samples)
            
            source = self.sources[source_idx]
            source.audio_file_data = audio_data.astype(np.float32)
            source.audio_file_pos = 0
            source.use_audio_file = True
            source.audio_file_name = os.path.basename(filepath)
            return True
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return False

    def spatialize(self, mono: np.ndarray, source: AudioSource) -> np.ndarray:
        source_pos = source.position
        v = (source_pos - self.listener).astype(np.float32)
        dist = float(np.linalg.norm(v))
        dist = max(dist, self.params.min_dist)

        # facing vector (unit)
        fwd = np.array([np.cos(self.theta), np.sin(self.theta)], dtype=np.float32)
        u = v / dist

        # "frontness" in [-1, +1]: +1 in front, -1 behind
        frontness = float(np.dot(u, fwd))

        # left/right pan from right-vector dot
        right = np.array([np.cos(self.theta + np.pi/2), np.sin(self.theta + np.pi/2)], dtype=np.float32)
        pan = float(clamp(-np.dot(u, right), -1.0, 1.0))

        # Equal-power panning
        gL = float(np.sqrt(0.5 * (1.0 - pan)))
        gR = float(np.sqrt(0.5 * (1.0 + pan)))

        # ITD: more accurate calculation based on head size and angle
        # ITD = (head_radius / speed_of_sound) * (angle + sin(angle))
        # Simplified: use pan as proxy for angle, but with head radius consideration
        angle_rad = np.arcsin(clamp(pan, -0.999, 0.999))
        itd = (self.params.head_radius_m / self.params.speed_of_sound) * (angle_rad + np.sin(angle_rad))
        itd = clamp(itd, -self.params.max_itd, self.params.max_itd)
        delayL_s = max(0.0,  itd)
        delayR_s = max(0.0, -itd)

        # Distance attenuation: inverse-square law with rolloff
        # Realistic: 1/distance^2, but we add a rolloff to prevent extreme volumes
        att = 1.0 / (1.0 + dist * dist * 0.5)

        # === Psychoacoustic front/back cheat ===
        # behindness: 0 (front) -> 1 (behind)
        behindness = float(clamp((-frontness + 1.0) * 0.5, 0.0, 1.0))

        # direct brightness: behind => duller
        cutoff_direct = (16000.0 * (1.0 - 0.65 * behindness)) / (1.0 + 1.6 * dist)

        # direct level: behind => slightly quieter
        direct_scale = 1.0 - 0.20 * behindness

        # wet mix: behind => roomier
        wet_mix = 0.06 + 0.28 * behindness

        # reverb delay & feedback: behind => longer tail
        rev_delay = 0.055 + 0.055 * behindness
        rev_fb = 0.35 + 0.45 * behindness

        # wet tone shaping: behind => even duller
        cutoff_wet = 6000.0 * (1.0 - 0.55 * behindness)

        # Apply direct path (use per-source processing objects)
        x = mono.astype(np.float32) * (att * self.params.gain) * direct_scale
        x_dir = source.lpf_direct.process(x, cutoff_hz=cutoff_direct)

        # Reverb (feed a bit of the direct path) - use per-source reverb
        wet_in = x_dir
        yL_wet = source.revL.process(wet_in, delay_s=rev_delay, feedback=rev_fb, mix=wet_mix)
        yR_wet = source.revR.process(wet_in, delay_s=rev_delay * 1.07, feedback=rev_fb, mix=wet_mix)  # slight stereo difference
        yL_wet = source.lpf_wet.process(yL_wet, cutoff_hz=cutoff_wet)
        yR_wet = source.lpf_wet.process(yR_wet, cutoff_hz=cutoff_wet)

        # Combine direct + wet
        xL = x_dir + yL_wet
        xR = x_dir + yR_wet

        # Apply ITD (delay far ear) then ILD (gains) - use per-source delays
        dL = delayL_s * self.fs
        dR = delayR_s * self.fs
        yL = source.delayL.process(xL, dL) * gL
        yR = source.delayR.process(xR, dR) * gR

        return np.stack([yL, yR], axis=1).astype(np.float32)

    def generate_source_audio(self, source: AudioSource, frames: int) -> np.ndarray:
        """Generate or load audio for a single source."""
        if source.use_audio_file and source.audio_file_data is not None:
            # Play loaded audio file
            mono = np.zeros(frames, dtype=np.float32)
            remaining = len(source.audio_file_data) - source.audio_file_pos
            
            if remaining > 0:
                # Copy available samples
                copy_len = min(frames, remaining)
                mono[:copy_len] = source.audio_file_data[source.audio_file_pos:source.audio_file_pos + copy_len]
                source.audio_file_pos += copy_len
                
                # Loop if we've reached the end
                if source.audio_file_pos >= len(source.audio_file_data):
                    source.audio_file_pos = 0
                    # Fill remaining with beginning of file
                    if copy_len < frames:
                        remaining_after_loop = frames - copy_len
                        mono[copy_len:] = source.audio_file_data[:remaining_after_loop]
                        source.audio_file_pos = remaining_after_loop
            return mono * source.gain
        else:
            # Generate noise and tone
            noise = np.random.randn(frames).astype(np.float32) * 0.25
            noise = 0

            # Calculate current frequency with Doppler effect
            v = (source.position - self.listener).astype(np.float32)
            dist = float(np.linalg.norm(v))
            dist = max(dist, self.params.min_dist)
            u = v / dist
            rel_vel = np.dot(source.velocity - self.listener_vel, u)
            doppler_factor = clamp(1.0 + (rel_vel / self.params.speed_of_sound), 0.5, 2.0)
            current_freq = source.base_freq * doppler_factor

            t = np.arange(frames, dtype=np.float32) / self.fs
            ph = source.phase + (2.0 * np.pi * current_freq) * t
            tone = np.sin(ph).astype(np.float32) * 0.12 if source.enable_tone else 0.0
            source.phase = float((ph[-1] + (2.0 * np.pi * current_freq / self.fs)) % (2.0 * np.pi))

            return (noise + tone) * source.gain

    def audio_callback(self, outdata, frames, time, status):
        if not self.running:
            outdata[:] = 0
            return

        # Process all sources and mix them together
        mixed_output = np.zeros((frames, 2), dtype=np.float32)
        
        for source in self.sources:
            # Skip paused sources
            if source.paused:
                continue
                
            # Generate audio for this source
            mono = self.generate_source_audio(source, frames)
            
            # Spatialize this source (pass the source object, not just position)
            spatialized = self.spatialize(mono, source)
            
            # Mix into output
            mixed_output += spatialized
        
        # Better clipping prevention: scale down if too many sources
        # Each source contributes, so divide by number of sources to prevent overload
        num_sources = max(1, len(self.sources))
        mixed_output = mixed_output / (num_sources * 0.7)  # Scale down per source
        
        # Hard limit to prevent clipping
        mixed_output = np.clip(mixed_output, -1.0, 1.0)
        
        outdata[:] = mixed_output

    def run_ui(self):
        pygame.init()
        W, H = 760, 560
        fullscreen = False
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("2D Immersive Audio + Front/Back Cheat (WASD move relative to look, arrows source, mouse look, F11=fullscreen, L=load audio)")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 22)

        room_px = 400
        cx, cy = W // 2, H // 2
        half = room_px // 2

        def world_to_screen(p):
            s = room_px / (2.0 * self.params.room_half_size_m)
            x = cx + int(p[0] * s)
            y = cy - int(p[1] * s)
            return x, y

        speed = 1.6
        mouse_sensitivity = 0.01  # radians per pixel of horizontal mouse movement

        while self.running:
            dt = clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F11:
                        # Toggle fullscreen
                        fullscreen = not fullscreen
                        if fullscreen:
                            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                            W, H = screen.get_size()
                        else:
                            screen = pygame.display.set_mode((760, 560))
                            W, H = 760, 560
                        cx, cy = W // 2, H // 2
                    elif event.key == pygame.K_l:
                        # Load audio file
                        root = tk.Tk()
                        root.withdraw()  # Hide the main window
                        filepath = filedialog.askopenfilename(
                            title="Select Audio File (WAV)",
                            filetypes=[
                                ("WAV files", "*.wav"),
                                ("All files", "*.*")
                            ]
                        )
                        root.destroy()
                        
                        if filepath:
                            if self.load_audio_file(filepath, self.selected_source_idx):
                                print(f"Loaded audio file: {os.path.basename(filepath)} for source {self.selected_source_idx+1}")
                            else:
                                print(f"Failed to load audio file: {filepath}")
                    elif event.key == pygame.K_n:
                        # Add new source with N key (event-based, only triggers once)
                        if len(self.sources) < 10:
                            # Add source at mouse position (in world coordinates)
                            mx, my = pygame.mouse.get_pos()
                            # Convert screen to world coordinates
                            s = (2.0 * self.params.room_half_size_m) / room_px
                            world_x = (mx - cx) * s
                            world_y = (cy - my) * s  # Invert Y
                            new_pos = np.array([world_x, world_y], dtype=np.float32)
                            self.sources.append(AudioSource(
                                position=new_pos,
                                velocity=np.array([0.0, 0.0], dtype=np.float32),
                                prev_position=new_pos.copy(),
                                fs=self.fs,
                                base_freq=220.0 + len(self.sources) * 50.0,  # Different frequency per source
                                enable_tone=True
                            ))
                            self.selected_source_idx = len(self.sources) - 1
                    elif event.key == pygame.K_DELETE:
                        # Delete selected source with DELETE key (event-based)
                        if len(self.sources) > 1:
                            self.sources.pop(self.selected_source_idx)
                            self.selected_source_idx = min(self.selected_source_idx, len(self.sources) - 1)
                    elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, 
                                       pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]:
                        # Select source with number keys (event-based)
                        num_keys = [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, 
                                   pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]
                        if event.key in num_keys[:len(self.sources)]:
                            idx = num_keys.index(event.key)
                            if idx < len(self.sources):
                                self.selected_source_idx = idx
                    elif event.key == pygame.K_SPACE or event.key == pygame.K_p:
                        # Toggle pause for selected source
                        if len(self.sources) > 0 and self.selected_source_idx < len(self.sources):
                            source = self.sources[self.selected_source_idx]
                            source.paused = not source.paused
                    elif event.key == pygame.K_SPACE or event.key == pygame.K_p:
                        # Toggle pause for selected source
                        if len(self.sources) > 0 and self.selected_source_idx < len(self.sources):
                            source = self.sources[self.selected_source_idx]
                            source.paused = not source.paused

            keys = pygame.key.get_pressed()

            # Mouse look: only left-right movement controls head rotation
            # Center X is reference (North), left rotates left, right rotates right
            mx, my = pygame.mouse.get_pos()
            dx = mx - cx  # Horizontal offset from center
            # Map mouse X position to rotation angle (North = π/2 is center)
            # Negative sign: mouse left = rotate left, mouse right = rotate right
            self.theta = np.pi / 2.0 - dx * mouse_sensitivity

            # Move listener (WASD) relative to look direction
            # Forward/backward along look direction, left/right perpendicular
            fwd = np.array([np.cos(self.theta), np.sin(self.theta)], dtype=np.float32)
            right = np.array([np.cos(self.theta + np.pi/2), np.sin(self.theta + np.pi/2)], dtype=np.float32)
            
            mv = np.array([0.0, 0.0], dtype=np.float32)
            if keys[pygame.K_w]: mv += fwd      # Forward
            if keys[pygame.K_s]: mv -= fwd      # Backward
            if keys[pygame.K_a]: mv += right    # Left (strafe)
            if keys[pygame.K_d]: mv -= right    # Right (strafe)
            
            n = float(np.linalg.norm(mv))
            if n > 0:
                mv /= n
            self.listener += mv * speed * dt
            
            # Update velocities for Doppler (simple finite difference)
            self.listener_vel = (self.listener - self.prev_listener) / max(dt, 1e-6)
            self.prev_listener = self.listener.copy()
            
            # Move selected source (arrows)
            if len(self.sources) > 0:
                selected_source = self.sources[self.selected_source_idx]
                mv2 = np.array([0.0, 0.0], dtype=np.float32)
                if keys[pygame.K_UP]:    mv2[1] += 1
                if keys[pygame.K_DOWN]:  mv2[1] -= 1
                if keys[pygame.K_LEFT]:  mv2[0] -= 1
                if keys[pygame.K_RIGHT]: mv2[0] += 1
                n2 = float(np.linalg.norm(mv2))
                if n2 > 0:
                    mv2 /= n2
                selected_source.position += mv2 * speed * dt
                
                # Update source velocity for Doppler
                selected_source.velocity = (selected_source.position - selected_source.prev_position) / max(dt, 1e-6)
                selected_source.prev_position = selected_source.position.copy()

            # clamp to room
            r = self.params.room_half_size_m
            self.listener[0] = clamp(float(self.listener[0]), -r, r)
            self.listener[1] = clamp(float(self.listener[1]), -r, r)
            for source in self.sources:
                source.position[0] = clamp(float(source.position[0]), -r, r)
                source.position[1] = clamp(float(source.position[1]), -r, r)

            # draw
            screen.fill((245, 245, 245))
            pygame.draw.rect(screen, (20, 20, 20), (cx - half, cy - half, room_px, room_px), width=3)
            pygame.draw.line(screen, (150, 150, 150), (cx - half, cy), (cx + half, cy), width=1)
            pygame.draw.line(screen, (150, 150, 150), (cx, cy - half), (cx, cy + half), width=1)

            lx, ly = world_to_screen(self.listener)

            # Draw lines from listener to all sources
            for i, source in enumerate(self.sources):
                sx, sy = world_to_screen(source.position)
                # Highlight selected source
                if i == self.selected_source_idx:
                    if source.paused:
                        # Paused selected source - grayed out
                        pygame.draw.line(screen, (150, 150, 150), (lx, ly), (sx, sy), width=2)
                        pygame.draw.circle(screen, (100, 100, 100), (sx, sy), 10)
                        pygame.draw.circle(screen, (150, 150, 150), (sx, sy), 10, width=2)  # Outline
                    else:
                        # Active selected source
                        pygame.draw.line(screen, (200, 100, 100), (lx, ly), (sx, sy), width=2)
                        pygame.draw.circle(screen, (200, 50, 50), (sx, sy), 10)  # Larger, brighter
                    # Draw source number
                    num_text = font.render(str(i+1), True, (255, 255, 255))
                    screen.blit(num_text, (sx - 6, sy - 8))
                else:
                    if source.paused:
                        # Paused unselected source - grayed out
                        pygame.draw.line(screen, (120, 120, 120), (lx, ly), (sx, sy), width=1)
                        pygame.draw.circle(screen, (80, 80, 80), (sx, sy), 8)
                    else:
                        # Active unselected source
                        pygame.draw.line(screen, (90, 90, 90), (lx, ly), (sx, sy), width=1)
                        pygame.draw.circle(screen, (150, 30, 30), (sx, sy), 8)
                    # Draw source number
                    num_text = font.render(str(i+1), True, (255, 255, 255))
                    screen.blit(num_text, (sx - 6, sy - 8))

            # Draw listener
            pygame.draw.circle(screen, (235, 190, 40), (lx, ly), 10)

            # facing arrow
            fwd = np.array([np.cos(self.theta), np.sin(self.theta)], dtype=np.float32)
            tip = (lx + int(22 * fwd[0]), ly - int(22 * fwd[1]))
            pygame.draw.line(screen, (40, 40, 40), (lx, ly), tip, width=3)

            # info - show selected source info
            if len(self.sources) > 0 and self.selected_source_idx < len(self.sources):
                selected_source = self.sources[self.selected_source_idx]
                v = selected_source.position - self.listener
                dist = float(np.linalg.norm(v))
                dist = max(dist, self.params.min_dist)
                u = v / dist
                frontness = float(np.dot(u, fwd))
                behindness = float(clamp((-frontness + 1.0) * 0.5, 0.0, 1.0))
                state = "BEHIND" if frontness < 0 else "FRONT"
                paused_text = " [PAUSED]" if selected_source.paused else ""
                info = f"Source {self.selected_source_idx+1}/{len(self.sources)}: dist={dist:.2f}m  frontness={frontness:+.2f}  ({state}){paused_text}"
                color = (150, 150, 150) if selected_source.paused else (0, 0, 0)
                screen.blit(font.render(info, True, color), (18, 18))
            
            controls = "WASD=listener, arrows=move source, 1-9=select, N=add, DEL=delete, L=load audio, SPACE/P=pause"
            screen.blit(font.render(controls, True, (0, 0, 0)), (18, 42))
            
            # Show audio file info for selected source
            if len(self.sources) > 0 and self.selected_source_idx < len(self.sources):
                selected_source = self.sources[self.selected_source_idx]
                if selected_source.use_audio_file and selected_source.audio_file_name:
                    audio_info = f"Source {self.selected_source_idx+1}: {selected_source.audio_file_name} (L=load)"
                    screen.blit(font.render(audio_info, True, (0, 100, 0)), (18, 66))
                else:
                    audio_info = f"Source {self.selected_source_idx+1}: Noise/Tone (L=load audio)"
                    screen.blit(font.render(audio_info, True, (100, 100, 100)), (18, 66))

            pygame.display.flip()

        pygame.quit()

    def run(self):
        with sd.OutputStream(
            samplerate=self.fs,
            blocksize=self.block,
            channels=2,
            dtype="float32",
            callback=self.audio_callback,
            latency="low",
        ):
            self.run_ui()
        self.running = False

if __name__ == "__main__":
    Demo().run()
