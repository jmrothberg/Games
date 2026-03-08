"""
ESKF Indoor Drone Localization — Error-State Kalman Filter for GPS-denied flight.

This implements a manifold-based ESKF that fuses:
  - IMU (accelerometer + gyroscope) as the high-rate propagation model
  - UWB range beacons (like Decawave DWM1001) for position corrections
  - Barometer for altitude corrections
  - Optical flow for velocity corrections (simulating a downward camera)

The ESKF works on SO(3) manifolds for rotation, meaning it tracks a *small
error rotation* (3-vector) instead of a full quaternion in the filter state.
This avoids quaternion normalization hacks and gives better linearization.

Run this file directly to see a simulated drone flying a figure-8 inside a
building with noisy sensors, and compare the ESKF estimate vs. ground truth.

Usage:
    python eskf_indoor_localization.py

Author: JMR / Claude — for indoor drone navigation experiments
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import time


# ─────────────────────────────────────────────
#  Quaternion utilities (Hamilton convention)
# ─────────────────────────────────────────────

def quat_mult(q, p):
    """Multiply two quaternions q * p. Format: [w, x, y, z]."""
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = p
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def quat_conj(q):
    """Quaternion conjugate (inverse for unit quaternions)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_to_rot(q):
    """Convert unit quaternion to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])

def rot_vec_to_quat(dtheta):
    """Small rotation vector (3,) -> quaternion. First-order for ESKF."""
    angle = np.linalg.norm(dtheta)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    half = angle / 2.0
    axis = dtheta / angle
    return np.array([np.cos(half), *(axis * np.sin(half))])

def skew(v):
    """Skew-symmetric matrix from 3-vector."""
    return np.array([
        [ 0,   -v[2],  v[1]],
        [ v[2], 0,    -v[0]],
        [-v[1], v[0],  0   ],
    ])


# ─────────────────────────────────────────────
#  ESKF state and filter
# ─────────────────────────────────────────────

@dataclass
class NominalState:
    """Nominal (large) state on the manifold."""
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))       # position [m]
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))       # velocity [m/s]
    q: np.ndarray = field(default_factory=lambda: np.array([1.,0.,0.,0.]))  # orientation quaternion
    ab: np.ndarray = field(default_factory=lambda: np.zeros(3))      # accel bias [m/s^2]
    gb: np.ndarray = field(default_factory=lambda: np.zeros(3))      # gyro bias [rad/s]


class ESKF:
    """
    Error-State Kalman Filter for indoor drone localization.

    Error state vector (15-dimensional):
        [dp(3), dv(3), dtheta(3), dab(3), dgb(3)]

    This is the filter from Solà (2017) "Quaternion kinematics for the
    error-state Kalman filter" — the gold standard for IMU fusion.
    """

    STATE_DIM = 15  # error state dimension

    def __init__(self, gravity=np.array([0., 0., -9.81])):
        self.nominal = NominalState()
        self.gravity = gravity

        # Error-state covariance
        self.P = np.eye(self.STATE_DIM) * 0.01

        # IMU noise parameters (tune these for your hardware)
        self.accel_noise_density = 0.02       # m/s^2/√Hz  (typical MEMS)
        self.gyro_noise_density = 0.001       # rad/s/√Hz
        self.accel_bias_random_walk = 0.0002  # m/s^3/√Hz
        self.gyro_bias_random_walk = 0.00004  # rad/s^2/√Hz

    # ── IMU propagation ──────────────────────

    def predict(self, accel_meas: np.ndarray, gyro_meas: np.ndarray, dt: float):
        """
        Propagate nominal state and error-state covariance with IMU.

        accel_meas: raw accelerometer reading [m/s^2] in body frame
        gyro_meas:  raw gyroscope reading [rad/s] in body frame
        dt:         time step [s]
        """
        # Bias-corrected measurements
        a_b = accel_meas - self.nominal.ab
        w_b = gyro_meas - self.nominal.gb

        R = quat_to_rot(self.nominal.q)

        # --- Propagate nominal state ---
        # Position
        self.nominal.p = self.nominal.p + self.nominal.v * dt + 0.5 * (R @ a_b + self.gravity) * dt**2
        # Velocity
        self.nominal.v = self.nominal.v + (R @ a_b + self.gravity) * dt
        # Orientation
        dq = rot_vec_to_quat(w_b * dt)
        self.nominal.q = quat_mult(self.nominal.q, dq)
        self.nominal.q /= np.linalg.norm(self.nominal.q)  # renormalize
        # Biases (random walk — no deterministic update)

        # --- Propagate error-state covariance ---
        # State transition matrix F (15x15)
        F = np.eye(self.STATE_DIM)
        F[0:3, 3:6] = np.eye(3) * dt                         # dp/dv
        F[3:6, 6:9] = -R @ skew(a_b) * dt                    # dv/dtheta
        F[3:6, 9:12] = -R * dt                                # dv/dab
        F[6:9, 12:15] = -np.eye(3) * dt                       # dtheta/dgb

        # Noise covariance Q
        Q = np.zeros((self.STATE_DIM, self.STATE_DIM))
        Q[3:6, 3:6] = np.eye(3) * (self.accel_noise_density * dt)**2
        Q[6:9, 6:9] = np.eye(3) * (self.gyro_noise_density * dt)**2
        Q[9:12, 9:12] = np.eye(3) * (self.accel_bias_random_walk * dt)**2
        Q[12:15, 12:15] = np.eye(3) * (self.gyro_bias_random_walk * dt)**2

        self.P = F @ self.P @ F.T + Q

    # ── Measurement updates ──────────────────

    def _update(self, H: np.ndarray, residual: np.ndarray, R_noise: np.ndarray):
        """Generic ESKF measurement update (inject error into nominal)."""
        S = H @ self.P @ H.T + R_noise
        K = self.P @ H.T @ np.linalg.inv(S)

        dx = K @ residual  # error-state correction

        # Inject error state into nominal state
        self.nominal.p += dx[0:3]
        self.nominal.v += dx[3:6]
        dq = rot_vec_to_quat(dx[6:9])
        self.nominal.q = quat_mult(self.nominal.q, dq)
        self.nominal.q /= np.linalg.norm(self.nominal.q)
        self.nominal.ab += dx[9:12]
        self.nominal.gb += dx[12:15]

        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_noise @ K.T

        # ESKF reset: after injection, error state is zero, but we need
        # to update P for the attitude reset (Solà eq. 297)
        G = np.eye(self.STATE_DIM)
        G[6:9, 6:9] = np.eye(3) - skew(0.5 * dx[6:9])
        self.P = G @ self.P @ G.T

    def update_position(self, pos_meas: np.ndarray, noise_std: float):
        """
        Update with a 3D position measurement (e.g., from UWB trilateration).

        pos_meas:  measured position [x, y, z] in world frame
        noise_std: measurement noise standard deviation [m]
        """
        H = np.zeros((3, self.STATE_DIM))
        H[0:3, 0:3] = np.eye(3)  # position is directly observed
        residual = pos_meas - self.nominal.p
        R = np.eye(3) * noise_std**2
        self._update(H, residual, R)

    def update_altitude(self, alt_meas: float, noise_std: float):
        """
        Update with a barometer altitude measurement.

        alt_meas:  measured altitude [m]
        noise_std: measurement noise standard deviation [m]
        """
        H = np.zeros((1, self.STATE_DIM))
        H[0, 2] = 1.0  # z-component of position
        residual = np.array([alt_meas - self.nominal.p[2]])
        R = np.array([[noise_std**2]])
        self._update(H, residual, R)

    def update_velocity(self, vel_meas: np.ndarray, noise_std: float):
        """
        Update with a velocity measurement (e.g., from optical flow).

        vel_meas:  measured velocity [vx, vy, vz] in world frame
        noise_std: measurement noise standard deviation [m/s]
        """
        H = np.zeros((3, self.STATE_DIM))
        H[0:3, 3:6] = np.eye(3)
        residual = vel_meas - self.nominal.v
        R = np.eye(3) * noise_std**2
        self._update(H, residual, R)

    def update_uwb_range(self, anchor_pos: np.ndarray, range_meas: float, noise_std: float):
        """
        Update with a single UWB range measurement to a known anchor.

        This is more realistic than full 3D position — each UWB beacon
        gives you ONE range, and you fuse them individually.

        anchor_pos: known anchor position [x, y, z] in world frame
        range_meas: measured range [m]
        noise_std:  measurement noise standard deviation [m]
        """
        diff = self.nominal.p - anchor_pos
        predicted_range = np.linalg.norm(diff)
        if predicted_range < 1e-6:
            return  # avoid division by zero

        H = np.zeros((1, self.STATE_DIM))
        H[0, 0:3] = diff / predicted_range  # Jacobian of range w.r.t. position
        residual = np.array([range_meas - predicted_range])
        R = np.array([[noise_std**2]])
        self._update(H, residual, R)

    def get_position(self) -> np.ndarray:
        return self.nominal.p.copy()

    def get_velocity(self) -> np.ndarray:
        return self.nominal.v.copy()

    def get_orientation_matrix(self) -> np.ndarray:
        return quat_to_rot(self.nominal.q)

    def get_position_uncertainty(self) -> np.ndarray:
        """Return 3-sigma position uncertainty [m] per axis."""
        return 3.0 * np.sqrt(np.diag(self.P[0:3, 0:3]))


# ─────────────────────────────────────────────
#  Indoor environment simulation
# ─────────────────────────────────────────────

@dataclass
class IndoorEnvironment:
    """
    Simulates a building interior with UWB anchors, barometer, and optical flow.

    Default setup: 10m x 10m x 3m room with 4 UWB anchors in corners + 4 on ceiling.
    This is a realistic deployment for a warehouse, gym, or lab.
    """
    # UWB anchor positions (mount at room corners, floor and ceiling)
    uwb_anchors: np.ndarray = field(default_factory=lambda: np.array([
        [0, 0, 0],      # corner floor
        [10, 0, 0],
        [10, 10, 0],
        [0, 10, 0],
        [0, 0, 3],      # corner ceiling
        [10, 0, 3],
        [10, 10, 3],
        [0, 10, 3],
    ], dtype=float))

    # Noise parameters (realistic for indoor MEMS + UWB)
    uwb_noise_std: float = 0.10          # UWB range noise [m] (DWM1001 ≈ 10cm)
    uwb_nlos_probability: float = 0.05   # chance of NLOS (non-line-of-sight) outlier
    uwb_nlos_bias: float = 1.5           # NLOS adds positive bias [m]
    baro_noise_std: float = 0.5          # barometer noise [m]
    baro_drift_rate: float = 0.01        # barometer drift [m/s]
    flow_noise_std: float = 0.15         # optical flow velocity noise [m/s]
    accel_noise_std: float = 0.05        # accelerometer noise [m/s^2]
    gyro_noise_std: float = 0.003        # gyroscope noise [rad/s]
    accel_bias: np.ndarray = field(default_factory=lambda: np.array([0.02, -0.01, 0.03]))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.array([0.001, -0.0005, 0.0008]))

    def simulate_imu(self, true_accel_world: np.ndarray, true_omega_body: np.ndarray,
                     R_true: np.ndarray) -> tuple:
        """Generate noisy IMU readings from true dynamics."""
        gravity = np.array([0., 0., -9.81])
        # Accelerometer measures specific force in body frame
        specific_force = R_true.T @ (true_accel_world - gravity)
        accel_meas = specific_force + self.accel_bias + np.random.randn(3) * self.accel_noise_std
        gyro_meas = true_omega_body + self.gyro_bias + np.random.randn(3) * self.gyro_noise_std
        return accel_meas, gyro_meas

    def simulate_uwb(self, true_pos: np.ndarray) -> list:
        """Generate noisy UWB range measurements to all anchors."""
        ranges = []
        for anchor in self.uwb_anchors:
            true_range = np.linalg.norm(true_pos - anchor)
            noise = np.random.randn() * self.uwb_noise_std
            # Simulate NLOS: occasional positive bias
            if np.random.rand() < self.uwb_nlos_probability:
                noise += self.uwb_nlos_bias * np.random.rand()
            ranges.append((anchor, true_range + noise))
        return ranges

    def simulate_barometer(self, true_alt: float, t: float) -> float:
        """Generate noisy barometer reading with slow drift."""
        drift = self.baro_drift_rate * t
        return true_alt + drift + np.random.randn() * self.baro_noise_std

    def simulate_optical_flow(self, true_vel: np.ndarray) -> np.ndarray:
        """Generate noisy optical flow velocity estimate."""
        return true_vel + np.random.randn(3) * self.flow_noise_std


def generate_indoor_trajectory(duration: float, dt: float):
    """
    Generate a figure-8 drone trajectory inside a 10x10x3m room.
    Returns arrays of (time, position, velocity, acceleration, orientation, angular_velocity).
    """
    N = int(duration / dt)
    t = np.arange(N) * dt

    # Figure-8 centered at (5, 5, 1.5) — staying well inside the room
    freq = 0.1  # Hz — slow enough for indoor flight
    radius = 3.0

    px = 5.0 + radius * np.sin(2 * np.pi * freq * t)
    py = 5.0 + radius * np.sin(4 * np.pi * freq * t) * 0.5  # figure-8
    pz = 1.5 + 0.3 * np.sin(2 * np.pi * freq * 0.5 * t)     # gentle altitude variation

    # Velocities (analytical derivatives)
    vx = radius * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t)
    vy = radius * 0.5 * 4 * np.pi * freq * np.cos(4 * np.pi * freq * t)
    vz = 0.3 * 2 * np.pi * freq * 0.5 * np.cos(2 * np.pi * freq * 0.5 * t)

    # Accelerations
    ax = -radius * (2 * np.pi * freq)**2 * np.sin(2 * np.pi * freq * t)
    ay = -radius * 0.5 * (4 * np.pi * freq)**2 * np.sin(4 * np.pi * freq * t)
    az = -0.3 * (2 * np.pi * freq * 0.5)**2 * np.sin(2 * np.pi * freq * 0.5 * t)

    positions = np.column_stack([px, py, pz])
    velocities = np.column_stack([vx, vy, vz])
    accelerations = np.column_stack([ax, ay, az])

    # Simple orientation: drone faces velocity direction, mostly level
    orientations = []
    angular_velocities = []
    for i in range(N):
        # Identity rotation for simplicity (drone is approximately level indoors)
        orientations.append(np.eye(3))
        angular_velocities.append(np.array([0.0, 0.0, 0.02 * np.sin(2*np.pi*freq*t[i])]))

    return t, positions, velocities, accelerations, orientations, angular_velocities


# ─────────────────────────────────────────────
#  Main simulation
# ─────────────────────────────────────────────

def run_simulation():
    """Run the full indoor localization simulation and print results."""

    print("=" * 65)
    print("  ESKF Indoor Drone Localization Simulation")
    print("  Room: 10m x 10m x 3m  |  8 UWB anchors  |  IMU @ 200 Hz")
    print("=" * 65)

    # Simulation parameters
    duration = 30.0   # seconds
    imu_dt = 0.005    # 200 Hz IMU
    uwb_rate = 10.0   # 10 Hz UWB updates
    baro_rate = 25.0  # 25 Hz barometer
    flow_rate = 30.0  # 30 Hz optical flow

    # Generate ground truth trajectory
    t_arr, pos_true, vel_true, acc_true, rot_true, omega_true = \
        generate_indoor_trajectory(duration, imu_dt)
    N = len(t_arr)

    # Setup
    env = IndoorEnvironment()
    eskf = ESKF()

    # Initialize ESKF at true starting position
    eskf.nominal.p = pos_true[0].copy()
    eskf.nominal.v = vel_true[0].copy()

    # Logging
    pos_estimates = np.zeros((N, 3))
    pos_uncertainties = np.zeros((N, 3))
    imu_only_pos = np.zeros((N, 3))

    # Also run a "dead reckoning only" version for comparison
    dr_pos = pos_true[0].copy()
    dr_vel = vel_true[0].copy()

    uwb_interval = int(1.0 / (uwb_rate * imu_dt))
    baro_interval = int(1.0 / (baro_rate * imu_dt))
    flow_interval = int(1.0 / (flow_rate * imu_dt))

    start_time = time.time()

    for i in range(N):
        t = t_arr[i]

        # ── Generate sensor data ──
        accel_meas, gyro_meas = env.simulate_imu(
            acc_true[i], omega_true[i], rot_true[i]
        )

        # ── ESKF prediction (every IMU sample) ──
        eskf.predict(accel_meas, gyro_meas, imu_dt)

        # ── Dead-reckoning prediction (for comparison) ──
        R_dr = np.eye(3)  # simplified
        a_world = R_dr @ (accel_meas - env.accel_bias) + np.array([0, 0, -9.81])
        dr_vel = dr_vel + a_world * imu_dt
        dr_pos = dr_pos + dr_vel * imu_dt

        # ── Sensor updates ──
        if i > 0 and i % uwb_interval == 0:
            uwb_ranges = env.simulate_uwb(pos_true[i])
            for anchor, range_meas in uwb_ranges:
                eskf.update_uwb_range(anchor, range_meas, env.uwb_noise_std)

        if i > 0 and i % baro_interval == 0:
            baro_alt = env.simulate_barometer(pos_true[i][2], t)
            eskf.update_altitude(baro_alt, env.baro_noise_std)

        if i > 0 and i % flow_interval == 0:
            flow_vel = env.simulate_optical_flow(vel_true[i])
            eskf.update_velocity(flow_vel, env.flow_noise_std)

        # Log
        pos_estimates[i] = eskf.get_position()
        pos_uncertainties[i] = eskf.get_position_uncertainty()
        imu_only_pos[i] = dr_pos.copy()

    elapsed = time.time() - start_time

    # ── Results ──────────────────────────────
    eskf_errors = np.linalg.norm(pos_estimates - pos_true, axis=1)
    dr_errors = np.linalg.norm(imu_only_pos - pos_true, axis=1)

    print(f"\nSimulation complete: {N} steps in {elapsed:.2f}s "
          f"({N/elapsed:.0f} steps/sec)")
    print(f"\n{'Metric':<35} {'ESKF':>10} {'IMU-only':>10}")
    print("-" * 57)
    print(f"{'Mean position error [m]':<35} {np.mean(eskf_errors):>10.4f} {np.mean(dr_errors):>10.4f}")
    print(f"{'Max position error [m]':<35} {np.max(eskf_errors):>10.4f} {np.max(dr_errors):>10.4f}")
    print(f"{'RMS position error [m]':<35} {np.sqrt(np.mean(eskf_errors**2)):>10.4f} {np.sqrt(np.mean(dr_errors**2)):>10.4f}")
    print(f"{'Final position error [m]':<35} {eskf_errors[-1]:>10.4f} {dr_errors[-1]:>10.4f}")
    print(f"{'Mean 3σ uncertainty [m]':<35} {np.mean(pos_uncertainties):>10.4f} {'N/A':>10}")

    print(f"\n  ESKF improvement over IMU-only: "
          f"{np.mean(dr_errors)/np.mean(eskf_errors):.1f}x better\n")

    # Per-axis breakdown
    print("Per-axis RMS errors (ESKF):")
    for axis, name in enumerate(['X', 'Y', 'Z']):
        rms = np.sqrt(np.mean((pos_estimates[:, axis] - pos_true[:, axis])**2))
        print(f"  {name}: {rms:.4f} m")

    print("\n" + "=" * 65)
    print("  HOW TO USE THIS FOR YOUR REAL DRONE")
    print("=" * 65)
    print("""
  1. HARDWARE YOU NEED:
     - IMU: Already on your flight controller (MPU6050, ICM20689, BMI270)
     - UWB: 4-8x Decawave DWM1001-DEV anchors (~$25 each) + 1 tag on drone
     - Barometer: Already on most FCs (BMP280, MS5611)
     - Optical flow: PMW3901 or similar (~$15) + rangefinder

  2. UWB ANCHOR PLACEMENT (critical for accuracy):
     - Mount at room corners, mix of floor-level and ceiling
     - Need 4+ anchors with good geometric spread (not all in a line)
     - More anchors = better NLOS rejection and redundancy
     - This simulation uses 8 anchors in a 10x10x3m room

  3. INTEGRATION APPROACH:
     a) Flight controller (best): Port this ESKF to run on your FC
        - ArduPilot: Modify EKF3 or add as external position source
        - PX4: Feed via MAVLink VISION_POSITION_ESTIMATE
        - Betaflight: More limited, use as external nav

     b) Companion computer (easier): Run on Raspberry Pi / Jetson
        - Read IMU via SPI/I2C or from FC via MAVLink
        - Read UWB via UART/SPI
        - Send fused position back to FC as "GPS" via MAVLink

  4. TUNING TIPS:
     - Start with UWB-only updates, add sensors one at a time
     - If position jumps: increase UWB noise_std or add outlier rejection
     - If position drifts: decrease IMU noise parameters
     - Log everything and compare against known waypoints
     - The uncertainty output tells you when the filter is confident

  5. EXPECTED ACCURACY:
     - UWB alone: ~15-30cm
     - ESKF (IMU + UWB): ~5-15cm with good anchor geometry
     - Adding optical flow: ~3-10cm, much smoother tracking
     - Adding barometer: better altitude hold, ~5-10cm vertical
""")

    # Attempt to save plot
    try:
        _save_plot(t_arr, pos_true, pos_estimates, imu_only_pos,
                   pos_uncertainties, eskf_errors, dr_errors, env)
    except ImportError:
        print("  (Install matplotlib for trajectory plots: pip install matplotlib)")
    except Exception as e:
        print(f"  (Could not save plot: {e})")


def _save_plot(t, pos_true, pos_eskf, pos_dr, uncertainties,
               eskf_errors, dr_errors, env):
    """Save a visualization of the results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ESKF Indoor Drone Localization', fontsize=14, fontweight='bold')

    # Top-left: 2D trajectory (top view)
    ax = axes[0, 0]
    ax.plot(pos_true[:, 0], pos_true[:, 1], 'g-', linewidth=2, label='Ground truth', alpha=0.7)
    ax.plot(pos_eskf[:, 0], pos_eskf[:, 1], 'b-', linewidth=1, label='ESKF estimate')
    ax.plot(pos_dr[:, 0], pos_dr[:, 1], 'r-', linewidth=0.5, alpha=0.5, label='IMU-only (drift)')
    anchors = env.uwb_anchors
    ax.scatter(anchors[:, 0], anchors[:, 1], c='orange', marker='^', s=100,
               zorder=5, label='UWB anchors')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Top View (XY plane)')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Top-right: altitude over time
    ax = axes[0, 1]
    ax.plot(t, pos_true[:, 2], 'g-', linewidth=2, label='True altitude')
    ax.plot(t, pos_eskf[:, 2], 'b-', linewidth=1, label='ESKF estimate')
    ax.fill_between(t,
                    pos_eskf[:, 2] - uncertainties[:, 2],
                    pos_eskf[:, 2] + uncertainties[:, 2],
                    alpha=0.2, color='blue', label='3σ uncertainty')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Altitude [m]')
    ax.set_title('Altitude Tracking')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-left: position error over time
    ax = axes[1, 0]
    ax.plot(t, eskf_errors, 'b-', linewidth=1, label='ESKF error')
    ax.plot(t, dr_errors, 'r-', linewidth=0.5, alpha=0.5, label='IMU-only error')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position Error [m]')
    ax.set_title('Position Error Comparison')
    ax.legend(fontsize=8)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Bottom-right: per-axis errors
    ax = axes[1, 1]
    for axis, (name, color) in enumerate(zip(['X', 'Y', 'Z'], ['r', 'g', 'b'])):
        error = np.abs(pos_eskf[:, axis] - pos_true[:, axis])
        ax.plot(t, error, color=color, linewidth=1, label=f'{name} error')
        ax.fill_between(t, 0, uncertainties[:, axis], alpha=0.1, color=color)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Error [m]')
    ax.set_title('Per-Axis ESKF Errors with 3σ Bounds')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/Games/eskf_indoor_results.png', dpi=150)
    print(f"\n  Plot saved to: eskf_indoor_results.png")


if __name__ == '__main__':
    np.random.seed(42)
    run_simulation()
