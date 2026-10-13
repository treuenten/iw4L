use glam::{Mat3, Quat, Vec3};

const SUBSTEP_S: f32 = 0.025;
const GRAVITY: f32 = 800.0;
const CONTACT_SLOP: f32 = 0.125;
const GROUND_NZ: f32 = 0.7;
pub(crate) const SLEEP_SPEED: f32 = 4.0;
pub(crate) const SLEEP_SPIN: f32 = 20.0 * std::f32::consts::PI / 180.0;
const SLEEP_S: f32 = 0.2;
const UNSTICK_MAX_LIFT: f32 = 16.0;
pub(crate) const MAX_ANG_SPEED: f32 = 15.0;
const MAX_SPEED: f32 = 800.0;
const MAX_FALL_SPEED: f32 = 3000.0;
const ESCAPE_STEPS: [f32; 13] = [
    0.25, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0,
];
const CORE_HALF: f32 = 1.0;
const BURIED_ESCAPE_COS: f32 = 0.26;
const FALL_LIMIT: f32 = 4096.0;
const BELOW_LAUNCH: f32 = 256.0;
const FACE_SNAP_COS: f32 = 0.965_925_8;
const CONTACT_PLANE_TOL: f32 = 1.0;
const REST_SPEED: f32 = 2.0 * GRAVITY * SUBSTEP_S;
const GROUND_SPIN_KEEP: f32 = 0.9;
const SETTLE_SPEED: f32 = 20.0;
const SETTLE_SPIN: f32 = 1.5;
const SETTLE_KEEP: f32 = 0.7;
const LEVEL_S: f32 = 0.1;
const BROAD_FACE: f32 = 0.5;
const TIP_RATE: f32 = 3.0;
const LATE_SETTLE_S: f32 = 3.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Phys {
    pub mass: f32,
    pub bounce: f32,
    pub friction: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Sweep {
    pub fraction: f32,
    pub end: Vec3,
    pub normal: Vec3,
    pub start_solid: bool,
}

impl Sweep {
    pub(crate) fn from_trace(trace: &trace_iw4::Trace) -> Self {
        Self {
            fraction: trace.fraction,
            end: Vec3::from_array(trace.endpos),
            normal: Vec3::from_array(trace.normal),
            start_solid: trace.startsolid != 0 || trace.allsolid != 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MapLimits {
    pub lowest_ground: f32,
    pub lo: Vec3,
    pub hi: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RigidBody {
    pub center: Vec3,
    pub rotation: Quat,
    pub velocity: Vec3,
    pub spin: Vec3,
    mid: Vec3,
    pub half: Vec3,
    phys: Phys,
    pub on_ground: bool,
    calm_s: f32,
    free: bool,
    last_free: (Vec3, Quat),
    last_valid: (Vec3, Quat),
    touched_any: bool,
    put_back: bool,
    floor_z: f32,
    bounds: Option<(Vec3, Vec3)>,
    awake_s: f32,
    lost: bool,
}

impl RigidBody {
    pub(crate) fn launch(
        origin: Vec3,
        axis: [[f32; 3]; 3],
        mid: Vec3,
        half: Vec3,
        phys: Phys,
        point: Vec3,
        force: Vec3,
    ) -> Self {
        let rotation = rotation_from_axis(axis);
        let half = half.abs().max(Vec3::splat(0.5));
        let mut body = Self {
            center: origin + rotation * mid,
            rotation,
            velocity: Vec3::ZERO,
            spin: Vec3::ZERO,
            mid,
            half,
            phys: Phys {
                mass: phys.mass.max(0.001),
                bounce: phys.bounce.clamp(0.0, 1.0),
                friction: phys.friction.clamp(0.0, 1.0),
            },
            on_ground: false,
            calm_s: 0.0,
            free: false,
            last_free: (origin + rotation * mid, rotation),
            last_valid: (origin + rotation * mid, rotation),
            touched_any: false,
            put_back: false,
            floor_z: origin.z - FALL_LIMIT,
            bounds: None,
            awake_s: 0.0,
            lost: false,
        };
        let lever = point - body.center;
        let lever = if lever.length() <= body.half.length() + CONTACT_PLANE_TOL {
            lever
        } else {
            Vec3::ZERO
        };
        body.apply_impulse(force, lever);
        body.clamp_spin();
        body
    }

    pub(crate) fn keep_within(&mut self, limits: MapLimits) {
        self.floor_z = limits.lowest_ground.min(self.center.z - BELOW_LAUNCH);
        self.bounds = Some((limits.lo, limits.hi));
    }

    pub(crate) fn origin(&self) -> Vec3 {
        self.center - self.rotation * self.mid
    }

    pub(crate) fn axis(&self) -> [[f32; 3]; 3] {
        let m = Mat3::from_quat(self.rotation);
        [
            m.x_axis.to_array(),
            m.y_axis.to_array(),
            m.z_axis.to_array(),
        ]
    }

    pub(crate) fn world_half(&self) -> Vec3 {
        world_half(self.rotation, self.half)
    }

    pub(crate) fn push(&mut self, impulse: Vec3, lever: Vec3, spin_share: f32) {
        self.velocity += impulse / self.phys.mass;
        self.spin += self.inverse_inertia_world() * lever.cross(impulse * spin_share);
        self.clamp_spin();
        self.calm_s = 0.0;
        self.awake_s = 0.0;
        self.on_ground = false;
    }

    pub(crate) fn step(
        &mut self,
        dt: f32,
        sweep: &mut impl FnMut(Vec3, Vec3, Vec3) -> Sweep,
        solid: &mut impl FnMut(Vec3, Vec3, Vec3) -> Sweep,
    ) {
        let steps = (dt / SUBSTEP_S).round().max(1.0) as u32;
        let sub = dt / steps as f32;
        for _ in 0..steps {
            let before = *self;
            self.advance(sub, sweep, solid);
            if !self.is_finite() {
                *self = before;
                self.velocity = Vec3::ZERO;
                self.spin = Vec3::ZERO;
                self.calm_s = SLEEP_S;
                break;
            }
        }
    }

    fn is_finite(&self) -> bool {
        self.center.is_finite()
            && self.rotation.is_finite()
            && self.velocity.is_finite()
            && self.spin.is_finite()
    }

    pub(crate) fn ready_to_sleep(&self) -> bool {
        self.calm_s >= SLEEP_S
    }

    fn advance(
        &mut self,
        dt: f32,
        trace: &mut impl FnMut(Vec3, Vec3, Vec3) -> Sweep,
        solid: &mut impl FnMut(Vec3, Vec3, Vec3) -> Sweep,
    ) {
        self.velocity.z -= GRAVITY * dt;
        self.clamp_speed(MAX_FALL_SPEED);
        self.awake_s += dt;
        if self.lost {
            self.stop_at_last_free(dt);
            return;
        }
        let old_rotation = self.rotation;
        let old_half = self.sweep_half(old_rotation);
        if !self.free && !self.get_free(trace, solid, old_half) {
            self.lost = true;
            self.stop_at_last_free(dt);
            return;
        }
        let mut rotation = turned(old_rotation, self.spin, dt);
        let mut half = self.sweep_half(rotation);
        let mut start = self.center;
        let mut rested = None;
        if rotation != old_rotation {
            // Terrain is one-sided: a box that starts more than its radius
            // under a triangle never meets it again, so a turn that lowers
            // the bottom turns about whatever is under it.
            let grown = half.z - old_half.z;
            if grown > 0.0 {
                let top = start + Vec3::Z * (grown + CONTACT_SLOP);
                let probe = trace(top, start, half);
                if !probe.start_solid && probe.fraction < 1.0 {
                    start = probe.end;
                    rested = Some(probe.normal.try_normalize().unwrap_or(Vec3::Z));
                }
            }
            if !fits(trace, start, half) {
                rotation = old_rotation;
                half = old_half;
                start = self.center;
                rested = None;
                self.spin = Vec3::ZERO;
            }
        }
        let delta = self.velocity * dt;
        let mut hit = trace(start, start + delta, half);
        if hit.start_solid && rotation != old_rotation {
            let unturned = trace(self.center, self.center + delta, old_half);
            if !unturned.start_solid {
                rotation = old_rotation;
                half = old_half;
                start = self.center;
                rested = None;
                self.spin = Vec3::ZERO;
                hit = unturned;
            }
        }
        if hit.start_solid
            && let Some(lifted) = unstick(trace, start, half)
        {
            start = lifted;
            rested = Some(Vec3::Z);
            hit = trace(start, start + delta, half);
        }
        if hit.start_solid {
            self.stop_at_last_free(dt);
            return;
        }
        self.rotation = rotation;
        let before = start;
        let mut touched = hit.fraction < 1.0 || rested.is_some();
        if hit.fraction < 1.0 {
            let normal = hit.normal.try_normalize().unwrap_or(Vec3::Z);
            self.center = hit.end + separation(normal);
            self.collide(normal);
            self.on_ground = normal.z > GROUND_NZ;
            let rest = (1.0 - hit.fraction.max(0.0)) * dt;
            if rest > dt * 0.01 {
                let slide = trace(self.center, self.center + self.velocity * rest, half);
                if !slide.start_solid {
                    if slide.fraction < 1.0 {
                        let normal = slide.normal.try_normalize().unwrap_or(Vec3::Z);
                        self.center = slide.end + separation(normal);
                        self.collide(normal);
                        self.on_ground |= normal.z > GROUND_NZ;
                    } else {
                        self.center = slide.end;
                    }
                }
            }
        } else {
            self.center = hit.end;
            self.on_ground = rested.is_some_and(|normal| normal.z > GROUND_NZ);
            if let Some(normal) = rested {
                // Being raised stands in for turning about the corner that
                // touched; without its contact every lift adds energy.
                self.collide(normal);
            }
        }
        // A sweep that meets terrain forgets that it started inside a brush,
        // so a pose it reports clear can still overlap one.
        if self.center != before && !fits(trace, self.center, half) {
            self.center = before;
            touched = true;
        }
        if self.left_the_map() {
            self.go_back(dt);
            return;
        }
        self.last_free = (self.center, self.rotation);
        if touched {
            self.last_valid = (self.center, self.rotation);
            self.touched_any = true;
            self.keep_to_what_was_achieved(delta, self.center - before, dt);
        }
        let slow = self.velocity.length() < SETTLE_SPEED && self.spin.length() < SETTLE_SPIN;
        if self.on_ground && (slow || self.awake_s > LATE_SETTLE_S) {
            self.velocity *= SETTLE_KEEP;
            self.spin *= SETTLE_KEEP;
        }
        let moved = (self.center - before).length() / dt;
        let turned_rate = old_rotation.angle_between(self.rotation) / dt;
        let calm = touched
            && moved.min(self.velocity.length()) < SLEEP_SPEED
            && turned_rate.min(self.spin.length()) < SLEEP_SPIN;
        self.calm_s = if calm { self.calm_s + dt } else { 0.0 };
    }

    fn keep_to_what_was_achieved(&mut self, intended: Vec3, achieved: Vec3, dt: f32) {
        let Some(direction) = intended.try_normalize() else {
            return;
        };
        let allowed = achieved.dot(direction).max(0.0) / dt;
        let along = self.velocity.dot(direction);
        if along > allowed {
            self.velocity -= direction * (along - allowed);
        }
    }

    fn get_free(
        &mut self,
        trace: &mut impl FnMut(Vec3, Vec3, Vec3) -> Sweep,
        solid: &mut impl FnMut(Vec3, Vec3, Vec3) -> Sweep,
        half: Vec3,
    ) -> bool {
        let at = if fits(trace, self.center, half) {
            Some(self.center)
        } else {
            self.escape(trace, solid, half)
        };
        if let Some(at) = at {
            self.center = at;
            self.free = true;
            self.last_free = (at, self.rotation);
            self.last_valid = (at, self.rotation);
        }
        self.free
    }

    fn escape(
        &self,
        trace: &mut impl FnMut(Vec3, Vec3, Vec3) -> Sweep,
        solid: &mut impl FnMut(Vec3, Vec3, Vec3) -> Sweep,
        half: Vec3,
    ) -> Option<Vec3> {
        let at = self.center;
        let launch = self.velocity.with_z(self.velocity.z.max(0.0));
        let forward = (self.rotation * Vec3::X)
            .with_z(0.0)
            .try_normalize()
            .unwrap_or(Vec3::X);
        let side = Vec3::Z.cross(forward);
        let level = launch.with_z(0.0).normalize_or_zero();
        let mut ways = [Vec3::ZERO; 11];
        ways[0] = launch.try_normalize().unwrap_or(Vec3::ZERO);
        ways[1] = Vec3::Z;
        if level != Vec3::ZERO {
            ways[2] = (level + Vec3::Z).normalize();
        }
        for (k, around) in [forward, side, -forward, -side].into_iter().enumerate() {
            ways[3 + k] = around;
            ways[7 + k] = (around + Vec3::Z).normalize();
        }
        let core = Vec3::splat(CORE_HALF);
        let buried = solid(at, at, core).start_solid;
        for reach in ESCAPE_STEPS {
            for way in ways {
                if way == Vec3::ZERO
                    || (reach < 2.0 && way != Vec3::Z)
                    || (buried && way.dot(level) > BURIED_ESCAPE_COS)
                {
                    continue;
                }
                let to = at + way * reach;
                if fits(trace, to, half)
                    && solid(at, to, core).fraction >= 1.0
                    && !crosses_down(solid(at, to, half))
                {
                    return Some(to);
                }
            }
        }
        None
    }

    fn left_the_map(&self) -> bool {
        let c = self.center;
        c.z < self.floor_z
            || self.bounds.is_some_and(|(lo, hi)| {
                c.x < lo.x || c.y < lo.y || c.z < lo.z || c.x > hi.x || c.y > hi.y
            })
    }

    fn go_back(&mut self, dt: f32) {
        (self.center, self.rotation) = self.last_valid;
        self.last_free = self.last_valid;
        self.velocity = Vec3::ZERO;
        self.spin = Vec3::ZERO;
        if self.put_back || !self.touched_any {
            self.lost = true;
        }
        self.put_back = true;
        self.calm_s += dt;
    }

    fn stop_at_last_free(&mut self, dt: f32) {
        (self.center, self.rotation) = self.last_free;
        self.velocity = Vec3::ZERO;
        self.spin = Vec3::ZERO;
        self.calm_s += dt;
    }

    // The world trace takes the box's x extent as its capsule radius.
    fn sweep_half(&self, rotation: Quat) -> Vec3 {
        let half = world_half(rotation, self.half);
        let reach = half.x.max(half.y);
        Vec3::new(reach, reach, half.z)
    }

    fn collide(&mut self, normal: Vec3) {
        let ground = normal.z > GROUND_NZ;
        if ground {
            self.spin *= GROUND_SPIN_KEEP;
        }
        let mut lever = contact_lever(self.rotation, self.half, normal);
        let inverse = self.inverse_inertia_world();
        let approach = self.point_velocity(lever).dot(normal);
        if approach >= 0.0 {
            return;
        }
        let resting = -approach < REST_SPEED;
        let bounce = if resting { 0.0 } else { self.phys.bounce };
        let lying = (ground && resting)
            .then(|| stable_face(self.rotation, self.half, normal))
            .flatten();
        if lying.is_some() {
            lever = normal * lever.dot(normal);
        }
        let carried = (-self.velocity.dot(normal)).max(0.0) * self.phys.mass;
        let k_normal = effective_inverse_mass(self.phys.mass, inverse, lever, normal);
        let jn = -(1.0 + bounce) * approach / k_normal;
        self.apply_impulse(normal * jn, lever);
        if ground && resting {
            let tangent = self.velocity - normal * self.velocity.dot(normal);
            let slip = tangent.length();
            if slip > 1e-4 {
                let held = if self.awake_s > LATE_SETTLE_S {
                    slip
                } else {
                    slip.min(self.phys.friction * carried / self.phys.mass)
                };
                self.velocity -= tangent / slip * held;
            }
        } else {
            let relative = self.point_velocity(lever);
            let tangent = relative - normal * relative.dot(normal);
            let slip = tangent.length();
            if slip > 1e-4 {
                let direction = tangent / slip;
                let k_tangent = effective_inverse_mass(self.phys.mass, inverse, lever, direction);
                let jt = (slip / k_tangent).min(self.phys.friction * jn);
                self.apply_impulse(-direction * jt, lever);
            }
        }
        let again = self.point_velocity(lever).dot(normal);
        if again < 0.0 {
            self.apply_impulse(normal * (-again / k_normal), lever);
        }
        if let Some((axis, target)) = lying {
            let level = axis.cross(target);
            let angle = axis.dot(target).clamp(-1.0, 1.0).acos();
            let level = level.try_normalize().unwrap_or(Vec3::ZERO) * angle / LEVEL_S;
            self.spin = normal * self.spin.dot(normal) + level;
        } else if ground
            && resting
            && self.velocity.length() < SETTLE_SPEED
            && self.spin.length() < SETTLE_SPIN
            && let Some(tip) = toward_broad_face(self.rotation, self.half, normal)
        {
            self.spin = normal * self.spin.dot(normal) + tip * TIP_RATE;
        }
        self.clamp_spin();
    }

    fn point_velocity(&self, lever: Vec3) -> Vec3 {
        self.velocity + self.spin.cross(lever)
    }

    pub(crate) fn settle(&mut self, sweep: &mut impl FnMut(Vec3, Vec3, Vec3) -> Sweep) {
        self.velocity = Vec3::ZERO;
        self.spin = Vec3::ZERO;
        let old_rotation = self.rotation;
        let old_half = self.sweep_half(old_rotation);
        let flat = flatten(self.rotation);
        self.rotation = flat;
        let half = self.sweep_half(flat);
        let lift = (half.z - old_half.z).max(0.0) + CONTACT_SLOP;
        let start = self.center + Vec3::Z * lift;
        let down = sweep(start, start - Vec3::Z * (lift + UNSTICK_MAX_LIFT), half);
        if down.start_solid {
            self.rotation = old_rotation;
            return;
        }
        let old_center = self.center;
        if down.fraction < 1.0 && down.normal.z > GROUND_NZ {
            self.center = down.end + Vec3::Z * CONTACT_SLOP;
        } else if flat != old_rotation {
            self.rotation = old_rotation;
        }
        if (self.center, self.rotation) != (old_center, old_rotation)
            && !fits(sweep, self.center, self.sweep_half(self.rotation))
        {
            (self.center, self.rotation) = (old_center, old_rotation);
        }
    }

    fn apply_impulse(&mut self, impulse: Vec3, lever: Vec3) {
        self.velocity += impulse / self.phys.mass;
        self.spin += self.inverse_inertia_world() * lever.cross(impulse);
    }

    fn clamp_spin(&mut self) {
        let speed = self.spin.length();
        if speed > MAX_ANG_SPEED {
            self.spin *= MAX_ANG_SPEED / speed;
        }
        self.clamp_speed(MAX_SPEED);
    }

    fn clamp_speed(&mut self, max: f32) {
        let speed = self.velocity.length();
        if speed > max {
            self.velocity *= max / speed;
        }
    }

    fn inverse_inertia_world(&self) -> Mat3 {
        let h2 = self.half * self.half;
        let m = self.phys.mass / 3.0;
        let local = Vec3::new(
            1.0 / (m * (h2.y + h2.z)),
            1.0 / (m * (h2.x + h2.z)),
            1.0 / (m * (h2.x + h2.y)),
        );
        let r = Mat3::from_quat(self.rotation);
        r * Mat3::from_diagonal(local) * r.transpose()
    }
}

pub(crate) fn rotation_from_axis(axis: [[f32; 3]; 3]) -> Quat {
    Quat::from_mat3(&Mat3::from_cols(
        Vec3::from_array(axis[0]),
        Vec3::from_array(axis[1]),
        Vec3::from_array(axis[2]),
    ))
    .normalize()
}

fn world_half(rotation: Quat, half: Vec3) -> Vec3 {
    let m = Mat3::from_quat(rotation);
    m.x_axis.abs() * half.x + m.y_axis.abs() * half.y + m.z_axis.abs() * half.z
}

fn turned(rotation: Quat, spin: Vec3, dt: f32) -> Quat {
    let angle = spin.length() * dt;
    if angle <= 1e-6 {
        return rotation;
    }
    (Quat::from_axis_angle(spin / spin.length(), angle) * rotation).normalize()
}

fn separation(normal: Vec3) -> Vec3 {
    if normal.z > GROUND_NZ {
        Vec3::Z * (CONTACT_SLOP / normal.z)
    } else {
        normal * CONTACT_SLOP
    }
}

fn crosses_down(hit: Sweep) -> bool {
    hit.fraction < 1.0 && hit.normal.z >= 0.0
}

fn fits(trace: &mut impl FnMut(Vec3, Vec3, Vec3) -> Sweep, at: Vec3, half: Vec3) -> bool {
    !trace(at, at, half).start_solid
}

fn unstick(
    trace: &mut impl FnMut(Vec3, Vec3, Vec3) -> Sweep,
    at: Vec3,
    half: Vec3,
) -> Option<Vec3> {
    let mut dz = CONTACT_SLOP;
    while dz <= UNSTICK_MAX_LIFT {
        let lifted = at + Vec3::Z * dz;
        if fits(trace, lifted, half) {
            return Some(lifted);
        }
        dz *= 2.0;
    }
    None
}

fn contact_lever(rotation: Quat, half: Vec3, normal: Vec3) -> Vec3 {
    let corners: [Vec3; 8] = std::array::from_fn(|i| {
        let sign = Vec3::new(
            if i & 1 == 0 { -1.0 } else { 1.0 },
            if i & 2 == 0 { -1.0 } else { 1.0 },
            if i & 4 == 0 { -1.0 } else { 1.0 },
        );
        rotation * (half * sign)
    });
    let deepest = corners
        .iter()
        .map(|c| c.dot(normal))
        .fold(f32::INFINITY, f32::min);
    let (sum, n) = corners
        .iter()
        .filter(|c| c.dot(normal) <= deepest + CONTACT_PLANE_TOL)
        .fold((Vec3::ZERO, 0.0), |(sum, n), c| (sum + *c, n + 1.0));
    sum / n
}

fn effective_inverse_mass(mass: f32, inverse: Mat3, lever: Vec3, direction: Vec3) -> f32 {
    1.0 / mass + direction.dot((inverse * lever.cross(direction)).cross(lever))
}

fn nearest_face(rotation: Quat, up: Vec3) -> Option<(Vec3, Vec3)> {
    let m = Mat3::from_quat(rotation);
    let (axis, cos) = [m.x_axis, m.y_axis, m.z_axis]
        .into_iter()
        .map(|a| (a, a.dot(up).abs()))
        .fold((up, -1.0), |best, c| if c.1 > best.1 { c } else { best });
    (cos >= FACE_SNAP_COS).then(|| (axis, up * axis.dot(up).signum()))
}

fn stable_face(rotation: Quat, half: Vec3, up: Vec3) -> Option<(Vec3, Vec3)> {
    let (axis, target) = nearest_face(rotation, up)?;
    let m = Mat3::from_quat(rotation);
    let along = [m.x_axis, m.y_axis, m.z_axis].map(|a| a.dot(axis).abs());
    let i = (0..3).fold(0, |best, k| if along[k] > along[best] { k } else { best });
    let height = half[i];
    let side = (0..3)
        .filter(|&k| k != i)
        .map(|k| half[k])
        .fold(f32::INFINITY, f32::min);
    (side >= BROAD_FACE * height).then_some((axis, target))
}

fn toward_broad_face(rotation: Quat, half: Vec3, up: Vec3) -> Option<Vec3> {
    let m = Mat3::from_quat(rotation);
    let axes = [m.x_axis, m.y_axis, m.z_axis];
    let broad = |i: usize| {
        let side = (0..3)
            .filter(|&k| k != i)
            .map(|k| half[k])
            .fold(f32::INFINITY, f32::min);
        side >= BROAD_FACE * half[i]
    };
    let face = (0..3)
        .filter(|&i| broad(i))
        .max_by(|&a, &b| axes[a].dot(up).abs().total_cmp(&axes[b].dot(up).abs()))?;
    let axis = axes[face] * if axes[face].dot(up) < 0.0 { -1.0 } else { 1.0 };
    axis.cross(up).try_normalize()
}

fn flatten(rotation: Quat) -> Quat {
    match nearest_face(rotation, Vec3::Z) {
        Some((axis, target)) if axis.dot(target) < 1.0 => {
            (Quat::from_rotation_arc(axis, target) * rotation).normalize()
        }
        _ => rotation,
    }
}

pub(crate) fn swept_box_overlaps(
    start: Vec3,
    end: Vec3,
    mins: Vec3,
    maxs: Vec3,
    lo: Vec3,
    hi: Vec3,
) -> bool {
    let lo = lo - maxs;
    let hi = hi - mins;
    let delta = end - start;
    let (mut enter, mut exit) = (0.0f32, 1.0f32);
    for i in 0..3 {
        let (s, d) = (start[i], delta[i]);
        if d.abs() < 1e-6 {
            if s <= lo[i] || s >= hi[i] {
                return false;
            }
            continue;
        }
        let (a, b) = ((lo[i] - s) / d, (hi[i] - s) / d);
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        enter = enter.max(a);
        exit = exit.min(b);
        if enter >= exit {
            return false;
        }
    }
    true
}
