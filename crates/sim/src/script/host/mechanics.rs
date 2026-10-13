use super::entities::Link;
use super::rigid_body::{MapLimits, RigidBody, Sweep};
use crate::script::runtime::raise;
use crate::script::{BTreeMap, Resource, RoundScript, Value};
use bevy_ecs::prelude::World;
use glam::Vec3;

/// Script-driven entity mechanics: timed moves, launched physics bodies and
/// entity/player links. They run before the VM each tick and settle native
/// collision, so a thread woken by movedone or physics_finished already
/// queries the final pose.
#[derive(Resource, Clone, Default)]
pub(crate) struct Mechanics {
    motions: BTreeMap<u64, Vec<Motion>>,
    bodies: BTreeMap<u64, Body>,
    slides: BTreeMap<u64, Slide>,
    finished: Vec<(u64, &'static str)>,
    settled: Vec<u64>,
    fallback_logged: std::collections::BTreeSet<String>,
    limits: Option<MapLimits>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Body {
    motion: BodyMotion,
    ticks: u32,
    active: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum BodyMotion {
    Piece {
        velocity: [f32; 3],
        offset: [f32; 3],
        mins: [f32; 3],
        maxs: [f32; 3],
    },
    Rigid {
        body: RigidBody,
        client: bool,
        explosive_force_scale: f32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LaunchSpec {
    pub origin: [f32; 3],
    pub angles: [f32; 3],
    pub point: [f32; 3],
    pub force: [f32; 3],
    pub mid: [f32; 3],
    pub half: [f32; 3],
    pub phys: xmodel_runtime::ModelPhys,
    pub client: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BodyTrajectory {
    pub velocity: [f32; 3],
    pub angle_rates: [f32; 3],
    pub falling: bool,
}

const TICK_S: f32 = crate::MATCH_TICK_MS as f32 / 1000.0;
const SETTLE_TICKS: u32 = 200;
const EXPLOSION_FORCE: f32 = 12500.0;
const EXPLOSION_UPBIAS: f32 = 0.5;
const EXPLOSION_MIN_FORCE: f32 = 40.0;
const EXPLOSION_SPIN_SHARE: f32 = 0.25;
const CRUSH_MIN_DROP: f32 = 1.0;
const LOWEST_GROUND_BELOW_SPAWN: f32 = 512.0;
const MAP_BOUNDS_MARGIN: f32 = 512.0;
const SOLID_WORLD: u32 = crate::bullet_collision::CONTENTS_SOLID | CONTENTS_GLASS;
const CONTENTS_GLASS: u32 = 0x0000_0010;

impl Mechanics {
    pub(crate) fn start(&mut self, object: u64, motion: Motion) {
        if motion.field == "origin" {
            self.bodies.remove(&object);
            self.slides.remove(&object);
        }
        let motions = self.motions.entry(object).or_default();
        motions.retain(|m| m.field != motion.field);
        motions.push(motion);
    }

    pub(crate) fn stop(&mut self, object: u64, field: &str) {
        if let Some(motions) = self.motions.get_mut(&object) {
            motions.retain(|m| m.field != field);
        }
    }

    pub(crate) fn launch(&mut self, object: u64, spec: LaunchSpec) {
        self.motions.remove(&object);
        self.slides.remove(&object);
        let mut body = RigidBody::launch(
            Vec3::from_array(spec.origin),
            math_iw4::angles_to_axis(spec.angles),
            Vec3::from_array(spec.mid),
            Vec3::from_array(spec.half),
            super::rigid_body::Phys {
                mass: spec.phys.mass,
                bounce: spec.phys.bounce,
                friction: spec.phys.friction,
            },
            Vec3::from_array(spec.point),
            Vec3::from_array(spec.force),
        );
        if let Some(limits) = self.limits {
            body.keep_within(limits);
        }
        self.bodies.insert(
            object,
            Body {
                motion: BodyMotion::Rigid {
                    body,
                    client: spec.client,
                    explosive_force_scale: spec.phys.explosive_force_scale,
                },
                ticks: 0,
                active: true,
            },
        );
    }

    pub(crate) fn launch_piece(
        &mut self,
        object: u64,
        velocity: [f32; 3],
        offset: [f32; 3],
        half: [f32; 3],
    ) {
        self.bodies.insert(
            object,
            Body {
                motion: BodyMotion::Piece {
                    velocity,
                    offset,
                    mins: half.map(|h| -h),
                    maxs: half,
                },
                ticks: 0,
                active: true,
            },
        );
    }

    fn ensure_limits(world: &mut World) {
        if world.resource::<Mechanics>().limits.is_none() {
            let limits = map_limits(world);
            world.resource_mut::<Mechanics>().limits = Some(limits);
        }
    }

    fn note_fallback_phys(&mut self, model: &str) -> bool {
        self.fallback_logged.insert(model.to_owned())
    }

    pub(crate) fn bodies_to_present(&self) -> std::collections::BTreeSet<u64> {
        self.bodies
            .iter()
            .filter(|(_, b)| b.active && matches!(b.motion, BodyMotion::Rigid { .. }))
            .map(|(object, _)| *object)
            .chain(self.settled.iter().copied())
            .collect()
    }

    pub(crate) fn body_trajectory(&self, object: u64) -> Option<BodyTrajectory> {
        let body = self.bodies.get(&object)?;
        let BodyMotion::Rigid { body: rigid, .. } = &body.motion else {
            return None;
        };
        let still = BodyTrajectory {
            velocity: [0.0; 3],
            angle_rates: [0.0; 3],
            falling: false,
        };
        if !body.active {
            return Some(still);
        }
        // A pose that stops changing is not sent again, so a rate left on a
        // resting body would drift on clients for good.
        let slow = rigid.on_ground && rigid.velocity.length() < super::rigid_body::SLEEP_SPEED;
        let calm = rigid.on_ground && rigid.spin.length() < super::rigid_body::SLEEP_SPIN;
        let mut velocity = rigid.velocity;
        if rigid.on_ground {
            velocity.z = velocity.z.max(0.0);
        }
        Some(BodyTrajectory {
            velocity: if slow {
                still.velocity
            } else {
                velocity.to_array()
            },
            angle_rates: if calm {
                still.angle_rates
            } else {
                angle_rates(rigid)
            },
            falling: !rigid.on_ground,
        })
    }

    pub(crate) fn cancel(&mut self, object: u64) {
        self.motions.remove(&object);
        self.bodies.remove(&object);
        self.slides.remove(&object);
    }

    pub(crate) fn slide(&mut self, object: u64, slide: Slide) {
        self.stop(object, "origin");
        self.bodies.remove(&object);
        self.slides.insert(object, slide);
    }

    pub(crate) fn sliding(&self, object: u64) -> bool {
        self.slides.contains_key(&object)
    }

    pub(crate) fn stop_slide(&mut self, object: u64) {
        self.slides.remove(&object);
    }

    pub(crate) fn velocity(&self, object: u64, now: i64) -> [f32; 3] {
        if let Some(slide) = self.slides.get(&object) {
            return slide.velocity;
        }
        if let Some(body) = self.bodies.get(&object) {
            return match body.motion {
                BodyMotion::Piece { velocity, .. } => velocity,
                BodyMotion::Rigid { body, .. } => body.velocity.to_array(),
            };
        }
        self.motions
            .get(&object)
            .and_then(|motions| motions.iter().find(|m| m.field == "origin"))
            .map_or([0.0; 3], |m| m.sample(now).1)
    }

    pub(crate) fn explode(
        &mut self,
        runtime: &mut RoundScript,
        center: [f32; 3],
        outer: f32,
        inner: f32,
        magnitude: f32,
    ) {
        for (object, body) in &mut self.bodies {
            if !runtime.entities.contains_key(object) {
                continue;
            }
            let at = match &body.motion {
                BodyMotion::Rigid { body, .. } => body.center.to_array(),
                BodyMotion::Piece { .. } => match runtime.object_field(*object, "origin") {
                    Value::Vector(origin) => origin,
                    _ => continue,
                },
            };
            let scale = match &body.motion {
                BodyMotion::Rigid {
                    explosive_force_scale,
                    ..
                } => *explosive_force_scale,
                BodyMotion::Piece { .. } => 1.0,
            };
            let Some(impulse) = explosion_impulse(
                Vec3::from_array(at),
                Vec3::from_array(center),
                outer,
                inner,
                magnitude,
                scale,
            ) else {
                continue;
            };
            let delta = Vec3::from_array(at) - Vec3::from_array(center);
            match &mut body.motion {
                BodyMotion::Piece { velocity, .. } => {
                    *velocity = (glam::Vec3::from_array(*velocity) + impulse).to_array();
                }
                BodyMotion::Rigid { body, .. } => {
                    let toward = -delta.try_normalize().unwrap_or(glam::Vec3::NEG_Z);
                    let lever = toward * body.half.min_element();
                    body.push(impulse, lever, EXPLOSION_SPIN_SHARE);
                }
            }
            body.ticks = 0;
            body.active = true;
        }
    }
}

fn explosion_impulse(
    at: Vec3,
    center: Vec3,
    outer: f32,
    inner: f32,
    magnitude: f32,
    scale: f32,
) -> Option<Vec3> {
    if !at.is_finite() {
        return None;
    }
    let delta = at - center;
    let distance = delta.length();
    if outer <= 0.0 || distance >= outer {
        return None;
    }
    let fraction = if distance <= inner {
        1.0
    } else {
        (outer - distance) / (outer - inner)
    };
    let force = magnitude * fraction * EXPLOSION_FORCE * scale;
    if !force.is_finite() || force < EXPLOSION_MIN_FORCE {
        return None;
    }
    let mut direction = delta.try_normalize().unwrap_or(Vec3::Z);
    direction.z += EXPLOSION_UPBIAS;
    Some(direction.normalize() * force)
}

fn crushes(client: bool, drop: f32) -> bool {
    !client && drop > CRUSH_MIN_DROP
}

fn angle_rates(body: &RigidBody) -> [f32; 3] {
    const LOOK_AHEAD_S: f32 = 0.01;
    const POLE_PITCH: f32 = 80.0;
    const POLE_STRAY_DEG: f32 = 6.0;
    if body.spin.length_squared() < 1e-8 {
        return [0.0; 3];
    }
    let turned = |dt: f32| {
        (glam::Quat::from_axis_angle(body.spin.normalize(), body.spin.length() * dt)
            * body.rotation)
            .normalize()
    };
    let euler = |rotation: glam::Quat| {
        let m = glam::Mat3::from_quat(rotation);
        math_iw4::axis_to_angles([
            m.x_axis.to_array(),
            m.y_axis.to_array(),
            m.z_axis.to_array(),
        ])
    };
    let now = euler(body.rotation);
    let later = euler(turned(LOOK_AHEAD_S));
    let near_pole = |angles: [f32; 3]| math_iw4::angle_subtract(angles[0], 0.0).abs() > POLE_PITCH;
    if near_pole(now) || near_pole(later) {
        let next = euler(turned(TICK_S));
        let rates: [f32; 3] =
            std::array::from_fn(|i| math_iw4::angle_subtract(next[i], now[i]) / TICK_S);
        let stays = [0.25f32, 0.5, 0.75].into_iter().all(|share| {
            let t = TICK_S * share;
            let along = super::rigid_body::rotation_from_axis(math_iw4::angles_to_axis(
                std::array::from_fn(|i| now[i] + rates[i] * t),
            ));
            along.angle_between(turned(t)) < POLE_STRAY_DEG.to_radians()
        });
        return if stays { rates } else { [0.0; 3] };
    }
    let cap = 2.0 * super::rigid_body::MAX_ANG_SPEED.to_degrees();
    std::array::from_fn(|i| {
        (math_iw4::angle_subtract(later[i], now[i]) / LOOK_AHEAD_S).clamp(-cap, cap)
    })
}

fn launch_angles(angles: [f32; 3], client: bool) -> [f32; 3] {
    if client {
        angles
    } else {
        [0.0, angles[1], 0.0]
    }
}

fn map_limits(world: &mut World) -> MapLimits {
    let lowest_spawn = {
        let mut runtime = world.resource_mut::<RoundScript>();
        let spawns: Vec<(u64, bool)> = runtime
            .entities
            .iter()
            .filter_map(|(id, e)| {
                let class: &str = &e.classname;
                let tdm = class == "mp_tdm_spawn";
                (tdm || (class.starts_with("mp_") && class.contains("_spawn")))
                    .then_some((*id, tdm))
            })
            .collect();
        let (mut tdm, mut any) = (f32::INFINITY, f32::INFINITY);
        for (id, is_tdm) in spawns {
            if let Value::Vector(origin) = runtime.object_field(id, "origin")
                && origin[2].is_finite()
            {
                any = any.min(origin[2]);
                if is_tdm {
                    tdm = tdm.min(origin[2]);
                }
            }
        }
        if tdm.is_finite() { tdm } else { any }
    };
    let frame = crate::frame::FrameWorld::from_world(world);
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for vert in &frame.clip_mesh().tables.verts {
        let v = Vec3::from_array(*vert);
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    for brush in frame.clip_brushes() {
        if brush.contents & crate::bullet_collision::MASK_PHYS_WORLD == 0 {
            continue;
        }
        let mut bmin = Vec3::splat(f32::NEG_INFINITY);
        let mut bmax = Vec3::splat(f32::INFINITY);
        for plane in &brush.planes {
            for axis in 0..3 {
                let along = plane[axis];
                let across = (0..3).filter(|&k| k != axis).all(|k| plane[k].abs() < 1e-4);
                if across && along > 0.9999 {
                    bmax[axis] = bmax[axis].min(plane[3]);
                } else if across && along < -0.9999 {
                    bmin[axis] = bmin[axis].max(-plane[3]);
                }
            }
        }
        if bmin.is_finite() && bmax.is_finite() && bmin.cmple(bmax).all() {
            lo = lo.min(bmin);
            hi = hi.max(bmax);
        }
    }
    let (lo, hi) = if lo.is_finite() && hi.is_finite() {
        (
            lo - Vec3::splat(MAP_BOUNDS_MARGIN),
            hi + Vec3::splat(MAP_BOUNDS_MARGIN),
        )
    } else {
        (Vec3::splat(-1.0e6), Vec3::splat(1.0e6))
    };
    let lowest_ground = if lowest_spawn.is_finite() {
        lowest_spawn - LOWEST_GROUND_BELOW_SPAWN
    } else {
        lo.z
    };
    MapLimits {
        lowest_ground,
        lo,
        hi,
    }
}

const FALLBACK_PHYS: xmodel_runtime::ModelPhys = xmodel_runtime::ModelPhys {
    mass: 25.0,
    bounce: 0.1,
    friction: 0.25,
    explosive_force_scale: 0.18,
};
const FALLBACK_BOX: ([f32; 3], [f32; 3]) = ([0.0, 0.0, 12.0], [12.0; 3]);

pub(crate) fn physics_launch(
    world: &mut World,
    receiver: &Value,
    args: &[Value],
    client: bool,
) -> Result<Value, String> {
    let object = super::natives::engine::entity_id(world, receiver)?;
    let point = super::args::vector(args, 0)?;
    let force = super::args::vector(args, 1)?;
    if point.iter().chain(&force).any(|v| !v.is_finite()) {
        return Err("physics launch requires a finite point and force".into());
    }
    Mechanics::ensure_limits(world);
    let (model, origin, angles) = {
        let mut runtime = world.resource_mut::<RoundScript>();
        let model = match runtime.object_field(object, "model") {
            Value::String(model) => model.to_string(),
            _ => String::new(),
        };
        let vector = |runtime: &mut RoundScript, name| match runtime.object_field(object, name) {
            Value::Vector(v) => v,
            _ => [0.0; 3],
        };
        let origin = vector(&mut runtime, "origin");
        let angles = vector(&mut runtime, "angles");
        (model, origin, angles)
    };
    let capability = crate::frame::FrameWorld::from_world(world)
        .model_capability(&model)
        .flatten();
    let (mid, half) = capability
        .as_ref()
        .and_then(|c| c.bounds)
        .unwrap_or(FALLBACK_BOX);
    let phys = match capability.as_ref().and_then(|c| c.phys) {
        Some(phys) => phys,
        None => {
            if world.resource_mut::<Mechanics>().note_fallback_phys(&model) {
                diag::warn!(
                    Sim,
                    "physics launch: model `{model}` has no physics preset with a mass; using the car door preset"
                );
            }
            FALLBACK_PHYS
        }
    };
    let angles = launch_angles(angles, client);
    world.resource_mut::<Mechanics>().launch(
        object,
        LaunchSpec {
            origin,
            angles,
            point,
            force,
            mid,
            half,
            phys,
            client,
        },
    );
    let mut runtime = world.resource_mut::<RoundScript>();
    let entity = runtime.entities.get_mut(&object).unwrap();
    entity.linked_to = None;
    if client {
        entity.solid = false;
    } else {
        runtime.set_object_field(object, "angles", Value::Vector(angles));
    }
    Ok(Value::Undefined)
}

pub(crate) fn advance_mechanics(world: &mut World) {
    let request = world.resource::<crate::step::StepRequest>();
    if !request.reason.advances_authority_world() {
        return;
    }
    let now = i64::from(request.tick.0) * i64::from(crate::MATCH_TICK_MS);
    let runtime = world.resource::<RoundScript>();
    if runtime.fault.is_some() || runtime.program.is_none() {
        return;
    }
    advance_motions(world, now);
    advance_bodies(world);
    advance_slides(world);
    apply_entity_links(world);
    super::players::apply_player_links(world);
    super::presence::settle_collision(world);
}

pub(crate) fn deliver_finished(world: &mut World) {
    let finished = std::mem::take(&mut world.resource_mut::<Mechanics>().finished);
    for (object, name) in finished {
        raise(world, Value::Object(object), name, Vec::new());
    }
}

fn advance_motions(world: &mut World, now: i64) {
    world.resource_scope::<Mechanics, _>(|world, mut mechanics| {
        let mut runtime = world.resource_mut::<RoundScript>();
        let Mechanics {
            motions, finished, ..
        } = &mut *mechanics;
        motions.retain(|object, _| runtime.entities.contains_key(object));
        for (object, list) in motions.iter_mut() {
            list.retain(|motion| {
                let (value, _) = motion.sample(now);
                runtime.set_object_field(*object, motion.field, Value::Vector(value));
                let done = now - motion.start_ms >= motion.duration_ms;
                if done {
                    finished.push((*object, motion.done));
                }
                !done
            });
        }
        motions.retain(|_, list| !list.is_empty());
    });
}

fn advance_bodies(world: &mut World) {
    let mut crushed = Vec::new();
    world.resource_scope::<Mechanics, _>(|world, mut mechanics| {
        let Mechanics {
            bodies,
            finished,
            settled,
            ..
        } = &mut *mechanics;
        settled.clear();
        bodies.retain(|object, body| {
            if !world
                .resource::<RoundScript>()
                .entities
                .contains_key(object)
            {
                return false;
            }
            if !body.active {
                return true;
            }
            match body.motion {
                BodyMotion::Piece { .. } => {
                    advance_piece(world, *object, body, finished, &mut crushed)
                }
                BodyMotion::Rigid { .. } => {
                    advance_rigid(world, *object, body, finished, &mut crushed);
                    if !body.active {
                        settled.push(*object);
                    }
                }
            }
            true
        });
    });
    for (victim, pusher, point) in crushed {
        super::players::crush_player(world, victim, pusher, point);
    }
}

type Crushed = Vec<(crate::ClientId, u64, [f32; 3])>;

fn advance_piece(
    world: &mut World,
    object: u64,
    body: &mut Body,
    finished: &mut Vec<(u64, &'static str)>,
    crushed: &mut Crushed,
) {
    let BodyMotion::Piece {
        velocity,
        offset,
        mins,
        maxs,
    } = &mut body.motion
    else {
        return;
    };
    let (origin, angles) = {
        let mut runtime = world.resource_mut::<RoundScript>();
        match (
            runtime.object_field(object, "origin"),
            runtime.object_field(object, "angles"),
        ) {
            (Value::Vector(o), Value::Vector(a)) => (o, a),
            (Value::Vector(o), _) => (o, [0.0; 3]),
            _ => return,
        }
    };
    velocity[2] -= GRAVITY * TICK_S;
    let start: [f32; 3] = std::array::from_fn(|i| origin[i] + offset[i]);
    let end: [f32; 3] = std::array::from_fn(|i| start[i] + velocity[i] * TICK_S);
    let trace = crate::frame::FrameWorld::from_world(world).trace_static_world(
        start,
        end,
        *mins,
        *maxs,
        crate::bullet_collision::MASK_PHYS_WORLD,
    );
    let fraction = if trace.startsolid != 0 {
        0.0
    } else {
        trace.fraction
    };
    let at: [f32; 3] = std::array::from_fn(|i| origin[i] + velocity[i] * TICK_S * fraction);
    if velocity[2] < 0.0 {
        crushed.extend(
            crush_victims(
                world,
                object,
                origin,
                at,
                *mins,
                *maxs,
                CrushVolume::AroundMove,
            )
            .into_iter()
            .map(|(victim, point)| (victim, object, point)),
        );
    }
    body.ticks += 1;
    let rested = fraction < 1.0 || body.ticks >= SETTLE_TICKS;
    let mut runtime = world.resource_mut::<RoundScript>();
    runtime.set_object_field(object, "origin", Value::Vector(at));
    if rested {
        runtime.set_object_field(object, "angles", Value::Vector([0.0, angles[1], 0.0]));
        finished.push((object, "physics_finished"));
        *velocity = [0.0; 3];
        body.active = false;
    }
}

fn advance_rigid(
    world: &mut World,
    object: u64,
    body: &mut Body,
    finished: &mut Vec<(u64, &'static str)>,
    crushed: &mut Crushed,
) {
    let BodyMotion::Rigid {
        body: rigid,
        client,
        ..
    } = &mut body.motion
    else {
        return;
    };
    let before = (rigid.origin(), rigid.center);
    body.ticks += 1;
    let asleep = {
        let frame = crate::frame::FrameWorld::from_world(world);
        let mut sweep = |start: Vec3, end: Vec3, half: Vec3| {
            Sweep::from_trace(&frame.trace_static_world(
                start.to_array(),
                end.to_array(),
                (-half).to_array(),
                half.to_array(),
                crate::bullet_collision::MASK_PHYS_WORLD,
            ))
        };
        let mut solid = |start: Vec3, end: Vec3, half: Vec3| {
            Sweep::from_trace(&frame.trace_static_world(
                start.to_array(),
                end.to_array(),
                (-half).to_array(),
                half.to_array(),
                SOLID_WORLD,
            ))
        };
        rigid.step(TICK_S, &mut sweep, &mut solid);
        let asleep = rigid.ready_to_sleep() || body.ticks >= SETTLE_TICKS;
        if asleep {
            rigid.settle(&mut sweep);
        }
        asleep
    };
    let origin = rigid.origin();
    if crushes(*client, before.1.z - rigid.center.z) {
        let half = rigid.world_half();
        let offset = rigid.center - origin;
        crushed.extend(
            crush_victims(
                world,
                object,
                before.0.to_array(),
                origin.to_array(),
                (offset - half).to_array(),
                (offset + half).to_array(),
                CrushVolume::Swept,
            )
            .into_iter()
            .map(|(victim, point)| (victim, object, point)),
        );
    }
    let angles = math_iw4::axis_to_angles(rigid.axis());
    let mut runtime = world.resource_mut::<RoundScript>();
    runtime.set_object_field(object, "origin", Value::Vector(origin.to_array()));
    runtime.set_object_field(object, "angles", Value::Vector(angles));
    if asleep {
        finished.push((object, "physics_finished"));
        body.active = false;
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum CrushVolume {
    Swept,
    AroundMove,
}

fn crush_victims(
    world: &mut World,
    object: u64,
    origin: [f32; 3],
    at: [f32; 3],
    mins: [f32; 3],
    maxs: [f32; 3],
    volume: CrushVolume,
) -> Vec<(crate::ClientId, [f32; 3])> {
    let presence = world.resource::<RoundScript>().entities[&object].presence;
    let frame = crate::frame::FrameWorld::from_world(world);
    let (mins, maxs) = presence
        .and_then(|id| {
            let row = frame
                .entity_collision_capabilities()
                .iter()
                .find(|row| row.owner.script_model() == Some(id))?;
            let brush = row.linked_brushes.first()?;
            let model = frame
                .clip_cmodels()
                .models
                .get(brush.cmodel_handle as usize)?;
            let reach = (0..2)
                .map(|i| model.mins[i].abs().max(model.maxs[i].abs()))
                .fold(0.0f32, f32::max);
            let offset: [f32; 3] = std::array::from_fn(|i| brush.origin[i] - origin[i]);
            Some((
                [
                    offset[0] - reach,
                    offset[1] - reach,
                    offset[2] + model.mins[2],
                ],
                [
                    offset[0] + reach,
                    offset[1] + reach,
                    offset[2] + model.maxs[2],
                ],
            ))
        })
        .unwrap_or((mins, maxs));
    let (pmins, pmaxs) = (
        crate::bullet_collision::PLAYER_MINS,
        crate::bullet_collision::PLAYER_MAXS,
    );
    let around_lo: [f32; 3] = std::array::from_fn(|i| origin[i].min(at[i]) + mins[i]);
    let around_hi: [f32; 3] = std::array::from_fn(|i| origin[i].max(at[i]) + maxs[i]);
    let mut victims = Vec::new();
    frame.visit_players(|id, ps| {
        if ps.pm_type >= playerstate_iw4::PM_TYPE_DEAD {
            return;
        }
        let inside = match volume {
            CrushVolume::AroundMove => (0..3).all(|i| {
                ps.origin[i] + pmins[i] < around_hi[i] && ps.origin[i] + pmaxs[i] > around_lo[i]
            }),
            CrushVolume::Swept => {
                let lo = Vec3::from_array(std::array::from_fn(|i| ps.origin[i] + pmins[i]));
                let hi = Vec3::from_array(std::array::from_fn(|i| ps.origin[i] + pmaxs[i]));
                super::rigid_body::swept_box_overlaps(
                    Vec3::from_array(origin),
                    Vec3::from_array(at),
                    Vec3::from_array(mins),
                    Vec3::from_array(maxs),
                    lo,
                    hi,
                )
            }
        };
        if inside {
            victims.push((id, ps.origin));
        }
    });
    victims
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Slide {
    pub center: [f32; 3],
    pub radius: f32,
    pub velocity: [f32; 3],
}

pub(crate) fn load_slide_field(world: &World, object: u64, name: &str) -> Option<Value> {
    (name == "slidevelocity"
        && world
            .resource::<RoundScript>()
            .entities
            .contains_key(&object))
    .then(|| {
        world
            .resource::<Mechanics>()
            .slides
            .get(&object)
            .map_or(Value::Vector([0.0; 3]), |slide| {
                Value::Vector(slide.velocity)
            })
    })
}

pub(crate) fn store_slide_field(
    world: &mut World,
    object: u64,
    name: &str,
    value: &Value,
) -> Result<bool, String> {
    if name != "slidevelocity"
        || !world
            .resource::<RoundScript>()
            .entities
            .contains_key(&object)
    {
        return Ok(false);
    }
    let Value::Vector(velocity) = value else {
        return Err("slideVelocity requires a vector".into());
    };
    if velocity.iter().any(|v| !v.is_finite()) {
        return Err("slideVelocity must be finite".into());
    }
    let mut mechanics = world.resource_mut::<Mechanics>();
    let slide = mechanics
        .slides
        .get_mut(&object)
        .ok_or("cannot set slideVelocity on an entity that is not sliding")?;
    slide.velocity = *velocity;
    Ok(true)
}

fn advance_slides(world: &mut World) {
    world.resource_scope::<Mechanics, _>(|world, mut mechanics| {
        mechanics.slides.retain(|object, slide| {
            let origin = {
                let mut runtime = world.resource_mut::<RoundScript>();
                if !runtime.entities.contains_key(object) {
                    return false;
                }
                let Value::Vector(origin) = runtime.object_field(*object, "origin") else {
                    return false;
                };
                origin
            };
            if origin.iter().any(|v| !v.is_finite()) {
                return false;
            }
            super::presence::settle_collision(world);
            let at = crate::step::script_slide(world, *object, origin, slide);
            world.resource_mut::<RoundScript>().set_object_field(
                *object,
                "origin",
                Value::Vector(at),
            );
            true
        });
    });
}

impl Slide {
    pub(crate) fn advance<C: movement_iw4::CollisionBackend>(
        &mut self,
        origin: [f32; 3],
        collision: &C,
        mask: u32,
    ) -> [f32; 3] {
        use movement_iw4::GroundTraceInput;
        let mins = self.center.map(|v| v - self.radius);
        let maxs = self.center.map(|v| v + self.radius);
        let trace = |start, end| {
            collision.trace(GroundTraceInput {
                start,
                end,
                mins,
                maxs,
                tracemask: mask,
            })
        };
        let start_velocity = self.velocity;
        let mut at = origin;
        if !self.sweep(&mut at, collision, mask, mins, maxs) {
            return at;
        }
        let down = trace(origin, [origin[0], origin[1], origin[2] - 18.0]);
        if start_velocity[2] > 0.0 && (down.fraction == 1.0 || down.normal[2] < 0.7) {
            return at;
        }
        let up = trace(origin, [origin[0], origin[1], origin[2] + 18.0]);
        if up.startsolid != 0 {
            return at;
        }
        at = up.endpos;
        let elevation = at[2] - origin[2];
        self.velocity = start_velocity;
        self.sweep(&mut at, collision, mask, mins, maxs);
        let down = trace(at, [at[0], at[1], at[2] - elevation]);
        if down.startsolid == 0 {
            at = down.endpos;
        }
        if down.fraction < 1.0 {
            self.velocity = clip_slide(self.velocity, down.normal);
        }
        at
    }

    fn sweep<C: movement_iw4::CollisionBackend>(
        &mut self,
        origin: &mut [f32; 3],
        collision: &C,
        mask: u32,
        mins: [f32; 3],
        maxs: [f32; 3],
    ) -> bool {
        let mut end_velocity = self.velocity;
        end_velocity[2] -= GRAVITY * TICK_S;
        self.velocity[2] = (self.velocity[2] + end_velocity[2]) * 0.5;
        let mut planes = [[0.0; 3]; 5];
        planes[0] = glam::Vec3::from_array(self.velocity)
            .normalize_or_zero()
            .to_array();
        let mut plane_count = 1;
        let mut time = TICK_S;
        let mut bumped = false;
        for _ in 0..4 {
            let end = std::array::from_fn(|i| origin[i] + self.velocity[i] * time);
            if end.iter().any(|v: &f32| !v.is_finite()) {
                self.velocity = [0.0; 3];
                return true;
            }
            let hit = collision.trace(movement_iw4::GroundTraceInput {
                start: *origin,
                end,
                mins,
                maxs,
                tracemask: mask,
            });
            if hit.allsolid != 0 {
                self.velocity[2] = 0.0;
                return true;
            }
            if hit.fraction > 0.0 {
                *origin = hit.endpos;
            }
            if hit.fraction == 1.0 {
                break;
            }
            bumped = true;
            time *= 1.0 - hit.fraction;
            if plane_count == planes.len() {
                self.velocity = [0.0; 3];
                return true;
            }
            if planes[..plane_count]
                .iter()
                .any(|p| slide_dot(hit.normal, *p) > 0.99)
            {
                self.velocity = std::array::from_fn(|i| self.velocity[i] + hit.normal[i]);
                continue;
            }
            planes[plane_count] = hit.normal;
            plane_count += 1;
            for i in 0..plane_count {
                if slide_dot(self.velocity, planes[i]) >= 0.1 {
                    continue;
                }
                let mut velocity = clip_slide(self.velocity, planes[i]);
                let mut final_velocity = clip_slide(end_velocity, planes[i]);
                for j in 0..plane_count {
                    if i == j || slide_dot(velocity, planes[j]) >= 0.1 {
                        continue;
                    }
                    velocity = clip_slide(velocity, planes[j]);
                    final_velocity = clip_slide(final_velocity, planes[j]);
                    if slide_dot(velocity, planes[i]) >= 0.0 {
                        continue;
                    }
                    let crease = glam::Vec3::from_array(planes[i])
                        .cross(glam::Vec3::from_array(planes[j]))
                        .normalize_or_zero();
                    velocity =
                        (crease * crease.dot(glam::Vec3::from_array(self.velocity))).to_array();
                    final_velocity =
                        (crease * crease.dot(glam::Vec3::from_array(end_velocity))).to_array();
                    if planes[..plane_count]
                        .iter()
                        .enumerate()
                        .any(|(k, p)| k != i && k != j && slide_dot(velocity, *p) < 0.1)
                    {
                        self.velocity = [0.0; 3];
                        return true;
                    }
                }
                self.velocity = velocity;
                end_velocity = final_velocity;
                break;
            }
        }
        self.velocity = end_velocity;
        bumped
    }
}

fn slide_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn clip_slide(velocity: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
    let dot = slide_dot(velocity, normal);
    let backoff = -(dot - dot.abs() * (1.001 - 1.0));
    std::array::from_fn(|i| velocity[i] + backoff * normal[i])
}

fn apply_entity_links(world: &mut World) {
    let mut runtime = world.resource_mut::<RoundScript>();
    let linked: Vec<(u64, Link)> = runtime
        .entities
        .iter()
        .filter_map(|(id, e)| Some((*id, e.linked_to.clone()?)))
        .collect();
    for (id, link) in linked {
        if !runtime.live(&link.parent) {
            runtime.entities.get_mut(&id).unwrap().linked_to = None;
            continue;
        }
        let field = |runtime: &mut RoundScript, name| match runtime.object_field(link.parent, name)
        {
            Value::Vector(v) => v,
            _ => [0.0; 3],
        };
        let (base, base_angles) = (field(&mut runtime, "origin"), field(&mut runtime, "angles"));
        let axis = math_iw4::angles_to_axis(base_angles);
        let offset = link.tag_offset.unwrap_or([0.0; 3]);
        let local = std::array::from_fn(|i| offset[i] + link.origin[i]);
        let (child_axis, origin) =
            math_iw4::matrix_multiply43(math_iw4::angles_to_axis(link.angles), local, axis, base);
        runtime.set_object_field(id, "origin", Value::Vector(origin));
        runtime.set_object_field(
            id,
            "angles",
            Value::Vector(math_iw4::axis_to_angles(child_axis)),
        );
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Motion {
    pub field: &'static str,
    pub path: MotionPath,
    pub start_ms: i64,
    pub duration_ms: i64,
    pub done: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum MotionPath {
    Linear {
        from: [f32; 3],
        to: [f32; 3],
        accel: f32,
        decel: f32,
    },
    Ballistic {
        from: [f32; 3],
        velocity: [f32; 3],
    },
}

const GRAVITY: f32 = 800.0;

impl Motion {
    pub(crate) fn sample(&self, now: i64) -> ([f32; 3], [f32; 3]) {
        let elapsed = (now - self.start_ms).clamp(0, self.duration_ms);
        let t = elapsed as f32 / 1000.0;
        match self.path {
            MotionPath::Linear {
                from,
                to,
                accel,
                decel,
            } => {
                let total = self.duration_ms as f32 / 1000.0;
                let peak = 2.0 / (2.0 * total - accel - decel);
                let cruise_end = total - decel;
                let (fraction, rate) = if t < accel {
                    (0.5 * peak * t * t / accel, peak * t / accel)
                } else if t <= cruise_end {
                    (0.5 * peak * accel + peak * (t - accel), peak)
                } else {
                    let u = t - cruise_end;
                    (
                        0.5 * peak * accel + peak * (cruise_end - accel) + peak * u
                            - 0.5 * peak * u * u / decel,
                        peak * (1.0 - u / decel),
                    )
                };
                let fraction = if elapsed >= self.duration_ms {
                    1.0
                } else {
                    fraction
                };
                let rate = if elapsed >= self.duration_ms {
                    0.0
                } else {
                    rate
                };
                (
                    std::array::from_fn(|i| from[i] + (to[i] - from[i]) * fraction),
                    std::array::from_fn(|i| (to[i] - from[i]) * rate),
                )
            }
            MotionPath::Ballistic { from, velocity } => {
                let mut p: [f32; 3] = std::array::from_fn(|i| from[i] + velocity[i] * t);
                p[2] -= 0.5 * GRAVITY * t * t;
                let mut v = velocity;
                v[2] -= GRAVITY * t;
                (p, v)
            }
        }
    }
}
