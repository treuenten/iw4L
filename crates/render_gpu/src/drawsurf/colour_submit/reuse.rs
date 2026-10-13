use std::sync::Arc;

use render_backend::DrawRunKeys;
use render_frame::{MaterialExecFrame, RetainedDrawItem, RetainedDrawKind};
use render_material::{MaterialGenerationId, RuntimeCodeSources, TechType};

use super::constants::ArenaSpanRef;
use bevy::platform::collections::HashMap;

use super::{ColourRowPlan, PreparedExactDraw};

pub(super) const DRAW_REUSE_ENV: &str = "IW4L_DRAW_REUSE";

pub(super) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| perf::switch(DRAW_REUSE_ENV))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ReuseFrameKey {
    pub(super) scene_epoch: super::TableEpoch,
    pub(super) binding_clears: u64,
    pub(super) material_generation: MaterialGenerationId,
    pub(super) catalog: usize,
    pub(super) prepared_table: usize,
    pub(super) pipeline_generation: MaterialGenerationId,
    pub(super) pipeline_ports: usize,
    pub(super) registry_world: frame::WorldStamp,
    pub(super) geometry_generation: MaterialGenerationId,
    pub(super) world_generation: frame::WorldStamp,
    pub(super) smodel_index_count: usize,
    pub(super) world_ready: bool,
    pub(super) smodel_ready: bool,
    pub(super) world_vertex_refused: bool,
    pub(super) smodel_vertex_refused: bool,
    pub(super) sun_shadow_view_ready: bool,
    pub(super) target: bevy::render::render_resource::TextureFormat,
    pub(super) samples: u32,
}

impl ReuseFrameKey {
    fn changes(&self, other: &Self) -> impl Iterator<Item = &'static str> {
        let fields: [(&'static str, bool); 18] = [
            ("scene_epoch", self.scene_epoch != other.scene_epoch),
            (
                "binding_clears",
                self.binding_clears != other.binding_clears,
            ),
            (
                "material_generation",
                self.material_generation != other.material_generation,
            ),
            ("catalog", self.catalog != other.catalog),
            (
                "prepared_table",
                self.prepared_table != other.prepared_table,
            ),
            (
                "pipeline_generation",
                self.pipeline_generation != other.pipeline_generation,
            ),
            (
                "pipeline_ports",
                self.pipeline_ports != other.pipeline_ports,
            ),
            (
                "registry_world",
                self.registry_world != other.registry_world,
            ),
            (
                "geometry_generation",
                self.geometry_generation != other.geometry_generation,
            ),
            (
                "world_generation",
                self.world_generation != other.world_generation,
            ),
            (
                "smodel_index_count",
                self.smodel_index_count != other.smodel_index_count,
            ),
            ("world_ready", self.world_ready != other.world_ready),
            ("smodel_ready", self.smodel_ready != other.smodel_ready),
            (
                "world_vertex_refused",
                self.world_vertex_refused != other.world_vertex_refused,
            ),
            (
                "smodel_vertex_refused",
                self.smodel_vertex_refused != other.smodel_vertex_refused,
            ),
            (
                "sun_shadow_view_ready",
                self.sun_shadow_view_ready != other.sun_shadow_view_ready,
            ),
            ("target", self.target != other.target),
            ("samples", self.samples != other.samples),
        ];
        fields
            .into_iter()
            .filter_map(|(name, changed)| changed.then_some(name))
    }
}

fn first_exec_change(now: &MaterialExecFrame, previous: &MaterialExecFrame) -> &'static str {
    if now.view_origin.to_array().map(f32::to_bits)
        != previous.view_origin.to_array().map(f32::to_bits)
    {
        "view_origin"
    } else if mat_bits(now.clip_from_world) != mat_bits(previous.clip_from_world)
        || mat_bits(now.view_from_world) != mat_bits(previous.view_from_world)
        || mat_bits(now.clip_from_view) != mat_bits(previous.clip_from_view)
        || mat_bits(now.viewmodel_clip_from_world) != mat_bits(previous.viewmodel_clip_from_world)
    {
        "camera"
    } else if now.reflection_probe_sh != previous.reflection_probe_sh
        || !(Arc::ptr_eq(&now.smodel_lighting_sh, &previous.smodel_lighting_sh)
            || now.smodel_lighting_sh == previous.smodel_lighting_sh)
    {
        "sh"
    } else if !now.code_sources.same_textures(&previous.code_sources) {
        "code textures"
    } else {
        "other"
    }
}

fn mat_bits(m: Option<bevy::math::Mat4>) -> Option<[u32; 16]> {
    m.map(|m| m.to_cols_array().map(f32::to_bits))
}

pub(super) fn same_exec_inputs(now: &MaterialExecFrame, previous: &MaterialExecFrame) -> bool {
    now.view_origin.to_array().map(f32::to_bits)
        == previous.view_origin.to_array().map(f32::to_bits)
        && mat_bits(now.clip_from_world) == mat_bits(previous.clip_from_world)
        && mat_bits(now.view_from_world) == mat_bits(previous.view_from_world)
        && mat_bits(now.clip_from_view) == mat_bits(previous.clip_from_view)
        && mat_bits(now.viewmodel_clip_from_world) == mat_bits(previous.viewmodel_clip_from_world)
        && now.viewmodel_near.map(f32::to_bits) == previous.viewmodel_near.map(f32::to_bits)
        && now.inv_image_height.map(f32::to_bits) == previous.inv_image_height.map(f32::to_bits)
        && now.outdoor.map(|o| (o.image, o.lookup)) == previous.outdoor.map(|o| (o.image, o.lookup))
        && now.reflection_probe_sh == previous.reflection_probe_sh
        && (Arc::ptr_eq(&now.smodel_lighting_sh, &previous.smodel_lighting_sh)
            || now.smodel_lighting_sh == previous.smodel_lighting_sh)
        && now.code_sources.same_textures(&previous.code_sources)
}

#[derive(Default)]
pub(super) struct ChangedLights {
    bits: [u64; 4],
}

impl ChangedLights {
    fn fill(&mut self, now: &MaterialExecFrame, previous: &MaterialExecFrame) {
        fn mark<T: PartialEq>(bits: &mut [u64; 4], now: &[T], previous: &[T]) {
            for index in 0..now.len().max(previous.len()).min(256) {
                if now.get(index) != previous.get(index) {
                    bits[index / 64] |= 1 << (index % 64);
                }
            }
            if now.len().max(previous.len()) > 256 {
                *bits = [u64::MAX; 4];
            }
        }
        self.bits = [0; 4];
        mark(
            &mut self.bits,
            &now.primary_lights,
            &previous.primary_lights,
        );
        mark(&mut self.bits, &now.attenuation, &previous.attenuation);
        mark(
            &mut self.bits,
            &now.local_light_bindings,
            &previous.local_light_bindings,
        );
        mark(
            &mut self.bits,
            &now.spot_receivers,
            &previous.spot_receivers,
        );
    }

    pub(super) fn contains(&self, light: u8) -> bool {
        self.bits[usize::from(light) / 64] & (1 << (light % 64)) != 0
    }

    fn count(&self) -> u32 {
        self.bits.iter().map(|word| word.count_ones()).sum()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SmcLayoutKey {
    pub(super) revision: u64,
    pub(super) pretess_indices: usize,
    pub(super) baked: usize,
}

#[derive(Default)]
pub(super) struct ChangedConstants {
    bits: Vec<u64>,
    list: Vec<u16>,
}

impl ChangedConstants {
    fn fill(&mut self, now: &RuntimeCodeSources, previous: &RuntimeCodeSources) {
        now.changed_constants(previous, &mut self.list);
        self.bits.fill(0);
        for &index in &self.list {
            let word = usize::from(index) / 64;
            if self.bits.len() <= word {
                self.bits.resize(word + 1, 0);
            }
            self.bits[word] |= 1 << (index % 64);
        }
    }

    pub(super) fn any(&self, reads: &[u16]) -> bool {
        reads.iter().any(|&index| {
            self.bits
                .get(usize::from(index) / 64)
                .is_some_and(|word| word & (1 << (index % 64)) != 0)
        })
    }

    pub(super) fn indices(&self) -> &[u16] {
        &self.list
    }
}

#[derive(Default)]
pub(super) struct ReuseFrame {
    pub(super) on: bool,
    pub(super) valid: bool,
    pub(super) store: bool,
    pub(super) changed: ChangedConstants,
    pub(super) lights: ChangedLights,
    pub(super) smc_changed: bool,
}

#[derive(Default)]
pub(super) struct ReuseState {
    last_key: Option<ReuseFrameKey>,
    last_smc: Option<SmcLayoutKey>,
    last_exec: Option<MaterialExecFrame>,
    pub(super) frame: ReuseFrame,
    pub(super) stats: ReuseStats,
}

impl ReuseState {
    pub(super) fn open(
        &mut self,
        key: Option<ReuseFrameKey>,
        smc: SmcLayoutKey,
        exec: &MaterialExecFrame,
    ) {
        self.frame.on = enabled();
        if !self.frame.on {
            self.frame.valid = false;
            self.frame.store = false;
            return;
        }
        if self.stats.frames == 0 {
            diag::info!(World, "draw reuse: on ({DRAW_REUSE_ENV}=1)");
        }
        let same_key = key.is_some() && key == self.last_key;
        let same_exec = self
            .last_exec
            .as_ref()
            .is_some_and(|last| same_exec_inputs(exec, last));
        match self.last_exec.as_ref() {
            Some(last) => {
                self.frame
                    .changed
                    .fill(&exec.code_sources, &last.code_sources);
                self.frame.lights.fill(exec, last);
            }
            None => {
                self.frame
                    .changed
                    .fill(&exec.code_sources, &exec.code_sources);
                self.frame.lights.fill(exec, exec);
            }
        }
        self.stats.lights_changed += u64::from(self.frame.lights.count());
        self.frame.valid = same_key && same_exec;
        self.frame.store = same_exec;
        self.frame.smc_changed = self.last_smc != Some(smc);
        self.last_smc = Some(smc);
        self.stats.frames += 1;
        if self.frame.smc_changed {
            self.stats.smc_changes += 1;
        }
        if !self.frame.valid {
            self.stats.invalid_frames += 1;
            if !same_key {
                self.stats.key_changes += 1;
                match (key.as_ref(), self.last_key.as_ref()) {
                    (Some(now), Some(last)) => {
                        for name in now.changes(last) {
                            self.stats.note_change(name);
                        }
                    }
                    _ => self.stats.note_change("first"),
                }
            } else {
                self.stats.exec_changes += 1;
                let name = self
                    .last_exec
                    .as_ref()
                    .map_or("first", |last| first_exec_change(exec, last));
                self.stats.note_change(name);
            }
        }
        self.last_key = key;
        match self.last_exec.as_mut() {
            Some(last) => last.clone_from(exec),
            None => self.last_exec = Some(exec.clone()),
        }
    }
}

#[derive(Default)]
pub(super) struct ReuseStats {
    pub(super) frames: u64,
    pub(super) invalid_frames: u64,
    pub(super) key_changes: u64,
    pub(super) exec_changes: u64,
    pub(super) rows: u64,
    pub(super) eligible: u64,
    pub(super) reused: u64,
    pub(super) miss_row: u64,
    pub(super) miss_reads: u64,
    pub(super) miss_span: u64,
    pub(super) miss_light: u64,
    lights_changed: u64,
    smc_changes: u64,
    changes: Vec<(&'static str, u64)>,
    logged_at: u64,
}

impl ReuseStats {
    fn note_change(&mut self, name: &'static str) {
        match self.changes.iter_mut().find(|(known, _)| *known == name) {
            Some((_, n)) => *n += 1,
            None => self.changes.push((name, 1)),
        }
    }

    pub(super) fn absorb(&mut self, lane: &LaneReuseTally) {
        self.rows += lane.rows;
        self.eligible += lane.stored;
        self.reused += lane.reused;
        self.miss_row += lane.miss_row;
        self.miss_reads += lane.miss_reads;
        self.miss_span += lane.miss_span;
        self.miss_light += lane.miss_light;
    }

    pub(super) fn maybe_log(&mut self, changed: &[u16]) {
        if self.frames.saturating_sub(self.logged_at) < 600 {
            return;
        }
        self.logged_at = self.frames;
        diag::info!(
            World,
            "draw reuse: frames={} invalid={} (key={} inputs={} why={:?}) rows={} kept={} ({:.1}%) stored={} miss_row={} miss_reads={} miss_light={} miss_span={} lights_changed={} smc_changes={} changed_now={:?}",
            self.frames,
            self.invalid_frames,
            self.key_changes,
            self.exec_changes,
            self.changes,
            self.rows,
            self.reused,
            if self.rows == 0 {
                0.0
            } else {
                self.reused as f64 * 100.0 / self.rows as f64
            },
            self.eligible,
            self.miss_row,
            self.miss_reads,
            self.miss_light,
            self.miss_span,
            self.lights_changed,
            self.smc_changes,
            changed,
        );
    }
}

#[derive(Default, Clone, Copy)]
pub(super) struct LaneReuseTally {
    pub(super) rows: u64,
    pub(super) stored: u64,
    pub(super) reused: u64,
    pub(super) miss_row: u64,
    pub(super) miss_reads: u64,
    pub(super) miss_span: u64,
    pub(super) miss_light: u64,
}

pub(super) fn eligible_kind(kind: &RetainedDrawKind) -> bool {
    match *kind {
        RetainedDrawKind::World { .. } => true,
        RetainedDrawKind::Smodel { stream, .. } => matches!(
            stream,
            Some(
                lighting_iw4::SmodelSurfPath::Rigid
                    | lighting_iw4::SmodelSurfPath::Cached
                    | lighting_iw4::SmodelSurfPath::Pretess
            )
        ),
        _ => false,
    }
}

fn reads_smc_layout(kind: &RetainedDrawKind) -> bool {
    matches!(
        kind,
        RetainedDrawKind::Smodel {
            stream: Some(
                lighting_iw4::SmodelSurfPath::Cached | lighting_iw4::SmodelSurfPath::Pretess
            ),
            ..
        }
    )
}

fn mat4_bits(m: &bevy::math::Mat4) -> [u32; 16] {
    m.to_cols_array().map(f32::to_bits)
}

pub(super) fn same_item(a: &RetainedDrawItem, b: &RetainedDrawItem) -> bool {
    if a.material_id != b.material_id
        || a.key != b.key
        || a.material_rank != b.material_rank
        || a.surface_samplers != b.surface_samplers
        || a.camera_region != b.camera_region
    {
        return false;
    }
    match (&a.kind, &b.kind) {
        (
            RetainedDrawKind::World {
                surf: s0,
                run: r0,
                run_off: o0,
                bsp_kind: k0,
                bsp_run_first: f0,
                setup_key_changed: c0,
                world_from_local: m0,
            },
            RetainedDrawKind::World {
                surf: s1,
                run: r1,
                run_off: o1,
                bsp_kind: k1,
                bsp_run_first: f1,
                setup_key_changed: c1,
                world_from_local: m1,
            },
        ) => {
            s0 == s1
                && r0 == r1
                && o0 == o1
                && k0 == k1
                && f0 == f1
                && c0 == c1
                && mat4_bits(m0) == mat4_bits(m1)
        }
        (
            RetainedDrawKind::Smodel {
                surface: s0,
                material: t0,
                world_from_local: m0,
                lighting_handle: h0,
                packed_lighting: p0,
                placement: l0,
                stream: st0,
                cache_index: c0,
                pretess: r0,
            },
            RetainedDrawKind::Smodel {
                surface: s1,
                material: t1,
                world_from_local: m1,
                lighting_handle: h1,
                packed_lighting: p1,
                placement: l1,
                stream: st1,
                cache_index: c1,
                pretess: r1,
            },
        ) => {
            s0 == s1
                && t0 == t1
                && mat4_bits(m0) == mat4_bits(m1)
                && h0 == h1
                && p0 == p1
                && l0 == l1
                && st0 == st1
                && c0 == c1
                && r0 == r1
        }
        _ => false,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct RowId {
    key: u64,
    first: u32,
    second: u32,
    world_surf: Option<u16>,
    technique: u8,
    tech: u8,
}

impl RowId {
    pub(super) fn of(
        planned: &ColourRowPlan,
        live: &RetainedDrawItem,
        tech: TechType,
    ) -> Option<Self> {
        let (first, second) = match live.kind {
            RetainedDrawKind::World {
                surf, run, run_off, ..
            } => (u32::from(surf) | u32::from(run) << 16, run_off),
            RetainedDrawKind::Smodel {
                placement, surface, ..
            } => (placement, surface),
            _ => return None,
        };
        eligible_kind(&live.kind).then_some(Self {
            key: live.key,
            first,
            second,
            world_surf: planned.world_surface_override,
            technique: planned.technique.0,
            tech: tech.0,
        })
    }
}

#[derive(Clone, Copy)]
struct CachedRow {
    world_surface_override: Option<u16>,
    technique: TechType,
    index_span: Option<(u32, u32)>,
    layout_epoch: u64,
    item: RetainedDrawItem,
    tech: TechType,
    run_keys: DrawRunKeys,
    records: (u32, u32),
    spans: (u32, u32),
    reads: (u32, u32),
}

impl CachedRow {
    fn same_plan(&self, planned: &ColourRowPlan) -> bool {
        self.world_surface_override == planned.world_surface_override
            && self.technique == planned.technique
            && self.index_span == planned.index_span
            && (!matches!(self.item.kind, RetainedDrawKind::World { .. })
                || self.layout_epoch == planned.layout_epoch)
    }
}

#[derive(Default)]
struct RowPools {
    rows: HashMap<RowId, CachedRow>,
    records: Vec<PreparedExactDraw>,
    spans: Vec<ArenaSpanRef>,
    reads: Vec<u16>,
}

impl RowPools {
    fn clear(&mut self) {
        self.rows.clear();
        self.records.clear();
        self.spans.clear();
        self.reads.clear();
    }
}

fn range(r: (u32, u32)) -> std::ops::Range<usize> {
    r.0 as usize..(r.0 + r.1) as usize
}

fn span_of(start: usize, end: usize) -> (u32, u32) {
    (
        u32::try_from(start).expect("reuse pool fits u32"),
        u32::try_from(end - start).expect("reuse pool fits u32"),
    )
}

pub(super) struct KeptRow<'a> {
    pub(super) records: &'a [PreparedExactDraw],
    pub(super) spans: &'a [ArenaSpanRef],
    pub(super) run_keys: DrawRunKeys,
}

pub(super) enum Lookup<'a> {
    Kept(KeptRow<'a>),
    Changed,
    Reads,
    Light,
    None,
}

#[derive(Default)]
pub(super) struct LaneReuse {
    last: RowPools,
    next: RowPools,
    pub(super) tally: LaneReuseTally,
}

impl LaneReuse {
    pub(super) fn begin(&mut self, rows: usize) {
        self.next.clear();
        self.tally = LaneReuseTally {
            rows: rows as u64,
            ..LaneReuseTally::default()
        };
    }

    pub(super) fn finish(&mut self) {
        std::mem::swap(&mut self.last, &mut self.next);
        self.next.clear();
    }

    pub(super) fn lookup(
        &self,
        id: RowId,
        planned: &ColourRowPlan,
        live: &RetainedDrawItem,
        tech: TechType,
        frame: &ReuseFrame,
    ) -> Lookup<'_> {
        let Some(row) = self.last.rows.get(&id) else {
            return Lookup::None;
        };
        if !row.same_plan(planned) || row.tech != tech || !same_item(&row.item, live) {
            return Lookup::Changed;
        }
        if frame.smc_changed && reads_smc_layout(&live.kind) {
            return Lookup::Changed;
        }
        if frame
            .lights
            .contains(dpvs_iw4::GfxDrawSurf { packed: live.key }.scene_light_index())
        {
            return Lookup::Light;
        }
        let changed = &frame.changed;
        if changed.any(&self.last.reads[range(row.reads)]) {
            return Lookup::Reads;
        }
        Lookup::Kept(KeptRow {
            records: &self.last.records[range(row.records)],
            spans: &self.last.spans[range(row.spans)],
            run_keys: row.run_keys,
        })
    }

    pub(super) fn keep(&mut self, id: RowId) {
        let Some(row) = self.last.rows.get(&id).copied() else {
            return;
        };
        let records = self.next.records.len();
        self.next
            .records
            .extend_from_slice(&self.last.records[range(row.records)]);
        let spans = self.next.spans.len();
        self.next
            .spans
            .extend_from_slice(&self.last.spans[range(row.spans)]);
        let reads = self.next.reads.len();
        self.next
            .reads
            .extend_from_slice(&self.last.reads[range(row.reads)]);
        self.next.rows.insert(
            id,
            CachedRow {
                records: span_of(records, self.next.records.len()),
                spans: span_of(spans, self.next.spans.len()),
                reads: span_of(reads, self.next.reads.len()),
                ..row
            },
        );
        self.tally.reused += 1;
        self.tally.stored += 1;
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn store(
        &mut self,
        id: RowId,
        run_keys: DrawRunKeys,
        planned: &ColourRowPlan,
        live: &RetainedDrawItem,
        tech: TechType,
        records: &[PreparedExactDraw],
        spans: &[ArenaSpanRef],
        reads: impl Iterator<Item = u16>,
    ) {
        if records.len() != spans.len() {
            return;
        }
        let r0 = self.next.records.len();
        self.next.records.extend_from_slice(records);
        let s0 = self.next.spans.len();
        self.next.spans.extend_from_slice(spans);
        let d0 = self.next.reads.len();
        self.next.reads.extend(reads);
        self.next.rows.insert(
            id,
            CachedRow {
                world_surface_override: planned.world_surface_override,
                technique: planned.technique,
                index_span: planned.index_span,
                layout_epoch: planned.layout_epoch,
                item: *live,
                tech,
                run_keys,
                records: span_of(r0, self.next.records.len()),
                spans: span_of(s0, self.next.spans.len()),
                reads: span_of(d0, self.next.reads.len()),
            },
        );
        self.tally.stored += 1;
    }
}
