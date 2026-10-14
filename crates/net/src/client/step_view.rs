const STEP_VIEW_MIN: f32 = 1.0;
const STEP_VIEW_MAX: f32 = 16.0;
const STEP_VIEW_TIME_MS: i32 = 100;

pub const fn smooths_pm_type(pm_type: i32) -> bool {
    matches!(pm_type, 0 | 2 | 3)
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepView {
    start_ms: i32,
    change: f32,
    pending_ms: i32,
    pending_change: f32,
    newest_cmd_ms: Option<i32>,
    suppressed: bool,
}

impl StepView {
    pub fn note_command(&mut self, cmd_ms: i32, step: f32) {
        if self
            .newest_cmd_ms
            .is_none_or(|newest| cmd_ms.wrapping_sub(newest) > 0)
        {
            self.newest_cmd_ms = Some(cmd_ms);
        }
        if step != 0.0 && cmd_ms.wrapping_sub(self.pending_ms) > 0 {
            self.pending_change += step;
            self.pending_ms = cmd_ms;
        }
    }

    pub fn suppress(&mut self, latest_cmd_ms: Option<i32>) {
        self.change = 0.0;
        self.pending_change = 0.0;
        self.suppressed = true;
        if let Some(latest) = latest_cmd_ms {
            self.note_command(latest, 0.0);
        }
    }

    pub fn end_frame(&mut self, now_ms: i32, eligible: bool) {
        let change = core::mem::take(&mut self.pending_change);
        let time = self.pending_ms;
        if self.suppressed {
            self.suppressed = false;
            self.change = 0.0;
            if let Some(newest) = self.newest_cmd_ms
                && newest.wrapping_sub(self.start_ms) > 0
            {
                self.start_ms = newest;
            }
        } else if change != 0.0
            && time != self.start_ms
            && eligible
            && change.abs() >= STEP_VIEW_MIN
        {
            let carry = if now_ms.wrapping_sub(self.start_ms) < STEP_VIEW_TIME_MS {
                let progressed = time.wrapping_sub(self.start_ms) as f32 / STEP_VIEW_TIME_MS as f32;
                (1.0 - progressed).clamp(0.0, 1.0) * self.change
            } else {
                0.0
            };
            self.change = (change + carry).clamp(-STEP_VIEW_MAX, STEP_VIEW_MAX);
            self.start_ms = time;
        } else if now_ms.wrapping_sub(self.start_ms) > STEP_VIEW_TIME_MS {
            self.change = 0.0;
        }
        self.pending_ms = self.start_ms;
        self.newest_cmd_ms = None;
    }

    pub fn offset(&self, now_ms: i32) -> f32 {
        if self.change == 0.0 {
            return 0.0;
        }
        let elapsed = now_ms.wrapping_sub(self.start_ms);
        if elapsed < 0 {
            return 0.0;
        }
        let progressed = (elapsed as f32 / STEP_VIEW_TIME_MS as f32).min(1.0);
        self.change * (1.0 - progressed)
    }
}
