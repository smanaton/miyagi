use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::architecture::Projection;
use crate::backend::MiyagiBackend;
use crate::error::{Error, Result};
use crate::fitness::{FitnessMode, compute_fitness};
use crate::patch::{Patch, PatchFlip};
use crate::probe::{
    CompiledProbe, ProbeMeasurement, measure_probes, measure_probes_allowing_non_finite,
};

const CHECKPOINT_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub struct SearchConfig {
    pub search_layers: Vec<usize>,
    pub search_projections: Vec<Projection>,
    pub max_iters: usize,
    pub control_penalty: f32,
    pub fitness_mode: FitnessMode,
    pub seed: u64,
    pub screen_probe_count: usize,
    pub patch_name: String,
    pub patch_description: String,
    pub base_model: String,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            // Empty means "resolve from the live backend in run_search via
            // default_search_layers(layer_count)". Do not bake a fixed layer
            // count here — that reintroduces 8B-era under-search on larger models.
            search_layers: Vec::new(),
            search_projections: vec![Projection::Gate, Projection::Up],
            max_iters: 200,
            control_penalty: 2.0,
            fitness_mode: FitnessMode::Mean,
            seed: 42,
            screen_probe_count: 2,
            patch_name: "untitled".to_owned(),
            patch_description: String::new(),
            base_model: "unknown".to_owned(),
        }
    }
}

/// Upper bound on how many layers the computed default samples.
const MAX_DEFAULT_LAYERS: usize = 12;

/// Evenly spaced layer indices spanning the model's full depth, used when the
/// caller does not pass `--layers` (or leaves `SearchConfig.search_layers` empty).
///
/// A fixed absolute list (the old `[1, 2, 3, 4, 34]`) silently under-searches
/// any model deeper than the 8B it was tuned for — e.g. on a 64-layer model it
/// never samples past layer 34, capping results with no warning.
///
/// - **Large models** (`layer_count > MAX_DEFAULT_LAYERS`): up to
///   `MAX_DEFAULT_LAYERS` points across `1..=layer_count-1` (skips layer 0).
/// - **Small models** (`layer_count ≤ MAX_DEFAULT_LAYERS`): every layer index
///   in `0..layer_count` (includes layer 0).
pub fn default_search_layers(layer_count: usize) -> Vec<usize> {
    if layer_count == 0 {
        return Vec::new();
    }
    if layer_count <= MAX_DEFAULT_LAYERS {
        return (0..layer_count).collect();
    }
    let last = layer_count - 1;
    let span = last - 1; // spread across [1, last]
    let steps = MAX_DEFAULT_LAYERS - 1;
    let mut layers: Vec<usize> = (0..MAX_DEFAULT_LAYERS)
        .map(|i| 1 + (span * i + steps / 2) / steps) // integer round
        .collect();
    layers.dedup();
    layers
}

/// Fill empty `search_layers` from the live backend architecture.
pub fn resolve_search_layers(layer_count: usize, search_layers: &[usize]) -> Vec<usize> {
    if search_layers.is_empty() {
        default_search_layers(layer_count)
    } else {
        search_layers.to_vec()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SearchCheckpoint {
    pub version: u32,
    pub architecture_signature: String,
    pub model_label: String,
    pub config: SearchConfig,
    pub completed_iterations: usize,
    pub accepted: Vec<PatchFlip>,
    pub tried: BTreeSet<PatchFlip>,
    pub current_fitness: f32,
    pub rng_state: u64,
    pub target_baseline: Vec<ProbeMeasurement>,
    pub control_baseline: Vec<ProbeMeasurement>,
}

impl SearchCheckpoint {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
    }

    pub fn save_atomic(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let temporary = path.with_extension(format!(
            "{}.{}.tmp",
            path.extension()
                .and_then(|extension| extension.to_str())
                .unwrap_or("checkpoint"),
            std::process::id()
        ));
        let bytes = serde_json::to_vec_pretty(self)?;
        fs::write(&temporary, bytes)?;
        if let Err(error) = fs::rename(&temporary, path) {
            let _ = fs::remove_file(&temporary);
            return Err(error.into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelPatchState {
    Baseline,
    AcceptedPatchApplied,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SearchResult {
    pub patch: Patch,
    pub target_baseline: Vec<ProbeMeasurement>,
    pub control_baseline: Vec<ProbeMeasurement>,
    pub final_target: Vec<ProbeMeasurement>,
    pub final_control: Vec<ProbeMeasurement>,
    pub final_fitness: f32,
    pub completed_iterations: usize,
    pub tried_candidates: usize,
    pub screened_out: usize,
    pub model_state: ModelPatchState,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum SearchEvent {
    Baseline {
        target: Vec<ProbeMeasurement>,
        control: Vec<ProbeMeasurement>,
    },
    Candidate {
        iteration: usize,
        flip: PatchFlip,
    },
    ScreenedOut {
        iteration: usize,
        flip: PatchFlip,
    },
    Rejected {
        iteration: usize,
        flip: PatchFlip,
        fitness: f32,
        best_fitness: f32,
    },
    Accepted {
        iteration: usize,
        flip: PatchFlip,
        fitness: f32,
        accepted: usize,
    },
    Checkpoint {
        iteration: usize,
    },
    Completed {
        iterations: usize,
        accepted: usize,
        fitness: f32,
    },
}

#[derive(Clone, Debug)]
struct Candidate {
    flip: PatchFlip,
    weight: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn run_search<B, F>(
    backend: &mut B,
    target_probes: &[CompiledProbe],
    control_probes: &[CompiledProbe],
    config: SearchConfig,
    resume: Option<SearchCheckpoint>,
    checkpoint_path: Option<&Path>,
    cancellation: Option<&AtomicBool>,
    mut on_event: F,
) -> Result<SearchResult>
where
    B: MiyagiBackend,
    F: FnMut(&SearchEvent),
{
    let mut config = config;
    config.search_layers = resolve_search_layers(
        backend.architecture().layer_count(),
        &config.search_layers,
    );
    validate_config(backend, target_probes, control_probes, &config)?;
    let candidates = build_candidates(backend, &config)?;
    if candidates.is_empty() {
        return Err(Error::InvalidSearch("candidate pool is empty".to_owned()));
    }

    // Baselines must be finite (strict measure_probes). Candidate flips may
    // produce NaN — those use measure_probes_allowing_non_finite below.
    let fresh_target_baseline = measure_probes(backend, target_probes)?;
    let fresh_control_baseline = measure_probes(backend, control_probes)?;
    let architecture_signature = backend.architecture().signature().to_owned();
    let model_label = backend.model_label().to_owned();

    let (
        target_baseline,
        control_baseline,
        mut accepted,
        mut tried,
        mut current_fitness,
        mut completed_iterations,
        mut rng,
    ) = if let Some(checkpoint) = resume {
        validate_checkpoint(
            &checkpoint,
            &config,
            &architecture_signature,
            &fresh_target_baseline,
            &fresh_control_baseline,
        )?;
        apply_coordinates(backend, &checkpoint.accepted, "checkpoint restoration")?;
        (
            checkpoint.target_baseline,
            checkpoint.control_baseline,
            checkpoint.accepted,
            checkpoint.tried,
            checkpoint.current_fitness,
            checkpoint.completed_iterations,
            SplitMix64::from_state(checkpoint.rng_state),
        )
    } else {
        (
            fresh_target_baseline,
            fresh_control_baseline,
            Vec::new(),
            BTreeSet::new(),
            0.0,
            0,
            SplitMix64::new(config.seed),
        )
    };

    on_event(&SearchEvent::Baseline {
        target: target_baseline.clone(),
        control: control_baseline.clone(),
    });
    let screen_indices = screen_indices(&target_baseline, config.screen_probe_count);
    let baseline_by_name = target_baseline
        .iter()
        .map(|measurement| (measurement.name.clone(), measurement.gap))
        .collect::<BTreeMap<_, _>>();
    let mut screened_out = 0;

    while completed_iterations < config.max_iters {
        if cancellation.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            let checkpoint = checkpoint(
                &architecture_signature,
                &model_label,
                &config,
                completed_iterations,
                &accepted,
                &tried,
                current_fitness,
                rng.state(),
                &target_baseline,
                &control_baseline,
            );
            if let Some(path) = checkpoint_path {
                checkpoint.save_atomic(path)?;
            }
            return Err(Error::SearchCancelled);
        }

        let Some(candidate) = sample_candidate(&candidates, &tried, &mut rng) else {
            break;
        };
        tried.insert(candidate.flip.clone());
        completed_iterations += 1;
        on_event(&SearchEvent::Candidate {
            iteration: completed_iterations,
            flip: candidate.flip.clone(),
        });

        backend.flip_row(
            candidate.flip.layer,
            candidate.flip.proj,
            candidate.flip.row,
        )?;
        let evaluation = evaluate_candidate(
            backend,
            target_probes,
            control_probes,
            &screen_indices,
            &baseline_by_name,
            &target_baseline,
            &control_baseline,
            &config,
        );

        match evaluation {
            Ok(CandidateEvaluation::ScreenedOut) => {
                revert_candidate(backend, &candidate.flip, "screened candidate")?;
                screened_out += 1;
                on_event(&SearchEvent::ScreenedOut {
                    iteration: completed_iterations,
                    flip: candidate.flip.clone(),
                });
            }
            Ok(CandidateEvaluation::Measured { fitness }) if fitness > current_fitness => {
                current_fitness = fitness;
                accepted.push(candidate.flip.clone());
                on_event(&SearchEvent::Accepted {
                    iteration: completed_iterations,
                    flip: candidate.flip.clone(),
                    fitness,
                    accepted: accepted.len(),
                });
            }
            Ok(CandidateEvaluation::Measured { fitness }) => {
                revert_candidate(backend, &candidate.flip, "rejected candidate")?;
                on_event(&SearchEvent::Rejected {
                    iteration: completed_iterations,
                    flip: candidate.flip.clone(),
                    fitness,
                    best_fitness: current_fitness,
                });
            }
            Err(error) => {
                revert_candidate(backend, &candidate.flip, "failed candidate evaluation")?;
                return Err(error);
            }
        }

        if let Some(path) = checkpoint_path {
            checkpoint(
                &architecture_signature,
                &model_label,
                &config,
                completed_iterations,
                &accepted,
                &tried,
                current_fitness,
                rng.state(),
                &target_baseline,
                &control_baseline,
            )
            .save_atomic(path)?;
            on_event(&SearchEvent::Checkpoint {
                iteration: completed_iterations,
            });
        }
    }

    let final_target = measure_probes(backend, target_probes)?;
    let final_control = measure_probes(backend, control_probes)?;
    let final_fitness = compute_fitness(
        config.fitness_mode,
        &final_target,
        &final_control,
        &target_baseline,
        &control_baseline,
        config.control_penalty,
    )?;
    let mut patch = Patch::new(
        config.patch_name.clone(),
        config.patch_description.clone(),
        config.base_model.clone(),
        accepted,
    );
    patch.metadata.insert(
        "architecture_signature".to_owned(),
        Value::String(architecture_signature),
    );
    patch.metadata.insert(
        "search_algorithm".to_owned(),
        Value::String(
            format!("greedy_hill_climbing_screened_{:?}", config.fitness_mode).to_lowercase(),
        ),
    );
    patch
        .metadata
        .insert("seed".to_owned(), Value::Number(config.seed.into()));
    patch.metadata.insert(
        "completed_iterations".to_owned(),
        Value::Number((completed_iterations as u64).into()),
    );
    patch.metadata.insert(
        "control_penalty".to_owned(),
        serde_json::Number::from_f64(config.control_penalty as f64)
            .map(Value::Number)
            .unwrap_or(Value::Null),
    );
    let validated = patch.validate(backend.architecture(), Default::default())?;
    let patch = validated.into_patch();
    on_event(&SearchEvent::Completed {
        iterations: completed_iterations,
        accepted: patch.flips.len(),
        fitness: final_fitness,
    });
    Ok(SearchResult {
        patch,
        target_baseline,
        control_baseline,
        final_target,
        final_control,
        final_fitness,
        completed_iterations,
        tried_candidates: tried.len(),
        screened_out,
        model_state: ModelPatchState::AcceptedPatchApplied,
    })
}

enum CandidateEvaluation {
    ScreenedOut,
    Measured { fitness: f32 },
}

#[allow(clippy::too_many_arguments)]
fn evaluate_candidate<B: MiyagiBackend>(
    backend: &mut B,
    target_probes: &[CompiledProbe],
    control_probes: &[CompiledProbe],
    screen_indices: &[usize],
    baseline_by_name: &BTreeMap<String, f32>,
    target_baseline: &[ProbeMeasurement],
    control_baseline: &[ProbeMeasurement],
    config: &SearchConfig,
) -> Result<CandidateEvaluation> {
    let screen_probes = screen_indices
        .iter()
        .map(|index| target_probes[*index].clone())
        .collect::<Vec<_>>();
    // Candidate path: allow non-finite gaps (reject this flip, keep searching).
    let screen_measurements = measure_probes_allowing_non_finite(backend, &screen_probes)?;
    if !screen_measurements.iter().any(|measurement| {
        baseline_by_name.get(&measurement.name).is_some_and(|baseline| {
            measurement.gap.is_finite() && measurement.gap > *baseline
        })
    }) {
        return Ok(CandidateEvaluation::ScreenedOut);
    }

    let mut measured_by_index = screen_indices
        .iter()
        .copied()
        .zip(screen_measurements)
        .collect::<BTreeMap<_, _>>();
    for (index, probe) in target_probes.iter().enumerate() {
        if let std::collections::btree_map::Entry::Vacant(entry) = measured_by_index.entry(index) {
            let measurement = measure_probes_allowing_non_finite(backend, std::slice::from_ref(probe))?
                .into_iter()
                .next()
                .expect("one probe returns one measurement");
            entry.insert(measurement);
        }
    }
    let target = (0..target_probes.len())
        .map(|index| {
            measured_by_index
                .remove(&index)
                .expect("every target probe was measured")
        })
        .collect::<Vec<_>>();
    let control = measure_probes_allowing_non_finite(backend, control_probes)?;
    // Non-finite candidate gap → reject this flip only (NaN fails fitness >).
    if target
        .iter()
        .chain(control.iter())
        .any(|measurement| !measurement.gap.is_finite())
    {
        return Ok(CandidateEvaluation::Measured { fitness: f32::NAN });
    }
    let fitness = compute_fitness(
        config.fitness_mode,
        &target,
        &control,
        target_baseline,
        control_baseline,
        config.control_penalty,
    )?;
    Ok(CandidateEvaluation::Measured { fitness })
}

fn validate_config<B: MiyagiBackend>(
    backend: &B,
    target_probes: &[CompiledProbe],
    control_probes: &[CompiledProbe],
    config: &SearchConfig,
) -> Result<()> {
    if target_probes.is_empty() || control_probes.is_empty() {
        return Err(Error::InvalidSearch(
            "target and control probes must both be non-empty".to_owned(),
        ));
    }
    if config.search_layers.is_empty() || config.search_projections.is_empty() {
        return Err(Error::InvalidSearch(
            "search layers and projections must both be non-empty".to_owned(),
        ));
    }
    if config.max_iters == 0 {
        return Err(Error::InvalidSearch(
            "max_iters must be greater than zero".to_owned(),
        ));
    }
    if config.screen_probe_count == 0 {
        return Err(Error::InvalidSearch(
            "screen_probe_count must be greater than zero".to_owned(),
        ));
    }
    if !config.control_penalty.is_finite() || config.control_penalty < 0.0 {
        return Err(Error::InvalidSearch(
            "control penalty must be finite and non-negative".to_owned(),
        ));
    }
    for layer in &config.search_layers {
        for projection in &config.search_projections {
            backend.architecture().tensor(*layer, *projection)?;
        }
    }
    Ok(())
}

fn build_candidates<B: MiyagiBackend>(
    backend: &mut B,
    config: &SearchConfig,
) -> Result<Vec<Candidate>> {
    let coordinates = config
        .search_layers
        .iter()
        .flat_map(|layer| {
            config
                .search_projections
                .iter()
                .map(move |projection| (*layer, *projection))
        })
        .collect::<Vec<_>>();
    let mut candidates = Vec::new();
    for (layer, projection) in coordinates {
        let rows = backend.architecture().tensor(layer, projection)?.rows;
        let scales = backend.row_scales(layer, projection)?;
        if scales.len() != rows {
            return Err(Error::InvalidSearch(format!(
                "L{layer}.{projection} returned {} scales for {rows} rows",
                scales.len()
            )));
        }
        for (row, scale) in scales.into_iter().enumerate() {
            candidates.push(Candidate {
                flip: PatchFlip {
                    layer,
                    proj: projection,
                    row,
                },
                weight: if scale.is_finite() && scale > 0.0 {
                    f64::from(scale)
                } else {
                    0.0
                },
            });
        }
    }
    Ok(candidates)
}

fn sample_candidate<'a>(
    candidates: &'a [Candidate],
    tried: &BTreeSet<PatchFlip>,
    rng: &mut SplitMix64,
) -> Option<&'a Candidate> {
    let available = candidates
        .iter()
        .filter(|candidate| !tried.contains(&candidate.flip))
        .collect::<Vec<_>>();
    if available.is_empty() {
        return None;
    }
    let total = available
        .iter()
        .map(|candidate| candidate.weight)
        .sum::<f64>();
    if total <= 0.0 || !total.is_finite() {
        let index = (rng.next_u64() as usize) % available.len();
        return Some(available[index]);
    }
    let mut threshold = rng.next_f64() * total;
    for candidate in &available {
        if threshold < candidate.weight {
            return Some(candidate);
        }
        threshold -= candidate.weight;
    }
    available.last().copied()
}

fn screen_indices(baseline: &[ProbeMeasurement], count: usize) -> Vec<usize> {
    let mut indices = (0..baseline.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        baseline[*left]
            .gap
            .total_cmp(&baseline[*right].gap)
            .then_with(|| baseline[*left].name.cmp(&baseline[*right].name))
    });
    indices.truncate(count.min(indices.len()));
    indices
}

fn apply_coordinates<B: MiyagiBackend>(
    backend: &mut B,
    flips: &[PatchFlip],
    operation: &str,
) -> Result<()> {
    let mut applied: Vec<PatchFlip> = Vec::new();
    for flip in flips {
        if let Err(error) = backend.flip_row(flip.layer, flip.proj, flip.row) {
            for prior in applied.iter().rev() {
                if let Err(source) = backend.flip_row(prior.layer, prior.proj, prior.row) {
                    return Err(Error::RestorationFailed {
                        operation: operation.to_owned(),
                        source: Box::new(source),
                    });
                }
            }
            return Err(error);
        }
        applied.push(flip.clone());
    }
    Ok(())
}

fn revert_candidate<B: MiyagiBackend>(
    backend: &mut B,
    flip: &PatchFlip,
    operation: &str,
) -> Result<()> {
    backend
        .flip_row(flip.layer, flip.proj, flip.row)
        .map_err(|source| Error::RestorationFailed {
            operation: operation.to_owned(),
            source: Box::new(source),
        })
}

#[allow(clippy::too_many_arguments)]
fn checkpoint(
    architecture_signature: &str,
    model_label: &str,
    config: &SearchConfig,
    completed_iterations: usize,
    accepted: &[PatchFlip],
    tried: &BTreeSet<PatchFlip>,
    current_fitness: f32,
    rng_state: u64,
    target_baseline: &[ProbeMeasurement],
    control_baseline: &[ProbeMeasurement],
) -> SearchCheckpoint {
    SearchCheckpoint {
        version: CHECKPOINT_VERSION,
        architecture_signature: architecture_signature.to_owned(),
        model_label: model_label.to_owned(),
        config: config.clone(),
        completed_iterations,
        accepted: accepted.to_vec(),
        tried: tried.clone(),
        current_fitness,
        rng_state,
        target_baseline: target_baseline.to_vec(),
        control_baseline: control_baseline.to_vec(),
    }
}

fn validate_checkpoint(
    checkpoint: &SearchCheckpoint,
    config: &SearchConfig,
    architecture_signature: &str,
    target_baseline: &[ProbeMeasurement],
    control_baseline: &[ProbeMeasurement],
) -> Result<()> {
    if checkpoint.version != CHECKPOINT_VERSION {
        return Err(Error::IncompatibleCheckpoint(format!(
            "unsupported checkpoint version {}",
            checkpoint.version
        )));
    }
    let mut checkpoint_config = checkpoint.config.clone();
    checkpoint_config.max_iters = config.max_iters;
    if checkpoint_config != *config || config.max_iters < checkpoint.completed_iterations {
        return Err(Error::IncompatibleCheckpoint(
            "search configuration changed".to_owned(),
        ));
    }
    if checkpoint.architecture_signature != architecture_signature {
        return Err(Error::IncompatibleCheckpoint(
            "model architecture signature changed".to_owned(),
        ));
    }
    compare_baselines(&checkpoint.target_baseline, target_baseline)?;
    compare_baselines(&checkpoint.control_baseline, control_baseline)?;
    Ok(())
}

fn compare_baselines(expected: &[ProbeMeasurement], actual: &[ProbeMeasurement]) -> Result<()> {
    if expected.len() != actual.len() {
        return Err(Error::IncompatibleCheckpoint(
            "probe count changed".to_owned(),
        ));
    }
    for (expected, actual) in expected.iter().zip(actual) {
        if expected.name != actual.name || (expected.gap - actual.gap).abs() > 1e-5 {
            return Err(Error::IncompatibleCheckpoint(format!(
                "baseline changed for probe {}",
                expected.name
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn from_state(state: u64) -> Self {
        Self { state }
    }

    fn state(self) -> u64 {
        self.state
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * SCALE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_state_can_resume_exactly() {
        let mut first = SplitMix64::new(42);
        let _ = first.next_u64();
        let state = first.state();
        let expected = first.next_u64();
        let mut resumed = SplitMix64::from_state(state);
        assert_eq!(resumed.next_u64(), expected);
    }

    #[test]
    fn default_layers_span_full_depth_on_large_models() {
        // 64-layer model must reach deep layers, unlike the old fixed list.
        let layers = default_search_layers(64);
        assert_eq!(layers.len(), MAX_DEFAULT_LAYERS);
        assert_eq!(*layers.first().unwrap(), 1);
        assert_eq!(*layers.last().unwrap(), 63);
        assert!(layers.iter().all(|&l| l < 64));
        assert!(layers.windows(2).all(|w| w[0] < w[1]), "strictly increasing");
        // The old default's deepest layer was 34; the new one goes well past it.
        assert!(layers.iter().any(|&l| l > 34));
    }

    #[test]
    fn default_layers_cover_every_layer_on_small_models() {
        // Small models intentionally include layer 0 (see default_search_layers docs).
        assert_eq!(default_search_layers(4), vec![0, 1, 2, 3]);
        assert_eq!(default_search_layers(0), Vec::<usize>::new());
    }

    #[test]
    fn empty_search_layers_resolve_from_backend_depth() {
        let resolved = resolve_search_layers(64, &[]);
        assert_eq!(resolved, default_search_layers(64));
        assert!(resolved.iter().any(|&l| l > 34));
        let explicit = resolve_search_layers(64, &[1, 2, 3]);
        assert_eq!(explicit, vec![1, 2, 3]);
    }

    #[test]
    fn screen_uses_worst_gaps_then_name() {
        fn measurement(name: &str, gap: f32) -> ProbeMeasurement {
            ProbeMeasurement {
                name: name.to_owned(),
                category: String::new(),
                prompt: String::new(),
                correct_token: String::new(),
                wrong_token: String::new(),
                correct_id: 0,
                wrong_id: 1,
                gap,
            }
        }
        let baseline = [
            measurement("b", -1.0),
            measurement("a", -1.0),
            measurement("c", 0.0),
        ];
        assert_eq!(screen_indices(&baseline, 2), vec![1, 0]);
    }
}
