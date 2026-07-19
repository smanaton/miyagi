use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use clap::{Args, Parser, Subcommand, ValueEnum};
use regex::Regex;
use serde::Serialize;

use crate::architecture::{ArchitectureMap, Projection};
use crate::backend::{BackendConfig, GenerateConfig, MiyagiBackend, TokenMode, WwamaBackend};
use crate::dataset::{DatasetConfig, evaluate_dataset, load_records};
use crate::error::{Error, Result};
use crate::evaluation::compare_measurements;
use crate::fitness::FitnessMode;
use crate::patch::{Patch, PatchValidation};
use crate::probe::{Probe, built_in, compile_probes, load_probe_file, measure_probes};
use crate::search::{SearchCheckpoint, SearchConfig, SearchEvent, default_search_layers, run_search};

#[derive(Parser)]
#[command(
    name = "miyagi",
    about = "Sparse XOR adaptation for true binary GGUF models"
)]
struct Cli {
    #[arg(long, global = true, help = "Emit machine-readable JSON")]
    json: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Inspect(InspectArgs),
    Info(InfoArgs),
    Compose(ComposeArgs),
    Eval(EvalArgs),
    Apply(ApplyArgs),
    Search(SearchArgs),
    Benchmark(BenchmarkArgs),
}

#[derive(Clone, Args)]
struct ModelArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value_t = 2048)]
    n_ctx: u32,
    #[arg(long, default_value_t = 512)]
    n_batch: u32,
    #[arg(long, default_value_t = 512)]
    n_ubatch: u32,
    #[arg(long, default_value_t = 0)]
    n_threads: i32,
    #[arg(long, default_value_t = 0)]
    n_threads_batch: i32,
    #[arg(long, default_value_t = 999)]
    n_gpu_layers: i32,
    #[arg(long, default_value_t = false)]
    add_special: bool,
}

impl ModelArgs {
    fn config(&self, mutable_tensors: bool) -> BackendConfig {
        BackendConfig {
            n_ctx: self.n_ctx,
            n_batch: self.n_batch,
            n_ubatch: self.n_ubatch,
            n_threads: self.n_threads,
            n_threads_batch: self.n_threads_batch,
            n_gpu_layers: self.n_gpu_layers,
            mutable_tensors,
            add_special: self.add_special,
            parse_special: true,
        }
    }

    fn model_str(&self) -> Result<&str> {
        self.model
            .to_str()
            .ok_or_else(|| Error::InvalidSearch("model path is not valid UTF-8".to_owned()))
    }
}

#[derive(Args)]
struct InspectArgs {
    #[command(flatten)]
    model: ModelArgs,
    #[arg(long, help = "Print every model tensor, not only Miyagi mappings")]
    all_tensors: bool,
}

#[derive(Args)]
struct InfoArgs {
    #[arg(long)]
    patch: PathBuf,
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    n_gpu_layers: i32,
    #[arg(long)]
    allow_model_mismatch: bool,
}

#[derive(Args)]
struct ComposeArgs {
    #[arg(long, required = true, num_args = 2..)]
    patch: Vec<PathBuf>,
    #[arg(long)]
    name: String,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Args)]
struct EvalArgs {
    #[command(flatten)]
    model: ModelArgs,
    #[arg(long)]
    patch: PathBuf,
    #[arg(long, value_delimiter = ',', default_value = "math,code,knowledge")]
    probes: Vec<String>,
    #[arg(long, value_enum, default_value_t = TokenModeArg::Compatibility)]
    token_mode: TokenModeArg,
    #[arg(long, default_value_t = 0.1)]
    change_threshold: f32,
    #[arg(long)]
    allow_model_mismatch: bool,
    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Args)]
struct ApplyArgs {
    #[command(flatten)]
    model: ModelArgs,
    #[arg(long)]
    patch: PathBuf,
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value_t = 100)]
    max_tokens: usize,
    #[arg(long, default_value_t = 42)]
    seed: u32,
    #[arg(long)]
    allow_model_mismatch: bool,
}

#[derive(Args)]
struct SearchArgs {
    #[command(flatten)]
    model: ModelArgs,
    #[arg(long, help = "Built-in probe set name or JSON file")]
    target: String,
    #[arg(long, help = "Built-in probe set or JSON file; may be repeated")]
    control: Vec<String>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    report: Option<PathBuf>,
    #[arg(long, default_value_t = 200)]
    iters: usize,
    #[arg(
        long,
        value_delimiter = ',',
        help = "Comma-separated layer indices to search. Default: an even spread across the model's full depth (derived from its layer count)."
    )]
    layers: Option<Vec<usize>>,
    #[arg(long, value_delimiter = ',', default_value = "gate_proj,up_proj")]
    projections: Vec<String>,
    #[arg(long, default_value_t = 2.0)]
    penalty: f32,
    #[arg(long, value_enum, default_value_t = FitnessModeArg::Mean)]
    fitness: FitnessModeArg,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 2)]
    screen_probes: usize,
    #[arg(long, value_enum, default_value_t = TokenModeArg::Compatibility)]
    token_mode: TokenModeArg,
    #[arg(long, default_value = "untitled")]
    name: String,
    #[arg(long, default_value = "")]
    description: String,
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    #[arg(long)]
    resume: Option<PathBuf>,
}

#[derive(Args)]
struct BenchmarkArgs {
    #[command(flatten)]
    model: ModelArgs,
    #[arg(long)]
    dataset: PathBuf,
    #[arg(long)]
    patch: Option<PathBuf>,
    #[arg(long, default_value = "question")]
    question_field: String,
    #[arg(long, default_value = "answer")]
    answer_field: String,
    #[arg(
        long,
        default_value = "Solve this problem and end with 'The answer is [number]'.\n\n{question}"
    )]
    prompt_template: String,
    #[arg(long, default_value = r"(?i)the answer is[:\s]*\$?([\-\d,]+)")]
    answer_regex: String,
    #[arg(long, default_value = r"####\s*([\-\d,]+)")]
    gold_regex: String,
    #[arg(long)]
    limit: Option<usize>,
    #[arg(long, default_value_t = 400)]
    max_tokens: usize,
    #[arg(long, default_value_t = 42)]
    seed: u32,
    #[arg(long)]
    allow_model_mismatch: bool,
    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum TokenModeArg {
    Compatibility,
    Strict,
}

impl From<TokenModeArg> for TokenMode {
    fn from(value: TokenModeArg) -> Self {
        match value {
            TokenModeArg::Compatibility => Self::LastTokenCompatibility,
            TokenModeArg::Strict => Self::StrictSingle,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum FitnessModeArg {
    Mean,
    Min,
}

impl From<FitnessModeArg> for FitnessMode {
    fn from(value: FitnessModeArg) -> Self {
        match value {
            FitnessModeArg::Mean => Self::Mean,
            FitnessModeArg::Min => Self::Min,
        }
    }
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect(args) => inspect(args, cli.json),
        Command::Info(args) => info(args, cli.json),
        Command::Compose(args) => compose(args, cli.json),
        Command::Eval(args) => eval(args, cli.json),
        Command::Apply(args) => apply(args, cli.json),
        Command::Search(args) => search(args, cli.json),
        Command::Benchmark(args) => benchmark(args, cli.json),
    }
}

#[derive(Serialize)]
struct InspectReport {
    model: String,
    tensor_count: usize,
    miyagi_supported: bool,
    architecture: Option<ArchitectureMap>,
    architecture_error: Option<String>,
    tensors: Option<Vec<InventoryTensor>>,
}

#[derive(Serialize)]
struct InventoryTensor {
    name: String,
    type_name: String,
    dimensions: [u64; 4],
    strides: [usize; 4],
    nbytes: usize,
    backend: String,
}

fn inspect(args: InspectArgs, json: bool) -> Result<()> {
    let mut config = args.model.config(false);
    config.mutable_tensors = false;
    let session =
        wwama::Session::load_from_path(args.model.model_str()?, config.session_options())?;
    let descriptors = session.model().tensors()?;
    let discovered = ArchitectureMap::discover(&descriptors);
    let report = InspectReport {
        model: args.model.model.display().to_string(),
        tensor_count: descriptors.len(),
        miyagi_supported: discovered.is_ok(),
        architecture_error: discovered.as_ref().err().map(ToString::to_string),
        architecture: discovered.ok(),
        tensors: args.all_tensors.then(|| {
            descriptors
                .into_iter()
                .map(|tensor| InventoryTensor {
                    name: tensor.name,
                    type_name: tensor.type_name,
                    dimensions: tensor.dimensions,
                    strides: tensor.strides,
                    nbytes: tensor.nbytes,
                    backend: tensor.backend,
                })
                .collect()
        }),
    };
    if json {
        print_json(&report)
    } else {
        println!("Model: {}", report.model);
        println!("Tensors: {}", report.tensor_count);
        println!("Miyagi Q1_0 support: {}", report.miyagi_supported);
        if let Some(architecture) = &report.architecture {
            println!("Layers: {}", architecture.layer_count());
            println!("Architecture signature: {}", architecture.signature());
            for tensor in architecture.tensors() {
                println!(
                    "  L{}.{} -> {} rows={} width={} backend={}",
                    tensor.layer,
                    tensor.projection,
                    tensor.name,
                    tensor.rows,
                    tensor.width,
                    tensor.backend
                );
            }
        } else if let Some(error) = &report.architecture_error {
            println!("Capability: {error}");
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct PatchInfoReport {
    patch: Patch,
    json_size_bytes: u64,
    live_model_validated: bool,
    logical_bits_flipped: Option<u64>,
    architecture_signature: Option<String>,
}

fn info(args: InfoArgs, json: bool) -> Result<()> {
    let patch = Patch::load(&args.patch)?;
    let json_size_bytes = fs::metadata(&args.patch)?.len();
    let (patch, live_model_validated, logical_bits_flipped, signature) =
        if let Some(model) = args.model {
            let backend = WwamaBackend::load(
                model.to_str().ok_or_else(|| {
                    Error::InvalidSearch("model path is not valid UTF-8".to_owned())
                })?,
                BackendConfig {
                    n_gpu_layers: args.n_gpu_layers,
                    mutable_tensors: false,
                    ..BackendConfig::default()
                },
            )?;
            let validated = patch.validate(
                backend.architecture(),
                PatchValidation {
                    allow_model_signature_mismatch: args.allow_model_mismatch,
                },
            )?;
            let bits = validated.logical_bits_flipped();
            let signature = backend.architecture().signature().to_owned();
            (validated.into_patch(), true, Some(bits), Some(signature))
        } else {
            (patch, false, None, None)
        };
    let report = PatchInfoReport {
        patch,
        json_size_bytes,
        live_model_validated,
        logical_bits_flipped,
        architecture_signature: signature,
    };
    if json {
        print_json(&report)
    } else {
        println!("Patch: {}", report.patch.name);
        println!("Description: {}", report.patch.description);
        println!("Base model: {}", report.patch.base_model);
        println!("Flips: {}", report.patch.flips.len());
        println!("JSON size: {} bytes", report.json_size_bytes);
        println!(
            "Compact binary estimate: {} bytes",
            report.patch.stats.compact_binary_estimate_bytes
        );
        if let Some(bits) = report.logical_bits_flipped {
            println!("Logical bits flipped: {bits}");
        }
        println!("Live model validated: {}", report.live_model_validated);
        Ok(())
    }
}

#[derive(Serialize)]
struct ComposeReport {
    patch: Patch,
    output: String,
    json_size_bytes: usize,
}

fn compose(args: ComposeArgs, json: bool) -> Result<()> {
    let patches = args
        .patch
        .iter()
        .map(Patch::load)
        .collect::<Result<Vec<_>>>()?;
    let patch = Patch::compose(args.name, &patches)?;
    let json_size_bytes = patch.save_atomic(&args.output)?;
    let report = ComposeReport {
        patch,
        output: args.output.display().to_string(),
        json_size_bytes,
    };
    if json {
        print_json(&report)
    } else {
        println!("Composed patch saved to {}", report.output);
        println!("Flips: {}", report.patch.flips.len());
        println!("JSON size: {} bytes", report.json_size_bytes);
        Ok(())
    }
}

fn eval(args: EvalArgs, json: bool) -> Result<()> {
    let probes = load_selectors(&args.probes)?;
    let mut backend = WwamaBackend::load(args.model.model_str()?, args.model.config(true))?;
    let compiled = compile_probes(&backend, &probes, args.token_mode.into())?;
    let patch = Patch::load(&args.patch)?;
    let validated = patch.validate(
        backend.architecture(),
        PatchValidation {
            allow_model_signature_mismatch: args.allow_model_mismatch,
        },
    )?;
    let baseline = measure_probes(&mut backend, &compiled)?;
    validated.apply(&mut backend)?;
    let patched_result = measure_probes(&mut backend, &compiled);
    let removal_result = validated.remove(&mut backend);
    let patched = finish_patched_operation(patched_result, removal_result, "patch evaluation")?;
    let report = compare_measurements(&baseline, &patched, args.change_threshold)?;
    if let Some(path) = args.report {
        write_json_atomic(&path, &report)?;
    }
    if json {
        print_json(&report)
    } else {
        print_evaluation(&report);
        Ok(())
    }
}

#[derive(Serialize)]
struct ApplyReport {
    patch: String,
    prompt: String,
    baseline: String,
    patched: String,
    restored: bool,
}

fn apply(args: ApplyArgs, json: bool) -> Result<()> {
    let mut backend = WwamaBackend::load(args.model.model_str()?, args.model.config(true))?;
    let patch = Patch::load(&args.patch)?;
    let validated = patch.validate(
        backend.architecture(),
        PatchValidation {
            allow_model_signature_mismatch: args.allow_model_mismatch,
        },
    )?;
    let generation = GenerateConfig {
        max_new_tokens: args.max_tokens,
        seed: args.seed,
        ..GenerateConfig::default()
    };
    let baseline = backend.generate(&args.prompt, &generation)?;
    validated.apply(&mut backend)?;
    let patched_result = backend.generate(&args.prompt, &generation);
    let removal_result = validated.remove(&mut backend);
    let patched = finish_patched_operation(patched_result, removal_result, "patched generation")?;
    let report = ApplyReport {
        patch: validated.patch().name.clone(),
        prompt: args.prompt,
        baseline,
        patched,
        restored: true,
    };
    if json {
        print_json(&report)
    } else {
        println!("Without patch:\n{}", report.baseline);
        println!("\nWith patch {}:\n{}", report.patch, report.patched);
        println!("\nModel state restored before exit.");
        Ok(())
    }
}

fn search(args: SearchArgs, json: bool) -> Result<()> {
    let target = load_selector(&args.target)?;
    let controls = if args.control.is_empty() {
        default_controls(&args.target)?
    } else {
        load_selectors(&args.control)?
    };
    let mut backend = WwamaBackend::load(args.model.model_str()?, args.model.config(true))?;
    let token_mode = args.token_mode.into();
    let target = compile_probes(&backend, &target, token_mode)?;
    let controls = compile_probes(&backend, &controls, token_mode)?;
    let projections = args
        .projections
        .iter()
        .map(|projection| Projection::from_str(projection))
        .collect::<Result<Vec<_>>>()?;
    let config = SearchConfig {
        search_layers: args
            .layers
            .unwrap_or_else(|| default_search_layers(backend.architecture().layer_count())),
        search_projections: projections,
        max_iters: args.iters,
        control_penalty: args.penalty,
        fitness_mode: args.fitness.into(),
        seed: args.seed,
        screen_probe_count: args.screen_probes,
        patch_name: args.name,
        patch_description: args.description,
        base_model: args.model.model.display().to_string(),
    };
    let resume = args
        .resume
        .as_ref()
        .map(SearchCheckpoint::load)
        .transpose()?;
    let checkpoint_path = args.checkpoint.as_deref().or(args.resume.as_deref());
    let cancelled = Arc::new(AtomicBool::new(false));
    #[cfg(not(target_arch = "wasm32"))]
    {
        let signal_flag = Arc::clone(&cancelled);
        ctrlc::set_handler(move || {
            signal_flag.store(true, std::sync::atomic::Ordering::Relaxed);
        })
        .map_err(|error| {
            Error::InvalidSearch(format!("failed to install signal handler: {error}"))
        })?;
    }

    let result = run_search(
        &mut backend,
        &target,
        &controls,
        config,
        resume,
        checkpoint_path,
        Some(&cancelled),
        |event| print_search_event(event, json),
    )?;
    let json_size_bytes = result.patch.save_atomic(&args.output)?;
    if let Some(path) = args.report {
        write_json_atomic(&path, &result)?;
    }
    if json {
        print_json(&result)
    } else {
        println!("Patch saved to {}", args.output.display());
        println!("Flips: {}", result.patch.flips.len());
        println!("Fitness: {:+.6}", result.final_fitness);
        println!("JSON size: {json_size_bytes} bytes");
        println!("Accepted patch remained applied until process exit.");
        Ok(())
    }
}

#[derive(Serialize)]
struct BenchmarkComparison {
    baseline: crate::dataset::DatasetReport,
    patched: Option<crate::dataset::DatasetReport>,
    model_restored: bool,
}

fn benchmark(args: BenchmarkArgs, json: bool) -> Result<()> {
    let records = load_records(&args.dataset)?;
    let mut backend = WwamaBackend::load(
        args.model.model_str()?,
        args.model.config(args.patch.is_some()),
    )?;
    let config = DatasetConfig {
        question_field: args.question_field,
        answer_field: args.answer_field,
        prompt_template: args.prompt_template,
        answer_regex: compile_capture_regex(&args.answer_regex)?,
        gold_regex: compile_capture_regex(&args.gold_regex)?,
        limit: args.limit,
        generation: GenerateConfig {
            max_new_tokens: args.max_tokens,
            seed: args.seed,
            ..GenerateConfig::default()
        },
    };
    let baseline = evaluate_dataset(&mut backend, &records, &config)?;
    let patched = if let Some(path) = args.patch {
        let patch = Patch::load(path)?;
        let validated = patch.validate(
            backend.architecture(),
            PatchValidation {
                allow_model_signature_mismatch: args.allow_model_mismatch,
            },
        )?;
        validated.apply(&mut backend)?;
        let patched_result = evaluate_dataset(&mut backend, &records, &config);
        let removal_result = validated.remove(&mut backend);
        Some(finish_patched_operation(
            patched_result,
            removal_result,
            "dataset evaluation",
        )?)
    } else {
        None
    };
    let report = BenchmarkComparison {
        baseline,
        patched,
        model_restored: true,
    };
    if let Some(path) = args.report {
        write_json_atomic(&path, &report)?;
    }
    if json {
        print_json(&report)
    } else {
        println!(
            "Baseline: {}/{} correct",
            report.baseline.correct, report.baseline.total
        );
        if let Some(patched) = &report.patched {
            println!("Patched: {}/{} correct", patched.correct, patched.total);
        }
        println!("Model state restored before exit.");
        Ok(())
    }
}

fn load_selector(selector: &str) -> Result<Vec<Probe>> {
    if let Some(probes) = built_in(selector) {
        return Ok(probes);
    }
    let path = Path::new(selector);
    if path.is_file() {
        return load_probe_file(path);
    }
    Err(Error::InvalidSearch(format!(
        "unknown probe selector {selector:?}"
    )))
}

fn load_selectors(selectors: &[String]) -> Result<Vec<Probe>> {
    let mut probes = Vec::new();
    for selector in selectors {
        probes.extend(load_selector(selector)?);
    }
    Ok(probes)
}

fn default_controls(target: &str) -> Result<Vec<Probe>> {
    let mut controls = Vec::new();
    for name in ["math", "code", "knowledge"] {
        if name != target {
            controls.extend(built_in(name).expect("known built-in probe set"));
        }
    }
    if controls.is_empty() {
        return Err(Error::EmptyProbeSet);
    }
    Ok(controls)
}

fn finish_patched_operation<T>(operation: Result<T>, removal: Result<()>, name: &str) -> Result<T> {
    if let Err(source) = removal {
        return Err(Error::RestorationFailed {
            operation: name.to_owned(),
            source: Box::new(source),
        });
    }
    operation
}

fn compile_capture_regex(pattern: &str) -> Result<Regex> {
    let regex = Regex::new(pattern)
        .map_err(|error| Error::InvalidSearch(format!("invalid regex {pattern:?}: {error}")))?;
    if regex.captures_len() < 2 {
        return Err(Error::MissingRegexCapture(pattern.to_owned()));
    }
    Ok(regex)
}

fn print_search_event(event: &SearchEvent, json: bool) {
    if json {
        if let Ok(line) = serde_json::to_string(event) {
            println!("{line}");
        }
        return;
    }
    match event {
        SearchEvent::Baseline { target, control } => {
            println!(
                "Baseline measured: {} target probes, {} controls",
                target.len(),
                control.len()
            );
        }
        SearchEvent::Accepted {
            iteration,
            flip,
            fitness,
            accepted,
        } => println!(
            "[{iteration}] accept {} fitness={fitness:+.6} accepted={accepted}",
            flip.coordinate()
        ),
        SearchEvent::Completed {
            iterations,
            accepted,
            fitness,
        } => println!(
            "Search complete: iterations={iterations} accepted={accepted} fitness={fitness:+.6}"
        ),
        _ => {}
    }
}

fn print_evaluation(report: &crate::evaluation::EvaluationSummary) {
    println!(
        "{:<24} {:>10} {:>10} {:>10} {:>14}",
        "Probe", "Baseline", "Patched", "Delta", "Transition"
    );
    for probe in &report.probes {
        println!(
            "{:<24} {:+10.3} {:+10.3} {:+10.3} {:>14?}",
            probe.name, probe.baseline, probe.patched, probe.delta, probe.transition
        );
    }
    println!(
        "fixed={} broke={} stayed_right={} stayed_wrong={}",
        report.fixed, report.broke, report.stayed_right, report.stayed_wrong
    );
}

fn print_json<T: Serialize>(value: &T) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let temporary = path.with_extension(format!(
        "{}.{}.tmp",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("json"),
        std::process::id()
    ));
    fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(error.into());
    }
    Ok(())
}
