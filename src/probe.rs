use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::backend::{MiyagiBackend, TokenMode};
use crate::error::{Error, Result};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Probe {
    pub prompt: String,
    #[serde(rename = "correct", alias = "correct_token")]
    pub correct_token: String,
    #[serde(rename = "wrong", alias = "wrong_token")]
    pub wrong_token: String,
    pub name: String,
    #[serde(default = "default_category")]
    pub category: String,
}

fn default_category() -> String {
    "general".to_owned()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CompiledProbe {
    pub probe: Probe,
    pub prompt_tokens: Vec<i32>,
    pub correct_id: i32,
    pub wrong_id: i32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ProbeMeasurement {
    pub name: String,
    pub category: String,
    pub prompt: String,
    pub correct_token: String,
    pub wrong_token: String,
    pub correct_id: i32,
    pub wrong_id: i32,
    pub gap: f32,
}

impl Probe {
    pub fn new(
        prompt: impl Into<String>,
        correct_token: impl Into<String>,
        wrong_token: impl Into<String>,
        name: impl Into<String>,
        category: impl Into<String>,
    ) -> Self {
        Self {
            prompt: prompt.into(),
            correct_token: correct_token.into(),
            wrong_token: wrong_token.into(),
            name: name.into(),
            category: category.into(),
        }
    }

    pub fn compile<B: MiyagiBackend>(&self, backend: &B, mode: TokenMode) -> Result<CompiledProbe> {
        validate_probe(self)?;
        let prompt_tokens = backend.tokenize(&self.prompt)?;
        if prompt_tokens.is_empty() {
            return Err(Error::EmptyProbeField {
                probe: self.name.clone(),
                field: "tokenized prompt",
            });
        }
        let correct_id = resolve_answer(backend, &self.name, &self.correct_token, mode)?;
        let wrong_id = resolve_answer(backend, &self.name, &self.wrong_token, mode)?;
        Ok(CompiledProbe {
            probe: self.clone(),
            prompt_tokens,
            correct_id,
            wrong_id,
        })
    }
}

pub fn compile_probes<B: MiyagiBackend>(
    backend: &B,
    probes: &[Probe],
    mode: TokenMode,
) -> Result<Vec<CompiledProbe>> {
    validate_probe_set(probes)?;
    probes
        .iter()
        .map(|probe| probe.compile(backend, mode))
        .collect()
}

/// Measure probe logit gaps. Non-finite gaps are a hard error — used by `eval`
/// and search baselines so reports never contain silent NaN transitions.
pub fn measure_probes<B: MiyagiBackend>(
    backend: &mut B,
    probes: &[CompiledProbe],
) -> Result<Vec<ProbeMeasurement>> {
    let measurements = measure_probes_allowing_non_finite(backend, probes)?;
    for measurement in &measurements {
        if !measurement.gap.is_finite() {
            return Err(Error::MeasurementMismatch(format!(
                "probe {} produced non-finite gap {}",
                measurement.name, measurement.gap
            )));
        }
    }
    Ok(measurements)
}

/// Like [`measure_probes`], but returns non-finite gaps as data.
///
/// Search candidate evaluation uses this so a NaN from one flip can be rejected
/// and reverted without aborting the whole search. Callers must handle NaN.
pub fn measure_probes_allowing_non_finite<B: MiyagiBackend>(
    backend: &mut B,
    probes: &[CompiledProbe],
) -> Result<Vec<ProbeMeasurement>> {
    if probes.is_empty() {
        return Err(Error::EmptyProbeSet);
    }
    probes
        .iter()
        .map(|compiled| {
            let gap = backend.logit_gap(
                &compiled.prompt_tokens,
                compiled.correct_id,
                compiled.wrong_id,
            )?;
            Ok(ProbeMeasurement {
                name: compiled.probe.name.clone(),
                category: compiled.probe.category.clone(),
                prompt: compiled.probe.prompt.clone(),
                correct_token: compiled.probe.correct_token.clone(),
                wrong_token: compiled.probe.wrong_token.clone(),
                correct_id: compiled.correct_id,
                wrong_id: compiled.wrong_id,
                gap,
            })
        })
        .collect()
}

pub fn load_probe_file(path: impl AsRef<Path>) -> Result<Vec<Probe>> {
    let probes = serde_json::from_str::<Vec<Probe>>(&fs::read_to_string(path)?)?;
    validate_probe_set(&probes)?;
    Ok(probes)
}

pub fn built_in(name: &str) -> Option<Vec<Probe>> {
    match name {
        "math" => Some(math_probes()),
        "code" => Some(code_probes()),
        "knowledge" => Some(knowledge_probes()),
        _ => None,
    }
}

pub fn math_probes() -> Vec<Probe> {
    vec![
        Probe::new("1 + 1 =", " 2", " 3", "add_1", "math"),
        Probe::new("2 + 2 =", " 4", " 5", "add_2", "math"),
        Probe::new("7 * 8 =", " 56", " 54", "mul_1", "math"),
        Probe::new("The square root of 144 is", " 12", " 14", "sqrt_1", "math"),
        Probe::new("If x = 3, then x^2 =", " 9", " 8", "algebra_1", "math"),
        Probe::new("100 / 4 =", " 25", " 20", "div_1", "math"),
    ]
}

pub fn code_probes() -> Vec<Probe> {
    vec![
        Probe::new(
            "def hello():\n    print(\"Hello",
            " World",
            " Goodbye",
            "hello_world",
            "code",
        ),
        Probe::new(
            "In Python, to open a file you use the",
            " open",
            " close",
            "python_open",
            "code",
        ),
        Probe::new(
            "for i in range(10):\n    print(",
            "i",
            "x",
            "for_loop",
            "code",
        ),
        Probe::new(
            "import json\ndata = json.",
            "loads",
            "dump",
            "json_loads",
            "code",
        ),
    ]
}

pub fn knowledge_probes() -> Vec<Probe> {
    vec![
        Probe::new(
            "The capital of France is",
            " Paris",
            " London",
            "france_capital",
            "knowledge",
        ),
        Probe::new(
            "The capital of Japan is",
            " Tokyo",
            " Beijing",
            "japan_capital",
            "knowledge",
        ),
        Probe::new(
            "The color of the sky is",
            " blue",
            " red",
            "sky_color",
            "knowledge",
        ),
        Probe::new(
            "Einstein is famous for the theory of",
            " relativity",
            " evolution",
            "einstein",
            "knowledge",
        ),
        Probe::new(
            "The chemical formula for water is H",
            "2",
            "3",
            "water_formula",
            "knowledge",
        ),
    ]
}

fn validate_probe_set(probes: &[Probe]) -> Result<()> {
    if probes.is_empty() {
        return Err(Error::EmptyProbeSet);
    }
    let mut names = BTreeSet::new();
    for probe in probes {
        validate_probe(probe)?;
        if !names.insert(probe.name.as_str()) {
            return Err(Error::DuplicateProbe(probe.name.clone()));
        }
    }
    Ok(())
}

fn validate_probe(probe: &Probe) -> Result<()> {
    for (field, value) in [
        ("name", probe.name.as_str()),
        ("prompt", probe.prompt.as_str()),
        ("correct", probe.correct_token.as_str()),
        ("wrong", probe.wrong_token.as_str()),
    ] {
        if value.is_empty() {
            return Err(Error::EmptyProbeField {
                probe: probe.name.clone(),
                field,
            });
        }
    }
    Ok(())
}

fn resolve_answer<B: MiyagiBackend>(
    backend: &B,
    probe_name: &str,
    text: &str,
    mode: TokenMode,
) -> Result<i32> {
    let tokens = backend.tokenize(text)?;
    if tokens.is_empty() {
        return Err(Error::EmptyAnswerToken {
            probe: probe_name.to_owned(),
        });
    }
    if mode == TokenMode::StrictSingle && tokens.len() != 1 {
        return Err(Error::AmbiguousAnswerToken {
            probe: probe_name.to_owned(),
            count: tokens.len(),
        });
    }
    Ok(*tokens.last().expect("token list checked as non-empty"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_probe_field_aliases_parse() {
        let probes: Vec<Probe> =
            serde_json::from_str(r#"[{"prompt":"p","correct":" a","wrong":" b","name":"n"}]"#)
                .unwrap();
        assert_eq!(probes[0].correct_token, " a");
        assert_eq!(probes[0].category, "general");
    }

    #[test]
    fn built_in_names_are_unique() {
        for probes in [math_probes(), code_probes(), knowledge_probes()] {
            validate_probe_set(&probes).unwrap();
        }
    }
}
