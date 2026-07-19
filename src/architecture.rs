use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use wwama::TensorDescriptor;

use crate::error::{Error, Result};

const Q1_0_TYPE_ID: i32 = 41;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Projection {
    Gate,
    Up,
    Down,
}

impl Projection {
    pub const ALL: [Self; 3] = [Self::Gate, Self::Up, Self::Down];

    pub const fn projection_name(self) -> &'static str {
        match self {
            Self::Gate => "gate_proj",
            Self::Up => "up_proj",
            Self::Down => "down_proj",
        }
    }

    const fn gguf_suffix(self) -> &'static str {
        match self {
            Self::Gate => ".ffn_gate.weight",
            Self::Up => ".ffn_up.weight",
            Self::Down => ".ffn_down.weight",
        }
    }
}

impl fmt::Display for Projection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.projection_name())
    }
}

impl FromStr for Projection {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "gate" | "gate_proj" | "ffn_gate" => Ok(Self::Gate),
            "up" | "up_proj" | "ffn_up" => Ok(Self::Up),
            "down" | "down_proj" | "ffn_down" => Ok(Self::Down),
            _ => Err(Error::InvalidProjection(value.to_owned())),
        }
    }
}

impl Serialize for Projection {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.projection_name())
    }
}

impl<'de> Deserialize<'de> for Projection {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TensorInfo {
    pub layer: usize,
    pub projection: Projection,
    pub name: String,
    pub type_id: i32,
    pub type_name: String,
    pub width: usize,
    pub rows: usize,
    pub strides: [usize; 4],
    pub nbytes: usize,
    pub backend: String,
}

impl TensorInfo {
    pub fn coordinate(&self) -> String {
        format!("L{}.{}", self.layer, self.projection)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ArchitectureMap {
    // Tuple keys cannot be JSON map keys ("key must be a string"); emit the
    // tensor list instead — each TensorInfo already carries layer+projection.
    #[serde(serialize_with = "serialize_tensor_map")]
    tensors: BTreeMap<(usize, Projection), TensorInfo>,
    layer_count: usize,
    signature: String,
}

fn serialize_tensor_map<S>(
    tensors: &BTreeMap<(usize, Projection), TensorInfo>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.collect_seq(tensors.values())
}

impl ArchitectureMap {
    pub fn discover(descriptors: &[TensorDescriptor]) -> Result<Self> {
        let mut tensors = BTreeMap::new();
        let mut layers = BTreeSet::new();

        for descriptor in descriptors {
            let Some((layer, projection)) = parse_qwen_tensor_name(&descriptor.name) else {
                continue;
            };
            let coordinate = format!("L{layer}.{projection}");
            if descriptor.type_id != Q1_0_TYPE_ID {
                return Err(Error::UnsupportedTensor {
                    coordinate,
                    reason: format!(
                        "expected Q1_0 type id {Q1_0_TYPE_ID}, got {} ({})",
                        descriptor.type_id, descriptor.type_name
                    ),
                });
            }
            let rows = descriptor
                .row_count()
                .map_err(|error| Error::UnsupportedTensor {
                    coordinate: coordinate.clone(),
                    reason: error.to_string(),
                })?;
            let width = usize::try_from(descriptor.dimensions[0]).map_err(|_| {
                Error::UnsupportedTensor {
                    coordinate: coordinate.clone(),
                    reason: "tensor width does not fit usize".to_owned(),
                }
            })?;
            if width == 0 || rows == 0 {
                return Err(Error::UnsupportedTensor {
                    coordinate,
                    reason: "tensor dimensions must be non-zero".to_owned(),
                });
            }
            let info = TensorInfo {
                layer,
                projection,
                name: descriptor.name.clone(),
                type_id: descriptor.type_id,
                type_name: descriptor.type_name.clone(),
                width,
                rows,
                strides: descriptor.strides,
                nbytes: descriptor.nbytes,
                backend: descriptor.backend.clone(),
            };
            if tensors.insert((layer, projection), info).is_some() {
                return Err(Error::DuplicateTensor(coordinate));
            }
            layers.insert(layer);
        }

        let Some(last_layer) = layers.last().copied() else {
            return Err(Error::UnsupportedArchitecture(
                "no blk.<layer>.ffn_{gate,up,down}.weight Q1_0 tensors were found".to_owned(),
            ));
        };
        let layer_count = last_layer + 1;
        for layer in 0..layer_count {
            if !layers.contains(&layer) {
                return Err(Error::UnsupportedArchitecture(format!(
                    "transformer layer {layer} is missing from the MLP tensor map"
                )));
            }
            for projection in Projection::ALL {
                if !tensors.contains_key(&(layer, projection)) {
                    return Err(Error::MissingTensor(format!("L{layer}.{projection}")));
                }
            }
        }

        let signature = architecture_signature(&tensors);
        Ok(Self {
            tensors,
            layer_count,
            signature,
        })
    }

    pub fn layer_count(&self) -> usize {
        self.layer_count
    }

    pub fn signature(&self) -> &str {
        &self.signature
    }

    pub fn tensor(&self, layer: usize, projection: Projection) -> Result<&TensorInfo> {
        self.tensors
            .get(&(layer, projection))
            .ok_or_else(|| Error::MissingTensor(format!("L{layer}.{projection}")))
    }

    pub fn tensors(&self) -> impl Iterator<Item = &TensorInfo> {
        self.tensors.values()
    }

    pub fn layers(&self) -> std::ops::Range<usize> {
        0..self.layer_count
    }
}

fn parse_qwen_tensor_name(name: &str) -> Option<(usize, Projection)> {
    let rest = name.strip_prefix("blk.")?;
    let (layer, suffix) = rest.split_once('.')?;
    let layer = layer.parse().ok()?;
    let suffix = format!(".{suffix}");
    Projection::ALL
        .into_iter()
        .find(|projection| suffix == projection.gguf_suffix())
        .map(|projection| (layer, projection))
}

fn architecture_signature(tensors: &BTreeMap<(usize, Projection), TensorInfo>) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for info in tensors.values() {
        for byte in format!(
            "{}|{}|{}|{}|{}|{};",
            info.layer, info.projection, info.name, info.type_id, info.width, info.rows
        )
        .bytes()
        {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("fnv1a64:{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(name: &str, width: u64, rows: u64) -> TensorDescriptor {
        TensorDescriptor {
            name: name.to_owned(),
            type_id: Q1_0_TYPE_ID,
            type_name: "Q1_0".to_owned(),
            dimensions: [width, rows, 1, 1],
            strides: [18, (width as usize / 128) * 18, 0, 0],
            n_dims: 2,
            nbytes: (width as usize / 128) * 18 * rows as usize,
            backend: "CPU".to_owned(),
        }
    }

    #[test]
    fn discovers_contiguous_qwen_layers() {
        let mut descriptors = Vec::new();
        for layer in 0..2 {
            descriptors.push(descriptor(
                &format!("blk.{layer}.ffn_gate.weight"),
                4096,
                12288,
            ));
            descriptors.push(descriptor(
                &format!("blk.{layer}.ffn_up.weight"),
                4096,
                12288,
            ));
            descriptors.push(descriptor(
                &format!("blk.{layer}.ffn_down.weight"),
                12288,
                4096,
            ));
        }
        let map = ArchitectureMap::discover(&descriptors).unwrap();
        assert_eq!(map.layer_count(), 2);
        assert_eq!(map.tensor(1, Projection::Down).unwrap().rows, 4096);
        assert_eq!(map.tensor(1, Projection::Down).unwrap().width, 12288);
        assert!(map.signature().starts_with("fnv1a64:"));
    }

    #[test]
    fn rejects_sparse_or_incomplete_maps() {
        let descriptors = [descriptor("blk.1.ffn_gate.weight", 4096, 12288)];
        assert!(matches!(
            ArchitectureMap::discover(&descriptors),
            Err(Error::UnsupportedArchitecture(_))
        ));
    }

    #[test]
    fn projection_aliases_round_trip() {
        assert_eq!("gate_proj".parse::<Projection>().unwrap(), Projection::Gate);
        assert_eq!("ffn_up".parse::<Projection>().unwrap(), Projection::Up);
        assert_eq!(
            serde_json::to_string(&Projection::Down).unwrap(),
            "\"down_proj\""
        );
    }

    #[test]
    fn architecture_map_json_serializes_tensors_as_array() {
        let mut descriptors = Vec::new();
        for layer in 0..2 {
            descriptors.push(descriptor(
                &format!("blk.{layer}.ffn_gate.weight"),
                4096,
                12288,
            ));
            descriptors.push(descriptor(
                &format!("blk.{layer}.ffn_up.weight"),
                4096,
                12288,
            ));
            descriptors.push(descriptor(
                &format!("blk.{layer}.ffn_down.weight"),
                12288,
                4096,
            ));
        }
        let map = ArchitectureMap::discover(&descriptors).unwrap();
        let value = serde_json::to_value(&map).expect("JSON serialize must not fail on tuple keys");
        let tensors = value
            .get("tensors")
            .and_then(|t| t.as_array())
            .expect("tensors must serialize as a JSON array");
        assert_eq!(tensors.len(), 6);
        assert!(tensors.iter().all(|t| t.get("layer").is_some() && t.get("projection").is_some()));
    }
}
