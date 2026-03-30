use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const MANIFEST_VERSION: u32 = 1;
const MANIFEST_SIDECAR_SUFFIX: &str = ".manifest.json";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelProvenance {
    pub version: u32,
    pub source: ProvenanceSource,
    pub identity: ProvenanceIdentity,
    #[serde(default)]
    pub compatibility: ProvenanceCompatibility,
    #[serde(default)]
    pub local: ProvenanceLocal,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProvenanceSource {
    pub provider: ProvenanceProvider,
    pub repo: Option<String>,
    pub revision: Option<String>,
    pub file: Option<String>,
    pub resolved_url: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProvenanceIdentity {
    pub canonical_id: String,
    pub display_name: String,
    pub family: Option<String>,
    pub architecture: Option<String>,
    pub format: String,
    pub quantization: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProvenanceCompatibility {
    pub tokenizer_hash: Option<String>,
    pub chat_template_hash: Option<String>,
    pub base_model: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProvenanceLocal {
    pub downloaded_at: Option<String>,
    pub sha256: Option<String>,
    pub size_bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceProvider {
    HuggingFace,
    DirectUrl,
    Local,
    #[default]
    Unknown,
}

pub fn model_manifest_path(path: &Path) -> PathBuf {
    let filename = path
        .file_name()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_string());
    path.with_file_name(format!("{filename}{MANIFEST_SIDECAR_SUFFIX}"))
}

pub fn load_model_provenance(path: &Path) -> Option<ModelProvenance> {
    let sidecar = model_manifest_path(path);
    let raw = std::fs::read_to_string(sidecar).ok()?;
    let provenance: ModelProvenance = serde_json::from_str(&raw).ok()?;
    if provenance.version == MANIFEST_VERSION {
        Some(provenance)
    } else {
        None
    }
}

pub fn to_proto_manifest(
    route_model: &str,
    provenance: &ModelProvenance,
) -> crate::proto::node::ModelManifest {
    crate::proto::node::ModelManifest {
        version: provenance.version,
        route_model: route_model.to_string(),
        canonical_id: provenance.identity.canonical_id.clone(),
        display_name: if provenance.identity.display_name.is_empty() {
            None
        } else {
            Some(provenance.identity.display_name.clone())
        },
        family: provenance.identity.family.clone(),
        architecture: provenance.identity.architecture.clone(),
        format: if provenance.identity.format.is_empty() {
            None
        } else {
            Some(provenance.identity.format.clone())
        },
        quantization: provenance.identity.quantization.clone(),
        source: Some(crate::proto::node::ModelManifestSource {
            provider: match provenance.source.provider {
                ProvenanceProvider::HuggingFace => {
                    crate::proto::node::ManifestProvider::HuggingFace as i32
                }
                ProvenanceProvider::DirectUrl => {
                    crate::proto::node::ManifestProvider::DirectUrl as i32
                }
                ProvenanceProvider::Local => crate::proto::node::ManifestProvider::Local as i32,
                ProvenanceProvider::Unknown => crate::proto::node::ManifestProvider::Unknown as i32,
            },
            repo: provenance.source.repo.clone(),
            revision: provenance.source.revision.clone(),
            file: provenance.source.file.clone(),
        }),
        compatibility: Some(crate::proto::node::ModelManifestCompatibility {
            tokenizer_hash: provenance.compatibility.tokenizer_hash.clone(),
            chat_template_hash: provenance.compatibility.chat_template_hash.clone(),
            base_model: provenance.compatibility.base_model.clone(),
        }),
    }
}
