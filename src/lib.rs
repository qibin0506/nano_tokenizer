use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use std::path::Path;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

#[pyclass]
pub struct NanoTokenizer {
    tokenizer: Tokenizer,
}

#[pymethods]
impl NanoTokenizer {
    #[new]
    fn new(dir_path: &str) -> PyResult<Self> {
        let file_path = Path::new(dir_path).join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&file_path).map_err(|e| {
            PyIOError::new_err(format!("Failed to load tokenizer.json: {}", e))
        })?;
        Ok(NanoTokenizer { tokenizer })
    }

    fn get_vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    fn convert_tokens_to_ids(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    fn encode(&self, text: &str) -> PyResult<Vec<u32>> {
        let encoding = self.tokenizer
            .encode(text, false)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        self.tokenizer
            .decode(&ids, false)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn batch_decode(&self, batch_ids: Vec<Vec<u32>>) -> PyResult<Vec<String>> {
        let mut results = Vec::with_capacity(batch_ids.len());
        for ids in batch_ids {
            let text = self.tokenizer
                .decode(&ids, false)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            results.push(text);
        }
        Ok(results)
    }

    fn batch_encode(
        &self,
        texts: Vec<String>,
        padding: bool,
        truncation: bool,
        pad_id: u32,
        pad_token: &str,
    ) -> PyResult<Vec<Vec<u32>>> {
        let mut tokenizer = self.tokenizer.clone();

        if padding {
            let mut params = PaddingParams::default();
            params.pad_id = pad_id;
            params.pad_token = pad_token.to_string();
            tokenizer.with_padding(Some(params));
        }

        if truncation {
            tokenizer.with_truncation(Some(TruncationParams::default()));
        }

        let encodings = tokenizer
            .encode_batch(texts, false)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let mut input_ids = Vec::with_capacity(encodings.len());

        for encoding in encodings {
            input_ids.push(encoding.get_ids().to_vec());
        }

        Ok(input_ids)
    }
}

#[pymodule]
fn nano_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NanoTokenizer>()?;
    Ok(())
}