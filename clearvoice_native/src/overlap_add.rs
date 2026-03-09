use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Assembles decoded audio segments using overlap-add with give-up-length.
///
/// Args:
///   segments: 2D array of shape (num_segments, window_len) - decoded segments
///   window: window length in samples
///   stride: stride between windows
///   total_len: total output length
///
/// Returns:
///   1D array of assembled audio
#[pyfunction]
pub fn overlap_add_segments<'py>(
    py: Python<'py>,
    segments: PyReadonlyArray2<'py, f64>,
    window: usize,
    stride: usize,
    total_len: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let segments = segments.as_array();
    let give_up_length = (window - stride) / 2;

    let mut output = vec![0.0f64; total_len];

    for (i, segment) in segments.outer_iter().enumerate() {
        let current_idx = i * stride;
        let seg_slice = segment.as_slice().unwrap();

        if i == 0 {
            // First segment: use everything except the trailing give_up_length
            let end = (current_idx + window - give_up_length).min(total_len);
            let copy_len = end - current_idx;
            output[current_idx..end].copy_from_slice(&seg_slice[..copy_len]);
        } else {
            // Subsequent segments: skip give_up_length on both sides
            let start = current_idx + give_up_length;
            let end = (current_idx + window - give_up_length).min(total_len);
            if start < end && end <= total_len {
                let src_start = give_up_length;
                let src_end = src_start + (end - start);
                output[start..end].copy_from_slice(&seg_slice[src_start..src_end]);
            }
        }
    }

    Ok(output.into_pyarray_bound(py))
}
