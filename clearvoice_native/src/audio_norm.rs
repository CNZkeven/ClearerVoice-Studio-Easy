use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

const EPS: f64 = 1e-6;

/// Two-stage RMS normalization of audio signal.
/// Returns (normalized_audio, inverse_scalar).
#[pyfunction]
pub fn audio_norm<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    let x_slice = x.as_slice()?;
    let len = x_slice.len() as f64;

    // Stage 1: RMS normalization to -25 dB
    let sum_sq: f64 = x_slice.par_iter().map(|&v| v * v).sum();
    let rms = (sum_sq / len).sqrt();
    let scalar = 10.0_f64.powf(-25.0 / 20.0) / (rms + EPS);

    let scaled: Vec<f64> = x_slice.par_iter().map(|&v| v * scalar).collect();

    // Stage 2: RMS of above-average-power segments
    let pow_x: Vec<f64> = scaled.par_iter().map(|&v| v * v).collect();
    let avg_pow: f64 = pow_x.par_iter().sum::<f64>() / len;

    let (high_pow_sum, high_pow_count) = pow_x
        .par_iter()
        .filter(|&&p| p > avg_pow)
        .fold(
            || (0.0_f64, 0usize),
            |(sum, count), &p| (sum + p, count + 1),
        )
        .reduce(
            || (0.0, 0),
            |(s1, c1), (s2, c2)| (s1 + s2, c1 + c2),
        );

    let rmsx = if high_pow_count > 0 {
        (high_pow_sum / high_pow_count as f64).sqrt()
    } else {
        EPS
    };
    let scalarx = 10.0_f64.powf(-25.0 / 20.0) / (rmsx + EPS);

    let result: Vec<f64> = scaled.par_iter().map(|&v| v * scalarx).collect();
    let inverse_scalar = 1.0 / (scalar * scalarx + EPS);

    let result_array = result.into_pyarray_bound(py);
    Ok((result_array, inverse_scalar))
}
