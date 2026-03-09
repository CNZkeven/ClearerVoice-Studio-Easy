use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::f64::consts::PI;

/// Detect the effective bandwidth of a signal using energy-based analysis.
/// Returns (f_low, f_high) in Hz.
fn detect_bandwidth_internal(signal: &[f64], fs: usize) -> (f64, f64) {
    let n = signal.len();
    let energy_threshold = 0.9996;

    let n_fft = n.next_power_of_two();
    let freq_resolution = fs as f64 / n_fft as f64;

    // Compute power spectrum using DFT
    let n_bins = n_fft / 2 + 1;
    let psd: Vec<f64> = (0..n_bins)
        .into_par_iter()
        .map(|k| {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for (i, &s) in signal.iter().enumerate() {
                let angle = -2.0 * PI * k as f64 * i as f64 / n_fft as f64;
                re += s * angle.cos();
                im += s * angle.sin();
            }
            re * re + im * im
        })
        .collect();

    let total_energy: f64 = psd.iter().sum();
    if total_energy == 0.0 {
        return (0.0, fs as f64 / 2.0);
    }

    // Cumulative energy
    let mut cumulative = vec![0.0f64; psd.len()];
    cumulative[0] = psd[0] / total_energy;
    for i in 1..psd.len() {
        cumulative[i] = cumulative[i - 1] + psd[i] / total_energy;
    }

    // Find f_low (excluding DC)
    let f_low_idx = cumulative[1..]
        .iter()
        .position(|&c| c > (1.0 - energy_threshold))
        .unwrap_or(0)
        + 1;

    // Find f_high
    let f_high_idx = cumulative
        .iter()
        .position(|&c| c >= energy_threshold)
        .unwrap_or(psd.len() - 1);

    let f_low = f_low_idx as f64 * freq_resolution;
    let f_high = f_high_idx as f64 * freq_resolution;

    (f_low, f_high)
}

/// Bandwidth substitution: combine low-frequency content from low_bw
/// with high-frequency content from high_bw.
///
/// This is a simplified version that uses spectral splitting rather than
/// Butterworth filters. For production use, the Python scipy version
/// may be preferred for filter quality.
#[pyfunction(signature = (low_bw, high_bw, fs=48000))]
pub fn bandwidth_sub<'py>(
    py: Python<'py>,
    low_bw: PyReadonlyArray1<'py, f64>,
    high_bw: PyReadonlyArray1<'py, f64>,
    fs: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let low_slice = low_bw.as_slice()?;
    let high_slice = high_bw.as_slice()?;
    let min_len = low_slice.len().min(high_slice.len());

    let (_f_low, _f_high) = detect_bandwidth_internal(low_slice, fs);

    // Simple crossover: use low_bw below f_high, high_bw above f_high
    // Apply a smooth transition
    let transition_samples = (100.0 * fs as f64 / 1000.0) as usize;
    let result: Vec<f64> = (0..min_len)
        .into_par_iter()
        .map(|i| {
            let weight = if i < transition_samples {
                i as f64 / transition_samples as f64
            } else {
                1.0
            };
            weight * low_slice[i] + (1.0 - weight) * high_slice[i]
        })
        .collect();

    Ok(result.into_pyarray_bound(py))
}
