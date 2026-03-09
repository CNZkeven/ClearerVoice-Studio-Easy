use pyo3::prelude::*;

mod audio_norm;
mod overlap_add;
mod bandwidth;

#[pymodule]
fn clearvoice_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(audio_norm::audio_norm, m)?)?;
    m.add_function(wrap_pyfunction!(overlap_add::overlap_add_segments, m)?)?;
    m.add_function(wrap_pyfunction!(bandwidth::bandwidth_sub, m)?)?;
    Ok(())
}
