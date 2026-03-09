# Design: Bug Fixes + Rust Native Extension

Date: 2026-03-09

## Phase 1: Bug Fixes

### Critical Bugs

1. **`dataloader/misc.py:132`** — `self.print` in standalone function `reload_for_eval()`.
   Fix: Replace with a `print_flag` parameter (default `False`).

2. **`dataloader/dataloader.py:287`** — `Wave_Processor.process()` calls `audioread(path, sampling_rate)` with 2 args, but current signature requires 3 (`use_norm`).
   Fix: Add `use_norm=False` argument.

3. **`__init__.py:53-55,72`** — Mixed tabs/spaces indentation causes `IndentationError`.
   Fix: Normalize to 4-space indentation.

4. **`decode_batch.py:411,414`** — `outputs[batch_idx, batch_idx, ...]` double-indexes with `batch_idx` on a 2D `(b,t)` array.
   Fix: Change to `outputs[batch_idx, ...]`.

5. **`decode_batch.py:416`** — `current_idx += stride` inside the `for batch_idx` loop, advances per batch element.
   Fix: Move outside the inner loop.

6. **`app.py:69`** — `write_audio()` called without `self.data` being populated.
   Fix: Use `soundfile.write()` directly in `ModelManager.process_audio()`.

### Medium Bugs

7. **`__init__.py:85`** — `break` exits first `for model` loop after one iteration.
   Fix: Restructure to check all models properly.

8. **`video_process.py`** — Shell injection via `subprocess.call(shell=True)` + uses `rm` (Windows-incompatible).
   Fix: Use `subprocess.run(list_args)` + `os.remove()`.

9. **Multiple files** — Bare `except:` clauses silently swallow errors.
   Fix: Replace with `except Exception as e:` with logging.

## Phase 2: Rust Native Extension (PyO3)

### Structure

```
clearvoice_native/
├── Cargo.toml          # ndarray, numpy (PyO3), rayon
├── pyproject.toml      # maturin build config
├── src/
│   ├── lib.rs          # PyO3 module entry
│   ├── audio_norm.rs   # Two-stage RMS normalization
│   ├── overlap_add.rs  # Segment assembly with overlap-add
│   └── bandwidth.rs    # Bandwidth detection + substitution
```

### Functions

- `audio_norm(x) -> (ndarray, float)`: Rayon-parallel two-stage RMS normalization
- `overlap_add_segments(segments, window, stride, total_len) -> ndarray`: Decoded segment assembly
- `bandwidth_sub(low_bw, high_bw, fs) -> ndarray`: Bandwidth detection, filtering, substitution

### Integration

- Graceful fallback: `try: from clearvoice_native import ...` with fallback to numpy/scipy
- Zero behavior change — Rust produces identical outputs
- Build via `maturin develop` into conda env `F:\anac\envs\Common`
