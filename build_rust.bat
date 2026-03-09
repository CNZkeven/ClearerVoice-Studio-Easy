@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>nul
cd /d D:\test\ClearerVoice-Studio-Easy\clearvoice_native
cargo clean >nul 2>nul
F:\anac\envs\Common\python.exe -m maturin build --release --interpreter F:\anac\envs\Common\python.exe > D:\test\ClearerVoice-Studio-Easy\build_log.txt 2>&1
echo DONE >> D:\test\ClearerVoice-Studio-Easy\build_log.txt
