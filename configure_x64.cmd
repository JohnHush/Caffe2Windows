@echo off
@setlocal EnableDelayedExpansion

set MSVC_VERSION=14

:: Setup the environement for VS x64
set batch_file=!VS%MSVC_VERSION%0COMNTOOLS!..\..\VC\vcvarsall.bat

echo INFO: batchfile     = !batch_file!

call "%batch_file%" amd64

@endlocal
