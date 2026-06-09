$ErrorActionPreference = "Stop"

# Resolve CUDA installation from environment or nvcc on PATH.
$cudaHome = $env:CUDA_PATH
if (-not $cudaHome) {
    $cudaHome = $env:CUDA_HOME
}
if (-not $cudaHome) {
    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($nvcc) {
        $cudaHome = Split-Path (Split-Path $nvcc.Source -Parent) -Parent
    }
}

$nvccPath = Join-Path $cudaHome "bin\nvcc.exe"
if (-not $cudaHome -or -not (Test-Path $nvccPath)) {
    throw "CUDA Toolkit not found. Set CUDA_PATH/CUDA_HOME or add nvcc to PATH."
}

$env:CUDA_HOME = $cudaHome
$cudaBin = Join-Path $cudaHome "bin"
if (-not (($env:Path -split ";") -contains $cudaBin)) {
    $env:Path = "$cudaBin;$env:Path"
}

$buildDir = Join-Path $PSScriptRoot "build"

if (Test-Path $buildDir) {
    Remove-Item -Path $buildDir -Recurse -Force
}
New-Item -ItemType Directory -Path $buildDir | Out-Null

Push-Location $buildDir
try {
    cmake -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_COMPILER=$nvccPath" ..
    cmake --build . --config Release --parallel ([Environment]::ProcessorCount)

    # The Python extension is usually a .pyd on Windows.
    $artifact = Get-ChildItem -Path . -Recurse -File -Include "sph_cuda*.pyd", "sph_cuda*.dll" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $artifact) {
        throw "Build succeeded but no sph_cuda artifact was found in the build output."
    }

    Copy-Item -Path $artifact.FullName -Destination (Join-Path $PSScriptRoot $artifact.Name) -Force
}
finally {
    Pop-Location
    if (Test-Path $buildDir) {
        Remove-Item -Path $buildDir -Recurse -Force
    }
}