[CmdletBinding()]
param(
    [Parameter(Position = 0, Mandatory = $true)]
    [ValidateSet("ablation", "validation", "scalability", "comparison", "all")]
    [string]$Experiment,

    [switch]$Clean,
    [switch]$Resume,
    [switch]$PlotsOnly,
    [switch]$SkipWarmup
)

$ErrorActionPreference = "Stop"
$UvRecommendedVersion = "0.11.29"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir ".."))
$ResultsRoot = Join-Path $RepoRoot "results_replication"
$RunId = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")
$LogDir = Join-Path $ResultsRoot (Join-Path "reproduction" $RunId)
$LogFile = Join-Path $LogDir "execution.log"
$ManifestFile = Join-Path $LogDir "environment.json"
$SummaryFile = Join-Path $LogDir "summary.json"
$MainPackagesFile = Join-Path $LogDir "main_packages.txt"
$GprPackagesFile = Join-Path $LogDir "gpr_packages.txt"
$StartTime = [DateTime]::UtcNow
$Mode = if ($Clean) { "clean" } else { "resume" }
$ModeExplicit = $Clean -or $Resume
$UvCommand = $null
$CurrentStage = "initialization"
$CompletedStages = [System.Collections.Generic.List[string]]::new()

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Log {
    param([Parameter(Mandatory = $true)][string]$Message)
    $line = "[{0}] {1}" -f [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ"), $Message
    $line | Tee-Object -FilePath $LogFile -Append
}

function Invoke-LoggedCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $previousErrorActionPreference = $ErrorActionPreference

    try {
        $ErrorActionPreference = "Continue"

        & $Command @Arguments 2>&1 |
            ForEach-Object {
                if (
                    $_ -is
                    [System.Management.Automation.ErrorRecord]
                ) {
                    $line = $_.Exception.Message
                }
                else {
                    $line = $_.ToString()
                }

                Write-Host $line

                Add-Content -LiteralPath $LogFile -Value $line -Encoding UTF8
            }

        # Capture the native exit code immediately after the pipeline.
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference =  $previousErrorActionPreference
    }

    if ($exitCode -ne 0) {
        throw (
            "Command failed with exit code ${exitCode}: " +
            "$Command $($Arguments -join ' ')"
        )
    }
}

function Update-ProcessPath {
    # The current PowerShell process may have been started before uv updated
    # the user PATH. Merge the current, user-level, and machine-level PATH
    # values without modifying persistent system settings.

    $candidatePaths = @(
        $env:Path
        [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::User)
        [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::Machine)
        (Join-Path $HOME ".local\bin")
        (Join-Path $HOME ".cargo\bin")
    )

    $uniquePaths = @(
        $candidatePaths |
            Where-Object {
                -not [string]::IsNullOrWhiteSpace($_)
            } |
            ForEach-Object {
                $_ -split [System.IO.Path]::PathSeparator
            } |
            Where-Object {
                -not [string]::IsNullOrWhiteSpace($_)
            } |
            ForEach-Object {
                $_.Trim()
            } |
            Select-Object -Unique
    )

    $env:Path = $uniquePaths -join [System.IO.Path]::PathSeparator
}

function Find-Uv {
    Update-ProcessPath

    $command = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }
    $candidates = @(
        (Join-Path $HOME ".local\bin\uv.exe"),
        (Join-Path $HOME ".cargo\bin\uv.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    return $null
}

function Install-Uv {
    $url = "https://astral.sh/uv/$UvRecommendedVersion/install.ps1"
    Write-Log "uv was not found. Installing recommended uv $UvRecommendedVersion with the official installer."
    $installer = Invoke-RestMethod -Uri $url
    Invoke-Expression $installer

    # The installer may modify the persistent user PATH, but the current
    # process does not receive that modification automatically.
    Update-ProcessPath
    
    $script:UvCommand = Find-Uv
    if ($null -eq $script:UvCommand) {
        throw "uv installation completed, but uv.exe could not be found. Open a new PowerShell session and retry."
    }

    Write-Log (
        "uv installation completed. Executable: " +
        $script:UvCommand
    )
}

function Get-UvVersion {
    $text = & $UvCommand --version
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to determine the uv version."
    }
    return (($text -split '\s+')[1]).Trim()
}

function Sync-Environment {
    param([Parameter(Mandatory = $true)][string]$Project)
    $script:CurrentStage = "sync:$Project"
    Write-Log "Synchronizing $Project from its lockfile."
    try {
        Invoke-LoggedCommand -Command $UvCommand -Arguments @(
            "sync", "--project", (Join-Path $RepoRoot $Project), "--locked"
        )
    }
    catch {
        $detected = Get-UvVersion
        Write-Log "Environment synchronization failed with uv $detected."
        Write-Log "The recommended uv version for these experiments is $UvRecommendedVersion."
        throw
    }
}

function Get-ProjectPythonPath {
    param([Parameter(Mandatory = $true)][string]$Project)
    return Join-Path $RepoRoot (Join-Path $Project ".venv\Scripts\python.exe")
}

function Get-TotalMemoryBytes {
    try {
        return [int64](Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
    }
    catch {
        return $null
    }
}

function Write-EnvironmentManifest {
    $mainPython = (& $UvCommand run --project (Join-Path $RepoRoot "environments\main") --locked python --version 2>&1 | Out-String).Trim()
    $gprPython = (& $UvCommand run --project (Join-Path $RepoRoot "environments\gpr") --locked python --version 2>&1 | Out-String).Trim()
    $manifest = [ordered]@{
        start_time_utc = $StartTime.ToString("o")
        operating_system = [System.Environment]::OSVersion.Platform.ToString()
        operating_system_version = [System.Environment]::OSVersion.VersionString
        architecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
        hostname = [System.Environment]::MachineName
        logical_cpu_count = [System.Environment]::ProcessorCount
        total_memory_bytes = Get-TotalMemoryBytes
        uv_version = (& $UvCommand --version | Out-String).Trim()
        recommended_uv_version = $UvRecommendedVersion
        main_python_version = $mainPython
        gpr_python_version = $gprPython
        selected_experiment = $Experiment
        mode = $Mode
        plots_only = [bool]$PlotsOnly
        skip_warmup = [bool]$SkipWarmup
        results_root = $ResultsRoot
    }
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $ManifestFile -Encoding UTF8

    & $UvCommand pip freeze --python (Get-ProjectPythonPath "environments\main") |
        Set-Content -LiteralPath $MainPackagesFile -Encoding UTF8
    if ($LASTEXITCODE -ne 0) { throw "Could not record main environment packages." }

    & $UvCommand pip freeze --python (Get-ProjectPythonPath "environments\gpr") |
        Set-Content -LiteralPath $GprPackagesFile -Encoding UTF8
    if ($LASTEXITCODE -ne 0) { throw "Could not record GPR environment packages." }
}

function Invoke-SmokeTests {
    $script:CurrentStage = "smoke-test"
    Write-Log "Running import smoke tests."
    Invoke-LoggedCommand -Command $UvCommand -Arguments @(
        "run", "--project", (Join-Path $RepoRoot "environments\main"), "--locked",
        "python", "-c",
        "import numpy, pandas, sklearn, numba, aeon, imodels; from random_fuzzy_rules import RandomFuzzyRulesClassifier; import experiments.utils; print('Main environment smoke test passed.')"
    )
    Invoke-LoggedCommand -Command $UvCommand -Arguments @(
        "run", "--project", (Join-Path $RepoRoot "environments\gpr"), "--locked",
        "python", "-c",
        "import geppy, deap; from gpr_fast import GPR_FAST; print('GPR environment smoke test passed.')"
    )
}

function Remove-SafeReplicationPath {
    param([Parameter(Mandatory = $true)][string]$Target)
    if (-not (Test-Path -LiteralPath $Target)) { return }
    $canonicalRoot = [System.IO.Path]::GetFullPath($ResultsRoot).TrimEnd('\', '/')
    $canonicalTarget = [System.IO.Path]::GetFullPath($Target).TrimEnd('\', '/')
    $prefix = $canonicalRoot + [System.IO.Path]::DirectorySeparatorChar
    if (-not $canonicalTarget.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a path outside results_replication: $canonicalTarget"
    }
    if ($canonicalTarget -eq $canonicalRoot) {
        throw "Refusing to remove the results_replication root."
    }
    Write-Log "Removing replicated output: $canonicalTarget"
    Remove-Item -LiteralPath $canonicalTarget -Recurse -Force
}

function Clear-SelectedOutputs {
    if ($Mode -ne "clean") { return }
    switch ($Experiment) {
        "ablation" { Remove-SafeReplicationPath (Join-Path $ResultsRoot "ablation") }
        "validation" { Remove-SafeReplicationPath (Join-Path $ResultsRoot "ablation\default_configuration_validation") }
        "scalability" { Remove-SafeReplicationPath (Join-Path $ResultsRoot "scalability") }
        "comparison" { Remove-SafeReplicationPath (Join-Path $ResultsRoot "comparison") }
        "all" {
            Remove-SafeReplicationPath (Join-Path $ResultsRoot "ablation")
            Remove-SafeReplicationPath (Join-Path $ResultsRoot "scalability")
            Remove-SafeReplicationPath (Join-Path $ResultsRoot "comparison")
        }
    }
}

function Invoke-ExperimentStage {
    param([Parameter(Mandatory = $true)][string]$Stage)
    $arguments = [System.Collections.Generic.List[string]]::new()
    $arguments.AddRange([string[]]@("run", "--project", (Join-Path $RepoRoot "environments\main"), "--locked", "python", "-m"))
    switch ($Stage) {
        "ablation" {
            $arguments.Add("experiments.ablation.run")
            $arguments.AddRange([string[]]@("--study", "all"))
        }
        "validation" { $arguments.Add("experiments.ablation.validate_default_configuration") }
        "scalability" {
            $arguments.Add("experiments.scalability.run")
            $arguments.AddRange([string[]]@("--study", "all"))
        }
        "comparison" { $arguments.Add("experiments.comparison.run") }
    }
    $arguments.AddRange([string[]]@("--results-root", $ResultsRoot))
    if ($PlotsOnly) { $arguments.Add("--plots-only") }
    if ($SkipWarmup) { $arguments.Add("--skip-warmup") }

    $script:CurrentStage = $Stage
    Write-Log "Starting experiment stage: $Stage"
    Invoke-LoggedCommand -Command $UvCommand -Arguments $arguments.ToArray()
    $CompletedStages.Add($Stage)
    Write-Log "Completed experiment stage: $Stage"
}

function Write-Summary {
    param(
        [Parameter(Mandatory = $true)][string]$Status,
        [Parameter(Mandatory = $true)][int]$ExitCode
    )
    $endTime = [DateTime]::UtcNow
    $summary = [ordered]@{
        status = $Status
        exit_code = $ExitCode
        start_time_utc = $StartTime.ToString("o")
        end_time_utc = $endTime.ToString("o")
        duration_seconds = [math]::Round(($endTime - $StartTime).TotalSeconds, 3)
        selected_experiment = $Experiment
        mode = $Mode
        plots_only = [bool]$PlotsOnly
        skip_warmup = [bool]$SkipWarmup
        current_stage = $CurrentStage
        completed_stages = @($CompletedStages)
        results_root = $ResultsRoot
    }
    $summary | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $SummaryFile -Encoding UTF8
}

$exitCode = 0
try {
    if ($Clean -and $Resume) { throw "--clean and --resume cannot be used together." }
    if ($Clean -and $PlotsOnly) { throw "--clean and --plots-only cannot be used together." }

    Set-Location $RepoRoot
    if (-not $ModeExplicit) {
        Write-Log "Neither --clean nor --resume was specified. Resume mode is used by default."
        Write-Log "Only files under $ResultsRoot are reused. The reference results under $(Join-Path $RepoRoot 'results') are not modified."
    }

    Update-ProcessPath

    $UvCommand = Find-Uv
    if ($null -eq $UvCommand) { Install-Uv }
    $detectedVersion = Get-UvVersion
    Write-Log "Using uv $detectedVersion at $UvCommand"
    if ($detectedVersion -ne $UvRecommendedVersion) {
        Write-Log "WARNING: uv $detectedVersion differs from the recommended version $UvRecommendedVersion. Continuing with the installed version."
    }

    Sync-Environment "environments\main"
    Sync-Environment "environments\gpr"
    Write-EnvironmentManifest
    Invoke-SmokeTests
    Clear-SelectedOutputs

    if ($Experiment -eq "all") {
        foreach ($stage in @("ablation", "validation", "scalability", "comparison")) {
            Invoke-ExperimentStage $stage
        }
    }
    else {
        Invoke-ExperimentStage $Experiment
    }
    $CurrentStage = "completed"
    Write-Summary -Status "success" -ExitCode 0
    Write-Log "Replication completed successfully. Results: $ResultsRoot"
}
catch {
    $exitCode = 1
    Write-Log "ERROR: $($_.Exception.Message)"
    Write-Summary -Status "failed" -ExitCode $exitCode
}
finally {
    Set-Location $RepoRoot
}
exit $exitCode
