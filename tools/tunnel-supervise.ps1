<#
.SYNOPSIS
  Keep an SSH tunnel to the image box alive, reconnecting when it drops.

.DESCRIPTION
  A single `ssh -N -L` dies on a network blip, a box reboot, or an instance port
  change, and the app then sees ECONNREFUSED with no obvious cause. This wraps it
  in a supervised loop, logs every reconnect, and exits loudly only when the box
  is genuinely unreachable for several attempts in a row.

.EXAMPLE
  .\tunnel-supervise.ps1 -Box 93.91.156.106 -Port 46075
  .\tunnel-supervise.ps1 -Box 93.91.156.106 -Port 46075 -LocalPort 18080 -Log ..\..\scratch\tunnel.log
#>
param(
  [Parameter(Mandatory = $true)][string]$Box,
  [Parameter(Mandatory = $true)][int]$Port,
  [int]$LocalPort = 18080,
  [int]$RemotePort = 10100,
  [string]$Key = "$env:USERPROFILE\.ssh\id_ed25519",
  [string]$Log = "$env:TEMP\qwen-tunnel.log",
  [int]$MaxConsecutiveFailures = 5
)

$ErrorActionPreference = 'Continue'

function Write-Log([string]$Message) {
  $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $Message
  Write-Host $line
  Add-Content -Path $Log -Value $line -ErrorAction SilentlyContinue
}

Write-Log "supervising 127.0.0.1:$LocalPort -> ${Box}:$RemotePort (ssh port $Port)"
Write-Log "log: $Log"

$failures = 0
$attempt = 0
while ($true) {
  $attempt++
  Write-Log "connecting (attempt $attempt)"

  $args = @(
    '-N', '-T',
    '-o', 'ExitOnForwardFailure=yes',
    '-o', 'ServerAliveInterval=15',
    '-o', 'ServerAliveCountMax=3',
    '-o', 'TCPKeepAlive=yes',
    '-o', 'StrictHostKeyChecking=no',
    '-o', 'UserKnownHostsFile=NUL',
    '-o', 'ConnectTimeout=15',
    '-L', "${LocalPort}:127.0.0.1:$RemotePort",
    '-i', $Key,
    '-p', $Port,
    "root@$Box"
  )

  $started = Get-Date
  & ssh @args
  $code = $LASTEXITCODE
  $lived = ((Get-Date) - $started).TotalSeconds

  # a tunnel that stayed up for a while means the drop was transient
  if ($lived -gt 30) { $failures = 0 }
  else { $failures++ }

  Write-Log "tunnel exited (code $code) after $([math]::Round($lived))s"

  if ($failures -ge $MaxConsecutiveFailures) {
    Write-Log "FAIL: $failures consecutive short-lived attempts. Is the box still running, and is port $Port still correct?"
    exit 1
  }

  $backoff = [math]::Min(5 * $failures, 30)
  Write-Log "reconnecting in ${backoff}s"
  Start-Sleep -Seconds $backoff
}
