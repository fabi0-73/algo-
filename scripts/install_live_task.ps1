# Registers the persistent live-scanner Scheduled Task (idempotent).
#
#   install:   powershell -ExecutionPolicy Bypass -File scripts\install_live_task.ps1
#   start:     Start-ScheduledTask AMD_Live_Scanner
#   stop:      Stop-ScheduledTask AMD_Live_Scanner   (kills the supervisor;
#              also Stop-Process the python child if one is mid-scan)
#   status:    Get-ScheduledTask AMD_Live_Scanner | Get-ScheduledTaskInfo
#   uninstall: Unregister-ScheduledTask AMD_Live_Scanner -Confirm:$false
#
# Trigger is AT LOG ON (interactive token): MT5 is a GUI terminal and needs
# the user session, so the scanner runs while you are logged in and resumes
# at next logon after a reboot. Log: logs\live_scanner.log in the repo.

$name = "AMD_Live_Scanner"
$repo = "C:\Users\FT\Documents\FT\Market"
$bat = Join-Path $repo "scripts\run_live_forever.bat"

if (-not (Test-Path $bat)) { throw "missing $bat" }

try {
    Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction Stop
    Write-Host "replaced existing task '$name'"
} catch {}

$action = New-ScheduledTaskAction -Execute $bat -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable

Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger `
    -Settings $settings `
    -Description "AMD XAUUSD live signal scanner (signal-only -> Telegram); supervised loop, restarts on crash" | Out-Null

Write-Host "task '$name' registered (runs at logon; start now with: Start-ScheduledTask $name)"
