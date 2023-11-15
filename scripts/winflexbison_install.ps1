$success = $false
$tries = 0
$maxTries = 20

while (-not $success -and $tries -lt $maxTries) {
    Write-Host "Installing winflexbison..."
    $output = choco install -y winflexbison --ignore-checksums
    if ($LASTEXITCODE -eq 0) {
        $success = $true
        Write-Host "Installation successful!"
    } else {
        Write-Host "Installation failed. Retrying in 10 seconds..."
        Start-Sleep -Seconds 10
        $tries++
    }
}

if (-not $success) {
    Write-Host "Installation failed after $maxTries attempts."
    exit 1
 }
