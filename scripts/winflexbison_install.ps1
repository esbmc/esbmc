$success = $false
while (-not $success) {
    Write-Host "Installing winflexbison..."
    $output = choco install -y winflexbison --ignore-checksums
    if ($LASTEXITCODE -eq 0) {
        $success = $true
        Write-Host "Installation successful!"
    } else {
        Write-Host "Installation failed. Retrying in 10 seconds..."
        Start-Sleep -Seconds 10
    }
}
