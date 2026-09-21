# Menjalankan backend (FastAPI) + membuka frontend sekaligus.
$root = $PSScriptRoot
$backendDir = Join-Path $root "kukangsort-backend\backend"
$venvPython = Join-Path $backendDir "venv\Scripts\python.exe"
$html = Join-Path $root "kukangsort-ai.html"
$frontendPort = 5500

Write-Host "Menjalankan backend di background..."
Start-Process -FilePath $venvPython `
    -ArgumentList "-m uvicorn main:app --port 8000" `
    -WorkingDirectory $backendDir `
    -WindowStyle Normal

Write-Host "Menjalankan server frontend di background..."
Start-Process -FilePath $venvPython `
    -ArgumentList "-m http.server $frontendPort" `
    -WorkingDirectory $root `
    -WindowStyle Hidden

Write-Host "Menunggu backend siap..."
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 2 -UseBasicParsing
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch {}
}

if ($ready) {
    Write-Host "Backend siap. Membuka frontend..."
} else {
    Write-Host "Backend belum merespons setelah 30 detik, tetap membuka frontend (cek jendela backend untuk error)."
}

Start-Process "http://localhost:$frontendPort/kukangsort-ai.html"
