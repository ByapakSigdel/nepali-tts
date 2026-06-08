# Bundle docs/paper.tex + the current figures into the NEXT versioned Overleaf zip.
# Never overwrites an existing version — auto-increments paper_overleaf_vN.zip.
# Run from anywhere:  pwsh -File scripts\bundle_paper.ps1
$docs  = Join-Path $PSScriptRoot "..\docs"
$files = @(
    (Join-Path $docs "paper.tex"),
    (Join-Path $docs "training_losses.png"),
    (Join-Path $docs "headline_loss.png"),
    (Join-Path $docs "dataset_speakers.png")
)

$existing = Get-ChildItem (Join-Path $docs "paper_overleaf_v*.zip") -ErrorAction SilentlyContinue
$next = 1
if ($existing) {
    $maxv = ($existing | ForEach-Object {
        if ($_.BaseName -match 'v(\d+)$') { [int]$Matches[1] } else { 0 }
    } | Measure-Object -Maximum).Maximum
    $next = $maxv + 1
}

$zip = Join-Path $docs "paper_overleaf_v$next.zip"
Compress-Archive -Path $files -DestinationPath $zip
Write-Output "Created paper_overleaf_v$next.zip  (previous versions kept)."
Get-ChildItem (Join-Path $docs "paper_overleaf_v*.zip") | Select-Object Name, LastWriteTime
