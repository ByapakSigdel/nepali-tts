# Bundle docs/paper.tex + the current figures into the NEXT versioned Overleaf zip.
# Never overwrites an existing version — auto-increments paper_overleaf_vN.zip.
# Run from anywhere:  pwsh -File scripts\bundle_paper.ps1
$docs  = Join-Path $PSScriptRoot "..\docs"
# paper.tex + EVERY figure in docs/ (so new figures are always bundled for Overleaf).
$files = @((Join-Path $docs "paper.tex")) +
         (Get-ChildItem (Join-Path $docs "*.png") | ForEach-Object { $_.FullName })

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
