# Converts report\report.docx to report\report.pdf with Microsoft Word.
#   powershell -ExecutionPolicy Bypass -File report\export_pdf.ps1
# If Word hangs (e.g. waiting on a first-run dialog), open report.docx in Word and use
# File > Save As > PDF instead.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$docx = Join-Path $here "report.docx"
$pdf  = Join-Path $here "report.pdf"
$word = New-Object -ComObject Word.Application
$word.Visible = $true
$word.DisplayAlerts = 0        # wdAlertsNone
try {
    # Open(FileName, ConfirmConversions, ReadOnly, AddToRecentFiles)
    $doc = $word.Documents.Open($docx, $false, $true, $false)
    $doc.SaveAs2($pdf, 17)     # wdFormatPDF
    $doc.Close($false)
    Write-Output "saved $pdf"
} finally {
    if ($word.Documents.Count -eq 0) { $word.Quit() }   # never close Word if the user has other documents open
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
}
