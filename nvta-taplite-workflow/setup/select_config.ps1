Add-Type -AssemblyName System.Windows.Forms

$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.InitialDirectory = (Resolve-Path "configs").Path
$dialog.Filter = "JSON config files (*.json)|*.json|All files (*.*)|*.*"
$dialog.Title = "Select DTALite pipeline config file"

if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    $dialog.FileName
}
