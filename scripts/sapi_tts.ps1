param(
    [Parameter(Mandatory = $true)]
    [string]$TextFile,

    [Parameter(Mandatory = $true)]
    [string]$OutputFile,

    [string]$Voice,

    [int]$Rate = 0,

    [int]$Volume = 100
)

Add-Type -AssemblyName System.Speech

$text = Get-Content -LiteralPath $TextFile -Raw -Encoding UTF8
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

try {
    if ($Voice) {
        $synth.SelectVoice($Voice)
    }

    $synth.Rate = $Rate
    $synth.Volume = $Volume
    $synth.SetOutputToWaveFile($OutputFile)
    $synth.Speak($text)
}
finally {
    $synth.Dispose()
}

