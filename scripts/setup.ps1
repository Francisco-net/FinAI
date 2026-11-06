# Instala tudo que o FinAI precisa (Windows): uv, Python 3.13 e todas as dependencias
# nas versoes exatas travadas no uv.lock. Pode ser executado mais de uma vez com seguranca.
#
# Uso (PowerShell, dentro da pasta do projeto):
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "==> Instalando o uv (gerenciador de projetos Python)..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:USERPROFILE\.cargo\bin;$env:Path"
}

Write-Host "==> Instalando o Python e as dependencias (versoes exatas do uv.lock)..."
uv sync --all-extras --frozen
if ($LASTEXITCODE -ne 0) { throw "Falha ao instalar as dependencias." }

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "==> Arquivo .env criado. Preencha ANTHROPIC_API_KEY para usar o chat com IA."
}

if (Test-Path ".git") {
    uv run pre-commit install
}

Write-Host "==> Verificando a instalacao (testes automatizados)..."
uv run pytest
if ($LASTEXITCODE -ne 0) { throw "Os testes falharam. Veja as mensagens acima." }

Write-Host ""
Write-Host "Instalacao concluida!"
Write-Host ""
Write-Host "  Interface web:      uv run streamlit run app/streamlit_app.py   (http://localhost:8501)"
Write-Host "  Chat no terminal:   uv run finai chat"
Write-Host "  API REST:           uv run uvicorn finai.api.main:app --reload  (http://localhost:8000/docs)"
Write-Host "  Sem chave de API:   uv run finai carteira examples/perfil_exemplo.json"
Write-Host ""
Write-Host "Para o chat com IA, preencha ANTHROPIC_API_KEY no arquivo .env."
