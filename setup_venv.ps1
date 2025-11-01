# Fix virtual environment and install requirements
Write-Host "Removing old virtual environment..." -ForegroundColor Yellow
Remove-Item -Recurse -Force venv -ErrorAction SilentlyContinue

Write-Host "Creating new virtual environment..." -ForegroundColor Green
python -m venv venv

Write-Host "Activating virtual environment..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip

Write-Host "Installing requirements..." -ForegroundColor Green
pip install -r requirements.txt

Write-Host "`nSetup complete! You can now run the app with:" -ForegroundColor Cyan
Write-Host "  streamlit run app.py" -ForegroundColor White

