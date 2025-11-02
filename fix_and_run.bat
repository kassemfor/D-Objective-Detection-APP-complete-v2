@echo off
echo Removing old virtual environment...
rmdir /s /q venv 2>nul

echo Creating new virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing requirements...
pip install -r requirements.txt

echo.
echo ========================================
echo Setup complete! Starting the app...
echo ========================================
echo.
streamlit run app.py

pause



