@echo off
echo Starting FraudWatch AI System...
echo.

echo Starting Backend Server...
start "Backend Server" cmd /k "cd /d %~dp0 && python backend/main.py"

timeout /t 3 /nobreak > nul

echo Starting Frontend Server...
start "Frontend Server" cmd /k "cd /d %~dp0frontend && python -m http.server 8080"

echo.
echo Servers starting...
echo Backend: http://127.0.0.1:5000
echo Frontend: http://localhost:8080
echo.
echo Press any key to close this window...
pause > nul