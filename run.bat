@echo off
set /p python_path=Please enter the full path to your Python interpreter:  
echo Running Python script...
"%python_path%" 2_predict.py
pause