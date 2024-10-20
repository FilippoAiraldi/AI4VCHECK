@echo off

if "%~1"=="" (
    echo ERROR: No directory provided. Please specify a directory to search for images.
    exit /b 1
)

set circles=1 2 3 4 5

for /D %%d in (%~1\*) do (
    for %%f in ("%%d\cornea.png") do (
		if exist "%%f" (
            echo Generating circles "%circles%" for "%%~ff"
            python circles.py %%~ff %circles%
		)
    )
)
