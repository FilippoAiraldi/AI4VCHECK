@echo off

if "%~1"=="" (
    echo ERROR: No directory provided. Please specify a directory to search for images.
    exit /b 1
)

set masks=1 2
set rotations=90
::,90,180,270

for /D %%d in (%~1\*) do (
    for %%f in ("%%d\cornea.png") do (
		if exist "%%f" (
			for %%i in (%rotations%) do (
				echo Segmenting "%%~ff" for masks "%masks%" and with rotation %%i
                python segment.py %%~ff -y --rotate=%%i --save-masks %masks%
			)
		)
    )
)
