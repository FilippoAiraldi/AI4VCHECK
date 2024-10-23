@echo off

if "%~1"=="" (
    echo ERROR: No directory provided. Please specify a directory to search for images.
    exit /b 1
)

setlocal EnableDelayedExpansion
for /D %%d in (%~1\*) do (
    for %%f in ("%%d\masks\cornea-mask-1-at-0.0-manually-segmented.png") do (
        set pred="%%d\masks\cornea-mask-1-at-0.0.png"
		if exist "%%f"  (
            if exist !pred! (
                echo Comparing "%%~f" with !pred!
                python compare.py %%~ff !pred!
            )
		)
    )
)
endlocal
