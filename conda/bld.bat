:: Build FastAtoms into the conda prefix: the C++ core, the Python extension,
:: the headers and the CMake package files, in one configure/build/install pass.
::
:: The Windows counterpart of conda/build.sh. The differences are conda's own:
:: C and C++ artifacts live under %LIBRARY_PREFIX% (that is %PREFIX%\Library),
:: while the Python package still belongs in %SP_DIR%.
setlocal EnableDelayedExpansion

:: CMake wants forward slashes even on Windows; backslashes in a -D value are
:: read as escapes and quietly mangle the path.
set "PREFIX_FWD=%LIBRARY_PREFIX:\=/%"
set "SP_DIR_FWD=%SP_DIR:\=/%"

cmake -S "%SRC_DIR%" -B build ^
    -G "Ninja" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_INSTALL_PREFIX="%PREFIX_FWD%" ^
    -DCMAKE_PREFIX_PATH="%PREFIX_FWD%" ^
    -DPython3_EXECUTABLE="%PYTHON%" ^
    -DPython3_FIND_STRATEGY=LOCATION ^
    -DFASTATOMS_PYTHON_INSTALL_DIR="%SP_DIR_FWD%/fastatoms" ^
    :: Windows has no rpath - DLLs are found on PATH - so the equivalent of
    :: build.sh's CMAKE_INSTALL_RPATH is simply not needed here.
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON ^
    -DFASTATOMS_INSTALL=ON ^
    -DFASTATOMS_BUILD_PYTHON=ON ^
    -DFASTATOMS_BUILD_EXAMPLES=ON ^
    -DFASTATOMS_BUILD_TESTS=ON ^
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON ^
    -DFASTATOMS_WITH_TORCH=OFF ^
    %CMAKE_ARGS%
if errorlevel 1 exit 1

cmake --build build --parallel %CPU_COUNT%
if errorlevel 1 exit 1

:: The C++ suites, before anything is packaged. The Python suite runs in the
:: test: section against the installed package.
ctest --test-dir build --output-on-failure -E "python"
if errorlevel 1 exit 1

cmake --install build
if errorlevel 1 exit 1
