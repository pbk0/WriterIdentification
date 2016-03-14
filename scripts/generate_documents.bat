rem rmdir /S /Q html
echo generating documentation
cd ..\documentation
D:\InstalledPrograms\doxygen\bin\doxygen ..\documentation\Doxyfile
cd ..\scripts
echo documentation completed :)
echo committing to repo
rem cd E:\GitHub\praveenneuron.github.io\writer_identification_doc
rem git add -A
rem git commit "auto commit"
rem git push
rem cd E:\GitHub\WriterIdentification\documentation
