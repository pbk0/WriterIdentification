rmdir /S /Q html
echo generating documentation
D:\InstalledPrograms\doxygen\bin\doxygen Doxyfile
echo documentation completed :)
echo committing to repo
cd E:\GitHub\praveenneuron.github.io\writer_identification_doc
git add -A
git commit "auto commit"
git push
