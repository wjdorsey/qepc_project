cd C:\Users\wdors\qepc_project

Get-ChildItem -Recurse -Filter *.ipynb | ForEach-Object {
    python -m jupyter nbconvert --clear-output --inplace $_.FullName
}
