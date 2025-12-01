cd C:\Users\wdors\qepc_project

Get-ChildItem -Path .\notebooks -Filter *.ipynb -Recurse | ForEach-Object {
    jupyter nbconvert --clear-output --inplace $_.FullName
}
