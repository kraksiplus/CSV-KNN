# Script to run main.py 10 times consecutively

for ($i = 1; $i -le 100; $i++) {
    Write-Output "Running iteration $i"
    python main.py
}