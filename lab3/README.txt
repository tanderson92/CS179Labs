[Haru:tanderson]~/lab3>./fft_convolve 512 512
Index of the GPU with the lowest temperature: 0 (52 C)
Time limit for this program set to 120 seconds

N (number of samples per channel):    10000000

Impulse length (number of samples per channel):    2001

CPU convolution...
GPU convolution...
No kernel error detected
Comparing...
GPU time (convolve): 1016.43 milliseconds

Speedup factor (convolution): 21.9361


CPU normalization...
GPU normalization...
No kernel error detected
No kernel error detected

CPU normalization constant: 0.504522
GPU normalization constant: 0.50445

CPU time (normalization): 34.5623 milliseconds
GPU time (normalization): 2.06803 milliseconds

Speedup factor (normalization): 16.7127




CPU convolution...
GPU convolution...
No kernel error detected
Comparing...

Successful output

CPU time (convolve): 22291.9 milliseconds
GPU time (convolve): 941.743 milliseconds

Speedup factor (convolution): 23.6709


CPU normalization...
GPU normalization...
No kernel error detected
No kernel error detected

CPU normalization constant: 0.502063
GPU normalization constant: 0.502002

CPU time (normalization): 34.6965 milliseconds
GPU time (normalization): 1.91683 milliseconds

Speedup factor (normalization): 18.101
