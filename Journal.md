# Journal

### AVX implementation

- AVX has been implemented for matrix multiplication, now the next step would be to add more avx instructions so that they get pipelined well.
- Multiple AVX instructions have been added in one loop in hopes that it will pipeline effectively, this has almost halved the time taken.
- An attempt was made to swap the last loop of matrix multiplication with the second last one (iterating rows of second matrix with iterating columns of the first), however it seemed to worsen the performance and increase the computation time.