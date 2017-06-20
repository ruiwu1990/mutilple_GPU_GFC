Compile:
nvcc -O3 -arch=sm_20 GFC_11.cu -o GFC

tsgpu bash -c "./GFC 2 file.gfc GPUresult.txt decompressedDatasets/obs_info.trace.fpc file.out GPUDecompressResult.txt"


./GFC 2 file.gfc GPUresult.txt decompressedDatasets/obs_info.trace.fpc file.out GPUDecompressResult.txt

# More details in this paper
More details in https://scholar.google.com/citations?view_op=view_citation&amp;hl=en&amp;user=403C7GMAAAAJ&amp;citation_for_view=403C7GMAAAAJ:j3f4tGmQtD8C
