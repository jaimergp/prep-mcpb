# Further recommendations

According to the experience of former colleagues in our lab, you might need to apply some changes in the files proposed by MCPB.py.

## Step 1

- Increase `%nproc` and `%mem` as needed in your Gaussian input files.
- Use B3LYP with SDD basis set plus Grimme dispersion (`EmpiricalDispersion=GD3`).
- In the `*_large_mk.com` file, change the VdW radius of the metal ion at the bottom of the file to reflect the values suggested in Table 9 of _S. S. Batsanov, Inorganic
metal-X bonds. Materials. Vol. 37, No 9, 2001, pp 871-885_. DOI: `10.1023/A:1011625728803`.
