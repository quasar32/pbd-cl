# Position Based Dynamics 

Position based dynamics example using randomly generated beads on a wire.

## Build

Use `make` with the name of the program you wish to create.
If no name is provided will create `pbd`.

## `pbd`

`pbd` outputs a `csv` of 10 seconds of simulation.
If a filename is not provided `pbd` outputs to standard output. 

## `vid`
`vid` turns a `csv` from `pbd` into `mp4`. 
First argument is the source `csv`.
If this argument is not provided default to standard input.
Second argument is the output `mp4`.
If this argument is not provided default to `pbd.mp4`.
