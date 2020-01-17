# CV Final Project Stereo Matching
## File Placements:

```
.
    ├── data/                        # Place the data directory here
          ├── Output/                # Output directory
          ├── Real/                  # Real images
          └── Synthetic/             # Synthetic images
    ├── log/                         # Logging information
          ├── log.txt                # Log when running `run_syn.h`
          ├── log_error.txt          # Log image errors to compute average
          └── arguments.txt          # Backup for arguments
    └── ...
```

## Run the Code

```sh
python3 main.py --input-left $LEFT_PATH --input-right $RIGHT_PATH --output $OUT_PATH
```



## Development Notes

- Follow the notes when developing

### requirements.txt
- Add the modules you installed in the `requirements.txt` file

### Shell scripts
- `. run_one.sh` to run a single image, you can uncomment `main.py -c` to read from config, or uncomment `. vis1.sh $1` to visualize the output
- `. run_syn.sh` to run all synthetic images. Outputs are logged to `log/log.txt` and errors for each image are logged to `log_error.txt`
- `. vis1.sh ${num}` to visualize an image

### Arguments
- Add arguments specific to file in the `parse-from-{file}` functions in each file
- Run `python main.py -h` to understand arguments
- If using base method for a certain phase, specify in the argument using `--CM_base` or `--OP_base` or `--RF_base`

### Config.json
- Run `python main.py -c` if you want to read arguments from the config
- Use the function `parser.write_config({output_path})` to write out config

### Cost manager
- Defined in `costmgr.py` to implement cost definition and aggregation
- Add own methods and add it under the `run()` method

### Optimizer
- Defined in `optimizer.py` to optimize the disparity selection process
- Add own methods and add it under the `run()` method

### Refiner
- Defined in `refiner.py` to implement the refinement process
- Add own methods and add it under the `run()` method
