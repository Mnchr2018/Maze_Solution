from pathlib import Path
import os
import cv2
import pandas as pd
import numpy as np
import random
from solve_maze import solve_maze_2

if __name__ == "__main__":
    file="img004.tif"
    cwd=Path.cwd().resolve()
    im=solve_maze.master_solver(image=os.path.join(cwd,"images",file),
                                    black_white_threshold=200, 
                                    pixel_to_matrix_factor=0.8,
                                    pixel_margin=1, 
                                    block_threshold=0.98)

    cv2.imwrite('img.png',im)