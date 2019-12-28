LMC
run 0 - fresh run with adam and adding dropout to conv2 and removing bnorm after fc1 (2952181)
run 1 - now with max pooling instead of stride on conv4 (2952878)
run 2 - using loads of padding with the strides in the paper (2953195) - failed run
run 3 - reverting to normal code but with main function at bottom (2953196)
run 4 - switching height and width and using strides + padding... (2953228)
run 5 - Data:(C,H,W) and normal padding with no strides (2953230)

MC
run 0 - fresh run with adam and adding dropout to conv2 and removing bnorm after fc1 (2952123)
run 1 - run 1 - now with max pooling instead of stride on conv4 (2952877)

MLMC
run 0 - fresh run with adam and adding dropout to conv2 and removing bnorm after fc1 (2952829)
run 1 - now with max pooling instead of stride on conv4 (2952876)
