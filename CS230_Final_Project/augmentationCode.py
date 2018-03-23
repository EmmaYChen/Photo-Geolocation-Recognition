
import Augmentor
p = Augmentor.Pipeline("photodirectory")
p.random_distortion(probability=1, grid_width=6, grid_height=6, magnitude=6)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.2)
p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
p.flip_left_right(probability=0.5)
p.sample(9)
