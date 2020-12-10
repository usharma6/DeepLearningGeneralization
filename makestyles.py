import os
import numpy as np
import shlex

styles = np.array(['examples/inputs/the_scream.jpg', 'examples/inputs/starry_night.jpg', 'examples/inputs/escher_sphere.jpg', 'examples/inputs/picasso_selfport1907.jpg', 'examples/inputs/woman-with-hat-matisse.jpg'])

for root, dirs, files in os.walk("tiny-imagenet-200/train", topdown=False):
   #for name in files:
    #  print(os.path.join(root, name))
   for name in dirs:
      directory = os.path.join(root, name, "images")
      #directory = os.path.join(directory, "images")
      for _, _, s3 in os.walk(directory):
        for s in s3:
          inputpath = os.path.join(directory, s)
          style = np.random.choice(styles)
          args = shlex.split("python3 neural_style.py " + " -content_image " + inputpath + " -style_image " + style + " -output_image " + inputpath)
          #p = subprocess.Popen(args)
          #p.wait()
          print(args)
