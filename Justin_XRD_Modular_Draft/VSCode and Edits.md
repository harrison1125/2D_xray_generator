# VSCode Setup and Edits Documentation
### Homebrew for package management on Mac
- paste code into terminal: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
- sota password is just your computer password (ubuntu system)
- Follow brew instructions to install onto path
- brew update -> brew install python -> brew upgrade python
### VSCode
- Follow installation instructions on the tutorial
### Edits Summary (see inline comments for more explanation)
Main:
- plt.show() rearrangement to show heat map later
- and statement for coords_on_detector removed
- gaussian parameters put into a separate file for modularization

Grain:
- grain.size_average -> self.size_average
- grain.strain_average -> self.strain_average
- simplified last few lines in randomize_rotation() function

Detector:
- lowercase "experiment" vs "Experiment" variable to avoid confusion with class Experiment
- removed unused line
- vectorized corner calculations with numpy slicing for speed and readability
- deleted duplicate return statement and moved first return statement outside the loop

EwaldSphere:
- untouched

Experiment:
- untouched

Sample:
- untouched

PointOnDetector:
- appears to be unused as of now, so I did not make a separate module for it. In the future, it can be added

