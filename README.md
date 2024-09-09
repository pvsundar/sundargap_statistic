# Sundar-Tibshirani Gap Statistic Package

This package implements the Sundar-Tibshirani Gap Statistic for cluster analysis.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/pvsundar/sundar_gap_stat.git
```

2. Navigate to the package directory:
```bash
cd sundar_gap_stat
```
3. Install the package and its dependencies:

```bash
pip install .
```

Alternatively, you can install directly from the GitHub repository:

```bash
pip install git+https://github.com/pvsundar/sundar_gap_stat.git
```

## Usage Example
Here’s a basic example of how to use the package with the Iris dataset:

```python
from sundar_gap_stat import sundar_tibshirani_gap_statistic_main
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
sundar_tibshirani_gap_statistic_main(X, y, k=3)
```
## Contributing
Feel free to open an issue or submit a pull request if you'd like to contribute.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments
Special thanks to all contributors and the open-source community.
in particular to the code by Mavengence who created the package for the Tibshirani Gap Statistic.
https://github.com/Mavengence/GapStatistics/blob/main/gapstatistics/gapstatistics.py
