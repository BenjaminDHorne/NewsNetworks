# NewsNetworks

## Install
**Note:** this requirements file is outdated, will be updated soon. The current code in generate_network.py uses a limited set of external libraries.

This code is built in Python 3. Requirements can be installed using:

```
pip install -r requirements.txt
```

## Use

To generate a news outlet network using a NELA-GT database, simply run generate_network.py with command line arguments for: path to the nela database, path to write pair CSV file to, and path to save GML file to (GML file is the network file). Optionally, you can add the argument --initial_date in the form of YYYY-mm-dd string to start the network building on a specific date. Here is a more detailed look at the arguments:

```
parser.add_argument("input", type=str, help="Path to nela database")
parser.add_argument("output_pair_file", type=str, help="Path to write pair CSV file to")
parser.add_argument("output_network_file", type=str, help="Path to save GML file to")
parser.add_arguement("--heuristics_off", type="store_true", help="Turn off heuristic functions (We strongly recommend not doing this)")
parser.add_argument("--language", type=str, help="Language of the database")
parser.add_argument("--initial_date", type=str, help="YYYY-mm-dd string for initial date of articles")
parser.add_argument("--verbose", action="store_true", help="Verbose mode")
```

By default the network will normalize the edge weights by the number of total articles a source has published. 

## Legacy

The folder legacy contains code from a very old version of this project. While the two code bases work the same, the updated one is optimized and significantly faster. 

## Citation when using code
Please cite the following work when using this code:

Horne, B. D., Nørregaard, J., & Adalı, S. (2019, July). Different spirals of sameness: A study of content sharing in mainstream and alternative media. In Proceedings of the International AAAI Conference on Web and Social Media (Vol. 13, pp. 257-266).

Bibtex:

```
@inproceedings{horne2019different,
  title={Different spirals of sameness: A study of content sharing in mainstream and alternative media},
  author={Horne, Benjamin D and N{\o}rregaard, Jeppe and Adal{\i}, Sibel},
  booktitle={Proceedings of the International AAAI Conference on Web and Social Media},
  volume={13},
  pages={257--266},
  year={2019}
}
```
