# NewsNetworks

## Install
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
parser.add_argument("--language", type=str, help="Language of the database")
parser.add_argument("--initial_date", type=str, help="YYYY-mm-dd string for initial date of articles")
parser.add_argument("--verbose", action="store_true", help="Verbose mode")
```

By default the network will normalize the edge weights by the number of total articles a source has published. 

## Legacy

The folder legacy contains code from a very old version of this project. While the two code bases work the same, the updated one is optimized and significantly faster. 
