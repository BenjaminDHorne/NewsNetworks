# NewsNetworks

## Install
This code is built in Python 3. Requirements can be installed using:

```
pip install -r requirements.txt
```

## Use

To generate a news outlet network using a NELA-GT database, simply run generate_network.py with command line arguments for: path to the nela database, path to write pair CSV file to, and path to save GML file to (GML file is the network file). Optionally, you can add the argument --initial_date in the form of YYYY-mm-dd string to start the network building on a specific date.

By default the network will normalize the edge weights by the number of total articles a source has published. 

## Legacy

The folder legacy contains code from a very old version of this project. While the two code bases work the same, the updated one is optimized and significantly faster. 
