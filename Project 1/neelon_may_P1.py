import numpy as np

coeffs = np.array([ 2.36729952e+03, 1.16447584e+00, -1.04770170e+02])
powers = np.array([ 0, 1, 0.5 ])
year_unparsed = input("Enter a year: ")
year = None
try:
    year = int(year_unparsed)
except:
    print("The input entered was not a valid year")

if year != None:
    time = (year ** powers) @ coeffs
    print(f"Predicted time is {time} seconds")
