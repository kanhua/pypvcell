from units_system import UnitsSystem
from random import random, randint, choice


if __name__ == "__main__":
    print ("Solcore Units example, requires 'Default Units' plugin.\n\nAvailable units:")
    UnitsSystem().list_dimensions()
    
    print ("\nConverting some random simple units:")
    
    dimensions = UnitsSystem().dimensions

    u=UnitsSystem()
    
    for dim in dimensions: # in each dimension
        from_unit = (choice(list(dimensions[dim].keys())))
        to_unit = (choice(list(dimensions[dim].keys())))
        some_number = random()*10**randint(-10,10)

        some_number_converted = u.convert(some_number,from_unit,to_unit)   ### EXAMPLE 1

        print ("{}: {:.2e} {} = {:.2e} {}".format( dim,
            some_number, from_unit,
            some_number_converted, to_unit
        ))
        
    
    print ("\nSome compound dimensions into SI units:")
    
    print ("using 'si': 1 um2 mOhm-1 eV-1 = {} m2 Ohm-1 J-1".format(
        u.si("1 um2 mOhm-1 eV-1")                                          ### EXAMPLE 2
    ))
    print ("using 'siUnits': 1 um2 mOhm-1 eV-1 = {} m2 Ohm-1 J-1".format(
        u.siUnits(1,"um2 mOhm-1 eV-1")                                     ### EXAMPLE 3
    ))
    print ("using 'convert': 1 um2 mOhm-1 eV-1 = {} m2 Ohm-1 J-1".format(
        u.convert(1,"um2 mOhm-1 eV-1", "m2 Ohm-1 J-1")                     ### EXAMPLE 4
    ))

